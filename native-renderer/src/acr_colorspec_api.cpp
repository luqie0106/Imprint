#include "imprint_renderer.h"
#include "acr_colorspec.hpp"
#include "acr_camera_prepare.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
bool valid_buffers(const double *controls, size_t count, size_t expected_count,
                   double *output, size_t output_count, size_t expected_output) {
    if (!controls || !output || count != expected_count || output_count != expected_output)
        return false;
    const auto a = reinterpret_cast<uintptr_t>(controls);
    const auto b = reinterpret_cast<uintptr_t>(output);
    if (a > std::numeric_limits<uintptr_t>::max() - count * sizeof(double) ||
        b > std::numeric_limits<uintptr_t>::max() - output_count * sizeof(double) ||
        (a < b + output_count * sizeof(double) && b < a + count * sizeof(double)))
        return false;
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(controls[i])) return false;
    return true;
}
bool valid_colorspec_flags(const double *controls) {
    for (size_t i = 61; i < 64; ++i)
        if (controls[i] != 0 && controls[i] != 1) return false;
    return true;
}
imprint::acr::ColorSpecInput read_colorspec(const double *controls) {
    imprint::acr::ColorSpecInput input;
    std::memcpy(input.analog_balance.data(), controls, 3 * sizeof(double));
    std::memcpy(input.raw_cm1.data(), controls + 3, 9 * sizeof(double));
    std::memcpy(input.raw_cm2.data(), controls + 12, 9 * sizeof(double));
    std::memcpy(input.cc1.data(), controls + 21, 9 * sizeof(double));
    std::memcpy(input.cc2.data(), controls + 30, 9 * sizeof(double));
    std::memcpy(input.forward1.data(), controls + 39, 9 * sizeof(double));
    std::memcpy(input.forward2.data(), controls + 48, 9 * sizeof(double));
    input.endpoint_kelvin1 = controls[57];
    input.endpoint_kelvin2 = controls[58];
    input.white_xy = {controls[59], controls[60]};
    input.has_second_cm = controls[61] == 1;
    input.has_forward1 = controls[62] == 1;
    input.has_forward2 = controls[63] == 1;
    return input;
}
void write_colorspec(const imprint::acr::ColorSpecPrepared &prepared, double *values) {
    std::memcpy(values, prepared.camera_white.data(), 3 * sizeof(double));
    std::memcpy(values + 3, prepared.camera_to_pcs.data(), 9 * sizeof(double));
    std::memcpy(values + 12, prepared.pcs_to_camera.data(), 9 * sizeof(double));
    values[21] = prepared.mix_weight;
}
}  // namespace

im_status im_native_acr_prepare_colorspec(const double *controls, size_t count,
                                          double *output, size_t output_count) {
    if (!valid_buffers(controls, count, 64, output, output_count, 22) ||
        !valid_colorspec_flags(controls))
        return IM_STATUS_INVALID_ARGUMENT;
    const auto input = read_colorspec(controls);
    imprint::acr::ColorSpecPrepared prepared;
    if (imprint::acr::prepare_acr_colorspec(input, prepared) != imprint::acr::ColorSpecStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;
    std::array<double, 22> values;
    write_colorspec(prepared, values.data());
    std::memcpy(output, values.data(), values.size() * sizeof(double));
    return IM_STATUS_OK;
}

im_status im_native_acr_resolve_white(const double *controls, size_t count,
                                      double *output, size_t output_count) {
    if (!valid_buffers(controls, count, 68, output, output_count, 26) ||
        !valid_colorspec_flags(controls) ||
        (controls[67] != 0 && controls[67] != 1))
        return IM_STATUS_INVALID_ARGUMENT;
    imprint::acr::WhiteResolveInput input;
    input.profile = read_colorspec(controls);
    std::memcpy(input.camera_neutral.data(), controls + 64, 3 * sizeof(double));
    input.output_referred_mode = controls[67] == 1;
    imprint::acr::WhiteResolvePrepared prepared;
    if (imprint::acr::resolve_acr_white(input, prepared) != imprint::acr::ColorSpecStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;
    std::array<double, 26> values;
    std::memcpy(values.data(), prepared.neutral_xy.data(), 2 * sizeof(double));
    std::memcpy(values.data() + 2, prepared.adjusted_xy.data(), 2 * sizeof(double));
    write_colorspec(prepared.colorspec, values.data() + 4);
    std::memcpy(output, values.data(), values.size() * sizeof(double));
    return IM_STATUS_OK;
}

im_status im_native_acr_prepare_camera_neutral(const double *neutral, size_t count,
                                               double *output, size_t output_count) {
    if (!valid_buffers(neutral, count, 3, output, output_count, 3))
        return IM_STATUS_INVALID_ARGUMENT;
    double maximum = neutral[0];
    if (maximum < neutral[1]) maximum = neutral[1];
    if (maximum < neutral[2]) maximum = neutral[2];
    if (maximum <= 0) return IM_STATUS_INVALID_ARGUMENT;
    const double reciprocal = 1.0 / maximum;
    if (!std::isfinite(reciprocal)) return IM_STATUS_INVALID_ARGUMENT;
    std::array<double, 3> prepared;
    for (size_t i = 0; i < 3; ++i) {
        const double normalized = neutral[i] * reciprocal;
        const double scaled = normalized * 1.0e6;
        const double rounded = scaled + (scaled > 0 ? 0.5 : -0.5);
        if (!std::isfinite(normalized) || !std::isfinite(scaled) ||
            !std::isfinite(rounded) || rounded <= -2147483649.0 || rounded >= 2147483648.0)
            return IM_STATUS_INVALID_ARGUMENT;
        prepared[i] = 1.0e-6 * static_cast<int32_t>(rounded);
    }
    std::memcpy(output, prepared.data(), prepared.size() * sizeof(double));
    return IM_STATUS_OK;
}

im_status im_native_acr_prepare_profile_camera(const double *controls, size_t count,
                                               double *output, size_t output_count) {
    if (!valid_buffers(controls, count, 75, output, output_count, 65) ||
        !valid_colorspec_flags(controls) || (controls[67] != 0 && controls[67] != 1))
        return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 68; i < 75; ++i)
        if (controls[i] != std::trunc(controls[i]) || controls[i] < -100 || controls[i] > 100)
            return IM_STATUS_INVALID_ARGUMENT;
    imprint::acr::WhiteResolveInput white_input;
    white_input.profile = read_colorspec(controls);
    if (im_native_acr_prepare_camera_neutral(controls + 64, 3,
                                           white_input.camera_neutral.data(), 3) != IM_STATUS_OK)
        return IM_STATUS_INVALID_ARGUMENT;
    white_input.output_referred_mode = controls[67] == 1;
    imprint::acr::WhiteResolvePrepared white;
    if (imprint::acr::resolve_acr_white(white_input, white) != imprint::acr::ColorSpecStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;
    imprint::acr::CameraStageInput camera_input;
    camera_input.camera_white = white.colorspec.camera_white;
    camera_input.camera_to_pcs = white.colorspec.camera_to_pcs;
    camera_input.source_vector = white_input.profile.analog_balance;
    std::array<double, 9> prophoto_matrix;
    if (imprint::acr::prepare_normalized_prophoto_inverse(camera_input.prophoto_inverse,
                                                         &prophoto_matrix) != imprint::acr::ColorSpecStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < 6; ++i)
        camera_input.calibration[i] = static_cast<int32_t>(controls[68 + i]);
    camera_input.shadow_tint = static_cast<int32_t>(controls[74]);
    imprint::acr::CameraStagePrepared camera;
    if (imprint::acr::prepare_camera_stage(camera_input, camera) != imprint::acr::CameraStageStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;
    std::array<double, 65> values;
    std::memcpy(values.data(), white.neutral_xy.data(), 2 * sizeof(double));
    std::memcpy(values.data() + 2, white.adjusted_xy.data(), 2 * sizeof(double));
    write_colorspec(white.colorspec, values.data() + 4);
    for (size_t i = 0; i < 9; ++i) {
        values[26 + i] = camera.matrix[i];
        values[35 + i] = camera.sorted_matrix[i];
        values[56 + i] = prophoto_matrix[i];
    }
    for (size_t i = 0; i < 3; ++i) {
        values[44 + i] = camera.gain_q12[i];
        values[47 + i] = camera.gain_float[i];
        values[50 + i] = camera.black_compensation[i];
        values[53 + i] = camera.sorted_plane_indices[i];
    }
    std::memcpy(output, values.data(), values.size() * sizeof(double));
    return IM_STATUS_OK;
}
