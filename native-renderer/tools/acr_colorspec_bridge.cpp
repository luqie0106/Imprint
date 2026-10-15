#include "../src/acr_colorspec.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr std::size_t kControlCount = 64;
constexpr std::size_t kOutputCount = 22;
constexpr std::size_t kWhiteResolveControlCount = 68;
constexpr std::size_t kWhiteResolveOutputCount = 26;

bool overlaps(const void *left, std::size_t left_size,
              const void *right, std::size_t right_size) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_size;
    return left_start - right_start < right_size;
}

bool read_flag(double value, bool &flag) noexcept {
    if (!std::isfinite(value) || (value != 0.0 && value != 1.0)) return false;
    flag = value == 1.0;
    return true;
}

}  // namespace

// Controls: AB[0:3], rawCM1[3:12], rawCM2[12:21], CC1[21:30],
// CC2[30:39], Forward1[39:48], Forward2[48:57], endpointKelvin[57:59],
// whiteXY[59:61], hasSecondCM, hasForward1, hasForward2[61:64].
// Output: CameraWhite[0:3], CameraToPCS[3:12], PCStoCamera[12:21], mixWeight.
extern "C" int acr_test_prepare_colorspec(
    const double *controls, std::size_t control_count,
    double *output, std::size_t output_count) {
    if (controls == nullptr || output == nullptr || control_count != kControlCount ||
        output_count != kOutputCount ||
        overlaps(controls, kControlCount * sizeof(double),
                 output, kOutputCount * sizeof(double))) {
        return static_cast<int>(imprint::acr::ColorSpecStatus::invalid_input);
    }

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
    if (!read_flag(controls[61], input.has_second_cm) ||
        !read_flag(controls[62], input.has_forward1) ||
        !read_flag(controls[63], input.has_forward2)) {
        return static_cast<int>(imprint::acr::ColorSpecStatus::invalid_input);
    }

    imprint::acr::ColorSpecPrepared prepared;
    const auto status = imprint::acr::prepare_acr_colorspec(input, prepared);
    if (status != imprint::acr::ColorSpecStatus::ok) return static_cast<int>(status);

    std::array<double, kOutputCount> result{};
    std::memcpy(result.data(), prepared.camera_white.data(), 3 * sizeof(double));
    std::memcpy(result.data() + 3, prepared.camera_to_pcs.data(), 9 * sizeof(double));
    std::memcpy(result.data() + 12, prepared.pcs_to_camera.data(), 9 * sizeof(double));
    result[21] = prepared.mix_weight;
    std::memcpy(output, result.data(), result.size() * sizeof(double));
    return 0;
}

// Controls[0:64] use acr_test_prepare_colorspec's layout, with its finite
// whiteXY fields ignored by this resolver. CameraNeutral[64:67] is already
// source-max-normalized and quantized by 1e6; outputReferredMode is [67].
// Output: NeutralXY[0:2], adjustedXY[2:4], then the 22 ColorSpec outputs.
extern "C" int acr_test_resolve_white(
    const double *controls, std::size_t control_count,
    double *output, std::size_t output_count) {
    if (controls == nullptr || output == nullptr ||
        control_count != kWhiteResolveControlCount ||
        output_count != kWhiteResolveOutputCount ||
        overlaps(controls, kWhiteResolveControlCount * sizeof(double),
                 output, kWhiteResolveOutputCount * sizeof(double))) {
        return static_cast<int>(imprint::acr::ColorSpecStatus::invalid_input);
    }

    imprint::acr::WhiteResolveInput input;
    std::memcpy(input.profile.analog_balance.data(), controls, 3 * sizeof(double));
    std::memcpy(input.profile.raw_cm1.data(), controls + 3, 9 * sizeof(double));
    std::memcpy(input.profile.raw_cm2.data(), controls + 12, 9 * sizeof(double));
    std::memcpy(input.profile.cc1.data(), controls + 21, 9 * sizeof(double));
    std::memcpy(input.profile.cc2.data(), controls + 30, 9 * sizeof(double));
    std::memcpy(input.profile.forward1.data(), controls + 39, 9 * sizeof(double));
    std::memcpy(input.profile.forward2.data(), controls + 48, 9 * sizeof(double));
    input.profile.endpoint_kelvin1 = controls[57];
    input.profile.endpoint_kelvin2 = controls[58];
    input.profile.white_xy = {controls[59], controls[60]};
    if (!read_flag(controls[61], input.profile.has_second_cm) ||
        !read_flag(controls[62], input.profile.has_forward1) ||
        !read_flag(controls[63], input.profile.has_forward2) ||
        !read_flag(controls[67], input.output_referred_mode)) {
        return static_cast<int>(imprint::acr::ColorSpecStatus::invalid_input);
    }
    std::memcpy(input.camera_neutral.data(), controls + 64, 3 * sizeof(double));

    imprint::acr::WhiteResolvePrepared prepared;
    const auto status = imprint::acr::resolve_acr_white(input, prepared);
    if (status != imprint::acr::ColorSpecStatus::ok) return static_cast<int>(status);

    std::array<double, kWhiteResolveOutputCount> result{};
    std::memcpy(result.data(), prepared.neutral_xy.data(), 2 * sizeof(double));
    std::memcpy(result.data() + 2, prepared.adjusted_xy.data(), 2 * sizeof(double));
    std::memcpy(result.data() + 4, prepared.colorspec.camera_white.data(), 3 * sizeof(double));
    std::memcpy(result.data() + 7, prepared.colorspec.camera_to_pcs.data(), 9 * sizeof(double));
    std::memcpy(result.data() + 16, prepared.colorspec.pcs_to_camera.data(), 9 * sizeof(double));
    result[25] = prepared.colorspec.mix_weight;
    std::memcpy(output, result.data(), result.size() * sizeof(double));
    return 0;
}
