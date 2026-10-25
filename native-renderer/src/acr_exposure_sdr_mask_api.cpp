#include "imprint_renderer.h"
#include "acr_exposure_sdr_mask.hpp"

#include <cstdint>
#include <limits>

namespace {
bool valid_read_span(const void* values, size_t bytes, size_t alignment,
                     const float* destination, size_t destination_count) noexcept {
    if (values == nullptr || destination == nullptr ||
        destination_count > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float))
        return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(values);
    const auto output = reinterpret_cast<std::uintptr_t>(destination);
    const size_t output_bytes = destination_count * sizeof(float);
    if (begin % alignment != 0 ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin ||
        output_bytes > std::numeric_limits<std::uintptr_t>::max() - output)
        return false;
    return !(begin < output + output_bytes && output < begin + bytes);
}
}  // namespace

extern "C" IMPRINT_API im_status im_native_acr_sdr_mask_exposure_prepared_f32(
    const float* source, size_t source_count, size_t source_row_stride,
    size_t height, size_t width, const float* mask, size_t mask_count,
    size_t mask_row_stride, const float* controls, size_t control_count,
    const float* main_lut, size_t main_lut_count,
    float* destination, size_t destination_count, size_t destination_row_stride) {
    if (control_count != 6 || controls == nullptr || destination == nullptr)
        return IM_STATUS_INVALID_ARGUMENT;
    const auto control_address = reinterpret_cast<std::uintptr_t>(controls);
    const auto destination_address = reinterpret_cast<std::uintptr_t>(destination);
    if (control_address % alignof(float) != 0 ||
        6 * sizeof(float) > std::numeric_limits<std::uintptr_t>::max() - control_address ||
        destination_count > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float) ||
        destination_count * sizeof(float) >
            std::numeric_limits<std::uintptr_t>::max() - destination_address)
        return IM_STATUS_INVALID_ARGUMENT;
    if (destination_address < control_address + 6 * sizeof(float) &&
        control_address < destination_address + destination_count * sizeof(float))
        return IM_STATUS_INVALID_ARGUMENT;
    imprint::acr::PreparedExposureSdrMask prepared;
    prepared.gain = controls[0];
    prepared.main_lut_scale = controls[1];
    prepared.rational_scale = controls[2];
    prepared.sqrt_amount = controls[3];
    prepared.rational_denominator = controls[4];
    prepared.epsilon = controls[5];
    prepared.main_lut = main_lut;
    prepared.main_lut_count = main_lut_count;
    using imprint::acr::ExposureSdrMaskStatus;
    const auto status = imprint::acr::apply_prepared_exposure_sdr_mask(
        source, source_count, source_row_stride, height, width,
        mask, mask_count, mask_row_stride, &prepared,
        destination, destination_count, destination_row_stride);
    switch (status) {
        case ExposureSdrMaskStatus::ok: return IM_STATUS_OK;
        case ExposureSdrMaskStatus::buffer_too_small: return IM_STATUS_BUFFER_TOO_SMALL;
        case ExposureSdrMaskStatus::allocation_failure:
        case ExposureSdrMaskStatus::resource_limit: return IM_STATUS_RUNTIME_ERROR;
        default: return IM_STATUS_INVALID_ARGUMENT;
    }
}

extern "C" IMPRINT_API im_status im_native_acr_sdr_mask_exposure_context_prepared_f32(
    const float* source, size_t source_count, size_t source_row_stride,
    size_t height, size_t width, const float* mask, size_t mask_count,
    size_t mask_row_stride, const float* controls, size_t control_count,
    const float* main_lut, size_t main_lut_count,
    const im_acr_exposure_lut_f32* tables, size_t table_count,
    float* destination, size_t destination_count, size_t destination_row_stride) {
    if (control_count != 6 || table_count != 5 ||
        !valid_read_span(controls, 6 * sizeof(float), alignof(float),
                         destination, destination_count) ||
        !valid_read_span(tables, 5 * sizeof(*tables), alignof(im_acr_exposure_lut_f32),
                         destination, destination_count))
        return IM_STATUS_INVALID_ARGUMENT;
    imprint::acr::PreparedExposureSdrMask prepared;
    prepared.gain = controls[0];
    prepared.main_lut_scale = controls[1];
    prepared.rational_scale = controls[2];
    prepared.sqrt_amount = controls[3];
    prepared.rational_denominator = controls[4];
    prepared.epsilon = controls[5];
    prepared.main_lut = main_lut;
    prepared.main_lut_count = main_lut_count;
    for (size_t i = 0; i < 5; ++i) {
        prepared.context[i].values = tables[i].values;
        prepared.context[i].count = tables[i].count;
        prepared.context[i].max_index = tables[i].max_index;
        prepared.context[i].scale = tables[i].scale;
    }
    using imprint::acr::ExposureSdrMaskStatus;
    const auto status = imprint::acr::apply_prepared_exposure_sdr_mask(
        source, source_count, source_row_stride, height, width,
        mask, mask_count, mask_row_stride, &prepared,
        destination, destination_count, destination_row_stride);
    switch (status) {
        case ExposureSdrMaskStatus::ok: return IM_STATUS_OK;
        case ExposureSdrMaskStatus::buffer_too_small: return IM_STATUS_BUFFER_TOO_SMALL;
        case ExposureSdrMaskStatus::allocation_failure:
        case ExposureSdrMaskStatus::resource_limit: return IM_STATUS_RUNTIME_ERROR;
        default: return IM_STATUS_INVALID_ARGUMENT;
    }
}

extern "C" IMPRINT_API im_status im_native_acr_sdr_nomask_exposure_context_prepared_f32(
    const float* source, size_t source_count, size_t source_row_stride,
    size_t height, size_t width, const float* controls, size_t control_count,
    const float* main_lut, size_t main_lut_count,
    const im_acr_exposure_lut_f32* tables, size_t table_count,
    float* destination, size_t destination_count, size_t destination_row_stride) {
    if (control_count != 6 || table_count != 5 ||
        !valid_read_span(controls, 6 * sizeof(float), alignof(float),
                         destination, destination_count) ||
        !valid_read_span(tables, 5 * sizeof(*tables), alignof(im_acr_exposure_lut_f32),
                         destination, destination_count))
        return IM_STATUS_INVALID_ARGUMENT;
    imprint::acr::PreparedExposureSdrMask prepared;
    prepared.gain = controls[0];
    prepared.main_lut_scale = controls[1];
    prepared.rational_scale = controls[2];
    prepared.sqrt_amount = controls[3];
    prepared.rational_denominator = controls[4];
    prepared.epsilon = controls[5];
    prepared.main_lut = main_lut;
    prepared.main_lut_count = main_lut_count;
    for (size_t i = 0; i < 5; ++i) {
        prepared.context[i].values = tables[i].values;
        prepared.context[i].count = tables[i].count;
        prepared.context[i].max_index = tables[i].max_index;
        prepared.context[i].scale = tables[i].scale;
    }
    using imprint::acr::ExposureSdrMaskStatus;
    const auto status = imprint::acr::apply_prepared_exposure_sdr_nomask(
        source, source_count, source_row_stride, height, width, &prepared,
        destination, destination_count, destination_row_stride);
    switch (status) {
        case ExposureSdrMaskStatus::ok: return IM_STATUS_OK;
        case ExposureSdrMaskStatus::buffer_too_small: return IM_STATUS_BUFFER_TOO_SMALL;
        case ExposureSdrMaskStatus::allocation_failure:
        case ExposureSdrMaskStatus::resource_limit: return IM_STATUS_RUNTIME_ERROR;
        default: return IM_STATUS_INVALID_ARGUMENT;
    }
}
