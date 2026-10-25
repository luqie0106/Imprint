#include "imprint_renderer.h"
#include "acr_postcamera_black.hpp"

namespace {
im_status mapped(imprint::acr::PostcameraBlackStatus status) noexcept {
    using imprint::acr::PostcameraBlackStatus;
    switch (status) {
        case PostcameraBlackStatus::ok: return IM_STATUS_OK;
        case PostcameraBlackStatus::buffer_too_small: return IM_STATUS_BUFFER_TOO_SMALL;
        case PostcameraBlackStatus::resource_limit:
        case PostcameraBlackStatus::allocation_failure:
        case PostcameraBlackStatus::length_error: return IM_STATUS_RUNTIME_ERROR;
        default: return IM_STATUS_INVALID_ARGUMENT;
    }
}
}

extern "C" IMPRINT_API im_status im_native_acr_postcamera_black_prepared_f32(
    const float* source, size_t source_count, size_t source_row_stride,
    size_t height, size_t row_samples, double black, float* destination,
    size_t destination_count, size_t destination_row_stride) {
    return mapped(imprint::acr::apply_postcamera_black_prepared_f32(
        source, source_count, source_row_stride, height, row_samples, black,
        destination, destination_count, destination_row_stride));
}

extern "C" IMPRINT_API im_status im_native_acr_postcamera_black_prepared_u16(
    const uint16_t* source, size_t source_count, size_t source_row_stride,
    size_t height, size_t row_samples, double black, uint16_t* destination,
    size_t destination_count, size_t destination_row_stride) {
    return mapped(imprint::acr::apply_postcamera_black_prepared_u16(
        source, source_count, source_row_stride, height, row_samples, black,
        destination, destination_count, destination_row_stride));
}
