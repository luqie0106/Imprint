#include "imprint_renderer.h"
#include "acr_buffer_format.hpp"

namespace {
im_status mapped(imprint::acr::BufferFormatStatus status) noexcept {
    using imprint::acr::BufferFormatStatus;
    switch (status) {
        case BufferFormatStatus::ok: return IM_STATUS_OK;
        case BufferFormatStatus::buffer_too_small: return IM_STATUS_BUFFER_TOO_SMALL;
        case BufferFormatStatus::resource_limit:
        case BufferFormatStatus::allocation_failure:
        case BufferFormatStatus::length_error: return IM_STATUS_RUNTIME_ERROR;
        default: return IM_STATUS_INVALID_ARGUMENT;
    }
}
} // namespace

extern "C" IMPRINT_API im_status im_native_acr_buffer_f32_to_u16(
    const float* source, size_t source_count, size_t source_row_stride,
    size_t height, size_t row_samples, uint16_t* destination,
    size_t destination_count, size_t destination_row_stride) {
    return mapped(imprint::acr::copy_buffer_f32_to_u16_012eaeb8(
        source, source_count, source_row_stride, height, row_samples,
        destination, destination_count, destination_row_stride));
}

extern "C" IMPRINT_API im_status im_native_acr_buffer_u16_to_f32(
    const uint16_t* source, size_t source_count, size_t source_row_stride,
    size_t height, size_t row_samples, float* destination,
    size_t destination_count, size_t destination_row_stride) {
    return mapped(imprint::acr::copy_buffer_u16_to_f32_012eac64(
        source, source_count, source_row_stride, height, row_samples,
        destination, destination_count, destination_row_stride));
}
