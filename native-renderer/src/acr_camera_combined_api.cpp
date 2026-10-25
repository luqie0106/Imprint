#include "imprint_renderer.h"
#include "acr_camera_combined.hpp"

extern "C" IMPRINT_API im_status im_native_acr_camera_combined_prepared_f32(
    const float* source, size_t source_count, size_t source_row_stride,
    size_t height, size_t width, const double* controls, size_t control_count,
    const uint32_t order3[3], uint32_t flags, float* destination,
    size_t destination_count, size_t destination_row_stride) {
    using imprint::acr::CombinedCameraStatus;
    const auto status = imprint::acr::apply_camera_combined_prepared_f32_to_buffer(
        source, source_count, source_row_stride, height, width, controls,
        control_count, order3, flags, destination, destination_count,
        destination_row_stride);
    switch (status) {
        case CombinedCameraStatus::ok: return IM_STATUS_OK;
        case CombinedCameraStatus::buffer_too_small: return IM_STATUS_BUFFER_TOO_SMALL;
        case CombinedCameraStatus::resource_limit:
        case CombinedCameraStatus::allocation_failure:
        case CombinedCameraStatus::length_error: return IM_STATUS_RUNTIME_ERROR;
        default: return IM_STATUS_INVALID_ARGUMENT;
    }
}
