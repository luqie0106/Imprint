#include "imprint_renderer.h"
#include "acr_stage3_prepared_image.hpp"

extern "C" IMPRINT_API im_status im_native_acr_stage3_prepared_image_u16(
    const uint16_t* source, size_t source_sample_count, size_t source_row_stride,
    const int32_t source_rect4[4], const double gains3[3], uint16_t source_floor,
    int cubic, double amount, int suppress_refine, uint16_t hot_threshold,
    uint16_t cold_threshold, size_t tile_width, size_t tile_height,
    uint16_t* destination, size_t destination_sample_count) {
    using imprint::acr::PreparedImageStatus;
    const auto status = imprint::acr::run_stage3_prepared_image_u16_to_buffer(
        source, source_sample_count, source_row_stride, source_rect4, gains3,
        source_floor, cubic, amount, suppress_refine, hot_threshold,
        cold_threshold, tile_width, tile_height, destination,
        destination_sample_count);
    switch (status) {
        case PreparedImageStatus::ok:
            return IM_STATUS_OK;
        case PreparedImageStatus::buffer_too_small:
            return IM_STATUS_BUFFER_TOO_SMALL;
        case PreparedImageStatus::resource_limit:
        case PreparedImageStatus::allocation_failure:
        case PreparedImageStatus::length_error:
            return IM_STATUS_RUNTIME_ERROR;
        default:
            return static_cast<int>(status) < 25 ? IM_STATUS_INVALID_ARGUMENT
                                               : IM_STATUS_RUNTIME_ERROR;
    }
}
