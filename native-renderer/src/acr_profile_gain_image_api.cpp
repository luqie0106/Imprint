#include "imprint_renderer.h"
#include "acr_profile_gain_image.hpp"
#include "acr_profile_gain_map.hpp"

im_status im_native_acr_profile_gain_payload_validate(
    const uint8_t *payload, size_t payload_bytes, uint32_t tag,
    uint32_t byte_order) {
    imprint::acr::ProfileGainMapData parsed;
    const auto status = imprint::acr::parse_profile_gain_map_payload(
        payload, payload_bytes, tag,
        static_cast<imprint::acr::ProfileGainMapByteOrder>(byte_order), parsed);
    using Status = imprint::acr::ProfileGainMapStatus;
    if (status == Status::ok) return IM_STATUS_OK;
    if (status == Status::allocation_failure) return IM_STATUS_RUNTIME_ERROR;
    return IM_STATUS_INVALID_ARGUMENT;
}

im_status im_native_acr_profile_gain_image_prepared_f32(
    const float *source, size_t source_count, size_t source_row_stride,
    float *destination, size_t destination_count, size_t destination_row_stride,
    const uint8_t *payload, size_t payload_bytes, uint32_t tag, uint32_t byte_order,
    const int32_t *rects, size_t rect_count, const double *matrix50,
    size_t matrix_count, double scalar, int32_t amount, uint32_t lookup_flag,
    uint32_t predicate, uint32_t has_coordinate_task) {
    const auto status = imprint::acr::process_profile_gain_image_rgbf32(
        source, source_count, source_row_stride,
        destination, destination_count, destination_row_stride,
        payload, payload_bytes, tag, byte_order, rects, rect_count,
        matrix50, matrix_count, scalar, amount, lookup_flag, predicate,
        has_coordinate_task);
    using Status = imprint::acr::ProfileGainImageStatus;
    switch (status) {
        case Status::ok:
            return IM_STATUS_OK;
        case Status::buffer_too_small:
            return IM_STATUS_BUFFER_TOO_SMALL;
        case Status::unsupported_transform:
            return IM_STATUS_BACKEND_UNAVAILABLE;
        case Status::allocation_failure:
        case Status::non_finite_intermediate:
            return IM_STATUS_RUNTIME_ERROR;
        default:
            return IM_STATUS_INVALID_ARGUMENT;
    }
}
