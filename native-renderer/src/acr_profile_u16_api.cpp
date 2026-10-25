#include "imprint_renderer.h"
#include "acr_profile_u16.hpp"

extern "C" IMPRINT_API im_status im_native_acr_profile_stage_sdr_u16(
    const uint16_t* source, size_t source_count, size_t source_row_stride,
    size_t height, size_t width, const float* table_data, size_t table_values,
    uint32_t hue_count, uint32_t saturation_count, uint32_t value_count,
    uint32_t encoding, double lookup_gain, double amount, uint32_t pixel_type,
    uint16_t* destination, size_t destination_count, size_t destination_row_stride) {
    if (table_values % 3 != 0) return IM_STATUS_INVALID_ARGUMENT;
    const imprint::acr::ProfileTableView table{
        table_data, table_values / 3, hue_count, saturation_count, value_count};
    const auto status = imprint::acr::apply_profile_stage_sdr_u16(
        source, source_count, source_row_stride, height, width, table,
        encoding, lookup_gain, amount, pixel_type,
        destination, destination_count, destination_row_stride);
    using imprint::acr::ProfileU16Status;
    switch (status) {
        case ProfileU16Status::ok: return IM_STATUS_OK;
        case ProfileU16Status::buffer_too_small: return IM_STATUS_BUFFER_TOO_SMALL;
        case ProfileU16Status::resource_limit:
        case ProfileU16Status::allocation_failure: return IM_STATUS_RUNTIME_ERROR;
        default: return IM_STATUS_INVALID_ARGUMENT;
    }
}
