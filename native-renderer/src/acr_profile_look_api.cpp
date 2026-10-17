#include "imprint_renderer.h"
#include "acr_profile_encoding.hpp"
#include "acr_profile_table.hpp"
#include "acr_profile_stage.hpp"

#include <cstdint>

im_status im_native_acr_profile_look_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *look_table, size_t look_values, uint32_t hue_count,
    uint32_t saturation_count, uint32_t value_count, uint32_t look_encoding,
    float *destination, size_t destination_values) {
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (!width || !height || pixels > (uint64_t{1} << 26) ||
        source_values != pixels * 3 || destination_values != pixels * 3 ||
        look_values % 3 != 0 || look_encoding > 1)
        return IM_STATUS_INVALID_ARGUMENT;
    const imprint::acr::ProfileTableView table{
        look_table, look_values / 3, hue_count, saturation_count, value_count,
    };
    imprint::acr::ProfileCurveView forward{}, inverse{};
    if (look_encoding == 1 && value_count > 1) {
        const auto &curves = imprint::acr::profile_srgb_tables();
        forward = {curves.forward.data(), curves.forward.size(),
                   static_cast<float>(imprint::acr::kProfileSrgbIntervals)};
        inverse = {curves.inverse.data(), curves.inverse.size(),
                   static_cast<float>(imprint::acr::kProfileSrgbIntervals)};
    }
    const auto status = imprint::acr::apply_profile_table(
        source, destination, static_cast<size_t>(pixels), table, forward, inverse);
    if (status == imprint::acr::ProfileTableStatus::ok) return IM_STATUS_OK;
    if (status == imprint::acr::ProfileTableStatus::allocation_failure)
        return IM_STATUS_RUNTIME_ERROR;
    return IM_STATUS_INVALID_ARGUMENT;
}

im_status im_native_acr_profile_stage_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *look_table, size_t look_values, uint32_t hue_count,
    uint32_t saturation_count, uint32_t value_count, uint32_t look_encoding,
    double lookup_gain, double amount,
    float *destination, size_t destination_values) {
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (!width || !height || pixels > (uint64_t{1} << 26) ||
        source_values != pixels * 3 || destination_values != pixels * 3 ||
        look_values % 3 != 0) return IM_STATUS_INVALID_ARGUMENT;
    const auto status = imprint::acr::apply_profile_stage_sdr(
        source, destination, static_cast<size_t>(pixels),
        {look_table, look_values / 3, hue_count, saturation_count, value_count},
        look_encoding, lookup_gain, amount);
    if (status == imprint::acr::ProfileTableStatus::ok) return IM_STATUS_OK;
    if (status == imprint::acr::ProfileTableStatus::allocation_failure)
        return IM_STATUS_RUNTIME_ERROR;
    return IM_STATUS_INVALID_ARGUMENT;
}
