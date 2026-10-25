#include "../src/acr_profile_u16.hpp"

extern "C" int im_native_acr_profile_stage_sdr_u16(
    const std::uint16_t *src, std::size_t src_count, std::size_t src_stride,
    std::size_t height, std::size_t width, const float *table_data,
    std::size_t table_values, std::uint32_t hue_count,
    std::uint32_t saturation_count, std::uint32_t value_count,
    std::uint32_t encoding, double lookup_gain, double amount,
    std::uint32_t pixel_type, std::uint16_t *dst, std::size_t dst_count,
    std::size_t dst_stride) noexcept {
    if (table_values % 3 != 0) {
        return static_cast<int>(imprint::acr::ProfileU16Status::invalid_argument);
    }
    const imprint::acr::ProfileTableView table{
        table_data, table_values / 3, hue_count, saturation_count, value_count};
    return static_cast<int>(imprint::acr::apply_profile_stage_sdr_u16(
        src, src_count, src_stride, height, width, table,
        static_cast<unsigned>(encoding),
        lookup_gain, amount, pixel_type, dst, dst_count, dst_stride));
}
