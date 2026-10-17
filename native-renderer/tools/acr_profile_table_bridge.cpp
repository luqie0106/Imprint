#include "../src/acr_profile_table.hpp"

#include <cstddef>

extern "C" int acr_test_apply_profile_table(
    const float *input_rgb, std::size_t input_element_count,
    const float *table_data, std::size_t table_count,
    std::size_t hue_count, std::size_t saturation_count,
    std::size_t value_count, const float *forward_data,
    std::size_t forward_count, float forward_scale,
    const float *inverse_data, std::size_t inverse_count,
    float inverse_scale, float *output_rgb,
    std::size_t output_element_count) noexcept {
    if (input_element_count % 3 != 0 ||
        input_element_count != output_element_count) {
        return static_cast<int>(imprint::acr::ProfileTableStatus::invalid_argument);
    }
    if (table_count % 3 != 0)
        return static_cast<int>(imprint::acr::ProfileTableStatus::invalid_table);
    const imprint::acr::ProfileTableView table{
        table_data, table_count / 3, hue_count, saturation_count, value_count};
    const imprint::acr::ProfileCurveView forward{
        forward_data, forward_count, forward_scale};
    const imprint::acr::ProfileCurveView inverse{
        inverse_data, inverse_count, inverse_scale};
    return static_cast<int>(imprint::acr::apply_profile_table(
        input_rgb, output_rgb, input_element_count / 3, table, forward, inverse));
}
