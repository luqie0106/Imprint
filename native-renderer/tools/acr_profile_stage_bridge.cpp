#include "../src/acr_profile_stage.hpp"

extern "C" int acr_test_profile_stage(
    const float *input, std::size_t input_values, const float *table,
    std::size_t table_values, std::size_t h, std::size_t s, std::size_t v,
    unsigned encoding, double gain, double amount, float *output,
    std::size_t output_values) noexcept {
    if (input_values % 3 || table_values % 3 || output_values != input_values) return 1;
    return static_cast<int>(imprint::acr::apply_profile_stage_sdr(
        input, output, input_values / 3,
        {table, table_values / 3, h, s, v}, encoding, gain, amount));
}
