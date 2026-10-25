#include "../src/acr_warp.hpp"

#include <cstddef>
#include <cstdint>

// Test-only C ABI. All metadata, alias, and destination checks live in the
// shared prepared-float32 adapter.
extern "C" int acr_test_warp(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const std::int32_t rect4[4],
    const std::uint32_t controls2[2], const double geometry3[3],
    const double* coefficients, std::size_t coefficient_count,
    std::size_t tile_width, std::size_t tile_height, float* destination,
    std::size_t destination_sample_count) noexcept {
    return static_cast<int>(imprint::acr::apply_warp_prepared_f32_to_buffer(
        source, source_sample_count, source_row_stride, rect4, controls2,
        geometry3, coefficients, coefficient_count, tile_width, tile_height,
        destination, destination_sample_count));
}
