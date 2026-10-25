#include "../src/acr_gainmap.hpp"

#include <cstddef>
#include <cstdint>

// Test-only C ABI used by the Python oracle. The implementation owns all
// metadata, alignment, capacity, and alias validation.
extern "C" int acr_test_gainmap(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const std::int32_t rects8[8],
    const std::uint32_t controls9[9], const double geometry4[4],
    const float* map, std::size_t map_sample_count, std::uint16_t black_floor,
    std::size_t tile_width, std::size_t tile_height, float* destination,
    std::size_t destination_sample_count) noexcept {
    return static_cast<int>(imprint::acr::apply_gainmap_prepared_f32_to_buffer(
        source, source_sample_count, source_row_stride, rects8, controls9,
        geometry4, map, map_sample_count, black_floor, tile_width, tile_height,
        destination, destination_sample_count));
}
