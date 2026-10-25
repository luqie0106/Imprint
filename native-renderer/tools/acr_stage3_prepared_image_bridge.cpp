#include "acr_stage3_prepared_image.hpp"

#include <cstddef>
#include <cstdint>

// Test-only C ABI for Python differential validation. rect4 is the half-open
// {top,left,bottom,right} BayerRepeatRect. Returns PreparedImageStatus as int.
extern "C" int acr_test_stage3_prepared_image(
    const std::uint16_t* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const std::int32_t rect4[4],
    const double gains3[3], std::uint16_t source_floor, int cubic,
    double amount, int suppress_refine, std::uint16_t hot_threshold,
    std::uint16_t cold_threshold, std::size_t tile_width,
    std::size_t tile_height, std::uint16_t* destination,
    std::size_t destination_sample_count) noexcept {
    return static_cast<int>(
        imprint::acr::run_stage3_prepared_image_u16_to_buffer(
            source, source_sample_count, source_row_stride, rect4, gains3,
            source_floor, cubic, amount, suppress_refine, hot_threshold,
            cold_threshold, tile_width, tile_height, destination,
            destination_sample_count));
}
