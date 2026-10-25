#include "../src/acr_stage3_core3_colour.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int acr_test_core3_colour(
    const float *source_center,
    const float *green_center,
    std::size_t source_stride,
    std::size_t green_stride,
    std::size_t width,
    std::uint32_t phase,
    float *out_a,
    float *out_b) noexcept {
    return static_cast<int>(imprint::acr::core3_colour_difference_row_008134b8(
        source_center, green_center, source_stride, green_stride, width, phase,
        out_a, out_b));
}
