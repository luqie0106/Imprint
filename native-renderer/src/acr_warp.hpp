#pragma once

#include "acr_bayer_border.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

// Explicit prepared-image contract for OpcodeList3 id1 WarpRectilinear.
// Pixels are interleaved HWC float32 over the complete image_bounds rectangle.
// geometry is {centerX, centerY, aspect}; coefficients contains six doubles
// per parameter plane in {k0, k1, k2, k3, t0, t1} order.
struct WarpSpec {
    BayerRepeatRect image_bounds{};
    std::uint32_t image_planes = 0;
    std::uint32_t param_planes = 0;
    double center_x = 0.5;
    double center_y = 0.5;
    double aspect = 1.0;
    std::size_t tile_width = 0;
    std::size_t tile_height = 0;
};

struct WarpOutput {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t planes = 0;
    // Tightly packed HWC float32.
    std::vector<float> pixels;
};

enum class WarpStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_rect = 2,
    invalid_dimensions = 3,
    invalid_stride = 4,
    buffer_too_small = 5,
    invalid_metadata = 6,
    unsupported_alignment = 7,
    overlap = 8,
    arithmetic_overflow = 9,
    resource_limit = 10,
    non_finite_input = 11,
    unsupported_domain = 12,
    allocation_failure = 13,
    length_error = 14,
};

// Applies an explicit prepared-float32 warp. The complete active source image
// must be finite; finite negative values are allowed. Output is committed only
// after a complete successful render. A NOP warp copies active source bits
// directly, bypassing the edge consumer.
WarpStatus apply_warp_prepared_f32(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const WarpSpec& spec,
    const double* coefficients, std::size_t coefficient_count,
    WarpOutput* output) noexcept;

// Shared buffer adapter. rect4={top,left,bottom,right}; controls2=
// {imagePlanes,paramPlanes}; geometry3={centerX,centerY,aspect}. Destination
// is compact HWC. Every declared span is checked for representability and
// overlap before any input metadata is read, and no destination bytes change
// on failure.
WarpStatus apply_warp_prepared_f32_to_buffer(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const std::int32_t rect4[4],
    const std::uint32_t controls2[2], const double geometry3[3],
    const double* coefficients, std::size_t coefficient_count,
    std::size_t tile_width, std::size_t tile_height, float* destination,
    std::size_t destination_sample_count) noexcept;

}  // namespace imprint::acr
