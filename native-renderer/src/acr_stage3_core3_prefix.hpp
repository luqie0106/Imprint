#pragma once

#include <cstddef>
#include <vector>

namespace imprint::acr {

// Statuses for the prepared-float prefix composition. Numeric values are
// stable so callers can report failures without depending on exception text.
enum class Core3PrefixStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    buffer_too_small = 4,
    unsupported_alignment = 5,
    resource_limit = 6,
    allocation_failure = 7,
    arithmetic_overflow = 8,
    kernel_failure = 9,
};

// Half-open tile-logical rectangle: [top,bottom) x [left,right).
struct Core3PrefixRegion {
    std::size_t top = 0;
    std::size_t left = 0;
    std::size_t bottom = 0;
    std::size_t right = 0;
};

// The recovered executor allocates nine float planes for each spatial row.
// A sample at (y, plane, x) is stored at
//   samples[y * row_stride + plane * plane_stride + x].
// `pitch` includes four samples of trailing padding after the logical width.
// Undefined regions, including plane 6 and all padding, remain quiet NaN.
struct Core3PrefixWorkspace {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t pitch = 0;
    std::size_t plane_stride = 0;
    std::size_t row_stride = 0;
    Core3PrefixRegion gradient_valid{};
    Core3PrefixRegion energy_valid{};
    std::vector<float> samples;
};

// Compose only the first two recovered core3 CPU stages over an explicitly
// prepared float tile. This does not decode CFA input, produce full-resolution
// RGB, or model the remaining workspace planes/consumer.
//
// Input layout: `padded_input` points to row 0's left guard. Every row has four
// left guard floats, `width` logical samples, at least four right guard floats,
// and `input_row_stride` samples to the next row. The base and row stride must
// preserve 16-byte alignment. Supported dimensions are width >= 12, width a
// multiple of four, and height >= 9. The input row stride must be at least
// width + 8 samples and a multiple of four. Workspace allocation is capped at
// 64 * 1024 * 1024 float samples. The caller remains responsible for the actual
// allocation capacity represented by `input_sample_count`.
//
// The original executor's gradient ROI starts at inset 2; that pointer can
// trigger its alignment backup and rounded tail stores. For widths divisible
// by four, this prepared path starts at aligned tile-logical x=0 and computes
// x=[0,width), the corresponding subset of those values. This API does not
// emulate the original backup or tail writes and is not a full ABI replica.
Core3PrefixStatus run_core3_gradient_energy_prefix(
    const float *padded_input,
    std::size_t input_sample_count,
    std::size_t width,
    std::size_t height,
    std::size_t input_row_stride,
    Core3PrefixWorkspace *output) noexcept;

}  // namespace imprint::acr
