#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class CombinedCameraStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    buffer_too_small = 4,
    unsupported_alignment = 5,
    overlap = 6,
    arithmetic_overflow = 7,
    resource_limit = 8,
    non_finite_input = 9,
    invalid_controls = 10,
    non_finite_intermediate = 11,
    allocation_failure = 12,
    length_error = 13,
};

// Applies the finite-value ordinary three-plane Camera kernel FUN_013466b4.
// Buffers are HWC-interleaved float32. Row strides and capacities are measured
// in float samples; each active row contains width * 3 samples. `order3[k]`
// selects the source channel within each HWC pixel for sorted gain channel k,
// and destination pixels are always RGB. Counts are sample capacities. The
// final row only needs space through its final active pixel.
//
// controls[13] is [A, B, U, b, M00..M22] in row-major matrix order. The
// columns have already been reordered by descending gain. A>=B>=1,
// float(32768/65535)<=U<=1 and 0<=b<1 are required. Only flags bit zero is
// consulted: zero clamps outputs to [0,1], one preserves finite overrange and
// negative values. Other bits are ignored.
//
// The source and destination may overlap only for exact in-place operation:
// same base pointer, row stride, and declared sample count. All validation and
// computation complete before any destination write; padding is preserved.
// Preflight status order is: null pointers, dimensions, width arithmetic,
// strides, control count, arithmetic/span overflow, resource limit,
// capacities, alignment, overlap, control values, then active source
// finiteness.
CombinedCameraStatus apply_camera_combined_prepared_f32_to_buffer(
    const float *src, std::size_t srcCount, std::size_t srcRowStride,
    std::size_t height, std::size_t width, const double *controls,
    std::size_t controlCount, const std::uint32_t order3[3],
    std::uint32_t flags, float *dst, std::size_t dstCount,
    std::size_t dstRowStride) noexcept;

}  // namespace imprint::acr
