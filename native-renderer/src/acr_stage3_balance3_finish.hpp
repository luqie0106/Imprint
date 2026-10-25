#pragma once

#include <cstddef>

namespace imprint::acr {

enum class Balance3FinishStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    unsupported_alignment = 4,
    invalid_parameters = 5,
    buffer_too_small = 6,
    arithmetic_overflow = 7,
    overlapping_buffers = 8,
    unsupported_domain = 9,
};

struct Balance3FinishLimits {
    float lower_even = 0.0f;
    float upper_even = 0.0f;
    float lower_odd = 0.0f;
    float upper_odd = 0.0f;
};

// Apply one of the two recovered balance3 finish consumers to a prepared
// binary32 row. `source` points at column zero in the center row and the
// caller supplies radius-one guard rows and columns. `stride` is in float
// samples. `diff` and `dst` each address a contiguous row; their counts are
// capacities in float samples. Width and stride must be divisible by four,
// and source, diff, and dst must be 16-byte aligned. The caller guarantees
// that the complete guarded source allocation is readable.
//
// `phase` and `row_even` are independent binary flags. The selected limits
// and correction sign are derived from `row_even`; lane selection is specific
// to each named entrypoint. The limits are copied before any output is
// written, so they may refer to storage overlapping `dst`. Read-only input
// spans may overlap each other; any conservative read-span overlap with the
// destination is rejected. Rejected calls leave the destination unchanged.
// This is a prepared CPU-row subset, not a full Stage3 implementation or a
// runtime binding choice between the recovered branches.
Balance3FinishStatus balance3_finish_prepared_row_0081403c(
    const float *source, std::size_t stride, const float *diff,
    std::size_t diff_count, std::size_t width, int phase, int row_even,
    const Balance3FinishLimits &limits, float *dst,
    std::size_t dst_count) noexcept;

Balance3FinishStatus balance3_finish_prepared_row_013312fc(
    const float *source, std::size_t stride, const float *diff,
    std::size_t diff_count, std::size_t width, int phase, int row_even,
    const Balance3FinishLimits &limits, float *dst,
    std::size_t dst_count) noexcept;

// Basic inline branch of 007ed860 (007edd3c..007ee150), with the same
// prepared allocation/alignment and atomic rejection contract above.
// phase=(rect.top+rect.left)&1 and row_even=(rect.top&1)==0. Only lanes
// (column&1)!=phase are written; other destination lanes retain their old
// values. This branch clamps the direct fused accumulator to [0,1] without
// subtracting/re-adding the center or applying bounded delta limits.
// The host must select the basic branch from the original raw controls and
// correction-vector state; this API does not infer that state from pixels.
Balance3FinishStatus balance3_basic_prepared_row_007ed860(
    const float *source, std::size_t stride, const float *diff,
    std::size_t diff_count, std::size_t width, int phase, int row_even,
    float *dst, std::size_t dst_count) noexcept;

}  // namespace imprint::acr
