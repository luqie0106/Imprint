#pragma once

#include <cstdint>

namespace imprint::acr {

enum class ResampleAxisKind : int {
    resample = 0,
    identity = 1,
    integer_translation = 2,
};

enum class ResampleAxisStatus : int {
    ok = 0,
    invalid_half_taps = 1,
    unsupported_skew = 2,
};

struct ResampleAxisPlan {
    ResampleAxisKind kind = ResampleAxisKind::resample;
    std::int64_t origin = 0;
    std::int64_t step = 0;
    std::int64_t lower = 0;
    std::int64_t upper = 0;
    std::int32_t integer_offset = 0;
};

// Prepares one ordinary factory axis from an already prepared affine tuple and
// the rectangle bounds supplied to that factory. Each binary64 input is
// independently multiplied by 2^32, receives +0.5 when the scaled value is
// nonnegative or -0.5 otherwise, and is converted with FCVTZS semantics:
// NaN becomes zero, out-of-range values saturate to int64 endpoints, and
// in-range values truncate toward zero. The conversion also covers every
// binary64 bit pattern; it does not claim those values can come from a
// constructed coefficient table.
//
// The quantized cross-axis term must be zero. Any failure leaves output
// unchanged. Quantized identity and integer-translation axes do not consume
// half_taps. A general resample axis requires half_taps > 0; its caller must
// ensure it equals the associated coefficient table's tap count divided by
// two. Origin additions wrap modulo 2^64, and rectangle bound arithmetic
// wraps modulo 2^32 before shifting into the high word.
//
// This helper does not prepare a render-level matrix, select a level or layer,
// generate coefficients, or materialize/check the source rectangle/window.
ResampleAxisStatus prepare_ordinary_resample_axis(
    double translation, double along_scale, double cross_scale,
    std::int32_t bound_start, std::int32_t bound_end,
    std::int32_t half_taps, ResampleAxisPlan &output) noexcept;

}  // namespace imprint::acr
