#pragma once

#include "acr_resample_axis.hpp"
#include "acr_resample_coefficients.hpp"
#include "acr_resample_matrix.hpp"
#include "acr_resample_separable.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace imprint::acr {

// Owning ordinary-axis geometry and coefficient tables. The plan stores no
// pointers into its own vectors, so copying or moving it preserves validity.
struct OrdinaryResamplePlan {
    ResampleAxisPlan vertical;
    ResampleAxisPlan horizontal;
    ResampleCoefficientTable vertical_coefficients;
    ResampleCoefficientTable horizontal_coefficients;
};

enum class ResamplePlanStatus : int {
    ok = 0,
    invalid_bounds = 1,
    matrix_failure = 2,
    unsupported_skew = 3,
    unsupported_scale = 4,
    coefficient_failure = 5,
    axis_failure = 6,
};

// Prepare the bounded ordinary affine path for [top, left, bottom, right].
// Both non-wrapping spans must be in [1, INT32_MAX]. This supports no-skew
// matrices with positive general scales and the recovered default filter. For
// a general axis, q is computed from its quantized Q32 step using
// SCVTF -> FMUL(2^-32) -> FDIV(1, scale); it is not computed from the original
// double scale. Coefficient generation retains the helper's 4096-tap cap.
// This is not a normalized request parser, Adobe ABI, or runtime-dispatch
// claim. Failure leaves output unchanged.
ResamplePlanStatus prepare_ordinary_resample_plan(
    const ResampleMatrix3 &matrix,
    const std::array<std::int32_t, 4> &bounds,
    OrdinaryResamplePlan &output) noexcept;

// Execute with the existing ordinary separable executor. Coefficient rows are
// borrowed from plan only for axes classified as general resamples.
ResampleSeparableStatus process_ordinary_resample(
    const SourceView &source, const DestinationRect &destination,
    const OrdinaryResamplePlan &plan,
    std::vector<float> &result) noexcept;

}  // namespace imprint::acr
