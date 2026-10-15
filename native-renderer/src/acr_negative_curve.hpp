#pragma once

#include <cstddef>
#include <vector>

namespace imprint::acr {

enum class NegativeCurveStatus {
    ok = 0,
    input_out_of_range,
    non_finite_intermediate,
};

// Evaluate the recovered SDR negative exposure block for one normalized
// scalar. Inputs must be finite, x must be in [0, 1], and curve_ev18 in
// [-120, 0]. `output` is written only when the function succeeds.
NegativeCurveStatus evaluate_negative_curve_scalar(
    double x, double curve_ev18, double &output) noexcept;

// Probe helper for the generic 00dc62d4 four-segment inverse. The selected
// SDR composite calls this with max(curve_ev18, -1), so its own curve uses
// only the first segment; this helper also permits static checks of all four
// recovered segments for curve_ev18 down to -120 EV.
NegativeCurveStatus evaluate_negative_curve_four_segment_inverse(
    double target, double curve_ev18, double &output) noexcept;

enum class PositiveCurveStatus {
    ok = 0,
    input_out_of_range,
    non_finite_intermediate,
    allocation_failure,
};

inline constexpr std::size_t kPositiveExposureLutSize = 4098;

// Evaluate the recovered positive exposure scalar wrapper. Inputs must be
// finite, x must be in [0, 1], and exposure_ev in [0, 8]. `output` is written
// only when the function succeeds. At 0 EV the gamma encode/decode roundtrip
// is intentionally retained.
PositiveCurveStatus evaluate_positive_exposure_curve_scalar(
    double x, double exposure_ev, double &output) noexcept;

// Build the 4096-interval exposure LUT plus its copied endpoint. The caller's
// vector is replaced only after all scalar samples have been evaluated.
PositiveCurveStatus build_positive_exposure_lut(
    double exposure_ev, std::vector<float> &output) noexcept;

}  // namespace imprint::acr
