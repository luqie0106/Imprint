#pragma once

#include "acr_float_lut.hpp"

namespace imprint::acr {

struct Stage3CurvePrepared {
    double pivot;
    double log_black;
    bool cubic;
};

enum class Stage3CurveStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_parameters = 2,
    overlap = 3,
    evaluation_failure = 4,
};

// Prepared scalar state for the recovered Stage3 log/cubic pair. The cubic
// selection is explicit state corresponding to the caller's mode bit; this
// helper does not infer or bind that metadata automatically.
Stage3CurveStatus prepare_stage3_curve(
    double pivot, bool cubic, Stage3CurvePrepared* out) noexcept;

// P2 order: forward is L(clamp(cubic ? C(sample) : sample)); inverse is
// cubic ? C^-1(L^-1(sample)) : L^-1(sample). Every failure leaves `out`
// unchanged. The prepared value must match prepare_stage3_curve exactly.
Stage3CurveStatus evaluate_stage3_curve(
    double sample, const Stage3CurvePrepared& prepared, bool inverse,
    double* out) noexcept;

// Builds the recovered 4096-segment float LUT using the shared 01300490
// builder. `prepared` is the explicit scalar mode state above.
Stage3CurveStatus build_stage3_curve_lut(
    const Stage3CurvePrepared& prepared, bool inverse,
    FloatLutPrepared* out) noexcept;

}  // namespace imprint::acr
