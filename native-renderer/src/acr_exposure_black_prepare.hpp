#pragma once

#include <cstdint>

namespace imprint::acr {

enum class BlackPrepareStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_curve = 2,
    non_finite_result = 3,
};

struct PreparedBlackResponse {
    double z;
    double post_strength;
    double channel_strength;
};

// Evaluate the closed Camera Raw black-control response for prepared inputs.
// Inputs must be finite and satisfy t0<t1<1. Unlike the producer below, this
// helper does not restrict q. Nonfinite roots are scanned/discarded as in the
// source; FMINNM may recover nonfinite intermediates. Constructed curve points
// and the final result must be finite. Original constructor rejection becomes
// invalid_curve. Output must be aligned and have a nonoverflowing address span;
// it is unchanged when validation or evaluation fails.
BlackPrepareStatus evaluate_black_control_response(
    double z, double t0, double t1, double k, double q,
    double* output) noexcept;

// Prepare the three scalar black-control values. gain_adjust_active and mode
// are resolved caller inputs; this function does not infer profile/runtime
// state or add a UI range to either integer parameter. k must be finite and q
// already prepared in [0.0625,0.9]. Success atomically writes all three values.
BlackPrepareStatus prepare_exposure_black_response(
    std::int32_t blacks, std::int32_t profile_gain_param,
    bool gain_adjust_active, bool mode, double k, double q,
    PreparedBlackResponse* output) noexcept;

}  // namespace imprint::acr
