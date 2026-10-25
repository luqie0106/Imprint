#include "acr_stage3_balance3_controls.hpp"
extern "C" int acr_test_balance3_controls(double lower, double upper, int direct, float *out) {
    if (!out) return 1;
    imprint::acr::Balance3FinishLimits result;
    auto status = imprint::acr::prepare_balance3_finish_limits_007ed860(lower, upper, direct != 0, &result);
    if (status != imprint::acr::Balance3ControlsStatus::ok) return static_cast<int>(status);
    out[0] = result.lower_even;
    out[1] = result.upper_even;
    out[2] = result.lower_odd;
    out[3] = result.upper_odd;
    return 0;
}
