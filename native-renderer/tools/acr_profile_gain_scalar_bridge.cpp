#include "../src/acr_profile_gain_scalar.hpp"

extern "C" int acr_profile_gain_scalar_bridge(double input, double *output) {
    // Bridge-only invalid-pointer status; core statuses remain 0 and 1.
    if (output == nullptr) return 2;

    double candidate = 0.0;
    const auto status =
        imprint::acr::quantize_profile_gain_scalar(input, candidate);
    if (status != imprint::acr::ProfileGainScalarStatus::ok) {
        return static_cast<int>(status);
    }

    *output = candidate;
    return static_cast<int>(status);
}
