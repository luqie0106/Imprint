#include "imprint_renderer.h"
#include "acr_profile_gain_scalar.hpp"

#include <cstdint>

im_status im_native_acr_profile_gain_scalar_quantize(double input, double *output) {
    if (output == nullptr ||
        reinterpret_cast<std::uintptr_t>(output) % alignof(double) != 0) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    double candidate = 0.0;
    if (imprint::acr::quantize_profile_gain_scalar(input, candidate) !=
        imprint::acr::ProfileGainScalarStatus::ok) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    *output = candidate;
    return IM_STATUS_OK;
}
