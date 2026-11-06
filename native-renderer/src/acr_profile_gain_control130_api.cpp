#include "imprint_renderer.h"
#include "acr_profile_gain_control130.hpp"

#include <cstdint>

im_status im_native_acr_profile_gain_control130(
    int32_t current, int32_t profile_default, uint32_t profile_version,
    double profile_scalar, im_acr_profile_gain_control130_result *output) {
    if (output == nullptr ||
        reinterpret_cast<std::uintptr_t>(output) %
            alignof(im_acr_profile_gain_control130_result) != 0) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    const imprint::acr::PreparedProfileGainControl130 input{
        current, profile_default, profile_version, profile_scalar};
    imprint::acr::ProfileGainControl130Result result{};
    const auto status = imprint::acr::process_profile_gain_control130(input, result);
    if (status != imprint::acr::ProfileGainControl130Status::ok) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    const im_acr_profile_gain_control130_result candidate{
        result.value, result.wrote ? 1u : 0u, result.blended};
    *output = candidate;
    return IM_STATUS_OK;
}
