#include "../src/acr_profile_gain_control130.hpp"

#include <cstdint>

extern "C" int acr_profile_gain_control130_bridge(
    std::int32_t current, std::int32_t profile_default,
    std::uint32_t profile_version, double profile_scalar,
    std::int32_t *value, std::uint32_t *wrote, double *blended) {
    // Bridge-only invalid-output status. Core statuses remain 0 and 1.
    if (value == nullptr || wrote == nullptr || blended == nullptr) return 2;

    const imprint::acr::PreparedProfileGainControl130 input{
        current, profile_default, profile_version, profile_scalar};
    imprint::acr::ProfileGainControl130Result result{};
    const auto status = imprint::acr::process_profile_gain_control130(
        input, result);
    if (status != imprint::acr::ProfileGainControl130Status::ok) {
        return static_cast<int>(status);
    }

    *value = result.value;
    *wrote = result.wrote ? 1u : 0u;
    *blended = result.blended;
    return static_cast<int>(status);
}
