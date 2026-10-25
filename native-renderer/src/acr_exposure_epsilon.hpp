#pragma once

#include <cstdint>

namespace imprint::acr {

enum class ExposureEpsilonStatus : int {
    ok = 0,
    invalid_argument = 1,
};

// Prepares the Exposure kernel's pixel epsilon from its recovered source
// fields. Invalid inputs leave output untouched.
ExposureEpsilonStatus prepare_exposure_pixel_epsilon(
    std::uint32_t colorimetric_reference, double source_flare,
    double original_ev90, float *output) noexcept;

}  // namespace imprint::acr
