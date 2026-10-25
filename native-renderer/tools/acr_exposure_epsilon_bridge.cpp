#include "../src/acr_exposure_epsilon.hpp"

#include <cstdint>

extern "C" int acr_test_exposure_pixel_epsilon(
    std::uint32_t colorimetric_reference, double source_flare,
    double original_ev90, float *output) noexcept {
    return static_cast<int>(imprint::acr::prepare_exposure_pixel_epsilon(
        colorimetric_reference, source_flare, original_ev90, output));
}
