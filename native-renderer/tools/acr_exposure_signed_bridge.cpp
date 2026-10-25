#include "../src/acr_exposure_signed.hpp"

#include <cstddef>

extern "C" int acr_test_signed_exposure(
    const float *input, float *output, std::size_t pixel_count,
    const float *mask, std::size_t mask_count,
    const imprint::acr::PreparedSignedExposure *prepared) noexcept {
    return static_cast<int>(imprint::acr::apply_prepared_signed_exposure_rgb(
        input, output, pixel_count, mask, mask_count, prepared));
}
