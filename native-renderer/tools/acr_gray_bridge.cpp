#include "../src/acr_gray.hpp"

#include <cstddef>

static_assert(offsetof(imprint::acr::PreparedRgbToGray, weights_rgb) == 8,
              "Python bridge expects a two-byte pad after the RGB uint16 lanes");

extern "C" int acr_test_prepare_rgb_to_gray(
    const double *coefficients_rgb, std::size_t coefficient_count,
    imprint::acr::PreparedRgbToGray *prepared) noexcept {
    if (coefficient_count != 3)
        return static_cast<int>(imprint::acr::GrayStatus::invalid_argument);
    return static_cast<int>(
        imprint::acr::prepare_rgb_to_gray(coefficients_rgb, prepared));
}

extern "C" int acr_test_apply_rgb_to_gray(
    const float *input_rgb, std::size_t input_element_count,
    const imprint::acr::PreparedRgbToGray *prepared, float *output_gray,
    std::size_t output_count) noexcept {
    if (input_element_count % 3 != 0 ||
        input_element_count / 3 != output_count) {
        return static_cast<int>(imprint::acr::GrayStatus::invalid_argument);
    }
    return static_cast<int>(imprint::acr::apply_rgb_to_gray(
        input_rgb, output_gray, output_count, prepared));
}
