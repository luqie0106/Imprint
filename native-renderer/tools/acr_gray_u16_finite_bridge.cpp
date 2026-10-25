#include "../src/acr_gray.hpp"
#include "../src/acr_u16_store.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

imprint::acr::PreparedRgbToGray make_prepared(std::uint16_t red,
                                              std::uint16_t green,
                                              std::uint16_t blue) noexcept {
    imprint::acr::PreparedRgbToGray prepared{};
    const std::uint16_t lanes[3]{red, green, blue};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        prepared.lanes_rgb[channel] = lanes[channel];
        prepared.weights_rgb[channel] =
            static_cast<float>(lanes[channel]) / 16384.0f;
    }
    return prepared;
}

}  // namespace

extern "C" int acr_test_gray_u16_finite_apply_gray(
    const float *input_rgb, float *output_gray, std::size_t pixel_count,
    std::uint16_t red_lane, std::uint16_t green_lane,
    std::uint16_t blue_lane) noexcept {
    const auto prepared = make_prepared(red_lane, green_lane, blue_lane);
    return static_cast<int>(imprint::acr::apply_rgb_to_gray(
        input_rgb, output_gray, pixel_count, &prepared));
}

extern "C" int acr_test_gray_u16_finite_store(
    const float *gray, std::uint16_t *output, std::size_t pixel_count,
    std::uint32_t width, std::uint32_t height, std::int32_t origin_x,
    std::int32_t origin_y, int dither) noexcept {
    return static_cast<int>(imprint::acr::apply_gray_u16_store(
        gray, output, pixel_count, width, height, origin_x, origin_y,
        dither != 0));
}

extern "C" int acr_test_gray_u16_finite_copy_tile(
    std::uint16_t *output, std::size_t output_count) noexcept {
    const auto &tile = imprint::acr::get_u16_dither_tile();
    if (output == nullptr || output_count != tile.size()) return 1;
    std::copy(tile.begin(), tile.end(), output);
    return 0;
}
