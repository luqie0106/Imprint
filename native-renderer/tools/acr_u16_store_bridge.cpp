#include "../src/acr_u16_store.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int acr_test_apply_gray_u16_store(
    const float *gray, std::uint16_t *output, std::size_t pixel_count,
    std::uint32_t width, std::uint32_t height, std::int32_t origin_x,
    std::int32_t origin_y, int dither) noexcept {
    if (dither != 0 && dither != 1)
        return static_cast<int>(imprint::acr::U16StoreStatus::invalid_argument);
    return static_cast<int>(imprint::acr::apply_gray_u16_store(
        gray, output, pixel_count, width, height, origin_x, origin_y,
        dither != 0));
}

extern "C" const std::uint16_t *acr_test_u16_dither_tile() noexcept {
    return imprint::acr::get_u16_dither_tile().data();
}

extern "C" std::size_t acr_test_u16_dither_tile_count() noexcept {
    return imprint::acr::get_u16_dither_tile().size();
}
