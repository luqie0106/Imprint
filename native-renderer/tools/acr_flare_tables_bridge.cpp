#include "../src/acr_flare.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int acr_test_flare_integer_lut(std::uint16_t black_offset,
                                           int kind,
                                           std::uint16_t* output,
                                           std::size_t count) noexcept {
    acr_flare::IntegerTableKind table_kind;
    switch (kind) {
        case 0:
            table_kind = acr_flare::IntegerTableKind::black;
            break;
        case 1:
            table_kind = acr_flare::IntegerTableKind::forward;
            break;
        case 2:
            table_kind = acr_flare::IntegerTableKind::inverse;
            break;
        default:
            return static_cast<int>(acr_flare::Status::invalid_argument);
    }
    return static_cast<int>(
        acr_flare::prepare_integer_lut(black_offset, table_kind, output, count));
}

extern "C" int acr_test_flare_calculate(const std::uint16_t* rgb,
                                         std::size_t width,
                                         std::size_t height,
                                         std::size_t crop_x,
                                         std::size_t crop_y,
                                         std::size_t crop_width,
                                         std::size_t crop_height,
                                         std::uint16_t black_offset,
                                         int nonlinear_enabled,
                                         int scene_referred,
                                         acr_flare::Result* output,
                                         std::uint16_t* last_level_rgb,
                                         std::size_t last_level_count) noexcept {
    return static_cast<int>(acr_flare::calculate(
        rgb,
        width,
        height,
        acr_flare::Crop{crop_x, crop_y, crop_width, crop_height},
        black_offset,
        nonlinear_enabled != 0,
        scene_referred != 0,
        output,
        last_level_rgb,
        last_level_count));
}
