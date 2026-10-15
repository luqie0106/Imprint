#include "../src/acr_flare.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

static_assert(std::is_standard_layout<acr_flare::Result>::value,
              "flare result must have a stable ctypes-compatible layout");

extern "C" int acr_test_calculate_flare(const std::uint16_t* packed_rgb,
                                         std::size_t image_width,
                                         std::size_t image_height,
                                         std::size_t crop_x,
                                         std::size_t crop_y,
                                         std::size_t crop_width,
                                         std::size_t crop_height,
                                         std::uint16_t black_offset,
                                         int nonlinear_enabled,
                                         int scene_referred,
                                         acr_flare::Result* output,
                                         std::uint16_t* last_level_rgb,
                                         std::size_t last_level_capacity_samples) noexcept {
    if ((nonlinear_enabled != 0 && nonlinear_enabled != 1) ||
        (scene_referred != 0 && scene_referred != 1)) {
        return static_cast<int>(acr_flare::Status::invalid_argument);
    }
    return static_cast<int>(acr_flare::calculate(
        packed_rgb,
        image_width,
        image_height,
        acr_flare::Crop{crop_x, crop_y, crop_width, crop_height},
        black_offset,
        nonlinear_enabled != 0,
        scene_referred != 0,
        output,
        last_level_rgb,
        last_level_capacity_samples));
}
