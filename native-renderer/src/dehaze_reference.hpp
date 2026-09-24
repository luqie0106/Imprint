#pragma once

#include "backend.hpp"

#include <cstddef>
#include <cstdint>

namespace imprint {

ImageStats dehaze_image_stats(const uint16_t *rgb16, size_t pixels);
float dehaze_brightness_gain(const uint16_t *rgb16, size_t pixels,
                             const im_dehaze_params &params, const ImageStats &stats);
void apply_dehaze_reference(const uint16_t *rgb16, size_t pixels,
                            const im_dehaze_params &params, ImageStats stats,
                            uint16_t *destination);

} // namespace imprint
