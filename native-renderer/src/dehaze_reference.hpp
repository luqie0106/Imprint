#pragma once

#include "backend.hpp"

#include <cstddef>
#include <cstdint>

namespace imprint {

ImageStats dehaze_image_stats(const uint16_t *rgb16, size_t pixels);
float dehaze_brightness_gain(const uint16_t *rgb16, size_t pixels,
                             const im_dehaze_params &params, const ImageStats &stats);
float dehaze_spatial_brightness_gain(const uint16_t *rgb16, uint32_t width, uint32_t height,
                                    const float *transmission, const float *airlight_rgb,
                                    const im_dehaze_params &params);
void apply_dehaze_reference(const uint16_t *rgb16, size_t pixels,
                            const im_dehaze_params &params, ImageStats stats,
                            uint16_t *destination);
void apply_dehaze_spatial_reference(const uint16_t *rgb16, uint32_t width, uint32_t height,
                                    const float *transmission,
                                    const float *airlight_rgb,
                                    const im_dehaze_params &params,
                                    uint16_t *destination);

} // namespace imprint
