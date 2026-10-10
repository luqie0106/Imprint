#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint {

bool validate_warp_rectilinear_rgb16(uint32_t width, uint32_t height,
                                     const uint16_t *source, size_t source_values,
                                     const float *constants, size_t constant_values,
                                     uint16_t *destination, size_t destination_values,
                                     size_t *pixel_count, float *radius);

} // namespace imprint
