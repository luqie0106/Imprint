#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t imprint_imageio_dimensions(const char *path, uint32_t *width,
                                   uint32_t *height, char *error,
                                   uint32_t error_capacity);
int32_t imprint_imageio_decode_rgba8_srgb(const char *path, uint8_t *rgba,
                                          uint32_t width, uint32_t height,
                                          uint64_t capacity, char *error,
                                          uint32_t error_capacity);

#ifdef __cplusplus
}
#endif
