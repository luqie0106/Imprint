#ifndef IMPRINT_GEO_NATIVE_PREVIEW_H
#define IMPRINT_GEO_NATIVE_PREVIEW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Run an identity-checked full-resolution pass through the existing native GPU
 * renderer. Returns 1 on success and 0 on error. The renderer library is loaded
 * from library_path at runtime; no renderer library is linked into this adapter.
 *
 * On success, out_rgb contains exactly the input RGB8 bytes and backend names
 * the backend actually selected by the renderer. On error, out_rgb is untouched;
 * backend is "unavailable" if no renderer instance was available, and error
 * contains a short diagnostic when its capacity is nonzero.
 *
 * rgb and out_rgb may alias. The metadata buffers must not overlap either pixel
 * buffer or each other. Pixel buffers contain width * height * 3 bytes.
 */
int32_t imprint_geo_native_preview(const char *library_path,
                                   const uint8_t *rgb,
                                   uint32_t width,
                                   uint32_t height,
                                   uint8_t *out_rgb,
                                   char *backend,
                                   uint32_t backend_cap,
                                   char *error,
                                   uint32_t errcap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IMPRINT_GEO_NATIVE_PREVIEW_H */
