#ifndef IMPRINT_RENDERER_H
#define IMPRINT_RENDERER_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(IMPRINT_RENDERER_BUILD)
#    define IMPRINT_API __declspec(dllexport)
#  else
#    define IMPRINT_API __declspec(dllimport)
#  endif
#else
#  define IMPRINT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct im_renderer im_renderer;

typedef enum im_status {
    IM_STATUS_OK = 0,
    IM_STATUS_INVALID_ARGUMENT = 1,
    IM_STATUS_BACKEND_UNAVAILABLE = 2,
    IM_STATUS_NOT_READY = 3,
    IM_STATUS_BUFFER_TOO_SMALL = 4,
    IM_STATUS_RUNTIME_ERROR = 5
} im_status;

typedef enum im_backend_kind {
    IM_BACKEND_AUTO = 0,
    IM_BACKEND_METAL = 1,
    IM_BACKEND_CUDA = 2,
    IM_BACKEND_D3D12 = 3
} im_backend_kind;

/* L0 is the cached preview base. L1 and L2 are cached at approximately 1/2
   and 1/4 size, respectively. FULL receives a newly decoded full-resolution image. */
typedef enum im_render_level {
    IM_RENDER_L0 = 0,
    IM_RENDER_L1 = 1,
    IM_RENDER_L2 = 2,
    IM_RENDER_FULL = 3
} im_render_level;

typedef struct im_dehaze_params {
    float strength;                 /* [0, 1] */
    float naturalness;              /* [0, 1] */
    float fog_retention;             /* [0, 1] */
    float local_contrast;           /* [0, 1] */
    float color_recovery;           /* [0, 1] */
    float color_protection;         /* [0, 1] */
    float highlight_protection;     /* [0, 1] */
    float shadow_protection;        /* [0, 1] */
    float brightness_protection;    /* [0, 1] */
} im_dehaze_params;

typedef struct im_basic_params {
    float exposure;                 /* [-5, 5] EV */
    float contrast;                 /* [-100, 100] */
    float highlights;               /* [-100, 100] */
    float shadows;                  /* [-100, 100] */
    float whites;                   /* [-100, 100] */
    float blacks;                   /* [-100, 100] */
    float vibrance;                 /* [-100, 100] */
    float saturation;               /* [-100, 100] */
} im_basic_params;

typedef struct im_filter_params {
    float curve_mix;                /* [0, 1] */
    float lut_mix;                  /* [0, 1] */
    float exposure_ev;              /* [-8, 8] */
    float contrast;                 /* [0, 4], 1 is neutral */
    float saturation;               /* [0, 3], 1 is neutral */
    float warmth;                   /* [-1, 1] */
    float tint;                     /* [-1, 1] */
    float fade;                     /* [-1, 1] */
} im_filter_params;

/* Curves are 3 x 256 RGB16 samples (R plane, then G, then B).
   A 3D LUT is optional; its RGB16 entries are ordered ((r * edge + g) * edge + b) * 3 + channel. */
IMPRINT_API im_status im_renderer_create(im_backend_kind backend, im_renderer **out_renderer);
IMPRINT_API void im_renderer_destroy(im_renderer *renderer);
IMPRINT_API const char *im_renderer_last_error(const im_renderer *renderer);
IMPRINT_API const char *im_renderer_backend_name(const im_renderer *renderer);

/* Upload Sidecar-decoded preview RGB16 once. L0 uses this exact image; L1/L2 are
   cached downsampled copies. value_count must equal width*height*3. */
IMPRINT_API im_status im_renderer_upload_preview_image(im_renderer *renderer,
                                                        uint32_t width,
                                                        uint32_t height,
                                                        const uint16_t *rgb16,
                                                        size_t value_count);

/* Upload runtime filter controls and optional curve/LUT resources. Pass count zero
   and a null pointer to leave curves or LUT disabled (identity behavior). */
IMPRINT_API im_status im_renderer_upload_filter(im_renderer *renderer,
                                                 const im_filter_params *params,
                                                 const uint16_t *curve_rgb16,
                                                 size_t curve_value_count,
                                                 const uint16_t *lut_rgb16,
                                                 size_t lut_value_count,
                                                 uint32_t lut_edge);

/* Each call uses the latest parameter structs and re-renders from the selected
   cached preview or full-resolution source. The GPU pipeline is dehaze -> basic ->
   filter controls/curves/LUT; no intermediate image is read back to CPU. */
IMPRINT_API im_status im_renderer_render(im_renderer *renderer,
                                          im_render_level level,
                                          const im_dehaze_params *dehaze,
                                          const im_basic_params *basic);

/* Export path: pass a freshly decoded full-resolution source for this render.
   The image is not added to or read from the preview pyramid. */
IMPRINT_API im_status im_renderer_render_full(im_renderer *renderer,
                                               uint32_t width,
                                               uint32_t height,
                                               const uint16_t *rgb16,
                                               size_t value_count,
                                               const im_dehaze_params *dehaze,
                                               const im_basic_params *basic);

/* Render one full-resolution RGB16 image through only the supplied 3D LUT.
   This resets filter controls to neutral and uses zero dehaze/basic controls. */
IMPRINT_API im_status im_renderer_render_ricoh_full(im_renderer *renderer,
                                                     uint32_t width,
                                                     uint32_t height,
                                                     const uint16_t *rgb16,
                                                     size_t value_count,
                                                     const uint16_t *lut_rgb16,
                                                     size_t lut_value_count,
                                                     uint32_t lut_edge);

IMPRINT_API im_status im_renderer_get_output_size(const im_renderer *renderer,
                                                   uint32_t *width,
                                                   uint32_t *height,
                                                   size_t *value_count);
IMPRINT_API im_status im_renderer_copy_output(const im_renderer *renderer,
                                               uint16_t *destination,
                                               size_t destination_value_count);

#ifdef __cplusplus
}
#endif

#endif
