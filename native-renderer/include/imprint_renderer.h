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

/* Apply the eight basic preview controls to interleaved RGB8 or RGB16 input.
   Sample counts must equal width*height*3. Parameters are validated before
   output is written, and source/destination buffers must not overlap. */
IMPRINT_API im_status im_native_basic_rgb8(uint32_t width,
                                          uint32_t height,
                                          const uint8_t *source,
                                          size_t source_samples,
                                          const im_basic_params *params,
                                          uint8_t *destination,
                                          size_t destination_samples);
IMPRINT_API im_status im_native_basic_rgb16(uint32_t width,
                                           uint32_t height,
                                           const uint16_t *source,
                                           size_t source_samples,
                                           const im_basic_params *params,
                                           uint16_t *destination,
                                           size_t destination_samples);

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

/* CPU reference operator for automatic spatial dehaze. rgb16 is interleaved
   RGB with width*height pixels; transmission_count must equal width*height;
   airlight_rgb contains three finite RAW-linear RGB values. destination_samples
   must be at least width*height*3. The input is never modified. */
IMPRINT_API im_status im_native_dehaze_spatial_run(const uint16_t *rgb16,
                                                    uint32_t width,
                                                    uint32_t height,
                                                    const float *transmission,
                                                    size_t transmission_count,
                                                    const float *airlight_rgb,
                                                    const im_dehaze_params *params,
                                                    uint16_t *destination,
                                                    size_t destination_samples);

/* Renderer-independent CPU dense operators. All RGB float inputs use
   interleaved normalized linear RGB. */
/* source_values must equal width*height*3. maps interleave absolute source
   (x,y) float coordinates for strip_height*width output pixels, with one
   coordinate pair shared by RGB or three pairs for per-channel correction.
   destination_values must be at least strip_height*width*3. Width, height,
   and strip_height are limited to 32766. interpolation_mode is 0 for OpenCV
   1/32 fixed-map interpolation or 1 for continuous float-map interpolation.
   Inputs are validated before output is written; source and destination must
   not overlap. */
IMPRINT_API im_status im_native_lens_remap_rgb16(uint32_t width,
                                                uint32_t height,
                                                const uint16_t *source,
                                                size_t source_values,
                                                const float *maps,
                                                size_t map_values,
                                                uint32_t strip_height,
                                                uint32_t map_channels,
                                                uint16_t *destination,
                                                size_t destination_values,
                                                uint32_t interpolation_mode);

/* source_values must equal width*height*3, transmission_count width*height,
   and destination_values must be at least width*height*3. Source, transmission,
   airlight, and params are validated before output is written. */
IMPRINT_API im_status im_native_physical_float_run(uint32_t width,
                                                   uint32_t height,
                                                   const float *source,
                                                   size_t source_values,
                                                   const float *transmission,
                                                   size_t transmission_count,
                                                   const float *airlight_rgb,
                                                   const im_dehaze_params *params,
                                                   float *destination,
                                                   size_t destination_values);

/* source_values and result_values must equal width*height*3. The destination
   may be the same buffer as result, but must not overlap source. A finite
   floor_level <= 1e-8 copies result unchanged. */
IMPRINT_API im_status im_native_dark_guard_run(uint32_t width,
                                               uint32_t height,
                                               const float *source,
                                               size_t source_values,
                                               const float *result,
                                               size_t result_values,
                                               float floor_level,
                                               float *destination,
                                               size_t destination_values);

/* source_values must equal width*height*3 and destination_count must be at
   least width*height. The finite scalar LUT contains edge^3 values at
   ((r*edge + g)*edge + b). A non-finite source pixel produces zero; source,
   LUT, and destination buffers must not overlap. */
IMPRINT_API im_status im_native_scalar_lut_run(uint32_t width,
                                               uint32_t height,
                                               const float *source,
                                               size_t source_values,
                                               const float *lut,
                                               size_t lut_values,
                                               uint32_t edge,
                                               float *destination,
                                               size_t destination_count);

/* depth_count and confidence_count must equal width*height; cube_count must
   equal edge^3. total and mass are zeroed, then accumulated in RGB-major
   order. Pixels with non-finite source/depth/confidence or confidence <= 0
   are ignored. The output cubes must not overlap the inputs or each other. */
IMPRINT_API im_status im_native_lut_splat_run(uint32_t width,
                                              uint32_t height,
                                              const float *source,
                                              size_t source_values,
                                              const float *depth,
                                              size_t depth_count,
                                              const float *confidence,
                                              size_t confidence_count,
                                              uint32_t edge,
                                              float depth_min,
                                              float depth_max,
                                              double *total,
                                              double *mass,
                                              size_t cube_count);

/* GPU spatial dehaze using a per-pixel transmission map. The renderer must be
   Metal or D3D12; unsupported backends return an error. Input is unchanged. */
IMPRINT_API im_status im_renderer_render_spatial_full(im_renderer *renderer,
                                                      uint32_t width,
                                                      uint32_t height,
                                                      const uint16_t *rgb16,
                                                      size_t value_count,
                                                      const float *transmission,
                                                      size_t transmission_count,
                                                      const float *airlight_rgb,
                                                      const im_dehaze_params *params,
                                                      uint16_t *destination,
                                                      size_t destination_samples);

/* Confidence-limited physical dehaze on normalized linear float RGB. The
   renderer must provide this optional operator (Metal or D3D12).
   source_values and transmission_count must exactly match width*height*3 and
   width*height; destination_values must be at least width*height*3. All input
   samples and airlight values must be finite and within [0, 1]. */
IMPRINT_API im_status im_renderer_render_physical_float(im_renderer *renderer,
                                                        uint32_t width,
                                                        uint32_t height,
                                                        const float *source,
                                                        size_t source_values,
                                                        const float *transmission,
                                                        size_t transmission_count,
                                                        const float *airlight_rgb,
                                                        const im_dehaze_params *params,
                                                        float *destination,
                                                        size_t destination_values);

/* Curves are 3 x 256 RGB16 samples (R plane, then G, then B).
   A 3D LUT is optional; its RGB16 entries are ordered ((r * edge + g) * edge + b) * 3 + channel. */
IMPRINT_API im_status im_renderer_create(im_backend_kind backend, im_renderer **out_renderer);
IMPRINT_API void im_renderer_destroy(im_renderer *renderer);
IMPRINT_API const char *im_renderer_last_error(const im_renderer *renderer);
IMPRINT_API const char *im_renderer_backend_name(const im_renderer *renderer);
/* Dense full-resolution stages. Sources are normalized finite RGB float32;
   dimensions/counts follow the dense CPU contract and outputs never alias inputs. */
IMPRINT_API im_status im_native_rgb_peak(uint32_t width, uint32_t height,
    const float *source, size_t source_values, float *peak);
IMPRINT_API im_status im_native_exposure_float(uint32_t width, uint32_t height,
    const float *source, size_t source_values, float gain,
    float *destination, size_t destination_values);
IMPRINT_API im_status im_native_refine_transmission(uint32_t width, uint32_t height,
    const float *source, size_t source_values, const float *slope, size_t slope_count,
    const float *intercept, size_t intercept_count, float depth_min, float depth_max,
    float *destination, size_t destination_count);
IMPRINT_API im_status im_native_relief_transmission(uint32_t width, uint32_t height,
    const float *source, size_t source_values, const float *relief, size_t relief_count,
    const float *airlight_rgb, float base, float knee, float initial_t,
    float *destination, size_t destination_count);

/* Same physical float contract, with the shared dark-background guard fused
   into the GPU pass. dark_floor must be finite and within [0, 2]. */
IMPRINT_API im_status im_renderer_render_physical_guarded_float(
    im_renderer *renderer, uint32_t width, uint32_t height,
    const float *source, size_t source_values,
    const float *transmission, size_t transmission_count,
    const float *airlight_rgb, const im_dehaze_params *params, float dark_floor,
    float *destination, size_t destination_values);
IMPRINT_API int im_renderer_supports_physical_guarded_float(const im_renderer *renderer);

/* Returns 1 only if this renderer implements the linear-float physical operator. */
IMPRINT_API int im_renderer_supports_physical_float(const im_renderer *renderer);

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
