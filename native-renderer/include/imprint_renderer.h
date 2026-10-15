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
typedef struct im_shared_buffer im_shared_buffer;

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

/* Recovered xy -> temperature/tint projection, using the fixed 31-row table.
 * output_values must be 2. The API accepts finite physical xy chromaticities
 * (x >= 0, y > 0, x+y < 1). Invalid input leaves output unchanged.
 */
IMPRINT_API im_status im_native_acr_white_temperature(
    double x, double y, double *output, size_t output_values);

/* Recovered modern SDR color stages on float32 linear ProPhoto RGB.
   These are individual CPU operators, not a complete Adobe rendering pipeline.
   HSL controls are 24 doubles: Hue[8], Saturation[8], Luminance[8], in
   Red/Orange/Yellow/Green/Aqua/Blue/Purple/Magenta order (-100..100).
   A 25th optional double is Vibrance (-100..100), executed in the same MMHG
   stage before HSL while retaining the original chroma for Luminance weighting.
   Grading controls are 14 doubles: shadow/mid/highlight/global Hue, Sat, Lum,
   followed by Blending and Balance. Hue=0..360, Sat=0..100, Lum=-100..100,
   Blending=0..100, Balance=-100..100. The normal color branch is used.
   Input must be finite in [0,1]; sample counts equal width*height*3.
   Invalid input is rejected before writing output. Buffers must not overlap. */
IMPRINT_API im_status im_native_acr_hsl_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *controls, size_t control_count, float *destination,
    size_t destination_samples);
/* Rendered-input color path: inverse default tone -> HSL/Vibrance -> default
   tone. Same control and buffer contract. Do not apply this normalization to
   a scene-linear camera stage that has not yet received its forward tone. */
IMPRINT_API im_status im_native_acr_rendered_hsl_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *controls, size_t control_count, float *destination,
    size_t destination_samples);
IMPRINT_API im_status im_native_acr_grading_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *controls, size_t control_count, float *destination,
    size_t destination_samples);

/* Explicit profile-only SDR curve: natural spline sampled into a 4096-segment
   float LUT, followed by ordinary RGB tone. Points are interleaved double x,y,
   with point_count >= 2; x strictly increases, x/y finite in [0,1]. Endpoint
   extrapolation is constant. modern must be 0 or 1 (MMHG coupling).
   This does not select Adobe's source override, process version or luma stage.
   Input finite [0,1]; destination must not overlap source or points.
   Invalid input leaves destination unchanged. */
IMPRINT_API im_status im_native_acr_profile_tone_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *points_xy, size_t point_count, int modern,
    float *destination, size_t destination_samples);

/* Same profile/MMHG path with the explicit recovered SDR negative curveEV18
   block composed before profile-spline evaluation while sampling the one LUT.
   curveEV18 is a prepared context coordinate in [-120,0], not manual Exposure.
   Zero is exactly the profile-only path. Positive/HDR/main-curve source
   overrides and other scalar operators remain outside this interface.
   Failure preserves destination, with the same buffer contract as above. */
IMPRINT_API im_status im_native_acr_profile_tone_exposure_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *points_xy, size_t point_count, int modern, double curve_ev18,
    float *destination, size_t destination_samples);

/* Modern exposure preparation with explicit, already-resolved source inputs.
   Controls[10]: manual Exposure x100 integer, Whites UI integer, resolved source
   EV, source scale, LinearResponseLimit, strongest Q12 gain integer,
   three-color-plane flag, HDR edit flag, prepared limit EV88, prepared ProfileGainTableMap amount (context +0x58).
   Output[19]: manual EV, Whites EV, source EV, source scale, context EV50,
   limit EV88, profile gain amount, first/second arguments, exposure gain,
   EV90, EV98, EV100, EV108, EV112, negative-curve EV18 change, camera scale,
   rounded source limit, source limit uint16 value (stored exactly as double).
   Zero ProfileGainTableMap amount allocates at most -1 EV per call to the
   negative curve and retains remaining negative exposure in EV98. Context+B8
   (scene gray-vs-max log difference) is a distinct field, not this input.
   Caller must apply source-specific manual Exposure limits first. This API does
   not infer metadata, WB/matrix gains, flare, PV or profile-gain-table selection.
   Invalid inputs or overlapping buffers leave output unchanged. */
IMPRINT_API im_status im_native_acr_prepare_exposure(
    const double *controls, size_t control_count, double *output, size_t output_count);

/* Three-plane prepared source-log statistics -> camera -> exposure-limit CPU.
   log_rgb is interleaved finite float32 log samples, before the camera matrix;
   it is NOT display RGB, a gray guide, or automatically decoded RAW pixels.
   epsilon is the resolved source log epsilon in (0,2^-12].
   Controls[17]: original-plane permutation[3], sorted camera A/B/U/S[4],
   sorted camera matrix[9 row-major], temporary EV90 post-scale.
   Output[15]: filtered high log[3], post-fallback populations[3], linear
   original-plane values[3], camera/post-scale RGB[3], limitEV88, contextB8
   gray-minus-max log difference, float32 post-scale represented as double.
   No source pixel, pyramid, black, default-curve or metadata preparation.
   Unsupported/invalid input and aliases leave the complete output unchanged. */
IMPRINT_API im_status im_native_acr_exposure_limit_log_rgbf32(
    uint32_t width, uint32_t height, const float *log_rgb, size_t source_samples,
    const double *controls, size_t control_count, double epsilon,
    double *output, size_t output_count);

/* Explicit zero-pedestal, scene-referred uint16 source subset, no Stage4
   filters and identity source-default curve. Caller must establish that domain.
   Builds the ordinary crop pyramid once, selects Flare and statistics levels,
   normalizes uint16 with float reciprocal multiplication, computes source log,
   per-plane high statistics and the prepared camera exposure limit in C++.
   Controls[22]: crop x/y/width/height, nonlinear-pyramid flag, then the 17
   prepared-camera controls documented above. Does not infer these from a RAW.
   Output[24]: Flare mean/flare/epsilon/coarsest width/height/lower-level count,
   statistics level/width/height, followed by the 15 outputs above.
   Nonzero internal black, float source, nonidentity source-default curve,
   Stage4 filters, metadata/decoder equivalence are outside this interface.
   Invalid inputs or aliases leave the entire output unchanged. */
IMPRINT_API im_status im_native_acr_source_limit_scene_u16_rgb(
    uint32_t width, uint32_t height, const uint16_t *source, size_t source_samples,
    const double *controls, size_t control_count, double *output, size_t output_count);

/* Finite RGB single/two-illuminant ColorSpec constructor + SetWhiteXY CPU.
   Controls[64]: AnalogBalance[3], raw ColorMatrix1/2[9 each], caller-selected
   CameraCalibration1/2[9 each], ForwardMatrix1/2[9 each], endpoint Kelvin1/2,
   already-resolved white xy[2], hasSecondColorMatrix, hasForward1, hasForward2.
   All matrices are row-major. All values must be finite; AB must be positive;
   flags are exactly 0/1. Caller chooses compatible calibration and endpoints.
   Output[22]: CameraWhite[3], CameraToPCS[9], PCStoCamera[9], first-slot weight.
   White xy must satisfy x>=0, y>0, x+y<1 (outer projection safety boundary).
   It is projected for interpolation; XYtoXYZ independently uses its
   recovered clipping rule. No metadata, NeutralToXY, xy quantization, automatic
   white selection, three-illuminant or non-square matrix path is performed.
   Explicit 3x3 inverse uses FMA; other arithmetic is strict unfused scalar.
   Invalid input, singular/nonfinite intermediate or overlap leaves output unchanged. */
IMPRINT_API im_status im_native_acr_prepare_colorspec(
    const double *controls, size_t control_count, double *output, size_t output_count);

/* Prepare exactly three CameraNeutral values: source max normalization and
   signed-int32 rounding at 1e6 followed by reciprocal-factor multiplication.
   Finite negative entries are supported, but maximum must be positive.
   This is a numerical source-setter subset, not a metadata parser. Empty or
   nonpositive source setters clear metadata in Adobe; this fixed-size API
   rejects them instead. Invalid input/overflow/overlap leaves output unchanged. */
IMPRINT_API im_status im_native_acr_prepare_camera_neutral(
    const double *neutral, size_t count, double *output, size_t output_count);

/* Two-illuminant RGB neutral -> adjusted white -> ColorSpec CPU resolution.
   Controls[68]: the same first 64 fields as im_native_acr_prepare_colorspec
   (white xy fields are finite placeholders ignored here), prepared source
   CameraNeutral[3], output-referred mode flag (exactly 0/1).
   Neutral must already be max-normalized and quantized by its source reader;
   finite negative entries are allowed, but its maximum must be positive.
   Mode equals the source ColorimetricReference != 0 predicate. Caller selects
   this metadata predicate; no automatic-WB or metadata inference is performed.
   Output[26]: neutral xy[2], adjusted xy[2], then the 22 ColorSpec fields.
   Intermediate white projection uses the existing physical xy safety boundary.
   Invalid input, singular/nonfinite intermediate or overlap leaves output unchanged. */
IMPRINT_API im_status im_native_acr_resolve_white(
    const double *controls, size_t control_count, double *output, size_t output_count);

/* Combined explicit native profile-camera parameter preparation.
   Controls[75]: first 68 fields of im_native_acr_resolve_white, except the
   supplied neutral[3] is raw and internally max-normalized/quantized; then
   six calibration controls and ShadowTint (integers [-100,100]).
   Caller supplies compatible calibration, AnalogBalance and source mode.
   Output[65]: resolved-white fields[26], camera-preparation fields[30], then
   normalized ProPhotoToPCS matrix[9]. The ProPhoto inverse is built in C++.
   This does not read metadata or apply pixels, black compensation, exposure
   or scene tone mapping. Invalid/overlap leaves output unchanged. */
IMPRINT_API im_status im_native_acr_prepare_profile_camera(
    const double *controls, size_t control_count, double *output, size_t output_count);

/* Imprint display projection for the explicit normalized ProPhoto working
   matrix. This is the existing sRGB display math migrated to CPU C++, not an
   Adobe ICC display-transform reconstruction. Nonnegative finite float32 RGB
   (including values >1) is accepted; output is clipped sRGB8, ties-to-even.
   Q has nine finite row-major double values. All buffers must be pairwise
   disjoint; samples equal width*height*3 and at most 2^26 pixels.
   Invalid/nonfinite intermediate/allocation failure leaves output unchanged. */
IMPRINT_API im_status im_native_acr_project_srgb_rgb8(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *normalized_prophoto_to_pcs, size_t matrix_count,
    uint8_t *destination, size_t destination_samples);

/* Native preservation of Imprint's existing clipped linear gain, pending
   the complete recovered exposure stage. Source finite [0,1], gain float32
   finite >=0; output min(source*gain,1). Not an Adobe exposure reconstruction.
   Source/destination may not overlap; failure leaves destination unchanged. */
IMPRINT_API im_status im_native_acr_scale_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    float gain, float *destination, size_t destination_samples);

/* Explicit unsigned16 packed-RGB ordinary edge2 flare-source subset.
   Controls[7] integer doubles: crop x/y/width/height, internal B16 black offset
   (0..65534), nonlinear-enabled flag, scene-referred flag (flags exactly 0/1).
   Crop must be positive and contained in actual image bounds. Ordinary borders
   clamp to actual image bounds; odd crop may read a pixel outside the crop.
   If both original crop dimensions <=64, no lower levels are built; otherwise
   five levels use round(originalCropDimension/2^(n+1)), not recursive ceil.
   Output[6]: normalized mean, flare, selected epsilon, last width/height,
   number of lower levels (0 or 5). At most 2^26 pixels; counts width*height*3.
   Caller supplies a decoded/prepared source domain and resolved flags; this
   does not infer RAW metadata or signed16/float/panorama/precomputed levels.
   Invalid/allocation failure or output overlap leaves output unchanged. */
IMPRINT_API im_status im_native_acr_flare_u16_rgb(
    uint32_t width, uint32_t height, const uint16_t *source, size_t source_samples,
    const double *controls, size_t control_count, double *output, size_t output_count);

/* Prepared three-plane ColorSpec camera parameter factory, CPU only.
   Controls[31]: CameraWhite[3], row-major CameraToPCS[9], normalized
   ProPhoto inverse[9], prepared AnalogBalance-derived source vector[3],
   integer RedHue/RedSat/GreenHue/GreenSat/BlueHue/BlueSat[6], integer ShadowTint.
   CameraWhite must be in [0.001,1] with maximum 1 or its preceding double
   (reciprocal-normalization rounding); source vector must
   be positive. Calibration and ShadowTint must be integers in [-100,100].
   Output[30]: unsorted float matrix[9], column-sorted float matrix[9],
   unsorted Q12 gains[3], unsorted float gains[3], black compensation[3],
   sorted source-plane indices[3]. Float/integer values are carried as doubles.
   Caller resolves metadata, white balance and profile before calling; this
   does not apply black compensation or transform image pixels.
   Invalid input, nonfinite intermediate or overlap leaves output unchanged. */
IMPRINT_API im_status im_native_acr_prepare_camera_stage(
    const double *controls, size_t control_count, double *output, size_t output_count);

/* Recovered camera stage on finite, nonnegative, interleaved float32 camera
   RGB that has already been sorted by descending channel gain. Values above
   one are supported. The 13 controls are [gain_max, gain_middle, upper,
   scale, row-major 3x3 matrix], and output channels have fixed ProPhoto RGB
   semantics. This is a CPU scalar operator, not a complete camera profile or
   rendering pipeline. Sample counts equal width*height*3. Destination must
   not overlap source or controls. Invalid input leaves destination unchanged. */
IMPRINT_API im_status im_native_acr_camera_float_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *controls, size_t control_count, float *destination,
    size_t destination_samples);

/* Recovered CPU tone subset on two prepared float32 RGB images of identical
   geometry. guide_rgb is the domain entering MakeGrayLog after camera and
   earlier preparation; tone_rgb is the domain at the enabled tone stage.
   Neither is assumed to be arbitrary sRGB. Controls are [source epsilon,
   EV90, EV98, EV100, Shadows2012 UI, Highlights2012 UI, positive
   Clarity2012 UI, sigma mode]. Negative clarity and GPU execution are not
   supported by this recovered operator. Sample counts equal width*height*3.
   Inputs must be finite and nonnegative. Destination must not overlap either
   source or controls; guide and tone may alias. Invalid input leaves
   destination unchanged. */
IMPRINT_API im_status im_native_acr_tone_prepared_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *guide_rgb,
    const float *tone_rgb, size_t samples, const double *controls,
    size_t control_count, float *destination, size_t destination_samples);

/* Same prepared SDR contract with nine controls: the preceding eight and
   output mode (0=ordinary RGB ratio, 1=clamped mapped gray in all channels).
   Calling mode 1 explicitly means the tone stage is present; it does not
   decide whether ACR would construct that stage. A zero-control mode-1 call
   still evaluates the prepared model. This is not automatic RAW statistics. */
IMPRINT_API im_status im_native_acr_tone_prepared_sdr_mode_rgbf32(
    uint32_t width, uint32_t height, const float *guide_rgb,
    const float *tone_rgb, size_t samples, const double *controls,
    size_t control_count, float *destination, size_t destination_samples);

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
/* Version 2 applies exposure as linear-light gain to display-sRGB samples.
   Callers requiring that transfer must probe these symbols; older libraries
   may expose only the original basic ABI. */
IMPRINT_API im_status im_native_basic_v2_rgb8(uint32_t width,
                                              uint32_t height,
                                              const uint8_t *source,
                                              size_t source_samples,
                                              const im_basic_params *params,
                                              uint8_t *destination,
                                              size_t destination_samples);
IMPRINT_API im_status im_native_basic_v2_rgb16(uint32_t width,
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
/* Convert interleaved RGB integer samples to normalized float32 linear RGB.
   Sample counts must equal width*height*3; source and destination buffers
   must not overlap. The caller owns both buffers. */
IMPRINT_API im_status im_native_rgb8_to_linear_float(uint32_t width,
                                                    uint32_t height,
                                                    const uint8_t *source,
                                                    size_t source_samples,
                                                    float *destination,
                                                    size_t destination_values);
IMPRINT_API im_status im_native_rgb16_to_linear_float(uint32_t width,
                                                     uint32_t height,
                                                     const uint16_t *source,
                                                     size_t source_samples,
                                                     float *destination,
                                                     size_t destination_values);
/* Quantize interleaved finite linear float32 RGB using NumPy-compatible
   float32 multiplication by 65535, ties-to-even rounding, and clipping.
   Sample counts must equal width*height*3; buffers must not overlap. */
IMPRINT_API im_status im_native_linear_float_to_rgb16(uint32_t width,
                                                      uint32_t height,
                                                      const float *source,
                                                      size_t source_values,
                                                      uint16_t *destination,
                                                      size_t destination_samples);
/* Expand RGB8 to RGB16 with the exact x*257 mapping, or compact RGB16 to RGB8
   with integer (x+128)/257 rounding. Sample counts must equal width*height*3. */
IMPRINT_API im_status im_native_expand_rgb8_to_rgb16(uint32_t width,
                                                     uint32_t height,
                                                     const uint8_t *source,
                                                     size_t source_samples,
                                                     uint16_t *destination,
                                                     size_t destination_samples);
IMPRINT_API im_status im_native_compact_rgb16_to_rgb8(uint32_t width,
                                                     uint32_t height,
                                                     const uint16_t *source,
                                                     size_t source_samples,
                                                     uint8_t *destination,
                                                     size_t destination_samples);
/* Return 1 only when every sample is finite and within the inclusive range.
   Returns 0 for invalid arguments or for the first invalid sample. */
IMPRINT_API int im_native_float_range_valid(const float *source,
                                            size_t source_values,
                                            float minimum,
                                            float maximum);
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

/* Apply Adobe WarpRectilinear sampling to an interleaved RGB16 image.
   constants contains three per-channel [k0,k1,k2,k3,t0,t1] groups followed
   by normalized center_x, normalized center_y, and vertical pixel scale.
   Source and destination counts must equal width*height*3; buffers must not
   overlap. */
IMPRINT_API im_status im_native_warp_rectilinear_rgb16(
    uint32_t width, uint32_t height,
    const uint16_t *source, size_t source_values,
    const float *constants, size_t constant_values,
    uint16_t *destination, size_t destination_values);

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
/* Streaming Metal RGB16 Lensfun remap. The source is uploaded once at begin;
   strips write into a full-size backend output and finish reads it back. Only
   backends that explicitly report support may be used. Coordinates are
   absolute source coordinates, with REFLECT101 borders. */
IMPRINT_API int im_renderer_supports_lens_remap(const im_renderer *renderer);
IMPRINT_API im_status im_renderer_lens_remap_begin(im_renderer *renderer,
                                                   uint32_t width,
                                                   uint32_t height,
                                                   const uint16_t *source,
                                                   size_t source_values);
IMPRINT_API im_status im_renderer_lens_remap_strip(im_renderer *renderer,
                                                   const float *maps,
                                                   size_t map_values,
                                                   uint32_t strip_y,
                                                   uint32_t strip_height,
                                                   uint32_t map_channels,
                                                   uint32_t interpolation_mode);
IMPRINT_API im_status im_renderer_lens_remap_finish(im_renderer *renderer,
                                                    uint16_t *destination,
                                                    size_t destination_values);
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

/* Camera profile render. constants22 is camera-to-ProPhoto row-major 3x3,
   white RGB, exposure gain, then ProPhoto-to-sRGB row-major 3x3. look_table
   is [value,hue,saturation,channel]; tone_curve is interleaved [x,y] pairs.
   Counts are exact interleaved RGB sample/value counts. Output never aliases
   any input. look_encoding is 0 for linear value or 1 for encoded sRGB value. */
IMPRINT_API im_status im_native_camera_profile_render_rgb16_to_rgb8(
    uint32_t width, uint32_t height,
    const uint16_t *camera_rgb, size_t source_values,
    const float *constants22, size_t constants_count,
    const float *look_table, size_t look_values,
    uint32_t hue_count, uint32_t saturation_count, uint32_t value_count,
    uint32_t look_encoding,
    const float *tone_curve, size_t tone_values,
    uint8_t *destination, size_t destination_values);

/* Transfer enhancement using processed/reference luma ratio and color residual.
   inverse_matrix9 is row-major and residual is multiplied on the left.
   Counts are exact interleaved RGB sample counts. Output never aliases inputs. */
IMPRINT_API im_status im_native_camera_profile_transfer_rgb16(
    uint32_t width, uint32_t height,
    const uint16_t *camera_rgb, size_t camera_values,
    const uint16_t *reference_rgb, size_t reference_values,
    const uint16_t *processed_rgb, size_t processed_values,
    const float *inverse_matrix9, size_t inverse_count,
    uint16_t *destination, size_t destination_values);

/* GPU camera-profile rendering keeps one immutable interleaved RGB16 source
   resident on the selected renderer. Upload a replacement source explicitly;
   look/tone resources are cached when their contents are unchanged. */
IMPRINT_API im_status im_renderer_set_camera_profile_source(
    im_renderer *renderer, uint32_t width, uint32_t height,
    const uint16_t *camera_rgb, size_t source_values);
IMPRINT_API im_status im_renderer_render_camera_profile(
    im_renderer *renderer, const float *constants22, size_t constants_count,
    const float *look_table, size_t look_values,
    uint32_t hue_count, uint32_t saturation_count, uint32_t value_count,
    uint32_t look_encoding, const float *tone_curve, size_t tone_values,
    uint8_t *destination, size_t destination_values);
IMPRINT_API im_status im_renderer_clear_camera_profile_source(im_renderer *renderer);
IMPRINT_API int im_renderer_supports_camera_profile_render(const im_renderer *renderer);

/* Dense enhancement transfer in camera space. Source pairs are copied into
   renderer-owned GPU storage; each render preserves uint16 quantization. */
IMPRINT_API im_status im_renderer_set_camera_profile_transfer_source(
    im_renderer *renderer, uint32_t width, uint32_t height,
    const uint16_t *camera_rgb, const uint16_t *reference_rgb, size_t source_values);
IMPRINT_API im_status im_renderer_render_camera_profile_transfer(
    im_renderer *renderer, const uint16_t *processed_rgb, size_t processed_values,
    const float *inverse_matrix9, size_t inverse_count,
    uint16_t *destination, size_t destination_values);
IMPRINT_API int im_renderer_supports_camera_profile_transfer(const im_renderer *renderer);

/* Metal shared-memory owners. byte_count is in bytes, must be nonzero, and
   is limited to 1 GiB. The owner remains valid after its renderer is destroyed. */
IMPRINT_API im_status im_renderer_create_shared_buffer(im_renderer *renderer,
                                                       size_t byte_count,
                                                       im_shared_buffer **out_buffer);
IMPRINT_API void *im_shared_buffer_data(const im_shared_buffer *buffer);
IMPRINT_API size_t im_shared_buffer_size(const im_shared_buffer *buffer);
IMPRINT_API void im_shared_buffer_destroy(im_shared_buffer *buffer);

/* Transfer RGB16 samples directly between Metal shared buffers. Both owners
   must be compatible with renderer, have exact source dimensions, and not
   overlap. inverse_matrix9 must contain nine finite values. */
IMPRINT_API im_status im_renderer_render_camera_profile_transfer_shared(
    im_renderer *renderer, const im_shared_buffer *processed,
    const float *inverse_matrix9, size_t inverse_count,
    im_shared_buffer *destination);

/* Retain the RGB16 source owner on this renderer until it is replaced or
   cleared. Its byte size must equal width*height*3*sizeof(uint16_t). */
IMPRINT_API im_status im_renderer_set_camera_profile_source_shared(
    im_renderer *renderer, uint32_t width, uint32_t height,
    const im_shared_buffer *source);

/* Render to the supplied independent shared RGB8 destination owner. Its byte
   size must equal the retained source's width*height*3 sample count. */
IMPRINT_API im_status im_renderer_render_camera_profile_shared(
    im_renderer *renderer, const float *constants22, size_t constants_count,
    const float *look_table, size_t look_values,
    uint32_t hue_count, uint32_t saturation_count, uint32_t value_count,
    uint32_t look_encoding, const float *tone_curve, size_t tone_values,
    im_shared_buffer *destination);

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

/* GPU implementation of the same WarpRectilinear contract as the CPU ABI. */
IMPRINT_API im_status im_renderer_warp_rectilinear_rgb16(
    im_renderer *renderer, uint32_t width, uint32_t height,
    const uint16_t *source, size_t source_values,
    const float *constants, size_t constant_values,
    uint16_t *destination, size_t destination_values);
IMPRINT_API int im_renderer_supports_warp_rectilinear(const im_renderer *renderer);

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
