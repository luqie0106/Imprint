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

/* Recovered ACR fixed illuminant-code to Kelvin lookup (FUN_01297c78).
   Exactly two uint32 codes are read and two doubles are written. Codes in
   0..65535 map known values to Kelvin and unknown values to zero; 255 is
   rejected because its descriptor-dependent path is not prepared. Input and
   output must be aligned, valid, and non-overlapping. Failure leaves output
   unchanged. */
IMPRINT_API im_status im_native_acr_illuminant_temperatures(
    const uint32_t *codes, size_t count, double *output, size_t output_count);

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

/* Explicit SDR bulk profile/HSM mode1 generic wrapper (00bd60e0, header
   range word 0). Compact active HWC rows; counts/strides are in samples.
   pixel_type 3 stores unsigned words; type 8 stores the same normalized
   values biased by XOR 0x8000. The caller resolves this format separately
   from profile encoding 0/1 and the pipeline's selected mode/encoding byte.
   The float bulk stage is bracketed by original integer/float conversion;
   writeback is fmaf(value,65535,.5), ARM signed FCVTZS, low16 then optional
   XOR, with no extra clamp. No HDR/range2/4 post-processing or automatic
   host selection. Profile encoding 0/1 controls optional transfer curves;
   it is separate from the HSM object's HDR byte, fixed to zero here.
   Table is [value,hue,saturation,3] float32; table_values counts floats.
   Exact same-address/count/stride source/destination is allowed; other
   overlap and output overlap with the map are rejected. Row padding and
   all destination bytes on failure are preserved. */
IMPRINT_API im_status im_native_acr_profile_stage_sdr_u16(
    const uint16_t *source, size_t source_count, size_t source_row_stride,
    size_t height, size_t width, const float *table_data, size_t table_values,
    uint32_t hue_count, uint32_t saturation_count, uint32_t value_count,
    uint32_t encoding, double lookup_gain, double amount, uint32_t pixel_type,
    uint16_t *destination, size_t destination_count, size_t destination_row_stride);

/* Explicit finite-value FUN_01372fb8 CPU path. Packed HWC RGB, including
   negative and overrange samples, is clamped to [-1000,1000]. Controls[4] are
   gain, compression, expansion, epsilon; these are resolved internal floats,
   not UI amounts. Gain>=0, the other three values>0, all finite.
   Tables[4] contain R/G/B and gray descriptors. A channel group runs only when
   all three pointers are nonnull; gray is enabled independently. An enabled
   table needs count>=max_index+2 (one interpolation guard), finite scale>=0.
   Mask is absent (null,count0) or contains at least pixel_count finite values.
   No host/HDR mode selection or table preparation is inferred here. All pixel
   work is C++. Exact source/destination alias is allowed; partial overlap and
   overlap of destination with active tables/mask/controls/descriptors are
   rejected. Errors leave the entire destination unchanged. */
typedef struct im_acr_exposure_lut_f32 {
    const float *values;
    size_t count;
    uint32_t max_index;
    float scale;
} im_acr_exposure_lut_f32;
IMPRINT_API im_status im_native_acr_signed_exposure_prepared_f32(
    const float *source, size_t source_samples, size_t pixel_count,
    const float *controls, size_t control_count,
    const im_acr_exposure_lut_f32 *tables, size_t table_count,
    const float *mask, size_t mask_count,
    float *destination, size_t destination_samples);

/* Prepared baseline 013729ac SDR mask subset, with all five optional
   luminance/channel LUT contexts disabled. HWC float RGB is finite and
   nonnegative; a finite HxW mask is mandatory. Counts/strides are samples.
   Controls[6]: gain G, main-curve scale S, rational scale A, sqrt amount B,
   rational denominator Q, epsilon E. G/S>=0; A/Q/E>0; B finite.
   Main LUT is read only if A>1 and each actual index needs a guard sample.
   Mask is clamped to +/-16 and approximates 2^(4*mask) using original FMA
   and floor instructions. Arithmetic must remain finite; nonfinite
   intermediates are rejected, so this is narrower than the original ABI.
   Exact source/destination address/count/stride alias is allowed. Other
   overlap, including destination with mask, controls or active LUT, fails
   without writing; successful calls preserve row padding. No automatic
   stage selection, upstream mask/LUT preparation, or UI mapping. */
IMPRINT_API im_status im_native_acr_sdr_mask_exposure_prepared_f32(
    const float *source, size_t source_count, size_t source_row_stride,
    size_t height, size_t width, const float *mask, size_t mask_count,
    size_t mask_row_stride, const float *controls, size_t control_count,
    const float *main_lut, size_t main_lut_count,
    float *destination, size_t destination_count, size_t destination_row_stride);

/* Same prepared SDR mask operation, with five optional LUT contexts in
   order [pre weighted-luminance, R, G, B, post weighted-luminance]. Exactly
   five descriptor objects are required; null values disable a descriptor.
   The R/G/B group runs only when all three value pointers are nonnull;
   otherwise its entire descriptor group is ignored. Pre and post run
   independently. Weighted luminance uses ((R+G)+G+B)*0.25 in that order;
   its mapped value scales RGB by mapped/max(luminance,E). Post processing
   alone adds a final channel upper clamp. LUT indices use ARM signed
   FCVTZS, including truncation of coordinates between -1 and 0 to zero;
   max_index is inclusive, and every actual access needs index+1<count.
   Active scales are finite/nonnegative; max_index<=INT32_MAX. Only accessed
   entries must be finite. Intermediate finiteness and atomic buffer/alias
   rules above apply, including output overlap with descriptor objects and
   active table data. Does not construct masks, curves, or host controls. */
IMPRINT_API im_status im_native_acr_sdr_mask_exposure_context_prepared_f32(
    const float *source, size_t source_count, size_t source_row_stride,
    size_t height, size_t width, const float *mask, size_t mask_count,
    size_t mask_row_stride, const float *controls, size_t control_count,
    const float *main_lut, size_t main_lut_count,
    const im_acr_exposure_lut_f32 *tables, size_t table_count,
    float *destination, size_t destination_count, size_t destination_row_stride);

/* Explicit no-mask baseline 01372428, with the same six controls and five
   context descriptors above. RGB is multiplied directly by G. After the
   rational transform, upper clipping and median reconstruction run only if
   the high endpoint exceeds 1. A supplied zero mask in baseline 013729ac
   does not implement this branch. Finite-intermediate, guard, stride, alias
   and transactional rules remain the same. This does not choose between
   the original no-mask fast/full slots or infer host flags. */
IMPRINT_API im_status im_native_acr_sdr_nomask_exposure_context_prepared_f32(
    const float *source, size_t source_count, size_t source_row_stride,
    size_t height, size_t width, const float *controls, size_t control_count,
    const float *main_lut, size_t main_lut_count,
    const im_acr_exposure_lut_f32 *tables, size_t table_count,
    float *destination, size_t destination_count, size_t destination_row_stride);

/* Prepare pixel epsilon E from the original 009e02b0 source fields:
   uint32 ColorimetricReference, source+1540 flare, and original child EV90.
   Flare is the source1540 value, not the separately named log epsilon.
   Finite double inputs are required. The two double clamps and optional
   exp2(-EV90) precede one float conversion. Intermediate exp2 overflow or
   underflow is clamped to the original finite bounds. Writes only output[0]
   on success; a short output or invalid inputs leave all output unchanged.
   Does not infer the selected source image, its flare or EV90 from a RAW. */
IMPRINT_API im_status im_native_acr_prepare_exposure_pixel_epsilon(
    uint32_t colorimetric_reference, double source_flare, double original_ev90,
    float *output, size_t output_count);

/* Sample an explicit prepared Exposure context curve into 4098 float slots.
   kind=1: channel curve 00926ba4, controls[3]=a,b,h. Finite a/b in [0,1),
   h in [0,0.5]; the constructor's a/b==1 cases are outside this subset.
   kind=2: post wrapper 00926758 from 009222f8(p), controls[1]=p in [0,1].
   Keeps binary64 scalar operations, including post encode/decode and pow.
   Samples x=(1/4096)*i for i=0..4096, converts each result once to float,
   and copies endpoint4096 to guard4097. Descriptor max_index=4096,
   scale=4096. Only the first 4098 output slots are written; output padding
   is preserved. Invalid/short/overlapping buffers fail without writing.
   This builds selected curves, not their upstream controls or enable flags.
   Does not prove a particular Adobe request enabled either optional curve. */
IMPRINT_API im_status im_native_acr_prepare_exposure_context_lut(
    uint32_t kind, const double *controls, size_t control_count,
    float *output, size_t output_count);

/* Prepare the observed Blacks2012 response from six explicit controls:
   [Blacks2012 int32, ProfileGainTableMap int32, gain-adjust-active 0/1,
    resolved mode 0/1, finite k, prepared q in [0.0625,0.9]].
   Output[3]=normalized z, post strength B, channel strength D. Keeps the
   original UI clamp, conditional gain correction and full 00922c4c curve
   mathematics, including its constructor rejection cases. Does not infer
   mode, gain-adjust predicate, k/q, profile metadata or params+895 gate.
   All controls must be finite. Success writes only three doubles; padding
   is preserved. Invalid/short/overlapping buffers fail without writing.
   This prepares controls, not RGB pixels or a complete Adobe request. */
IMPRINT_API im_status im_native_acr_prepare_black_response(
    const double *controls, size_t control_count,
    double *output, size_t output_count);

/* Statistics of an explicitly prepared normalized post-tone log plane.
   Finite float input is contiguous; count is in [1,2^26]. Epsilon is the
   source-log epsilon in (0,2^-12], not the Exposure pixel epsilon E.
   Output[5]=filtered p0.01%, filtered p1%, filtered p99%, high h, population M.
   These correspond to context+658/+670/+678/+690 and population diagnostic.
   Post-filter indices use min(floor(M*p),M-1), with p=.0001/.01/.99/.9999.
   An empty filtered population becomes [log(epsilon)/ln2] and M=1.
   Keeps the original low gate and repeated-high removal. This does not
   generate the normalized request or infer its stage/image from display RGB.
   Success writes only five doubles and preserves padding. Short/invalid/
   misaligned/overlapping buffers fail without writing. */
IMPRINT_API im_status im_native_acr_post_tone_stats_prepared_f32(
    const float *log_plane, size_t sample_count, double epsilon,
    double *output, size_t output_count);

/* Prepared basic-slot loop from normalized post-tone request 009dca10.
   Both int32 input arrays contain exactly 132 slots (indices 0..0x83).
   Input is the loop-entry state AFTER basic activation normalization;
   source_defaults is the separately prepared source/default params vector.
   This does not clone opaque params, prepare source defaults, normalize
   activation groups, serialize keys, infer request flags or render pixels.
   PV uses uint32(PV+1) wrap before its recovered threshold comparison.
   Every int32 value, including sentinels, is accepted. Only the first 132
   output values are written; output padding is preserved. Exact in-place
   operation with either input is allowed only with output_count==132;
   other full declared-output-span overlaps are rejected. Failure is atomic. */
IMPRINT_API im_status im_native_acr_normalize_post_tone_basic_i32(
    uint32_t process_version, const int32_t *input, size_t input_count,
    const int32_t *source_defaults, size_t source_count,
    int32_t *output, size_t output_count);

/* Prepared 009227d0 q response. Controls[8]: context+98,+a0,+b0,+80,
   filtered p1(+670), p99(+678), p0.01%(+658), high(+690).
   Output[3]: anchor v, raw response R(exp2(v)), q clamped [.0625,.9].
   This is not the separate per-channel cap R(exp2(A)/sqrt(2)).
   Supported domain: finite values; +98/+a0 in [-120,120], +b0/+80 in
   [-2,2], computed v in [-120,120]. Negative +b0 additionally needs a
   constructible positive white clip. These are adapter limits, not original
   request validation. Zero is canonicalized to +0 for curve evaluation.
   Preserves double white-clip squaring and each separate exp2 operation.
   Success writes three doubles, preserving padding; invalid/short/aliased
   buffers fail atomically. Does not resolve profile/mode/k or request state. */
IMPRINT_API im_status im_native_acr_prepare_exposure_qcap(
    const double *controls, size_t control_count,
    double *output, size_t output_count);

/* Prepared 00922f20 optional curves; no profile/runtime state inference.
   Controls[18]: Blacks(int32), ProfileGain(int32), producer gain-adjust flag,
   mode flag, k, context+98/+a0/+b0/+80/+670/+678/+658/+690,
   context+a8, context+30/+38/+40, caller gain-adjust flag.
   The two gain-adjust flags are independent resolved predicate invocations.
   Output[18]: q_v,q_raw,q,z,B,D,channel_cap,channel_active,post_active,
   aR,bR,aG,bG,aB,bB,h,post_p,caller_z. Inactive channel fields are zero;
   inactive post_p is zero. Flags are exact 0/1. If any channel is active,
   all three channel objects exist, including channels with zero controls.
   Adapter limits: q-context domain above, +a8 in [-120,120], nonnegative
   finite +30/+38/+40 and finite scaled products. Active a/b must be [0,1),
   h [0,.5], p [0,1]. These are supported domains, not original validation.
   Caller cap is sampled independently at R(exp2(A)/sqrt(2)), not q.
   No LUT identity shortcut. Success writes 18 doubles and preserves padding;
   short/invalid/misaligned/aliased buffers fail atomically. */
IMPRINT_API im_status im_native_acr_prepare_exposure_optional(
    const double *controls, size_t control_count,
    double *output, size_t output_count);

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

/* Prepare the source AnalogBalance used by ColorSpec and camera preparation.
   count is 0 (absent) or 3 (raw RGB metadata). Absent or a nonpositive minimum
   models the cleared source vector's per-channel getter default of 1. Otherwise
   multiply by reciprocal minimum, then round to 1e-6 using the source int32
   quantizer. This is distinct from CameraNeutral's maximum normalization.
   NaN/Inf, overflow, misalignment or overlap leaves output unchanged. */
IMPRINT_API im_status im_native_acr_prepare_analog_balance(
    const double *balance, size_t count, double *output, size_t output_count);

/* Quantize one present source RGB CameraCalibration matrix at 1e-4, using
   the source reader's separate multiply/add and int32 rounding. Signature
   compatibility and absent-matrix identity selection belong to the caller.
   Invalid/nonfinite/overflow/overlap leaves output unchanged. */
IMPRINT_API im_status im_native_acr_prepare_camera_calibration(
    const double *matrix, size_t count, double *output, size_t output_count);

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

/* Ordinary modern three-plane Camera kernel 013466b4, with explicit prepared
   controls [A,B,U,b, sorted row-major matrix9]. After float conversion:
   A>=B>=1, 32768/65535<=U<=1, 0<=b<1; all values must be finite.
   Input/output are float32 HWC RGB; order3 is a permutation selecting source
   channels in descending gain order. Matrix columns are already in that order.
   Counts and row strides are samples. flags bit0=0 clamps output to [0,1];
   bit0=1 preserves finite negative/overrange output. Other bits are ignored.
   Exact in-place use requires the same pointer, stride and declared count;
   other overlaps (including metadata) are rejected. Failure preserves output.
   This prepared CPU component retains the pedestal and does not select host
   state or perform post-camera black, exposure, Profile or display processing. */
IMPRINT_API im_status im_native_acr_camera_combined_prepared_f32(
    const float *source, size_t source_count, size_t source_row_stride,
    size_t height, size_t width, const double *controls, size_t control_count,
    const uint32_t order3[3], uint32_t flags, float *destination,
    size_t destination_count, size_t destination_row_stride);

/* Explicit mode2 float post-camera black stage: b>0 builds the recovered
   4096-segment LUT with double sampling, then applies float interpolation;
   b==0 copies active samples bit-for-bit without an identity LUT. black is
   the prepared retained floor in double (0<=black<1), not a DNG BlackLevel.
   Active input must be finite in [0,1]. Counts/strides are float samples.
   Exact same pointer/count/stride permits in-place use; other overlaps are
   rejected. Padding and all output bytes on failure are preserved.
   This entry explicitly requests float processing. It does not infer the
   actual RAW pipe format or consume host context state. */
IMPRINT_API im_status im_native_acr_postcamera_black_prepared_f32(
    const float *source, size_t source_count, size_t source_row_stride,
    size_t height, size_t row_samples, double black, float *destination,
    size_t destination_count, size_t destination_row_stride);

/* Explicit mode1 unsigned post-camera black: build the same float LUT, expand
   to 65536 uint16 entries via 01300600, then use 0132d484 lookup. black==0
   bypasses the LUT. Counts/strides, alias and failure rules match the float
   entry. All uint16 inputs are valid. This does not implement signed type8,
   select the actual request's format, or perform adjacent buffer conversion. */
IMPRINT_API im_status im_native_acr_postcamera_black_prepared_u16(
    const uint16_t *source, size_t source_count, size_t source_row_stride,
    size_t height, size_t row_samples, double black, uint16_t *destination,
    size_t destination_count, size_t destination_row_stride);

/* Explicit prepared Stage3 CPU composition from linearized canonical RGGB
   uint16 CFA to interleaved camera RGB16, before OpcodeList3 and profile.
   This experimental API does not infer metadata/host state or enable the
   original full-resolution dispatcher. rect4 is {top,left,bottom,right},
   gains3 contains exactly three doubles, and strides/capacities are samples.
   The source is at rect4's top-left; dimensions must be >=2. source_floor,
   cubic (0/1), amount, suppression (0/1), and hot/cold thresholds are explicit
   prepared inputs. Tile dimensions are 1..1024; output has H*W*3 samples.
   Destination must not overlap source, rect4, or gains3. All errors leave it
   unchanged. No lens correction, white balance selection, or GPU is applied. */
IMPRINT_API im_status im_native_acr_stage3_prepared_image_u16(
    const uint16_t *source, size_t source_samples, size_t source_row_stride,
    const int32_t source_rect4[4], const double gains3[3], uint16_t source_floor,
    int cubic, double amount, int suppress_refine, uint16_t hot_threshold,
    uint16_t cold_threshold, size_t tile_width, size_t tile_height,
    uint16_t *destination, size_t destination_samples);

/* Prepared Opcode id9 CPU GainMap with explicit image bounds and tile grid.
   rects8=[image T,L,B,R, area T,L,B,R]; controls9=[imagePlanes,
   firstPlane, areaPlanes, rowPitch, colPitch, mapV, mapH, mapPlanes, listStage];
   geometry4=[spacingV,spacingH,originV,originH]. Source starts at image bounds'
   top-left, interleaved HWC; row_stride and capacities are samples. Map layout
   is V,H,mapPlanes float32. The tile grid starts at image bounds' top-left.
   Different tile grids may differ in rounding because interpolation restarts.
   Black compensation uses the explicit source floor when listStage>=2.
   Inputs are finite prepared data; destination must not alias input/metadata.
   Errors preserve destination. This does not select host state or run Warp.
   The uint16 adapter uses recovered normalized-row format conversions and
   rejects results outside their [0,1] prepared domain before committing. */
IMPRINT_API im_status im_native_acr_gainmap_prepared_f32(
    const float *source, size_t source_samples, size_t source_row_stride,
    const int32_t rects8[8], const uint32_t controls9[9],
    const double geometry4[4], const float *map, size_t map_samples,
    uint16_t black_floor, size_t tile_width, size_t tile_height,
    float *destination, size_t destination_samples);
IMPRINT_API im_status im_native_acr_gainmap_prepared_u16(
    const uint16_t *source, size_t source_samples, size_t source_row_stride,
    const int32_t rects8[8], const uint32_t controls9[9],
    const double geometry4[4], const float *map, size_t map_samples,
    uint16_t black_floor, size_t tile_width, size_t tile_height,
    uint16_t *destination, size_t destination_samples);

/* Prepared id1 WarpRectilinear CPU consumer, pure radial subdomain.
   rect4=[image T,L,B,R]; controls2=[imagePlanes,parameterPlanes], each 1..4,
   parameterPlanes=1 or imagePlanes. geometry3=[centerX,centerY,aspect].
   Coefficients are ORIGINAL double [k0,k1,k2,k3,t0,t1] per parameter plane;
   t0/t1 must be zero. Finite HWC source, sample strides/capacities, tightly
   packed float32 destination. Explicit tiles begin at image bounds' origin.
   Metadata and declared buffers must not overlap. Failure preserves output.
   NOP copies samples. Other calls follow recovered SrcArea/anchor semantics,
   including rejecting source tile spans below five samples in either axis.
   u16_f32 uses the recovered normalized input conversion, with no final u16
   writeback claim. No GainMap black-floor wrapper is applied by Warp. */
IMPRINT_API im_status im_native_acr_warp_prepared_f32(
    const float *source, size_t source_samples, size_t source_row_stride,
    const int32_t rect4[4], const uint32_t controls2[2],
    const double geometry3[3], const double *coefficients, size_t coefficient_count,
    size_t tile_width, size_t tile_height, float *destination, size_t destination_samples);
IMPRINT_API im_status im_native_acr_warp_prepared_u16_f32(
    const uint16_t *source, size_t source_samples, size_t source_row_stride,
    const int32_t rect4[4], const uint32_t controls2[2],
    const double geometry3[3], const double *coefficients, size_t coefficient_count,
    size_t tile_width, size_t tile_height, float *destination, size_t destination_samples);

/* Static scalar dng_pixel_buffer::CopyArea format routes, with explicit row
   samples/strides/capacities. float->u16 follows 012eaeb8: comparisons clamp
   to [0,1] (NaN->0, +Inf->1, -Inf->0), binary32 FMA(v,65535,.5), then truncate.
   This differs from the core3 ties-to-even format converter. u16->float uses
   one binary32 reciprocal of 65535 followed by multiplication (012eac64).
   Height/row_samples are positive; capacities cover the last active sample.
   Natural alignment and nonoverlapping declared spans are required. Failure
   preserves destination; destination padding is never changed. These are
   scalar prepared routes, without a claim about runtime SIMD replacement. */
IMPRINT_API im_status im_native_acr_buffer_f32_to_u16(
    const float *source, size_t source_count, size_t source_row_stride,
    size_t height, size_t row_samples, uint16_t *destination,
    size_t destination_count, size_t destination_row_stride);
IMPRINT_API im_status im_native_acr_buffer_u16_to_f32(
    const uint16_t *source, size_t source_count, size_t source_row_stride,
    size_t height, size_t row_samples, float *destination,
    size_t destination_count, size_t destination_row_stride);

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

/* Prepared RGB-to-gray coefficient quantization, float CPU conversion and
   source logarithm. The caller supplies the correct prepared RGB domain and
   resolved RGB-to-gray matrix row (3 finite doubles in [-120,120]).
   No transfer curve, extra output stages, stage selection, uint16 render/store
   round trip or Pin is applied; this is not a complete RAW anchor renderer.
   source_epsilon must be positive, <=2^-12 and nonzero after float conversion.
   RGB is finite HxWx3, including signed/overrange samples; non-finite gray
   intermediates are rejected. Output is HxW log float32, <=2^26 pixels.
   Six diagnostics: quantized uint16 R/G/B lanes, then their float R/G/B weights.
   No output/input overlap is allowed. Both outputs are unchanged on failure. */
IMPRINT_API im_status im_native_acr_gray_log_rgbf32(
    uint32_t width, uint32_t height, const float *rgb, size_t rgb_samples,
    const double *coefficients, size_t coefficient_count, double source_epsilon,
    float *log_gray, size_t log_samples, double *diagnostics, size_t diagnostic_count);

/* As above, with a prepared uint16 store/reload between gray conversion and
   source log. Finite signed/overrange gray is accepted. Both writers use
   fused float scaling and ARM signed-int32 FCVTZS (including saturation of
   a scaled overflow); dither=0 stores its low 16 bits, while dither=1 applies
   the recovered post-conversion saturation to [0,65535]. These differ outside
   [0,1]; no upstream Pin or pre-quantization clamp is inserted here.
   dither is 0 or 1; origin_x/y give the absolute phase in the 128x128 tile.
   The deterministic recovered tile is generated in C++. Reload uses one float
   reciprocal of 65535 and a float multiply. Diagnostics remain the same six
   coefficient lanes. Both outputs remain unchanged on failure.
   Source/guide/EV preparation and upstream stage/Pin selection remain explicit
   caller requirements, not inferred RAW or full normalized-render equivalence. */
IMPRINT_API im_status im_native_acr_gray_u16_log_rgbf32(
    uint32_t width, uint32_t height, const float *rgb, size_t rgb_samples,
    const double *coefficients, size_t coefficient_count, double source_epsilon,
    int32_t origin_x, int32_t origin_y, uint32_t dither,
    float *log_gray, size_t log_samples, double *diagnostics, size_t diagnostic_count);

/* Prepared SDR exposure/Whites from a separate normalized post-tone log
   anchor. The caller supplies its correct pixel domain and stage selection.
   No local masks/calibration/optional color tables are implemented here.
   Seven controls: context98,contextA0,amount58,grayDiffB8,WhitesEV80,
   source epsilon (0..2^-12), final pixel epsilon (0..2^-16); both >0.
   The first four controls are in [-120,120], WhitesEV in [-2,2].
   Ten diagnostics: filtered1%,filtered99%,high99.99%,population,
   splitA,splitB,LUT_EV,remainderEV,positiveGain,whiteClipSquared.
   RGB is nonnegative finite float32; anchor is finite float32 log pixels.
   RGB pixel and anchor sample counts are nonzero and at most 2^26 each.
   Exact RGB in-place is allowed. All other output/input overlap is rejected.
   Both outputs remain unchanged on failure. */
IMPRINT_API im_status im_native_acr_exposure_from_anchor_rgbf32(
    uint32_t width, uint32_t height, const float *rgb, size_t samples,
    const float *anchor_log, size_t anchor_samples,
    const double *controls, size_t control_count,
    float *output, size_t output_samples, double *diagnostics, size_t diagnostic_count);

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

/* Existing Imprint profile LookTable operation only, not a complete recovered
   ACR Profile implementation. RGB is finite float32 in [0,1]. Table layout is
   [value,hue,saturation,channel], encoding 0=linear or 1=sRGB value. No aliases
   with destination are allowed; destination remains unchanged on failure. */
IMPRINT_API im_status im_native_profile_look_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *look_table, size_t look_values, uint32_t hue_count,
    uint32_t saturation_count, uint32_t value_count, uint32_t look_encoding,
    float *destination, size_t destination_values);

/* Recovered ACR prepared single-map helper (012ebae4). Table layout
   is [value,hue,saturation,channel], with hue>=1, saturation>=2, value>=1.
   RGB is finite float32 in [0,1]; encoding is 0=linear or 1=sRGB value-axis
   resources prepared in C++ (skipped for a single value layer). At most 2^26
   pixels and 1,000,000 modifier triples; hue shifts are in [-360,360] degrees,
   saturation/value multipliers in [0,16], all finite. This is one selected stage, not full RAW/Profile
   preparation, HDR or Adobe runtime equivalence. No output/input aliases are
   allowed; destination is unchanged on failure. Counts are float elements. */
IMPRINT_API im_status im_native_acr_profile_look_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *look_table, size_t look_values, uint32_t hue_count,
    uint32_t saturation_count, uint32_t value_count, uint32_t look_encoding,
    float *destination, size_t destination_values);

/* Baseline bulk CPU Profile/LookTable stage (00885b38 resource preparation;
   0135f574/0135f8f8), explicitly SDR with overrange disabled. Uses sector-scaled
   padded map entries and fast float encoding curves. Same map/RGB/count bounds
   as the helper above. lookup_gain [0,65536], amount [0,2], both finite;
   a single value layer ignores encoding and lookup_gain. No output aliases;
   destination remains unchanged on failure. Does not infer metadata, HDR,
   or the live Adobe SIMD dispatch. Counts are float elements. */
IMPRINT_API im_status im_native_acr_profile_stage_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *look_table, size_t look_values, uint32_t hue_count,
    uint32_t saturation_count, uint32_t value_count, uint32_t look_encoding,
    double lookup_gain, double amount,
    float *destination, size_t destination_values);

/* Prepare the recovered single/two-illuminant HueSatMap resource. Controls4
   contains white x,y and endpoint kelvin 1,2; metadata3 receives projected
   temperature,tint,first-table weight. A missing second table is null/0.
   Nonpositive/equal endpoints or a single map copy the first table and emit
   metadata [0,0,1]. Counts are float values; layout is [V,H,S,3]. No outputs
   may overlap inputs or each other; both outputs remain unchanged on failure.
   This does not perform metadata loading, three-illuminant choice, or pixels. */
IMPRINT_API im_status im_native_acr_prepare_huesatmap(
    const float *first, size_t first_values,
    const float *second, size_t second_values,
    uint32_t hue_count, uint32_t saturation_count, uint32_t value_count,
    const double *controls, size_t control_count,
    float *output, size_t output_values,
    double *metadata, size_t metadata_count);

/* Resolve 0088ca08 and its caller's reciprocal using explicit prepared state.
   Controls3: source+680 scale, context+20 scale, context+28 EV. Output2:
   source gain and HSM lookup gain. The exact uint32 process-version predicate
   selects the legacy scale or modern exp2 branch; this API does not read or
   infer metadata. Both gains must be finite and positive. Exact counts,
   aligned non-overlapping buffers; failure leaves output unchanged. */
IMPRINT_API im_status im_native_acr_prepare_huesatmap_gain(
    uint32_t process_version, const double *controls, size_t control_count,
    double *output, size_t output_count);

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
