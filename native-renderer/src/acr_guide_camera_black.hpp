#pragma once

#include "acr_tone_kernels.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

struct CameraStagePrepared;

enum class GuideCameraBlackStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    buffer_too_small = 4,
    arithmetic_overflow = 5,
    resource_limit = 6,
    unsupported_alignment = 7,
    overlap = 8,
    invalid_controls = 9,
    camera_failure = 10,
    conversion_failure = 11,
    black_failure = 12,
    allocation_failure = 13,
    length_error = 14,
    gray_log_failure = 15,
    mask_failure = 16,
    unsupported_geometry = 17,
    unsupported_boundary = 18,
    profile_preparation_required = 19,
};

struct GuideCameraBlackControls {
    // [A, B, U, retained double b, sorted row-major M9]. This is the
    // existing combined Camera contract, not the standalone [A,B,U,S,M9].
    std::array<double, 13> camera{};
    std::array<std::uint32_t, 3> order{0, 1, 2};
    // Bit zero is the same prepared context flag for Camera's overrange
    // output and the following 0088cb58 pin append. Only canonical source
    // values 0/1 are accepted; this is not the standalone kernel flag domain.
    std::uint32_t camera_flags = 0;
    // Resolved first selector from the guide's input image: 1 or 2.
    // No source format is inferred from the supplied float adapter buffer.
    std::uint32_t input_selector = 0;
};

// Explicit fields from one admitted modern request. response_limit_u16 is
// the already quantized Camera +a4 value (32768..65535), e.g. the matching
// ExposurePrepared::source_limit_u16. retained_black_u16 is source+688, not
// a raw BlackLevel tag or profile black compensation. The caller establishes
// that context+6f8/+700 still carry these source values at Camera construction
// and the following black/pin append. No metadata or pointer identity is inferred.
struct GuideCameraBlackSource {
    std::uint32_t response_limit_u16 = 0;
    std::uint32_t retained_black_u16 = 0;
    std::uint32_t source_float_flag = 0;  // source+174 / context+700, 0 or 1
    std::uint32_t guide_image_type = 0;  // selected image+1c: supported 3 or 11
};

// Bind a matching three-plane camera preparation without sorting its matrix
// twice. Uses float32 U16/65535 for U, but double B16*(1/65535) for retained b.
// Source float flag and selected guide image type remain independent inputs.
// Output is unchanged on failure. This creates controls only: it neither
// prepares source pixels nor proves the request's ordinary-stage predicates.
GuideCameraBlackStatus bind_guide_camera_black_controls_prepared(
    const CameraStagePrepared& camera, const GuideCameraBlackSource& source,
    GuideCameraBlackControls& output) noexcept;

struct GuideCameraBlackPlan {
    std::uint32_t camera_type = 0;
    std::uint32_t black_type = 0;  // zero when the double b is not positive
    std::uint8_t black_encoding = 0;
    bool quantized_before_black = false;
    std::uint32_t pin_type = 0;  // independent of b; 0 means no pin stage
};

struct GuideCameraBlackResult {
    std::size_t height = 0;
    std::size_t width = 0;
    GuideCameraBlackPlan plan{};
    std::vector<float> rgb;  // compact HWC RGB, before the gray/log suffix
};

// Compose the recovered ordinary three-plane Camera and post-camera black
// block for the modern guide, with initially unresolved format selections
// and a following float gray/log consumer. The caller establishes the actual
// prepared Camera input and controls, and excludes intervening optional
// stages, local Temp/Tint, WB-only, identity bypass and GPU processing. This
// adapter does not discover those predicates, prepare RAW/Stage4 pixels,
// apply the gray/log suffix, or enable automatic page Highlights.
//
// Source is finite float32 HWC with explicit sample capacity/row stride;
// input_selector describes the original guide selection, not this adapter's
// storage. At most 128 Mi active samples are supported. Padding is not read.
// Camera selects type2. If original double b>0, selector1 chooses unsigned
// type1 black: CopyArea f32->u16, u16 black LUT, reciprocal-multiply u16->f32.
// Selector2 uses the float black LUT. With bit zero set, the CPU pipe pins
// Camera output to [0,1] before that range=0 LUT. Independently, bit zero
// appends a range=0 pin stage even when double b==0 (including -0): selector1
// then still quantizes f32->u16->f32, while selector2 pins finite float values
// (nonpositive -> +0, over-one -> 1). With both b==0 and bit zero clear, no
// conversion is inserted. A positive double that rounds to float zero still
// appends the black stage. The original source flag must remain the same
// through this block; intervening context mutations are outside its contract.
//
// All failures leave the entire old result intact. Source may use the old
// result.rgb allocation, since it is fully consumed before the result is
// replaced. Source's declared span and controls must not overlap the result
// object itself. Valid C++ objects and ordinary nonoverlapping pixel/control
// storage remain caller preconditions.
GuideCameraBlackStatus apply_guide_camera_black_prepared(
    const float* source, std::size_t source_count,
    std::size_t source_row_stride, std::size_t height, std::size_t width,
    const GuideCameraBlackControls& controls,
    GuideCameraBlackResult& output) noexcept;

struct GuideGrayLogControls {
    GuideCameraBlackControls camera;
    double ev90 = 0.0;  // matching context+90, not manual Exposure UI
    double source_epsilon = 0.0;  // resolved 009e01e8 result; must be explicit
};

struct GuideGrayLogResult {
    std::size_t height = 0;
    std::size_t width = 0;
    GuideCameraBlackPlan camera_plan{};
    bool gain_stage_present = false;
    std::vector<float> log_gray;  // compact single plane before mask generation
};

// Extends the same prepared ordinary Camera/black/pin contract through
// 00889340 fixed-weight nonnegative gray, optional 0089a81c EV90 gain, and
// 0088a0a0 source log with b=0. Suffix stages are float; no Q14 output-gray
// transform, integer gray store or post-gain pin is inserted. EV90 is limited
// to +/-120 and epsilon to (0,2^-12], representably positive as float. Width
// and height must also be <=65535. These are defensive native API bounds.
// All failures preserve the complete old result. Old log_gray allocation may
// be the source; controls/source may not overlap the result object itself.
// The caller still establishes Stage4 identity, context and optional-stage
// predicates; this function does not discover a request or produce its mask.
GuideCameraBlackStatus apply_guide_gray_log_prepared(
    const float* source, std::size_t source_count,
    std::size_t source_row_stride, std::size_t height, std::size_t width,
    const GuideGrayLogControls& controls, GuideGrayLogResult& output) noexcept;

struct GuideLocalMaskRequest {
    // Explicit complete guide bounds [top,left,bottom,right]. Current native
    // path accepts only zero-origin complete images, not a tile or ROI crop.
    std::array<std::int32_t,4> bounds{};
    GuideGrayLogControls guide;
    int sigma_mode = 0;
    // Resolved source+1600 (009e035c), independent of sigma mode. Require 2
    // explicitly: 00a0855c's mode4 horizontal blend path is not implemented.
    std::uint32_t boundary_mode = 0;
};

struct GuideLocalMaskResult {
    GuideLocalMaskRequest request{};
    LocalToneGeometry geometry{};
    GuideGrayLogResult guide{};
    LocalToneMask mask{};
};

// State at the direct (params+b50 == null) db01a4 entry. The two byte flags
// are tested as nonzero, not bit0. Level tokens model source+678 and
// source+798..7b8; they are opaque and never dereferenced. No pixels/bounds or
// Stage4 format are inferred from them. A profile-bearing input must first
// execute the unresolved clone/Prepare/recursive branch and be resnapshotted.
struct GuideSourceLevelInput {
    bool profile_pair_present = false;
    std::uint8_t field_ad9 = 0;
    std::uint8_t field_adb = 0;
    std::array<const void*,6> level_images{};
};

struct GuideSourceLevelPlan {
    std::uint32_t initial_level = 0;
    std::uint32_t selected_level = 0;
    std::uint32_t tested_level_mask = 0;
    const void* selected_image = nullptr;
    // Level 0 is not tested by db01a4's search. A successful plan can have
    // available=false; e08058/Stage4 must reject that missing input later.
    bool available = false;
};

// A profile-bearing input returns profile_preparation_required without
// modifying output. Tokens remain borrowed and are never dereferenced.
GuideCameraBlackStatus prepare_guide_source_level(
    const GuideSourceLevelInput& input, GuideSourceLevelPlan& output) noexcept;

// db1228's independent owner-array selector, based on the caller's uint32
// params+21c. This is not a sigma mode, source level or internal stats level.
std::uint32_t select_guide_base_cache_slot(std::uint32_t strength) noexcept;

enum class GuideMaskTransfer : std::uint8_t {
    reuse = 0,
    upsample = 1,
    downsample = 2,
};
// db0444's unsigned comparison: cached entry+9c8 vs current request+88,
// after its unwarped request+100 mask-params pointer has passed the null gate.
// Only classifies the branch; it does not resample, transfer image ownership,
// copy the 0x218 statistics/aux payload, or prove a cache hit.
GuideMaskTransfer classify_guide_mask_transfer(std::uint32_t cached_source_level,
                                              std::uint32_t request_level) noexcept;

// One prepared invocation owns both guide and mask: same bounds and epsilon,
// statistics/collapse level, and a snapshot of the supplied controls. No
// independently prepared log/stats/mask inputs can be mixed into this call.
// It does not prove actual source/Stage4 identity, discover optional-stage
// predicates, rescale a mask to another source level, or execute final SH.
// Source is padded HWC as above, at least 2 pixels; result stays intact on
// every failure. Source may use an old guide.log_gray/mask.delta allocation;
// source/request must not overlap the result object itself or each other.
GuideCameraBlackStatus apply_guide_local_mask_prepared(
    const float* source, std::size_t source_count, std::size_t source_row_stride,
    const GuideLocalMaskRequest& request, GuideLocalMaskResult& output) noexcept;

}  // namespace imprint::acr
