#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

// Static Stage3 dispatch recovered from Camera Raw 17.2 ARM64:
//
//   source+0xe0 (FUN_01310ef0)
//     -> cr_mosaic_info+0x50 (FUN_00ab4b78)
//       -> mode 1, equal pair or (2,4): FUN_00ab4444
//         -> FUN_00ab4094
//           -> FUN_00ab3f24
//             -> FUN_00ab3964
//               -> pair (1,1): FUN_00ab35bc -> FUN_007f2124
//               -> pair (2,2): FUN_00ab35bc -> FUN_007f8a44
//               -> pair (2,4): FUN_00ab35bc -> FUN_007f9228
//
// The prepared contract intentionally supports only the direct, aligned
// three-plane canonical RGGB path that is reached when source+0x600 exists,
// the mosaic mode is 1, the CFA byte pattern is exactly {0,1,1,2}, the pair is
// (1,1) or (2,2), mosaic+0x7c is zero, mosaic+0x80/0x84 are zero, and the plane
// map is [0,1,2]. Other mode/pair/task/plane/CFA states return explicit
// unsupported statuses instead of falling back to another demosaic
// implementation.
//
// The canonical-RGGB restriction is deliberate. FUN_00ab21f0 leaves
// mosaic+0x80/+0x84 at zero only for the identity RGGB arrangement. GRBG,
// GBRG and BGGR take other branches that store a packed nonzero offset pair
// (0x100000000, 0x1, 0x100000001) at mosaic+0x80 and then enter the
// FUN_00ab4094 coordinate/normalisation path. That transform is not recovered,
// so zero-offset states for those arrangements are contradictions, not
// supported inputs.
//
// For pair (2,2) the recovered pixel body FUN_007f851c is a fixed-position 2x2
// downsample, not a generic three-colour averaging operator:
//   plane 0 = sample at the even/even position (top-left)
//   plane 1 = (top-right + bottom-left + 1) >> 1
//   plane 2 = sample at the odd/odd position (bottom-right)
// `run_stage3_bayer_2x2` implements that body directly for RGGB. It does not
// reuse the independent `downsample_aligned_bayer_2x2` helper, whose contract
// intentionally accepts all 36 three-plane 2x2 arrangements. It is not used
// for the full-resolution (1,1) branch.
//
// Pair (1,1) selects FUN_007f2124, which constructs the multi-stage
// cr_stage_bayer_core3 pipeline rather than a single recovered pixel loop.
// FUN_007eaf20 is its CPU row executor and the function-pointer table at
// 0x08197068 (initialized by FUN_013aad9c) names the 0081376c/00813368/...
// kernels. Those kernels are not ported yet. This contract reports the branch
// as unsupported_full_res_core3; it does not substitute a different demosaic
// and does not write output.
enum class Stage3Status : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_cfa = 3,
    unsupported_mode = 4,
    unsupported_pair = 5,
    unsupported_offset = 6,
    unsupported_plane_map = 7,
    unsupported_task_transform = 8,
    unsupported_full_res_core3 = 9,
    unsupported_kernel = 10,
    invalid_stride = 11,
    buffer_too_small = 12,
    unsupported_aliasing = 13,
    allocation_failure = 14,
    invalid_rect = 15,
};

struct Stage3Prepared {
    // Decoded Stage2 CFA buffer and its declared row stride, in samples.
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t input_stride = 0;

    // cr_mosaic_info CFA bytes for the 2x2 cell. FUN_00ab21f0/00ab3198 read
    // the real object at +0x10/+0x11/+0x18/+0x19 (2x2 positions, object row
    // stride 8; the +0x10..+0x13 range is not contiguous in the real object).
    // Stage3Prepared stores the four values extracted into this compact
    // row-major array; it is not a byte image of cr_mosaic_info+0x10.
    // Only canonical RGGB {0,1,1,2} is in the closed domain.
    std::uint8_t cfa_pattern[4] = {0, 1, 1, 2};
    // cr_mosaic_info+0x50 and +0x58. Only three planes / one sample per site
    // are in the closed domain.
    std::uint8_t plane_count = 3;
    // cr_mosaic_info+0x88, the distinct output-plane map. The closed direct
    // path requires [0,1,2].
    std::uint8_t plane_map[3] = {0, 1, 2};

    std::uint32_t mode = 0;  // cr_mosaic_info+0x78
    std::int32_t pair_x = 0;  // result of cr_mosaic_info+0x30
    std::int32_t pair_y = 0;

    // param_5 rect: top, left, bottom, right.
    std::int32_t rect_top = 0;
    std::int32_t rect_left = 0;
    std::int32_t rect_bottom = 0;
    std::int32_t rect_right = 0;

    std::uint32_t mosaic_7c = 0;  // cr_mosaic_info+0x7c
    std::int32_t offset_80 = 0;  // cr_mosaic_info+0x80
    std::int32_t offset_84 = 0;  // cr_mosaic_info+0x84
    // param_8 != 0 in FUN_00ab3964. The closed 2x2 branch currently requires
    // this to be false because the half-pixel task transform is not recovered.
    bool task_transform = false;
};

enum class Stage3Branch : int {
    direct_00ab3f24 = 0,
    task_0127cfe0 = 1,
    generic_012a4a60 = 2,
};

enum class Stage3Kernel : int {
    none = 0,
    bayer_2x2_007f851c = 1,
    bayer_2x4_007f9228 = 2,
    bayer_1x1_007f2124 = 3,
    generic_007fb724 = 4,
};

struct Stage3Dispatch {
    Stage3Status status = Stage3Status::invalid_argument;
    Stage3Branch branch = Stage3Branch::generic_012a4a60;
    Stage3Kernel kernel = Stage3Kernel::none;
    std::size_t output_width = 0;
    std::size_t output_height = 0;
    std::size_t output_samples = 0;
    bool half_pixel_shift = false;
    const char *reason = nullptr;
};

// Select the recovered branch and output geometry. This function never reads
// or writes pixels. On an unsupported branch it still fills branch/kernel when
// the selection is known, and sets status/reason to the exact stopping point.
Stage3Dispatch dispatch_stage3(const Stage3Prepared &prepared) noexcept;

// Run the closed 2x2 preview branch. Requires dispatch_stage3(...).status == ok,
// kernel == bayer_2x2_007f851c and task_transform == false. The rect may have a
// non-zero even origin; width and height of the rect must be even and positive.
// `input_sample_count` is the readable sample span of `input`. The output is a
// tightly packed interleaved RGB uint16 image of (rect_width/2) x
// (rect_height/2). Failure leaves `output_rgb` unchanged. Input may alias the
// current storage of `output_rgb`.
Stage3Status run_stage3_bayer_2x2(
    const std::uint16_t *input,
    std::size_t input_sample_count,
    const Stage3Prepared &prepared,
    std::vector<std::uint16_t> *output_rgb) noexcept;

}  // namespace imprint::acr
