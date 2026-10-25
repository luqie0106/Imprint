#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

struct ExposureSdrMaskLut {
    const float *values = nullptr;
    std::size_t count = 0;
    std::uint32_t max_index = 0;
    float scale = 1.0f;
};

// Recovered prepared controls shared by finite-intermediate subsets of the
// distinct SDR mask baseline 0x013729ac and no-mask baseline 0x01372428.
// Contexts are ordered pre-luma, R, G, B, post-luma. This does not include
// upstream table production or establish Adobe runtime binding.
struct PreparedExposureSdrMask {
    float gain = 1.0f;
    float main_lut_scale = 1.0f;
    float rational_scale = 1.0f;
    float sqrt_amount = 1.0f;
    float rational_denominator = 1.0f;
    float epsilon = 1.0e-6f;
    const float *main_lut = nullptr;
    std::size_t main_lut_count = 0;
    ExposureSdrMaskLut context[5]{};
};

enum class ExposureSdrMaskStatus : int {
    ok = 0,
    invalid_argument = 1,
    buffer_too_small = 2,
    unsupported_aliasing = 3,
    arithmetic_overflow = 4,
    lut_out_of_bounds = 5,
    allocation_failure = 6,
    resource_limit = 7,
    non_finite_intermediate = 8,
};

// Applies the prepared SDR mask exposure subset to HWC RGB float input.
// RGB row strides and counts are in float samples; mask row strides and counts
// are in one-channel float samples. Width and height are in pixels. Input RGB
// must be finite and nonnegative; mask values and all controls must be finite.
// The function rejects any non-finite intermediate or output and leaves dst
// untouched on failure. The main LUT is read only when rational_scale > 1 and
// must include every referenced pair of adjacent samples. Optional contexts
// use the recovered order pre-luma, all-RGB-channel LUTs, post-luma; each active
// table includes its guard sample. Exact in-place use is supported only when
// src/dst pointers, counts, and strides all match. The contract covers only
// finite-intermediate executions of baseline 0x013729ac; it does not establish
// upstream table production or Adobe runtime binding.
ExposureSdrMaskStatus apply_prepared_exposure_sdr_mask(
    const float *src, std::size_t src_count, std::size_t src_stride,
    std::size_t height, std::size_t width, const float *mask,
    std::size_t mask_count, std::size_t mask_stride,
    const PreparedExposureSdrMask *prepared, float *dst,
    std::size_t dst_count, std::size_t dst_stride) noexcept;

// Applies the distinct no-mask SDR exposure baseline 0x01372428 to HWC RGB
// float input using the same prepared controls and optional contexts. It does
// not accept, synthesize, validate, or read a mask. The conditional high>1
// upper-clip/median reconstruction follows 013726e0..013726e4; unlike the mask
// baseline, it does not reconstruct the median when high<=1. This is a prepared
// baseline entry point, not the earlier no-context fast baseline 0x01372168
// and not an automatic dispatcher. A caller that dispatches must preserve the
// upstream no-mask-versus-supplied-mask state. The same finite-intermediate,
// stride, bounds, aliasing, and atomic-output constraints as the mask entry
// point apply, without mask-specific validation.
ExposureSdrMaskStatus apply_prepared_exposure_sdr_nomask(
    const float *src, std::size_t src_count, std::size_t src_stride,
    std::size_t height, std::size_t width,
    const PreparedExposureSdrMask *prepared, float *dst,
    std::size_t dst_count, std::size_t dst_stride) noexcept;

}  // namespace imprint::acr
