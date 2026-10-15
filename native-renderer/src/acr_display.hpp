#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class DisplayStatus : int {
    ok = 0,
    invalid_input = 1,
    non_finite_intermediate = 2,
    allocation_failure = 3,
};

// Project nonnegative, finite, interleaved float32 ProPhoto RGB to display
// sRGB8 using the caller's row-major normalized ProPhoto-to-PCS Q matrix.
// Q and image storage must be pairwise disjoint; on every failure output is
// left byte-for-byte unchanged. Dimensions are limited to 2^26 pixels.
// Per channel the float32 matrix order is (m0*r + m1*g) + m2*b, with each
// multiply and add kept separate (compile with contraction disabled).
DisplayStatus project_display_srgb8(
    std::uint32_t width, std::uint32_t height,
    const float *source_rgb, std::size_t source_values,
    const double *normalized_prophoto_to_pcs_q, std::size_t q_values,
    std::uint8_t *output_rgb8, std::size_t output_values) noexcept;

// Preserve the existing exposure_scale behavior before the recovered ACR
// exposure stage is integrated: source RGB is finite float32 in [0,1], and
// each output is min(source * gain, 1). This is not a recovered Adobe stage.
// Source and destination may not overlap; failure preserves destination.
DisplayStatus scale_sdr_rgb(
    std::uint32_t width, std::uint32_t height,
    const float *source_rgb, std::size_t source_values, float gain,
    float *destination_rgb, std::size_t destination_values) noexcept;

}  // namespace imprint::acr
