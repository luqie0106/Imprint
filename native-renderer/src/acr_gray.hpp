#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class GrayStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_coefficients = 2,
    invalid_rgb = 3,
    overflow = 4,
    allocation_failure = 5,
};

struct PreparedRgbToGray {
    // Original byte-lane values, in RGB coefficient order.
    std::uint16_t lanes_rgb[3]{};
    // The effective weights reconstructed from those low 16-bit lanes.
    float weights_rgb[3]{};
};

// Reproduces the recovered coefficient error-diffusion quantizer and the
// subsequent nonnegative uint16 lane conversion. The output is committed only
// after every coefficient succeeds.
GrayStatus prepare_rgb_to_gray(const double *coefficients_rgb,
                               PreparedRgbToGray *prepared) noexcept;

// Applies the prepared RGB weights to interleaved float32 RGB. The output is
// one float32 gray value per pixel. Finite signed and overrange inputs are
// accepted; non-finite inputs and non-finite intermediate results are
// rejected. Output is not clamped. Input/output overlap is rejected and
// output is committed only after the complete image succeeds.
GrayStatus apply_rgb_to_gray(const float *input_rgb, float *output_gray,
                             std::size_t pixel_count,
                             const PreparedRgbToGray *prepared) noexcept;

}  // namespace imprint::acr
