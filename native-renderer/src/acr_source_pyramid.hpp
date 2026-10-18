#pragma once

#include "acr_flare.hpp"

#include <cstddef>
#include <vector>

namespace acr_source_pyramid {

enum class Status : int {
    ok = 0,
    invalid_argument = 1,
    size_overflow = 2,
    too_many_pixels = 3,
    non_finite_input = 4,
    non_finite_intermediate = 5,
    allocation_failure = 6,
};

struct Level {
    std::size_t width = 0;
    std::size_t height = 0;
    // Tightly packed RGB float32 samples.
    std::vector<float> pixels;
};

// Build the ordinary edge2 RGB float32 source pyramid from the full actual
// image. The input must already be the caller-selected Stage3/source domain;
// this function does not infer or construct that domain from rendered RGB.
// `nonlinear_enabled` must be resolved by the caller's source gate; this
// function reads no global state or metadata. The kernel transcribes the
// recovered scalar 2x2 path and float bracket arithmetic, without claiming
// bit parity with every NEON inverse path. Finite negative and over-range
// values are not clipped. NaN/Inf input, non-finite intermediates, and images
// over 100 million pixels are rejected as local safety guards. A crop with
// either dimension greater than 64 produces five levels; smaller crops
// produce none. `out` is replaced only after a successful build. Its existing
// pixel storage may be used as input; a successful replacement can invalidate
// that aliased storage after all input reads are complete.
Status build_float_rgb(const float* packed_rgb,
                       std::size_t image_width,
                       std::size_t image_height,
                       acr_flare::Crop crop,
                       bool nonlinear_enabled,
                       std::vector<Level>* out) noexcept;

}  // namespace acr_source_pyramid
