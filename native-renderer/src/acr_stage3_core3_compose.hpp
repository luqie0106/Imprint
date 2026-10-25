#pragma once

#include "acr_stage3_core3_green.hpp"
#include "acr_stage3_core3_prefix.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

enum class Core3ComposeStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    buffer_too_small = 4,
    unsupported_alignment = 5,
    invalid_phase = 6,
    invalid_parameters = 7,
    resource_limit = 8,
    allocation_failure = 9,
    arithmetic_overflow = 10,
    kernel_failure = 11,
    unsupported_domain = 12,
};

struct Core3ComposeControls {
    Core3GreenParameters green{};
    float lower = 0.0f;
    float upper = 0.0f;
    std::uint32_t row_origin_parity = 0;
    std::uint32_t column_origin_parity = 0;
};

struct Core3ComposeOutput {
    std::size_t width = 0;
    std::size_t height = 0;
    Core3PrefixRegion valid_region{};
    std::size_t output_width = 0;
    std::size_t output_height = 0;
    std::vector<float> rgb;
    std::vector<std::uint16_t> rgb_u16;
};

// Recovered preparation of the bounded controls at the original 007f2124
// entry. This accepts an explicit amount and host suppression flag; it does
// not read or bind any runtime camera state. Output must have natural object
// alignment and a non-wrapping address span. Every failure leaves *out intact.
Core3ComposeStatus prepare_core3_controls_007f2124(
    double amount, bool suppress_refine,
    std::uint32_t row_origin_parity,
    std::uint32_t column_origin_parity,
    Core3ComposeControls *out) noexcept;

// Compose the recovered six CPU kernels over explicitly prepared uint16 CFA
// values. The compact output contains only the valid interior; this is not the
// product/full-resolution Stage3 entry. Immediate u16 normalization is an
// explicit harness choice, not a binding of Adobe's pre/post balance stages.
//
// Width is divisible by four, input_row_stride is in samples and divisible by
// eight, and src is 16-byte aligned. The required readable span ends after
// width samples of the last row; trailing padding is not read. Parities locate
// the tile on a canonical global RGGB lattice. Controls are explicit prepared
// values; no runtime object state is inferred here. Arithmetic parity assumes
// the usual nearest-even floating-point environment, as for the row kernels.
//
// m==0 returns an inset-8 interior (minimum W=20,H=17); m!=0 returns inset-12
// (minimum W=28,H=25). Final RGB must be finite and in [0,1]; otherwise the
// whole call rejects without publishing a partial result. The nine-plane
// workspace is capped at 64M float samples. All failures leave *output intact.
// Source and padding are never written. Source may refer to the old output's
// rgb_u16 storage: it remains live through computation, then a successful
// output replacement invalidates that old storage normally.
Core3ComposeStatus run_core3_prepared_u16_tile(
    const std::uint16_t *src,
    std::size_t src_sample_count,
    std::size_t width,
    std::size_t height,
    std::size_t input_row_stride,
    const Core3ComposeControls &controls,
    Core3ComposeOutput *output) noexcept;

// Float-input counterpart to run_core3_prepared_u16_tile. It copies prepared
// float samples directly into the shared padded input, without scaling,
// clipping, or a u16 round trip. Only the width samples in each row are read
// and must be finite; row padding may contain any bit pattern. The source must
// be 16-byte aligned and input_row_stride (in float samples) must be divisible
// by four. Finite samples are accepted without an input [0,1] restriction;
// final RGB still must pass the existing [0,1] output encoding check.
//
// This interface only composes the recovered six CPU kernels over explicitly
// prepared input. It does not bind real camera state, constitute full-resolution
// Stage3, or enable a dispatcher, GPU path, or product default.
Core3ComposeStatus run_core3_prepared_f32_tile(
    const float *src,
    std::size_t src_sample_count,
    std::size_t width,
    std::size_t height,
    std::size_t input_row_stride,
    const Core3ComposeControls &controls,
    Core3ComposeOutput *output) noexcept;

}  // namespace imprint::acr
