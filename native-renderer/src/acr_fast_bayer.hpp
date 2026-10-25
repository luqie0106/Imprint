#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

enum class FastBayerStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    buffer_too_small = 4,
    invalid_pattern = 5,
    too_many_pixels = 6,
    allocation_failure = 7,
};

// Recovered aligned 2x2 `dng_fast_interpolator` scalar branch
// (`FUN_012a2b58` accumulate loop, `012a2eac`..`012a2ed8` average/store, ctor
// `FUN_012a297c` CFA-color to output-plane map).
//
// The input is one uint16 Stage2 plane buffer (CFA mosaic). Every aligned 2x2
// input block is demosaiced into one RGB output pixel by accumulating
// `sum[plane]`/`count[plane]` with the caller-provided CFA plane map:
//
//   for each block
//     for row in 0..1, col in 0..1
//       plane = pattern[row * 2 + col]
//       sum[plane]   += input[sample]
//       count[plane] += 1
//     output[plane] = (sum[plane] + (count[plane] >> 1)) / count[plane]
//
// This is unsigned integer division with a pre-division half-count add, so an
// odd two-sample sum rounds up. The recovered code then stores the quotient
// with a 16-bit store; with a 2x2 block and three present colors the quotient
// never exceeds 65535. No white balance, black subtraction, white scaling,
// profile or float conversion is applied here, and no extra output channel is
// created for the duplicated CFA color.
//
// `pattern` is row-major for the 2x2 block and must contain each of the three
// output plane indices 0, 1 and 2 at least once, so exactly one plane is seen
// twice and two planes are seen once. Callers are responsible for preparing the
// pattern from the real CFA phase; this operator does not read metadata or
// guess a Bayer arrangement.
//
// Scope limits (not implemented here): odd dimensions and edge padding, scale
// pairs other than (2, 2), other tile phases, region rectangles, and the
// remaining NEON chunking contract. `width`/`height` must be even and positive,
// `input_row_stride` is in samples and may exceed `width`, and unused samples
// in the padded region are never read. `input_sample_count` bounds the readable
// span so a short buffer is rejected instead of read past its end.
//
// Success replaces `output_rgb` with a tightly packed interleaved RGB uint16
// image of size (width / 2) x (height / 2). Failure leaves `output_rgb`
// unchanged. `input` may point into the current storage of `output_rgb`: all
// input reads complete before the replacement is swapped in. A new pixel
// allocation failure is reported as `allocation_failure` with no output change.
FastBayerStatus downsample_aligned_bayer_2x2(
    const std::uint16_t *input,
    std::size_t input_sample_count,
    std::size_t width,
    std::size_t height,
    std::size_t input_row_stride,
    const std::uint8_t pattern[4],
    std::vector<std::uint16_t> *output_rgb) noexcept;

}  // namespace imprint::acr
