#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

// Status for the recovered aligned, whole-vector subset of FUN_008134b8.
enum class Core3ColourStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    unsupported_alignment = 3,
    invalid_phase = 4,
};

// FUN_008134b8: one row of the core3 colour-difference consumer.
//
// `source_center` and `green_center` point to column zero in their respective
// center rows. Their strides are measured in float samples. The function also
// reads the rows immediately above and below and columns j-1/j+1, so the
// caller must provide those row and column guards, along with readable input
// capacity and writable output capacity for `width` floats.
//
// This port accepts only the recovered 16-byte-aligned subdomain: width is a
// positive multiple of four; both source strides are at least width and are
// multiples of four; all four pointers are 16-byte aligned. The caller must
// ensure that neither output overlaps either input region or the other output.
// The raw-source copy lanes are copied verbatim; only H/V/D candidates are
// clamped. Rejected calls do not write output. The original unaligned fixup and
// other widths are outside this function's supported ABI subset.
Core3ColourStatus core3_colour_difference_row_008134b8(
    const float *source_center,
    const float *green_center,
    std::size_t source_stride,
    std::size_t green_stride,
    std::size_t width,
    std::uint32_t phase,
    float *out_a,
    float *out_b) noexcept;

}  // namespace imprint::acr
