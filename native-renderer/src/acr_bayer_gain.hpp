#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class BayerGainStatus : int {
    ok = 0,
    null_pointer = 1,
    invalid_count = 2,
    invalid_gain = 3,
    invalid_pivot = 4,
    invalid_prepared = 5,
    invalid_width = 6,
    invalid_parity = 7,
    misaligned_pointer = 8,
    size_overflow = 9,
    address_overflow = 10,
    overlap = 11,
    nonfinite_sample = 12,
};

struct BayerGainPrepared {
    float gains[3];
    float bias[3];
    bool clip;
};

// Recovered prepared controls from FUN_007ef070. `gains` contains exactly
// three finite double values in (0, 128); inverse is evaluated in double
// before each result is rounded to float. The supported pivot is one whose
// float conversion is finite and in [0, 1). Both the three-double input span
// and output object must be naturally aligned, non-wrapping address spans, and
// must not overlap. The output is committed only when every control validates.
BayerGainStatus prepare_bayer_gain_007ef070(
    const double* gains, std::size_t gain_count, double pivot, bool inverse,
    BayerGainPrepared* out) noexcept;

// Adapts a cached Stage2 source floor to the pivot consumed by FUN_007ef070.
// The floor is converted to double and multiplied by the recovered binary64
// reciprocal of 65535 before delegating to the existing validator. This helper
// does not select source_gate or change the cached floor.
BayerGainStatus prepare_bayer_gain_from_source_floor_007ef070(
    const double* gains, std::size_t gain_count, std::uint16_t source_floor,
    bool inverse, BayerGainPrepared* out) noexcept;

// Recovered neutral-to-RGB gains preparation from FUN_007efc64. Exactly three
// finite neutral values are normalized through separately rounded binary64
// reciprocal, multiply, and divide operations. A nonpositive minimum or a
// normalized minimum below 0.125 selects the explicit default triplet. Any
// count other than three selects that default without reading `neutral`; this
// is the supported prepared subdomain, not a claim about the complete ABI.
// `out` must describe exactly three naturally aligned doubles in a non-wrapping
// span. When `neutral_count == 3`, the input span has the same requirements and
// must not overlap `out`. Output is committed only after all checks succeed.
BayerGainStatus prepare_bayer_gains_from_neutral_007efc64(
    const double* neutral, std::size_t neutral_count, double* out,
    std::size_t out_count) noexcept;

// Recovered two-plane planar RGB consumer from FUN_007ef1f4. Counts are sample
// capacities and width is the processed row span. All three processed spans
// must be disjoint. The recovered selector intentionally leaves one plane
// untouched, including when all three gains are non-unity.
BayerGainStatus apply_bayer_gain_rgb_row_007ef1f4(
    float* r, std::size_t r_count, float* g, std::size_t g_count,
    float* b, std::size_t b_count, std::size_t width,
    const BayerGainPrepared& prepared) noexcept;

// Recovered one-row CFA consumer from FUN_007ef1f4 on canonical global RGGB.
// row_parity and col_parity identify the global parity of row[0]. Width must
// be positive and even; padding after width is preserved.
BayerGainStatus apply_bayer_gain_cfa_row_007ef1f4(
    float* row, std::size_t count, std::size_t width,
    unsigned row_parity, unsigned col_parity,
    const BayerGainPrepared& prepared) noexcept;

}  // namespace imprint::acr
