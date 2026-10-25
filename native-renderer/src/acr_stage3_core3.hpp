#pragma once

#include <cstddef>

namespace imprint::acr {

// Recovered cr_stage_bayer_core3 CPU row kernels from Camera Raw 17.2 ARM64.
//
// The full-resolution pair (1,1) branch constructs cr_stage_bayer_core3 and
// FUN_007eaf20 is its CPU row executor. FUN_007eaf20 calls the function table
// initialized by FUN_013aad9c; the first two entries are:
//
//   0x08197068 -> FUN_0081376c   six gradient/edge rows
//   0x08197070 -> FUN_00813368   five-tap local energy (horizontal + vertical)
//
// The remaining entries (0080f2b8, 008134b8, 00814230, 008145b4) and the
// producer of the float input / consumer of the RGB output are not ported yet.
// These functions only implement the two recovered pixel consumers; they do
// not claim full-resolution RGB output. They are phase-agnostic: they consume
// float rows produced by the caller's input stage and do not read CFA
// metadata. The caller owns the CFA phase/scale, rect origin and tile/halo
// mapping; the row contracts below only describe the local guard/capacity
// requirements.
//
// ABI boundary (important): the original 0081376c/00813368 bodies begin with
// an alignment fixup. They compute (pointer >> 2) & 3 in float units, shift
// the source and output row pointers left by that many floats, add the same
// amount to the width, round the enlarged width up to a whole number of
// 4-float groups, and store that many floats. Relative to the original column
// 0 that fallback covers columns -shift .. width+3-shift, so it writes up to
// three floats before column 0 and up to three floats past width, and needs
// output slack. This port implements only the aligned path ((pointer & 0xF)
// == 0, so the fixup is a no-op), writes exactly `width` floats and rejects
// unaligned pointers with unsupported_alignment. It is a validated subset of
// the original ABI, not a full replica of the alignment/tail stores. Callers
// that feed the original alignment fallback must keep using the original
// kernel until that path is ported and tested separately.
enum class Core3Status : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    unsupported_remaining = 3,
    unsupported_alignment = 4,
};

// FUN_0081376c: writes six float rows of `width` samples each.
//
// Contract (aligned path):
//   * `center` points at column 0 of the center source row and is 16-byte
//     aligned.
//   * `source_stride` is in float samples, >= width, a multiple of 4, and
//     keeps center +/- stride and center +/- 2*stride 16-byte aligned.
//   * Rows center - 2*stride .. center + 2*stride must be readable.
//   * Each source row must expose 2 samples before column 0 and 2 samples
//     after column width-1 (the recovered kernel reads j-2 and j+2).
//   * `width` must be a positive multiple of 4 (the recovered loop stores
//     whole 4-float groups).
//   * Every output row must be 16-byte aligned and writable for `width`
//     floats, with `output_stride >= width`. Outputs may not alias the source
//     rows. This port does not use the original +3-float tail slack.
//   * On failure no output is written.
//
// Semantics, per output column j (confirmed by the 0081376c disassembly:
// uVar4 = stride*8 bytes selects the +/-2 rows, uVar5 = stride*4 bytes selects
// the +/-1 rows; the loads are C, C[j-2..j+1], C[j+2..j+5], Vminus2, Vplus2):
//   out0[j] = |C[j] - C[j+2]|
//   out1[j] = |C[j] - C[j-2]| + |C[j] - C[j+2]|
//   out2[j] = |C[j] - Vplus2[j]|
//   out3[j] = |C[j] - Vminus2[j]| + |C[j] - Vplus2[j]|
//   out4[j] = |(C[j-1] + C[j+1]) - (C[j] + C[j])|
//   out5[j] = |(Vminus1[j] + Vplus1[j]) - (C[j] + C[j])|
// where C is the center row, Vminus1/Vplus1 are center -/+ stride, and
// Vminus2/Vplus2 are center -/+ 2*stride. The per-column form is equivalent to
// the recovered 4-float group body for aligned, finite inputs.
//
// FUN_007eaf20 passes the six output rows in this order:
//   out0 -> workspace row 0,  out1 -> row 1,  out2 -> row 2,
//   out3 -> workspace row 5,  out4 -> row 3,  out5 -> row 4.
// FUN_00813368 then reads workspace rows 3 and 4 and writes rows 7 and 8.
Core3Status core3_gradient_rows_0081376c(
    const float *center,
    std::size_t source_stride,
    std::size_t width,
    float *const out[6],
    std::size_t output_stride) noexcept;

// FUN_00813368: writes two float rows of `width` samples each.
//
// The recovered body has two independent halves:
//   * `horizontal_row` -> five-tap local standard deviation along the row,
//     written to `out_horizontal`.
//   * `vertical_center` -> five-tap local standard deviation across rows
//     -2*stride .. +2*stride at the same column, written to `out_vertical`.
//
// Contract (aligned path):
//   * `horizontal_row` must expose 2 samples before and 2 after the width and
//     be 16-byte aligned.
//   * `vertical_center` must be 16-byte aligned and expose rows
//     -2*stride .. +2*stride; `row_stride` is in float samples, >= width and a
//     multiple of 4. Each of those rows must expose 2 samples after column
//     width-1 (the vertical phase reads columns up to width+1).
//   * `width` must be a positive multiple of 4.
//   * `out_horizontal`/`out_vertical` must be 16-byte aligned, hold `width`
//     floats and may not alias the source rows.
//   * On failure no output is written.
//
// For a row R and a five-sample window x[-2..2]:
//   mean = (x[-2] + x[-1] + x[0] + x[1] + x[2]) * 0.2f
//   out[j] = sqrt((x[2]-mean)^2 + (x[1]-mean)^2 + (x[0]-mean)^2
//                 + (x[-2]-mean)^2 + (x[-1]-mean)^2)
// The mean addition order and the squared-sum +2,+1,0,-2,-1 order are part of
// the recovered instruction order. For finite inputs the result is bit-stable
// against the independent float32 oracle; non-finite inputs propagate per
// IEEE-754 (NaN in -> NaN out, Inf differences/squares -> Inf or NaN) and are
// not clamped or rejected by the kernel.
Core3Status core3_local_energy_rows_00813368(
    const float *horizontal_row,
    const float *vertical_center,
    std::size_t row_stride,
    std::size_t width,
    float *out_horizontal,
    float *out_vertical) noexcept;

}  // namespace imprint::acr
