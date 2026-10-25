#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

// Recovered Camera Raw 17.2 ARM64 source-linearization stage.
//
// Static evidence (saved Ghidra project `/Camera_Raw_arm64.analysis-383152`):
//   FUN_013051cc  dng_linearization_info::Parse / tag copy (vtable 080a4780 +0x10)
//   FUN_01304c54  post-parse rational quantization of black/delta tables
//   FUN_01305598  object vtable +0x20: destination floor + stage submission
//   FUN_01304618  dng_linearization_image construction (one plane per input plane)
//   FUN_0130312c  dng_linearize_plane constructor: builds per-plane LUTs
//   FUN_01303c4c  dng_linearize_plane::Process: per-pixel consumer
//   FUN_01303a18  dng_linearization_info::MaxBlackLevel(plane)
//   FUN_00af4830  source vtable +0xc0: linearization driver, identity bypass
//   FUN_01310650  builds Stage2 and invokes object vtable +0x20
//
// This header intentionally models only the recovered integer paths that the
// actual Bayer CFA pipeline uses: input type 1 (uint8) or 3 (uint16) and output
// type 3 (uint16). The recovered float paths (input types 4 / 0xb, output type
// 0xb) are documented but rejected here because they are not on the closed
// Stage2 target path; see `LinearizationStatus::unsupported_type`.
//
// The recovered math, in the order the original executes it:
//
//   scale      = 1.0 / (white[p] - MaxBlackLevel(p))            // double
//   rows2D     = black_repeat_rows      when DeltaH exists, else
//                black_repeat_rows      when black_repeat_cols >= 2, else 0
//   cols2D     = rect_width (DeltaH) or black_repeat_cols
//   rowCount   = rect_height           when DeltaV exists, else
//                black_repeat_rows     when rows2D == 0 && repeat_rows > 1,
//                else 0
//
//   Branch A (rowCount == 0 && rows2D == 0):
//     n = (input == uint8) ? 0x100 : 0x10000
//     for i in [0, n):
//       v = linearization_table ? table[min(i, count - 1)] : i
//       d = scale * (v - black[0][0][p])
//       if clamp_flag: d = clamp01_fminnm(d)     // NaN -> 1.0
//       main_u16[i] = sat_u16(round_half_away(max16 + (0xffff - max16) * d))
//
//   Branch B (otherwise):
//     for i in [0, 0x10000):
//       v = linearization_table ? table[min(i, count - 1)] : i
//       main_i24[i] = round_half_away((max16 + (0xffff - max16) * (scale * v)) * 256.0)
//     black2D[r][c] = round_half_away(scale * (black[r][c % cols][p] + DeltaH[c]) * W8)
//     rowLUT[r]     = round_half_away(scale * (black[r % rows][0][p] + DeltaV[r]) * W8)
//                     (the black term is omitted when rows2D != 0)
//     W8 = (int)((max16 ^ 0xffff) << 8)  == (0xffff - max16) * 256
//
//   Process (integer, output uint16):
//     rowAbs = region.top - rect.top + rowRel
//     colAbs = region.left - rect.left + colRel
//     Branch A:  out = main_u16[pixel]
//     Branch B:  off = (uint32)(rowLUT ? rowLUT[rowAbs % rowCount] : 0) - 0x80
//                v   = (uint32)main_i24[pixel] - off        // wraps mod 2^32
//                if (cols2D) v -= (uint32)black2D[...]      // wraps mod 2^32
//                out = (v >> 24 == 0) ? (uint16)(v >> 8)
//                                     : ((v >> 31) ? 0 : 0xffff)
//
// The Branch B consumer is 32-bit throughout: 01303fe8 (`sub w22,w22,#0x80`),
// 01304050/0130405c (u8) and 01304440/0130444c (u16) are `sub w` operations
// whose results wrap, and 0130401c..01304030 selects the stored halfword from
// the bit pattern. Widening to int64 changes which values saturate.
//
// The destination floor (FUN_01305598) is:
//
//   ratio = max over planes of black_p / white_p   (only 0 < black < white)
//   max16 = round_half_up(ratio * 65535)           (<= 13107; ratio >= 0.2 -> 13107)
//   if (source has a mosaic, i.e. source byte +0x600 != 0) max16 = max(max16, 0x404)
//   clamp_flag = source_vtable_0x70() ^ 1
//
// `FUN_01305598` reuses the cached source byte +0x688 when the gate returns 0,
// when source dword +0x170 != 0, or when the output image type is not 3.

constexpr int kLinearizationMaxPlanes = 4;
constexpr int kLinearizationMaxBlackRepeat = 8;
constexpr std::uint16_t kLinearizationDestinationFloor = 0x0404;
constexpr std::uint32_t kLinearizationMax16Cap = 13107;
// Implementation resource guard, not a recovered Adobe constant. The recovered
// code allocates the 2D table as rows*cols*4 bytes without a visible cap; this
// bound keeps a hostile prepared state from exhausting host memory.
constexpr std::size_t kLinearizationMaxBlackTableEntries = 16u * 1024u * 1024u;

enum class LinearizationStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_plane = 2,
    invalid_black_repeat = 3,
    invalid_value = 4,          // non-finite / out-of-range recovered field
    invalid_rect = 5,
    invalid_region = 6,
    invalid_table = 7,
    unsupported_type = 8,       // only uint8/uint16 -> uint16 is implemented
    resource_limit = 9,         // implementation guard
    arithmetic_overflow = 10,   // recovered Round_int32 / Floor_uint32 guard
    allocation_failure = 11,
    not_built = 12,
    unsupported_aliasing = 13,  // overlapping buffers other than u16 in place
};

// `dng_linearization_info` as consumed by FUN_0130312c. Values are the tag
// values as FUN_013051cc would copy them, before the FUN_01304c54 quantization;
// `quantize_linearization_info` applies that quantization in place and must be
// called once before `SourceLinearization::build`.
struct LinearizationInfo {
    // linObj +0x08..+0x17, `dng_rect {t,l,b,r}`.
    std::int32_t rect_top = 0;
    std::int32_t rect_left = 0;
    std::int32_t rect_bottom = 0;
    std::int32_t rect_right = 0;

    // linObj +0x68 / +0x6c: fBlackLevelRepeatRows / fBlackLevelRepeatCols.
    std::uint32_t black_repeat_rows = 0;
    std::uint32_t black_repeat_cols = 0;

    // linObj +0x70..+0x86f: fBlackLevel[row][col][plane], row stride 0x100,
    // col stride 0x20, plane stride 8. Rows and cols are capped at 8 by the
    // recovered layout.
    double black_level[kLinearizationMaxBlackRepeat][kLinearizationMaxBlackRepeat]
                      [kLinearizationMaxPlanes] = {};

    // linObj +0x870: per-column black offset (DeltaH); +0x878: per-row offset
    // (DeltaV). The optional objects carry a byte count and a double array.
    bool has_black_level_delta_h = false;
    std::vector<double> black_level_delta_h;
    bool has_black_level_delta_v = false;
    std::vector<double> black_level_delta_v;

    // linObj +0x880 + 8*plane: absolute white level per plane.
    double white_level[kLinearizationMaxPlanes] = {0.0, 0.0, 0.0, 0.0};

    // linObj +0x60: optional uint16 LinearizationTable (+0x8 byte count,
    // +0x10 data). Index is clamped to count - 1.
    bool has_linearization_table = false;
    std::vector<std::uint16_t> linearization_table;

    // Input image +0x18: number of planes in Stage1/Stage2.
    std::uint32_t plane_count = 1;
};

enum class LinearizationInputType : std::uint32_t {
    u8 = 1,
    u16 = 3,
    u32 = 4,
    f32 = 11,
};

enum class LinearizationOutputType : std::uint32_t {
    u16 = 3,
    f32 = 11,
};

struct DestinationFloor {
    std::uint16_t max16 = 0;
    bool clamp_flag = true;  // == gate ^ 1, preserving the recovered XOR
    // `ok` unless the prepared plane/repeat fields are outside the recovered
    // layout; in that case max16 falls back to the cached value and no table
    // indexing happens.
    LinearizationStatus status = LinearizationStatus::ok;
};

// FUN_01304c54: quantize every fBlackLevel / DeltaH / DeltaV to a signed or
// unsigned rational with a common power-of-two denominator. `fBlackLevel` uses
// an unsigned rational (non-positive -> 0, round-half-up); deltas use a signed
// rational (round-half-away-from-zero). The denominator is the largest power of
// two <= 0x100 for which absMax * denominator < 1966080000.0.
LinearizationStatus quantize_linearization_info(LinearizationInfo* info) noexcept;

// FUN_01303a18: maximum black level over the repeat grid for one plane, with
// fBlack[r][c][p] + max(DeltaH over columns congruent to c) +
// max(DeltaV over rows congruent to r). (0,0) seeds the maximum; rows or cols
// equal to 0 make the function return 0.0.
//
// This is a public helper: a plane >= 4 or a repeat dimension outside [1, 8]
// is not representable in the fixed [8][8][4] layout and returns quiet NaN
// instead of indexing out of bounds (the build and the floor decision reject
// the same state explicitly). Non-finite black/delta values are outside the
// supported domain: `quantize_linearization_info` rejects them, and the folded
// comparison here matches the recovered `fcmp/fcsel` for finite inputs.
double max_black_level(const LinearizationInfo& info, std::uint32_t plane) noexcept;

// FUN_01305598 destination-floor decision. `source_gate` is the value returned
// by the source vtable +0x70 method (00affadc on vtable 08078068);
// `skip_recompute` models source dword +0x170 != 0, `cached_max16` models the
// u16 cache at source byte +0x688, and `source_has_mosaic` models source byte
// +0x600 != 0 (the cr_mosaic_info pointer). When the recovered early-out is
// taken the cached value is returned unchanged. An impossible plane count or
// repeat dimension is rejected through `DestinationFloor::status` (cached value
// returned, no table indexing).
DestinationFloor compute_destination_floor(
    const LinearizationInfo& info,
    std::uint32_t plane_count,
    LinearizationOutputType output_type,
    bool source_gate,
    bool skip_recompute,
    std::uint16_t cached_max16,
    bool source_has_mosaic) noexcept;

// One `dng_linearize_plane` (object size 0x60, FUN_0130312c). Tables are owned
// by the object and reused across Process calls, matching the recovered stage
// lifetime. `build` is failure-atomic: on error the previous tables and build
// state are preserved.
class SourceLinearization {
public:
    SourceLinearization() = default;
    SourceLinearization(const SourceLinearization&) = delete;
    SourceLinearization& operator=(const SourceLinearization&) = delete;

    // Build the per-plane LUTs. `input_type` must be u8 or u16 and `output_type`
    // must be u16; every other recovered combination is rejected with
    // `unsupported_type`. `info` must have been quantized first.
    LinearizationStatus build(
        const LinearizationInfo& info,
        std::uint32_t plane,
        LinearizationInputType input_type,
        LinearizationOutputType output_type,
        std::uint16_t max16,
        bool clamp_flag) noexcept;

    bool built() const noexcept { return built_; }
    std::uint32_t plane() const noexcept { return plane_; }
    bool branch_b() const noexcept { return !main_i24_.empty(); }
    std::uint32_t black_2d_rows() const noexcept { return black_2d_rows_; }
    std::uint32_t black_2d_cols() const noexcept { return black_2d_cols_; }
    std::uint32_t row_lut_count() const noexcept { return row_lut_count_; }
    const std::vector<std::uint16_t>& main_u16() const noexcept { return main_u16_; }
    const std::vector<std::int32_t>& main_i24() const noexcept { return main_i24_; }
    const std::vector<std::int32_t>& black_2d() const noexcept { return black_2d_; }
    const std::vector<std::int32_t>& row_lut() const noexcept { return row_lut_; }

    // FUN_01303c4c for one plane and one region. `input`/`output` point at the
    // first pixel of `region`; row strides are in samples.
    //
    // Aliasing: only the recovered u16 -> u16 call with identical base pointer
    // and identical row stride is supported in place. Every other overlapping
    // layout (u8 -> u16 at the same address, u16 partial overlap, different
    // strides) is rejected with `unsupported_aliasing` before the first store.
    //
    // Capacity: the caller owns both buffers and must provide
    // `(height - 1) * stride + width` elements at each pointer. A raw pointer
    // carries no length, so this function only rejects geometry whose span or
    // sample->byte conversion would overflow size_t; it cannot validate the
    // actual allocation. All checks run before the first write, so a failing
    // call leaves `output` untouched.
    //
    // Overlap is decided from the *declared* spans: a declared span that covers
    // the output buffer is rejected as `unsupported_aliasing` even when the real
    // allocations happen not to overlap, and an allocation shorter than the
    // declared span cannot be detected.
    LinearizationStatus process(
        const void* input,
        std::size_t input_row_stride_samples,
        void* output,
        std::size_t output_row_stride_samples,
        std::int32_t region_top,
        std::int32_t region_left,
        std::int32_t region_bottom,
        std::int32_t region_right) const noexcept;

    void reset() noexcept;

private:
    LinearizationStatus build_impl(
        const LinearizationInfo& info,
        std::uint32_t plane,
        LinearizationInputType input_type,
        LinearizationOutputType output_type,
        std::uint16_t max16,
        bool clamp_flag) noexcept(false);

    bool built_ = false;
    std::uint32_t plane_ = 0;
    LinearizationInputType input_type_ = LinearizationInputType::u16;
    LinearizationOutputType output_type_ = LinearizationOutputType::u16;
    std::int32_t rect_top_ = 0;
    std::int32_t rect_left_ = 0;
    std::int32_t rect_bottom_ = 0;
    std::int32_t rect_right_ = 0;
    std::uint32_t black_2d_rows_ = 0;
    std::uint32_t black_2d_cols_ = 0;
    std::uint32_t row_lut_count_ = 0;
    std::vector<std::uint16_t> main_u16_;
    std::vector<std::int32_t> main_i24_;
    std::vector<std::int32_t> black_2d_;
    std::vector<std::int32_t> row_lut_;
};

}  // namespace imprint::acr
