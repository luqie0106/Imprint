/*
 * Integer Nikon HE Bayer reconstruction kernels.
 *
 * Clean, per-pixel Metal reimplementation of the equations in
 * nikon_he_bayer.cpp by Dmitri Sotnikov (2026), from LibRaw. The reference is
 * distributed under the LGPL-2.1 or CDDL-1.0. No floating-point operations
 * are used here; signed lifting shifts explicitly round toward negative
 * infinity to match the reference C++ arithmetic shifts.
 */
#include <metal_stdlib>
using namespace metal;

constant uint kLutSize = 81792u;
constant int kLutShift = 2;
constant int kLutBitWidth = 14;
constant int kClipMax = (1 << kLutBitWidth) - 1;

struct TicoParams {
    uint rows;
    uint cols;
    uint lut_size;
};

// Define arithmetic right shift explicitly so negative coefficients have the
// same floor semantics on every Metal target.
static inline int floor_shift(int value, uint shift) {
    if (shift == 0u) return value;
    uint bits = as_type<uint>(value);
    uint shifted = bits >> shift;
    if (value < 0) shifted |= (~0u << (32u - shift));
    return as_type<int>(shifted);
}

static inline int clamped_next_col(uint c, uint cols) {
    return int((c + 1u < cols) ? c + 1u : c);
}

kernel void step1_merge_4_to_2(
    device const int* ll [[buffer(0)]],
    device const int* lh [[buffer(1)]],
    device const int* hh [[buffer(2)]],
    device const int* hl [[buffer(3)]],
    device int* out_l [[buffer(4)]],
    device int* out_h [[buffer(5)]],
    constant TicoParams& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = p.rows * p.cols;
    if (tid >= count) return;

    const uint r = tid / p.cols;
    const uint c = tid - r * p.cols;
    const uint prev_r = (r == 0u) ? 0u : r - 1u;
    const uint next_r = (r + 1u == p.rows) ? r : r + 1u;
    const uint row = r * p.cols;
    const uint prev_row = prev_r * p.cols;
    const uint next_row = next_r * p.cols;
    const uint cp = uint(clamped_next_col(c, p.cols));

    const int hh_col_sum = hh[row + c] + hh[row + cp];
    const int cur_predict = lh[row + c] -
        floor_shift(hh_col_sum + hh[prev_row + c] + hh[prev_row + cp], 3u);

    // At the final image row, extend this row's prediction. Recomputing from
    // the clamped source row changes the original inverse-lifting boundary.
    const int next_predict = (r + 1u == p.rows)
        ? cur_predict
        : lh[next_row + c] - floor_shift(
              hh_col_sum + hh[next_row + c] + hh[next_row + cp], 3u);

    const int ll_hl_sum = ll[row + c] + hl[row + c];
    out_l[tid] = cur_predict - floor_shift(
        ll_hl_sum + ll[row + cp] + hl[prev_row + c], 3u);

    if (c == 0u) {
        const int hh_predicted = hh[row + c] - floor_shift(
            ll_hl_sum + hl[row + c] + ll[next_row + c], 3u);
        // The first column has no incoming carry. The reference computes this
        // one-column carry as ((cur + next) * 2) >> 2.
        out_h[tid] = hh_predicted + floor_shift((cur_predict + next_predict) * 2, 2u);
    } else {
        const uint prev_c = c - 1u;
        const int hh_predicted = hh[row + c] - floor_shift(
            ll_hl_sum + hl[row + prev_c] + ll[next_row + c], 3u);
        const int prev_hh_col_sum = hh[row + prev_c] + hh[row + c];
        const int prev_cur_predict = lh[row + prev_c] - floor_shift(
            prev_hh_col_sum + hh[prev_row + prev_c] + hh[prev_row + c], 3u);
        const int prev_next_predict = (r + 1u == p.rows)
            ? prev_cur_predict
            : lh[next_row + prev_c] - floor_shift(
                  prev_hh_col_sum + hh[next_row + prev_c] + hh[next_row + c], 3u);
        out_h[tid] = hh_predicted + floor_shift(
            prev_cur_predict + cur_predict + next_predict + prev_next_predict, 2u);
    }
}

static inline ushort tone_lookup(device const int* lut, int value, uint lut_size) {
    const int max_index = int(min(lut_size, kLutSize)) - 1;
    const int index = clamp(value, 0, max_index);
    int result = floor_shift(lut[index] + (1 << (kLutShift - 1)), uint(kLutShift));
    result = clamp(result, 0, kClipMax);
    return ushort(result);
}

kernel void step2_bayer_rows(
    device const int* ll [[buffer(0)]],
    device const int* in_l [[buffer(1)]],
    device const int* in_h [[buffer(2)]],
    device const int* hl [[buffer(3)]],
    device const int* lut [[buffer(4)]],
    device ushort* out [[buffer(5)]],
    constant TicoParams& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = p.rows * p.cols;
    if (tid >= count) return;

    const uint r = tid / p.cols;
    const uint c = tid - r * p.cols;
    const uint prev_r = (r == 0u) ? 0u : r - 1u;
    const uint next_r = (r + 1u == p.rows) ? r : r + 1u;
    const uint row = r * p.cols;
    const uint prev_row = prev_r * p.cols;
    const uint next_row = next_r * p.cols;
    const uint cp = uint(clamped_next_col(c, p.cols));

    const int hh_c = in_h[row + c];
    const int hh_cp = in_h[row + cp];
    const int lh_c = in_l[row + c];
    const int lh_next_c = in_l[next_row + c];
    const int midpoint_bias = 1 << (kLutShift + kLutBitWidth - 1);

    int top0;
    int bot1;
    if (c == 0u) {
        top0 = floor_shift(in_h[prev_row + c] + lh_c + hh_c + lh_c, 2u)
             + midpoint_bias + ll[row + c];
        bot1 = hl[row + c] + midpoint_bias + floor_shift(
            hh_cp + hh_c + lh_c + lh_next_c, 2u);
    } else {
        top0 = floor_shift(in_l[row + c - 1u] + in_h[prev_row + c] + hh_c + lh_c, 2u)
             + midpoint_bias + ll[row + c];
        bot1 = hl[row + c] + midpoint_bias + floor_shift(
            hh_cp + hh_c + lh_c + lh_next_c, 2u);
    }

    const uint out_width = p.cols * 2u;
    const uint top = (r * 2u) * out_width + c * 2u;
    const uint bot = top + out_width;
    out[top] = tone_lookup(lut, top0, p.lut_size);
    out[top + 1u] = tone_lookup(lut, lh_c + midpoint_bias, p.lut_size);
    out[bot] = tone_lookup(lut, hh_c + midpoint_bias, p.lut_size);
    out[bot + 1u] = tone_lookup(lut, bot1, p.lut_size);
}
