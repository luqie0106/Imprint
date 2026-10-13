/* -*- C++ -*-
 * Imprint TicoRAW inverse wavelet and Bayer reconstruction operators.
 *
 * Copyright (C) 2026 Imprint contributors.
 * The 5/3 lifting layout, vertical state transitions, and integer Bayer
 * equations are adapted from redlamp-libraw's Nikon HE implementation by
 * Dmitri Sotnikov. That implementation is available under either the GNU
 * Lesser General Public License version 2.1 or the Common Development and
 * Distribution License version 1.0. See the fixed LibRaw source tree's
 * LICENSE.LGPL and LICENSE.CDDL for the applicable terms.
 */

#include "ticoraw_lifting.hpp"

#include "decoders/nikon_he/nikon_he_subband_config.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>

namespace imprint_tico {
namespace {

constexpr int kLutShift = 2;
constexpr int kLutBitWidth = 14;
constexpr int kLutSize = 81792;
constexpr int kClipMax = (1 << kLutBitWidth) - 1;

int round_up(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

int64_t floor_shift(int64_t value, unsigned bits) {
    const int64_t divisor = int64_t{1} << bits;
    if (value >= 0) return value / divisor;
    return -((-value + divisor - 1) / divisor);
}

bool narrow_i32(int64_t value, int32_t& out) {
    if (value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    out = static_cast<int32_t>(value);
    return true;
}

bool inverse_53_one_level(const int32_t* low,
                          int low_count,
                          const int32_t* high,
                          int high_count,
                          int32_t* output) {
    const int count = low_count + high_count;
    if (count < 0 || (count > 0 && (!low || !high || !output))) return false;
    for (int i = 0; i < low_count; ++i) output[2 * i] = low[i];
    for (int i = 0; i < high_count; ++i) output[2 * i + 1] = high[i];

    // Inverse update on the even samples with whole-sample symmetric edges.
    for (int i = 0; i < count; i += 2) {
        const int32_t left = i > 0 ? output[i - 1]
                                   : (count > 1 ? output[1] : 0);
        const int32_t right = i + 1 < count ? output[i + 1]
                                            : (i > 0 ? output[i - 1] : 0);
        int32_t updated = 0;
        if (!narrow_i32(static_cast<int64_t>(output[i]) -
                            floor_shift(static_cast<int64_t>(left) + right + 2, 2),
                        updated)) {
            return false;
        }
        output[i] = updated;
    }

    // Inverse predict on the odd samples.
    for (int i = 1; i < count; i += 2) {
        const int32_t left = output[i - 1];
        const int32_t right = i + 1 < count ? output[i + 1] : output[i - 1];
        int32_t predicted = 0;
        if (!narrow_i32(static_cast<int64_t>(output[i]) +
                            floor_shift(static_cast<int64_t>(left) + right, 1),
                        predicted)) {
            return false;
        }
        output[i] = predicted;
    }
    return true;
}

bool inverse_horizontal_levels(const int32_t* packed,
                               int sample_count,
                               int levels,
                               const int* hl_offsets,
                               int32_t* output,
                               int32_t* work) {
    if (!packed || !hl_offsets || !output || !work || sample_count <= 0 ||
        levels < 1 || levels > 5) {
        return false;
    }
    std::array<int, 6> sizes{};
    sizes[0] = sample_count;
    for (int level = 1; level <= levels; ++level) {
        sizes[level] = (sizes[level - 1] + 1) / 2;
    }

    int32_t* current = (levels & 1) ? work : output;
    int32_t* destination = current == output ? work : output;
    const int deepest_size = sizes[levels];
    std::copy_n(packed, deepest_size, current);

    for (int level = levels; level >= 1; --level) {
        const int low_count = sizes[level];
        const int high_count = sizes[level - 1] - low_count;
        if (!inverse_53_one_level(current, low_count,
                                  packed + hl_offsets[level], high_count,
                                  destination)) {
            return false;
        }
        std::swap(current, destination);
    }
    return current == output;
}

bool write_sample(uint16_t* output, int64_t index, int64_t sample,
                  int64_t sample_count) {
    if (index < 0 || index >= sample_count) return true;
    if (sample < 0) sample = 0;
    if (sample > kClipMax) sample = kClipMax;
    output[index] = static_cast<uint16_t>(sample);
    return true;
}

}  // namespace

bool horizontal_pass(const int32_t* input,
                     const nikon_he::SubbandConfig config[26],
                     bool pass_a,
                     int32_t* output,
                     int32_t* work) noexcept {
    if (!input || !config || !config[0].layout_info || !output || !work) return false;
    const nikon_he::LayoutInfo* layout = config[0].layout_info;
    const int lift_st = layout->lift_LB * 4;
    const int memcpy_st = layout->memcpy_LB * 4;
    const int width = layout->ng_LL * 4;
    if (lift_st <= 0 || memcpy_st <= 0 || width <= 0) return false;

    if (pass_a) {
        constexpr int levels = 5;
        std::array<int, 6> offsets{};
        int cumulative = 0;
        for (int index = 0; index < 6; ++index) {
            cumulative += round_up(layout->ng_lift[index], 8);
            offsets[levels - index] = cumulative * 4;
        }
        offsets[0] = cumulative * 4;

        if (!inverse_horizontal_levels(input, width, levels, offsets.data(),
                                       output, work) ||
            !inverse_horizontal_levels(input + lift_st, width, levels,
                                       offsets.data(), output + lift_st, work)) {
            return false;
        }

        const int32_t* ll = input + config[12].x24 * 4;
        if (config[12].ng < 0 ||
            config[12].ng * 4 > memcpy_st) {
            return false;
        }
        std::copy_n(ll, config[12].ng * 4, output + 2 * lift_st);

        const int lb3_input_offset = 2 * lift_st + memcpy_st;
        const int lb3_output_offset = 2 * lift_st + memcpy_st;
        return inverse_horizontal_levels(input + lb3_input_offset, width,
                                         levels, offsets.data(),
                                         output + lb3_output_offset, work);
    }

    constexpr int levels = 1;
    const int high_offset = round_up(layout->ng_max, 8) * 4;
    // The reverse-cumulative convention uses [0] as the end sentinel and
    // [1] as the high-band start for a one-level transform.
    const int pass_b_offsets[2] = {2 * high_offset, high_offset};

    if (!inverse_horizontal_levels(input, width, levels, pass_b_offsets,
                                   output, work) ||
        !inverse_horizontal_levels(input + lift_st, width, levels,
                                   pass_b_offsets, output + lift_st, work)) {
        return false;
    }
    if (config[23].ng < 0 || config[23].ng * 4 > memcpy_st) return false;
    std::copy_n(input + config[23].x24 * 4, config[23].ng * 4,
                output + 2 * lift_st);

    const int lb7_offset = 2 * lift_st + memcpy_st;
    return inverse_horizontal_levels(input + lb7_offset, width, levels,
                                     pass_b_offsets,
                                     output + lb7_offset, work);
}

void vertical_lift_step(const int32_t* input,
                        int32_t* output,
                        int count,
                        VerticalLiftState& state) noexcept {
    if (state.failed || count < 0 || !state.x2_carry || !state.x3_carry ||
        (count > 0 && !output)) {
        state.failed = true;
        return;
    }
    if (input == nullptr && count > 0 &&
        state.state != VerticalState::State7 &&
        state.state != VerticalState::State9 &&
        state.state != VerticalState::State11) {
        state.failed = true;
        return;
    }

    const VerticalState initial_state = state.state;
    for (int i = 0; i < count; ++i) {
        int32_t next_x2 = state.x2_carry[i];
        int32_t next_x3 = state.x3_carry[i];
        int32_t value = 0;
        switch (initial_state) {
            case VerticalState::Init0:
                next_x2 = input[i];
                break;
            case VerticalState::Init1:
                if (!narrow_i32(static_cast<int64_t>(input[i]) * 4 - next_x2, next_x2)) {
                    state.failed = true;
                    return;
                }
                break;
            case VerticalState::Init4: {
                int32_t residual = 0;
                if (!narrow_i32(floor_shift(static_cast<int64_t>(next_x2) - input[i] + 1, 2), residual)) {
                    state.failed = true;
                    return;
                }
                next_x3 = residual;
                next_x2 = input[i];
                break;
            }
            case VerticalState::Init2:
                if (!narrow_i32(static_cast<int64_t>(input[i]) * 4, next_x2)) {
                    state.failed = true;
                    return;
                }
                break;
            case VerticalState::State5: {
                int32_t residual = 0;
                if (!narrow_i32(floor_shift(static_cast<int64_t>(next_x2) -
                                                static_cast<int64_t>(input[i]) * 2 + 1,
                                            2), residual)) {
                    state.failed = true;
                    return;
                }
                next_x3 = residual;
                next_x2 = input[i];
                break;
            }
            case VerticalState::State7:
                if (input == nullptr) {
                    value = next_x3;
                } else {
                    const int32_t previous_x3 = next_x3;
                    value = previous_x3;
                    if (!narrow_i32(static_cast<int64_t>(previous_x3) +
                                        static_cast<int64_t>(next_x2) * 2,
                                    next_x3) ||
                        !narrow_i32(static_cast<int64_t>(input[i]) * 4 - next_x2,
                                    next_x2)) {
                        state.failed = true;
                        return;
                    }
                }
                break;
            case VerticalState::State8: {
                int32_t residual = 0;
                if (!narrow_i32(floor_shift(static_cast<int64_t>(next_x2) - input[i] + 1, 2), residual) ||
                    !narrow_i32(floor_shift(static_cast<int64_t>(residual) + next_x3, 1), value)) {
                    state.failed = true;
                    return;
                }
                next_x2 = input[i];
                next_x3 = residual;
                break;
            }
            case VerticalState::State9:
                if (!narrow_i32(static_cast<int64_t>(next_x3) + next_x2, value)) {
                    state.failed = true;
                    return;
                }
                break;
            case VerticalState::State11:
                continue;
            default:
                state.failed = true;
                return;
        }
        if (initial_state == VerticalState::State7 ||
            initial_state == VerticalState::State8 ||
            initial_state == VerticalState::State9) {
            output[i] = value;
        }
        state.x2_carry[i] = next_x2;
        state.x3_carry[i] = next_x3;
    }

    // State transitions apply once per whole vector, including a zero-length
    // memcpy LB. The State7/8 changes above are only a per-element marker.
    if (count == 0) {
        switch (initial_state) {
            case VerticalState::Init0: state.state = VerticalState::Init1; break;
            case VerticalState::Init1: state.state = VerticalState::Init4; break;
            case VerticalState::Init4: state.state = VerticalState::State7; break;
            case VerticalState::Init2: state.state = VerticalState::State5; break;
            case VerticalState::State5: state.state = VerticalState::State7; break;
            case VerticalState::State7:
                state.state = input == nullptr ? VerticalState::State9 : VerticalState::State8;
                break;
            case VerticalState::State8: state.state = VerticalState::State7; break;
            case VerticalState::State9: state.state = VerticalState::State11; break;
            case VerticalState::State11: break;
        }
        return;
    }

    // For non-empty vectors, simple states transition after all elements.
    switch (initial_state) {
        case VerticalState::Init0: state.state = VerticalState::Init1; break;
        case VerticalState::Init1: state.state = VerticalState::Init4; break;
        case VerticalState::Init4: state.state = VerticalState::State7; break;
        case VerticalState::Init2: state.state = VerticalState::State5; break;
        case VerticalState::State5: state.state = VerticalState::State7; break;
        case VerticalState::State7:
            state.state = input == nullptr ? VerticalState::State9 : VerticalState::State8;
            break;
        case VerticalState::State8: state.state = VerticalState::State7; break;
        case VerticalState::State9: state.state = VerticalState::State11; break;
        case VerticalState::State11: break;
    }
}

bool reconstruct_cpu(const int32_t* ll,
                     const int32_t* lh,
                     const int32_t* hh,
                     const int32_t* hl,
                     int rows,
                     int cols,
                     int stride,
                     const int32_t* lut,
                     int lut_size,
                     uint16_t* output) noexcept {
    try {
    if (!ll || !lh || !hh || !hl || !lut || !output || rows <= 0 || cols <= 0 ||
        rows > std::numeric_limits<int>::max() / 2 ||
        cols > std::numeric_limits<int>::max() / 2 ||
            stride < cols || lut_size != kLutSize ||
            static_cast<uint64_t>(rows) * static_cast<uint64_t>(stride) >
                std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
            static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) * 4 >
                std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
            return false;
        }

        const size_t plane_count = static_cast<size_t>(rows) * stride;
        if (plane_count > std::numeric_limits<size_t>::max() / (2 * sizeof(int32_t))) {
            return false;
        }
        const size_t scratch_count = plane_count * 2;
        std::vector<int32_t> scratch(scratch_count, 0);
        constexpr int64_t midpoint_bias = int64_t{1} << (kLutShift + kLutBitWidth - 1);
        constexpr int64_t lut_rounding = int64_t{1} << (kLutShift - 1);
        const int64_t output_count = static_cast<int64_t>(rows) * 2 * cols * 2;

        auto p = [stride](const int32_t* plane, int row, int col) -> int32_t {
            return plane[static_cast<size_t>(row) * stride + col];
        };

        // First reconstruction pass: four wavelet bands become two planes.
        for (int row = 0; row < rows; ++row) {
            const int prev = row == 0 ? 0 : row - 1;
            const int next = row + 1 == rows ? row : row + 1;
            int32_t carry_current = 0;
            int32_t carry_next = 0;
            for (int col = 0; col < cols; ++col) {
                const int cp = col + 1 < cols ? col + 1 : col;
                const int32_t hh_cur = p(hh, row, col);
                const int32_t hh_cp = p(hh, row, cp);
                const int64_t hh_sum = static_cast<int64_t>(hh_cur) + hh_cp;

                const int64_t pred_cur_value = static_cast<int64_t>(p(lh, row, col)) -
                    floor_shift(hh_sum + p(hh, prev, col) + p(hh, prev, cp), 3);
                const int64_t pred_next_value = static_cast<int64_t>(p(lh, next, col)) -
                    floor_shift(hh_sum + p(hh, next, col) + p(hh, next, cp), 3);
                int32_t pred_cur = 0;
                int32_t pred_next = 0;
                if (!narrow_i32(pred_cur_value, pred_cur) ||
                    !narrow_i32(pred_next_value, pred_next)) return false;
                if (row + 1 == rows) pred_next = pred_cur;

                const int64_t llhl = static_cast<int64_t>(p(ll, row, col)) + p(hl, row, col);
                const int64_t out_low = static_cast<int64_t>(pred_cur) -
                    floor_shift(llhl + p(ll, row, cp) + p(hl, prev, col), 3);
                const int32_t hl_left = col == 0 ? p(hl, row, col)
                                                 : p(hl, row, col - 1);
                const int64_t hh_predicted = static_cast<int64_t>(hh_cur) -
                    floor_shift(llhl + hl_left + p(ll, next, col), 3);
                const int64_t merge = col == 0
                    ? floor_shift((static_cast<int64_t>(pred_cur) + pred_next) * 2, 2)
                    : floor_shift(static_cast<int64_t>(carry_current) + pred_cur +
                                      pred_next + carry_next,
                                  2);
                int32_t out_low_i32 = 0;
                int32_t out_high_i32 = 0;
                if (!narrow_i32(out_low, out_low_i32) ||
                    !narrow_i32(hh_predicted + merge, out_high_i32)) return false;

                scratch[static_cast<size_t>(row) * stride + col] = out_low_i32;
                scratch[plane_count + static_cast<size_t>(row) * stride + col] = out_high_i32;
                carry_current = pred_cur;
                carry_next = pred_next;
            }
        }

        // Second reconstruction pass: apply the transfer curve and write RGGB.
        for (int row = 0; row < rows; ++row) {
            const int prev = row == 0 ? 0 : row - 1;
            const int next = row + 1 == rows ? row : row + 1;
            for (int col = 0; col < cols; ++col) {
                const int cp = col + 1 < cols ? col + 1 : col;
                const int32_t ll_value = p(ll, row, col);
                const int32_t lh_value = scratch[static_cast<size_t>(row) * stride + col];
                const int32_t h_value = scratch[plane_count + static_cast<size_t>(row) * stride + col];
                const int32_t hl_value = p(hl, row, col);
                const int32_t lh_left = scratch[static_cast<size_t>(row) * stride +
                                                (col == 0 ? col : col - 1)];
                const int32_t lh_next = scratch[static_cast<size_t>(next) * stride + col];

                const int64_t top_even = static_cast<int64_t>(ll_value) + midpoint_bias +
                    floor_shift(static_cast<int64_t>(lh_left) +
                                    scratch[plane_count + static_cast<size_t>(prev) * stride + col] +
                                    lh_value + h_value,
                                2);
                const int64_t top_odd = static_cast<int64_t>(lh_value) + midpoint_bias;
                const int64_t bottom_even = static_cast<int64_t>(h_value) + midpoint_bias;
                const int64_t bottom_odd = static_cast<int64_t>(hl_value) + midpoint_bias +
                    floor_shift(static_cast<int64_t>(scratch[plane_count + static_cast<size_t>(row) * stride + cp]) +
                                    lh_value + h_value +
                                    lh_next,
                                2);

                const int64_t top_index = top_even < 0 ? 0 :
                    (top_even >= lut_size ? lut_size - 1 : top_even);
                const int64_t top_odd_index = top_odd < 0 ? 0 :
                    (top_odd >= lut_size ? lut_size - 1 : top_odd);
                const int64_t bottom_index = bottom_even < 0 ? 0 :
                    (bottom_even >= lut_size ? lut_size - 1 : bottom_even);
                const int64_t bottom_odd_index = bottom_odd < 0 ? 0 :
                    (bottom_odd >= lut_size ? lut_size - 1 : bottom_odd);
                const int64_t base = (static_cast<int64_t>(row) * 2) * (cols * 2) + col * 2;
                if (!write_sample(output, base, floor_shift(lut[top_index] + lut_rounding, kLutShift), output_count) ||
                    !write_sample(output, base + 1, floor_shift(lut[top_odd_index] + lut_rounding, kLutShift), output_count) ||
                    !write_sample(output, base + cols * 2, floor_shift(lut[bottom_index] + lut_rounding, kLutShift), output_count) ||
                    !write_sample(output, base + cols * 2 + 1, floor_shift(lut[bottom_odd_index] + lut_rounding, kLutShift), output_count)) {
                    return false;
                }
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace imprint_tico
