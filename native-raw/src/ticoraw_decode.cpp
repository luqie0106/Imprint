/* -*- C++ -*-
 * Imprint production Nikon TicoRAW entropy and image decoder.
 *
 * Copyright (C) 2026 Imprint contributors.
 * The Nikon HE profile layout and coding behavior are adapted from
 * redlamp-libraw's implementation by Dmitri Sotnikov, distributed under
 * LGPL-2.1 or CDDL-1.0. The bit readers, packet walk, entropy decode, strict
 * validation, and image-level orchestration below are Imprint C++ code.
 */

#include "ticoraw_decode.hpp"

#include "ticoraw_lifting.hpp"
#include "decoders/nikon_he/nikon_he_gtli_table.h"
#include "decoders/nikon_he/nikon_he_precinct_header.h"
#include "decoders/nikon_he/nikon_he_picture_header.h"
#include "decoders/nikon_he/nikon_he_predict_lut.h"
#include "decoders/nikon_he/nikon_he_subband_config.h"

#if defined(__APPLE__)
#include "ticoraw_metal.hpp"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace nikon_he {
namespace {

constexpr int kSubbandCount = 26;
constexpr int kLineBlockCount = 8;
constexpr int kDpbCount = 28;
constexpr int kPrecinctsPerTile = 18;
constexpr int kFilePrecinctsPerTile = 16;
constexpr int kStripesPerTile = 32;
constexpr int kLutSize = 81792;
constexpr int kMaxGcli = 30;

// ACR's 16-entry scalar inverse-quantization table. The final item is the
// format's unreachable sentinel for a 16-plane group.
constexpr int32_t kDequantScale[16] = {
    87381, 74898, 69905, 67650, 66576, 66052, 65793, 65664,
    65600, 65568, 65552, 65544, 65540, 65538, 65537, 0,
};

thread_local const char* g_last_backend = "none";

struct Packet {
    size_t offset = 0;
    size_t size = 0;
};

struct BitReader {
    BitReader(const uint8_t* source, size_t source_size) : data(source), size(source_size) {}

    bool read_bits(unsigned count, uint32_t& value) {
        if (count > 32 || count > remaining_bits()) return false;
        value = 0;
        for (unsigned bit = 0; bit < count; ++bit) {
            const size_t byte_index = bit_position >> 3;
            const unsigned bit_in_byte = static_cast<unsigned>(bit_position & 7u);
            value = (value << 1) |
                ((data[byte_index] >> (7u - bit_in_byte)) & 1u);
            ++bit_position;
        }
        return true;
    }

    bool read_unary(uint32_t& value) {
        value = 0;
        for (;;) {
            uint32_t bit = 0;
            if (!read_bits(1, bit)) return false;
            if (bit == 0) return true;
            if (++value > 31) return false;
        }
    }

    bool remaining_is_zero() const {
        for (size_t position = bit_position; position < size * 8; ++position) {
            const size_t byte_index = position >> 3;
            const unsigned bit_in_byte = static_cast<unsigned>(position & 7u);
            if ((data[byte_index] >> (7u - bit_in_byte)) & 1u) return false;
        }
        return true;
    }

    size_t remaining_bits() const {
        if (size > std::numeric_limits<size_t>::max() / 8) return 0;
        const size_t total_bits = size * 8;
        return bit_position <= total_bits ? total_bits - bit_position : 0;
    }

    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t bit_position = 0;
};

uint32_t read_be24(const uint8_t* bytes) {
    return (static_cast<uint32_t>(bytes[0]) << 16) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           static_cast<uint32_t>(bytes[2]);
}

uint16_t read_be16(const uint8_t* bytes) {
    return static_cast<uint16_t>((static_cast<uint16_t>(bytes[0]) << 8) |
                                 static_cast<uint16_t>(bytes[1]));
}

bool all_zero(const uint8_t* bytes, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (bytes[i] != 0) return false;
    }
    return true;
}

bool parse_packet_stream(const uint8_t* stream,
                         size_t stream_size,
                         int image_height,
                         std::vector<Packet>& packets) {
    if (!stream || image_height <= 0 || stream_size < 5) return false;
    const size_t expected_count = (static_cast<size_t>(image_height) + 3) / 4;
    if (expected_count == 0 || expected_count > static_cast<size_t>(INT32_MAX)) return false;
    packets.clear();
    packets.reserve(expected_count);

    size_t cursor = 0;
    for (size_t index = 0; index < expected_count; ++index) {
        if (index > 0 && index % kFilePrecinctsPerTile == 0) {
            if (cursor > stream_size || stream_size - cursor < 6 ||
                stream[cursor] != 0xff || stream[cursor + 1] != 0x20 ||
                read_be16(stream + cursor + 2) != 4 ||
                read_be16(stream + cursor + 4) != index / kFilePrecinctsPerTile) {
                return false;
            }
            cursor += 6;
        }

        if (cursor > stream_size || stream_size - cursor < 3) return false;
        if (stream[cursor] == 0xff && stream[cursor + 1] == 0x11) return false;
        const uint32_t payload_size = read_be24(stream + cursor);
        if (payload_size < 7) return false;
        const size_t packet_size = static_cast<size_t>(payload_size) + 12;
        if (packet_size > stream_size - cursor) return false;
        packets.push_back(Packet{cursor, packet_size});
        cursor += packet_size;
    }

    // The stream starts at the first precinct (the Nikon loader has already
    // consumed SLH 0). Require the exact image-derived count and terminal EOC.
    return cursor <= stream_size && stream_size - cursor == 2 &&
           stream[cursor] == 0xff && stream[cursor + 1] == 0x11;
}

bool parse_precinct(const uint8_t* data,
                    size_t size,
                    const LayoutInfo& layout,
                    PrecinctSizes& header) {
    if (!data || size < 12 || size > 0x00ffffffu + 12u) return false;
    const uint32_t declared_payload = read_be24(data);
    if (static_cast<uint64_t>(declared_payload) + 12u != size) return false;

    header = PrecinctSizes{};
    header.total_size = declared_payload;
    header.Bp = data[3];
    header.Br = data[4];
    if (header.Bp > 18 || header.Br > 31) return false;
    for (int index = 0; index < 7; ++index) {
        const uint8_t byte = data[5 + index];
        for (int lane = 0; lane < 4; ++lane) {
            const int dpb = index * 4 + lane;
            if (dpb < kDpbCount) {
                header.Dpb[dpb] = static_cast<uint8_t>((byte >> (6 - 2 * lane)) & 3u);
            }
        }
    }

    size_t cursor = 12;
    for (int lb = 0; lb < kLineBlockCount; ++lb) {
        if (cursor > size || size - cursor < 7) return false;
        const uint8_t* mini = data + cursor;
        uint64_t packed = 0;
        for (int byte = 0; byte < 7; ++byte) packed = (packed << 8) | mini[byte];
        const bool raw_gcli = ((packed >> 55) & 1u) != 0;
        const uint32_t data_bytes = static_cast<uint32_t>((packed >> 35) & 0xfffffu);
        const uint32_t gcli_bytes = static_cast<uint32_t>((packed >> 15) & 0xfffffu);
        const uint32_t sign_bytes = static_cast<uint32_t>(packed & 0x7fffu);
        const uint32_t sig_bytes = raw_gcli ? 0u : static_cast<uint32_t>(layout.sig_bytes[lb]);
        if (layout.sig_bytes[lb] < 0) return false;
        cursor += 7;

        const uint64_t payload_bytes = static_cast<uint64_t>(sig_bytes) + gcli_bytes +
                                       data_bytes + sign_bytes;
        if (payload_bytes > size - cursor) return false;
        header.lb_f20_sign[lb] = raw_gcli ? 1 : 0;
        header.lb_sig_bytes[lb] = sig_bytes;
        header.lb_gcli_bytes[lb] = gcli_bytes;
        header.lb_data_bytes[lb] = data_bytes;
        header.lb_sign_bytes[lb] = sign_bytes;
        header.lb_sig_offset[lb] = static_cast<uint32_t>(cursor);
        cursor += static_cast<size_t>(payload_bytes);
    }
    return cursor <= size && all_zero(data + cursor, size - cursor);
}

bool decode_gcli(int sb,
                 int group_count,
                 int gtli,
                 uint8_t mode,
                 bool raw_gcli,
                 BitReader& significance,
                 BitReader& gcli_reader,
                 const uint8_t* prediction_lut,
                 const std::vector<uint8_t>& previous,
                 const std::vector<uint8_t>* current_sb12,
                 std::vector<uint8_t>& decoded) {
    if (group_count <= 0 || gtli < 0 || gtli > 15 ||
        (mode != 0x70 && mode != 0x71 && mode != 0x73)) return false;
    decoded.assign(static_cast<size_t>(group_count), 0);
    if (raw_gcli) {
        for (int group = 0; group < group_count; ++group) {
            uint32_t value = 0;
            if (!gcli_reader.read_bits(4, value)) return false;
            decoded[static_cast<size_t>(group)] = static_cast<uint8_t>(value);
        }
        return true;
    }

    const std::vector<uint8_t>* prior = &previous;
    if (sb == 12) {
        prior = &previous;  // Previous precinct's sub-band 23 was moved here by the caller.
    } else if (sb == 23) {
        if (!current_sb12) return false;
        prior = current_sb12;
    }
    if (prior->size() < static_cast<size_t>(group_count)) return false;

    for (int base = 0; base < group_count; base += 8) {
        const int count = std::min(8, group_count - base);
        uint32_t significance_bit = 0;
        if (!significance.read_bits(1, significance_bit)) return false;
        if (significance_bit) {
            for (int i = 0; i < count; ++i) {
                const int prior_value = (*prior)[static_cast<size_t>(base + i)];
                if (prior_value > kMaxGcli) return false;
                decoded[static_cast<size_t>(base + i)] =
                    static_cast<uint8_t>(mode == 0x71 ? gtli : std::max(prior_value, gtli));
            }
            continue;
        }

        for (int i = 0; i < count; ++i) {
            uint32_t unary = 0;
            if (!gcli_reader.read_unary(unary)) return false;
            int value = 0;
            if (mode == 0x71) {
                value = gtli + static_cast<int>(unary);
            } else {
                const int prior_value = (*prior)[static_cast<size_t>(base + i)];
                if (prior_value > 15) return false;
                const uint8_t predicted = lookup_prediction(
                    prediction_lut, gtli, prior_value, static_cast<int>(unary));
                if (predicted == kPredictionLutSentinel) return false;
                value = predicted;
            }
            if (value < 0 || value > kMaxGcli) return false;
            decoded[static_cast<size_t>(base + i)] = static_cast<uint8_t>(value);
        }
    }
    return true;
}

bool dequantize_coefficient(int32_t signed_value,
                            int gcli,
                            int gtli,
                            int32_t& result) {
    result = 0;
    if (signed_value == 0 || gcli <= gtli) return true;
    const int bitplanes = gcli - gtli;
    if (bitplanes < 1 || bitplanes > 15) return false;
    const int64_t magnitude = (signed_value < 0 ? -static_cast<int64_t>(signed_value)
                                                : static_cast<int64_t>(signed_value)) >> gtli;
    const int64_t product = magnitude * kDequantScale[bitplanes - 1];
    const int64_t scaled = (product >> (16 - gtli)) << 4;
    if (scaled > std::numeric_limits<int32_t>::max()) return false;
    result = static_cast<int32_t>(signed_value < 0 ? -scaled : scaled);
    return true;
}

bool decode_tico_precinct(const uint8_t* data,
                          size_t size,
                          const SubbandConfig config[kSubbandCount],
                          const LayoutInfo& layout,
                          const uint8_t* prediction_lut,
                          std::array<std::vector<uint8_t>, kSubbandCount>& previous,
                          std::vector<int32_t>& buffer_a,
                          std::vector<int32_t>& buffer_b) {
    PrecinctSizes header{};
    if (!parse_precinct(data, size, layout, header)) return false;

    // The fixed profile uses at most 18 coded bit planes. The GTLI helper
    // computes this call's sub-band thresholds from LibRaw's active WGT header.
    std::array<uint8_t, kSubbandCount> gtli{};
    for (int sb = 0; sb < kSubbandCount; ++sb) {
        const uint8_t value = lookup_gtli_for_sub_band(header.Bp, header.Br, sb);
        if (value == 0xff || value > 15) return false;
        gtli[static_cast<size_t>(sb)] = value;
    }

    std::fill(buffer_a.begin(), buffer_a.end(), 0);
    std::fill(buffer_b.begin(), buffer_b.end(), 0);
    std::array<std::vector<uint8_t>, kSubbandCount> current;

    static constexpr int ranges[kLineBlockCount][2] = {
        {0, 6}, {6, 12}, {12, 13}, {13, 19},
        {19, 21}, {21, 23}, {23, 24}, {24, 26},
    };

    for (int lb = 0; lb < kLineBlockCount; ++lb) {
        const size_t sig_offset = header.lb_sig_offset[lb];
        const size_t sig_size = header.lb_sig_bytes[lb];
        const size_t gcli_offset = sig_offset + sig_size;
        const size_t gcli_size = header.lb_gcli_bytes[lb];
        const size_t data_offset = gcli_offset + gcli_size;
        const size_t data_size = header.lb_data_bytes[lb];
        const size_t sign_offset = data_offset + data_size;
        const size_t sign_size = header.lb_sign_bytes[lb];
        if (sign_offset > size || sign_size > size - sign_offset) return false;

        BitReader sig_reader(data + sig_offset, sig_size);
        BitReader gcli_reader(data + gcli_offset, gcli_size);
        BitReader magnitude_reader(data + data_offset, data_size);
        BitReader sign_reader(data + sign_offset, sign_size);
        const bool raw_gcli = header.lb_f20_sign[lb] != 0;

        for (int sb = ranges[lb][0]; sb < ranges[lb][1]; ++sb) {
            const int dpb_index = sb < 16 ? sb : 16 + (sb - 16);
            if (dpb_index < 0 || dpb_index >= kDpbCount) return false;
            const uint8_t mode = static_cast<uint8_t>(0x70u | header.Dpb[dpb_index]);
            const int groups = config[sb].ng;
            if (groups <= 0) return false;

            // sb12 uses the preceding precinct's sb23; sb23 uses this
            // precinct's sb12. Other bands keep their own prior GCLI vector.
            std::vector<uint8_t> previous_for_decode;
            const std::vector<uint8_t>* prior = &previous[static_cast<size_t>(sb)];
            const std::vector<uint8_t>* current_sb12 = nullptr;
            if (sb == 12) {
                previous_for_decode = previous[23];
                prior = &previous_for_decode;
            } else if (sb == 23) {
                current_sb12 = &current[12];
            }

            if (!decode_gcli(sb, groups, gtli[static_cast<size_t>(sb)], mode,
                             raw_gcli, sig_reader, gcli_reader, prediction_lut,
                             *prior, current_sb12, current[static_cast<size_t>(sb)])) {
                return false;
            }

            const size_t coefficient_count = static_cast<size_t>(groups) * 4;
            std::vector<int32_t> coefficients(coefficient_count, 0);
            const int threshold = gtli[static_cast<size_t>(sb)];
            for (int group = 0; group < groups; ++group) {
                const int bitplanes = static_cast<int>(current[static_cast<size_t>(sb)][group]) - threshold;
                if (bitplanes > 15) return false;
                if (bitplanes <= 0) continue;

                uint32_t magnitudes[4] = {0, 0, 0, 0};
                for (int plane = 0; plane < bitplanes; ++plane) {
                    uint32_t nibble = 0;
                    if (!magnitude_reader.read_bits(4, nibble)) return false;
                    magnitudes[0] = (magnitudes[0] << 1) | ((nibble >> 3) & 1u);
                    magnitudes[1] = (magnitudes[1] << 1) | ((nibble >> 2) & 1u);
                    magnitudes[2] = (magnitudes[2] << 1) | ((nibble >> 1) & 1u);
                    magnitudes[3] = (magnitudes[3] << 1) | (nibble & 1u);
                }
                for (int lane = 0; lane < 4; ++lane) {
                    const uint32_t magnitude = magnitudes[lane] << threshold;
                    if (magnitude > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) return false;
                    int32_t signed_value = static_cast<int32_t>(magnitude);
                    if (signed_value != 0) {
                        uint32_t sign = 0;
                        if (!sign_reader.read_bits(1, sign)) return false;
                        if (sign) signed_value = -signed_value;
                    }
                    if (!dequantize_coefficient(signed_value,
                                                current[static_cast<size_t>(sb)][group],
                                                threshold,
                                                coefficients[static_cast<size_t>(group) * 4 + lane])) {
                        return false;
                    }
                }
            }

            const int64_t start = static_cast<int64_t>(config[sb].x24) * 4;
            if (start < 0 || start + static_cast<int64_t>(coefficient_count) >
                                 static_cast<int64_t>(buffer_a.size())) return false;
            std::vector<int32_t>& target = config[sb].buffer_idx == 0 ? buffer_a : buffer_b;
            if (start + static_cast<int64_t>(coefficient_count) >
                static_cast<int64_t>(target.size())) return false;
            std::copy(coefficients.begin(), coefficients.end(), target.begin() + start);
        }

        if (!sig_reader.remaining_is_zero() || !gcli_reader.remaining_is_zero() ||
            !magnitude_reader.remaining_is_zero() || !sign_reader.remaining_is_zero()) {
            return false;
        }
    }

    previous = std::move(current);
    return true;
}

bool decode_tile(const std::vector<Packet>& packets,
                 const uint8_t* stream,
                 int tile_index,
                 int tile_packet_count,
                 int stripe_ints,
                 int pass_a_stride,
                 int lift_st,
                 int memcpy_st,
                 const SubbandConfig config[kSubbandCount],
                 const uint8_t* prediction_lut,
                 std::vector<int32_t>& tile_coefficients,
                 std::vector<int32_t>& overflow) {
    if (tile_packet_count <= 0 || tile_packet_count > kPrecinctsPerTile ||
        stripe_ints != pass_a_stride * 4 || lift_st <= 0 || memcpy_st <= 0) return false;

    const size_t bands_per_tile = static_cast<size_t>(kStripesPerTile) * stripe_ints;
    const size_t tile_base = static_cast<size_t>(tile_index) * bands_per_tile;
    if (tile_base + bands_per_tile > tile_coefficients.size() || overflow.size() !=
        static_cast<size_t>(2) * stripe_ints) return false;

    std::array<std::vector<uint8_t>, kSubbandCount> previous;
    for (int sb = 0; sb < kSubbandCount; ++sb) {
        previous[static_cast<size_t>(sb)].assign(static_cast<size_t>(config[sb].ng), 0);
    }

    std::vector<int32_t> x2_carry(static_cast<size_t>(4) * lift_st, 0);
    std::vector<int32_t> x3_carry(static_cast<size_t>(4) * lift_st, 0);
    std::array<imprint_tico::VerticalLiftState, 4> vertical{};
    for (int lb = 0; lb < 4; ++lb) {
        vertical[lb].state = tile_index == 0 ? imprint_tico::VerticalState::Init2
                                            : imprint_tico::VerticalState::Init0;
        vertical[lb].x2_carry = x2_carry.data() + lb * lift_st;
        vertical[lb].x3_carry = x3_carry.data() + lb * lift_st;
    }

    std::vector<int32_t> buffer_a(static_cast<size_t>(stripe_ints), 0);
    std::vector<int32_t> buffer_b(static_cast<size_t>(stripe_ints), 0);
    std::vector<int32_t> horizontal(static_cast<size_t>(4 * lift_st), 0);
    std::vector<int32_t> horizontal_work(static_cast<size_t>(4 * lift_st + 16), 0);
    std::vector<int32_t> tile_buffer(static_cast<size_t>(40) * stripe_ints, 0);
    if (tile_index > 0) {
        std::copy(overflow.begin(), overflow.end(), tile_buffer.begin());
    }

    int output_offset = tile_index == 0 ? 0 : 8;
    int memcpy_cursor = 0;
    const int kband = lift_st;
    for (int local = 0; local < tile_packet_count; ++local) {
        const int file_index = tile_index * kFilePrecinctsPerTile + local;
        if (file_index < 0 || static_cast<size_t>(file_index) >= packets.size()) return false;
        if (local == 16) {
            for (auto& band : previous) std::fill(band.begin(), band.end(), 0);
        }

        const Packet& packet = packets[static_cast<size_t>(file_index)];
        if (!decode_tico_precinct(stream + packet.offset, packet.size, config,
                                  *config[0].layout_info, prediction_lut,
                                  previous, buffer_a, buffer_b)) return false;

        std::fill(horizontal.begin(), horizontal.end(), 0);
        std::fill(horizontal_work.begin(), horizontal_work.end(), 0);
        if (!imprint_tico::horizontal_pass(buffer_a.data(), config, true,
                                           horizontal.data(), horizontal_work.data())) return false;

        const int64_t ll_write = static_cast<int64_t>(memcpy_cursor) + 3LL * kband;
        const int64_t ll_input = static_cast<int64_t>(2) * lift_st;
        if (ll_write < 0 || ll_write + memcpy_st > static_cast<int64_t>(tile_buffer.size()) ||
            ll_input + memcpy_st > static_cast<int64_t>(horizontal.size())) return false;
        std::copy_n(horizontal.data() + ll_input, memcpy_st,
                    tile_buffer.data() + ll_write);
        memcpy_cursor += stripe_ints;

        const int32_t* pass_a_inputs[4] = {
            horizontal.data(),
            horizontal.data() + lift_st,
            nullptr,
            horizontal.data() + 2 * lift_st + memcpy_st,
        };
        const bool skip_pass_a = tile_index > 0 && local == 0;
        if (!skip_pass_a) {
            const int pre_state = static_cast<int>(vertical[0].state);
            const int64_t at = static_cast<int64_t>(output_offset) * pass_a_stride;
            if (at < 0 || at + 3LL * lift_st > static_cast<int64_t>(tile_buffer.size())) return false;
            int32_t* destination = tile_buffer.data() + at;
            for (int lb = 0; lb < 4; ++lb) {
                vertical[lb].x2_carry = x2_carry.data() + lb * lift_st;
                vertical[lb].x3_carry = x3_carry.data() + lb * lift_st;
                const int count = lb == 2 ? 0 : lift_st;
                imprint_tico::vertical_lift_step(pass_a_inputs[lb], destination,
                                                  count, vertical[lb]);
                if (vertical[lb].failed) return false;
                if (lb != 2) destination += lift_st;
            }
            if (pre_state > static_cast<int>(imprint_tico::VerticalState::State5)) {
                if (output_offset > std::numeric_limits<int>::max() - 4) return false;
                output_offset += 4;
            }
        }

        std::fill(horizontal.begin(), horizontal.end(), 0);
        std::fill(horizontal_work.begin(), horizontal_work.end(), 0);
        if (!imprint_tico::horizontal_pass(buffer_b.data(), config, false,
                                           horizontal.data(), horizontal_work.data())) return false;
        const int64_t ll_write_b = static_cast<int64_t>(memcpy_cursor) + 3LL * kband;
        if (ll_write_b < 0 || ll_write_b + memcpy_st > static_cast<int64_t>(tile_buffer.size())) return false;
        std::copy_n(horizontal.data() + 2 * lift_st, memcpy_st,
                    tile_buffer.data() + ll_write_b);
        memcpy_cursor += stripe_ints;

        const int32_t* pass_b_inputs[4] = {
            horizontal.data(),
            horizontal.data() + lift_st,
            nullptr,
            horizontal.data() + 2 * lift_st + memcpy_st,
        };
        const int pre_state = static_cast<int>(vertical[0].state);
        const int64_t at = static_cast<int64_t>(output_offset) * pass_a_stride;
        if (at < 0 || at + 3LL * lift_st > static_cast<int64_t>(tile_buffer.size())) return false;
        int32_t* destination = tile_buffer.data() + at;
        for (int lb = 0; lb < 4; ++lb) {
            vertical[lb].x2_carry = x2_carry.data() + lb * lift_st;
            vertical[lb].x3_carry = x3_carry.data() + lb * lift_st;
            const int count = lb == 2 ? 0 : lift_st;
            imprint_tico::vertical_lift_step(pass_b_inputs[lb], destination,
                                              count, vertical[lb]);
            if (vertical[lb].failed) return false;
            if (lb != 2) destination += lift_st;
        }
        if (pre_state > static_cast<int>(imprint_tico::VerticalState::State5)) {
            if (output_offset > std::numeric_limits<int>::max() - 4) return false;
            output_offset += 4;
        }
    }

    const int32_t* tail_inputs[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int tail = 0; tail < 2; ++tail) {
        const int pre_state = static_cast<int>(vertical[0].state);
        const int64_t at = static_cast<int64_t>(output_offset) * pass_a_stride;
        if (at < 0 || at + 3LL * lift_st > static_cast<int64_t>(tile_buffer.size())) return false;
        int32_t* destination = tile_buffer.data() + at;
        for (int lb = 0; lb < 4; ++lb) {
            vertical[lb].x2_carry = x2_carry.data() + lb * lift_st;
            vertical[lb].x3_carry = x3_carry.data() + lb * lift_st;
            imprint_tico::vertical_lift_step(tail_inputs[lb], destination,
                                              lift_st, vertical[lb]);
            if (vertical[lb].failed) return false;
            if (lb != 2) destination += lift_st;
        }
        if (pre_state > static_cast<int>(imprint_tico::VerticalState::State5)) {
            if (output_offset > std::numeric_limits<int>::max() - 4) return false;
            output_offset += 4;
        }
    }

    std::copy_n(tile_buffer.begin(), static_cast<size_t>(kStripesPerTile) * stripe_ints,
                tile_coefficients.begin() + tile_base);
    std::copy_n(tile_buffer.begin() + static_cast<size_t>(kStripesPerTile) * stripe_ints,
                static_cast<size_t>(2) * stripe_ints, overflow.begin());
    return true;
}

enum class BackendRequest { Auto, Cpu, Metal, Invalid };

BackendRequest requested_backend() {
    const char* raw = std::getenv("IMPRINT_TICORAW_BACKEND");
    if (!raw || !*raw) return BackendRequest::Auto;
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "auto") return BackendRequest::Auto;
    if (value == "cpu") return BackendRequest::Cpu;
    if (value == "metal") return BackendRequest::Metal;
    return BackendRequest::Invalid;
}

}  // namespace

HeDecodeResult decode_nikon_he_image(const uint8_t* precinct_data,
                                     size_t precinct_stream_size,
                                     int image_width,
                                     int image_height,
                                     const int32_t* lut,
                                     uint16_t* out_bayer) noexcept {
    HeDecodeResult result{false, 0, 0};
    g_last_backend = "none";
    try {
        if (!precinct_data || !lut || !out_bayer || image_width < 64 ||
            image_height < 64 || (image_width & 1) || (image_height & 1) ||
            ((image_width / 2) & 3) != 0) {
            return result;
        }

        const BackendRequest backend = requested_backend();
        if (backend == BackendRequest::Invalid) return result;

        SubbandConfig config[kSubbandCount]{};
        LayoutInfo layout{};
        compute_subband_layout(image_width / 2, config, layout);
        if (layout.ng_LL <= 0 || layout.ng_max <= 0 || layout.lift_LB <= 0 ||
            layout.memcpy_LB <= 0) return result;
        const int pass_a_stride = 3 * layout.lift_LB + layout.memcpy_LB;
        const int stripe_ints = 4 * pass_a_stride;
        const int lift_st = 4 * layout.lift_LB;
        const int memcpy_st = 4 * layout.memcpy_LB;
        if (pass_a_stride <= 0 || stripe_ints <= 0 || lift_st > 8192) return result;
        for (int sb = 0; sb < kSubbandCount; ++sb) {
            const int64_t end = static_cast<int64_t>(config[sb].x24) * 4 +
                                static_cast<int64_t>(config[sb].ng) * 4;
            if (config[sb].ng <= 0 || end > stripe_ints) return result;
        }

        std::vector<Packet> packets;
        if (!parse_packet_stream(precinct_data, precinct_stream_size,
                                 image_height, packets)) return result;
        const int tile_count = (image_height + 63) / 64;
        if (tile_count <= 0 || packets.size() < static_cast<size_t>((tile_count - 1) * 16 + 1)) return result;
        const size_t tile_buffer_count = static_cast<size_t>(tile_count) *
                                         kStripesPerTile * stripe_ints;
        if (tile_buffer_count > std::numeric_limits<size_t>::max() / sizeof(int32_t)) return result;
        std::vector<int32_t> tile_coefficients(tile_buffer_count, 0);
        std::vector<int32_t> overflow(static_cast<size_t>(2) * stripe_ints, 0);
        const uint8_t* prediction_lut = build_prediction_lookup_table();
        if (!prediction_lut) return result;

        for (int tile = 0; tile < tile_count; ++tile) {
            const size_t base = static_cast<size_t>(tile) * kFilePrecinctsPerTile;
            if (base >= packets.size()) return HeDecodeResult{false, result.tiles_decoded, result.total_precincts};
            const int count = static_cast<int>(std::min<size_t>(
                kPrecinctsPerTile, packets.size() - base));
            if (!decode_tile(packets, precinct_data, tile, count,
                             stripe_ints, pass_a_stride, lift_st,
                             memcpy_st, config, prediction_lut,
                             tile_coefficients, overflow)) {
                return HeDecodeResult{false, result.tiles_decoded, result.total_precincts};
            }
            result.tiles_decoded++;
            result.total_precincts += count;
        }

        const int rows = image_height / 2;
        const int cols = image_width / 2;
        const int32_t* const tile_base = tile_coefficients.data();
        const int32_t* const ll = tile_base + lift_st;
        const int32_t* const lh = tile_base;
        const int32_t* const hh = tile_base + 3 * lift_st;
        const int32_t* const hl = tile_base + 2 * lift_st;

        bool reconstructed = false;
        if (backend != BackendRequest::Cpu) {
#if defined(__APPLE__)
            reconstructed = imprint_tico_reconstruct_metal(
                ll, lh, hh, hl, rows, cols, stripe_ints, lut,
                kLutSize, out_bayer);
#endif
            if (reconstructed) {
                g_last_backend = "metal";
            } else if (backend == BackendRequest::Metal) {
                return HeDecodeResult{false, result.tiles_decoded, result.total_precincts};
            }
        }
        if (!reconstructed) {
            reconstructed = imprint_tico::reconstruct_cpu(
                ll, lh, hh, hl, rows, cols, stripe_ints, lut,
                kLutSize, out_bayer);
            if (!reconstructed) return HeDecodeResult{false, result.tiles_decoded, result.total_precincts};
            g_last_backend = "cpu";
        }

        result.success = true;
        return result;
    } catch (...) {
        return HeDecodeResult{false, result.tiles_decoded, result.total_precincts};
    }
}

}  // namespace nikon_he

extern "C" const char* imprint_tico_last_backend() noexcept {
    return nikon_he::g_last_backend;
}
