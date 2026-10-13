// Research-only continuous TicoRAW entropy harness for pinned redlamp-libraw.
// It calls the public decode_precinct API and writes complete A/B scatter
// stripes for each image-orchestrated precinct call. Each fixed-size record
// is bufA followed by bufB. No Adobe code is used.
#include "src/decoders/nikon_he/nikon_he_picture_header.h"
#include "src/decoders/nikon_he/nikon_he_subband_config.h"
#include "src/decoders/nikon_he/nikon_he_precinct_header.h"
#include "src/decoders/nikon_he/nikon_he_precinct_decode.h"
#include "src/decoders/nikon_he/nikon_he_predecessor.h"
#include "src/decoders/nikon_he/nikon_he_tile.h"
#include "src/decoders/nikon_he/nikon_he_predict_lut.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace nikon_he;

namespace {
struct Packet {
    size_t offset;
    size_t size;
};

uint32_t be24(const uint8_t* p) {
    return (uint32_t(p[0]) << 16) | (uint32_t(p[1]) << 8) | uint32_t(p[2]);
}

struct CreatedFile {
    std::string path;
    std::FILE* file = nullptr;
    bool created = false;
    bool committed = false;

    bool open_exclusive() {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd < 0) return false;
        created = true;
        file = ::fdopen(fd, "wb");
        if (!file) {
            ::close(fd);
            ::unlink(path.c_str());
            created = false;
            return false;
        }
        return true;
    }

    bool close_file() {
        if (!file) return true;
        std::FILE* owned = file;
        file = nullptr;
        return std::fclose(owned) == 0;
    }

    ~CreatedFile() {
        if (file) std::fclose(file);
        if (created && !committed) ::unlink(path.c_str());
    }
};

bool write_i32le(std::FILE* out, const int32_t* values, size_t count,
                 std::vector<unsigned char>& encoded) {
    encoded.resize(count * 4);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t value = static_cast<uint32_t>(values[i]);
        encoded[i * 4] = static_cast<unsigned char>(value & 0xff);
        encoded[i * 4 + 1] = static_cast<unsigned char>((value >> 8) & 0xff);
        encoded[i * 4 + 2] = static_cast<unsigned char>((value >> 16) & 0xff);
        encoded[i * 4 + 3] = static_cast<unsigned char>((value >> 24) & 0xff);
    }
    return std::fwrite(encoded.data(), 1, encoded.size(), out) == encoded.size();
}

bool write_escaped(std::ostream& out, const std::string& value) {
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out.put('\\');
        if (c < 0x20) return false;
        out.put(static_cast<char>(c));
    }
    return bool(out);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
            "usage: harness NEF strip_offset strip_size AB.i32le metadata.json\n");
        return 64;
    }
    std::error_code path_error;
    const auto input_path = std::filesystem::absolute(argv[1], path_error).lexically_normal();
    if (path_error) return 65;
    const auto scatter_path = std::filesystem::absolute(argv[4], path_error).lexically_normal();
    if (path_error) return 65;
    const auto metadata_path = std::filesystem::absolute(argv[5], path_error).lexically_normal();
    if (path_error || scatter_path == input_path || metadata_path == input_path ||
        scatter_path == metadata_path) {
        std::fprintf(stderr, "input and output paths must be distinct normalized paths\n");
        return 65;
    }

    const uint64_t strip_offset = std::strtoull(argv[2], nullptr, 10);
    const uint64_t strip_size = std::strtoull(argv[3], nullptr, 10);
    if (strip_size < 100 || strip_size > 100000000) {
        std::fprintf(stderr, "invalid strip size: %llu\n",
                     static_cast<unsigned long long>(strip_size));
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 3;
    input.seekg(static_cast<std::streamoff>(strip_offset));
    std::vector<uint8_t> strip(static_cast<size_t>(strip_size));
    input.read(reinterpret_cast<char*>(strip.data()),
               static_cast<std::streamsize>(strip.size()));
    if (!input) {
        std::fprintf(stderr, "short read for requested RAW strip\n");
        return 3;
    }

    PictureHeader ph{};
    if (!parse_picture_header(strip.data(), strip.size(), ph) ||
        !is_supported_picture_header(ph, strip.size())) {
        std::fprintf(stderr, "public parser rejected picture header\n");
        return 4;
    }
    if (ph.precinct_offset > strip.size()) return 5;

    SubbandConfig config[26];
    LayoutInfo layout;
    compute_subband_layout(ph.hdr_width / 2, config, layout);
    const int stripe_ints = compute_buf_stripe_ints(&layout);
    const int pass_a_stride = 3 * layout.lift_LB + layout.memcpy_LB;
    if (stripe_ints != 4 * pass_a_stride || stripe_ints <= 0) return 6;
    for (int sb = 0; sb < 26; ++sb) {
        const int end = config[sb].x24 * 4 + config[sb].ng * 4;
        if (end > stripe_ints) {
            std::fprintf(stderr, "sub-band %d scatter end exceeds full stripe\n", sb);
            return 6;
        }
    }

    // Mirror nikon_he_decode.cpp's file-precinct packet walk, including the
    // 6-byte SLH after each group of 16 file precincts.
    std::vector<Packet> packets;
    size_t cursor = ph.precinct_offset;
    const int n_tiles = (ph.hdr_height + 63) / 64;
    const int max_file_packets = n_tiles * 16 + 2;
    for (int p = 0; p < max_file_packets; ++p) {
        if (cursor + 2 <= strip.size() && strip[cursor] == 0xff &&
            strip[cursor + 1] == 0x11) break;
        if ((p & 15) == 0 && p > 0) {
            if (cursor + 6 > strip.size() || strip[cursor] != 0xff ||
                strip[cursor + 1] != 0x20 || strip[cursor + 2] != 0 ||
                strip[cursor + 3] != 4 ||
                ((uint32_t(strip[cursor + 4]) << 8) | strip[cursor + 5]) !=
                    uint32_t(p / 16)) {
                std::fprintf(stderr, "invalid/missing SLH before file precinct %d\n", p);
                return 7;
            }
            cursor += 6;
        }
        if (cursor + 3 > strip.size()) break;
        const uint32_t payload_size = be24(strip.data() + cursor);
        if (payload_size == 0) break;
        const size_t full_size = static_cast<size_t>(payload_size) + 12;
        if (full_size > strip.size() - cursor) {
            std::fprintf(stderr, "file precinct %d exceeds strip boundary\n", p);
            return 7;
        }
        packets.push_back(Packet{cursor, full_size});
        cursor += full_size;
    }
    if (packets.empty()) return 8;

    CreatedFile out_ab;
    out_ab.path = scatter_path.string();
    if (!out_ab.open_exclusive()) {
        std::fprintf(stderr, "cannot exclusively create scatter output: %s\n", std::strerror(errno));
        return 9;
    }
    const uint8_t* predict_lut = build_prediction_lookup_table();
    set_active_picture_header(&ph);
    std::vector<int32_t> buf_a(static_cast<size_t>(stripe_ints), 0);
    std::vector<int32_t> buf_b(static_cast<size_t>(stripe_ints), 0);
    std::vector<unsigned char> encoded;
    struct CallInfo { int tile, local, file; int bp, br; bool reset; };
    std::vector<CallInfo> calls;
    int call_index = 0;
    int reset_calls = 0;

    for (int tile = 0; tile < n_tiles; ++tile) {
        PrecinctPredecessorState pred_state;
        pred_state.init(config);
        const int base = tile * 16;
        int count = 0;
        for (int local = 0; local < 18; ++local) {
            const int file_index = base + local;
            if (file_index >= static_cast<int>(packets.size())) break;
            const Packet& packet = packets[static_cast<size_t>(file_index)];
            std::fill(buf_a.begin(), buf_a.end(), 0);
            std::fill(buf_b.begin(), buf_b.end(), 0);
            PrecinctSizes sizes{};
            if (!parse_precinct_header(strip.data() + packet.offset,
                                       packet.size, layout.sig_bytes, sizes)) {
                pred_state.destroy();
                set_active_picture_header(nullptr);
                return 10;
            }
            const bool reset = should_reset_gcli(pred_state.precinct_index(),
                                                 sizes.Bp, sizes.Br);
            const auto result = decode_precinct(
                strip.data() + packet.offset, packet.size, ph.hdr_width,
                config, pred_state, predict_lut, buf_a.data(), buf_b.data());
            if (!result.success || result.total_sub_bands_decoded != 26) {
                std::fprintf(stderr,
                    "decode_precinct failed call=%d tile=%d local=%d file=%d\n",
                    call_index, tile, local, file_index);
                pred_state.destroy();
                set_active_picture_header(nullptr);
                return 11;
            }
            if (!write_i32le(out_ab.file, buf_a.data(), buf_a.size(), encoded) ||
                !write_i32le(out_ab.file, buf_b.data(), buf_b.size(), encoded)) {
                pred_state.destroy();
                set_active_picture_header(nullptr);
                return 12;
            }
            calls.push_back(CallInfo{tile, local, file_index, sizes.Bp,
                                     sizes.Br, reset});
            if (reset) ++reset_calls;
            ++call_index;
            ++count;
        }
        pred_state.destroy();
        (void)count;
    }
    set_active_picture_header(nullptr);
    if (!out_ab.close_file()) return 13;

    CreatedFile meta_file;
    meta_file.path = metadata_path.string();
    if (!meta_file.open_exclusive()) {
        std::fprintf(stderr, "cannot exclusively create metadata output: %s\n", std::strerror(errno));
        return 14;
    }
    std::ostringstream meta;
    meta << "{\n  \"source_commit\":\"4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812\",\n"
         << "  \"public_api\":\"nikon_he::decode_precinct\",\n"
         << "  \"image_width\":" << ph.hdr_width << ",\n"
         << "  \"image_height\":" << ph.hdr_height << ",\n"
         << "  \"tile_count\":" << n_tiles << ",\n"
         << "  \"file_precinct_count\":" << packets.size() << ",\n"
         << "  \"calls\":" << calls.size() << ",\n"
         << "  \"stripe_ints\":" << stripe_ints << ",\n"
         << "  \"buffer_ints\":" << stripe_ints << ",\n"
         << "  \"record_order\":\"A then B; each is stripe_ints little-endian int32\",\n"
         << "  \"byte_order\":\"little_endian\",\n"
         << "  \"output\":\"";
    if (!write_escaped(meta, scatter_path.string())) return 15;
    meta << "\",\n  \"layout\":[";
    for (int sb = 0; sb < 26; ++sb) {
        if (sb) meta << ',';
        meta << "{\"sb\":" << sb << ",\"ng\":" << config[sb].ng
             << ",\"x24\":" << config[sb].x24
             << ",\"buffer_idx\":" << config[sb].buffer_idx << '}';
    }
    meta << "],\n  \"reset_call_count\":" << reset_calls
         << ",\n  \"call_map\":[";
    for (size_t i = 0; i < calls.size(); ++i) {
        if (i) meta << ',';
        const auto& c = calls[i];
        meta << "{\"call\":" << i << ",\"tile\":" << c.tile
             << ",\"local\":" << c.local << ",\"file_precinct\":" << c.file
             << ",\"Bp\":" << c.bp << ",\"Br\":" << c.br
             << ",\"reset\":" << (c.reset ? "true" : "false") << '}';
    }
    meta << "]\n}\n";
    const std::string metadata = meta.str();
    if (std::fwrite(metadata.data(), 1, metadata.size(), meta_file.file) != metadata.size()) return 16;
    if (!meta_file.close_file()) return 16;
    out_ab.committed = true;
    meta_file.committed = true;
    return 0;
}
