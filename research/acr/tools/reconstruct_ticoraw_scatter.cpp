// Research harness: independently decoded scatter buffers -> public IDWT/CFA.
// The entropy function below REPLACES the static library's decode_precinct.
// Lifting, tile carry, Bayer reconstruction and transfer-curve scheme remain
// redlamp-libraw 4abfcd2 reference code, not an independent Adobe implementation.
#include "src/decoders/nikon_he/nikon_he_decode.h"
#include "src/decoders/nikon_he/nikon_he_picture_header.h"
#include "src/decoders/nikon_he/nikon_he_precinct_decode.h"
#include "src/decoders/nikon_he/nikon_he_iqx_iqp_lut_data.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

static std::ifstream coefficients;
static size_t calls = 0;
static size_t buffer_ints = 0;

namespace nikon_he {
PrecinctDecodeResult decode_precinct(const uint8_t*, size_t, int,
    const SubbandConfig config[26], PrecinctPredecessorState&,
    const uint8_t*, int32_t* a, int32_t* b) {
    const size_t n = compute_buf_stripe_ints(config[0].layout_info);
    if (n != buffer_ints) return {0, false};
    coefficients.read(reinterpret_cast<char*>(a), n * sizeof(int32_t));
    coefficients.read(reinterpret_cast<char*>(b), n * sizeof(int32_t));
    if (!coefficients) return {0, false};
    ++calls;
    return {26, true};
}
}

static uint64_t number(const char* text) {
    char* end = nullptr;
    if (!*text || *text == '-') throw std::runtime_error("invalid nonnegative number");
    auto value = std::strtoull(text, &end, 10);
    if (*end) throw std::runtime_error("invalid decimal number");
    return value;
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error(
            "usage: reconstruct NEF strip_offset strip_size scatter.i32 output.cfa.u16");
        const auto output = std::filesystem::absolute(argv[5]).lexically_normal();
        if (std::filesystem::exists(output)) throw std::runtime_error("output already exists");
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
            throw std::runtime_error("little-endian host required");
        std::ifstream raw(argv[1], std::ios::binary);
        const auto offset = number(argv[2]), length = number(argv[3]);
        if (!raw || length < 100 || length > 100000000)
            throw std::runtime_error("invalid input or strip size");
        std::vector<uint8_t> strip(length);
        raw.seekg(offset);
        raw.read(reinterpret_cast<char*>(strip.data()), length);
        if (!raw) throw std::runtime_error("truncated RAW strip");
        nikon_he::PictureHeader ph{};
        if (!nikon_he::parse_picture_header(strip.data(), strip.size(), ph) ||
            !nikon_he::is_supported_picture_header(ph, strip.size()) ||
            !ph.has_vendor_tail || ph.precinct_offset >= strip.size())
            throw std::runtime_error("unsupported picture header");
        // Deliberately bounded to the real fixture whose packet stream and
        // complete coefficient records have been independently validated.
        if (ph.hdr_width != 6064 || ph.hdr_height != 4040)
            throw std::runtime_error("only validated 6064x4040 fixture supported");
        nikon_he::SubbandConfig cfg[26];
        nikon_he::LayoutInfo layout;
        nikon_he::compute_subband_layout(ph.hdr_width / 2, cfg, layout);
        buffer_ints = nikon_he::compute_buf_stripe_ints(&layout);
        const size_t expected_calls = 1136;
        if (std::filesystem::file_size(argv[4]) != expected_calls * 2 * buffer_ints * 4)
            throw std::runtime_error("scatter size does not match complete tile schedule");
        coefficients.open(argv[4], std::ios::binary);
        if (!coefficients) throw std::runtime_error("cannot read scatter file");
        // This transfer curve is the fixed public implementation's scheme.
        // It has NOT been independently recovered from ACR parameters.
        const uint8_t* t = ph.vendor_tail;
        const int64_t B = ((int64_t)(t[3] & 15) << 8) | t[4];
        const int64_t A = ((int64_t)(t[7] & 3) << 16) | ((int64_t)t[8] << 8) | t[9];
        const int64_t C = ((int64_t)t[10] << 8) | t[11];
        std::vector<int32_t> lut(nikon_he::kIqxIqpLutSize);
        for (size_t i = 0; i < lut.size(); ++i) {
            const int64_t d = std::min<int64_t>(i, 65535) - C;
            const int64_t g = (d * A + 32768) >> 16;
            const int64_t q = d < 0 ? -d * d : g * g;
            const int64_t value = std::clamp<int64_t>(B + ((q + 163840) >> 18), 0, 16383);
            lut[i] = int32_t(4 * value);
        }
        std::vector<uint16_t> cfa(size_t(ph.hdr_width) * ph.hdr_height);
        nikon_he::set_active_picture_header(&ph);
        const auto start = std::chrono::steady_clock::now();
        const auto result = nikon_he::decode_nikon_he_image(
            strip.data() + ph.precinct_offset, strip.size() - ph.precinct_offset,
            ph.hdr_width, ph.hdr_height, lut.data(), cfa.data());
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        nikon_he::set_active_picture_header(nullptr);
        if (!result.success || calls != expected_calls || coefficients.peek() != EOF)
            throw std::runtime_error("reconstruction failed or unconsumed coefficients");
        FILE* target = std::fopen(output.c_str(), "wx");
        if (!target) throw std::runtime_error("cannot exclusively create output");
        const bool written = std::fwrite(cfa.data(), 2, cfa.size(), target) == cfa.size();
        const bool closed = std::fclose(target) == 0;
        if (!written || !closed) {
            std::filesystem::remove(output);
            throw std::runtime_error("CFA write failed");
        }
        std::printf("{\"success\":true,\"width\":%u,\"height\":%u,"
            "\"tiles\":%d,\"scatter_calls\":%zu,\"buffer_ints\":%zu,"
            "\"transfer_B\":%lld,\"transfer_A\":%lld,\"transfer_C\":%lld,"
            "\"reconstruction_ms\":%.3f,\"public_entropy_called\":false}\n",
            ph.hdr_width, ph.hdr_height, result.tiles_decoded, calls, buffer_ints,
            (long long)B, (long long)A, (long long)C, ms);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "reconstruct error: %s\n", e.what());
        return 1;
    }
}
