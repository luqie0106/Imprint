#include "acr_profile_stage.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace imprint::acr {
namespace {
float bits(std::uint32_t value) noexcept {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

// Exact binary32 constants from 0135fb14..0135fbd4 and 0135ff70..01360078.
float encode(float x) noexcept {
    if (x <= bits(0x3b4d2e1c)) return x * bits(0x414eb852);
    float numerator = std::fma(x, bits(0xc5375cbc), bits(0x4686555f));
    for (auto value : {0x46d9a6f8u, 0x45308d78u, 0x41c4250cu, 0xbc9aaf34u})
        numerator = std::fma(numerator, x, bits(value));
    float denominator = std::fma(x, bits(0x4705a8a5), bits(0x4622fd1f));
    denominator = std::fma(denominator, x, bits(0x43a4f30a));
    denominator = std::fma(denominator, x, 1.0f);
    return numerator / denominator;
}

float decode(float x) noexcept {
    if (x <= bits(0x3d25aed5)) return x * bits(0x3d9e8391);
    float result = std::fma(x, bits(0xbdfcfc81), bits(0x3ed58f8e));
    for (auto value : {0xbf2169d3u, 0x3f4f6537u, 0x3efb441cu, 0x3d0fb070u, 0x3a60b5a8u})
        result = std::fma(result, x, bits(value));
    return result;
}

float clip(float value) noexcept {
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
}

bool span(const void *pointer, std::size_t count) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return pointer && address % alignof(float) == 0 &&
        count <= (std::numeric_limits<std::uintptr_t>::max() - address) / sizeof(float);
}

bool overlap(const void *a, std::size_t an, const void *b, std::size_t bn) noexcept {
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(b);
    return x < y + bn * sizeof(float) && y < x + an * sizeof(float);
}

float blend(float lower, float upper, float fraction) noexcept {
    const float product = fraction * upper;
    return std::fma(1.0f - fraction, lower, product);
}
}  // namespace

ProfileTableStatus apply_profile_stage_sdr(
    const float *input, float *output, std::size_t pixels,
    const ProfileTableView &table, unsigned encoding,
    double lookup_gain, double amount) noexcept {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "the bulk stage requires IEEE binary32");
    if (!pixels || pixels > (std::size_t{1} << 26) || encoding > 1 ||
        !std::isfinite(lookup_gain) || lookup_gain < 0 || lookup_gain > 65536 ||
        !std::isfinite(amount) || amount < 0 || amount > 2)
        return ProfileTableStatus::invalid_argument;
    const std::size_t values = pixels * 3;
    if (!span(input, values) || !span(output, values))
        return ProfileTableStatus::invalid_argument;
    if (validate_profile_table_view(table) != ProfileTableStatus::ok)
        return ProfileTableStatus::invalid_table;
    if (overlap(input, values, output, values) ||
        overlap(table.data, table.count * 3, output, values))
        return ProfileTableStatus::invalid_argument;
    try {
        const auto hcount = table.hue_count;
        const auto scount = table.saturation_count;
        const auto vcount = table.value_count;
        std::vector<float> prepared(vcount * (hcount + 1) * scount * 4, 0.0f);
        for (std::size_t v = 0; v < vcount; ++v) {
            for (std::size_t h = 0; h <= hcount; ++h) {
                for (std::size_t s = 0; s < scount; ++s) {
                    const auto src = ((v * hcount + (h == hcount ? 0 : h)) * scount + s) * 3;
                    const auto dst = ((v * (hcount + 1) + h) * scount + s) * 4;
                    prepared[dst] = table.data[src] * bits(0x3c888889);
                    prepared[dst + 1] = table.data[src + 1];
                    prepared[dst + 2] = table.data[src + 2];
                }
            }
        }
        std::vector<float> candidate(values);
        const float hue_scale = hcount > 1 ? static_cast<float>(hcount) * bits(0x3e2aaaab) : 0;
        const float gain = static_cast<float>(lookup_gain);
        const float intensity = static_cast<float>(amount);
        const bool encoded = encoding == 1 && vcount > 1;
        for (std::size_t p = 0; p < pixels; ++p) {
            const float r = input[p * 3], g = input[p * 3 + 1], b = input[p * 3 + 2];
            if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b) ||
                r < 0 || g < 0 || b < 0 || r > 1 || g > 1 || b > 1)
                return ProfileTableStatus::invalid_rgb;
            const float max_gb = g > b ? g : b, min_gb = g < b ? g : b;
            const float maximum = max_gb < r ? r : max_gb;
            const float minimum = min_gb > r ? r : min_gb;
            const float delta = maximum - minimum;
            float hue = 0, saturation = 0;
            if (delta > 0) {
                if (maximum == r) {
                    hue = (g - b) / delta;
                    if (hue < 0) hue += 6;
                } else if (maximum == g) hue = (b - r) / delta + 2;
                else hue = (r - g) / delta + 4;
                saturation = delta / maximum;
            }
            const float hc = hue_scale * hue;
            const float sc = saturation * static_cast<float>(scount - 1);
            const auto hi = std::min(static_cast<std::size_t>(hc), hcount - 1);
            const auto si = std::min(static_cast<std::size_t>(sc), scount - 2);
            const float hf = hc - static_cast<float>(hi), sf = sc - static_cast<float>(si);
            std::size_t vi = 0;
            float vf = 0;
            if (vcount > 1) {
                float lookup = clip(maximum * gain);
                if (encoded) lookup = encode(lookup);
                const float vc = lookup * static_cast<float>(vcount - 1);
                vi = std::min(static_cast<std::size_t>(vc), vcount - 2);
                vf = vc - static_cast<float>(vi);
            }
            auto at = [&](std::size_t v, std::size_t h, std::size_t s, unsigned c) {
                return prepared[((v * (hcount + 1) + h) * scount + s) * 4 + c];
            };
            float modifiers[3];
            for (unsigned c = 0; c < 3; ++c) {
                auto hs = [&](std::size_t v, std::size_t s) {
                    return blend(at(v, hi, s, c), at(v, hi + 1, s, c), hf);
                };
                const float low_s = vcount > 1 ? blend(hs(vi, si), hs(vi + 1, si), vf) : hs(vi, si);
                const float high_s = vcount > 1 ? blend(hs(vi, si + 1), hs(vi + 1, si + 1), vf) : hs(vi, si + 1);
                modifiers[c] = blend(low_s, high_s, sf);
            }
            const float sm = std::fma(intensity, modifiers[1] - 1.0f, 1.0f);
            const float vm = std::fma(intensity, modifiers[2] - 1.0f, 1.0f);
            const float mapped_s = std::min(saturation * (sm < 0 ? 0 : sm), 1.0f);
            const float base_v = encoded ? encode(maximum) : maximum;
            const float clipped_v = clip(base_v * (vm < 0 ? 0 : vm));
            const float mapped_v = encoded ? decode(clipped_v) : clipped_v;
            float red = mapped_v, green = mapped_v, blue = mapped_v;
            if (mapped_s > 0) {
                const float hue_delta = modifiers[0] * intensity;
                float wrapped = std::fmod(hue + hue_delta, 6.0f);
                if (wrapped < 0) wrapped += 6;
                const int sector = static_cast<int>(wrapped);
                const float fraction = wrapped - static_cast<float>(sector);
                const float pv = (1.0f - mapped_s) * mapped_v;
                const float qv = std::fma(-mapped_s, fraction, 1.0f) * mapped_v;
                const float tv = std::fma(-mapped_s, 1.0f - fraction, 1.0f) * mapped_v;
                switch (sector) {
                    case 1: red = qv; green = mapped_v; blue = pv; break;
                    case 2: red = pv; green = mapped_v; blue = tv; break;
                    case 3: red = pv; green = qv; blue = mapped_v; break;
                    case 4: red = tv; green = pv; blue = mapped_v; break;
                    case 5: red = mapped_v; green = pv; blue = qv; break;
                    default: red = mapped_v; green = tv; blue = pv; break;
                }
            }
            if (!std::isfinite(red) || !std::isfinite(green) || !std::isfinite(blue))
                return ProfileTableStatus::overflow;
            candidate[p * 3] = red;
            candidate[p * 3 + 1] = green;
            candidate[p * 3 + 2] = blue;
        }
        std::copy(candidate.begin(), candidate.end(), output);
        return ProfileTableStatus::ok;
    } catch (const std::bad_alloc &) {
        return ProfileTableStatus::allocation_failure;
    } catch (...) {
        return ProfileTableStatus::allocation_failure;
    }
}
}  // namespace imprint::acr
