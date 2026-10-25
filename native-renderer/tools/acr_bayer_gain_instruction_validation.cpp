#include "acr_bayer_gain.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>

#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
namespace {
float original_bias(float gain, float pivot) {
    float result, t, den, negative_t, positive, zero;
    // Transcription of 007ef334..364 / 007ef5b0..604, with b==1 excluded
    // by the prepared contract. This executes local arithmetic, not Adobe.
    asm volatile(
        "fmov %s[t], #-1.0\n"
        "fadd %s[d], %s[b], %s[t]\n"
        "fdiv %s[t], %s[t], %s[d]\n"
        "fneg %s[n], %s[t]\n"
        "fmov %s[z], wzr\n"
        "fmsub %s[d], %s[t], %s[b], %s[z]\n"
        "fdiv %s[d], %s[d], %s[t]\n"
        "fnmadd %s[p], %s[n], %s[b], %s[z]\n"
        "fdiv %s[p], %s[p], %s[t]\n"
        "fmadd %s[o], %s[g], %s[d], %s[p]\n"
        : [o] "=&w"(result), [t] "=&w"(t), [d] "=&w"(den),
          [n] "=&w"(negative_t), [p] "=&w"(positive), [z] "=&w"(zero)
        : [g] "w"(gain), [b] "w"(pivot));
    return result;
}

float original_pixel(float sample, float gain, float bias, bool clip) {
    float result, one, zero;
    asm volatile("fmadd %s[o], %s[x], %s[g], %s[b]"
        : [o] "=w"(result) : [x] "w"(sample), [g] "w"(gain), [b] "w"(bias));
    if (clip) {
        asm volatile(
            "fmov %s[one], #1.0\n"
            "fmov %s[zero], wzr\n"
            "fminnm %s[o], %s[o], %s[one]\n"
            "fcmp %s[o], #0.0\n"
            "fcsel %s[o], %s[zero], %s[o], mi\n"
            : [o] "+w"(result), [one] "=&w"(one), [zero] "=&w"(zero)
            : : "cc");
    }
    return result;
}

std::uint32_t bits(float x) {
    std::uint32_t result;
    std::memcpy(&result, &x, sizeof(result));
    return result;
}
}
#endif

int main() {
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    using namespace imprint::acr;
    std::uint32_t random = 0x7a53d917;
    auto next = [&]() { random = random * 1664525u + 1013904223u; return random; };
    for (int i = 0; i < 10000; ++i) {
        const double gains[3] = {0.01 + (next() % 100000) * .001, 1.0,
                                0.01 + (next() % 100000) * .001};
        const double pivot = (next() % 65535) / 65535.0;
        BayerGainPrepared prepared{};
        if (prepare_bayer_gain_007ef070(gains, 3, pivot, i & 1, &prepared) != BayerGainStatus::ok) return 1;
        for (int c = 0; c < 3; ++c) {
            if (bits(prepared.bias[c]) != bits(original_bias(prepared.gains[c], static_cast<float>(pivot)))) {
                std::cerr << "instruction bias mismatch at " << i << ',' << c << '\n';
                return 2;
            }
        }
        float row[2] = {static_cast<float>(static_cast<int>(next() % 6000) - 2000) / 1000.f, -0.f};
        const float expected0 = original_pixel(row[0], prepared.gains[0], prepared.bias[0], prepared.clip);
        const float expected1 = original_pixel(row[1], prepared.gains[1], prepared.bias[1], prepared.clip);
        if (apply_bayer_gain_cfa_row_007ef1f4(row, 2, 2, 0, 0, prepared) != BayerGainStatus::ok ||
            bits(row[0]) != bits(expected0) || bits(row[1]) != bits(expected1)) return 3;
    }
    std::cout << "10000 local ARM instruction controls/pixels matched\n";
    return 0;
#else
    std::cout << "ARM arithmetic instruction validation unavailable on this compiler/architecture\n";
    return 77;
#endif
}
