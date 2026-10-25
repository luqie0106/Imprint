#include "acr_bayer_gain.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

using imprint::acr::BayerGainPrepared;
using imprint::acr::BayerGainStatus;
using imprint::acr::apply_bayer_gain_cfa_row_007ef1f4;
using imprint::acr::apply_bayer_gain_rgb_row_007ef1f4;
using imprint::acr::prepare_bayer_gain_007ef070;
using imprint::acr::prepare_bayer_gain_from_source_floor_007ef070;
using imprint::acr::prepare_bayer_gains_from_neutral_007efc64;

namespace {

int failures = 0;

#define CHECK(condition)                                                               \
    do {                                                                               \
        if (!(condition)) {                                                            \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

#define CHECK_STATUS(expression, expected)                                             \
    do {                                                                               \
        const BayerGainStatus status_value = (expression);                             \
        if (status_value != (expected)) {                                              \
            std::fprintf(stderr, "FAIL %s:%d: status %d != %d for %s\n", __FILE__,    \
                         __LINE__, static_cast<int>(status_value),                    \
                         static_cast<int>(expected), #expression);                    \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    static_assert(sizeof(result) == sizeof(value), "float bit helper requires binary32");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float from_bits(std::uint32_t value) {
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::uint64_t double_bits(double value) {
    std::uint64_t result = 0;
    static_assert(sizeof(result) == sizeof(value), "double bit helper requires binary64");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

BayerGainPrepared state(float gr, float gg, float gb,
                        float br, float bg, float bb, bool clip) {
    BayerGainPrepared result{{gr, gg, gb}, {br, bg, bb}, clip};
    return result;
}

bool same_prepared(const BayerGainPrepared& a, const BayerGainPrepared& b) {
    if (a.clip != b.clip) return false;
    for (int channel = 0; channel < 3; ++channel)
        if (bits(a.gains[channel]) != bits(b.gains[channel]) ||
            bits(a.bias[channel]) != bits(b.bias[channel]))
            return false;
    return true;
}

void test_prepare_exact_controls() {
    const double input[3] = {2.0, 3.0, 4.0};
    BayerGainPrepared prepared{};
    CHECK_STATUS(prepare_bayer_gain_007ef070(input, 3, 0.25, false, &prepared),
                 BayerGainStatus::ok);
    CHECK(bits(prepared.gains[0]) == 0x40000000u);
    CHECK(bits(prepared.gains[1]) == 0x40400000u);
    CHECK(bits(prepared.gains[2]) == 0x40800000u);
    CHECK(prepared.clip);

    // Hard-coded binary32 outputs from the documented sequence, including the
    // two divisions and explicit fused multiply-adds.
    CHECK(bits(prepared.bias[0]) == 0xbe800000u);
    CHECK(bits(prepared.bias[1]) == 0xbf000000u);
    CHECK(bits(prepared.bias[2]) == 0xbf400000u);

    // Inversion is in double before conversion. This input rounds to 1.0f,
    // while its double reciprocal rounds to the preceding binary32 value.
    const double near_one[3] = {1.00000004, 2.0, 3.0};
    CHECK_STATUS(prepare_bayer_gain_007ef070(near_one, 3, 0.25, true, &prepared),
                 BayerGainStatus::ok);
    CHECK(bits(static_cast<float>(1.0 / near_one[0])) == 0x3f7fffffu);
    CHECK(bits(prepared.gains[0]) == 0x3f7fffffu);
    CHECK(bits(static_cast<float>(1.0 / static_cast<double>(
              static_cast<float>(near_one[0])))) == 0x3f800000u);
    CHECK(!prepared.clip);

    // Clip is computed from the resulting double, even when a non-inverse
    // gain just above one rounds to a float gain of exactly one.
    CHECK_STATUS(prepare_bayer_gain_007ef070(near_one, 3, 0.25, false, &prepared),
                 BayerGainStatus::ok);
    CHECK(bits(prepared.gains[0]) == 0x3f800000u);
    CHECK(prepared.clip);
}

void test_prepare_failure_atomicity() {
    BayerGainPrepared out = state(7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f, true);
    const BayerGainPrepared original = out;
    const double good[3] = {1.0, 2.0, 3.0};
    const double bad_gain[3] = {1.0, 128.0, 3.0};
    CHECK_STATUS(prepare_bayer_gain_007ef070(good, 2, 0.25, false, &out),
                 BayerGainStatus::invalid_count);
    CHECK(same_prepared(out, original));
    CHECK_STATUS(prepare_bayer_gain_007ef070(bad_gain, 3, 0.25, false, &out),
                 BayerGainStatus::invalid_gain);
    CHECK(same_prepared(out, original));
    CHECK_STATUS(prepare_bayer_gain_007ef070(good, 3,
                 std::numeric_limits<double>::quiet_NaN(), false, &out),
                 BayerGainStatus::invalid_pivot);
    CHECK(same_prepared(out, original));
    CHECK_STATUS(prepare_bayer_gain_007ef070(good, 3, 1.0, false, &out),
                 BayerGainStatus::invalid_pivot);
    CHECK(same_prepared(out, original));
    const double inverse_overflow[3] = {1.0e-40, 2.0, 3.0};
    CHECK_STATUS(prepare_bayer_gain_007ef070(inverse_overflow, 3, 0.25, true, &out),
                 BayerGainStatus::invalid_gain);
    CHECK(same_prepared(out, original));
    CHECK_STATUS(prepare_bayer_gain_007ef070(nullptr, 3, 0.25, false, &out),
                 BayerGainStatus::null_pointer);
    CHECK_STATUS(prepare_bayer_gain_007ef070(good, 3, 0.25, false, nullptr),
                 BayerGainStatus::null_pointer);

    alignas(double) unsigned char misaligned_gain_storage[3 * sizeof(double) + 1] = {};
    const double* misaligned_gains =
        reinterpret_cast<const double*>(misaligned_gain_storage + 1);
    CHECK_STATUS(prepare_bayer_gain_007ef070(misaligned_gains, 3, 0.25, false, &out),
                 BayerGainStatus::misaligned_pointer);
    CHECK(same_prepared(out, original));

    alignas(BayerGainPrepared) unsigned char misaligned_out_storage[
        sizeof(BayerGainPrepared) + 1] = {};
    std::memset(misaligned_out_storage, 0x5a, sizeof(misaligned_out_storage));
    unsigned char original_out_bytes[sizeof(misaligned_out_storage)];
    std::memcpy(original_out_bytes, misaligned_out_storage, sizeof(misaligned_out_storage));
    BayerGainPrepared* misaligned_out =
        reinterpret_cast<BayerGainPrepared*>(misaligned_out_storage + 1);
    CHECK_STATUS(prepare_bayer_gain_007ef070(good, 3, 0.25, false, misaligned_out),
                 BayerGainStatus::misaligned_pointer);
    CHECK(std::memcmp(misaligned_out_storage, original_out_bytes,
                      sizeof(misaligned_out_storage)) == 0);
    // An aligned but unreadable gain address paired with a misaligned output
    // proves output alignment is rejected before the gain list is accessed.
    const double* unreadable_aligned_gains =
        reinterpret_cast<const double*>(static_cast<std::uintptr_t>(0x1000u));
    CHECK_STATUS(prepare_bayer_gain_007ef070(unreadable_aligned_gains, 3, 0.25,
                                             false, misaligned_out),
                 BayerGainStatus::misaligned_pointer);

    const std::uintptr_t max_address = std::numeric_limits<std::uintptr_t>::max();
    const std::uintptr_t wrapping_gain_address =
        max_address - (max_address % alignof(double));
    const double* wrapping_gains = reinterpret_cast<const double*>(wrapping_gain_address);
    CHECK_STATUS(prepare_bayer_gain_007ef070(wrapping_gains, 3, 0.25, false, &out),
                 BayerGainStatus::address_overflow);
    CHECK(same_prepared(out, original));
    const std::uintptr_t wrapping_output_address =
        max_address - (max_address % alignof(BayerGainPrepared));
    BayerGainPrepared* wrapping_out =
        reinterpret_cast<BayerGainPrepared*>(wrapping_output_address);
    CHECK_STATUS(prepare_bayer_gain_007ef070(good, 3, 0.25, false, wrapping_out),
                 BayerGainStatus::address_overflow);
    CHECK(same_prepared(out, original));

    BayerGainPrepared* overlapping_out =
        reinterpret_cast<BayerGainPrepared*>(const_cast<double*>(good));
    CHECK_STATUS(prepare_bayer_gain_007ef070(good, 3, 0.25, false, overlapping_out),
                 BayerGainStatus::overlap);
    CHECK(good[0] == 1.0 && good[1] == 2.0 && good[2] == 3.0);
}

void test_prepare_source_floor_adapter() {
    const double gains[3] = {2.0, 3.0, 4.0};
    const std::uint16_t floors[] = {0, 1040, 3992, 65535};
    static constexpr double reciprocal65535 = 0x1.0001000100010p-16;
    CHECK(double_bits(reciprocal65535) == 0x3ef0001000100010ULL);

    for (const std::uint16_t floor : floors) {
        const double pivot = static_cast<double>(floor) * reciprocal65535;
        BayerGainPrepared expected = state(7.0f, 8.0f, 9.0f,
                                           1.0f, 2.0f, 3.0f, true);
        BayerGainPrepared actual = expected;
        const BayerGainStatus expected_status =
            prepare_bayer_gain_007ef070(gains, 3, pivot, false, &expected);
        const BayerGainStatus actual_status =
            prepare_bayer_gain_from_source_floor_007ef070(
                gains, 3, floor, false, &actual);
        CHECK(actual_status == expected_status);
        CHECK(same_prepared(actual, expected));
        if (floor == 65535)
            CHECK(actual_status == BayerGainStatus::invalid_pivot);
        else
            CHECK(actual_status == BayerGainStatus::ok);
    }
}

void test_neutral_prepare_defaults_threshold_and_order() {
    double output[3] = {-9.0, -8.0, -7.0};
    const std::uint64_t default_gain_bits[3] = {
        0x3ff0000000000000ULL, 0x3fd999999999999aULL,
        0x3fe5555555555556ULL};
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(nullptr, 2, output, 3),
                 BayerGainStatus::ok);
    for (int channel = 0; channel < 3; ++channel)
        CHECK(double_bits(output[channel]) == default_gain_bits[channel]);

    // A deliberately unreadable but aligned address proves non-three counts
    // select the prepared default without touching the neutral pointer.
    const double* unreadable =
        reinterpret_cast<const double*>(static_cast<std::uintptr_t>(0x1000u));
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(unreadable, 2, output, 3),
                 BayerGainStatus::ok);
    for (int channel = 0; channel < 3; ++channel)
        CHECK(double_bits(output[channel]) == default_gain_bits[channel]);

    const double exact_threshold[3] = {0.125, 1.0, 0.5};
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(
                     exact_threshold, 3, output, 3),
                 BayerGainStatus::ok);
    CHECK(double_bits(output[0]) == double_bits(1.0));
    CHECK(double_bits(output[1]) == double_bits(0.125));
    CHECK(double_bits(output[2]) == double_bits(0.25));

    const double below_threshold[3] = {
        std::nextafter(0.125, 0.0), 1.0, 0.5};
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(
                     below_threshold, 3, output, 3),
                 BayerGainStatus::ok);
    for (int channel = 0; channel < 3; ++channel)
        CHECK(double_bits(output[channel]) == default_gain_bits[channel]);

    const double nonpositive[][3] = {{0.0, 1.0, 0.5}, {-1.0, 1.0, 0.5}};
    for (const auto& input : nonpositive) {
        CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(input, 3, output, 3),
                     BayerGainStatus::ok);
        for (int channel = 0; channel < 3; ++channel)
            CHECK(double_bits(output[channel]) == default_gain_bits[channel]);
    }

    // Hard-coded binary64 results preserve reciprocal, scaled multiplies,
    // minimum selection, and final divisions as separate rounded operations.
    const double ordered[3] = {
        0x1.b3935c7c631c6p-1, 0x1.c5f444851b987p-1,
        0x1.b3facd23b500ep-3};
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(ordered, 3, output, 3),
                 BayerGainStatus::ok);
    CHECK(double_bits(output[0]) == 0x3fd003ccb6c04d6bULL);
    CHECK(double_bits(output[1]) == 0x3fcebba284182a16ULL);
    CHECK(double_bits(output[2]) == 0x3ff0000000000000ULL);
    CHECK(double_bits(output[1]) != double_bits(
        std::min(ordered[0], std::min(ordered[1], ordered[2])) / ordered[1]));
}

void test_neutral_prepare_rejections_are_atomic() {
    double output[3] = {7.0, 8.0, 9.0};
    const double original[3] = {7.0, 8.0, 9.0};
    const double finite[3] = {0.5, 1.0, 0.75};
    const double nonfinite[][3] = {
        {std::numeric_limits<double>::quiet_NaN(), 1.0, 0.5},
        {0.5, std::numeric_limits<double>::infinity(), 1.0},
        {0.5, 1.0, -std::numeric_limits<double>::infinity()},
    };
    for (const auto& input : nonfinite) {
        CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(input, 3, output, 3),
                     BayerGainStatus::invalid_gain);
        CHECK(std::memcmp(output, original, sizeof(output)) == 0);
    }

    const double reciprocal_overflow[3] = {
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::denorm_min()};
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(
                     reciprocal_overflow, 3, output, 3),
                 BayerGainStatus::invalid_gain);
    CHECK(std::memcmp(output, original, sizeof(output)) == 0);

    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(nullptr, 3, output, 3),
                 BayerGainStatus::null_pointer);
    CHECK(std::memcmp(output, original, sizeof(output)) == 0);
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(finite, 3, output, 2),
                 BayerGainStatus::invalid_count);
    CHECK(std::memcmp(output, original, sizeof(output)) == 0);
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(finite, 3, nullptr, 3),
                 BayerGainStatus::null_pointer);

    alignas(double) unsigned char misaligned_storage[3 * sizeof(double) + 1] = {};
    const double* misaligned = reinterpret_cast<const double*>(misaligned_storage + 1);
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(misaligned, 3, output, 3),
                 BayerGainStatus::misaligned_pointer);
    CHECK(std::memcmp(output, original, sizeof(output)) == 0);
    double* misaligned_output = reinterpret_cast<double*>(misaligned_storage + 1);
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(finite, 3,
                                                           misaligned_output, 3),
                 BayerGainStatus::misaligned_pointer);
    CHECK(std::memcmp(output, original, sizeof(output)) == 0);

    const std::uintptr_t max_address = std::numeric_limits<std::uintptr_t>::max();
    const std::uintptr_t wrapping_address =
        max_address - (max_address % alignof(double));
    const double* wrapping_input = reinterpret_cast<const double*>(wrapping_address);
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(wrapping_input, 3, output, 3),
                 BayerGainStatus::address_overflow);
    CHECK(std::memcmp(output, original, sizeof(output)) == 0);
    double* wrapping_output = reinterpret_cast<double*>(wrapping_address);
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(finite, 3,
                                                           wrapping_output, 3),
                 BayerGainStatus::address_overflow);
    CHECK(std::memcmp(output, original, sizeof(output)) == 0);

    double aliased[3] = {0.5, 1.0, 0.75};
    const double aliased_original[3] = {0.5, 1.0, 0.75};
    CHECK_STATUS(prepare_bayer_gains_from_neutral_007efc64(aliased, 3, aliased, 3),
                 BayerGainStatus::overlap);
    CHECK(std::memcmp(aliased, aliased_original, sizeof(aliased)) == 0);
}

void test_rgb_selection_and_fma() {
    float r[3] = {0.0f, 1.0f, 2.0f};
    float g[3] = {0.0f, 1.0f, 2.0f};
    float b[3] = {7.0f, 8.0f, 9.0f};
    const BayerGainPrepared all_nonunity = state(2.0f, 3.0f, 4.0f,
                                                  0.25f, 0.5f, -0.75f, false);
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(r, 3, g, 3, b, 3, 3,
                                                   all_nonunity),
                 BayerGainStatus::ok);
    CHECK(bits(r[0]) == bits(0.25f) && bits(r[1]) == bits(2.25f) &&
          bits(r[2]) == bits(4.25f));
    CHECK(bits(g[0]) == bits(0.5f) && bits(g[1]) == bits(3.5f) &&
          bits(g[2]) == bits(6.5f));
    CHECK(b[0] == 7.0f && b[1] == 8.0f && b[2] == 9.0f);

    // FMA result differs from separately rounded multiply then add.
    BayerGainPrepared fused = state(from_bits(0x3f7ffffeu), 1.0f, 1.0f,
                                    -1.0f, 0.0f, 0.0f, false);
    float fr[1] = {from_bits(0x3f800001u)};
    float fg[1] = {11.0f};
    float fb[1] = {12.0f};
    const float separate = fr[0] * fused.gains[0] + fused.bias[0];
    const float fused_reference = std::fma(fr[0], fused.gains[0], fused.bias[0]);
    CHECK(bits(separate) != bits(fused_reference));
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(fr, 1, fg, 1, fb, 1, 1, fused),
                 BayerGainStatus::ok);
    CHECK(bits(fr[0]) == 0xa8800000u);
    CHECK(fg[0] == 11.0f && fb[0] == 12.0f);
}

void test_rgb_unity_placements() {
    struct Case { float gain[3]; int preserved; };
    const Case cases[] = {
        {{1.0f, 2.0f, 3.0f}, 0},
        {{2.0f, 1.0f, 3.0f}, 1},
        {{2.0f, 3.0f, 1.0f}, 2},
    };
    for (const Case& item : cases) {
        float r[2] = {1.0f, 2.0f};
        float g[2] = {1.0f, 2.0f};
        float b[2] = {1.0f, 2.0f};
        BayerGainPrepared prepared = state(item.gain[0], item.gain[1], item.gain[2],
                                           0.0f, 0.0f, 0.0f, false);
        CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(r, 2, g, 2, b, 2, 2, prepared),
                     BayerGainStatus::ok);
        float* planes[] = {r, g, b};
        CHECK(planes[item.preserved][0] == 1.0f && planes[item.preserved][1] == 2.0f);
        for (int channel = 0; channel < 3; ++channel) {
            if (channel == item.preserved) continue;
            CHECK(planes[channel][0] == item.gain[channel]);
            CHECK(planes[channel][1] == 2.0f * item.gain[channel]);
        }
    }

    float r[2] = {1.0f, 2.0f};
    float g[2] = {3.0f, 4.0f};
    float b[2] = {5.0f, 6.0f};
    const BayerGainPrepared unity = state(1.0f, 1.0f, 1.0f,
                                          0.0f, 0.0f, 0.0f, false);
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(r, 2, g, 2, b, 2, 2, unity),
                 BayerGainStatus::ok);
    CHECK(r[0] == 1.0f && r[1] == 2.0f && g[0] == 3.0f && g[1] == 4.0f &&
          b[0] == 5.0f && b[1] == 6.0f);

    // Selection is based on rounded float gain, while clip retains the double
    // decision made by prepare. The R plane rounds to unity and stays untouched.
    const double rounded_unity[3] = {1.00000004, 2.0, 3.0};
    BayerGainPrepared rounded{};
    CHECK_STATUS(prepare_bayer_gain_007ef070(rounded_unity, 3, 0.25, false, &rounded),
                 BayerGainStatus::ok);
    CHECK(bits(rounded.gains[0]) == 0x3f800000u && rounded.clip);
    float rr[1] = {4.0f};
    float gg[1] = {0.25f};
    float bb[1] = {0.25f};
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(rr, 1, gg, 1, bb, 1, 1, rounded),
                 BayerGainStatus::ok);
    CHECK(rr[0] == 4.0f);
}

void test_clip_signed_zero_and_cfa_parities() {
    const BayerGainPrepared clipped = state(2.0f, 3.0f, 4.0f,
                                            -0.0f, -0.0f, -0.0f, true);
    for (unsigned row_parity = 0; row_parity < 2; ++row_parity) {
        for (unsigned col_parity = 0; col_parity < 2; ++col_parity) {
            float row[8] = {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -0.0f};
            const float padding[2] = {91.0f, 92.0f};
            row[6] = padding[0];
            row[7] = padding[1];
            CHECK_STATUS(apply_bayer_gain_cfa_row_007ef1f4(row, 8, 6,
                                                           row_parity, col_parity,
                                                           clipped),
                         BayerGainStatus::ok);
            for (int i = 0; i < 4; ++i) CHECK(row[i] == 1.0f);
            CHECK(bits(row[4]) == 0x00000000u);  // strict-negative becomes +0
            CHECK(bits(row[5]) == 0x80000000u);  // -0 survives FMA/FMINNM/compare
            CHECK(row[6] == padding[0] && row[7] == padding[1]);
        }
    }

    const BayerGainPrepared mapping = state(2.0f, 3.0f, 4.0f,
                                            10.0f, 20.0f, 30.0f, false);
    const float expected[2][2][2] = {
        {{12.0f, 23.0f}, {23.0f, 12.0f}},
        {{23.0f, 34.0f}, {34.0f, 23.0f}},
    };
    for (unsigned rp = 0; rp < 2; ++rp) {
        for (unsigned cp = 0; cp < 2; ++cp) {
            float row[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            CHECK_STATUS(apply_bayer_gain_cfa_row_007ef1f4(row, 4, 4, rp, cp, mapping),
                         BayerGainStatus::ok);
            CHECK(row[0] == expected[rp][cp][0]);
            CHECK(row[1] == expected[rp][cp][1]);
            CHECK(row[2] == expected[rp][cp][0]);
            CHECK(row[3] == expected[rp][cp][1]);
        }
    }

    const BayerGainPrepared unclipped = state(2.0f, 3.0f, 4.0f,
                                              0.0f, 0.0f, 0.0f, false);
    float r[1] = {2.0f};
    float g[1] = {2.0f};
    float b[1] = {8.0f};
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(r, 1, g, 1, b, 1, 1, unclipped),
                 BayerGainStatus::ok);
    CHECK(r[0] == 4.0f && g[0] == 6.0f && b[0] == 8.0f);
}

void test_preflight_and_atomicity() {
    const BayerGainPrepared prepared = state(2.0f, 3.0f, 4.0f,
                                             0.0f, 0.0f, 0.0f, false);
    float r[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float g[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float b[4] = {9.0f, 10.0f, 11.0f, 12.0f};
    const float original_r[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float original_g[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    const float original_b[4] = {9.0f, 10.0f, 11.0f, 12.0f};
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(r, 4, g, 4, b, 4, 0, prepared),
                 BayerGainStatus::invalid_width);
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(r, 3, g, 4, b, 4, 4, prepared),
                 BayerGainStatus::invalid_count);
    CHECK(std::memcmp(r, original_r, sizeof(r)) == 0 &&
          std::memcmp(g, original_g, sizeof(g)) == 0 &&
          std::memcmp(b, original_b, sizeof(b)) == 0);

    b[2] = std::numeric_limits<float>::infinity();  // even the preserved plane is checked
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(r, 4, g, 4, b, 4, 4, prepared),
                 BayerGainStatus::nonfinite_sample);
    CHECK(std::memcmp(r, original_r, sizeof(r)) == 0 &&
          std::memcmp(g, original_g, sizeof(g)) == 0);
    std::memcpy(b, original_b, sizeof(b));

    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(r, 4, r + 1, 3, b, 4, 2, prepared),
                 BayerGainStatus::overlap);
    CHECK(std::memcmp(r, original_r, sizeof(r)) == 0);
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(nullptr, 4, g, 4, b, 4, 1, prepared),
                 BayerGainStatus::null_pointer);
    const std::size_t overlarge_count =
        static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(float) + 1;
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(r, overlarge_count,
                 g, overlarge_count, b, overlarge_count,
                 overlarge_count, prepared), BayerGainStatus::size_overflow);

    const std::uintptr_t overflow_address =
        std::numeric_limits<std::uintptr_t>::max() - sizeof(float) + 1u;
    float* overflow_pointer = reinterpret_cast<float*>(overflow_address);
    CHECK_STATUS(apply_bayer_gain_cfa_row_007ef1f4(overflow_pointer, 2, 2, 0, 0,
                                                   prepared),
                 BayerGainStatus::address_overflow);

    alignas(float) unsigned char misaligned_storage[sizeof(float) * 2 + 1] = {};
    float* misaligned = reinterpret_cast<float*>(misaligned_storage + 1);
    CHECK_STATUS(apply_bayer_gain_cfa_row_007ef1f4(misaligned, 2, 2, 0, 0, prepared),
                 BayerGainStatus::misaligned_pointer);
    CHECK_STATUS(apply_bayer_gain_cfa_row_007ef1f4(r, 4, 3, 0, 0, prepared),
                 BayerGainStatus::invalid_width);
    CHECK_STATUS(apply_bayer_gain_cfa_row_007ef1f4(r, 4, 2, 2, 0, prepared),
                 BayerGainStatus::invalid_parity);

    float cfa[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float original_cfa[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    cfa[3] = std::numeric_limits<float>::quiet_NaN();
    CHECK_STATUS(apply_bayer_gain_cfa_row_007ef1f4(cfa, 4, 4, 0, 0, prepared),
                 BayerGainStatus::nonfinite_sample);
    CHECK(bits(cfa[0]) == bits(original_cfa[0]) && bits(cfa[1]) == bits(original_cfa[1]) &&
          bits(cfa[2]) == bits(original_cfa[2]));

    BayerGainPrepared invalid = prepared;
    invalid.gains[1] = 0.0f;
    CHECK_STATUS(apply_bayer_gain_rgb_row_007ef1f4(r, 4, g, 4, b, 4, 1, invalid),
                 BayerGainStatus::invalid_prepared);
}

}  // namespace

int main() {
    test_prepare_exact_controls();
    test_prepare_failure_atomicity();
    test_prepare_source_floor_adapter();
    test_neutral_prepare_defaults_threshold_and_order();
    test_neutral_prepare_rejections_are_atomic();
    test_rgb_selection_and_fma();
    test_rgb_unity_placements();
    test_clip_signed_zero_and_cfa_parities();
    test_preflight_and_atomicity();
    if (failures != 0) {
        std::fprintf(stderr, "%d Bayer gain validation failure(s)\n", failures);
        return 1;
    }
    std::puts("ACR Bayer gain prepared validation passed");
    return 0;
}
