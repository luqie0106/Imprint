#include "../src/acr_postcamera_black.hpp"
#include "../src/acr_u16_lut.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

extern "C" {
int acr_test_postcamera_black(
    const float* src, std::size_t srcCount, std::size_t srcRowStride,
    std::size_t height, std::size_t rowSamples, double b, float* dst,
    std::size_t dstCount, std::size_t dstRowStride) noexcept;
int acr_test_postcamera_black_lut(double b, float* out,
                                  std::size_t count) noexcept;
int acr_test_postcamera_black_u16(
    const std::uint16_t* src, std::size_t srcCount,
    std::size_t srcRowStride, std::size_t height, std::size_t rowSamples,
    double b, std::uint16_t* dst, std::size_t dstCount,
    std::size_t dstRowStride) noexcept;
int acr_test_postcamera_black_u16_lut(double b, std::uint16_t* out,
                                      std::size_t count) noexcept;
}

namespace {

using imprint::acr::FloatLutPrepared;
using imprint::acr::PostcameraBlackStatus;
using imprint::acr::U16LutPrepared;

constexpr float kSentinel = -1234.5f;
int failures = 0;

void check(bool condition, const char* label) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", label);
    ++failures;
}

bool same_bits(float left, float right) {
    std::uint32_t left_bits = 0u;
    std::uint32_t right_bits = 0u;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0u;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float from_bits(std::uint32_t value) {
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

double black_sample(double x, double b) {
    const double d = b - 1.0;
    const double scale = -1.0 / d;
    const double u = x - b;
    const double v = std::fma(scale, u, +0.0);
    double result = std::fmin(v, 1.0);
    if (result < 0.0) result = +0.0;
    return result;
}

FloatLutPrepared reference_lut(double b) {
    FloatLutPrepared result{};
    for (std::size_t i = 0u; i <= imprint::acr::kFloatLutSegments; ++i) {
        const double x = static_cast<double>(i) * (1.0 / 4096.0);
        result.values[i] = static_cast<float>(black_sample(x, b));
    }
    result.values[imprint::acr::kFloatLutSegments + 1u] =
        result.values[imprint::acr::kFloatLutSegments];
    return result;
}

U16LutPrepared reference_u16_lut(const FloatLutPrepared& table) {
    U16LutPrepared result{};
    constexpr double kScale = 65535.0;
    const double step = static_cast<double>(imprint::acr::kFloatLutSegments) /
                        kScale;
    double left = static_cast<double>(table.values[0]);
    double right = static_cast<double>(table.values[1]);
    double base = std::fma(left, kScale, 0.5);
    double slope = (right - left) * kScale;
    std::size_t index = 1u;
    double fraction = +0.0;
    for (std::size_t i = 0u; i < imprint::acr::kU16LutEntryCount; ++i) {
        const double value = std::fma(slope, fraction, base);
        const std::int32_t signed_value = static_cast<std::int32_t>(value);
        result.values[i] = static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(signed_value));
        fraction = step + fraction;
        if (fraction > 1.0) {
            ++index;
            fraction = fraction + -1.0;
            const double next = static_cast<double>(table.values[index]);
            base = std::fma(right, kScale, 0.5);
            slope = (next - right) * kScale;
            right = next;
        }
    }
    return result;
}

float reference_apply(const FloatLutPrepared& table, float sample) {
    const float scaled = sample * 4096.0f;
    const int index = static_cast<int>(scaled);
    const float fraction = scaled - static_cast<float>(index);
    const float product = fraction * table.values[static_cast<std::size_t>(index) + 1u];
    const float complement = 1.0f - fraction;
    return std::fma(table.values[static_cast<std::size_t>(index)],
                    complement, product);
}

bool all_bits(const float* values, std::size_t count, float expected) {
    for (std::size_t i = 0u; i < count; ++i) {
        if (!same_bits(values[i], expected)) return false;
    }
    return true;
}

bool all_u16(const std::uint16_t* values, std::size_t count,
             std::uint16_t expected) {
    for (std::size_t i = 0u; i < count; ++i) {
        if (values[i] != expected) return false;
    }
    return true;
}

void test_table_construction() {
    constexpr double kBlack = 4100.0 / 65535.0;
    FloatLutPrepared actual{};
    const PostcameraBlackStatus status =
        imprint::acr::build_postcamera_black_lut(kBlack, &actual);
    check(status == PostcameraBlackStatus::ok,
          "build LUT for b=4100/65535 succeeds");

    const FloatLutPrepared expected = reference_lut(kBlack);
    for (std::size_t i = 0u; i < actual.values.size(); ++i) {
        if (!same_bits(actual.values[i], expected.values[i])) {
            check(false, "4098 LUT samples match independent double-FMA oracle");
            break;
        }
    }
    check(same_bits(actual.values[4096], actual.values[4097]),
          "table guard copies endpoint bits");

    FloatLutPrepared pivot{};
    check(imprint::acr::build_postcamera_black_lut(0.5, &pivot) ==
              PostcameraBlackStatus::ok,
          "build exact-grid pivot LUT succeeds");
    check(bits(pivot.values[2047]) == 0x00000000u &&
              bits(pivot.values[2048]) == 0x00000000u &&
              bits(pivot.values[2049]) == 0x3a000000u &&
              bits(pivot.values[4096]) == 0x3f800000u,
          "hard-coded b=0.5 table breakpoints match expected float bits");

    float table_out[imprint::acr::kFloatLutEntryCount];
    for (float& value : table_out) value = kSentinel;
    const int bridge_status = acr_test_postcamera_black_lut(
        kBlack, table_out, imprint::acr::kFloatLutEntryCount);
    check(bridge_status == static_cast<int>(PostcameraBlackStatus::ok) &&
              std::memcmp(table_out, actual.values.data(), sizeof(table_out)) == 0,
          "LUT bridge copies all 4098 entries");
    for (float& value : table_out) value = kSentinel;
    const int bad_count_status = acr_test_postcamera_black_lut(
        kBlack, table_out, imprint::acr::kFloatLutEntryCount - 1u);
    check(bad_count_status ==
              static_cast<int>(PostcameraBlackStatus::invalid_argument) &&
              all_bits(table_out, imprint::acr::kFloatLutEntryCount, kSentinel),
          "LUT bridge rejects a non-4098 count atomically");
    const int bad_black_status = acr_test_postcamera_black_lut(
        1.0, table_out, imprint::acr::kFloatLutEntryCount);
    check(bad_black_status == static_cast<int>(PostcameraBlackStatus::invalid_black) &&
              all_bits(table_out, imprint::acr::kFloatLutEntryCount, kSentinel),
          "LUT bridge rejects b=1 without changing output");

    FloatLutPrepared sentinel{};
    for (float& value : sentinel.values) value = kSentinel;
    check(imprint::acr::build_postcamera_black_lut(
              std::numeric_limits<double>::quiet_NaN(), &sentinel) ==
              PostcameraBlackStatus::invalid_black &&
              all_bits(sentinel.values.data(), sentinel.values.size(), kSentinel),
          "NaN black is rejected atomically");
    check(imprint::acr::build_postcamera_black_lut(-0.001, &sentinel) ==
              PostcameraBlackStatus::invalid_black &&
              all_bits(sentinel.values.data(), sentinel.values.size(), kSentinel),
          "negative black is rejected atomically");

    const double just_below_one = std::nextafter(1.0, 0.0);
    FloatLutPrepared near_one{};
    check(imprint::acr::build_postcamera_black_lut(just_below_one, &near_one) ==
              PostcameraBlackStatus::ok &&
              std::isfinite(near_one.values[4096]) &&
              same_bits(near_one.values[4096], near_one.values[4097]),
          "largest representable b below one builds a finite guarded LUT");
}

void test_lut_vs_direct_formula() {
    constexpr double kBlack = 4100.0 / 65535.0;
    const FloatLutPrepared table = reference_lut(kBlack);
    const float before_black = 256.25f / 4096.0f;
    const float table_value = reference_apply(table, before_black);
    const float direct_value = static_cast<float>(black_sample(before_black, kBlack));
    check(direct_value == 0.0f && table_value > 0.0f,
          "LUT interpolation expands the off-grid pivot below b");
    check(bits(table_value) == 0x384bbee9u,
          "hard-coded off-grid pivot LUT sample has recovered float bits");
}

void test_buffer_execution() {
    constexpr double kBlack = 4100.0 / 65535.0;
    const FloatLutPrepared table = reference_lut(kBlack);
    const float src[13] = {
        0.0f, 256.25f / 4096.0f, 256.5f / 4096.0f, 257.0f / 4096.0f,
        0.5f, 1.0f, 77.0f,
        0.125f, 0.25f, 0.75f, 0.875f, 0.4f, 0.55f};
    float dst[16];
    for (float& value : dst) value = kSentinel;
    const int status = acr_test_postcamera_black(
        src, 13u, 7u, 2u, 6u, kBlack, dst, 16u, 8u);
    check(status == static_cast<int>(PostcameraBlackStatus::ok),
          "multi-row arbitrary-sample-row black LUT succeeds");
    for (std::size_t y = 0u; y < 2u; ++y) {
        for (std::size_t x = 0u; x < 6u; ++x) {
            const float expected = reference_apply(table, src[y * 7u + x]);
            check(same_bits(dst[y * 8u + x], expected),
                  "image output matches unsigned LUT interpolation bits");
        }
    }
    check(same_bits(dst[6], kSentinel) && same_bits(dst[7], kSentinel) &&
              same_bits(dst[14], kSentinel) && same_bits(dst[15], kSentinel),
          "destination row padding and tail remain unchanged");

    const float pivot_src[] = {
        2047.0f / 4096.0f, 2048.0f / 4096.0f,
        2049.0f / 4096.0f, 1.0f};
    float pivot_dst[4]{};
    const int pivot_status = acr_test_postcamera_black(
        pivot_src, 4u, 4u, 1u, 4u, 0.5, pivot_dst, 4u, 4u);
    check(pivot_status == static_cast<int>(PostcameraBlackStatus::ok) &&
              bits(pivot_dst[0]) == 0x00000000u &&
              bits(pivot_dst[1]) == 0x00000000u &&
              bits(pivot_dst[2]) == 0x3a000000u &&
              bits(pivot_dst[3]) == 0x3f800000u,
          "hard-coded float image values straddling the table pivot match");

    float same_storage[] = {
        0.125f, 0.5f, 99.0f,
        0.75f, 1.0f, 88.0f};
    const int inplace_status = acr_test_postcamera_black(
        same_storage, 6u, 3u, 2u, 2u, 0.5, same_storage, 6u, 3u);
    check(inplace_status == static_cast<int>(PostcameraBlackStatus::ok) &&
              same_bits(same_storage[0], reference_apply(reference_lut(0.5), 0.125f)) &&
              same_bits(same_storage[1], reference_apply(reference_lut(0.5), 0.5f)) &&
              same_bits(same_storage[3], reference_apply(reference_lut(0.5), 0.75f)) &&
              same_bits(same_storage[4], 1.0f) && same_storage[2] == 99.0f &&
              same_storage[5] == 88.0f,
          "exact in-place rows succeed and preserve padding");

    const float bypass_src[] = {from_bits(0x80000000u), 1.0e-30f, 0.25f, 1.0f,
                                0.75f, 0.125f, 0.5f};
    float bypass_dst[8];
    for (float& value : bypass_dst) value = kSentinel;
    const int bypass_status = acr_test_postcamera_black(
        bypass_src, 7u, 4u, 2u, 3u, 0.0, bypass_dst, 8u, 4u);
    check(bypass_status == static_cast<int>(PostcameraBlackStatus::ok) &&
              same_bits(bypass_dst[0], bypass_src[0]) &&
              same_bits(bypass_dst[1], bypass_src[1]) &&
              same_bits(bypass_dst[2], bypass_src[2]) &&
              same_bits(bypass_dst[4], bypass_src[4]) &&
              same_bits(bypass_dst[5], bypass_src[5]) &&
              same_bits(bypass_dst[6], bypass_src[6]) &&
              same_bits(bypass_dst[3], kSentinel) &&
              same_bits(bypass_dst[7], kSentinel),
          "b=0 bypass preserves tiny values, negative zero, and padding bitwise");
}

void test_validation_and_atomicity() {
    float src[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 70.0f, 80.0f};
    float dst[10];
    float before[10];
    for (float& value : dst) value = kSentinel;

    auto invoke = [&](const float* input, std::size_t input_count,
                      std::size_t input_stride, std::size_t height,
                      std::size_t row_samples, double black, float* output,
                      std::size_t output_count,
                      std::size_t output_stride) {
        return static_cast<PostcameraBlackStatus>(acr_test_postcamera_black(
            input, input_count, input_stride, height, row_samples, black,
            output, output_count, output_stride));
    };

    std::memcpy(before, dst, sizeof(dst));
    check(invoke(src, 8u, 4u, 2u, 3u, 1.0, dst, 10u, 5u) ==
              PostcameraBlackStatus::invalid_black &&
              std::memcmp(dst, before, sizeof(dst)) == 0,
          "invalid black leaves every destination byte unchanged");

    src[1] = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(before, dst, sizeof(dst));
    check(invoke(src, 8u, 4u, 2u, 3u, 0.25, dst, 10u, 5u) ==
              PostcameraBlackStatus::non_finite_input &&
              std::memcmp(dst, before, sizeof(dst)) == 0,
          "non-finite active input is rejected atomically");
    src[1] = 1.25f;
    check(invoke(src, 8u, 4u, 2u, 3u, 0.25, dst, 10u, 5u) ==
              PostcameraBlackStatus::out_of_domain &&
              std::memcmp(dst, before, sizeof(dst)) == 0,
          "finite input outside [0,1] is rejected atomically");
    src[1] = 0.2f;

    check(invoke(src, 8u, 4u, 0u, 3u, 0.25, dst, 10u, 5u) ==
              PostcameraBlackStatus::invalid_dimensions,
          "zero height is rejected");
    check(invoke(src, 8u, 2u, 2u, 3u, 0.25, dst, 10u, 5u) ==
              PostcameraBlackStatus::invalid_stride,
          "short source stride is rejected");
    check(invoke(src, 5u, 4u, 2u, 3u, 0.25, dst, 10u, 5u) ==
              PostcameraBlackStatus::buffer_too_small,
          "short source capacity is rejected before reads");

    float aliased[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float alias_before[8];
    std::memcpy(alias_before, aliased, sizeof(aliased));
    check(invoke(aliased, 8u, 4u, 2u, 3u, 0.25, aliased + 1u, 7u, 4u) ==
              PostcameraBlackStatus::overlap &&
              std::memcmp(aliased, alias_before, sizeof(aliased)) == 0,
          "partial declared-span overlap is rejected atomically");

    const std::uintptr_t max_address =
        std::numeric_limits<std::uintptr_t>::max();
    const auto* wrapping_source = reinterpret_cast<const float*>(max_address - 3u);
    check(invoke(wrapping_source, 1u, 1u, 1u, 1u, 0.25, dst, 1u, 1u) ==
              PostcameraBlackStatus::arithmetic_overflow &&
              std::memcmp(dst, before, sizeof(dst)) == 0,
          "uintptr span overflow is rejected before input read");

    constexpr std::size_t kOverLimit = 128u * 1024u * 1024u + 1u;
    check(invoke(src, kOverLimit, kOverLimit, 1u, kOverLimit, 0.25,
                 dst, kOverLimit, kOverLimit) ==
              PostcameraBlackStatus::resource_limit,
          "active sample resource cap is enforced before buffer reads");
}

void test_u16_table_construction() {
    constexpr double kBlack = 4100.0 / 65535.0;
    U16LutPrepared actual{};
    const PostcameraBlackStatus status =
        imprint::acr::build_postcamera_black_u16_lut(kBlack, &actual);
    check(status == PostcameraBlackStatus::ok,
          "build unsigned mode1 table for b=4100/65535 succeeds");

    const U16LutPrepared expected = reference_u16_lut(reference_lut(kBlack));
    for (std::size_t i = 0u; i < actual.values.size(); ++i) {
        if (actual.values[i] != expected.values[i]) {
            std::fprintf(stderr,
                         "u16 LUT mismatch at %zu: got %u expected %u\n", i,
                         static_cast<unsigned>(actual.values[i]),
                         static_cast<unsigned>(expected.values[i]));
            check(false, "65536 mode1 entries match the independent FMA oracle");
            break;
        }
    }

    U16LutPrepared identity{};
    check(imprint::acr::build_postcamera_black_u16_lut(0.0, &identity) ==
              PostcameraBlackStatus::ok,
          "b=0 mode1 LUT builds successfully");
    bool identity_matches = true;
    for (std::size_t i = 0u; i < identity.values.size(); ++i) {
        if (identity.values[i] != static_cast<std::uint16_t>(i)) {
            identity_matches = false;
            break;
        }
    }
    check(identity_matches, "b=0 mode1 LUT is identity across all uint16 values");

    std::uint16_t bridge_table[imprint::acr::kU16LutEntryCount];
    for (std::uint16_t& value : bridge_table) value = 0xa55au;
    int bridge_status = acr_test_postcamera_black_u16_lut(
        kBlack, bridge_table, imprint::acr::kU16LutEntryCount);
    check(bridge_status == static_cast<int>(PostcameraBlackStatus::ok) &&
              std::memcmp(bridge_table, actual.values.data(),
                          sizeof(bridge_table)) == 0,
          "mode1 LUT bridge copies all 65536 entries");

    for (std::uint16_t& value : bridge_table) value = 0xa55au;
    bridge_status = acr_test_postcamera_black_u16_lut(
        kBlack, bridge_table, imprint::acr::kU16LutEntryCount - 1u);
    check(bridge_status ==
              static_cast<int>(PostcameraBlackStatus::invalid_argument) &&
              all_u16(bridge_table, imprint::acr::kU16LutEntryCount, 0xa55au),
          "mode1 LUT bridge rejects a non-65536 count atomically");
    bridge_status = acr_test_postcamera_black_u16_lut(
        1.0, bridge_table, imprint::acr::kU16LutEntryCount);
    check(bridge_status ==
              static_cast<int>(PostcameraBlackStatus::invalid_black) &&
              all_u16(bridge_table, imprint::acr::kU16LutEntryCount, 0xa55au),
          "invalid mode1 black leaves bridge table unchanged");

    U16LutPrepared sentinel{};
    for (std::uint16_t& value : sentinel.values) value = 0xa55au;
    check(imprint::acr::build_postcamera_black_u16_lut(
              std::numeric_limits<double>::quiet_NaN(), &sentinel) ==
              PostcameraBlackStatus::invalid_black &&
              all_u16(sentinel.values.data(), sentinel.values.size(), 0xa55au),
          "NaN mode1 black is rejected atomically");
}

void test_u16_buffer_execution() {
    constexpr double kBlack = 4100.0 / 65535.0;
    const U16LutPrepared table = reference_u16_lut(reference_lut(kBlack));
    const std::uint16_t src[12] = {
        0u, 1028u, 2049u, 32768u, 65535u, 0x1111u,
        4096u, 4100u, 8192u, 16384u, 32767u, 0x2222u};
    std::uint16_t dst[14];
    for (std::uint16_t& value : dst) value = 0xa55au;
    int status = acr_test_postcamera_black_u16(
        src, 12u, 6u, 2u, 5u, kBlack, dst, 14u, 7u);
    check(status == static_cast<int>(PostcameraBlackStatus::ok),
          "multi-row uint16 mode1 LUT execution succeeds");
    for (std::size_t y = 0u; y < 2u; ++y) {
        for (std::size_t x = 0u; x < 5u; ++x) {
            check(dst[y * 7u + x] == table.values[src[y * 6u + x]],
                  "uint16 output uses the full prepared table lookup");
        }
    }
    check(dst[5] == 0xa55au && dst[6] == 0xa55au && dst[12] == 0xa55au &&
              dst[13] == 0xa55au,
          "uint16 row padding and destination tail remain unchanged");

    std::uint16_t bypass_dst[14];
    for (std::uint16_t& value : bypass_dst) value = 0xa55au;
    status = acr_test_postcamera_black_u16(
        src, 12u, 6u, 2u, 5u, 0.0, bypass_dst, 14u, 7u);
    check(status == static_cast<int>(PostcameraBlackStatus::ok),
          "b=0 uint16 buffer takes the bypass path");
    for (std::size_t y = 0u; y < 2u; ++y) {
        for (std::size_t x = 0u; x < 5u; ++x) {
            check(bypass_dst[y * 7u + x] == src[y * 6u + x],
                  "b=0 uint16 bypass copies active values exactly");
        }
    }
    check(bypass_dst[5] == 0xa55au && bypass_dst[6] == 0xa55au &&
              bypass_dst[12] == 0xa55au && bypass_dst[13] == 0xa55au,
          "b=0 uint16 bypass preserves row padding and tail");

    std::uint16_t in_place[8] = {0u, 4096u, 0xbeefu, 32768u,
                                 65535u, 0xcafeu, 0x1234u, 0x5678u};
    const std::uint16_t in_place_before[8] = {
        0u, 4096u, 0xbeefu, 32768u, 65535u, 0xcafeu, 0x1234u, 0x5678u};
    status = acr_test_postcamera_black_u16(
        in_place, 8u, 4u, 2u, 3u, kBlack, in_place, 8u, 4u);
    check(status == static_cast<int>(PostcameraBlackStatus::ok) &&
              in_place[0] == table.values[in_place_before[0]] &&
              in_place[1] == table.values[in_place_before[1]] &&
              in_place[4] == table.values[in_place_before[4]] &&
              in_place[5] == table.values[in_place_before[5]] &&
              in_place[6] == table.values[in_place_before[6]] &&
              in_place[2] == table.values[in_place_before[2]] &&
              in_place[3] == in_place_before[3] &&
              in_place[7] == in_place_before[7],
          "exact uint16 in-place rows work and preserve padding");

    std::uint16_t invalid_dst[10];
    for (std::uint16_t& value : invalid_dst) value = 0xa55au;
    const std::uint16_t invalid_before[10] = {
        0xa55au, 0xa55au, 0xa55au, 0xa55au, 0xa55au,
        0xa55au, 0xa55au, 0xa55au, 0xa55au, 0xa55au};
    status = acr_test_postcamera_black_u16(
        src, 12u, 6u, 2u, 5u, 1.0, invalid_dst, 10u, 5u);
    check(status == static_cast<int>(PostcameraBlackStatus::invalid_black) &&
              std::memcmp(invalid_dst, invalid_before, sizeof(invalid_dst)) == 0,
          "invalid uint16 black leaves destination bytes unchanged");
    status = acr_test_postcamera_black_u16(
        src, 10u, 6u, 2u, 5u, kBlack, invalid_dst, 10u, 5u);
    check(status == static_cast<int>(PostcameraBlackStatus::buffer_too_small) &&
              std::memcmp(invalid_dst, invalid_before, sizeof(invalid_dst)) == 0,
          "short uint16 input capacity is rejected atomically");

    std::uint16_t alias[8] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
    const std::uint16_t alias_before[8] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
    status = acr_test_postcamera_black_u16(
        alias, 8u, 4u, 2u, 3u, kBlack, alias + 1u, 7u, 4u);
    check(status == static_cast<int>(PostcameraBlackStatus::overlap) &&
              std::memcmp(alias, alias_before, sizeof(alias)) == 0,
          "partial declared uint16 span overlap is rejected atomically");

    const std::uintptr_t max_address =
        std::numeric_limits<std::uintptr_t>::max();
    const auto* wrapping_source = reinterpret_cast<const std::uint16_t*>(
        max_address - 1u);
    status = acr_test_postcamera_black_u16(
        wrapping_source, 1u, 1u, 1u, 1u, kBlack, invalid_dst, 1u, 1u);
    check(status == static_cast<int>(PostcameraBlackStatus::arithmetic_overflow) &&
              std::memcmp(invalid_dst, invalid_before, sizeof(invalid_dst)) == 0,
          "uint16 uintptr span wrap is rejected before reads");

    constexpr std::size_t kOverLimit = 128u * 1024u * 1024u + 1u;
    status = acr_test_postcamera_black_u16(
        src, kOverLimit, kOverLimit, 1u, kOverLimit, kBlack,
        invalid_dst, kOverLimit, kOverLimit);
    check(status == static_cast<int>(PostcameraBlackStatus::resource_limit),
          "uint16 active sample resource cap is enforced");
}

}  // namespace

int main() {
    test_table_construction();
    test_lut_vs_direct_formula();
    test_buffer_execution();
    test_validation_and_atomicity();
    test_u16_table_construction();
    test_u16_buffer_execution();
    if (failures != 0) {
        std::fprintf(stderr, "%d post-camera black validation checks failed\n",
                     failures);
        return 1;
    }
    std::puts("post-camera black validation passed");
    return 0;
}
