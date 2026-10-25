#include "../src/acr_stage3_core3_format.hpp"

#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

using imprint::acr::Core3FormatStatus;
using imprint::acr::core3_f32_to_u16_prepared_row_008103dc;
using imprint::acr::core3_u16_to_f32_prepared_row_008096e4;

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float float_from_bits(std::uint32_t value) {
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void test_exhaustive_u16_domain() {
    constexpr std::size_t count = 65536;
    alignas(16) static std::array<std::uint16_t, count> input{};
    alignas(16) static std::array<float, count> prepared{};
    alignas(16) static std::array<std::uint16_t, count> roundtrip{};

    for (std::size_t i = 0; i < count; ++i) {
        input[i] = static_cast<std::uint16_t>(i);
    }
    const auto input_before = input;
    check(core3_u16_to_f32_prepared_row_008096e4(
              input.data(), count, prepared.data(), count, count) ==
              Core3FormatStatus::ok,
          "forward conversion accepts all 65536 uint16 values");

    const float reciprocal = float_from_bits(0x37800080u);
    for (std::size_t i = 0; i < count; ++i) {
        const float expected = static_cast<float>(
            static_cast<double>(input[i]) * static_cast<double>(reciprocal));
        if (bits(prepared[i]) != bits(expected)) {
            check(false, "forward conversion matches reciprocal multiply oracle");
            break;
        }
    }
    check(bits(prepared[257]) == 0x3b808080u,
          "sample 257 uses the reciprocal multiply result 0x3b808080");
    volatile float direct_division =
        static_cast<float>(input[257]) / 65535.0f;
    check(bits(direct_division) == 0x3b808081u,
          "sample 257 distinguishes direct division by one ULP");
    check(core3_f32_to_u16_prepared_row_008103dc(
              prepared.data(), count, roundtrip.data(), count, count) ==
              Core3FormatStatus::ok,
          "reverse conversion accepts the complete prepared uint16 row");
    check(roundtrip == input,
          "the normalized roundtrip preserves every uint16 code value");
    check(input == input_before, "forward conversion leaves input unchanged");
}

void test_padding_and_nearest_even() {
    alignas(16) std::array<std::uint16_t, 12> integers{};
    alignas(16) std::array<float, 12> prepared{};
    alignas(16) std::array<float, 12> tie_values{};
    alignas(16) std::array<std::uint16_t, 12> output{};
    integers.fill(91u);
    prepared.fill(-27.5f);
    tie_values.fill(0.25f);
    tie_values[0] = 0.0f;
    tie_values[1] = 0.5f;
    tie_values[2] = 1.0f;
    tie_values[3] = 0.25f;
    output.fill(0xA55Au);
    const auto tie_values_before = tie_values;

    check(core3_u16_to_f32_prepared_row_008096e4(
              integers.data(), integers.size(), prepared.data(),
              prepared.size(), 8) == Core3FormatStatus::ok,
          "forward conversion accepts an aligned width-four-multiple row");
    for (std::size_t i = 8; i < prepared.size(); ++i) {
        check(bits(prepared[i]) == bits(-27.5f),
              "forward conversion leaves destination padding unchanged");
    }

    check(core3_f32_to_u16_prepared_row_008103dc(
              tie_values.data(), tie_values.size(), output.data(),
              output.size(), 8) == Core3FormatStatus::ok,
          "reverse conversion accepts finite normalized values");
    check(output[0] == 0u && output[1] == 32768u && output[2] == 65535u &&
              output[3] == 16384u,
          "zero, half-tie, one, and quarter convert as expected");
    check(tie_values == tie_values_before,
          "reverse conversion leaves its float input unchanged");
    for (std::size_t i = 8; i < output.size(); ++i) {
        check(output[i] == 0xA55Au,
              "reverse conversion leaves destination padding unchanged");
    }

    const int original_rounding = std::fegetround();
    const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    for (const int mode : modes) {
        if (std::fesetround(mode) != 0) continue;
        output.fill(0xA55Au);
        check(core3_f32_to_u16_prepared_row_008103dc(
                  tie_values.data(), tie_values.size(), output.data(),
                  output.size(), 4) == Core3FormatStatus::ok,
              "half-tie conversion succeeds under each available rounding mode");
        check(output[1] == 32768u,
              "integer conversion rounds a half tie to even independent of fenv");
    }
    if (original_rounding != -1) std::fesetround(original_rounding);
}

void test_rejected_calls_do_not_write() {
    alignas(16) std::array<std::uint16_t, 8> integer_input{};
    alignas(16) std::array<float, 8> float_input{};
    alignas(16) std::array<float, 8> float_output{};
    alignas(16) std::array<std::uint16_t, 8> integer_output{};

    const auto expect_domain_rejection = [&](float invalid,
                                             const char *label) {
        float_input.fill(0.25f);
        float_input[7] = invalid;
        integer_output.fill(0xBEEFu);
        const auto before = integer_output;
        check(core3_f32_to_u16_prepared_row_008103dc(
                  float_input.data(), float_input.size(), integer_output.data(),
                  integer_output.size(), float_input.size()) ==
                  Core3FormatStatus::unsupported_domain,
              label);
        check(integer_output == before,
              "late unsupported sample rejects the entire row without writes");
    };
    expect_domain_rejection(std::numeric_limits<float>::quiet_NaN(),
                            "NaN is outside the reverse conversion domain");
    expect_domain_rejection(std::numeric_limits<float>::infinity(),
                            "positive infinity is outside the reverse domain");
    expect_domain_rejection(-std::numeric_limits<float>::infinity(),
                            "negative infinity is outside the reverse domain");
    expect_domain_rejection(-0.001f,
                            "negative finite value is outside the reverse domain");
    expect_domain_rejection(1.001f,
                            "value above one is outside the reverse domain");

    float_input.fill(0.5f);
    float_output.fill(-17.0f);
    const auto float_before = float_output;
    check(core3_u16_to_f32_prepared_row_008096e4(
              integer_input.data(), integer_input.size(), float_output.data(),
              float_output.size(), 6) == Core3FormatStatus::invalid_dimensions,
          "width not divisible by four is rejected");
    check(float_output == float_before,
          "invalid dimensions leave forward output unchanged");
    check(core3_u16_to_f32_prepared_row_008096e4(
              integer_input.data(), integer_input.size(), float_output.data(),
              float_output.size(), 0) == Core3FormatStatus::invalid_dimensions,
          "zero width is rejected");
    check(core3_u16_to_f32_prepared_row_008096e4(
              integer_input.data(), integer_input.size(), float_output.data(),
              float_output.size(), 8) == Core3FormatStatus::ok,
          "fixture row is valid before capacity checks");

    float_output.fill(-17.0f);
    const auto capacity_before = float_output;
    check(core3_u16_to_f32_prepared_row_008096e4(
              integer_input.data(), 7, float_output.data(),
              float_output.size(), 8) == Core3FormatStatus::insufficient_capacity,
          "short source capacity is rejected");
    check(float_output == capacity_before,
          "short source capacity leaves output unchanged");
    check(core3_u16_to_f32_prepared_row_008096e4(
              integer_input.data(), integer_input.size(), float_output.data(),
              7, 8) == Core3FormatStatus::insufficient_capacity,
          "short destination capacity is rejected");
    check(float_output == capacity_before,
          "short destination capacity leaves output unchanged");

    alignas(16) std::array<std::byte, 64> storage{};
    const auto *misaligned_source = reinterpret_cast<const std::uint16_t *>(
        storage.data() + sizeof(std::uint16_t));
    float_output.fill(-17.0f);
    const auto alignment_before = float_output;
    check(core3_u16_to_f32_prepared_row_008096e4(
              misaligned_source, 8, float_output.data(), float_output.size(),
              8) == Core3FormatStatus::unsupported_alignment,
          "non-16-byte-aligned source is rejected");
    check(float_output == alignment_before,
          "alignment rejection leaves output unchanged");

    storage.fill(std::byte{0x5A});
    const auto storage_before = storage;
    auto *misaligned_destination =
        reinterpret_cast<float *>(storage.data() + sizeof(float));
    check(core3_u16_to_f32_prepared_row_008096e4(
              integer_input.data(), integer_input.size(), misaligned_destination,
              8, 8) == Core3FormatStatus::unsupported_alignment,
          "non-16-byte-aligned destination is rejected");
    check(storage == storage_before,
          "destination alignment rejection leaves storage unchanged");

    check(core3_u16_to_f32_prepared_row_008096e4(
              nullptr, integer_input.size(), float_output.data(),
              float_output.size(), 8) == Core3FormatStatus::invalid_argument,
          "null source is rejected");

    auto *overlapping_float = reinterpret_cast<float *>(storage.data());
    const auto *overlapping_integer =
        reinterpret_cast<const std::uint16_t *>(storage.data());
    check(core3_u16_to_f32_prepared_row_008096e4(
              overlapping_integer, 8, overlapping_float, 8, 8) ==
              Core3FormatStatus::overlapping_buffers,
          "overlapping width spans are rejected before access");

    const auto too_wide_base = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max() / sizeof(float));
    std::size_t too_wide = (too_wide_base + 4u) & ~std::size_t{3u};
    if (too_wide <= too_wide_base) too_wide += 4u;
    check(core3_f32_to_u16_prepared_row_008103dc(
              float_input.data(), too_wide, integer_output.data(), too_wide,
              too_wide) == Core3FormatStatus::arithmetic_overflow,
          "byte span beyond PTRDIFF_MAX is rejected before access");

    const auto near_address_limit =
        std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{0xFu};
    const auto *near_limit_input =
        reinterpret_cast<const float *>(near_address_limit);
    integer_output.fill(0xBEEFu);
    const auto pointer_overflow_before = integer_output;
    check(core3_f32_to_u16_prepared_row_008103dc(
              near_limit_input, 4, integer_output.data(), integer_output.size(),
              4) == Core3FormatStatus::arithmetic_overflow,
          "uintptr span overflow is rejected before pointer arithmetic");
    check(integer_output == pointer_overflow_before,
          "pointer-span overflow leaves output unchanged");
}

}  // namespace

int main() {
    const int original_rounding = std::fegetround();
    if (original_rounding != -1) std::fesetround(FE_TONEAREST);
    test_exhaustive_u16_domain();
    test_padding_and_nearest_even();
    test_rejected_calls_do_not_write();
    if (original_rounding != -1) std::fesetround(original_rounding);

    if (failures != 0) {
        std::fprintf(stderr, "%d validation check(s) failed\n", failures);
        return 1;
    }
    std::puts("core3 format validation passed");
    return 0;
}
