#include "../src/acr_buffer_format.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>

extern "C" int acr_test_buffer_f32_to_u16(
    const float *, std::size_t, std::size_t, std::size_t, std::size_t,
    std::uint16_t *, std::size_t, std::size_t);
extern "C" int acr_test_buffer_u16_to_f32(
    const std::uint16_t *, std::size_t, std::size_t, std::size_t,
    std::size_t, float *, std::size_t, std::size_t);

namespace {

using imprint::acr::BufferFormatStatus;
using imprint::acr::copy_buffer_f32_to_u16_012eaeb8;
using imprint::acr::copy_buffer_u16_to_f32_012eac64;

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

std::uint16_t core3_style_ties_even(float value) {
    const float product = value * 65535.0f;
    const double exact = static_cast<double>(product);
    const double lower = std::floor(exact);
    auto integer = static_cast<std::uint32_t>(lower);
    const double fraction = exact - lower;
    if (fraction > 0.5 ||
        (fraction == 0.5 && (integer & 1u) != 0u)) {
        ++integer;
    }
    return static_cast<std::uint16_t>(integer);
}

void test_f32_to_u16_active_rows_and_fp_semantics() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::array<float, 14> source = {
        0.0f, 0.5f, 1.0f, 0.25f, nan, nan, nan,
        -1.0f, 2.0f, std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 0.75f, nan, nan};
    std::array<std::uint16_t, 16> output{};
    output.fill(0xA55Au);

    check(copy_buffer_f32_to_u16_012eaeb8(
              source.data(), source.size(), 7, 2, 5, output.data(),
              output.size(), 8) == BufferFormatStatus::ok,
          "float-to-u16 supports arbitrary width and independent row strides");
    const std::array<std::uint16_t, 5> first = {0, 32768, 65535, 16384, 0};
    const std::array<std::uint16_t, 5> second = {0, 65535, 65535, 0, 49151};
    for (std::size_t i = 0; i < first.size(); ++i) {
        check(output[i] == first[i], "first active row matches ACR values");
        check(output[8 + i] == second[i], "second active row matches ACR values");
    }
    for (const std::size_t padding : {5u, 6u, 7u, 13u, 14u, 15u}) {
        check(output[padding] == 0xA55Au,
              "float-to-u16 leaves destination row padding untouched");
    }

    const float half_tie = 0.5f;
    output.fill(0xA55Au);
    check(copy_buffer_f32_to_u16_012eaeb8(
              &half_tie, 1, 1, 1, 1, output.data(), output.size(), 1) ==
              BufferFormatStatus::ok,
          "single-sample float-to-u16 accepts natural alignment");
    check(output[0] == 32768u,
          "ACR fused operation maps 65535 * 0.5 + 0.5 to 32768");
    output.fill(0xA55Au);
    check(acr_test_buffer_f32_to_u16(
              &half_tie, 1, 1, 1, 1, output.data(), output.size(), 1) == 0 &&
              output[0] == 32768u,
          "C bridge exposes the float-to-u16 conversion");

    // This binary32 value makes the exact product 0.5. ACR's fused +0.5
    // followed by truncation yields 1; core3's ties-to-even path yields 0.
    const float ties_even_probe = float_from_bits(0x37000080u);
    output.fill(0xA55Au);
    check(copy_buffer_f32_to_u16_012eaeb8(
              &ties_even_probe, 1, 1, 1, 1, output.data(), output.size(), 1) ==
              BufferFormatStatus::ok,
          "ties-even differential probe converts successfully");
    check(output[0] == 1u && core3_style_ties_even(ties_even_probe) == 0u,
          "hardcoded probe distinguishes ACR from core3 ties-to-even");

    // This probe has different pre-conversion float bits for fused and split
    // arithmetic. Both truncate to zero, so it documents the operation-level
    // difference without claiming an observable uint16 difference.
    const float split_probe = float_from_bits(0x2bc000c0u);
    volatile float split_product = split_probe * 65535.0f;
    volatile float split_sum = split_product + 0.5f;
    const float fused_sum = std::fma(split_probe, 65535.0f, 0.5f);
    check(bits(static_cast<float>(split_sum)) == 0x3f000002u &&
              bits(fused_sum) == 0x3f000001u,
          "hardcoded probe distinguishes fused from split intermediate bits");
    output.fill(0xA55Au);
    check(copy_buffer_f32_to_u16_012eaeb8(
              &split_probe, 1, 1, 1, 1, output.data(), output.size(), 1) ==
              BufferFormatStatus::ok &&
              output[0] == static_cast<std::uint16_t>(fused_sum),
          "split arithmetic probe follows fused result through truncation");
}

void test_u16_to_f32_active_rows_and_reciprocal() {
    const std::array<std::uint16_t, 8> source = {
        0, 257, 65535, 1, 91, 17, 32768, 40000};
    std::array<float, 12> output{};
    output.fill(-17.5f);
    check(copy_buffer_u16_to_f32_012eac64(
              source.data(), source.size(), 4, 2, 3, output.data(),
              output.size(), 5) == BufferFormatStatus::ok,
          "u16-to-float supports arbitrary width and independent row strides");
    constexpr float reciprocal = 1.0f / 65535.0f;
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            const float expected = reciprocal *
                static_cast<float>(source[row * 4 + col]);
            check(bits(output[row * 5 + col]) == bits(expected),
                  "u16-to-float uses reciprocal followed by a separate multiply");
        }
    }
    check(bits(output[1]) == 0x3b808080u,
          "sample 257 uses reciprocal multiply result 0x3b808080");
    volatile float direct_division = static_cast<float>(source[1]) / 65535.0f;
    check(bits(output[1]) != bits(static_cast<float>(direct_division)),
          "sample 257 distinguishes reciprocal multiply from direct division");
    std::array<float, 2> bridge_output{};
    check(acr_test_buffer_u16_to_f32(
              source.data(), source.size(), 4, 1, 2, bridge_output.data(),
              bridge_output.size(), 2) == 0 &&
              bits(bridge_output[1]) == bits(output[1]),
          "C bridge exposes the u16-to-float conversion");
    for (const std::size_t padding : {3u, 4u, 8u, 9u, 10u, 11u}) {
        check(bits(output[padding]) == bits(-17.5f),
              "u16-to-float leaves destination row padding untouched");
    }
}

void test_rejections_are_atomic() {
    std::array<float, 12> floats{};
    floats.fill(0.5f);
    std::array<std::uint16_t, 12> integers{};
    integers.fill(0xBEEFu);
    const auto unchanged = [&]() { return integers ==
        std::array<std::uint16_t, 12>{0xBEEF, 0xBEEF, 0xBEEF, 0xBEEF,
                                      0xBEEF, 0xBEEF, 0xBEEF, 0xBEEF,
                                      0xBEEF, 0xBEEF, 0xBEEF, 0xBEEF}; };

    check(copy_buffer_f32_to_u16_012eaeb8(
              nullptr, 12, 4, 2, 3, integers.data(), integers.size(), 4) ==
              BufferFormatStatus::invalid_argument && unchanged(),
          "null source rejects without changing destination");
    check(copy_buffer_f32_to_u16_012eaeb8(
              floats.data(), floats.size(), 4, 0, 3, integers.data(),
              integers.size(), 4) == BufferFormatStatus::invalid_dimensions &&
              unchanged(),
          "zero height rejects without changing destination");
    check(copy_buffer_f32_to_u16_012eaeb8(
              floats.data(), floats.size(), 2, 2, 3, integers.data(),
              integers.size(), 4) == BufferFormatStatus::invalid_stride &&
              unchanged(),
          "short source stride rejects without changing destination");
    check(copy_buffer_f32_to_u16_012eaeb8(
              floats.data(), 6, 4, 2, 3, integers.data(), integers.size(), 4) ==
              BufferFormatStatus::buffer_too_small && unchanged(),
          "short source capacity rejects without changing destination");

    const auto over_limit = 128u * 1024u * 1024u + 1u;
    check(copy_buffer_f32_to_u16_012eaeb8(
              floats.data(), floats.size(), over_limit, 1, over_limit,
              integers.data(), integers.size(), over_limit) ==
              BufferFormatStatus::resource_limit && unchanged(),
          "active sample resource limit rejects before writes");

    alignas(16) std::array<std::byte, 128> storage{};
    const auto *misaligned = reinterpret_cast<const float *>(storage.data() + 1);
    check(copy_buffer_f32_to_u16_012eaeb8(
              misaligned, 1, 1, 1, 1, integers.data(), integers.size(), 1) ==
              BufferFormatStatus::unsupported_alignment && unchanged(),
          "unaligned source rejects without changing destination");

    storage.fill(std::byte{0x5A});
    const auto storage_before_unaligned = storage;
    auto *misaligned_destination =
        reinterpret_cast<std::uint16_t *>(storage.data() + 1);
    check(copy_buffer_f32_to_u16_012eaeb8(
              floats.data(), floats.size(), 1, 1, 1, misaligned_destination, 1,
              1) == BufferFormatStatus::unsupported_alignment &&
              storage == storage_before_unaligned,
          "unaligned destination rejects without changing destination bytes");

    storage.fill(std::byte{0x5A});
    const auto storage_before_overlap = storage;
    const auto *alias_source = reinterpret_cast<const float *>(storage.data());
    auto *alias_destination =
        reinterpret_cast<std::uint16_t *>(storage.data() + 8);
    check(copy_buffer_f32_to_u16_012eaeb8(
              alias_source, 16, 1, 1, 1, alias_destination, 1, 1) ==
              BufferFormatStatus::overlap && storage == storage_before_overlap,
          "overlapping declared spans reject before access or writes");

    const auto huge_count = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max() / sizeof(float)) + 1u;
    check(copy_buffer_f32_to_u16_012eaeb8(
              floats.data(), huge_count, 1, 1, 1, integers.data(),
              integers.size(), 1) == BufferFormatStatus::arithmetic_overflow &&
              unchanged(),
          "declared source span beyond PTRDIFF_MAX rejects atomically");

    const auto near_limit = std::numeric_limits<std::uintptr_t>::max() &
                            ~static_cast<std::uintptr_t>(alignof(float) - 1u);
    const auto *near_address_end = reinterpret_cast<const float *>(near_limit);
    check(copy_buffer_f32_to_u16_012eaeb8(
              near_address_end, 1, 1, 1, 1, integers.data(), integers.size(), 1) ==
              BufferFormatStatus::arithmetic_overflow && unchanged(),
          "uintptr source span overflow rejects atomically");
}

}  // namespace

int main() {
    static_assert(static_cast<int>(BufferFormatStatus::ok) == 0);
    static_assert(static_cast<int>(BufferFormatStatus::length_error) == 10);
    test_f32_to_u16_active_rows_and_fp_semantics();
    test_u16_to_f32_active_rows_and_reciprocal();
    test_rejections_are_atomic();

    if (failures != 0) {
        std::fprintf(stderr, "%d validation check(s) failed\n", failures);
        return 1;
    }
    std::puts("ACR buffer format validation passed");
    return 0;
}
