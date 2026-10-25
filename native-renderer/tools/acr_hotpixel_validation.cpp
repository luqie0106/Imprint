#include "acr_hotpixel.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

extern "C" int acr_test_hotpixel(
    const std::uint16_t* src, std::size_t count, std::size_t stride,
    std::size_t width, std::uint32_t phase, std::uint16_t hot_threshold,
    std::uint16_t cold_threshold, std::uint16_t* dst,
    std::size_t dst_count);

using imprint::acr::HotPixelStatus;
using imprint::acr::hotpixel_u16_row_0132d7c8;

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
        const HotPixelStatus status_value = (expression);                              \
        if (status_value != (expected)) {                                              \
            std::fprintf(stderr, "FAIL %s:%d: status %d != %d for %s\n", __FILE__,    \
                         __LINE__, static_cast<int>(status_value),                    \
                         static_cast<int>(expected), #expression);                    \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

struct Rows {
    static constexpr std::size_t stride = 8;
    std::array<std::uint16_t, 5 * stride> samples{};
    const std::uint16_t* row_pointers[5]{};
    std::size_t row_counts[5]{};

    Rows() {
        for (std::size_t row = 0; row < 5; ++row) {
            row_pointers[row] = samples.data() + row * stride;
            row_counts[row] = stride;
        }
    }

    std::uint16_t& at(std::size_t row, std::size_t column) {
        return samples[row * stride + column];
    }

    HotPixelStatus run(std::uint16_t* dst, std::size_t dst_count,
                       std::size_t width = 1, std::uint32_t phase = 0,
                       std::uint16_t hot = 100, std::uint16_t cold = 100) {
        return hotpixel_u16_row_0132d7c8(row_pointers, row_counts, 2, dst,
                                        dst_count, width, phase, hot, cold);
    }
};

void test_phase_zero_hot_fixed_counterexample() {
    Rows input;
    input.at(2, 2) = 65535;
    input.at(0, 2) = 47923;
    input.at(4, 2) = 9067;
    input.at(2, 0) = 54993;
    input.at(2, 4) = 18027;

    std::uint16_t output[2] = {0xaaaa, 0xbbbb};
    CHECK_STATUS(input.run(output, 2, 1, 0, 1000, 100), HotPixelStatus::ok);
    CHECK(output[0] == 54993);  // Recovered hot replacement is max of four.
    CHECK(output[1] == 0xbbbb); // Output padding is untouched.
}

void test_phase_one_hot_and_cold() {
    Rows hot_input;
    hot_input.at(2, 2) = 65535;
    hot_input.at(1, 1) = 50000;
    hot_input.at(1, 3) = 40000;
    hot_input.at(3, 1) = 30000;
    hot_input.at(3, 3) = 20000;
    std::uint16_t hot_output = 0;
    CHECK_STATUS(hot_input.run(&hot_output, 1, 1, 1, 1000, 100), HotPixelStatus::ok);
    CHECK(hot_output == 50000);

    Rows cold_input;
    cold_input.samples.fill(1000);
    cold_input.at(2, 2) = 0;
    cold_input.at(0, 2) = 500;
    cold_input.at(4, 2) = 400;
    cold_input.at(2, 0) = 300;
    cold_input.at(2, 4) = 200;
    std::uint16_t cold_output = 0xffff;
    CHECK_STATUS(cold_input.run(&cold_output, 1, 1, 0, 65535, 100),
                 HotPixelStatus::ok);
    CHECK(cold_output == 200);
}

void test_phase_toggles_by_column() {
    Rows input;
    input.at(2, 3) = 65535;
    input.at(1, 2) = 50000;
    input.at(1, 4) = 40000;
    input.at(3, 2) = 30000;
    input.at(3, 4) = 20000;
    std::uint16_t output[3] = {0xffff, 0xffff, 0xcafe};
    CHECK_STATUS(input.run(output, 3, 2, 0, 1000, 100), HotPixelStatus::ok);
    CHECK(output[1] == 50000); // Column 1 uses phase 0 XOR 1.
    CHECK(output[2] == 0xcafe);
}

void test_threshold_equalities_and_no_change() {
    Rows equal_hot;
    equal_hot.samples.fill(0);
    equal_hot.at(2, 2) = 200;
    std::uint16_t output = 0xffff;
    CHECK_STATUS(equal_hot.run(&output, 1, 1, 0, 200, 65535), HotPixelStatus::ok);
    CHECK(output == 200); // C == hot threshold is not hot.

    Rows equal_hot_neighbor;
    equal_hot_neighbor.at(2, 2) = 1000;
    equal_hot_neighbor.at(0, 2) = 900; // C - hot_threshold, strict comparison.
    output = 0xffff;
    CHECK_STATUS(equal_hot_neighbor.run(&output, 1, 1, 0, 100, 65535),
                 HotPixelStatus::ok);
    CHECK(output == 1000);

    Rows equal_cold_neighbor;
    equal_cold_neighbor.samples.fill(2000);
    equal_cold_neighbor.at(2, 2) = 1000;
    equal_cold_neighbor.at(0, 2) = 1100; // C + cold_threshold, strict comparison.
    output = 0xffff;
    CHECK_STATUS(equal_cold_neighbor.run(&output, 1, 1, 0, 65535, 100),
                 HotPixelStatus::ok);
    CHECK(output == 1000);

    Rows unchanged;
    unchanged.samples.fill(4321);
    std::uint16_t padded[3] = {0, 0, 0xface};
    CHECK_STATUS(unchanged.run(padded, 3, 2, 0), HotPixelStatus::ok);
    CHECK(padded[0] == 4321 && padded[1] == 4321 && padded[2] == 0xface);
}

void test_repeated_rows_are_allowed() {
    std::array<std::uint16_t, 8> row{};
    row.fill(3210);
    const std::uint16_t* rows[5] = {row.data(), row.data(), row.data(),
                                    row.data(), row.data()};
    const std::size_t counts[5] = {row.size(), row.size(), row.size(),
                                   row.size(), row.size()};
    std::uint16_t output = 0;
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(rows, counts, 2, &output, 1,
                                           1, 0, 100, 100), HotPixelStatus::ok);
    CHECK(output == 3210);
}

void test_invalid_inputs_are_atomic() {
    Rows input;
    std::uint16_t output[2] = {0x1234, 0x5678};
    const auto unchanged = std::array<std::uint16_t, 2>{output[0], output[1]};
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(nullptr, input.row_counts, 2, output,
                                           2, 1, 0, 100, 100),
                 HotPixelStatus::invalid_argument);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(input.row_pointers, nullptr, 2,
                                           output, 2, 1, 0, 100, 100),
                 HotPixelStatus::invalid_argument);
    CHECK_STATUS(input.run(output, 2, 1, 2), HotPixelStatus::invalid_parameters);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);
    CHECK_STATUS(input.run(output, 2, 0), HotPixelStatus::invalid_parameters);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(input.row_pointers, input.row_counts,
                                           1, output, 2, 1, 0, 100, 100),
                 HotPixelStatus::invalid_parameters);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);
    CHECK_STATUS(input.run(output, 0, 1), HotPixelStatus::invalid_parameters);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);

    std::size_t short_counts[5] = {4, 8, 8, 8, 8};
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(input.row_pointers, short_counts, 2,
                                           output, 2, 1, 0, 100, 100),
                 HotPixelStatus::invalid_parameters);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);

    const auto max_address = std::numeric_limits<std::uintptr_t>::max();
    auto* wrapping_sample = reinterpret_cast<const std::uint16_t*>(max_address - 1u);
    const std::uint16_t* wrapping_rows[5] = {
        wrapping_sample, input.row_pointers[1], input.row_pointers[2],
        input.row_pointers[3], input.row_pointers[4]};
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(wrapping_rows, input.row_counts, 2,
                                           output, 2, 1, 0, 100, 100),
                 HotPixelStatus::invalid_parameters);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);

    auto* wrapping_dst = reinterpret_cast<std::uint16_t*>(max_address - 1u);
    CHECK_STATUS(input.run(wrapping_dst, 1), HotPixelStatus::invalid_parameters);

    const auto wrapping_metadata_address = max_address -
        (max_address % alignof(const std::uint16_t*));
    auto** wrapping_rows_array = reinterpret_cast<const std::uint16_t**>(
        wrapping_metadata_address);
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(wrapping_rows_array,
                                           input.row_counts, 2, output, 2,
                                           1, 0, 100, 100),
                 HotPixelStatus::invalid_parameters);
    const auto wrapping_count_address = max_address -
        (max_address % alignof(std::size_t));
    auto* wrapping_count_array = reinterpret_cast<const std::size_t*>(
        wrapping_count_address);
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(input.row_pointers,
                                           wrapping_count_array, 2, output, 2,
                                           1, 0, 100, 100),
                 HotPixelStatus::invalid_parameters);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);

    const auto max_size = std::numeric_limits<std::size_t>::max();
    std::size_t huge_counts[5] = {max_size, 8, 8, 8, 8};
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(input.row_pointers, huge_counts, 2,
                                           output, 2, 1, 0, 100, 100),
                 HotPixelStatus::invalid_parameters);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);
    std::size_t ptrdiff_counts[5] = {
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
                sizeof(std::uint16_t) + 1,
        8, 8, 8, 8};
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(input.row_pointers, ptrdiff_counts,
                                           2, output, 2, 1, 0, 100, 100),
                 HotPixelStatus::invalid_parameters);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);

    alignas(const std::uint16_t*) unsigned char pointer_array_bytes[
        sizeof(input.row_pointers) + 1] = {};
    auto** misaligned_rows = reinterpret_cast<const std::uint16_t**>(
        pointer_array_bytes + 1);
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(misaligned_rows, input.row_counts,
                                           2, output, 2, 1, 0, 100, 100),
                 HotPixelStatus::invalid_argument);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);

    alignas(std::size_t) unsigned char count_array_bytes[
        sizeof(input.row_counts) + 1] = {};
    auto* misaligned_counts = reinterpret_cast<std::size_t*>(count_array_bytes + 1);
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(input.row_pointers, misaligned_counts,
                                           2, output, 2, 1, 0, 100, 100),
                 HotPixelStatus::invalid_argument);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);

    const std::uint16_t* misaligned_input_rows[5] = {
        reinterpret_cast<const std::uint16_t*>(
            reinterpret_cast<const unsigned char*>(input.row_pointers[0]) + 1),
        input.row_pointers[1], input.row_pointers[2], input.row_pointers[3],
        input.row_pointers[4]};
    CHECK_STATUS(hotpixel_u16_row_0132d7c8(misaligned_input_rows,
                                           input.row_counts, 2, output, 2,
                                           1, 0, 100, 100),
                 HotPixelStatus::invalid_argument);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);

    auto* misaligned_dst = reinterpret_cast<std::uint16_t*>(
        reinterpret_cast<unsigned char*>(output) + 1);
    CHECK_STATUS(input.run(misaligned_dst, 1), HotPixelStatus::invalid_argument);
    CHECK(output[0] == unchanged[0] && output[1] == unchanged[1]);
}

void test_overlap_is_atomic() {
    Rows input;
    auto* aliased_destination = input.samples.data() + 1;
    const auto original = input.samples;
    CHECK_STATUS(input.run(aliased_destination, 1), HotPixelStatus::overlap);
    CHECK(input.samples == original);

    Rows metadata_input;
    auto* metadata_destination =
        reinterpret_cast<std::uint16_t*>(metadata_input.row_pointers);
    const auto metadata_original = metadata_input.samples;
    CHECK_STATUS(metadata_input.run(metadata_destination, 1), HotPixelStatus::overlap);
    CHECK(metadata_input.samples == metadata_original);

    Rows count_metadata_input;
    auto* count_destination =
        reinterpret_cast<std::uint16_t*>(count_metadata_input.row_counts);
    const auto counts_original = count_metadata_input.samples;
    CHECK_STATUS(count_metadata_input.run(count_destination, 1),
                 HotPixelStatus::overlap);
    CHECK(count_metadata_input.samples == counts_original);
}

void test_bridge_contract() {
    constexpr std::size_t width = 1;
    constexpr std::size_t stride = 8;
    constexpr std::size_t required = 4 * stride + width + 4;
    std::array<std::uint16_t, 5 * stride> source{};
    source.fill(5000);
    std::uint16_t output[2] = {0, 0xabcd};
    CHECK(acr_test_hotpixel(source.data(), source.size(), stride, width, 0,
                            100, 100, output, 2) == 0);
    CHECK(output[0] == 5000 && output[1] == 0xabcd);

    output[0] = 0x1234;
    CHECK(acr_test_hotpixel(source.data(), required - 1, stride, width, 0,
                            100, 100, output, 2) ==
          static_cast<int>(HotPixelStatus::invalid_parameters));
    CHECK(output[0] == 0x1234);
    CHECK(acr_test_hotpixel(source.data(), source.size(), width + 3, width, 0,
                            100, 100, output, 2) ==
          static_cast<int>(HotPixelStatus::invalid_parameters));
    CHECK(output[0] == 0x1234);

    const auto source_original = source;
    CHECK(acr_test_hotpixel(source.data(), source.size(), stride, width, 0,
                            100, 100, source.data() + 1, 1) ==
          static_cast<int>(HotPixelStatus::overlap));
    CHECK(source == source_original);

    const auto max_size = std::numeric_limits<std::size_t>::max();
    CHECK(acr_test_hotpixel(source.data(), source.size(), max_size / 2,
                            width, 0, 100, 100, output, 2) ==
          static_cast<int>(HotPixelStatus::invalid_parameters));
    CHECK(output[0] == 0x1234);
    CHECK(acr_test_hotpixel(source.data(), max_size, stride, width, 0,
                            100, 100, output, 2) ==
          static_cast<int>(HotPixelStatus::invalid_parameters));
    CHECK(output[0] == 0x1234);
    const std::size_t ptrdiff_count =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
            sizeof(std::uint16_t) + 1;
    CHECK(acr_test_hotpixel(source.data(), ptrdiff_count, stride, width, 0,
                            100, 100, output, 2) ==
          static_cast<int>(HotPixelStatus::invalid_parameters));
    CHECK(output[0] == 0x1234);
}

}  // namespace

int main() {
    test_phase_zero_hot_fixed_counterexample();
    test_phase_one_hot_and_cold();
    test_phase_toggles_by_column();
    test_threshold_equalities_and_no_change();
    test_repeated_rows_are_allowed();
    test_invalid_inputs_are_atomic();
    test_overlap_is_atomic();
    test_bridge_contract();
    if (failures != 0) {
        std::fprintf(stderr, "%d hotpixel validation checks failed\n", failures);
        return 1;
    }
    std::puts("hotpixel validation passed");
    return 0;
}
