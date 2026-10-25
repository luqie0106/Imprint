#include "../src/acr_stage3_balance3.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using imprint::acr::Balance3KernelStatus;
using imprint::acr::balance3_diagonal_filter_prepared_row_008139cc;
using imprint::acr::balance3_diagonal_subtract_prepared_row_00813cf8;
using imprint::acr::balance3_difference_prepared_row_00813870;

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

bool same_bits(float a, float b) { return bits(a) == bits(b); }

struct AlignedFloats {
    std::vector<float> storage;
    float *data = nullptr;

    explicit AlignedFloats(std::size_t count, float fill = 0.0f)
        : storage(count + 3u, fill) {
        const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
        const std::size_t misalignment =
            static_cast<std::size_t>(address & 0xFu);
        const std::size_t offset = ((16u - misalignment) & 0xFu) / sizeof(float);
        data = storage.data() + offset;
    }
};

struct RowFixture {
    std::size_t width;
    std::size_t stride;
    AlignedFloats memory;
    std::size_t center_index;
    float *center;

    RowFixture(std::size_t row_width, std::size_t row_stride,
               std::size_t backward, std::size_t forward)
        : width(row_width), stride(row_stride),
          memory(128u + backward + forward + width + 128u, -777.0f),
          center_index(0), center(nullptr) {
        const std::size_t head = 64u + ((4u - (backward & 3u)) & 3u);
        center_index = head + backward;
        center = memory.data + center_index;
        for (std::size_t i = 0; i < memory.storage.size() - 3u; ++i) {
            memory.data[i] = static_cast<float>(static_cast<int>((i * 37u) % 1009u) - 500) *
                             0.125f;
        }
    }
};

float round_f32(float value) {
    volatile float rounded = value;
    return rounded;
}

std::vector<float> reference_difference(const float *center,
                                        std::size_t stride,
                                        std::size_t width, int reverse) {
    std::vector<float> result(width);
    for (std::size_t j = 0; j < width; ++j) {
        float sum = round_f32(
            center[-static_cast<std::ptrdiff_t>(stride) +
                   static_cast<std::ptrdiff_t>(j) - 1] +
            center[-static_cast<std::ptrdiff_t>(stride) +
                   static_cast<std::ptrdiff_t>(j) + 1]);
        sum = round_f32(
            sum + center[static_cast<std::ptrdiff_t>(stride) +
                         static_cast<std::ptrdiff_t>(j) - 1]);
        sum = round_f32(
            sum + center[static_cast<std::ptrdiff_t>(stride) +
                         static_cast<std::ptrdiff_t>(j) + 1]);
        const float avg = round_f32(sum * 0.25f);
        result[j] = reverse == 0 ? round_f32(center[j] - avg)
                                 : round_f32(avg - center[j]);
    }
    return result;
}

std::vector<float> reference_diagonal(const float *center,
                                      std::size_t width,
                                      std::size_t distance,
                                      const float *coeff,
                                      std::size_t radius,
                                      const float *initial,
                                      bool subtract) {
    std::vector<float> result(width);
    for (std::size_t j = 0; j < width; ++j) {
        float acc = round_f32(coeff[0] * center[j]);
        for (std::size_t i = 1; i <= radius; ++i) {
            const std::size_t offset = i * distance;
            const float pair = round_f32(
                (center - static_cast<std::ptrdiff_t>(offset))[j] +
                (center + static_cast<std::ptrdiff_t>(offset))[j]);
            acc = std::fma(coeff[i], pair, acc);
        }
        result[j] = subtract ? round_f32(initial[j] - acc) : acc;
    }
    return result;
}

void check_row_bits(const float *actual, const std::vector<float> &expected,
                    const char *message) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!same_bits(actual[i], expected[i])) {
            check(false, message);
            return;
        }
    }
}

void test_difference_asymmetric() {
    constexpr std::size_t width = 4u;
    constexpr std::size_t stride = 8u;
    RowFixture fixture(width, stride, stride + 1u, stride + width);
    float *north = fixture.center - static_cast<std::ptrdiff_t>(stride);
    float *south = fixture.center + static_cast<std::ptrdiff_t>(stride);
    const float north_values[] = { -20.0f, 1.0f, 2.0f, 8.0f, 15.0f, 100.0f };
    const float south_values[] = { 300.0f, 4.0f, 16.0f, 32.0f, 64.0f, -999.0f };
    for (std::size_t i = 0; i < 6u; ++i) {
        north[static_cast<std::ptrdiff_t>(i) - 1] = north_values[i];
        south[static_cast<std::ptrdiff_t>(i) - 1] = south_values[i];
    }
    const float center_values[] = { 10.0f, 20.0f, 30.0f, 40.0f };
    for (std::size_t i = 0; i < width; ++i) fixture.center[i] = center_values[i];
    const auto source_before = fixture.memory.storage;

    AlignedFloats output(width + 4u, 0x1.23456p+8f);
    auto status = balance3_difference_prepared_row_00813870(
        fixture.center, stride, width, 0, output.data);
    check(status == Balance3KernelStatus::ok,
          "asymmetric difference forward succeeds");
    const float expected_forward[] = { -64.5f, 8.75f, 5.75f, 254.75f };
    check_row_bits(output.data,
                   reference_difference(fixture.center, stride, width, 0),
                   "difference forward matches binary32 oracle");
    for (std::size_t i = 0; i < width; ++i) {
        check(same_bits(output.data[i], expected_forward[i]),
              "asymmetric difference hard-coded forward values");
    }
    for (std::size_t i = width; i < width + 4u; ++i) {
        check(same_bits(output.data[i], 0x1.23456p+8f),
              "difference leaves destination padding unchanged");
    }
    check(fixture.memory.storage == source_before,
          "difference leaves source and guard storage unchanged");

    status = balance3_difference_prepared_row_00813870(
        fixture.center, stride, width, 1, output.data);
    check(status == Balance3KernelStatus::ok,
          "asymmetric difference reverse succeeds");
    const float expected_reverse[] = { 64.5f, -8.75f, -5.75f, -254.75f };
    check_row_bits(output.data,
                   reference_difference(fixture.center, stride, width, 1),
                   "difference reverse matches binary32 oracle");
    for (std::size_t i = 0; i < width; ++i) {
        check(same_bits(output.data[i], expected_reverse[i]),
              "asymmetric difference hard-coded reverse values");
    }
}

void test_diagonal_directions_hard_coded() {
    constexpr std::size_t width = 4u;
    constexpr std::size_t stride = 8u;
    constexpr std::size_t radius = 1u;
    constexpr std::size_t plus_distance = stride + 1u;
    RowFixture fixture(width, stride, plus_distance, plus_distance + width - 1u);
    const float centers[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    for (std::size_t j = 0; j < width; ++j) fixture.center[j] = centers[j];
    float *north = fixture.center - static_cast<std::ptrdiff_t>(stride);
    float *south = fixture.center + static_cast<std::ptrdiff_t>(stride);
    const float north_values[] = { 11.0f, 13.0f, 19.0f, 23.0f, 31.0f, 37.0f };
    const float south_values[] = { 97.0f, 107.0f, 109.0f, 113.0f, 127.0f, 131.0f };
    for (std::size_t i = 0; i < 6u; ++i) {
        north[static_cast<std::ptrdiff_t>(i) - 1] = north_values[i];
        south[static_cast<std::ptrdiff_t>(i) - 1] = south_values[i];
    }
    const float coeff[] = { 0.5f, 0.25f };
    const auto source_before = fixture.memory.storage;

    AlignedFloats output(width + 4u, -88.0f);
    auto status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, stride, width, coeff, radius, output.data);
    check(status == Balance3KernelStatus::ok,
          "asymmetric plus diagonal succeeds");
    const float expected_plus[] = { 30.5f, 32.5f, 38.0f, 40.5f };
    for (std::size_t i = 0; i < width; ++i) {
        check(same_bits(output.data[i], expected_plus[i]),
              "plus direction hard-coded asymmetric values");
    }

    const float initial_destination[] = { 100.0f, 101.0f, 102.0f, 103.0f };
    for (std::size_t i = 0; i < width; ++i) {
        output.data[i] = initial_destination[i];
    }
    status = balance3_diagonal_subtract_prepared_row_00813cf8(
        fixture.center, stride, width, coeff, radius, output.data);
    check(status == Balance3KernelStatus::ok,
          "asymmetric minus diagonal subtract succeeds");
    const float expected_minus_subtract[] = { 70.5f, 67.5f, 65.5f, 63.5f };
    for (std::size_t i = 0; i < width; ++i) {
        check(same_bits(output.data[i], expected_minus_subtract[i]),
              "minus direction hard-coded asymmetric values");
    }
    check(fixture.memory.storage == source_before,
          "diagonal filters leave source and guards unchanged");
    for (std::size_t i = width; i < width + 4u; ++i) {
        check(same_bits(output.data[i], -88.0f),
              "diagonal kernels leave destination padding unchanged");
    }
}

void test_radius_matrix_and_default_coefficients() {
    constexpr std::size_t width = 8u;
    for (const std::size_t radius : { 0u, 1u, 3u, 32u }) {
        const std::size_t minimum_stride = width + radius * 2u;
        const std::size_t stride = (minimum_stride + 3u) & ~std::size_t(3u);
        const std::size_t plus_distance = stride + 1u;
        const std::size_t minus_distance = radius == 0u ? stride : stride - 1u;
        const std::size_t max_distance = plus_distance;
        const std::size_t backward = radius * max_distance;
        const std::size_t forward = backward + width - 1u;
        RowFixture fixture(width, stride, backward, forward);
        for (std::size_t i = 0; i < fixture.memory.storage.size() - 3u; ++i) {
            const int sample = static_cast<int>((i * 101u + radius * 29u) % 4093u) - 2046;
            fixture.memory.data[i] = static_cast<float>(sample) * 0.03125f;
        }
        std::vector<float> coeff(radius + 1u);
        for (std::size_t i = 0; i <= radius; ++i) {
            const int numerator = static_cast<int>(i * 7u + 3u);
            coeff[i] = static_cast<float>(numerator) * 0.015625f;
        }
        AlignedFloats output(width + 4u, 123.25f);
        const auto source_before = fixture.memory.storage;
        const auto coeff_before = coeff;
        const auto expected_plus = reference_diagonal(
            fixture.center, width, plus_distance, coeff.data(), radius,
            output.data, false);
        auto status = balance3_diagonal_filter_prepared_row_008139cc(
            fixture.center, stride, width, coeff.data(), radius, output.data);
        if (status != Balance3KernelStatus::ok) {
            std::cerr << "radius=" << radius << " plus status="
                      << static_cast<int>(status) << '\n';
        }
        check(status == Balance3KernelStatus::ok,
              "radius matrix plus filter succeeds");
        check_row_bits(output.data, expected_plus,
                       "radius matrix plus filter matches binary32 oracle");
        check(fixture.memory.storage == source_before,
              "radius matrix plus filter preserves source");
        check(coeff == coeff_before,
              "radius matrix plus filter preserves coefficients");
        for (std::size_t i = width; i < width + 4u; ++i) {
            check(same_bits(output.data[i], 123.25f),
                  "radius matrix filter preserves destination padding");
        }

        AlignedFloats subtract_output(width + 4u, 456.5f);
        std::vector<float> initial(width);
        for (std::size_t j = 0; j < width; ++j) {
            initial[j] = subtract_output.data[j];
        }
        const auto expected_minus = reference_diagonal(
            fixture.center, width, minus_distance, coeff.data(), radius,
            initial.data(), true);
        status = balance3_diagonal_subtract_prepared_row_00813cf8(
            fixture.center, stride, width, coeff.data(), radius,
            subtract_output.data);
        if (status != Balance3KernelStatus::ok) {
            std::cerr << "radius=" << radius << " subtract status="
                      << static_cast<int>(status) << '\n';
        }
        check(status == Balance3KernelStatus::ok,
              "radius matrix minus subtract succeeds");
        check_row_bits(subtract_output.data, expected_minus,
                       "radius matrix minus subtract matches binary32 oracle");
        check(fixture.memory.storage == source_before,
              "radius matrix subtract preserves source");
        for (std::size_t i = width; i < width + 4u; ++i) {
            check(same_bits(subtract_output.data[i], 456.5f),
                  "radius matrix subtract preserves destination padding");
        }
    }

    constexpr std::size_t default_radius = 3u;
    constexpr float scale = 1.0f / 16384.0f;
    const float default_coeff[] = {
        5912.0f * scale, 3948.0f * scale, 1144.0f * scale, 144.0f * scale,
    };
    constexpr std::size_t minimum_stride = width + default_radius * 2u;
    constexpr std::size_t stride = (minimum_stride + 3u) & ~std::size_t(3u);
    constexpr std::size_t distance = stride + 1u;
    constexpr std::size_t backward = default_radius * distance;
    constexpr std::size_t forward = backward + width - 1u;
    RowFixture fixture(width, stride, backward, forward);
    for (std::size_t i = 0; i < fixture.memory.storage.size() - 3u; ++i) {
        fixture.memory.data[i] = static_cast<float>(static_cast<int>((i * 43u) % 2047u) - 1023) *
                                 0.0625f;
    }
    AlignedFloats output(width + 4u, -1.0f);
    const auto expected = reference_diagonal(
        fixture.center, width, distance, default_coeff, default_radius,
        output.data, false);
    const auto status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, stride, width, default_coeff, default_radius,
        output.data);
    check(status == Balance3KernelStatus::ok,
          "recovered radius-3 default coefficients succeed");
    check_row_bits(output.data, expected,
                   "recovered radius-3 defaults match bitwise oracle");
}

void test_explicit_fma_counterexample() {
    constexpr std::size_t width = 4u;
    constexpr std::size_t stride = 8u;
    constexpr std::size_t distance = stride + 1u;
    RowFixture fixture(width, stride, distance, distance + width - 1u);
    for (std::size_t i = 0; i < fixture.memory.storage.size() - 3u; ++i) {
        fixture.memory.data[i] = 0.0f;
    }
    fixture.center[0] = 1.0f;
    const float a = 1.0f + 0x1p-23f;
    const float pair_left = 1.0f - 0x1p-23f;
    (fixture.center - static_cast<std::ptrdiff_t>(distance))[0] = pair_left;
    (fixture.center + static_cast<std::ptrdiff_t>(distance))[0] = 0.0f;
    const float coeff[] = { -1.0f, a };
    AlignedFloats output(width + 4u, 99.0f);
    const auto status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, stride, width, coeff, 1u, output.data);
    check(status == Balance3KernelStatus::ok,
          "explicit FMA counterexample succeeds");
    check(same_bits(output.data[0], -0x1p-46f),
          "diagonal accumulator uses a single-rounding FMA");
    const float unfused = round_f32(round_f32(a * pair_left) - 1.0f);
    check(same_bits(unfused, 0.0f) && !same_bits(unfused, output.data[0]),
          "FMA counterexample differs from separate multiply and add");
}

void test_failures_are_atomic() {
    constexpr std::size_t width = 4u;
    constexpr std::size_t stride = 8u;
    constexpr std::size_t backward = stride + 1u;
    constexpr std::size_t forward = stride + width;
    RowFixture fixture(width, stride, backward, forward);
    AlignedFloats dst(width + 4u, 314.25f);
    const auto dst_before = dst.storage;
    const float valid_coeff[] = { 0.5f, 0.25f };

    auto status = balance3_difference_prepared_row_00813870(
        nullptr, stride, width, 0, dst.data);
    check(status == Balance3KernelStatus::invalid_argument,
          "null source is rejected");
    status = balance3_difference_prepared_row_00813870(
        fixture.center, stride, width, 0, nullptr);
    check(status == Balance3KernelStatus::invalid_argument,
          "null destination is rejected");
    check(dst.storage == dst_before, "null argument rejections are atomic");

    status = balance3_difference_prepared_row_00813870(
        fixture.center, stride, 0u, 0, dst.data);
    check(status == Balance3KernelStatus::invalid_dimensions,
          "zero width is rejected");
    check(dst.storage == dst_before, "zero width rejection is atomic");
    status = balance3_difference_prepared_row_00813870(
        fixture.center, stride, 6u, 0, dst.data);
    check(status == Balance3KernelStatus::invalid_dimensions,
          "non-multiple-of-four width is rejected");
    check(dst.storage == dst_before, "invalid width rejection is atomic");
    status = balance3_difference_prepared_row_00813870(
        fixture.center, 6u, width, 0, dst.data);
    check(status == Balance3KernelStatus::invalid_stride,
          "difference insufficient guarded stride is rejected");
    status = balance3_difference_prepared_row_00813870(
        fixture.center, 7u, width, 0, dst.data);
    check(status == Balance3KernelStatus::invalid_stride,
          "non-multiple-of-four stride is rejected");
    check(dst.storage == dst_before, "bad stride rejections are atomic");
    status = balance3_difference_prepared_row_00813870(
        fixture.center, stride, width, 2, dst.data);
    check(status == Balance3KernelStatus::invalid_parameters,
          "invalid reverse value is rejected");
    check(dst.storage == dst_before, "bad reverse rejection is atomic");

    status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, stride, width, valid_coeff, 33u, dst.data);
    check(status == Balance3KernelStatus::invalid_parameters,
          "radius above 32 is rejected");
    check(dst.storage == dst_before, "bad radius rejection is atomic");
    const float nonfinite_coeff[] = { 1.0f, std::numeric_limits<float>::infinity() };
    status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, stride, width, nonfinite_coeff, 1u, dst.data);
    check(status == Balance3KernelStatus::invalid_parameters,
          "infinite coefficient is rejected");
    check(dst.storage == dst_before, "nonfinite coefficient rejection is atomic");
    status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, stride, width, nullptr, 0u, dst.data);
    check(status == Balance3KernelStatus::invalid_argument,
          "null coefficient pointer is rejected");
    check(dst.storage == dst_before, "null coefficient rejection is atomic");
    const auto *unaligned_coeff = reinterpret_cast<const float *>(
        reinterpret_cast<const unsigned char *>(valid_coeff) + 1u);
    status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, stride, width, unaligned_coeff, 0u, dst.data);
    check(status == Balance3KernelStatus::unsupported_alignment,
          "naturally unaligned coefficients are rejected");
    check(dst.storage == dst_before, "coefficient alignment rejection is atomic");
    status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, 4u, width, valid_coeff, 2u, dst.data);
    check(status == Balance3KernelStatus::invalid_stride,
          "diagonal insufficient guarded stride is rejected");
    check(dst.storage == dst_before, "diagonal stride rejection is atomic");

    status = balance3_difference_prepared_row_00813870(
        fixture.center + 1, stride, width, 0, dst.data);
    check(status == Balance3KernelStatus::unsupported_alignment,
          "unaligned center is rejected");
    status = balance3_difference_prepared_row_00813870(
        fixture.center, stride, width, 0, dst.data + 1);
    check(status == Balance3KernelStatus::unsupported_alignment,
          "unaligned destination is rejected");
    check(dst.storage == dst_before, "alignment rejections are atomic");

    const auto source_before_overlap = fixture.memory.storage;
    status = balance3_difference_prepared_row_00813870(
        fixture.center, stride, width, 0, fixture.center);
    check(status == Balance3KernelStatus::overlapping_buffers,
          "source and destination overlap is rejected");
    check(fixture.memory.storage == source_before_overlap,
          "source/destination overlap rejection is atomic");

    status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, stride, width, valid_coeff, 1u,
        fixture.center - 4);
    check(status == Balance3KernelStatus::overlapping_buffers,
          "conservative source span overlap is rejected");
    check(fixture.memory.storage == source_before_overlap,
          "conservative source span rejection is atomic");

    AlignedFloats shared(width + 4u, 1.0f);
    const auto shared_before = shared.storage;
    status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, stride, width, shared.data, 0u, shared.data);
    check(status == Balance3KernelStatus::overlapping_buffers,
          "coefficient and destination overlap is rejected");
    check(shared.storage == shared_before,
          "coefficient/destination overlap rejection is atomic");

    const auto max_address = std::numeric_limits<std::uintptr_t>::max();
    const auto *overflow_source = reinterpret_cast<const float *>(max_address - 15u);
    status = balance3_difference_prepared_row_00813870(
        overflow_source, stride, width, 0, dst.data);
    check(status == Balance3KernelStatus::arithmetic_overflow,
          "source declared-span address overflow is rejected");
    auto *overflow_destination = reinterpret_cast<float *>(max_address - 15u);
    status = balance3_difference_prepared_row_00813870(
        fixture.center, stride, width, 0, overflow_destination);
    check(status == Balance3KernelStatus::arithmetic_overflow,
          "destination span address overflow is rejected");
    const auto *overflow_coeff = reinterpret_cast<const float *>(max_address - 7u);
    status = balance3_diagonal_filter_prepared_row_008139cc(
        fixture.center, stride, width, overflow_coeff, 1u, dst.data);
    check(status == Balance3KernelStatus::arithmetic_overflow,
          "coefficient span address overflow is rejected");
    status = balance3_difference_prepared_row_00813870(
        fixture.center, std::numeric_limits<std::size_t>::max() - 3u,
        width, 0, dst.data);
    check(status == Balance3KernelStatus::arithmetic_overflow,
          "PTRDIFF stride overflow is rejected");
    check(dst.storage == dst_before,
          "overflow rejections leave destination unchanged");
}

}  // namespace

int main() {
    test_difference_asymmetric();
    test_diagonal_directions_hard_coded();
    test_radius_matrix_and_default_coefficients();
    test_explicit_fma_counterexample();
    test_failures_are_atomic();
    if (failures != 0) {
        std::cerr << failures << " balance3 validation checks failed\n";
        return 1;
    }
    std::cout << "balance3 prepared-row validation passed\n";
    return 0;
}
