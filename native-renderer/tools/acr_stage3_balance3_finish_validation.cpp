#include "../src/acr_stage3_balance3_finish.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using imprint::acr::Balance3FinishLimits;
using imprint::acr::Balance3FinishStatus;
using Finish = Balance3FinishStatus (*)(
    const float *, std::size_t, const float *, std::size_t, std::size_t, int,
    int, const Balance3FinishLimits &, float *, std::size_t) noexcept;

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

bool same_bits(float left, float right) { return bits(left) == bits(right); }

bool same_storage(const std::vector<float> &left,
                  const std::vector<float> &right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (!same_bits(left[index], right[index])) return false;
    }
    return true;
}

struct AlignedFloats {
    std::vector<float> storage;
    float *data = nullptr;

    explicit AlignedFloats(std::size_t count, float fill = 0.0f)
        : storage(count + 4u, fill) {
        const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
        const std::size_t misalignment =
            static_cast<std::size_t>(address & 0xFu);
        const std::size_t offset = ((16u - misalignment) & 0xFu) / sizeof(float);
        data = storage.data() + offset;
    }
};

struct RowFixture {
    static constexpr std::size_t width = 8u;
    static constexpr std::size_t stride = 12u;
    AlignedFloats memory;
    std::size_t center_index;
    float *center;

    RowFixture()
        : memory(3u * stride + width + 128u, -777.0f),
          center_index(64u + stride), center(memory.data + center_index) {
        for (std::size_t index = 0; index < 3u * stride + width + 128u;
             ++index) {
            memory.data[index] =
                static_cast<float>(static_cast<int>((index * 19u) % 127u) - 61) /
                32.0f;
        }
    }
};

float round_f32(float value) {
    volatile float rounded = value;
    return rounded;
}

float reference_min(float left, float right) {
    if (left < right) return left;
    if (right < left) return right;
    if (left == 0.0f && right == 0.0f) {
        return (std::signbit(left) || std::signbit(right)) ? -0.0f : 0.0f;
    }
    return left;
}

float reference_max(float left, float right) {
    if (left > right) return left;
    if (right > left) return right;
    if (left == 0.0f && right == 0.0f) {
        return (std::signbit(left) && std::signbit(right)) ? -0.0f : 0.0f;
    }
    return left;
}

float reference_delta(const float *source, std::size_t stride,
                      const float *diff, std::size_t lane,
                      float sign) {
    const auto signed_stride = static_cast<std::ptrdiff_t>(stride);
    const auto signed_lane = static_cast<std::ptrdiff_t>(lane);
    float sum = round_f32(
        source[-signed_stride + signed_lane - 1] +
        source[-signed_stride + signed_lane + 1]);
    sum = round_f32(sum + source[signed_stride + signed_lane - 1]);
    sum = round_f32(sum + source[signed_stride + signed_lane + 1]);
    float acc = round_f32(sum * 0.125f);
    acc = std::fma(source[lane], 0.5f, acc);
    acc = std::fma(diff[lane], sign, acc);
    return round_f32(acc - source[lane]);
}

float reference_value(float delta, float source, float lower, float upper,
                      bool optimized) {
    float clamped_delta = delta;
    if (optimized) {
        clamped_delta = reference_max(lower, reference_min(delta, upper));
    } else {
        if (!(clamped_delta < upper)) clamped_delta = upper;
        if (lower > clamped_delta) clamped_delta = lower;
    }
    float value = round_f32(source + clamped_delta);
    value = reference_min(value, 1.0f);
    if (optimized) return reference_max(0.0f, value);
    if (value < 0.0f) value = 0.0f;
    return value;
}

void prepare_asymmetric(RowFixture &fixture, AlignedFloats &diff) {
    constexpr float center[] = {
        0x1.0p-3f, 0x1.0p-2f, 0x1.8p-2f, 0x1.0p-1f,
        0x1.4p-1f, 0x1.8p-1f, 0x1.cp-1f, 0x1.0p-2f,
    };
    constexpr float north[] = {
        -0x1.0p-2f, 0x1.0p-3f, 0x1.0p-2f, -0x1.0p-1f,
        0x1.8p-1f, 0x1.8p-2f, 0x1.4p-1f, 0x1.cp-1f,
        -0x1.0p-3f, 0x1.0p-1f,
    };
    constexpr float south[] = {
        0x1.0p-1f, -0x1.0p-3f, 0x1.4p-1f, 0x1.8p-2f,
        -0x1.0p-2f, 0x1.cp-1f, -0x1.0p-1f, 0x1.0p-2f,
        0x1.0p-3f, -0x1.8p-1f,
    };
    constexpr float differences[] = {
        0x1.0p-1f, -0x1.0p-1f, 0x1.0p-2f, -0x1.0p-2f,
        0x1.0p-3f, -0x1.0p-3f, 0x1.8p-1f, -0x1.8p-1f,
    };
    float *north_row = fixture.center -
                       static_cast<std::ptrdiff_t>(RowFixture::stride);
    float *south_row = fixture.center +
                       static_cast<std::ptrdiff_t>(RowFixture::stride);
    for (std::size_t index = 0; index < 10u; ++index) {
        north_row[static_cast<std::ptrdiff_t>(index) - 1] = north[index];
        south_row[static_cast<std::ptrdiff_t>(index) - 1] = south[index];
    }
    for (std::size_t lane = 0; lane < RowFixture::width; ++lane) {
        fixture.center[lane] = center[lane];
        diff.data[lane] = differences[lane];
    }
}

void run_hardcoded_case(Finish finish, const char *path_name, int phase,
                        int row_even, const float *expected) {
    RowFixture fixture;
    AlignedFloats diff(RowFixture::width + 4u, -55.0f);
    prepare_asymmetric(fixture, diff);
    const auto source_before = fixture.memory.storage;
    const auto diff_before = diff.storage;
    constexpr float sentinel = -91.0f;
    AlignedFloats output(RowFixture::width + 4u, sentinel);
    constexpr Balance3FinishLimits limits{
        -0x1.0p-2f, 0x1.8p-2f, -0x1.0p-3f, 0x1.0p-2f};
    const auto status = finish(
        fixture.center, RowFixture::stride, diff.data, RowFixture::width + 4u,
        RowFixture::width, phase, row_even, limits, output.data,
        RowFixture::width + 4u);
    check(status == Balance3FinishStatus::ok,
          "asymmetric hard-coded row succeeds");
    for (std::size_t lane = 0; lane < RowFixture::width; ++lane) {
        if (!same_bits(output.data[lane], expected[lane])) {
            std::cerr << "FAIL: " << path_name << " phase=" << phase
                      << " row_even=" << row_even << " lane=" << lane
                      << " expected=" << expected[lane]
                      << " actual=" << output.data[lane] << '\n';
            ++failures;
            break;
        }
    }
    for (std::size_t lane = RowFixture::width;
         lane < RowFixture::width + 4u; ++lane) {
        check(same_bits(output.data[lane], sentinel),
              "finish preserves destination padding");
    }
    check(same_storage(fixture.memory.storage, source_before),
          "finish preserves source and guard storage");
    check(same_storage(diff.storage, diff_before),
          "finish preserves diff input");
}

void test_hardcoded_asymmetric_lanes() {
    constexpr float sentinel = -91.0f;
    constexpr float optimized_even_phase0_row0[] = {
        0x1.8p-2f, 0x1.0p-2f, 0x1.fp-2f, 0x1.0p-1f,
        0x1.0p-1f, 0x1.8p-1f, 0x1.a8p-1f, 0x1.0p-2f,
    };
    constexpr float optimized_odd_phase1_row0[] = {
        0x1.0p-3f, 0x1.0p-3f, 0x1.8p-2f, 0x1.8p-2f,
        0x1.4p-1f, 0x1.4p-1f, 0x1.cp-1f, 0x1.0p-3f,
    };
    constexpr float optimized_even_phase0_row1[] = {
        0.0f, 0x1.0p-2f, 0x1.ep-3f, 0x1.0p-1f,
        0x1.8p-2f, 0x1.8p-1f, 0x1.4p-1f, 0x1.0p-2f,
    };
    constexpr float optimized_odd_phase1_row1[] = {
        0x1.0p-3f, 0x1.7p-2f, 0x1.8p-2f, 0x1.08p-1f,
        0x1.4p-1f, 0x1.78p-1f, 0x1.cp-1f, 0x1.38p-1f,
    };
    constexpr float default_even_phase0_row0[] = {
        sentinel, 0x1.0p-3f, sentinel, 0x1.8p-2f,
        sentinel, 0x1.4p-1f, sentinel, 0x1.0p-3f,
    };
    constexpr float default_odd_phase1_row0[] = {
        0x1.8p-2f, sentinel, 0x1.fp-2f, sentinel,
        0x1.0p-1f, sentinel, 0x1.a8p-1f, sentinel,
    };
    constexpr float default_even_phase0_row1[] = {
        sentinel, 0x1.7p-2f, sentinel, 0x1.08p-1f,
        sentinel, 0x1.78p-1f, sentinel, 0x1.38p-1f,
    };
    constexpr float default_odd_phase1_row1[] = {
        0.0f, sentinel, 0x1.ep-3f, sentinel,
        0x1.8p-2f, sentinel, 0x1.4p-1f, sentinel,
    };

    using imprint::acr::balance3_finish_prepared_row_0081403c;
    using imprint::acr::balance3_finish_prepared_row_013312fc;
    run_hardcoded_case(balance3_finish_prepared_row_0081403c, "0081403c", 0,
                       0, optimized_even_phase0_row0);
    run_hardcoded_case(balance3_finish_prepared_row_0081403c, "0081403c", 1,
                       0, optimized_odd_phase1_row0);
    run_hardcoded_case(balance3_finish_prepared_row_0081403c, "0081403c", 0,
                       1, optimized_even_phase0_row1);
    run_hardcoded_case(balance3_finish_prepared_row_0081403c, "0081403c", 1,
                       1, optimized_odd_phase1_row1);
    run_hardcoded_case(balance3_finish_prepared_row_013312fc, "013312fc", 0,
                       0, default_even_phase0_row0);
    run_hardcoded_case(balance3_finish_prepared_row_013312fc, "013312fc", 1,
                       0, default_odd_phase1_row0);
    run_hardcoded_case(balance3_finish_prepared_row_013312fc, "013312fc", 0,
                       1, default_even_phase0_row1);
    run_hardcoded_case(balance3_finish_prepared_row_013312fc, "013312fc", 1,
                       1, default_odd_phase1_row1);
}

std::uint32_t next_random(std::uint32_t *state) {
    *state ^= *state << 13u;
    *state ^= *state >> 17u;
    *state ^= *state << 5u;
    return *state;
}

float random_small(std::uint32_t *state) {
    const int numerator = static_cast<int>(next_random(state) % 8193u) - 4096;
    return static_cast<float>(numerator) * 0x1.0p-12f;
}

void test_random_finite_rows() {
    using imprint::acr::balance3_finish_prepared_row_0081403c;
    using imprint::acr::balance3_finish_prepared_row_013312fc;
    constexpr float sentinel = -123.5f;
    std::uint32_t random_state = 0x71ac93e5u;
    for (unsigned iteration = 0; iteration < 80u; ++iteration) {
        RowFixture fixture;
        for (std::size_t index = 0;
             index < 3u * RowFixture::stride + RowFixture::width + 128u;
             ++index) {
            fixture.memory.data[index] = random_small(&random_state);
        }
        AlignedFloats diff(RowFixture::width + 4u, 0.0f);
        for (std::size_t lane = 0; lane < RowFixture::width; ++lane) {
            diff.data[lane] = random_small(&random_state);
        }
        const auto source_before = fixture.memory.storage;
        const auto diff_before = diff.storage;
        float a = random_small(&random_state);
        float b = random_small(&random_state);
        float c = random_small(&random_state);
        float d = random_small(&random_state);
        Balance3FinishLimits limits{std::min(a, b), std::max(a, b),
                                    std::min(c, d), std::max(c, d)};

        for (const int row_even : {0, 1}) {
            const float sign = row_even != 0 ? -0.5f : 0.5f;
            const float lower = row_even != 0 ? limits.lower_even
                                               : limits.lower_odd;
            const float upper = row_even != 0 ? limits.upper_even
                                               : limits.upper_odd;
            for (const int phase : {0, 1}) {
                for (const bool optimized : {false, true}) {
                    AlignedFloats output(RowFixture::width + 4u, sentinel);
                    const auto expected_source = fixture.memory.storage;
                    const auto expected_diff = diff.storage;
                    const Finish finish = optimized
                                              ? balance3_finish_prepared_row_0081403c
                                              : balance3_finish_prepared_row_013312fc;
                    const auto status = finish(
                        fixture.center, RowFixture::stride, diff.data,
                        RowFixture::width + 4u, RowFixture::width, phase,
                        row_even, limits, output.data,
                        RowFixture::width + 4u);
                    check(status == Balance3FinishStatus::ok,
                          "random finite prepared row succeeds");
                    for (std::size_t lane = 0; lane < RowFixture::width;
                         ++lane) {
                        const bool selected = optimized
                                                  ? ((lane & 1u) ==
                                                     static_cast<unsigned>(phase))
                                                  : ((lane & 1u) !=
                                                     static_cast<unsigned>(phase));
                        float expected = sentinel;
                        if (selected) {
                            const float delta = reference_delta(
                                fixture.center, RowFixture::stride, diff.data,
                                lane, sign);
                            expected = reference_value(delta,
                                                       fixture.center[lane],
                                                       lower, upper, optimized);
                        } else if (optimized) {
                            expected = fixture.center[lane];
                        }
                        check(same_bits(output.data[lane], expected),
                              "random row matches independent binary32 oracle");
                    }
                    for (std::size_t lane = RowFixture::width;
                         lane < RowFixture::width + 4u; ++lane) {
                        check(same_bits(output.data[lane], sentinel),
                              "random row preserves destination padding");
                    }
                    check(same_storage(fixture.memory.storage, expected_source),
                          "random row preserves source");
                    check(same_storage(diff.storage, expected_diff),
                          "random row preserves diff");
                }
            }
        }
        check(same_storage(fixture.memory.storage, source_before),
              "random row case leaves source stable across flags");
        check(same_storage(diff.storage, diff_before),
              "random row case leaves diff stable across flags");
    }
}

void test_explicit_fma_subnormal_counterexamples() {
    using imprint::acr::balance3_finish_prepared_row_0081403c;
    using imprint::acr::balance3_finish_prepared_row_013312fc;
    const float subnormal = std::numeric_limits<float>::denorm_min();
    constexpr Balance3FinishLimits limits{-1.0f, 1.0f, -1.0f, 1.0f};

    for (const bool second_fma : {false, true}) {
        RowFixture fixture;
        std::fill(fixture.memory.data,
                  fixture.memory.data +
                      3u * RowFixture::stride + RowFixture::width + 128u,
                  0.0f);
        AlignedFloats diff(RowFixture::width + 4u, 0.0f);
        float *south = fixture.center +
                       static_cast<std::ptrdiff_t>(RowFixture::stride);
        south[1] = 8.0f * subnormal;
        if (second_fma) {
            diff.data[0] = subnormal;
        } else {
            fixture.center[0] = subnormal;
        }
        AlignedFloats output(RowFixture::width + 4u, -4.0f);
        const Finish finish = second_fma
                                  ? balance3_finish_prepared_row_013312fc
                                  : balance3_finish_prepared_row_0081403c;
        const int phase = second_fma ? 1 : 0;
        const auto status = finish(
            fixture.center, RowFixture::stride, diff.data,
            RowFixture::width + 4u, RowFixture::width, phase, 0, limits,
            output.data, RowFixture::width + 4u);
        check(status == Balance3FinishStatus::ok,
              "subnormal explicit-fma case succeeds");
        check(same_bits(output.data[0], 2.0f * subnormal),
              second_fma
                  ? "second explicit FMA rounds half-subnormal sum to even"
                  : "first explicit FMA rounds half-subnormal sum to even");
    }
}

void test_signed_zero_branch_difference() {
    using imprint::acr::balance3_finish_prepared_row_0081403c;
    using imprint::acr::balance3_finish_prepared_row_013312fc;
    RowFixture fixture;
    std::fill(fixture.memory.data,
              fixture.memory.data +
                  3u * RowFixture::stride + RowFixture::width + 128u,
              -0.0f);
    AlignedFloats diff(RowFixture::width + 4u, 0.0f);
    AlignedFloats optimized(RowFixture::width + 4u, 7.0f);
    AlignedFloats original(RowFixture::width + 4u, 7.0f);
    const Balance3FinishLimits negative_zero{
        -0.0f, -0.0f, -0.0f, -0.0f};
    const auto optimized_status = balance3_finish_prepared_row_0081403c(
        fixture.center, RowFixture::stride, diff.data, RowFixture::width,
        RowFixture::width, 0, 0, negative_zero, optimized.data,
        RowFixture::width);
    const auto default_status = balance3_finish_prepared_row_013312fc(
        fixture.center, RowFixture::stride, diff.data, RowFixture::width,
        RowFixture::width, 1, 0, negative_zero, original.data,
        RowFixture::width);
    check(optimized_status == Balance3FinishStatus::ok &&
              default_status == Balance3FinishStatus::ok,
          "signed-zero clamp rows succeed");
    check(same_bits(optimized.data[0], 0.0f),
          "optimized FMAX(+0, -0) produces positive zero");
    check(same_bits(original.data[0], -0.0f),
          "default FMINNM and negative test preserve negative zero");
    check(same_bits(optimized.data[1], -0.0f),
          "optimized unselected lane copies negative zero exactly");
    check(same_bits(original.data[1], 7.0f),
          "default unselected lane preserves destination exactly");
}

void test_source_diff_read_alias_allowed() {
    using imprint::acr::balance3_finish_prepared_row_0081403c;
    RowFixture fixture;
    AlignedFloats output(RowFixture::width + 4u, -18.0f);
    constexpr Balance3FinishLimits limits{-1.0f, 1.0f, -1.0f, 1.0f};
    const auto status = balance3_finish_prepared_row_0081403c(
        fixture.center, RowFixture::stride, fixture.center,
        RowFixture::width, RowFixture::width, 0, 0, limits, output.data,
        RowFixture::width);
    check(status == Balance3FinishStatus::ok,
          "two read-only inputs may overlap");
}

void test_rejections_are_atomic() {
    using imprint::acr::balance3_finish_prepared_row_0081403c;
    RowFixture fixture;
    AlignedFloats diff(RowFixture::width + 4u, 0.25f);
    AlignedFloats output(RowFixture::width + 4u, -18.0f);
    constexpr Balance3FinishLimits limits{-1.0f, 1.0f, -1.0f, 1.0f};
    const auto initial_output = output.storage;
    auto call = [&](const float *source, std::size_t stride,
                    const float *differences, std::size_t diff_count,
                    std::size_t width, int phase, int row_even,
                    const Balance3FinishLimits &control,
                    float *destination, std::size_t destination_count) {
        const auto before = output.storage;
        const auto status = balance3_finish_prepared_row_0081403c(
            source, stride, differences, diff_count, width, phase, row_even,
            control, destination, destination_count);
        check(same_storage(output.storage, before),
              "rejected call leaves output storage unchanged");
        return status;
    };

    check(call(nullptr, RowFixture::stride, diff.data, RowFixture::width,
               RowFixture::width, 0, 0, limits, output.data,
               RowFixture::width) == Balance3FinishStatus::invalid_argument,
          "null source rejected");
    check(call(fixture.center, RowFixture::stride, nullptr, RowFixture::width,
               RowFixture::width, 0, 0, limits, output.data,
               RowFixture::width) == Balance3FinishStatus::invalid_argument,
          "null diff rejected");
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, 0, limits, nullptr,
               RowFixture::width) == Balance3FinishStatus::invalid_argument,
          "null destination rejected");

    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, 0u, 0, 0, limits, output.data,
               RowFixture::width) == Balance3FinishStatus::invalid_dimensions,
          "zero width rejected");
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, 6u, 0, 0, limits, output.data,
               RowFixture::width) == Balance3FinishStatus::invalid_dimensions,
          "non-multiple-of-four width rejected");
    check(call(fixture.center, 10u, diff.data, RowFixture::width,
               RowFixture::width, 0, 0, limits, output.data,
               RowFixture::width) == Balance3FinishStatus::invalid_stride,
          "non-multiple-of-four stride rejected");
    check(call(fixture.center, 8u, diff.data, RowFixture::width,
               RowFixture::width, 0, 0, limits, output.data,
               RowFixture::width) == Balance3FinishStatus::invalid_stride,
          "insufficient guarded stride rejected");
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 2, 0, limits,
               output.data, RowFixture::width) ==
              Balance3FinishStatus::invalid_parameters,
          "invalid phase rejected");
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, -1, limits,
               output.data, RowFixture::width) ==
              Balance3FinishStatus::invalid_parameters,
          "invalid row_even rejected");
    const Balance3FinishLimits nan_limits{
        std::numeric_limits<float>::quiet_NaN(), 1.0f, -1.0f, 1.0f};
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, 0, nan_limits,
               output.data, RowFixture::width) ==
              Balance3FinishStatus::invalid_parameters,
          "nonfinite controls rejected");
    const Balance3FinishLimits reversed_limits{1.0f, -1.0f, -1.0f, 1.0f};
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, 0, reversed_limits,
               output.data, RowFixture::width) ==
              Balance3FinishStatus::invalid_parameters,
          "reversed controls rejected");
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width - 1u, RowFixture::width, 0, 0, limits,
               output.data, RowFixture::width) ==
              Balance3FinishStatus::buffer_too_small,
          "short diff capacity rejected");
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, 0, limits,
               output.data, RowFixture::width - 1u) ==
              Balance3FinishStatus::buffer_too_small,
          "short destination capacity rejected");
    check(call(fixture.center, RowFixture::stride, diff.data + 1,
               RowFixture::width, RowFixture::width, 0, 0, limits,
               output.data, RowFixture::width) ==
              Balance3FinishStatus::unsupported_alignment,
          "misaligned diff rejected");
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, 0, limits,
               output.data + 1, RowFixture::width) ==
              Balance3FinishStatus::unsupported_alignment,
          "misaligned destination rejected");
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, 0, limits,
               fixture.center, RowFixture::width) ==
              Balance3FinishStatus::overlapping_buffers,
          "source and destination overlap rejected");
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, 0, limits,
               diff.data, RowFixture::width) ==
              Balance3FinishStatus::overlapping_buffers,
          "diff and destination overlap rejected");

    const float original_center = fixture.center[RowFixture::width - 1u];
    fixture.center[RowFixture::width - 1u] =
        std::numeric_limits<float>::infinity();
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, 0, limits,
               output.data, RowFixture::width) ==
              Balance3FinishStatus::unsupported_domain,
          "nonfinite source sample rejected before writes");
    fixture.center[RowFixture::width - 1u] = original_center;
    const float original_diff = diff.data[RowFixture::width - 1u];
    diff.data[RowFixture::width - 1u] =
        std::numeric_limits<float>::quiet_NaN();
    check(call(fixture.center, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, 0, limits,
               output.data, RowFixture::width) ==
              Balance3FinishStatus::unsupported_domain,
          "nonfinite diff sample rejected before writes");
    diff.data[RowFixture::width - 1u] = original_diff;

    constexpr std::size_t huge_width =
        std::numeric_limits<std::size_t>::max() - 7u;
    constexpr std::size_t huge_stride =
        std::numeric_limits<std::size_t>::max() - 3u;
    check(call(fixture.center, huge_stride, diff.data, huge_width, huge_width,
               0, 0, limits, output.data, huge_width) ==
              Balance3FinishStatus::arithmetic_overflow,
          "PTRDIFF source or row overflow rejected");

    const auto top_address = std::numeric_limits<std::uintptr_t>::max() &
                             ~static_cast<std::uintptr_t>(0xFu);
    const auto *near_address_end = reinterpret_cast<const float *>(top_address);
    check(call(near_address_end, RowFixture::stride, diff.data,
               RowFixture::width, RowFixture::width, 0, 0, limits,
               output.data, RowFixture::width) ==
              Balance3FinishStatus::arithmetic_overflow,
          "source address span overflow rejected before reads");
    check(same_storage(output.storage, initial_output),
          "all rejection cases preserve entire destination storage");
}

void test_basic_inline_branch() {
    using imprint::acr::balance3_basic_prepared_row_007ed860;
    AlignedFloats source(48u, 0.25f), diff(8u, 0.5f), output(8u, -9.0f);
    auto *center = source.data + 16u;
    for (int phase = 0; phase != 2; ++phase) {
        for (int even = 0; even != 2; ++even) {
            std::fill(output.data, output.data + 8u, -9.0f);
            const auto status = balance3_basic_prepared_row_007ed860(
                center, 12u, diff.data, 8u, 4u, phase, even, output.data, 8u);
            check(status == Balance3FinishStatus::ok, "basic branch accepted");
            for (std::size_t j = 0; j != 8u; ++j) {
                const bool selected = j < 4u && (j & 1u) != static_cast<unsigned>(phase);
                check(same_bits(output.data[j], selected ? (even ? 0.0f : 0.5f) : -9.0f),
                      "basic selected lanes, sign and untouched tail");
            }
        }
    }
    const auto before = output.storage;
    center[12u + 4u] = std::numeric_limits<float>::quiet_NaN();
    check(balance3_basic_prepared_row_007ed860(center, 12u, diff.data, 8u,
          4u, 0, 0, output.data, 8u) == Balance3FinishStatus::unsupported_domain,
          "basic last guard rejected");
    check(same_storage(before, output.storage), "basic failure atomic");
}

}  // namespace

int main() {
    test_hardcoded_asymmetric_lanes();
    test_random_finite_rows();
    test_explicit_fma_subnormal_counterexamples();
    test_signed_zero_branch_difference();
    test_source_diff_read_alias_allowed();
    test_rejections_are_atomic();
    test_basic_inline_branch();
    if (failures != 0) {
        std::cerr << failures << " balance3 finish validation failure(s)\n";
        return 1;
    }
    std::cout << "balance3 finish validation passed\n";
    return 0;
}
