#include "../src/acr_camera_combined.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

extern "C" int acr_test_camera_combined(
    const float *src, std::uint64_t src_count, std::uint64_t src_row_stride,
    std::uint64_t height, std::uint64_t width, const double *controls,
    std::uint64_t control_count, const std::uint32_t order3[3],
    std::uint32_t flags, float *dst, std::uint64_t dst_count,
    std::uint64_t dst_row_stride) noexcept;

namespace {

using imprint::acr::CombinedCameraStatus;

constexpr std::array<std::uint32_t, 3> kRgbOrder = {0u, 1u, 2u};
constexpr std::array<double, 9> kMatrix = {
    0.8125, -0.0625, 0.125,
    0.1875, 0.75, 0.0625,
    -0.125, 0.25, 0.875};

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0u;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

bool same_bits(float left, float right) { return bits(left) == bits(right); }

float add32(float left, float right) {
    volatile float result = left + right;
    return result;
}

float sub32(float left, float right) {
    volatile float result = left - right;
    return result;
}

float mul32(float left, float right) {
    volatile float result = left * right;
    return result;
}

float div32(float left, float right) {
    volatile float result = left / right;
    return result;
}

float maxnm_zero(float value) { return value > 0.0f ? value : 0.0f; }
float max_gt(float left, float right) { return left > right ? left : right; }

float clamp32(float value) {
    const float capped = value > 1.0f ? 1.0f : value;
    return capped < 0.0f ? 0.0f : capped;
}

struct Image {
    std::size_t height;
    std::size_t width;
    std::size_t stride;
    std::vector<float> samples;

    Image(std::size_t h, std::size_t w, std::size_t row_padding,
          float fill = 0.0f)
        : height(h), width(w), stride(w * 3u + row_padding),
          samples(h * stride, fill) {}

    std::size_t active_row_samples() const { return width * 3u; }

    float &at(std::size_t y, std::size_t x, std::size_t channel) {
        return samples[y * stride + x * 3u + channel];
    }

    const float &at(std::size_t y, std::size_t x,
                    std::size_t channel) const {
        return samples[y * stride + x * 3u + channel];
    }
};

std::array<double, 13> make_controls(float a, float b_gain, float upper,
                                    float black,
                                    const std::array<double, 9> &matrix = {
                                        1.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 0.0, 1.0}) {
    std::array<double, 13> values{};
    values[0] = a;
    values[1] = b_gain;
    values[2] = upper;
    values[3] = black;
    for (std::size_t i = 0; i < 9u; ++i) values[i + 4u] = matrix[i];
    return values;
}

struct ReferenceConstants {
    float ratio;
    float inv_a;
    float q0;
    float q1;
    float q2;
    float i0;
    float i1;
    float i2;
    float delta;
    float inv_black_delta;
    float black_bias;
    float output_black;
    std::array<float, 9> matrix;
    std::array<float, 9> pedestal_matrix;
    float ratio_inv_black;
    float inv_a_inv_black;
    float ratio_bias;
    float inv_a_bias;
    bool pedestal;
    bool direct;
    bool clamp;
};

ReferenceConstants reference_constants(const std::array<double, 13> &controls,
                                       std::uint32_t flags) {
    // This oracle is kept in the validation translation unit and spells out
    // the reference steps independently of the production helpers.
    float c[13];
    for (std::size_t i = 0; i < 13u; ++i) c[i] = static_cast<float>(controls[i]);
    ReferenceConstants result{};
    result.ratio = div32(c[1], c[0]);
    result.inv_a = div32(1.0f, c[0]);
    result.q0 = mul32(c[2], 0.25f);
    result.q1 = mul32(result.ratio, result.q0);
    result.q2 = mul32(result.inv_a, result.q0);
    result.i0 = div32(1.0f, sub32(1.0f, result.q0));
    result.i1 = div32(1.0f, sub32(result.ratio, result.q1));
    result.i2 = div32(1.0f, sub32(result.inv_a, result.q2));
    result.delta = add32(c[3], -1.0f);
    result.inv_black_delta = c[3] == 1.0f ? 0.0f
                                           : div32(-1.0f, result.delta);
    result.pedestal = c[3] != 0.0f;
    result.direct = result.inv_a >= 1.0f;
    result.clamp = (flags & 1u) == 0u;
    for (std::size_t i = 0; i < 9u; ++i) result.matrix[i] = c[i + 4u];

    if (result.pedestal) {
        result.black_bias = std::fma(-result.inv_black_delta, c[3], 0.0f);
        result.output_black = std::fma(result.delta, 0.0f, c[3]);
        for (std::size_t i = 0; i < 9u; ++i) {
            result.pedestal_matrix[i] = -mul32(result.delta, result.matrix[i]);
        }
        result.ratio_inv_black = mul32(result.ratio,
                                       result.inv_black_delta);
        result.inv_a_inv_black = mul32(result.inv_a,
                                       result.inv_black_delta);
        result.ratio_bias = mul32(result.ratio, result.black_bias);
        result.inv_a_bias = mul32(result.inv_a, result.black_bias);
    }
    return result;
}

bool reference_smooth(float *z0, float *z1, float *z2,
                      const ReferenceConstants &k) {
    const float h = max_gt(*z0, *z1);
    float x = mul32(sub32(*z0, k.q0), k.i0);
    x = maxnm_zero(x);
    x = mul32(x, x);
    *z0 = std::fma(x, sub32(h, *z0), *z0);

    x = mul32(sub32(*z1, k.q1), k.i1);
    x = maxnm_zero(x);
    x = mul32(x, x);
    *z1 = std::fma(x, sub32(h, *z1), *z1);

    const float average = mul32(add32(*z0, *z1), 0.5f);
    const float h2 = max_gt(average, *z2);
    x = mul32(sub32(*z2, k.q2), k.i2);
    x = maxnm_zero(x);
    x = mul32(x, x);
    *z2 = std::fma(x, sub32(h2, *z2), *z2);
    return std::isfinite(*z0) && std::isfinite(*z1) && std::isfinite(*z2);
}

std::vector<float> reference(const Image &source,
                             const std::array<double, 13> &controls,
                             const std::array<std::uint32_t, 3> &order,
                             std::uint32_t flags) {
    const ReferenceConstants k = reference_constants(controls, flags);
    std::vector<float> output(source.height * source.width * 3u);
    for (std::size_t y = 0; y < source.height; ++y) {
        for (std::size_t x = 0; x < source.width; ++x) {
            float z0 = source.at(y, x, order[0]);
            float z1;
            float z2;
            if (k.pedestal) {
                z0 = std::fma(k.inv_black_delta, z0, k.black_bias);
                z1 = std::fma(k.ratio_inv_black,
                              source.at(y, x, order[1]), k.ratio_bias);
                z2 = std::fma(k.inv_a_inv_black,
                              source.at(y, x, order[2]), k.inv_a_bias);
            } else {
                z1 = mul32(k.ratio, source.at(y, x, order[1]));
                z2 = mul32(k.inv_a, source.at(y, x, order[2]));
            }
            if (!k.direct) reference_smooth(&z0, &z1, &z2, k);
            for (std::size_t row = 0; row < 3u; ++row) {
                const auto &m = k.pedestal ? k.pedestal_matrix : k.matrix;
                float value = mul32(m[row * 3u + 1u], z1);
                value = std::fma(z0, m[row * 3u], value);
                value = std::fma(z2, m[row * 3u + 2u], value);
                if (k.pedestal) value = add32(k.output_black, value);
                if (k.clamp) value = clamp32(value);
                output[(y * source.width + x) * 3u + row] = value;
            }
        }
    }
    return output;
}

int invoke(const float *src, std::size_t src_count, std::size_t src_stride,
           std::size_t height, std::size_t width, const double *controls,
           std::size_t control_count, const std::uint32_t order[3],
           std::uint32_t flags, float *dst, std::size_t dst_count,
           std::size_t dst_stride) {
    return acr_test_camera_combined(
        src, static_cast<std::uint64_t>(src_count),
        static_cast<std::uint64_t>(src_stride),
        static_cast<std::uint64_t>(height), static_cast<std::uint64_t>(width),
        controls, static_cast<std::uint64_t>(control_count), order, flags, dst,
        static_cast<std::uint64_t>(dst_count),
        static_cast<std::uint64_t>(dst_stride));
}

void check_active_matches(const Image &actual,
                          const std::vector<float> &expected,
                          const char *message) {
    bool matches = true;
    for (std::size_t y = 0; y < actual.height; ++y) {
        for (std::size_t sample = 0; sample < actual.active_row_samples(); ++sample) {
            const std::size_t expected_index = y * actual.active_row_samples() + sample;
            if (!same_bits(actual.samples[y * actual.stride + sample],
                           expected[expected_index])) {
                matches = false;
                break;
            }
        }
    }
    check(matches, message);
}

void fill_source(Image *image) {
    for (std::size_t y = 0; y < image->height; ++y) {
        for (std::size_t x = 0; x < image->width; ++x) {
            for (std::size_t channel = 0; channel < 3u; ++channel) {
                const int code = static_cast<int>((y * 11u + x * 7u + channel * 5u) % 23u);
                image->at(y, x, channel) = static_cast<float>(code - 5) * 0.0625f;
            }
        }
        for (std::size_t sample = image->active_row_samples();
             sample < image->stride; ++sample) {
            image->samples[y * image->stride + sample] =
                std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void run_reference_case(std::size_t width, float a, float b_gain,
                        float upper, float black, std::uint32_t flags,
                        const std::array<std::uint32_t, 3> &order,
                        const char *label) {
    constexpr std::size_t height = 2u;
    constexpr std::size_t row_padding = 5u;
    Image source(height, width, row_padding);
    fill_source(&source);
    Image destination(height, width, row_padding, 0x1.23456p+8f);
    const auto original = source.samples;
    const auto output_before = destination.samples;
    const auto controls = make_controls(a, b_gain, upper, black, kMatrix);
    const auto expected = reference(source, controls, order, flags);
    const int status = invoke(source.samples.data(), source.samples.size(),
                              source.stride, height, width, controls.data(),
                              controls.size(), order.data(), flags,
                              destination.samples.data(), destination.samples.size(),
                              destination.stride);
    check(status == static_cast<int>(CombinedCameraStatus::ok), label);
    check_active_matches(destination, expected,
                         "HWC output matches independent binary32 oracle");
    check(std::memcmp(source.samples.data(), original.data(),
                      original.size() * sizeof(float)) == 0,
          "kernel preserves source bytes including NaN row padding");
    bool padding_preserved = true;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t sample = destination.active_row_samples();
             sample < destination.stride; ++sample) {
            if (!same_bits(destination.samples[y * destination.stride + sample],
                           output_before[y * destination.stride + sample])) {
                padding_preserved = false;
            }
        }
    }
    check(padding_preserved, "destination row padding stays untouched");
}

void test_reference_paths_and_widths() {
    for (std::size_t width : {3u, 8u, 12u, 13u}) {
        run_reference_case(width, 1.0f, 1.0f, 0.75f, 0.0f, 0u,
                           kRgbOrder, "b0 direct clamp path succeeds");
        run_reference_case(width, 2.0f, 1.0f, 0.75f, 0.0f, 1u,
                           kRgbOrder, "b0 smooth preserve path succeeds");
        run_reference_case(width, 1.0f, 1.0f, 1.0f, 0.375f, 0u,
                           kRgbOrder, "pedestal direct clamp path succeeds");
        run_reference_case(width, 2.0f, 1.0f, 0.75f, 0.5f, 1u,
                           kRgbOrder, "pedestal smooth preserve path succeeds");
    }
}

void test_permutation_and_legacy_contrast_probe() {
    constexpr std::array<std::uint32_t, 3> permutation = {2u, 0u, 1u};
    Image source(1u, 1u, 0u);
    source.at(0u, 0u, 0u) = 0.125f;
    source.at(0u, 0u, 1u) = 0.25f;
    source.at(0u, 0u, 2u) = 0.5f;
    Image destination(1u, 1u, 0u, -9.0f);
    const auto controls = make_controls(1.0f, 1.0f, 0.75f, 0.0f);
    const int status = invoke(source.samples.data(), source.samples.size(), 3u,
                              1u, 1u, controls.data(), controls.size(),
                              permutation.data(), 1u, destination.samples.data(),
                              destination.samples.size(), 3u);
    check(status == static_cast<int>(CombinedCameraStatus::ok),
          "HWC channel permutation succeeds");
    check(same_bits(destination.samples[0], 0.5f) &&
              same_bits(destination.samples[1], 0.125f) &&
              same_bits(destination.samples[2], 0.25f),
          "permutation writes output in fixed RGB order");

    // Fixed contrast probe: the combined ordinary path has normalized
    // [x, (B/A)y, (1/A)z] input scaling. For this below-threshold sample its
    // exact result differs from the old standalone direct-scale signature
    // [A*x, B*y, z]. No standalone kernel is called or used as an oracle.
    Image contrast_source(1u, 1u, 0u);
    contrast_source.at(0u, 0u, 0u) = 0.125f;
    contrast_source.at(0u, 0u, 1u) = 0.125f;
    contrast_source.at(0u, 0u, 2u) = 0.125f;
    Image contrast_output(1u, 1u, 0u, -9.0f);
    const auto contrast_controls = make_controls(2.0f, 1.0f, 0.75f, 0.0f);
    const int contrast_status = invoke(
        contrast_source.samples.data(), contrast_source.samples.size(), 3u,
        1u, 1u, contrast_controls.data(), contrast_controls.size(),
        kRgbOrder.data(), 1u, contrast_output.samples.data(),
        contrast_output.samples.size(), 3u);
    check(contrast_status == static_cast<int>(CombinedCameraStatus::ok),
          "fixed combined-versus-standalone contrast probe succeeds");
    check(same_bits(contrast_output.samples[0], 0.125f) &&
              same_bits(contrast_output.samples[1], 0.0625f) &&
              same_bits(contrast_output.samples[2], 0.0625f),
          "fixed combined probe has hard-coded expected values");
    check(!(same_bits(contrast_output.samples[0], 0.25f) &&
            same_bits(contrast_output.samples[1], 0.125f) &&
            same_bits(contrast_output.samples[2], 0.125f)),
          "fixed probe differs from legacy direct-scale signature");
}

void test_flags_and_signed_zero() {
    Image source(1u, 1u, 0u);
    source.at(0u, 0u, 0u) = -0.25f;
    source.at(0u, 0u, 1u) = 1.5f;
    source.at(0u, 0u, 2u) = 0.375f;
    const auto controls = make_controls(1.0f, 1.0f, 1.0f, 0.0f);
    Image clamped(1u, 1u, 0u, 77.0f);
    Image preserved(1u, 1u, 0u, 77.0f);
    check(invoke(source.samples.data(), 3u, 3u, 1u, 1u, controls.data(),
                 controls.size(), kRgbOrder.data(), 0u, clamped.samples.data(),
                 3u, 3u) == static_cast<int>(CombinedCameraStatus::ok),
          "flag bit zero clamp request succeeds");
    check(invoke(source.samples.data(), 3u, 3u, 1u, 1u, controls.data(),
                 controls.size(), kRgbOrder.data(), 0xfffffffeu | 1u,
                 preserved.samples.data(), 3u, 3u) ==
              static_cast<int>(CombinedCameraStatus::ok),
          "flag bit one ignores all other bits");
    check(same_bits(clamped.samples[0], 0.0f) &&
              same_bits(clamped.samples[1], 1.0f) &&
              same_bits(clamped.samples[2], 0.375f),
          "clamp path bounds negative and overrange values");
    check(same_bits(preserved.samples[0], -0.25f) &&
              same_bits(preserved.samples[1], 1.5f) &&
              same_bits(preserved.samples[2], 0.375f),
          "preserve path retains negative and overrange values");

    Image zeros(1u, 1u, 0u);
    for (float &value : zeros.samples) value = -0.0f;
    Image zero_output(1u, 1u, 0u, 11.0f);
    check(invoke(zeros.samples.data(), 3u, 3u, 1u, 1u, controls.data(),
                 controls.size(), kRgbOrder.data(), 0u,
                 zero_output.samples.data(), 3u, 3u) ==
              static_cast<int>(CombinedCameraStatus::ok),
          "signed-zero clamp case succeeds");
    check(bits(zero_output.samples[0]) == 0x80000000u &&
              bits(zero_output.samples[1]) == 0x80000000u &&
              bits(zero_output.samples[2]) == 0x80000000u,
          "clamp preserves negative zero");
}

void test_exact_in_place_permutation_and_padding() {
    constexpr std::array<std::uint32_t, 3> permutation = {1u, 2u, 0u};
    Image source(2u, 4u, 4u, -999.0f);
    fill_source(&source);
    Image separate(2u, 4u, 4u, 123.0f);
    const auto controls = make_controls(2.0f, 1.25f, 0.875f, 0.25f, kMatrix);
    const auto expected = reference(source, controls, permutation, 1u);
    check(invoke(source.samples.data(), source.samples.size(), source.stride,
                 source.height, source.width, controls.data(), controls.size(),
                 permutation.data(), 1u, separate.samples.data(),
                 separate.samples.size(), separate.stride) ==
              static_cast<int>(CombinedCameraStatus::ok),
          "out-of-place stride fixture succeeds");
    Image in_place = source;
    check(invoke(in_place.samples.data(), in_place.samples.size(),
                 in_place.stride, in_place.height, in_place.width,
                 controls.data(), controls.size(), permutation.data(), 1u,
                 in_place.samples.data(), in_place.samples.size(),
                 in_place.stride) ==
              static_cast<int>(CombinedCameraStatus::ok),
          "exact in-place HWC permutation succeeds");
    check_active_matches(separate, expected,
                         "out-of-place result matches HWC oracle");
    check_active_matches(in_place, expected,
                         "in-place result matches HWC oracle");
    bool padding_equal = true;
    for (std::size_t y = 0; y < in_place.height; ++y) {
        for (std::size_t sample = in_place.active_row_samples();
             sample < in_place.stride; ++sample) {
            const auto index = y * in_place.stride + sample;
            if (!same_bits(in_place.samples[index], source.samples[index]))
                padding_equal = false;
        }
    }
    check(padding_equal, "in-place permutation leaves row padding unchanged");
}

void expect_atomic_status(int expected_status, const float *src,
                          std::size_t src_count, std::size_t src_stride,
                          std::size_t height, std::size_t width,
                          const double *controls, std::size_t control_count,
                          const std::uint32_t order[3], std::uint32_t flags,
                          float *dst, std::size_t dst_count,
                          std::size_t dst_stride, const char *label) {
    std::vector<float> before(dst, dst + dst_count);
    const int status = invoke(src, src_count, src_stride, height, width,
                              controls, control_count, order, flags, dst,
                              dst_count, dst_stride);
    check(status == expected_status, label);
    check(std::memcmp(dst, before.data(), before.size() * sizeof(float)) == 0,
          "rejected call leaves the full destination span unchanged");
}

void test_invalid_inputs_and_atomic_failures() {
    Image source(1u, 2u, 0u);
    fill_source(&source);
    Image destination(1u, 2u, 0u, 44.5f);
    const auto controls = make_controls(1.0f, 1.0f, 1.0f, 0.0f);
    auto invalid_controls = controls;

    expect_atomic_status(static_cast<int>(CombinedCameraStatus::invalid_argument),
                         nullptr, 6u, 6u, 1u, 2u, controls.data(),
                         controls.size(), kRgbOrder.data(), 0u,
                         destination.samples.data(), destination.samples.size(),
                         destination.stride, "null source is rejected");
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::invalid_dimensions),
                         source.samples.data(), source.samples.size(),
                         source.stride, 0u, 2u, controls.data(), controls.size(),
                         kRgbOrder.data(), 0u, destination.samples.data(),
                         destination.samples.size(), destination.stride,
                         "zero height is rejected");
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::invalid_stride),
                         source.samples.data(), source.samples.size(), 5u,
                         1u, 2u, controls.data(), controls.size(),
                         kRgbOrder.data(), 0u, destination.samples.data(),
                         destination.samples.size(), destination.stride,
                         "HWC row stride below width times three is rejected");
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::invalid_controls),
                         source.samples.data(), source.samples.size(),
                         source.stride, 1u, 2u, controls.data(), 12u,
                         kRgbOrder.data(), 0u, destination.samples.data(),
                         destination.samples.size(), destination.stride,
                         "control count must be exactly thirteen");

    const std::array<std::uint32_t, 3> bad_order = {0u, 0u, 2u};
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::invalid_controls),
                         source.samples.data(), source.samples.size(),
                         source.stride, 1u, 2u, controls.data(), controls.size(),
                         bad_order.data(), 0u, destination.samples.data(),
                         destination.samples.size(), destination.stride,
                         "channel order must be a permutation");

    invalid_controls[0] = 0.5;
    invalid_controls[1] = 1.0;
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::invalid_controls),
                         source.samples.data(), source.samples.size(),
                         source.stride, 1u, 2u, invalid_controls.data(),
                         invalid_controls.size(), kRgbOrder.data(), 0u,
                         destination.samples.data(), destination.samples.size(),
                         destination.stride, "A must be at least B");
    invalid_controls = controls;
    invalid_controls[2] = 0.4;
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::invalid_controls),
                         source.samples.data(), source.samples.size(),
                         source.stride, 1u, 2u, invalid_controls.data(),
                         invalid_controls.size(), kRgbOrder.data(), 0u,
                         destination.samples.data(), destination.samples.size(),
                         destination.stride, "U must meet the finite lower bound");
    invalid_controls = controls;
    invalid_controls[3] = 1.0;
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::invalid_controls),
                         source.samples.data(), source.samples.size(),
                         source.stride, 1u, 2u, invalid_controls.data(),
                         invalid_controls.size(), kRgbOrder.data(), 0u,
                         destination.samples.data(), destination.samples.size(),
                         destination.stride, "black pedestal must be below one");
    invalid_controls = controls;
    invalid_controls[5] = std::numeric_limits<double>::infinity();
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::invalid_controls),
                         source.samples.data(), source.samples.size(),
                         source.stride, 1u, 2u, invalid_controls.data(),
                         invalid_controls.size(), kRgbOrder.data(), 0u,
                         destination.samples.data(), destination.samples.size(),
                         destination.stride, "non-finite matrix control is rejected");
    invalid_controls = controls;
    invalid_controls[4] = std::numeric_limits<double>::max();
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::invalid_controls),
                         source.samples.data(), source.samples.size(),
                         source.stride, 1u, 2u, invalid_controls.data(),
                         invalid_controls.size(), kRgbOrder.data(), 0u,
                         destination.samples.data(), destination.samples.size(),
                         destination.stride,
                         "finite double outside float range is rejected before conversion");

    expect_atomic_status(static_cast<int>(CombinedCameraStatus::buffer_too_small),
                         source.samples.data(), 5u, source.stride, 1u, 2u,
                         controls.data(), controls.size(), kRgbOrder.data(), 0u,
                         destination.samples.data(), destination.samples.size(),
                         destination.stride, "undersized HWC input is rejected");

    Image bad_source(1u, 2u, 0u);
    fill_source(&bad_source);
    bad_source.at(0u, 1u, 1u) = std::numeric_limits<float>::quiet_NaN();
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::non_finite_input),
                         bad_source.samples.data(), bad_source.samples.size(),
                         bad_source.stride, 1u, 2u, controls.data(),
                         controls.size(), kRgbOrder.data(), 0u,
                         destination.samples.data(), destination.samples.size(),
                         destination.stride, "active NaN input is rejected atomically");

    auto overflow_controls = make_controls(1.0f, 1.0f, 1.0f, 0.0f,
        {2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    std::vector<float> huge_source = {
        std::numeric_limits<float>::max(), 0.0f, 0.0f};
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::non_finite_intermediate),
                         huge_source.data(), huge_source.size(), 3u, 1u, 1u,
                         overflow_controls.data(), overflow_controls.size(),
                         kRgbOrder.data(), 1u, destination.samples.data(),
                         destination.samples.size(), destination.stride,
                         "finite input with overflowing matrix result is rejected");
}

void test_alignment_overlap_and_overflow() {
    const auto controls = make_controls(1.0f, 1.0f, 1.0f, 0.0f);
    std::vector<float> source = {0.125f, 0.25f, 0.5f};
    std::vector<float> destination(3u, 88.0f);

    alignas(double) unsigned char misaligned_bytes[64] = {};
    const auto *misaligned_source =
        reinterpret_cast<const float *>(misaligned_bytes + 1u);
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::unsupported_alignment),
                         misaligned_source, 3u, 3u, 1u, 1u, controls.data(),
                         controls.size(), kRgbOrder.data(), 0u,
                         destination.data(), destination.size(), 3u,
                         "misaligned source is rejected before access");

    std::vector<float> shared = {0.125f, 0.25f, 0.5f, 0.75f,
                                 1.0f, 1.25f, 1.5f, 1.75f};
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::overlap),
                         shared.data(), shared.size(), 3u, 1u, 1u,
                         controls.data(), controls.size(), kRgbOrder.data(), 0u,
                         shared.data() + 1u, shared.size() - 1u, 3u,
                         "partial source destination overlap is rejected");

    alignas(double) unsigned char metadata_source[128] = {};
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::overlap),
                         reinterpret_cast<const float *>(metadata_source), 3u,
                         3u, 1u, 1u,
                         reinterpret_cast<const double *>(metadata_source),
                         13u, kRgbOrder.data(), 0u, destination.data(),
                         destination.size(), 3u,
                         "controls may not overlap source storage");

    std::vector<float> metadata_destination(32u, 71.0f);
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::overlap),
                         source.data(), source.size(), 3u, 1u, 1u,
                         controls.data(), controls.size(),
                         reinterpret_cast<const std::uint32_t *>(
                             metadata_destination.data()),
                         0u, metadata_destination.data(),
                         metadata_destination.size(), 3u,
                         "order metadata may not overlap destination storage");

    alignas(double) unsigned char metadata_pair[128] = {};
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::overlap),
                         source.data(), source.size(), 3u, 1u, 1u,
                         reinterpret_cast<const double *>(metadata_pair), 13u,
                         reinterpret_cast<const std::uint32_t *>(metadata_pair),
                         0u, destination.data(), destination.size(), 3u,
                         "control and order metadata may not overlap each other");

    const std::size_t max = std::numeric_limits<std::size_t>::max();
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::arithmetic_overflow),
                         source.data(), max, 3u, 1u, 1u, controls.data(),
                         controls.size(), kRgbOrder.data(), 0u,
                         destination.data(), destination.size(), 3u,
                         "declared source span byte multiplication overflow is rejected");
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::arithmetic_overflow),
                         source.data(), source.size(), max, 2u, 1u,
                         controls.data(), controls.size(), kRgbOrder.data(), 0u,
                         destination.data(), destination.size(), 3u,
                         "row extent arithmetic overflow is rejected");
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::arithmetic_overflow),
                         source.data(), source.size(), max, 1u, max,
                         controls.data(), controls.size(), kRgbOrder.data(), 0u,
                         destination.data(), destination.size(), max,
                         "HWC width multiplication overflow is rejected");

    constexpr std::size_t resource_width = 128u * 1024u * 1024u / 3u + 1u;
    expect_atomic_status(static_cast<int>(CombinedCameraStatus::resource_limit),
                         source.data(), 0u, resource_width * 3u, 1u,
                         resource_width, controls.data(), controls.size(),
                         kRgbOrder.data(), 0u, destination.data(), 0u,
                         resource_width * 3u,
                         "active sample resource ceiling is enforced before access");
}

}  // namespace

int main() {
    test_reference_paths_and_widths();
    test_permutation_and_legacy_contrast_probe();
    test_flags_and_signed_zero();
    test_exact_in_place_permutation_and_padding();
    test_invalid_inputs_and_atomic_failures();
    test_alignment_overlap_and_overflow();
    if (failures != 0) {
        std::cerr << failures << " combined Camera validation checks failed\n";
        return 1;
    }
    std::cout << "combined Camera 013466b4 validation passed\n";
    return 0;
}
