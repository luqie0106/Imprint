#include "acr_camera_combined.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kActiveSampleLimit = 128u * 1024u * 1024u;
constexpr std::size_t kControlCount = 13u;

static_assert(sizeof(float) == sizeof(std::uint32_t) &&
              std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24);

struct AddressSpan {
    std::uintptr_t begin;
    std::uintptr_t end;
};

struct KernelConstants {
    float a;
    float b_gain;
    float upper;
    float black;
    float matrix[9];
    float ratio;
    float reciprocal_a;
    float q0;
    float q1;
    float q2;
    float inverse0;
    float inverse1;
    float inverse2;
    float delta;
    float inverse_black_delta;
    float black_bias;
    float output_black;
    float pedestal_matrix[9];
    float ratio_inverse_black;
    float reciprocal_inverse_black;
    float ratio_bias;
    float reciprocal_bias;
    bool direct;
    bool pedestal;
    bool clamp_output;
};

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t *result) noexcept {
    if (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    *result = left * right;
    return true;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t *result) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    *result = left + right;
    return true;
}

bool make_span(const void *pointer, std::size_t count, std::size_t element_size,
               AddressSpan *span) noexcept {
    std::size_t byte_count = 0u;
    if (!checked_multiply(count, element_size, &byte_count) ||
        byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::ptrdiff_t>::max())) {
        return false;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (byte_count > std::numeric_limits<std::uintptr_t>::max() - begin)
        return false;
    span->begin = begin;
    span->end = begin + byte_count;
    return true;
}

bool spans_overlap(const AddressSpan &left, const AddressSpan &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

float rounded_add(float left, float right) noexcept {
    volatile float result = left + right;
    return result;
}

float rounded_subtract(float left, float right) noexcept {
    volatile float result = left - right;
    return result;
}

float rounded_multiply(float left, float right) noexcept {
    volatile float result = left * right;
    return result;
}

float rounded_divide(float left, float right) noexcept {
    volatile float result = left / right;
    return result;
}

float rounded_fma(float left, float right, float addend) noexcept {
    return std::fma(left, right, addend);
}

bool finite(float value) noexcept { return std::isfinite(value); }

// FMAXNM(x, +0) over the finite supported domain, including the +0 tie.
float max_number_with_zero(float value) noexcept {
    return value > 0.0f ? value : 0.0f;
}

// FCMP/FCSEL GT: equality chooses the second operand.
float max_greater_than(float first, float second) noexcept {
    return first > second ? first : second;
}

float clamp_output(float value) noexcept {
    // The inputs have already been checked finite. This preserves -0 because
    // the final comparison is strict, matching FMINNM followed by FCSEL MI.
    const float upper_limited = value > 1.0f ? 1.0f : value;
    return upper_limited < 0.0f ? 0.0f : upper_limited;
}

bool build_constants(const float controls[kControlCount], std::uint32_t flags,
                     KernelConstants *out) noexcept {
    KernelConstants candidate{};
    candidate.a = controls[0];
    candidate.b_gain = controls[1];
    candidate.upper = controls[2];
    candidate.black = controls[3];
    for (std::size_t i = 0; i < 9u; ++i) candidate.matrix[i] = controls[i + 4u];

    candidate.ratio = rounded_divide(candidate.b_gain, candidate.a);
    candidate.reciprocal_a = rounded_divide(1.0f, candidate.a);
    candidate.q0 = rounded_multiply(candidate.upper, 0.25f);
    candidate.q1 = rounded_multiply(candidate.ratio, candidate.q0);
    candidate.q2 = rounded_multiply(candidate.reciprocal_a, candidate.q0);
    const float denominator0 = rounded_subtract(1.0f, candidate.q0);
    const float denominator1 = rounded_subtract(candidate.ratio, candidate.q1);
    const float denominator2 = rounded_subtract(candidate.reciprocal_a, candidate.q2);
    candidate.inverse0 = rounded_divide(1.0f, denominator0);
    candidate.inverse1 = rounded_divide(1.0f, denominator1);
    candidate.inverse2 = rounded_divide(1.0f, denominator2);
    candidate.delta = rounded_add(candidate.black, -1.0f);
    candidate.inverse_black_delta = candidate.black == 1.0f
                                        ? 0.0f
                                        : rounded_divide(-1.0f, candidate.delta);
    candidate.direct = candidate.reciprocal_a >= 1.0f;
    candidate.pedestal = candidate.black != 0.0f;
    candidate.clamp_output = (flags & 1u) == 0u;

    const float prologue_values[] = {
        candidate.ratio, candidate.reciprocal_a, candidate.q0, candidate.q1,
        candidate.q2, denominator0, denominator1, denominator2,
        candidate.inverse0, candidate.inverse1, candidate.inverse2,
        candidate.delta, candidate.inverse_black_delta};
    for (float value : prologue_values) {
        if (!finite(value)) return false;
    }

    if (candidate.pedestal) {
        // Preserve the original scalar instruction sequence and operation
        // direction: +0 - inv*b, then d*(+0)+b. Matrix scaling is an
        // independent rounded multiply followed by sign inversion.
        candidate.black_bias = rounded_fma(-candidate.inverse_black_delta,
                                           candidate.black, 0.0f);
        candidate.output_black = rounded_fma(candidate.delta, 0.0f,
                                             candidate.black);
        for (std::size_t i = 0; i < 9u; ++i) {
            candidate.pedestal_matrix[i] =
                -rounded_multiply(candidate.delta, candidate.matrix[i]);
        }
        candidate.ratio_inverse_black = rounded_multiply(
            candidate.ratio, candidate.inverse_black_delta);
        candidate.reciprocal_inverse_black = rounded_multiply(
            candidate.reciprocal_a, candidate.inverse_black_delta);
        candidate.ratio_bias = rounded_multiply(candidate.ratio,
                                                candidate.black_bias);
        candidate.reciprocal_bias = rounded_multiply(
            candidate.reciprocal_a, candidate.black_bias);
        const float pedestal_values[] = {
            candidate.black_bias, candidate.output_black,
            candidate.ratio_inverse_black,
            candidate.reciprocal_inverse_black, candidate.ratio_bias,
            candidate.reciprocal_bias};
        for (float value : pedestal_values) {
            if (!finite(value)) return false;
        }
        for (float value : candidate.pedestal_matrix) {
            if (!finite(value)) return false;
        }
    }

    *out = candidate;
    return true;
}

bool smooth_highlights(float *z0, float *z1, float *z2,
                       const KernelConstants &kernel) noexcept {
    const float head = max_greater_than(*z0, *z1);

    float amount = rounded_multiply(rounded_subtract(*z0, kernel.q0),
                                    kernel.inverse0);
    if (!finite(amount)) return false;
    amount = max_number_with_zero(amount);
    amount = rounded_multiply(amount, amount);
    if (!finite(amount)) return false;
    float delta = rounded_subtract(head, *z0);
    if (!finite(delta)) return false;
    *z0 = rounded_fma(amount, delta, *z0);
    if (!finite(*z0)) return false;

    amount = rounded_multiply(rounded_subtract(*z1, kernel.q1),
                              kernel.inverse1);
    if (!finite(amount)) return false;
    amount = max_number_with_zero(amount);
    amount = rounded_multiply(amount, amount);
    if (!finite(amount)) return false;
    delta = rounded_subtract(head, *z1);
    if (!finite(delta)) return false;
    *z1 = rounded_fma(amount, delta, *z1);
    if (!finite(*z1)) return false;

    const float sum = rounded_add(*z0, *z1);
    if (!finite(sum)) return false;
    const float average = rounded_multiply(sum, 0.5f);
    if (!finite(average)) return false;
    const float head2 = max_greater_than(average, *z2);

    amount = rounded_multiply(rounded_subtract(*z2, kernel.q2),
                              kernel.inverse2);
    if (!finite(amount)) return false;
    amount = max_number_with_zero(amount);
    amount = rounded_multiply(amount, amount);
    if (!finite(amount)) return false;
    delta = rounded_subtract(head2, *z2);
    if (!finite(delta)) return false;
    *z2 = rounded_fma(amount, delta, *z2);
    return finite(*z2);
}

bool matrix_row(const float matrix[9], std::size_t row, float z0, float z1,
                float z2, float *result) noexcept {
    const std::size_t base = row * 3u;
    float value = rounded_multiply(matrix[base + 1u], z1);
    if (!finite(value)) return false;
    value = rounded_fma(z0, matrix[base], value);
    if (!finite(value)) return false;
    value = rounded_fma(z2, matrix[base + 2u], value);
    if (!finite(value)) return false;
    *result = value;
    return true;
}

bool process_pixel(float input0, float input1, float input2,
                   const KernelConstants &kernel, float output[3]) noexcept {
    float z0 = input0;
    float z1 = 0.0f;
    float z2 = 0.0f;

    if (kernel.pedestal) {
        z0 = rounded_fma(kernel.inverse_black_delta, input0,
                         kernel.black_bias);
        z1 = rounded_fma(kernel.ratio_inverse_black, input1,
                         kernel.ratio_bias);
        z2 = rounded_fma(kernel.reciprocal_inverse_black, input2,
                         kernel.reciprocal_bias);
    } else {
        z1 = rounded_multiply(kernel.ratio, input1);
        z2 = rounded_multiply(kernel.reciprocal_a, input2);
    }
    if (!finite(z0) || !finite(z1) || !finite(z2)) return false;

    if (!kernel.direct && !smooth_highlights(&z0, &z1, &z2, kernel))
        return false;

    for (std::size_t row = 0; row < 3u; ++row) {
        const float *matrix = kernel.pedestal ? kernel.pedestal_matrix
                                              : kernel.matrix;
        if (!matrix_row(matrix, row, z0, z1, z2, &output[row])) return false;
        if (kernel.pedestal) {
            output[row] = rounded_add(kernel.output_black, output[row]);
            if (!finite(output[row])) return false;
        }
        if (kernel.clamp_output) output[row] = clamp_output(output[row]);
    }
    return true;
}

}  // namespace

CombinedCameraStatus apply_camera_combined_prepared_f32_to_buffer(
    const float *src, std::size_t srcCount, std::size_t srcRowStride,
    std::size_t height, std::size_t width, const double *controls,
    std::size_t controlCount, const std::uint32_t order3[3],
    std::uint32_t flags, float *dst, std::size_t dstCount,
    std::size_t dstRowStride) noexcept {
    using Status = CombinedCameraStatus;

    if (src == nullptr || controls == nullptr || order3 == nullptr || dst == nullptr)
        return Status::invalid_argument;
    if (height == 0u || width == 0u) return Status::invalid_dimensions;

    std::size_t active_row_samples = 0u;
    if (!checked_multiply(width, 3u, &active_row_samples))
        return Status::arithmetic_overflow;
    if (srcRowStride < active_row_samples ||
        dstRowStride < active_row_samples)
        return Status::invalid_stride;
    if (controlCount != kControlCount) return Status::invalid_controls;

    std::size_t pixel_count = 0u;
    std::size_t active_sample_count = 0u;
    std::size_t last_row_src = 0u;
    std::size_t last_row_dst = 0u;
    std::size_t src_required = 0u;
    std::size_t dst_required = 0u;
    if (!checked_multiply(height, width, &pixel_count) ||
        !checked_multiply(pixel_count, 3u, &active_sample_count) ||
        !checked_multiply(height - 1u, srcRowStride, &last_row_src) ||
        !checked_multiply(height - 1u, dstRowStride, &last_row_dst) ||
        !checked_add(last_row_src, active_row_samples, &src_required) ||
        !checked_add(last_row_dst, active_row_samples, &dst_required)) {
        return Status::arithmetic_overflow;
    }
    const std::size_t ptrdiff_sample_limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
        sizeof(float);
    if (src_required > ptrdiff_sample_limit ||
        dst_required > ptrdiff_sample_limit) {
        return Status::arithmetic_overflow;
    }

    AddressSpan src_span{};
    AddressSpan dst_span{};
    AddressSpan controls_span{};
    AddressSpan order_span{};
    if (!make_span(src, srcCount, sizeof(float), &src_span) ||
        !make_span(dst, dstCount, sizeof(float), &dst_span) ||
        !make_span(controls, kControlCount, sizeof(double), &controls_span) ||
        !make_span(order3, 3u, sizeof(std::uint32_t), &order_span)) {
        return Status::arithmetic_overflow;
    }
    if (active_sample_count > kActiveSampleLimit)
        return Status::resource_limit;
    if (srcCount < src_required || dstCount < dst_required)
        return Status::buffer_too_small;

    if (reinterpret_cast<std::uintptr_t>(src) % alignof(float) != 0u ||
        reinterpret_cast<std::uintptr_t>(dst) % alignof(float) != 0u ||
        reinterpret_cast<std::uintptr_t>(controls) % alignof(double) != 0u ||
        reinterpret_cast<std::uintptr_t>(order3) % alignof(std::uint32_t) != 0u) {
        return Status::unsupported_alignment;
    }

    const bool source_destination_overlap = spans_overlap(src_span, dst_span);
    const bool exact_in_place = src == dst && srcRowStride == dstRowStride &&
                                srcCount == dstCount;
    if ((source_destination_overlap && !exact_in_place) ||
        spans_overlap(src_span, controls_span) ||
        spans_overlap(src_span, order_span) ||
        spans_overlap(dst_span, controls_span) ||
        spans_overlap(dst_span, order_span) ||
        spans_overlap(controls_span, order_span)) {
        return Status::overlap;
    }

    std::uint32_t input_order[3] = {order3[0], order3[1], order3[2]};
    if (input_order[0] > 2u || input_order[1] > 2u || input_order[2] > 2u ||
        input_order[0] == input_order[1] ||
        input_order[0] == input_order[2] ||
        input_order[1] == input_order[2]) {
        return Status::invalid_controls;
    }

    float control_values[kControlCount];
    for (std::size_t i = 0; i < kControlCount; ++i) {
        if (!std::isfinite(controls[i]) ||
            controls[i] > static_cast<double>(
                              std::numeric_limits<float>::max()) ||
            controls[i] < -static_cast<double>(
                               std::numeric_limits<float>::max())) {
            return Status::invalid_controls;
        }
        control_values[i] = static_cast<float>(controls[i]);
        if (!finite(control_values[i])) return Status::invalid_controls;
    }
    const float minimum_upper = static_cast<float>(32768.0 / 65535.0);
    if (!(control_values[0] >= control_values[1]) ||
        !(control_values[1] >= 1.0f) ||
        !(control_values[2] >= minimum_upper) ||
        !(control_values[2] <= 1.0f) ||
        !(control_values[3] >= 0.0f) ||
        !(control_values[3] < 1.0f)) {
        return Status::invalid_controls;
    }

    for (std::size_t y = 0; y < height; ++y) {
        const std::size_t row_base = y * srcRowStride;
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t pixel_base = row_base + x * 3u;
            for (std::size_t channel = 0; channel < 3u; ++channel) {
                if (!finite(src[pixel_base + input_order[channel]]))
                    return Status::non_finite_input;
            }
        }
    }

    KernelConstants kernel{};
    if (!build_constants(control_values, flags, &kernel))
        return Status::non_finite_intermediate;

    std::vector<float> candidate;
    try {
        candidate.resize(active_sample_count);
    } catch (const std::bad_alloc &) {
        return Status::allocation_failure;
    } catch (const std::length_error &) {
        return Status::length_error;
    } catch (...) {
        return Status::allocation_failure;
    }

    for (std::size_t y = 0; y < height; ++y) {
        const std::size_t src_row = y * srcRowStride;
        const std::size_t dst_candidate_row = y * active_row_samples;
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t src_pixel = src_row + x * 3u;
            const float input0 = src[src_pixel + input_order[0]];
            const float input1 = src[src_pixel + input_order[1]];
            const float input2 = src[src_pixel + input_order[2]];
            float output[3];
            if (!process_pixel(input0, input1, input2, kernel, output))
                return Status::non_finite_intermediate;
            const std::size_t pixel = dst_candidate_row + x * 3u;
            for (std::size_t channel = 0; channel < 3u; ++channel) {
                candidate[pixel + channel] = output[channel];
            }
        }
    }

    for (std::size_t y = 0; y < height; ++y) {
        const std::size_t destination_row = y * dstRowStride;
        const std::size_t candidate_row = y * active_row_samples;
        for (std::size_t sample = 0; sample < active_row_samples; ++sample) {
            dst[destination_row + sample] = candidate[candidate_row + sample];
        }
    }
    return Status::ok;
}

}  // namespace imprint::acr
