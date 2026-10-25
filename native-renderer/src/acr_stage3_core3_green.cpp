#include "acr_stage3_core3_green.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace imprint::acr {
namespace {

// Volatile stores force each ordinary operation to round to binary32 and
// prevent accidental contraction. The recovered kernel uses FMA only at the
// explicitly marked sites below.
float add_f32(float left, float right) noexcept {
    volatile float rounded = left + right;
    return rounded;
}

float sub_f32(float left, float right) noexcept {
    volatile float rounded = left - right;
    return rounded;
}

float mul_f32(float left, float right) noexcept {
    volatile float rounded = left * right;
    return rounded;
}

float div_f32(float numerator, float denominator) noexcept {
    volatile float rounded = numerator / denominator;
    return rounded;
}

float float_from_bits(std::uint32_t bits) noexcept {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float clamp01(float value) noexcept {
    if (std::isnan(value)) return value;

    const float upper = value <= 1.0f ? value : 1.0f;
    return upper > 0.0f ? upper : 0.0f;
}

bool aligned_16(const void *pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0xFu) == 0;
}

enum class ShapeResult { valid, invalid, overflow };

ShapeResult valid_shape(std::size_t width, std::size_t stride,
                        std::uintmax_t guard) noexcept {
    if (width == 0 || (width & 3u) != 0 || stride < width ||
        (stride & 3u) != 0) {
        return ShapeResult::invalid;
    }

    // Keep both the sample offset and its byte distance within ptrdiff_t
    // before forming any row or guard pointer. This bounds
    // 3 * stride + width + guard without evaluating an overflowing sum.
    const auto max_samples = static_cast<std::uintmax_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) /
        sizeof(float);
    const auto width_samples = static_cast<std::uintmax_t>(width);
    const auto stride_samples = static_cast<std::uintmax_t>(stride);
    if (width_samples > max_samples || guard > max_samples - width_samples) {
        return ShapeResult::overflow;
    }
    const auto remaining = max_samples - width_samples - guard;
    if (stride_samples > remaining / 3u) return ShapeResult::overflow;
    return ShapeResult::valid;
}

float sample(const float *center, std::ptrdiff_t row,
             std::ptrdiff_t column,
             std::ptrdiff_t stride) noexcept {
    return center[row * stride + column];
}

float inverse(float value) noexcept {
    return value == 0.0f ? 0.0f : div_f32(1.0f, value);
}

float polynomial(float ratio) noexcept {
    const float a = float_from_bits(0x4273e173u);
    const float b = float_from_bits(0xc17f85ccu);
    const float fused = std::fma(ratio, b, a);
    return mul_f32(ratio, fused);
}

float angle_approximation(float a, float b) noexcept {
    if (a == 0.0f && b == 0.0f) return 0.0f;
    if (a <= b) {
        const float ratio = mul_f32(a, inverse(b));
        return sub_f32(90.0f, polynomial(ratio));
    }
    const float ratio = mul_f32(b, inverse(a));
    return polynomial(ratio);
}

float sum5(float a, float b, float c, float d, float e) noexcept {
    const float ab = add_f32(a, b);
    const float abc = add_f32(ab, c);
    const float abcd = add_f32(abc, d);
    return add_f32(abcd, e);
}

float sum10(float a, float b, float c, float d, float e, float f, float g,
            float h, float i, float j) noexcept {
    const float ab = add_f32(a, b);
    const float abc = add_f32(ab, c);
    const float abcd = add_f32(abc, d);
    const float abcde = add_f32(abcd, e);
    const float abcdef = add_f32(abcde, f);
    const float abcdefg = add_f32(abcdef, g);
    const float abcdefgh = add_f32(abcdefg, h);
    const float abcdefghi = add_f32(abcdefgh, i);
    return add_f32(abcdefghi, j);
}

}  // namespace

Core3GreenStatus core3_green_prepared_float_row_0080f2b8(
    const float *source_center,
    const float *plane0_center,
    const float *plane1_center,
    const float *plane2_center,
    const float *plane5_center,
    const float *plane7_center,
    const float *plane8_center,
    std::size_t source_stride,
    std::size_t workspace_stride,
    std::size_t width,
    std::uint32_t phase,
    Core3GreenParameters parameters,
    float *out_green,
    float *out_aux) noexcept {
    if (source_center == nullptr || plane0_center == nullptr ||
        plane1_center == nullptr || plane2_center == nullptr ||
        plane5_center == nullptr || plane7_center == nullptr ||
        plane8_center == nullptr || out_green == nullptr ||
        (parameters.m != 0.0f && out_aux == nullptr)) {
        return Core3GreenStatus::invalid_argument;
    }
    if (phase > 1u) return Core3GreenStatus::invalid_phase;
    if (!std::isfinite(parameters.t) || !std::isfinite(parameters.k) ||
        !std::isfinite(parameters.m) || !std::isfinite(parameters.q) ||
        parameters.q < 0.0f) {
        return Core3GreenStatus::invalid_parameters;
    }

    const ShapeResult source_shape =
        valid_shape(width, source_stride, 2u);
    const ShapeResult workspace_shape =
        valid_shape(width, workspace_stride, 3u);
    if (source_shape == ShapeResult::invalid ||
        workspace_shape == ShapeResult::invalid) {
        return Core3GreenStatus::invalid_dimensions;
    }
    if (source_shape == ShapeResult::overflow ||
        workspace_shape == ShapeResult::overflow) {
        return Core3GreenStatus::arithmetic_overflow;
    }

    if (!aligned_16(source_center) || !aligned_16(plane0_center) ||
        !aligned_16(plane1_center) || !aligned_16(plane2_center) ||
        !aligned_16(plane5_center) || !aligned_16(plane7_center) ||
        !aligned_16(plane8_center) || !aligned_16(out_green) ||
        (parameters.m != 0.0f && !aligned_16(out_aux))) {
        return Core3GreenStatus::unsupported_alignment;
    }

    const auto source_row_stride = static_cast<std::ptrdiff_t>(source_stride);
    const auto workspace_row_stride =
        static_cast<std::ptrdiff_t>(workspace_stride);
    const float high_limit = sub_f32(90.0f, parameters.t);
    const bool write_aux = parameters.m != 0.0f;

    for (std::size_t column = 0; column < width; ++column) {
        const auto x = static_cast<std::ptrdiff_t>(column);
        const float eh = sum5(
            sample(plane8_center, 0, x - 2, workspace_row_stride),
            sample(plane8_center, 0, x - 1, workspace_row_stride),
            sample(plane8_center, 0, x, workspace_row_stride),
            sample(plane8_center, 0, x + 1, workspace_row_stride),
            sample(plane8_center, 0, x + 2, workspace_row_stride));
        const float ev = sum5(
            sample(plane7_center, -2, x, workspace_row_stride),
            sample(plane7_center, -1, x, workspace_row_stride),
            sample(plane7_center, 0, x, workspace_row_stride),
            sample(plane7_center, 1, x, workspace_row_stride),
            sample(plane7_center, 2, x, workspace_row_stride));
        const float dv = sum10(
            sample(plane1_center, -3, x, workspace_row_stride),
            sample(plane1_center, -2, x - 1, workspace_row_stride),
            sample(plane1_center, -1, x, workspace_row_stride),
            sample(plane1_center, 0, x - 1, workspace_row_stride),
            sample(plane1_center, 1, x, workspace_row_stride),
            sample(plane1_center, 2, x - 1, workspace_row_stride),
            sample(plane1_center, 3, x, workspace_row_stride),
            sample(plane0_center, -2, x + 1, workspace_row_stride),
            sample(plane0_center, 0, x + 1, workspace_row_stride),
            sample(plane0_center, 2, x + 1, workspace_row_stride));
        const float dh = sum10(
            sample(plane5_center, -1, x - 2, workspace_row_stride),
            sample(plane5_center, -1, x, workspace_row_stride),
            sample(plane5_center, -1, x + 2, workspace_row_stride),
            sample(plane5_center, 0, x - 3, workspace_row_stride),
            sample(plane5_center, 0, x - 1, workspace_row_stride),
            sample(plane5_center, 0, x + 1, workspace_row_stride),
            sample(plane5_center, 0, x + 3, workspace_row_stride),
            sample(plane2_center, 1, x - 2, workspace_row_stride),
            sample(plane2_center, 1, x, workspace_row_stride),
            sample(plane2_center, 1, x + 2, workspace_row_stride));

        const float theta_e = angle_approximation(eh, ev);
        const float theta_d = angle_approximation(dh, dv);
        const float angle_sum = add_f32(theta_e, theta_d);
        const float angle_mean = mul_f32(angle_sum, 0.5f);
        const float centered = sub_f32(angle_mean, parameters.t);
        const float scaled_base = mul_f32(centered, parameters.k);
        const float base = clamp01(scaled_base);

        float weight = 0.5f;
        if (parameters.q == 0.0f) {
            weight = 0.5f;
        } else if (parameters.q < 0.5f) {
            const float twice_q = add_f32(parameters.q, parameters.q);
            const float inverse_twice_q = div_f32(1.0f, twice_q);
            const float one_minus_q = sub_f32(1.0f, parameters.q);
            if (base <= parameters.q) {
                weight = mul_f32(base, inverse_twice_q);
            } else if (base >= one_minus_q) {
                const float base_minus_one = sub_f32(base, 1.0f);
                weight = std::fma(base_minus_one, inverse_twice_q, 1.0f);
            } else {
                weight = 0.5f;
            }
        } else {
            weight = base;
        }

        const float source = source_center[x];
        const float twice_source = add_f32(source, source);

        const float h_outer = sub_f32(
            sub_f32(twice_source,
                    sample(source_center, 0, x - 2, source_row_stride)),
            sample(source_center, 0, x + 2, source_row_stride));
        const float h_neighbors = add_f32(
            sample(source_center, 0, x - 1, source_row_stride),
            sample(source_center, 0, x + 1, source_row_stride));
        const float h_quarter = std::fma(h_outer, 0.25f, h_neighbors);
        const float horizontal = clamp01(mul_f32(h_quarter, 0.5f));

        const float v_outer = sub_f32(
            sub_f32(twice_source,
                    sample(source_center, -2, x, source_row_stride)),
            sample(source_center, 2, x, source_row_stride));
        const float v_neighbors = add_f32(
            sample(source_center, -1, x, source_row_stride),
            sample(source_center, 1, x, source_row_stride));
        const float v_quarter = std::fma(v_outer, 0.25f, v_neighbors);
        const float vertical = clamp01(mul_f32(v_quarter, 0.5f));

        const float blend_delta = sub_f32(vertical, horizontal);
        const float blend_scaled = mul_f32(blend_delta, weight);
        float green = add_f32(horizontal, blend_scaled);

        const bool use_vertical_e = theta_e > high_limit;
        const bool use_horizontal_e = theta_e < parameters.t;
        const bool use_vertical_d = theta_d > high_limit;
        const bool use_horizontal_d = theta_d < parameters.t;
        if (use_vertical_e) green = vertical;
        if (use_horizontal_e) green = horizontal;
        if (use_vertical_d) green = vertical;
        if (use_horizontal_d) green = horizontal;

        const bool copy_source = phase == 0u ? (column & 1u) != 0u
                                             : (column & 1u) == 0u;
        out_green[column] = copy_source ? source : green;

        if (write_aux) {
            if (use_vertical_e || use_horizontal_e || use_vertical_d ||
                use_horizontal_d) {
                out_aux[column] = 0.0f;
            } else {
                const float twice_weight = mul_f32(weight, 2.0f);
                const float centered_weight = sub_f32(twice_weight, 1.0f);
                const float weight_distance = std::fabs(centered_weight);
                const float inverse_distance = sub_f32(1.0f, weight_distance);
                const float twice_inverse_distance =
                    mul_f32(inverse_distance, 2.0f);
                const float factor = clamp01(twice_inverse_distance);
                out_aux[column] = mul_f32(factor, parameters.m);
            }
        }
    }

    return Core3GreenStatus::ok;
}

}  // namespace imprint::acr
