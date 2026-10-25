#include "acr_stage3_core3_refine.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace imprint::acr {
namespace {

static_assert(sizeof(float) == sizeof(std::uint32_t) &&
                  std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "core3 prepared-float rows require IEEE-754 binary32 float");

// Volatile temporaries make each non-FMA operation round to binary32 and
// prevent contraction even if the translation unit is built without the
// expected strict floating-point flags.
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

float inverse(float value) noexcept {
    return value == 0.0f ? 0.0f : div_f32(1.0f, value);
}

float fmin_propagating_nan(float left, float right) noexcept {
    if (std::isnan(left)) return left;
    if (std::isnan(right)) return right;
    return std::fmin(left, right);
}

float fmax_propagating_nan(float left, float right) noexcept {
    if (std::isnan(left)) return left;
    if (std::isnan(right)) return right;
    return std::fmax(left, right);
}

float clamp01(float value) noexcept {
    const float inner = fmin_propagating_nan(value, 1.0f);
    return fmax_propagating_nan(0.0f, inner);
}

bool aligned_16(const void *pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0xFu) == 0;
}

Core3RefineStatus validate(const RefineInputs &inputs,
                           const RefineOutputs &outputs) noexcept {
    if (inputs.guide == nullptr || inputs.channel_a == nullptr ||
        inputs.channel_b == nullptr || inputs.mask == nullptr ||
        outputs.guide == nullptr || outputs.a == nullptr ||
        outputs.b == nullptr) {
        return Core3RefineStatus::invalid_argument;
    }
    if (inputs.phase > 1u) return Core3RefineStatus::invalid_phase;
    if (!std::isfinite(inputs.lower) || !std::isfinite(inputs.upper) ||
        inputs.lower < 0.0f || inputs.upper < 0.0f) {
        return Core3RefineStatus::invalid_parameters;
    }
    if (inputs.width == 0 || (inputs.width & 3u) != 0 ||
        inputs.stride < inputs.width || (inputs.stride & 3u) != 0) {
        return Core3RefineStatus::invalid_dimensions;
    }

    // Every read is within +/-2 rows and +/-2 columns. Bound both sample
    // offsets and their byte distances before converting dimensions or making
    // any neighborhood pointer. This also protects the signed row*stride
    // calculations performed in sample().
    const auto max_samples = static_cast<std::uintmax_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) /
        sizeof(float);
    constexpr std::uintmax_t guard = 2;
    const auto width = static_cast<std::uintmax_t>(inputs.width);
    const auto stride = static_cast<std::uintmax_t>(inputs.stride);
    if (width > max_samples || guard > max_samples - width) {
        return Core3RefineStatus::arithmetic_overflow;
    }
    const auto remaining = max_samples - width - guard;
    if (stride > remaining / 2u) {
        return Core3RefineStatus::arithmetic_overflow;
    }

    if (!aligned_16(inputs.guide) || !aligned_16(inputs.channel_a) ||
        !aligned_16(inputs.channel_b) || !aligned_16(inputs.mask) ||
        !aligned_16(outputs.guide) || !aligned_16(outputs.a) ||
        !aligned_16(outputs.b)) {
        return Core3RefineStatus::unsupported_alignment;
    }
    return Core3RefineStatus::ok;
}

float sample(const float *center, std::ptrdiff_t row,
             std::ptrdiff_t column,
             std::ptrdiff_t stride) noexcept {
    return center[row * stride + column];
}

float abs_difference(float left, float right) noexcept {
    return std::fabs(sub_f32(left, right));
}

float candidate(const RefineInputs &inputs, const float *channel,
                std::ptrdiff_t x, float mix_mask) noexcept {
    constexpr std::ptrdiff_t center_row = 0;
    const auto stride = static_cast<std::ptrdiff_t>(inputs.stride);
    const float epsilon = float_from_bits(0x37800080u);

    const float channel_up = sample(channel, -1, x, stride);
    const float channel_down = sample(channel, 1, x, stride);
    const float channel_left = sample(channel, center_row, x - 1, stride);
    const float channel_right = sample(channel, center_row, x + 1, stride);
    const float vertical_base =
        add_f32(abs_difference(channel_up, channel_down), epsilon);
    const float horizontal_base =
        add_f32(abs_difference(channel_left, channel_right), epsilon);

    const float guide_center = sample(inputs.guide, 0, x, stride);
    const float weight_up = inverse(add_f32(
        abs_difference(guide_center, sample(inputs.guide, -2, x, stride)),
        vertical_base));
    const float weight_down = inverse(add_f32(
        abs_difference(guide_center, sample(inputs.guide, 2, x, stride)),
        vertical_base));
    const float weight_left = inverse(add_f32(
        abs_difference(guide_center, sample(inputs.guide, 0, x - 2, stride)),
        horizontal_base));
    const float weight_right = inverse(add_f32(
        abs_difference(guide_center, sample(inputs.guide, 0, x + 2, stride)),
        horizontal_base));

    const float denominator = add_f32(
        weight_up,
        add_f32(weight_down, add_f32(weight_left, weight_right)));

    const float term_up = mul_f32(
        sub_f32(sample(channel, -1, x, stride),
                sample(inputs.guide, -1, x, stride)),
        weight_up);
    const float term_down = mul_f32(
        sub_f32(sample(channel, 1, x, stride),
                sample(inputs.guide, 1, x, stride)),
        weight_down);
    const float term_left = mul_f32(
        sub_f32(sample(channel, 0, x - 1, stride),
                sample(inputs.guide, 0, x - 1, stride)),
        weight_left);
    const float term_right = mul_f32(
        sub_f32(sample(channel, 0, x + 1, stride),
                sample(inputs.guide, 0, x + 1, stride)),
        weight_right);
    const float numerator = add_f32(
        term_up,
        add_f32(term_down, add_f32(term_left, term_right)));

    const float inverse_denominator = inverse(denominator);
    const float filtered =
        std::fma(inverse_denominator, numerator, guide_center);

    const float channel_center = sample(channel, 0, x, stride);
    const float low = sub_f32(channel_center, inputs.lower);
    const float high = add_f32(channel_center, inputs.upper);
    const float bounded = fmax_propagating_nan(
        low, fmin_propagating_nan(high, filtered));
    const float delta = sub_f32(bounded, channel_center);
    const float mixed = std::fma(mix_mask, delta, channel_center);
    return clamp01(mixed);
}

void copy_sample_bits(float *output, const float *input) noexcept {
    std::memcpy(output, input, sizeof(float));
}

}  // namespace

Core3RefineStatus core3_refine_prepared_float_row_00814230(
    const RefineInputs &inputs, const RefineOutputs &outputs) noexcept {
    const Core3RefineStatus status = validate(inputs, outputs);
    if (status != Core3RefineStatus::ok) return status;

    const auto width = inputs.width;
    std::memcpy(outputs.guide, inputs.guide, width * sizeof(float));
    for (std::size_t column = 0; column < width; ++column) {
        const bool refine = (column & 1u) == (inputs.phase == 0u ? 0u : 1u);
        const auto x = static_cast<std::ptrdiff_t>(column);
        if (refine) {
            const float mask = sample(
                inputs.mask, 0, x,
                static_cast<std::ptrdiff_t>(inputs.stride));
            outputs.a[column] = candidate(inputs, inputs.channel_a, x, mask);
            outputs.b[column] = candidate(inputs, inputs.channel_b, x, mask);
        } else {
            copy_sample_bits(outputs.a + column, inputs.channel_a + column);
            copy_sample_bits(outputs.b + column, inputs.channel_b + column);
        }
    }
    return Core3RefineStatus::ok;
}

Core3RefineStatus core3_refine_prepared_float_row_008145b4(
    const RefineInputs &inputs, const RefineOutputs &outputs) noexcept {
    const Core3RefineStatus status = validate(inputs, outputs);
    if (status != Core3RefineStatus::ok) return status;

    const auto width = inputs.width;
    const auto stride = static_cast<std::ptrdiff_t>(inputs.stride);
    std::memcpy(outputs.guide, inputs.guide, width * sizeof(float));
    for (std::size_t column = 0; column < width; ++column) {
        const bool refine = (column & 1u) == (inputs.phase == 0u ? 1u : 0u);
        const auto x = static_cast<std::ptrdiff_t>(column);
        if (refine) {
            const float mask_up = sample(inputs.mask, -1, x, stride);
            const float mask_left = sample(inputs.mask, 0, x - 1, stride);
            const float mask_right = sample(inputs.mask, 0, x + 1, stride);
            const float mask_down = sample(inputs.mask, 1, x, stride);
            const float right_down = add_f32(mask_right, mask_down);
            const float left_and_rest = add_f32(mask_left, right_down);
            const float mask_sum = add_f32(mask_up, left_and_rest);
            const float mean = mul_f32(0.25f, mask_sum);
            outputs.a[column] = candidate(inputs, inputs.channel_a, x, mean);
            outputs.b[column] = candidate(inputs, inputs.channel_b, x, mean);
        } else {
            copy_sample_bits(outputs.a + column, inputs.channel_a + column);
            copy_sample_bits(outputs.b + column, inputs.channel_b + column);
        }
    }
    return Core3RefineStatus::ok;
}

}  // namespace imprint::acr
