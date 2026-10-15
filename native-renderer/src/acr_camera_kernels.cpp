#include "acr_camera_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {
constexpr float kOne = 1.0f;
constexpr float kTwo = 2.0f;
constexpr float kHalf = 0.5f;
constexpr float kOneThird = 0.33333334f;
constexpr float kMinimumUpper = 32768.0f / 65535.0f;

float checked_add(float left, float right) {
    const float result = left + right;
    if (!std::isfinite(result)) throw std::overflow_error("camera kernel float addition overflow");
    return result;
}

float checked_subtract(float left, float right) {
    const float result = left - right;
    if (!std::isfinite(result)) throw std::overflow_error("camera kernel float subtraction overflow");
    return result;
}

float checked_multiply(float left, float right) {
    const float result = left * right;
    if (!std::isfinite(result)) throw std::overflow_error("camera kernel float multiplication overflow");
    return result;
}

void validate_params(const CameraFloatParams &params) {
    if (!std::isfinite(params.gain_max) || !std::isfinite(params.gain_middle) ||
        !std::isfinite(params.upper) || !std::isfinite(params.scale)) {
        throw std::invalid_argument("camera kernel parameters must be finite");
    }
    if (params.gain_max < params.gain_middle || params.gain_middle < kOne) {
        throw std::invalid_argument("camera gains must satisfy A >= B >= 1");
    }
    if (params.upper < kMinimumUpper || params.upper > kOne) {
        throw std::invalid_argument("camera upper threshold is outside [32768/65535, 1]");
    }
    if (params.scale <= 0.0f) {
        throw std::invalid_argument("camera scale must be positive");
    }
    for (const float coefficient : params.matrix) {
        if (!std::isfinite(coefficient)) {
            throw std::invalid_argument("camera matrix coefficients must be finite");
        }
    }
}

// Preserve the scalar pseudo-C ordering: column 2 term, then column 1 term,
// then column 3 term. Coefficients are supplied row-major in the public API.
float matrix_row(const std::array<float, 3> &values, const float *row) {
    const float column_1 = checked_multiply(values[1], row[1]);
    const float column_0 = checked_multiply(row[0], values[0]);
    const float first_sum = checked_add(column_1, column_0);
    const float column_2 = checked_multiply(row[2], values[2]);
    return checked_add(first_sum, column_2);
}

std::array<float, 3> matrix(const std::array<float, 3> &values,
                            const CameraFloatParams &params) {
    return {matrix_row(values, params.matrix.data()),
            matrix_row(values, params.matrix.data() + 3),
            matrix_row(values, params.matrix.data() + 6)};
}

float residual_x(float x, float reciprocal_scale, float upper, float gain_middle,
                 float lower_third, float middle_origin) {
    if (x <= upper) return 0.0f;
    const float value = std::min(x, reciprocal_scale);
    if (value <= gain_middle) {
        if (value <= kOne) {
            return checked_multiply(checked_subtract(value, upper), kOneThird);
        }
        const float offset = checked_multiply(checked_subtract(value, kOne), kHalf);
        return checked_add(lower_third, offset);
    }
    return checked_add(middle_origin, checked_subtract(value, gain_middle));
}

float residual_y(float y, float reciprocal_scale, float upper, float lower_third) {
    if (y <= upper) return 0.0f;
    const float value = std::min(y, reciprocal_scale);
    if (value <= kOne) {
        return checked_multiply(checked_subtract(value, upper), kOneThird);
    }
    const float offset = checked_multiply(checked_subtract(value, kOne), kHalf);
    return checked_add(lower_third, offset);
}

float residual_z(float z, float upper) {
    if (z <= upper) return 0.0f;
    return checked_multiply(checked_subtract(z, upper), kOneThird);
}

float clamp_lower_zero_upper(float value, float upper) {
    return std::max(0.0f, std::min(value, upper));
}
}  // namespace

void apply_camera_float(const float *sorted_input_rgb, float *output,
                        std::size_t pixels, const CameraFloatParams &params) {
    validate_params(params);
    if (pixels > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::length_error("camera image extent overflows size_t");
    }
    const std::size_t sample_count = pixels * 3;
    if (sample_count > std::vector<float>().max_size()) {
        throw std::length_error("camera image extent exceeds vector capacity");
    }
    if (pixels == 0) return;
    if (!sorted_input_rgb || !output) {
        throw std::invalid_argument("camera kernel input and output must be non-null");
    }
    for (std::size_t i = 0; i < sample_count; ++i) {
        if (!std::isfinite(sorted_input_rgb[i]) || sorted_input_rgb[i] < 0.0f) {
            throw std::invalid_argument("camera input must be finite and nonnegative");
        }
    }

    // Keep all results private until every pixel has passed validation. This
    // also permits input/output aliasing when both buffers have enough space.
    std::vector<float> temporary(sample_count);
    const float upper = params.upper;
    const float scale = params.scale;
    const bool unit_special_case = upper == kOne && scale == kOne;

    float reciprocal_scale = 0.0f;
    float scale_over_upper = 0.0f;
    float lower_third = 0.0f;
    float middle_origin = 0.0f;
    if (!unit_special_case) {
        reciprocal_scale = 1.0f / scale;
        scale_over_upper = scale / upper;
        if (!std::isfinite(reciprocal_scale) || !std::isfinite(scale_over_upper)) {
            throw std::overflow_error("camera scale ratios overflow float");
        }
        lower_third = checked_multiply(checked_subtract(kOne, upper), kOneThird);
        const float middle_offset = checked_multiply(
            checked_subtract(params.gain_middle, kOne), kHalf);
        middle_origin = checked_add(lower_third, middle_offset);
    }

    for (std::size_t i = 0; i < pixels; ++i) {
        const float x = checked_multiply(sorted_input_rgb[3 * i], params.gain_max);
        const float y = checked_multiply(sorted_input_rgb[3 * i + 1], params.gain_middle);
        const float z = sorted_input_rgb[3 * i + 2];

        if (unit_special_case) {
            const std::array<float, 3> values{
                std::min(x, kOne), std::min(y, kOne), z};
            const auto mixed = matrix(values, params);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                temporary[3 * i + channel] = clamp_lower_zero_upper(mixed[channel], kOne);
            }
            continue;
        }

        const float rx = residual_x(x, reciprocal_scale, upper, params.gain_middle,
                                    lower_third, middle_origin);
        const float ry = residual_y(y, reciprocal_scale, upper, lower_third);
        const float rz = residual_z(z, upper);
        const float residual = checked_add(checked_add(rx, ry), rz);

        const std::array<float, 3> clipped{
            std::min(x, upper), std::min(y, upper), std::min(z, upper)};
        const auto mixed = matrix(clipped, params);
        bool residual_path = x > upper || y > upper || z > upper;
        std::array<float, 3> value{};
        const float gain_adjustment = checked_multiply(scale_over_upper, residual);
        const float gain = checked_add(scale, gain_adjustment);
        for (std::size_t channel = 0; channel < 3; ++channel) {
            if (mixed[channel] > upper) residual_path = true;
            const float q = clamp_lower_zero_upper(mixed[channel], upper);
            const float scaled = checked_multiply(gain, q);
            value[channel] = std::min(scaled, kOne);
        }

        if (residual_path) {
            const std::array<float, 3> residual_values{
                std::min(x, kTwo), std::min(y, kTwo), z};
            const auto residual_mixed = matrix(residual_values, params);
            const std::array<float, 3> adjusted{
                checked_multiply(residual_mixed[0], scale),
                checked_multiply(residual_mixed[1], scale),
                checked_multiply(residual_mixed[2], scale)};
            const float difference = std::max(
                std::max(std::fabs(value[0] - value[1]),
                         std::fabs(value[0] - value[2])),
                std::fabs(value[1] - value[2]));
            for (std::size_t channel = 0; channel < 3; ++channel) {
                if (adjusted[channel] > value[channel]) {
                    const float value_plus_difference = checked_add(value[channel], difference);
                    value[channel] = std::min(
                        adjusted[channel], std::min(value_plus_difference, kOne));
                }
            }
        }

        for (std::size_t channel = 0; channel < 3; ++channel) {
            if (!std::isfinite(value[channel])) {
                throw std::overflow_error("camera kernel output is non-finite");
            }
            temporary[3 * i + channel] = value[channel];
        }
    }

    std::copy(temporary.begin(), temporary.end(), output);
}

}  // namespace imprint::acr
