#include "acr_resample_coefficients.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <new>

namespace imprint::acr {
namespace {

constexpr double kOneSixth = 0x1.5555555555555p-3;
// Binary64 bits 0x3fefff2e48e8a71e, loaded by FUN_012e3a20.
constexpr double kTapCountBias = 0x1.fff2e48e8a71ep-1;

static_assert(sizeof(double) == 8 &&
                  std::numeric_limits<double>::is_iec559,
              "the recovered coefficient generator requires IEEE-754 binary64");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "the recovered coefficient rows require IEEE-754 binary32");

}  // namespace

namespace resample_detail {

double evaluate_resample_weight(double parameter_a, double parameter_b,
                                double x) noexcept {
    const double u = x < 0.0 ? -x : x;
    if (u >= 2.0) return 0.0;

    if (u <= 1.0) {
        const double t = parameter_b * 6.0;
        const double c0 = std::fma(parameter_a, -2.0, 6.0);
        double c1 = std::fma(parameter_a, 12.0, t);
        c1 += -18.0;
        double c2 = std::fma(parameter_a, -9.0, -t);
        c2 += 12.0;
        const double u2 = u * u;
        double value = std::fma(u, c2, c1);
        value = std::fma(u2, value, c0);
        return value * kOneSixth;
    }

    const double c0 = std::fma(parameter_a, 8.0, parameter_b * 24.0);
    const double c1 = std::fma(parameter_a, -12.0, parameter_b * -48.0);
    const double c2 = std::fma(parameter_a, 6.0, parameter_b * 30.0);
    const double c3 = parameter_b * -6.0 - parameter_a;
    double value = std::fma(u, c3, c2);
    value = std::fma(u, value, c1);
    value = std::fma(u, value, c0);
    return value * kOneSixth;
}

}  // namespace resample_detail

ResampleCoefficientStatus build_resample_coefficients(
    double q, ResampleCoefficientTable *result) noexcept {
    if (result == nullptr || !std::isfinite(q) || q <= 0.0) {
        return ResampleCoefficientStatus::invalid_input;
    }

    const double scale = q < 1.0 ? q : 1.0;
    const double requested_tap_radius = (2.0 / scale) + kTapCountBias;
    if (!std::isfinite(requested_tap_radius) || requested_tap_radius < 2.0 ||
        requested_tap_radius >= 2049.0) {
        return ResampleCoefficientStatus::unsupported_tap_count;
    }
    const std::size_t tap_radius =
        static_cast<std::size_t>(requested_tap_radius);
    const std::size_t tap_count = tap_radius * 2;
    if (tap_count > kResampleCoefficientMaximumTaps) {
        return ResampleCoefficientStatus::unsupported_tap_count;
    }
    const std::size_t stride =
        ((tap_count + kResampleCoefficientStrideQuantum - 1) /
         kResampleCoefficientStrideQuantum) * kResampleCoefficientStrideQuantum;

    double parameter_a = 0.0;
    double parameter_b = 0.5;
    if (q > 1.0) {
        // Preserve the source schedule: rounded subtraction followed by a
        // multiplication by the stored reciprocal, not division by six.
        const double offset = q + -2.0;
        const double scaled_offset = offset * kOneSixth;
        const double c = scaled_offset < 1.0 ? scaled_offset : 1.0;
        parameter_a = c < 0.0 ? 0.0 : c * 0.5;
        parameter_b = (1.0 - parameter_a) * 0.5;
    }

    try {
        ResampleCoefficientTable candidate;
        candidate.coefficients.assign(kResampleCoefficientPhases * stride,
                                      0.0f);
        candidate.q = q;
        candidate.scale = scale;
        candidate.parameter_a = parameter_a;
        candidate.parameter_b = parameter_b;
        candidate.tap_count = tap_count;
        candidate.stride = stride;

        for (std::size_t phase_index = 0;
             phase_index < kResampleCoefficientPhases; ++phase_index) {
            const double phase = static_cast<double>(phase_index) * 0x1p-7;
            const std::size_t row = phase_index * stride;
            double sum = 0.0;
            for (std::size_t tap = 0; tap < tap_count; ++tap) {
                const int integer_offset = static_cast<int>(tap) -
                                           static_cast<int>(tap_radius) + 1;
                const double x = scale *
                    (static_cast<double>(integer_offset) - phase);
                const double raw = resample_detail::evaluate_resample_weight(
                    parameter_a, parameter_b, x);
                const float stored = static_cast<float>(raw);
                if (!std::isfinite(raw) || !std::isfinite(stored)) {
                    return ResampleCoefficientStatus::invalid_coefficients;
                }
                candidate.coefficients[row + tap] = stored;
                sum += static_cast<double>(stored);
            }

            if (!std::isfinite(sum) || sum <= 0.0) {
                return ResampleCoefficientStatus::invalid_coefficients;
            }
            const float normalizer = static_cast<float>(1.0 / sum);
            if (!std::isfinite(normalizer)) {
                return ResampleCoefficientStatus::invalid_coefficients;
            }
            for (std::size_t tap = 0; tap < tap_count; ++tap) {
                float &coefficient = candidate.coefficients[row + tap];
                coefficient = coefficient * normalizer;
                if (!std::isfinite(coefficient)) {
                    return ResampleCoefficientStatus::invalid_coefficients;
                }
            }
        }

        result->q = candidate.q;
        result->scale = candidate.scale;
        result->parameter_a = candidate.parameter_a;
        result->parameter_b = candidate.parameter_b;
        result->tap_count = candidate.tap_count;
        result->stride = candidate.stride;
        result->coefficients.swap(candidate.coefficients);
        return ResampleCoefficientStatus::ok;
    } catch (const std::bad_alloc &) {
        return ResampleCoefficientStatus::allocation_failure;
    }
}

}  // namespace imprint::acr
