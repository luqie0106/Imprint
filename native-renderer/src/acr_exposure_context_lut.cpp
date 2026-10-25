#include "acr_exposure_context_lut.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace imprint::acr {
namespace {

constexpr std::size_t kExposureContextLutIntervals = 4096;

bool valid_output_pointer(double* output) noexcept {
    if (output == nullptr) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(output);
    if (address % alignof(double) != 0) return false;
    constexpr auto max_address = std::numeric_limits<std::uintptr_t>::max();
    return sizeof(double) <= static_cast<std::size_t>(max_address - address);
}

bool in_unit_interval(double value) noexcept {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

bool in_half_interval(double value) noexcept {
    return std::isfinite(value) && value >= 0.0 && value <= 0.5;
}

double clip01(double value) noexcept {
    value = std::fmin(value, 1.0);
    if (value < 0.0) value = 0.0;
    return value;
}

// Exact binary64 constants and operation order recovered for the shared
// Camera Raw encode/decode singleton. Keep these local to this testable curve
// implementation; the existing profile encoding API is intentionally private.
constexpr double kEncodeThreshold = 0x1.9a5c37387b719p-9;
constexpr double kEncodeExponent = 0x1.aaaaaaaaaaaabp-2;
constexpr double kEncodeSlope = 0x1.9d70a3d70a3d7p+3;
constexpr double kEncodeMultiplier = 0x1.0e147ae147ae1p+0;
constexpr double kEncodeOffset = -0x1.c28f5c28f5c29p-5;
constexpr double kDecodeThreshold = 0x1.4b5daa07d0fa8p-5;
constexpr double kDecodeOffset = 0x1.c28f5c28f5c29p-5;
constexpr double kDecodeMultiplier = 0x1.e54edcd0aeb60p-1;
constexpr double kDecodeExponent = 0x1.3333333333333p+1;
constexpr double kDecodeSlope = 0x1.3d0722149b580p-4;

double encode(double x) noexcept {
    if (x <= kEncodeThreshold) return x * kEncodeSlope;
    return std::fma(std::pow(x, kEncodeExponent), kEncodeMultiplier,
                    kEncodeOffset);
}

double decode(double x) noexcept {
    if (x <= kDecodeThreshold) return x * kDecodeSlope;
    const double shifted = x + kDecodeOffset;
    const double base = shifted * kDecodeMultiplier;
    return std::pow(base, kDecodeExponent);
}

bool evaluate_channel(double x, double a, double b, double h,
                      double* result) noexcept {
    const double scale_a = -1.0 / (a + -1.0);
    const double scale_b_candidate = -1.0 / (b + -1.0);
    const double scale_b = (b == 1.0) ? 0.0 : scale_b_candidate;
    const double threshold = b * h;
    const double x1 = std::fma(1.0 - b, 0.0, b);
    const double y1 = std::fma(scale_b, x1 - b, 0.0);
    const double x2 = std::fma(1.0 - b, 0.5, b);
    const double y2 = std::fma(scale_b, x2 - b, 0.0);

    double u = std::fma(scale_a, x - a, 0.0);
    u = clip01(u);

    if (threshold >= u) {
        *result = 0.0;
        return true;
    }
    if (x2 <= u) {
        *result = std::fma(scale_b, u - b, 0.0);
        return true;
    }

    const double x0 = threshold;
    const double A = x2 + std::fma(x1, -2.0, x0);
    const double B = std::fma(x0, -2.0, x1 + x1);
    const double C = x0 - u;

    double t = 0.0;
    if (A == 0.0) {
        if (B != 0.0) {
            const double root = (-C) / B;
            if (root >= 0.0 && root <= 1.0) t = root;
        }
    } else {
        double discriminant_term = A * -4.0;
        discriminant_term = discriminant_term * C;
        const double discriminant =
            std::fma(B, B, discriminant_term);
        if (!(discriminant < 0.0)) {
            const double sign = (B >= 0.0) ? 1.0 : -1.0;
            double q = std::fma(sign, std::sqrt(discriminant), B);
            q = q * -0.5;
            const double root1 = q / A;
            const double root2 = C / q;
            if (root1 >= 0.0 && root1 <= 1.0) {
                t = root1;
            } else if (root2 >= 0.0 && root2 <= 1.0) {
                t = root2;
            }
        }
    }

    const double v = 1.0 - t;
    double w = 0.0 * v;
    double z = t * (y1 + y1);
    z = v * z;
    double y = std::fma(w, v, z);
    w = t * y2;
    y = std::fma(w, t, y);
    *result = y;
    return true;
}

double evaluate_post(double x, double p) noexcept {
    const double clipped_p = clip01(p);
    const double A = clipped_p * 4.0;
    double K = std::fma(clipped_p, -2.0, 4.0);
    if (K > 3.0) K = 3.0;

    const double encoded = encode(x);
    const double decoded = decode(encoded);

    double u = std::exp2(-(K + A));
    const double v = (decoded + u) / (u + 1.0);
    u = std::exp2(-K);
    double w = (decoded + u) / (u + 1.0);
    w = 1.0 * w;
    const double ratio = w / v;

    const double scaled = ratio * x;
    const double difference = x - scaled;
    return std::fma(x, difference, scaled);
}

template <typename Evaluator, typename... Args>
ExposureContextLutStatus build_lut(Evaluator evaluator, std::vector<float>& output,
                                   Args... args) noexcept {
    std::vector<float> built;
    try {
        built.resize(kExposureContextLutSize);
    } catch (const std::bad_alloc&) {
        return ExposureContextLutStatus::allocation_failure;
    } catch (const std::length_error&) {
        return ExposureContextLutStatus::allocation_failure;
    } catch (...) {
        return ExposureContextLutStatus::allocation_failure;
    }

    const double reciprocal = 1.0 / static_cast<double>(kExposureContextLutIntervals);
    for (std::size_t i = 0; i <= kExposureContextLutIntervals; ++i) {
        const double x = reciprocal * static_cast<double>(i);
        double value = 0.0;
        const ExposureContextLutStatus status = evaluator(x, args..., &value);
        if (status != ExposureContextLutStatus::ok) return status;
        const float converted = static_cast<float>(value);
        if (!std::isfinite(converted)) {
            return ExposureContextLutStatus::non_finite_result;
        }
        built[i] = converted;
    }
    built[kExposureContextLutIntervals + 1] =
        built[kExposureContextLutIntervals];
    output.swap(built);
    return ExposureContextLutStatus::ok;
}

}  // namespace

ExposureContextLutStatus evaluate_exposure_channel_curve(
    double x, double a, double b, double h, double* output) noexcept {
    if (!valid_output_pointer(output) || !in_unit_interval(x) || !in_unit_interval(a) ||
        a == 1.0 || !in_unit_interval(b) || b == 1.0 || !in_half_interval(h)) {
        return ExposureContextLutStatus::invalid_argument;
    }

    double result = 0.0;
    evaluate_channel(x, a, b, h, &result);
    if (!std::isfinite(result)) {
        return ExposureContextLutStatus::non_finite_result;
    }
    *output = result;
    return ExposureContextLutStatus::ok;
}

ExposureContextLutStatus evaluate_exposure_post_curve(
    double x, double p, double* output) noexcept {
    if (!valid_output_pointer(output) || !in_unit_interval(x) || !in_unit_interval(p)) {
        return ExposureContextLutStatus::invalid_argument;
    }

    const double result = evaluate_post(x, p);
    if (!std::isfinite(result)) {
        return ExposureContextLutStatus::non_finite_result;
    }
    *output = result;
    return ExposureContextLutStatus::ok;
}

ExposureContextLutStatus build_exposure_channel_lut(
    double a, double b, double h, std::vector<float>& output) noexcept {
    if (!in_unit_interval(a) || a == 1.0 || !in_unit_interval(b) ||
        b == 1.0 || !in_half_interval(h)) {
        return ExposureContextLutStatus::invalid_argument;
    }
    return build_lut(evaluate_exposure_channel_curve, output, a, b, h);
}

ExposureContextLutStatus build_exposure_post_lut(
    double p, std::vector<float>& output) noexcept {
    if (!in_unit_interval(p)) return ExposureContextLutStatus::invalid_argument;
    return build_lut(evaluate_exposure_post_curve, output, p);
}

}  // namespace imprint::acr
