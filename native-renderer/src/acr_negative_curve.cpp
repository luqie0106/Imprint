#include "acr_negative_curve.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace imprint::acr {
namespace {

constexpr double kBase = 1.3703509847201236;  // bits 0x3ff5ecf52776a115
constexpr double kMinSlope = 0.1;
constexpr double kInverseStopDy = 1.0e-10;
constexpr int kInverseIterations = 30;
constexpr double kGammaEncode = 0.45454545454545453;  // bits 0x3fdd1745d1745d17
constexpr double kGammaDecode = 2.2;                    // bits 0x400199999999999a
constexpr double kQuarter = 0.25;
constexpr double kSixteenNinths = 1.7777777777777777;

struct Segment {
    double strength = 0.0;
    double g = 1.0;
    double h = 0.1;
    double k = 0.5;
    double w = 0.5;
};

bool finite(double value) noexcept {
    return std::isfinite(value);
}

NegativeCurveStatus build_segments(double curve_ev18,
                                   std::array<Segment, 4> &segments) noexcept {
    if (!finite(curve_ev18) || curve_ev18 < -120.0 || curve_ev18 > 0.0) {
        return NegativeCurveStatus::input_out_of_range;
    }

    const double magnitude = -curve_ev18;
    for (std::size_t index = 0; index < segments.size(); ++index) {
        const double offset = static_cast<double>(index) * 2.0;
        const double remaining = magnitude - offset;
        const double strength = std::max(0.0, std::min(2.0, remaining));
        const double g = std::pow(kBase, strength);
        if (!finite(g) || g <= 0.0) {
            return NegativeCurveStatus::non_finite_intermediate;
        }

        // The source uses an ARM64 FMA for (g - 1) * 12 + 1. Keep that one
        // contraction explicit while compiling all other expressions with
        // contraction disabled.
        const double denominator = std::fma(g - 1.0, 12.0, 1.0);
        if (!finite(denominator) || denominator <= 0.0) {
            return NegativeCurveStatus::non_finite_intermediate;
        }
        const double reciprocal = 1.0 / denominator;
        const double h = std::max(kMinSlope, reciprocal);
        const double k = 0.5 / g;
        const double w = 1.0 - k;
        if (!finite(h) || !finite(k) || !finite(w) || w <= 0.0) {
            return NegativeCurveStatus::non_finite_intermediate;
        }
        segments[index] = Segment{strength, g, h, k, w};
    }
    return NegativeCurveStatus::ok;
}

NegativeCurveStatus evaluate_positive_segment(double u, const Segment &segment,
                                               double &output) noexcept {
    if (!finite(u)) return NegativeCurveStatus::non_finite_intermediate;

    if (u < segment.k) {
        const double result = segment.g * u;
        if (!finite(result)) return NegativeCurveStatus::non_finite_intermediate;
        output = result;
        return NegativeCurveStatus::ok;
    }

    const double t = (u - segment.k) / segment.w;
    const double q = 1.0 - t;
    const double two_minus_q = 2.0 - q;
    const double second_sum = t + two_minus_q;
    const double wg = segment.w * segment.g;
    const double twg = wg * t;
    const double second_inner = std::fma(second_sum, 0.5, twg);
    const double q_squared = q * q;
    const double first_sum = (2.0 - t) + q;
    const double wh = segment.w * segment.h;
    const double slope = q * wh;
    const double first_inner = first_sum - slope;
    const double t_squared = t * t;
    const double first_term = t_squared * first_inner;
    const double result = std::fma(second_inner, q_squared, first_term);

    if (!finite(t) || !finite(q) || !finite(two_minus_q) ||
        !finite(second_sum) || !finite(wg) || !finite(twg) ||
        !finite(second_inner) || !finite(q_squared) ||
        !finite(first_sum) || !finite(wh) || !finite(slope) ||
        !finite(first_inner) || !finite(t_squared) || !finite(first_term) ||
        !finite(result)) {
        return NegativeCurveStatus::non_finite_intermediate;
    }
    output = result;
    return NegativeCurveStatus::ok;
}

NegativeCurveStatus inverse_positive_segment(double target,
                                             const Segment &segment,
                                             double &output) noexcept {
    double x_old = 0.0;
    double y_old = 0.0;
    auto status = evaluate_positive_segment(x_old, segment, y_old);
    if (status != NegativeCurveStatus::ok) return status;

    double x_current = 1.0;
    double y_current = 0.0;
    status = evaluate_positive_segment(x_current, segment, y_current);
    if (status != NegativeCurveStatus::ok) return status;

    for (int iteration = 0; iteration < kInverseIterations; ++iteration) {
        const double dy = y_current - y_old;
        if (!finite(dy)) return NegativeCurveStatus::non_finite_intermediate;
        if (std::abs(dy) < kInverseStopDy) {
            output = x_current;
            return NegativeCurveStatus::ok;
        }

        const double target_delta = target - y_current;
        const double x_delta = x_current - x_old;
        const double numerator = target_delta * x_delta;
        const double quotient = numerator / dy;
        const double proposed = x_current + quotient;
        if (!finite(target_delta) || !finite(x_delta) || !finite(numerator) ||
            !finite(quotient) || !finite(proposed)) {
            return NegativeCurveStatus::non_finite_intermediate;
        }
        const double x_new = std::max(0.0, std::min(1.0, proposed));
        double y_new = 0.0;
        status = evaluate_positive_segment(x_new, segment, y_new);
        if (status != NegativeCurveStatus::ok) return status;

        x_old = x_current;
        y_old = y_current;
        x_current = x_new;
        y_current = y_new;
    }

    // The recovered routine returns the current estimate after its 30th
    // iteration even when the strict |dy| stop has not fired.
    output = x_current;
    return NegativeCurveStatus::ok;
}

NegativeCurveStatus evaluate_remainder(double x, double remainder_ev,
                                       double &output) noexcept {
    const double gain = std::exp2(remainder_ev);
    const double b = (1.0 - gain) * kSixteenNinths;
    const double half_b = 0.5 * b;
    const double c = gain - half_b;
    const double one_minus_b = 1.0 - b;
    const double d = one_minus_b - c;
    if (!finite(gain) || !finite(b) || !finite(half_b) || !finite(c) ||
        !finite(one_minus_b) || !finite(d)) {
        return NegativeCurveStatus::non_finite_intermediate;
    }

    if (x <= kQuarter) {
        const double result = gain * x;
        if (!finite(result)) return NegativeCurveStatus::non_finite_intermediate;
        output = result;
        return NegativeCurveStatus::ok;
    }

    const double inner = std::fma(b, x, c);
    const double result = std::fma(inner, x, d);
    if (!finite(inner) || !finite(result)) {
        return NegativeCurveStatus::non_finite_intermediate;
    }
    output = result;
    return NegativeCurveStatus::ok;
}

}  // namespace

NegativeCurveStatus evaluate_negative_curve_four_segment_inverse(
    double target, double curve_ev18, double &output) noexcept {
    if (!finite(target) || target < 0.0 || target > 1.0 ||
        !finite(curve_ev18) || curve_ev18 < -120.0 || curve_ev18 > 0.0) {
        return NegativeCurveStatus::input_out_of_range;
    }
    if (curve_ev18 == 0.0) {
        output = target;
        return NegativeCurveStatus::ok;
    }

    std::array<Segment, 4> segments{};
    auto status = build_segments(curve_ev18, segments);
    if (status != NegativeCurveStatus::ok) return status;

    double current = target;
    for (std::size_t index = segments.size(); index-- > 0;) {
        const Segment &segment = segments[index];
        if (segment.strength == 0.0) continue;
        double next = current;
        status = inverse_positive_segment(current, segment, next);
        if (status != NegativeCurveStatus::ok) return status;
        current = next;
    }
    output = current;
    return NegativeCurveStatus::ok;
}

NegativeCurveStatus evaluate_negative_curve_scalar(
    double x, double curve_ev18, double &output) noexcept {
    if (!finite(x) || x < 0.0 || x > 1.0 ||
        !finite(curve_ev18) || curve_ev18 < -120.0 || curve_ev18 > 0.0) {
        return NegativeCurveStatus::input_out_of_range;
    }
    if (curve_ev18 == 0.0) {
        output = x;
        return NegativeCurveStatus::ok;
    }

    double pre_curve = x;
    if (curve_ev18 < -1.0) {
        const double remainder_ev = curve_ev18 + 1.0;
        auto status = evaluate_remainder(x, remainder_ev, pre_curve);
        if (status != NegativeCurveStatus::ok) return status;
    }

    double encoded = pre_curve;
    if (pre_curve > 0.0) {
        encoded = std::pow(pre_curve, kGammaEncode);
        if (!finite(encoded)) return NegativeCurveStatus::non_finite_intermediate;
    }

    const double curve_param = std::max(curve_ev18, -1.0);
    double inverse = encoded;
    auto status = evaluate_negative_curve_four_segment_inverse(
        encoded, curve_param, inverse);
    if (status != NegativeCurveStatus::ok) return status;

    double decoded = inverse;
    if (inverse > 0.0) {
        decoded = std::pow(inverse, kGammaDecode);
        if (!finite(decoded)) return NegativeCurveStatus::non_finite_intermediate;
    }
    output = decoded;
    return NegativeCurveStatus::ok;
}

namespace {

PositiveCurveStatus to_positive_status(NegativeCurveStatus status) noexcept {
    switch (status) {
        case NegativeCurveStatus::ok:
            return PositiveCurveStatus::ok;
        case NegativeCurveStatus::input_out_of_range:
            return PositiveCurveStatus::input_out_of_range;
        case NegativeCurveStatus::non_finite_intermediate:
            return PositiveCurveStatus::non_finite_intermediate;
    }
    return PositiveCurveStatus::non_finite_intermediate;
}

}  // namespace

PositiveCurveStatus evaluate_positive_exposure_curve_scalar(
    double x, double exposure_ev, double &output) noexcept {
    if (!finite(x) || x < 0.0 || x > 1.0 ||
        !finite(exposure_ev) || exposure_ev < 0.0 || exposure_ev > 8.0) {
        return PositiveCurveStatus::input_out_of_range;
    }

    std::array<Segment, 4> segments{};
    auto status = build_segments(-exposure_ev, segments);
    if (status != NegativeCurveStatus::ok) return to_positive_status(status);

    double encoded = x;
    if (encoded > 0.0) {
        encoded = std::pow(encoded, kGammaEncode);
        if (!finite(encoded)) {
            return PositiveCurveStatus::non_finite_intermediate;
        }
    }

    double current = encoded;
    for (const Segment &segment : segments) {
        if (segment.strength == 0.0) continue;
        double next = current;
        status = evaluate_positive_segment(current, segment, next);
        if (status != NegativeCurveStatus::ok) return to_positive_status(status);
        current = next;
    }

    double decoded = current;
    if (decoded > 0.0) {
        decoded = std::pow(decoded, kGammaDecode);
        if (!finite(decoded)) {
            return PositiveCurveStatus::non_finite_intermediate;
        }
    }
    output = decoded;
    return PositiveCurveStatus::ok;
}

PositiveCurveStatus build_positive_exposure_lut(
    double exposure_ev, std::vector<float> &output) noexcept {
    if (!finite(exposure_ev) || exposure_ev < 0.0 || exposure_ev > 8.0) {
        return PositiveCurveStatus::input_out_of_range;
    }

    std::vector<float> generated;
    try {
        generated.resize(kPositiveExposureLutSize);
    } catch (...) {
        return PositiveCurveStatus::allocation_failure;
    }

    for (std::size_t index = 0; index <= 4096; ++index) {
        const double input = (1.0 / 4096.0) * static_cast<double>(index);
        double value = 0.0;
        const auto status = evaluate_positive_exposure_curve_scalar(
            input, exposure_ev, value);
        if (status != PositiveCurveStatus::ok) return status;
        generated[index] = static_cast<float>(value);
    }
    generated[4097] = generated[4096];
    output.swap(generated);
    return PositiveCurveStatus::ok;
}

}  // namespace imprint::acr
