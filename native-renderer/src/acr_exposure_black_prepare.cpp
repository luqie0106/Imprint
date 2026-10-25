#include "acr_exposure_black_prepare.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstring>

namespace imprint::acr {
namespace {

constexpr double kEpsilon = 0x1.5798ee2308c3ap-27;  // 1e-8 by recovered bits.
constexpr double kUpperCurveLimit = 0x1.ffffffaa19c47p-1;
constexpr double kBlackBlendScale = -0x1.5555555555555p+0;
constexpr double kMinimumPreparedQ = 0.0625;
constexpr double kMaximumPreparedQ = 0.9;

struct QuadraticResponseCurve {
    double x0;
    double y0;
    double x1;
    double y1;
    double x2;
    double y2;
    double xlo;
    double xhi;
    double y_at_xlo;
    double y_at_xhi;
};

bool valid_output_range(const void* pointer, std::size_t size,
                        std::size_t alignment) noexcept {
    if (pointer == nullptr || alignment == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) return false;
    constexpr auto max_address = std::numeric_limits<std::uintptr_t>::max();
    return size <= static_cast<std::size_t>(max_address - begin);
}

bool finite_inputs(double z, double t0, double t1, double k,
                   double q) noexcept {
    return std::isfinite(z) && std::isfinite(t0) && std::isfinite(t1) &&
           std::isfinite(k) && std::isfinite(q) && t0 < t1 && t1 < 1.0;
}

BlackPrepareStatus construct_curve(double x0, double y0, double x1, double y1,
                                   double x2, double y2,
                                   QuadraticResponseCurve* output) noexcept {
    if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) ||
        !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) {
        return BlackPrepareStatus::non_finite_result;
    }

    const bool increasing = x0 < x1 && x1 < x2;
    const bool decreasing = x0 > x1 && x1 > x2;
    if (!increasing && !decreasing) return BlackPrepareStatus::invalid_curve;

    QuadraticResponseCurve curve{};
    curve.x0 = x0;
    curve.y0 = y0;
    curve.x1 = x1;
    curve.y1 = y1;
    curve.x2 = x2;
    curve.y2 = y2;
    if (x0 < x2) {
        curve.xlo = x0;
        curve.xhi = x2;
        curve.y_at_xlo = y0;
        curve.y_at_xhi = y2;
    } else {
        curve.xlo = x2;
        curve.xhi = x0;
        curve.y_at_xlo = y2;
        curve.y_at_xhi = y0;
    }
    std::memcpy(output, &curve, sizeof(curve));
    return BlackPrepareStatus::ok;
}

struct Roots {
    double values[2];
    std::size_t count;
};

Roots quadratic_roots(double a, double b, double c) noexcept {
    Roots roots{{0.0, 0.0}, 0};
    if (a == 0.0) {
        if (b == 0.0) return roots;
        const double negated_c = -c;
        roots.values[0] = negated_c / b;
        roots.count = 1;
        return roots;
    }

    double product = a * -4.0;
    product = product * c;
    const double discriminant = std::fma(b, b, product);
    if (discriminant < 0.0) return roots;

    const double sign = (b >= 0.0) ? 1.0 : -1.0;
    double qroot = std::fma(sign, std::sqrt(discriminant), b);
    qroot = qroot * -0.5;
    // The recovered helper computes and stores r2 before r1, but returns
    // [r1, r2]; the evaluator scans that returned order.
    const double r2 = c / qroot;
    const double r1 = qroot / a;
    roots.values[0] = r1;
    roots.values[1] = r2;
    roots.count = 2;
    return roots;
}

double evaluate_curve(const QuadraticResponseCurve& curve, double z) noexcept {
    if (curve.xlo >= z) return curve.y_at_xlo;
    if (curve.xhi <= z) return curve.y_at_xhi;

    double a = std::fma(curve.x1, -2.0, curve.x0);
    a = a + curve.x2;
    double b = curve.x1 + curve.x1;
    b = std::fma(curve.x0, -2.0, b);
    const double c = curve.x0 - z;
    const Roots roots = quadratic_roots(a, b, c);

    double s = 0.0;
    for (std::size_t index = 0; index < roots.count; ++index) {
        const double root = roots.values[index];
        if (root >= 0.0 && root <= 1.0) {
            s = root;
            break;
        }
    }

    const double v = 1.0 - s;
    double w = curve.y0 * v;
    double j = curve.y1 + curve.y1;
    j = s * j;
    j = v * j;
    double result = std::fma(w, v, j);
    w = s * curve.y2;
    result = std::fma(w, s, result);
    return result;
}

BlackPrepareStatus make_curve(double t, double k, double m0, double q,
                              double m1,
                              QuadraticResponseCurve* output) noexcept {
    const double delta = m0 - m1;
    double absolute_delta = delta;
    if (absolute_delta < 0.0) absolute_delta = -absolute_delta;

    if (absolute_delta < kEpsilon) {
        const double mid_x = (t + 1.0) * 0.5;
        const double mid_y = (k + q) * 0.5;
        // Preserve the literal recovered constructor argument order.
        return construct_curve(t, 1.0, mid_x, mid_y, k, q, output);
    }

    double n = std::fma(m0, t, -m1);
    n = n - k;
    n = n + q;
    const double u = n / delta;
    const double y = std::fma(m0, u, k);
    return construct_curve(t, k, u, y, 1.0, q, output);
}

BlackPrepareStatus evaluate_response(double z, double t0, double t1, double k,
                                     double q, double* result) noexcept {
    if (z <= t1) {
        double slope = 0.0;
        if (t0 != t1) {
            const double numerator = 0.0 - k;
            const double denominator = t0 - t1;
            slope = numerator / denominator;
        }
        const double offset = z - t0;
        *result = std::fma(slope, offset, 0.0);
        return std::isfinite(*result) ? BlackPrepareStatus::ok
                                      : BlackPrepareStatus::non_finite_result;
    }

    const double span = t1 - t0;
    const double m0 = k / span;
    const double q_minus_k = q - k;
    const double m1 = q_minus_k / (1.0 - t1);

    QuadraticResponseCurve curve{};
    if (m1 == m0) {
        const auto status = make_curve(t1, k, m0, q, m1, &curve);
        if (status != BlackPrepareStatus::ok) return status;
    } else if (m1 < m0) {
        const auto status = make_curve(t1, k, m0, q, 0.0, &curve);
        if (status != BlackPrepareStatus::ok) return status;
    } else {
        const double a0 = std::atan(m0);
        const double a1 = std::atan(m1);
        double angle = a1 - a0;
        angle = a1 + angle;
        double w = std::tan(angle);
        w = w * 1.5;
        double n = std::fma(t1, m0, q_minus_k);
        n = n - w;
        n = n + kEpsilon;
        double d = m0 - w;
        d = d + kEpsilon;
        double u = n / d;
        const double lower = t1 + kEpsilon;
        u = std::fmin(u, kUpperCurveLimit);
        if (u < lower) u = lower;
        const double offset = u - t1;
        const double y = std::fma(m0, offset, k);
        const auto status = construct_curve(t1, k, u, y, 1.0, q, &curve);
        if (status != BlackPrepareStatus::ok) return status;
    }

    *result = evaluate_curve(curve, z);
    return std::isfinite(*result) ? BlackPrepareStatus::ok
                                  : BlackPrepareStatus::non_finite_result;
}

}  // namespace

BlackPrepareStatus evaluate_black_control_response(
    double z, double t0, double t1, double k, double q,
    double* output) noexcept {
    if (!valid_output_range(output, sizeof(double), alignof(double)) ||
        !finite_inputs(z, t0, t1, k, q)) {
        return BlackPrepareStatus::invalid_argument;
    }

    double value = 0.0;
    const auto status = evaluate_response(z, t0, t1, k, q, &value);
    if (status != BlackPrepareStatus::ok) return status;
    std::memcpy(output, &value, sizeof(value));
    return BlackPrepareStatus::ok;
}

BlackPrepareStatus prepare_exposure_black_response(
    std::int32_t blacks, std::int32_t profile_gain_param,
    bool gain_adjust_active, bool mode, double k, double q,
    PreparedBlackResponse* output) noexcept {
    if (!valid_output_range(output, sizeof(PreparedBlackResponse),
                            alignof(PreparedBlackResponse)) ||
        !std::isfinite(k) || !std::isfinite(q) ||
        q < kMinimumPreparedQ || q > kMaximumPreparedQ) {
        return BlackPrepareStatus::invalid_argument;
    }

    double z = static_cast<double>(blacks) * -0.01;
    z = std::fmin(z, 1.0);
    if (z < -1.0) z = -1.0;
    if (gain_adjust_active) {
        const double adjustment = (z <= 0.0) ? 0.25 : (1.0 - z) * 0.25;
        const double profile_gain =
            static_cast<double>(profile_gain_param) * -0.01;
        const double scaled_gain = adjustment * profile_gain;
        z = z + scaled_gain;
        z = std::fmin(z, 1.0);
        if (z < -1.0) z = -1.0;
    }

    double post_strength = 0.0;
    double channel_strength = 0.0;
    if (mode) {
        if (z < 0.0) {
            post_strength = -z;
        } else {
            if (z <= 0.0) {
                channel_strength = 0.0;
            } else {
                const auto status =
                    evaluate_response(z, 0.0, 0.25, k, q, &channel_strength);
                if (status != BlackPrepareStatus::ok) return status;
            }
        }
    } else if (z < -0.25) {
        const double shifted = z + 1.0;
        post_strength = std::fma(shifted, kBlackBlendScale, 1.0);
    } else if (z <= -0.25) {
        channel_strength = 0.0;
    } else {
        const auto status =
            evaluate_response(z, -0.25, 0.0, k, q, &channel_strength);
        if (status != BlackPrepareStatus::ok) return status;
    }

    if (!std::isfinite(z) || !std::isfinite(post_strength) ||
        !std::isfinite(channel_strength)) {
        return BlackPrepareStatus::non_finite_result;
    }
    const PreparedBlackResponse prepared{z, post_strength, channel_strength};
    std::memcpy(output, &prepared, sizeof(prepared));
    return BlackPrepareStatus::ok;
}

}  // namespace imprint::acr
