#include "acr_exposure_qcap.hpp"

#include "acr_negative_curve.hpp"
#include "acr_whites.hpp"

#include <algorithm>
#include <cmath>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace imprint::acr {
namespace {
bool bounded(double x, double lo, double hi) noexcept {
    return std::isfinite(x) && x >= lo && x <= hi;
}
bool valid(const ExposureQcapContext& c) noexcept {
    return bounded(c.exposure98, -120, 120) && bounded(c.curve_a0, -120, 120) &&
           bounded(c.whites_b0, -2, 2) && bounded(c.whites80, -2, 2) &&
           std::isfinite(c.filtered_p1) && std::isfinite(c.filtered_p99) &&
           std::isfinite(c.filtered_p0001) && std::isfinite(c.high690);
}
double clip01(double x) noexcept {
    x = std::fmin(x, 1.0);
    return x < 0.0 ? 0.0 : x;
}
}  // namespace

ExposureQcapStatus evaluate_exposure_context_response(
    const ExposureQcapContext& c, double input, double& output) noexcept {
    if (!valid(c) || !std::isfinite(input) || input < 0.0)
        return ExposureQcapStatus::invalid_argument;
    // 009226b4 -> 00926db0 always wraps dc81c4 in encode/decode pow.
    // Canonicalize zero before using the existing conditional-pow helpers.
    double x = clip01(std::exp2(c.exposure98) * input);
    if (x == 0.0) x = 0.0;
    double response;
    if (c.curve_a0 >= 0.0) {
        if (evaluate_positive_exposure_curve_scalar(
                x, std::min(c.curve_a0, 8.0), response) != PositiveCurveStatus::ok)
            return ExposureQcapStatus::curve_failure;
    } else {
        const double encoded = std::pow(x, 0x1.d1745d1745d17p-2);
        double inverse;
        if (evaluate_negative_curve_four_segment_inverse(encoded, c.curve_a0, inverse) !=
            NegativeCurveStatus::ok)
            return ExposureQcapStatus::curve_failure;
        response = std::pow(inverse, 2.2);
    }
    if (!std::isfinite(response)) return ExposureQcapStatus::curve_failure;
    if (c.whites_b0 > 0.0) {
        const double exponent = std::pow(0x1.bb67ae8584caap+0, c.whites_b0);
        response = std::fma(response, std::pow(response, exponent) - response, response);
    } else if (c.whites_b0 < 0.0) {
        WhiteClipResult white;
        if (prepare_white_clip_scalar(c.exposure98, c.curve_a0, c.whites80,
                                     c.filtered_p1, c.filtered_p99, white) != WhitesStatus::ok)
            return ExposureQcapStatus::curve_failure;
        if (!(white.clip > 0.0)) return ExposureQcapStatus::curve_failure;
        // 009267b0 constructs the rational object from the DOUBLE clip square.
        const double square = white.clip * white.clip;
        const double first = response / (response + 1.0);
        const double second = response / square + 1.0;
        response = std::fmin(first * second, 1.0);
    }
    if (!std::isfinite(response)) return ExposureQcapStatus::curve_failure;
    output = response;
    return ExposureQcapStatus::ok;
}

ExposureQcapStatus prepare_exposure_qcap(
    const ExposureQcapContext& c, ExposureQcapResult& output) noexcept {
    if (!valid(c)) return ExposureQcapStatus::invalid_argument;
    const double first = c.filtered_p0001 + 2.0;
    const double second = std::fma(c.high690 - c.filtered_p0001, 0.5,
                                   c.filtered_p0001);
    const double v = first < second ? first : second;
    if (!bounded(v, -120, 120)) return ExposureQcapStatus::invalid_argument;
    double response;
    const auto status = evaluate_exposure_context_response(c, std::exp2(v), response);
    if (status != ExposureQcapStatus::ok) return status;
    const double h = clip01(response);
    const double q = std::fmin(std::fmax(h, 0.0625), 0.9);
    output = ExposureQcapResult{v, response, q};
    return ExposureQcapStatus::ok;
}

}  // namespace imprint::acr
