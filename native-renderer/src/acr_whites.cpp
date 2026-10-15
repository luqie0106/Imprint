#include "acr_whites.hpp"

#include "acr_negative_curve.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace imprint::acr {
namespace {

constexpr double kWhitesBase = 0x1.bb67ae8584caap+0;  // bits 0x3ffbb67ae8584caa
constexpr double kGammaEncode = 0.45454545454545453;  // bits 0x3fdd1745d1745d17
constexpr double kGammaDecode = 2.2;                  // bits 0x400199999999999a
constexpr double kSqrtTwo = 0x1.6a09e667f3bcdp+0;     // bits 0x3ff6a09e667f3bcd
constexpr double kLnTwo = 0.6931471805599453;

bool finite(double value) noexcept {
    return std::isfinite(value);
}

WhitesStatus map_positive_status(PositiveCurveStatus status) noexcept {
    switch (status) {
        case PositiveCurveStatus::ok:
            return WhitesStatus::ok;
        case PositiveCurveStatus::input_out_of_range:
            return WhitesStatus::invalid_input;
        case PositiveCurveStatus::non_finite_intermediate:
            return WhitesStatus::non_finite_intermediate;
        case PositiveCurveStatus::allocation_failure:
            return WhitesStatus::allocation_failure;
    }
    return WhitesStatus::curve_failure;
}

WhitesStatus map_negative_status(NegativeCurveStatus status) noexcept {
    switch (status) {
        case NegativeCurveStatus::ok:
            return WhitesStatus::ok;
        case NegativeCurveStatus::input_out_of_range:
            return WhitesStatus::invalid_input;
        case NegativeCurveStatus::non_finite_intermediate:
            return WhitesStatus::non_finite_intermediate;
    }
    return WhitesStatus::curve_failure;
}

WhitesStatus evaluate_response(double linear, double curve_ev, double &output) noexcept {
    if (!finite(linear) || linear < 0.0 || linear > 1.0 || !finite(curve_ev)) {
        return WhitesStatus::invalid_input;
    }

    if (curve_ev >= 0.0) {
        double value = 0.0;
        const auto status = evaluate_positive_exposure_curve_scalar(
            linear, std::min(curve_ev, 8.0), value);
        if (status != PositiveCurveStatus::ok) return map_positive_status(status);
        output = value;
        return WhitesStatus::ok;
    }

    double encoded = linear;
    if (encoded > 0.0) {
        encoded = std::pow(encoded, kGammaEncode);
        if (!finite(encoded)) return WhitesStatus::non_finite_intermediate;
    }

    double inverse = 0.0;
    const auto status = evaluate_negative_curve_four_segment_inverse(
        encoded, curve_ev, inverse);
    if (status != NegativeCurveStatus::ok) return map_negative_status(status);

    double decoded = inverse;
    if (decoded > 0.0) {
        decoded = std::pow(decoded, kGammaDecode);
        if (!finite(decoded)) return WhitesStatus::non_finite_intermediate;
    }
    output = decoded;
    return WhitesStatus::ok;
}

}  // namespace

WhitesStatus build_negative_whites_lut(
    double prepared_b0, std::vector<float> &output) noexcept {
    if (!finite(prepared_b0) || prepared_b0 < 0.0 || prepared_b0 > 2.0) {
        return WhitesStatus::invalid_input;
    }

    const double exponent = std::pow(kWhitesBase, prepared_b0);
    if (!finite(exponent)) return WhitesStatus::non_finite_intermediate;

    std::vector<float> generated;
    try {
        generated.resize(kNegativeWhitesLutSize);
    } catch (...) {
        return WhitesStatus::allocation_failure;
    }

    for (std::size_t index = 0; index <= 4096; ++index) {
        const double x = static_cast<double>(index) / 4096.0;
        const double powered = std::pow(x, exponent);
        const double value = std::fma(x, powered - x, x);
        if (!finite(powered) || !finite(value)) {
            return WhitesStatus::non_finite_intermediate;
        }
        generated[index] = static_cast<float>(value);
        if (!std::isfinite(generated[index])) {
            return WhitesStatus::non_finite_intermediate;
        }
    }
    generated[4097] = generated[4096];
    output.swap(generated);
    return WhitesStatus::ok;
}

WhitesStatus prepare_white_clip_scalar(
    double split_a98, double split_ba0, double whites_ev80,
    double stat670, double stat678, WhiteClipResult &output) noexcept {
    if (!finite(split_a98) || !finite(split_ba0) || !finite(whites_ev80) ||
        !finite(stat670) || !finite(stat678) || split_a98 < -120.0 ||
        split_a98 > 120.0 || split_ba0 < -120.0 || split_ba0 > 120.0 ||
        whites_ev80 < -2.0 || whites_ev80 > 2.0) {
        return WhitesStatus::invalid_input;
    }

    const double range = stat678 - stat670;
    if (!finite(range)) return WhitesStatus::non_finite_intermediate;
    const double fraction = std::clamp(range / 5.0, 0.0, 1.0);
    const double upper_anchor = stat678 - fraction;
    const double lower_anchor = std::fma(range, 0.75, stat670);
    const double anchor = std::max(upper_anchor, lower_anchor);
    if (!finite(fraction) || !finite(upper_anchor) || !finite(lower_anchor) ||
        !finite(anchor)) {
        return WhitesStatus::non_finite_intermediate;
    }

    const double exposure_gain = std::exp2(split_a98);
    const double anchor_gain = std::exp2(anchor);
    const double unclamped_linear = exposure_gain * anchor_gain;
    if (!finite(exposure_gain) || !finite(anchor_gain) || !finite(unclamped_linear)) {
        return WhitesStatus::non_finite_intermediate;
    }
    const double linear = std::clamp(unclamped_linear, 0.0, 1.0);

    double response = 0.0;
    const auto curve_status = evaluate_response(linear, split_ba0, response);
    if (curve_status != WhitesStatus::ok) return curve_status;
    if (!finite(response)) return WhitesStatus::non_finite_intermediate;
    response = std::clamp(response, 0.0, 1.0);
    const double base = std::min(response, 0.5);

    const double half_whites = whites_ev80 * 0.5;
    const double nonnegative_half_whites = std::max(half_whites, 0.0);
    const double exponent = std::pow(nonnegative_half_whites, kSqrtTwo);
    if (!finite(half_whites) || !finite(exponent)) {
        return WhitesStatus::non_finite_intermediate;
    }

    const double log_base = std::log(base);
    const double log_base_over_ln2 = log_base / kLnTwo;
    const double log_gain = std::fma(exponent, log_base_over_ln2, 0.0);
    const double clip = std::exp2(log_gain);
    // A zero base with positive Whites legitimately produces -infinity here
    // and exp2(-infinity) == 0. A zero base with zero exponent instead makes
    // the fused product NaN and is rejected below without changing output.
    if (std::isnan(log_gain) || !finite(clip)) {
        return WhitesStatus::non_finite_intermediate;
    }

    WhiteClipResult result;
    result.anchor = anchor;
    result.response = response;
    result.clip = clip;
    result.clip_float = static_cast<float>(clip);
    if (!std::isfinite(result.clip_float)) return WhitesStatus::non_finite_intermediate;
    result.clip_squared = result.clip_float * result.clip_float;
    if (!std::isfinite(result.clip_squared)) return WhitesStatus::non_finite_intermediate;
    result.below_one = result.clip_squared < 1.0f;
    output = result;
    return WhitesStatus::ok;
}

}  // namespace imprint::acr
