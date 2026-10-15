#include "acr_exposure_rgb_prepare.hpp"

#include "acr_negative_curve.hpp"
#include "acr_whites.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace imprint::acr {
namespace {

constexpr double kInputLimit = 120.0;
constexpr double kPositiveLutLimit = 8.0;
constexpr float kLutScale = 4096.0f;

bool finite(double value) noexcept { return std::isfinite(value); }
bool finite(float value) noexcept { return std::isfinite(value); }

bool in_input_range(double value) noexcept {
    return finite(value) && value >= -kInputLimit && value <= kInputLimit;
}

ExposureRgbPrepareStatus validate_inputs(
    const ExposureRgbPrepareInputs &inputs) noexcept {
    if (!inputs.sdr_mode || inputs.whites_zero != (inputs.whites_ev == 0.0) || inputs.mask_enabled ||
        inputs.color_transform_enabled) {
        return ExposureRgbPrepareStatus::unsupported_context;
    }
    if (!finite(inputs.a_ev) || !finite(inputs.b_ev) ||
        !finite(inputs.amount58) || !finite(inputs.gray_diff_b8) ||
        !finite(inputs.anchor690) || !finite(inputs.pixel_epsilon) ||
        !finite(inputs.whites_ev)) {
        return ExposureRgbPrepareStatus::non_finite_input;
    }
    if (!in_input_range(inputs.a_ev) || !in_input_range(inputs.b_ev) ||
        !in_input_range(inputs.amount58) ||
        !in_input_range(inputs.gray_diff_b8) ||
        !in_input_range(inputs.anchor690) || inputs.pixel_epsilon <= 0.0f ||
        inputs.pixel_epsilon > static_cast<float>(kInputLimit) ||
        inputs.whites_ev < -2.0 || inputs.whites_ev > 2.0) {
        return ExposureRgbPrepareStatus::input_out_of_range;
    }
    if (inputs.whites_ev > 0 && (!inputs.post_tone_stats_prepared ||
        !finite(inputs.post_tone_stat670) || !finite(inputs.post_tone_stat678)))
        return ExposureRgbPrepareStatus::invalid_argument;
    return ExposureRgbPrepareStatus::ok;
}

ExposureRgbPrepareStatus to_float(double value, float *output) noexcept {
    if (output == nullptr || !finite(value)) {
        return ExposureRgbPrepareStatus::non_finite_intermediate;
    }
    if (value > static_cast<double>(std::numeric_limits<float>::max()) ||
        value < -static_cast<double>(std::numeric_limits<float>::max())) {
        return ExposureRgbPrepareStatus::overflow;
    }
    const float converted = static_cast<float>(value);
    if (!finite(converted)) return ExposureRgbPrepareStatus::overflow;
    *output = converted;
    return ExposureRgbPrepareStatus::ok;
}

}  // namespace

ExposureRgbPrepareStatus prepare_exposure_rgb(
    const ExposureRgbPrepareInputs &inputs,
    PreparedExposureRgbPlan *output) noexcept {
    if (output == nullptr) return ExposureRgbPrepareStatus::invalid_argument;
    const auto input_status = validate_inputs(inputs);
    if (input_status != ExposureRgbPrepareStatus::ok) return input_status;

    PreparedExposureRgbPlan candidate;
    candidate.pixel_epsilon = inputs.pixel_epsilon;
    candidate.lut_scale = kLutScale;

    const double sum = inputs.a_ev + inputs.b_ev;
    if (!finite(sum)) return ExposureRgbPrepareStatus::non_finite_intermediate;

    double split_a = inputs.a_ev;
    double split_b = inputs.b_ev;
    if (sum > 0.0) {
        const double amount = std::max(0.0, std::min(inputs.amount58, 1.0));
        const double context_limit =
            ((0.0 - inputs.anchor690) + inputs.gray_diff_b8) - (1.0 - amount);
        if (!finite(context_limit)) {
            return ExposureRgbPrepareStatus::non_finite_intermediate;
        }
        split_a = std::min(sum, std::max(context_limit, 0.0));
        split_b = std::max(sum - split_a, 0.0);
    }
    if (!finite(split_a) || !finite(split_b)) {
        return ExposureRgbPrepareStatus::non_finite_intermediate;
    }

    if (inputs.whites_ev < 0.0) {
        const double transferred = std::max(std::min(-inputs.whites_ev, split_a), 0.0);
        split_a -= transferred;
        split_b += transferred;
    } else if (inputs.whites_ev > 0.0) {
        const double transferred = std::min(inputs.whites_ev, split_b);
        split_a += transferred;
        split_b -= transferred;
    }

    candidate.split_a_ev = split_a;
    candidate.split_b_ev = split_b;
    candidate.whites_ev = inputs.whites_ev;

    if (split_a == 0.0) {
        candidate.gain = 1.0f;
    } else {
        float gain = 1.0f;
        const auto status = to_float(std::exp2(split_a), &gain);
        if (status != ExposureRgbPrepareStatus::ok) return status;
        candidate.gain = gain;
    }

    double first = 0.0;
    double remainder = 0.0;
    if (split_b > 0.0) {
        first = std::min(split_b, kPositiveLutLimit);
        remainder = std::max(split_b - first, 0.0);
    }
    if (!finite(first) || !finite(remainder)) {
        return ExposureRgbPrepareStatus::non_finite_intermediate;
    }
    candidate.positive_lut_ev = first;
    candidate.rational_remainder_ev = remainder;

    const double exp2_b = std::exp2(split_b);
    if (!finite(exp2_b)) return ExposureRgbPrepareStatus::non_finite_intermediate;
    const double floored_exp2_b = std::max(exp2_b, 1.0);
    if (floored_exp2_b > static_cast<double>(std::numeric_limits<float>::max())) {
        // The source's float conversion is +infinity here, which still passes
        // its strict `> 1` LUT enable comparison.
        candidate.lut_enabled = true;
    } else {
        candidate.lut_enabled = static_cast<float>(floored_exp2_b) > 1.0f;
    }
    const auto positive_gain_status = to_float(floored_exp2_b, &candidate.positive_gain);
    if (positive_gain_status != ExposureRgbPrepareStatus::ok) return positive_gain_status;

    if (remainder > 0.0) {
        const double positive_gain = std::exp2(remainder);
        if (!finite(positive_gain)) {
            return ExposureRgbPrepareStatus::non_finite_intermediate;
        }
        const double squared_gain = positive_gain * positive_gain;
        if (!finite(squared_gain)) {
            return ExposureRgbPrepareStatus::non_finite_intermediate;
        }
        const auto status = to_float(squared_gain, &candidate.rational_gain);
        if (status != ExposureRgbPrepareStatus::ok) return status;
    } else {
        candidate.rational_gain = 1.0f;
    }

    if (candidate.lut_enabled) {
        const auto curve_status =
            build_positive_exposure_lut(first, candidate.positive_lut);
        if (curve_status == PositiveCurveStatus::allocation_failure) {
            return ExposureRgbPrepareStatus::allocation_failure;
        }
        if (curve_status != PositiveCurveStatus::ok) {
            return ExposureRgbPrepareStatus::positive_curve_failure;
        }
        if (candidate.positive_lut.size() != kPositiveExposureLutSize) {
            return ExposureRgbPrepareStatus::positive_curve_failure;
        }
    }

    if (inputs.whites_ev < 0.0) {
        const auto status = build_negative_whites_lut(-inputs.whites_ev,
                                                      candidate.negative_whites_lut);
        if (status == WhitesStatus::allocation_failure)
            return ExposureRgbPrepareStatus::allocation_failure;
        if (status != WhitesStatus::ok) return ExposureRgbPrepareStatus::positive_curve_failure;
    } else if (inputs.whites_ev > 0.0) {
        WhiteClipResult clip;
        const auto status = prepare_white_clip_scalar(
            split_a, split_b, inputs.whites_ev,
            inputs.post_tone_stat670, inputs.post_tone_stat678, clip);
        if (status != WhitesStatus::ok || !finite(clip.clip_squared) || clip.clip_squared <= 0.0f)
            return ExposureRgbPrepareStatus::non_finite_intermediate;
        candidate.white_clip_squared = clip.clip_squared;
    }

    static_assert(std::is_nothrow_move_assignable<PreparedExposureRgbPlan>::value,
                  "prepared plan commit must be atomic");
    *output = std::move(candidate);
    return ExposureRgbPrepareStatus::ok;
}

ExposureRgbStatus PreparedExposureRgbPlan::apply(
    const float *input, float *output, std::size_t pixel_count) const noexcept {
    if (!finite(gain) || gain < 0.0f || !finite(rational_gain) ||
        rational_gain < 1.0f || !finite(pixel_epsilon) ||
        pixel_epsilon <= 0.0f || lut_scale != kLutScale ||
        (lut_enabled && positive_lut.size() != kPositiveExposureLutSize) ||
        (!lut_enabled && !positive_lut.empty()) ||
        !finite(whites_ev) || whites_ev < -2 || whites_ev > 2 ||
        (whites_ev < 0 && negative_whites_lut.size() != kNegativeWhitesLutSize) ||
        (whites_ev >= 0 && !negative_whites_lut.empty())) {
        return ExposureRgbStatus::invalid_argument;
    }

    PreparedExposureRgb prepared;
    prepared.gain = gain;
    prepared.lut_scale = lut_scale;
    prepared.rational_gain = rational_gain;
    prepared.epsilon = pixel_epsilon;
    prepared.lut = lut_enabled ? positive_lut.data() : nullptr;
    prepared.lut_size = lut_enabled ? positive_lut.size() : 0;
    prepared.positive_gain = positive_gain;
    prepared.white_clip_squared = white_clip_squared;
    prepared.whites_lut = negative_whites_lut.empty() ? nullptr : negative_whites_lut.data();
    prepared.whites_lut_size = negative_whites_lut.size();
    return apply_prepared_exposure_rgb(input, output, pixel_count, &prepared);
}

}  // namespace imprint::acr
