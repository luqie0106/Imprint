#include "acr_profile_hsl_controls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace imprint::acr {
namespace {

constexpr std::int32_t kLower = -100;
constexpr std::int32_t kUpper = 100;

double normalize(std::int32_t value) noexcept {
    if (value <= kLower) return -1.0;
    if (value >= kUpper) return 1.0;
    // dd9804..3c / dd98cc..9904 for {-100, 0, 100}. In particular,
    // (value+100)/200 followed by FMA is not bitwise equal to value/100.
    const double t = (static_cast<double>(value) - (-100.0)) / 200.0;
    return std::fma(t, 2.0, -1.0);
}

double shape(double normalized_default, double scalar) noexcept {
    // dd99f0: the nonpositive scalar branch returns positive zero before
    // testing the sign of D. Negative shaping is recursive only for S>1.
    if (scalar <= 0.0) return 0.0;
    if (scalar <= 1.0) return normalized_default * scalar;
    if (normalized_default < 0.0) return -shape(-normalized_default, scalar);
    const double numerator = normalized_default * scalar;
    const double scalar_minus_one = scalar + (-1.0);
    const double denominator = std::fma(scalar_minus_one, normalized_default, 1.0);
    return numerator / denominator;
}

double combine(double normalized_current, double shaped_default) noexcept {
    // dd9a54: reflect both operands only when h<0 (not for negative zero).
    if (shaped_default < 0.0) return -combine(-normalized_current, -shaped_default);
    if (normalized_current <= 0.0) return normalized_current + shaped_default;
    if (normalized_current >= 1.0) return normalized_current;
    const double remaining = 1.0 - normalized_current;
    const double denominator = std::fma(normalized_current, shaped_default, remaining);
    const double ratio = remaining / denominator;
    const double weighted = ratio * shaped_default;
    return weighted + normalized_current;
}

double denormalize(double combined) noexcept {
    if (combined <= -1.0) return -100.0;
    if (combined >= 1.0) return 100.0;
    // dd9998..cc: middle fraction=.5 and its ratio=1. Keep the original
    // multiply, FMA denominator, divide and final FMA rather than 100*q.
    const double t = (combined + 1.0) * 0.5;
    const double numerator = t * 1.0;
    const double denominator = std::fma(0.0, t, 1.0);
    const double z = numerator / denominator;
    const double y = std::fma(200.0, z, -100.0);
    return std::clamp(y, -100.0, 100.0);
}

std::int32_t round_and_clamp(double blended) noexcept {
    const double offset = blended > 0.0 ? 0.5 : -0.5;
    const double truncated = std::trunc(blended + offset);
    // D==0 can return any int32 C. Truncate INT32_MIN-.5 / INT32_MAX+.5
    // back to the representable endpoints before conversion (caller FCVTZS).
    const auto value = static_cast<std::int32_t>(truncated);
    return std::clamp(value, kLower, kUpper);
}

}  // namespace

ProfileHslControlsStatus process_profile_hsl_controls(
    const ProfileHslControlsInput &input,
    ProfileHslControlsResult &result) noexcept {
    if (!std::isfinite(input.profile_scalar)) {
        return ProfileHslControlsStatus::non_finite_scalar;
    }

    ProfileHslControlsResult candidate{};
    const bool enabled = !input.monochrome_profile && input.convert_to_gray != 1;
    for (std::size_t lane = 0; lane < candidate.values.size(); ++lane) {
        const auto current = input.current[lane];
        const auto profile_default = input.profile_defaults[lane];
        candidate.values[lane] = current;
        candidate.blended[lane] = static_cast<double>(current);
        if (!enabled || profile_default < kLower || profile_default > kUpper) continue;

        candidate.wrote_mask |= std::uint32_t{1} << lane;
        // dd9760 returns C directly for D==middle, before normalization/S.
        if (profile_default != 0) {
            const double x = normalize(current);
            const double d = normalize(profile_default);
            const double h = shape(d, input.profile_scalar);
            candidate.blended[lane] = denormalize(combine(x, h));
        }
        candidate.values[lane] = round_and_clamp(candidate.blended[lane]);
    }
    result = candidate;
    return ProfileHslControlsStatus::ok;
}

}  // namespace imprint::acr
