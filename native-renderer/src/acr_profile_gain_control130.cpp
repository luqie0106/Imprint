#include "acr_profile_gain_control130.hpp"

#include <cmath>

namespace imprint::acr {
namespace {

constexpr std::uint32_t kMinimumProfileVersion = 0x05070001u;
constexpr std::int32_t kMinimumProfileDefault = 0;
constexpr std::int32_t kMaximumProfileDefault = 200;
constexpr double kControlRange = 200.0;

double normalize_current(std::int32_t current) noexcept {
    if (current <= kMinimumProfileDefault) return 0.0;
    if (current >= kMaximumProfileDefault) return 1.0;
    return static_cast<double>(current) / kControlRange;
}

std::int32_t round_and_clamp(double blended) noexcept {
    const double offset = blended > 0.0 ? 0.5 : -0.5;
    const double truncated = std::trunc(blended + offset);
    // For D==0, blended can be any int32 current value. Truncation brings
    // INT32_MIN-0.5 and INT32_MAX+0.5 back to representable int32 endpoints
    // before the conversion, matching the caller's FCVTZS operation.
    std::int32_t value = static_cast<std::int32_t>(truncated);
    if (value < kMinimumProfileDefault) value = kMinimumProfileDefault;
    if (value > kMaximumProfileDefault) value = kMaximumProfileDefault;
    return value;
}

double dd99f0(double normalized_default, double scalar) noexcept {
    if (scalar <= 0.0) return 0.0;
    if (scalar <= 1.0) return normalized_default * scalar;

    const double numerator = normalized_default * scalar;
    const double scalar_minus_one = scalar - 1.0;
    const double denominator =
        std::fma(scalar_minus_one, normalized_default, 1.0);
    return numerator / denominator;
}

double dd9a54(double normalized_current, double shaped_default) noexcept {
    if (normalized_current <= 0.0) {
        return normalized_current + shaped_default;
    }
    if (normalized_current >= 1.0) return normalized_current;

    const double remaining = 1.0 - normalized_current;
    const double denominator =
        std::fma(normalized_current, shaped_default, remaining);
    const double ratio = remaining / denominator;
    const double weighted = ratio * shaped_default;
    return weighted + normalized_current;
}

double dd974c_id130(std::int32_t current, std::int32_t profile_default,
                    double scalar) noexcept {
    // dd974c compares D with T[1] before reading T[0]/T[2]. For ID130,
    // T[1] is zero and this early return bypasses S and normalization.
    if (profile_default == 0) return static_cast<double>(current);

    const double normalized_current = normalize_current(current);
    const double normalized_default =
        static_cast<double>(profile_default) / kControlRange;
    const double shaped_default = dd99f0(normalized_default, scalar);
    const double combined = dd9a54(normalized_current, shaped_default);

    if (combined <= 0.0) return 0.0;
    if (combined >= 1.0) return kControlRange;
    return std::fma(kControlRange, combined, 0.0);
}

}  // namespace

ProfileGainControl130Status process_profile_gain_control130(
    const PreparedProfileGainControl130 &input,
    ProfileGainControl130Result &result) noexcept {
    if (!std::isfinite(input.profile_scalar)) {
        return ProfileGainControl130Status::non_finite_scalar;
    }

    const double current = static_cast<double>(input.current);
    const bool eligible =
        input.profile_version >= kMinimumProfileVersion &&
        input.profile_default >= kMinimumProfileDefault &&
        input.profile_default <= kMaximumProfileDefault;
    if (!eligible) {
        result = {input.current, false, current};
        return ProfileGainControl130Status::ok;
    }

    const double blended = dd974c_id130(
        input.current, input.profile_default, input.profile_scalar);
    const std::int32_t value = round_and_clamp(blended);
    result = {value, true, blended};
    return ProfileGainControl130Status::ok;
}

}  // namespace imprint::acr
