#include "acr_stage3_balance3_controls.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace imprint::acr {
namespace {
static_assert(sizeof(double) == sizeof(std::uint64_t) &&
              std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::digits == 53);
double number(std::uint64_t bits) noexcept {
    double result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
bool convert(double value, double *out) noexcept {
    const double z = std::fma(value, number(0x3fefae147ae147aeULL),
                             number(0x3f847ae147ae147bULL));
    if (!(z > 0.0) || !std::isfinite(z)) return false;
    const double added = std::log(z) + number(0x40126bb1bbb55514ULL);
    const double scaled = added * number(0x3fcbcb7b1526e510ULL);
    *out = scaled - 1.0;
    return std::isfinite(*out);
}
double lower_limit(double v) noexcept {
    if (v < 0.0) return v;
    if (v == 0.0 && std::signbit(v)) return -0.0;
    return 0.0;
}
double upper_limit(double v) noexcept {
    return v > 0.0 ? v : 0.0;
}
}

Balance3ControlsStatus prepare_balance3_finish_limits_007ed860(
    double lower, double upper, bool direct, Balance3FinishLimits *out) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(out);
    if (!out || address % alignof(Balance3FinishLimits) != 0 ||
        sizeof(*out) > std::numeric_limits<std::uintptr_t>::max() - address)
        return Balance3ControlsStatus::invalid_argument;
    if (!std::isfinite(lower) || !std::isfinite(upper) || upper < lower)
        return Balance3ControlsStatus::invalid_range;
    if (lower <= 0.0 && upper <= 0.0)
        return Balance3ControlsStatus::unsupported_basic_branch;
    if (!direct && (!convert(lower, &lower) || !convert(upper, &upper)))
        return Balance3ControlsStatus::unsupported_domain;

    const double values[] = {lower_limit(-(upper * 0.5)),
        upper_limit(-(lower * 0.5)), lower_limit(lower * 0.5), upper_limit(upper * 0.5)};
    for (double value : values)
        if (!std::isfinite(value) || std::fabs(value) > std::numeric_limits<float>::max())
            return Balance3ControlsStatus::unsupported_domain;
    const Balance3FinishLimits candidate{static_cast<float>(values[0]),
        static_cast<float>(values[1]), static_cast<float>(values[2]), static_cast<float>(values[3])};
    if (!std::isfinite(candidate.lower_even) || !std::isfinite(candidate.upper_even) ||
        !std::isfinite(candidate.lower_odd) || !std::isfinite(candidate.upper_odd))
        return Balance3ControlsStatus::unsupported_domain;
    *out = candidate;
    return Balance3ControlsStatus::ok;
}
}
