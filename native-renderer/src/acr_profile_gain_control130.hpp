#pragma once

#include <cstdint>

namespace imprint::acr {

// Finite-scalar numerical subdomain for the normally entered ID 130
// indexed Basic profile write after profile selection. This is not all of
// Prepare, does not model other writers, and does not claim to produce the
// final prepared Q value.
struct PreparedProfileGainControl130 {
    std::int32_t current = 0;
    std::int32_t profile_default = 0;
    std::uint32_t profile_version = 0;
    double profile_scalar = 0.0;
};

// `blended` is the dd974c double result before the caller's int32
// round-away-from-zero and [0, 200] clamp. `value` is the caller's final
// value for this one indexed write.
struct ProfileGainControl130Result {
    std::int32_t value = 0;
    bool wrote = false;
    double blended = 0.0;
};

enum class ProfileGainControl130Status : int {
    ok = 0,
    non_finite_scalar = 1,
};

// Models the ID 130 loop gate, dd974c profile blend for T={0,0,200}, and the
// caller's round/clamp. A non-finite profile_scalar returns
// non_finite_scalar without changing `result`.
ProfileGainControl130Status process_profile_gain_control130(
    const PreparedProfileGainControl130 &input,
    ProfileGainControl130Result &result) noexcept;

}  // namespace imprint::acr
