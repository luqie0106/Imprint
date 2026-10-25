#pragma once

#include "acr_exposure_black_prepare.hpp"
#include "acr_exposure_qcap.hpp"

#include <array>
#include <cstdint>

namespace imprint::acr {

// Typed inputs to the prepared optional-exposure scalar factory. Runtime
// predicates have already been resolved by the caller; this helper does not
// infer a profile, mode, gain gate, or pixel request.
struct ExposureOptionalControls {
    std::int32_t blacks = 0;
    std::int32_t profile_gain_param = 0;
    bool producer_gain_active = false;
    bool mode = false;
    double k = 0.0;
    ExposureQcapContext context{};
    double a8 = 0.0;
    std::array<double, 3> base_gain{};
    bool caller_gain_active = false;
};

struct PreparedExposureOptional {
    double qv = 0.0;
    double qraw = 0.0;
    double q = 0.0;
    double z = 0.0;
    double B = 0.0;  // post_strength
    double D = 0.0;  // channel_strength
    double cap = 0.0;
    bool channel_active = false;
    bool post_active = false;
    std::array<double, 3> a{};
    std::array<double, 3> b{};
    double h = 0.0;
    double p = 0.0;
    double z_caller = 0.0;

    // Public API order: qv,qraw,q,z,B,D,cap,channel_active,post_active,
    // aR,bR,aG,bG,aB,bB,h,p,zCaller.
    std::array<double, 18> as_array() const noexcept;
};

enum class ExposureOptionalStatus : int {
    ok = 0,
    invalid_argument = 1,
    dependency_failure = 2,
};

// Prepared scalar composition only. Context validation is inherited from the
// existing qcap helper; a8 is bounded to [-120,120], each base gain is finite
// and nonnegative, and gain*base overflow is rejected. Output is committed
// only after both existing preparation helpers and all active-branch checks
// succeed.
ExposureOptionalStatus prepare_exposure_optional(
    const ExposureOptionalControls& controls,
    PreparedExposureOptional& output) noexcept;

}  // namespace imprint::acr
