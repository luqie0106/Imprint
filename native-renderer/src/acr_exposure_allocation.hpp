#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

// Compact numerical snapshot, not the original context ABI. Fields correspond
// to +08,+18,+28,+50,+58,+60,+80,+88,+90,+98,+a0,+a8,+b0, in that order.
// In particular +18 is accumulated and +50 is consumed/cleared on each call.
struct ModernExposureAllocation {
    double manual_ev, negative_curve_ev, first_arg, source_ev, profile_gain;
    double exposure_gain, whites_ev, limit_ev;
    double ev90, ev98, ev100, ev108, ev112;
};

enum class ExposureAllocationStatus : std::uint32_t {
    ok = 0, non_finite_input = 1, non_finite_result = 2,
};

// Recovered finite modern dbea08 branch. Caller selects the process-version
// route and supplies actual invocation arguments and HDR predicate. No source
// metadata, profile or gain inference occurs here. Can be applied repeatedly;
// input/output may be the same object. Every failure leaves output unchanged.
ExposureAllocationStatus update_modern_exposure_allocation(
    const ModernExposureAllocation &input, double first_arg, double second_arg,
    bool hdr_edit, ModernExposureAllocation &output) noexcept;

}  // namespace imprint::acr
