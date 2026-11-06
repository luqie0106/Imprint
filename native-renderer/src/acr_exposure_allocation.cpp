#include "acr_exposure_allocation.hpp"

#include <cmath>
#include <limits>

namespace imprint::acr {
namespace {
bool finite(const ModernExposureAllocation &s) noexcept {
    const double values[]{s.manual_ev, s.negative_curve_ev, s.first_arg,
        s.source_ev, s.profile_gain, s.exposure_gain, s.whites_ev, s.limit_ev,
        s.ev90, s.ev98, s.ev100, s.ev108, s.ev112};
    for (double value : values) if (!std::isfinite(value)) return false;
    return true;
}
}  // namespace

ExposureAllocationStatus update_modern_exposure_allocation(
    const ModernExposureAllocation &input, double first_arg, double second_arg,
    bool hdr_edit, ModernExposureAllocation &output) noexcept {
    static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53);
    if (!finite(input) || !std::isfinite(first_arg) || !std::isfinite(second_arg))
        return ExposureAllocationStatus::non_finite_input;

    auto candidate = input;
    candidate.first_arg = first_arg;
    const double manual_and_second = input.manual_ev + second_arg;
    const double total = manual_and_second + input.source_ev;
    const double gain_ev = input.source_ev + second_arg;
    candidate.exposure_gain = std::exp2(gain_ev);
    double remainder = 0.0;
    double base = 0.0;
    if (total > 0.0) {
        double selected = input.limit_ev + second_arg;
        if (hdr_edit) selected = total;
        // FCSEL chooses total at equality; retain the original +0 addition.
        if (!(selected < total)) selected = total;
        base = selected + 0.0;
        remainder = total - base;
    }
    double negative_white = input.whites_ev >= 0.0 ? 0.0 : -input.whites_ev;
    const double transfer = remainder < input.whites_ev ? remainder : input.whites_ev;
    if (input.whites_ev > 0.0) {
        const double after = remainder - transfer;
        const double moved = base + transfer;
        negative_white = 0.0 - input.whites_ev;
        remainder = after;
        base = moved;
    }
    if (total < 0.0) {
        double negative = hdr_edit ? 0.0 : total;
        const double remaining = hdr_edit ? total : -0.0;
        base = base + remaining;
        if (input.profile_gain == 0.0) {
            const double contribution = std::fmax(negative, -1.0);
            candidate.negative_curve_ev = contribution + input.negative_curve_ev;
            negative = negative - contribution;
        }
        base = base + negative;
    }
    candidate.ev90 = second_arg;
    candidate.ev98 = base;
    candidate.ev100 = remainder;
    const double combined = remainder + base;
    candidate.ev108 = combined - second_arg;
    candidate.ev112 = negative_white;
    candidate.source_ev = 0.0;
    if (!std::isfinite(total) || !std::isfinite(gain_ev) || !finite(candidate))
        return ExposureAllocationStatus::non_finite_result;
    output = candidate;
    return ExposureAllocationStatus::ok;
}
}  // namespace imprint::acr
