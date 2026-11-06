#include "acr_exposure.hpp"
#include "acr_exposure_allocation.hpp"

#include <algorithm>
#include <cmath>

namespace imprint::acr {
namespace {

constexpr double kLn2 = 0.693147180559945309417232121458176568;
constexpr double kMaxPreparedEv = 120.0;

// The +/-120 EV bound is a preparation API safety range, not an Adobe clamp.
bool finite_ev(double value) noexcept {
    return std::isfinite(value) && value >= -kMaxPreparedEv &&
           value <= kMaxPreparedEv;
}

}  // namespace

ExposureStatus prepare_exposure(const ExposureInput &input,
                                ExposurePrepared &prepared) noexcept {
    if (input.manual_exposure_x100 < -1000 ||
        input.manual_exposure_x100 > 1000) {
        return ExposureStatus::manual_exposure_out_of_range;
    }
    if (input.whites_ui < -100 || input.whites_ui > 100) {
        return ExposureStatus::whites_out_of_range;
    }
    if (!finite_ev(input.source_exposure_ev)) {
        return ExposureStatus::source_exposure_out_of_range;
    }
    if (!std::isfinite(input.source_scale) || input.source_scale <= 0.0) {
        return ExposureStatus::source_scale_invalid;
    }
    if (!std::isfinite(input.source_linear_response_limit) ||
        input.source_linear_response_limit < 0.0 ||
        input.source_linear_response_limit > 1.0) {
        return ExposureStatus::source_limit_out_of_range;
    }
    if (input.strongest_gain_q12 < 4096U ||
        input.strongest_gain_q12 > 65535U) {
        return ExposureStatus::strongest_gain_out_of_range;
    }
    if (!finite_ev(input.limit_ev88)) {
        return ExposureStatus::limit_ev88_out_of_range;
    }
    // The recovered allocation only tests whether this resolved context field
    // is zero. No amount range was recovered here; don't invent a UI limit.
    if (!std::isfinite(input.profile_gain_amount)) {
        return ExposureStatus::profile_gain_out_of_range;
    }

    ExposurePrepared result;
    result.manual_exposure_ev =
        static_cast<double>(input.manual_exposure_x100) * 0.01;
    result.whites_ev = static_cast<double>(input.whites_ui) * 0.02;
    result.source_exposure_ev = input.source_exposure_ev;
    result.source_scale = input.source_scale;
    result.context_ev50 = input.source_exposure_ev -
                          std::log(input.source_scale) / kLn2;
    result.limit_ev88 = input.limit_ev88;
    result.profile_gain_amount = input.profile_gain_amount;

    const float strongest_gain =
        static_cast<float>(input.strongest_gain_q12) * (1.0F / 4096.0F);
    result.first_arg = std::log(static_cast<double>(strongest_gain)) / kLn2;
    if (!input.three_color_planes && result.first_arg > 1.0) {
        result.first_arg = 1.0;
    }
    result.second_arg = input.three_color_planes ? result.first_arg : 0.0;

    // One recovered modern update from the constructor's initial state.
    // This API remains a single-call preparation; source-level pipelines that
    // invoke dbea08 more than once must carry ModernExposureAllocation state.
    const ModernExposureAllocation initial{
        result.manual_exposure_ev, 0.0, 0.0, result.context_ev50,
        input.profile_gain_amount, 1.0, result.whites_ev, input.limit_ev88,
        0.0, 0.0, 0.0, 0.0, 0.0};
    ModernExposureAllocation allocated{};
    if (update_modern_exposure_allocation(initial, result.first_arg,
            result.second_arg, input.hdr_edit, allocated) !=
        ExposureAllocationStatus::ok) {
        return ExposureStatus::result_not_finite;
    }
    result.exposure_gain = allocated.exposure_gain;
    result.ev90 = allocated.ev90;
    result.ev98 = allocated.ev98;
    result.ev100 = allocated.ev100;
    result.ev108 = allocated.ev108;
    result.ev112 = allocated.ev112;
    result.context_ev18_change = allocated.negative_curve_ev;

    result.camera_scale = 1.0;
    result.rounded_source_limit = std::floor(
        input.source_linear_response_limit * 100.0 + 0.5) / 100.0;
    double source_limit_quantized =
        std::floor(result.rounded_source_limit * 65535.0 + 0.5);
    source_limit_quantized = std::max(32768.0,
                                      std::min(65535.0, source_limit_quantized));
    result.source_limit_u16 = static_cast<std::uint32_t>(source_limit_quantized);

    const double ev_values[] = {
        result.manual_exposure_ev, result.whites_ev, result.context_ev50,
        result.first_arg, result.second_arg, result.ev90, result.ev98,
        result.ev100, result.ev108, result.ev112,
        result.context_ev18_change,
    };
    for (double value : ev_values) {
        if (!finite_ev(value)) return ExposureStatus::result_out_of_range;
    }
    if (!std::isfinite(result.exposure_gain) ||
        !std::isfinite(result.rounded_source_limit)) {
        return ExposureStatus::result_not_finite;
    }

    prepared = result;
    return ExposureStatus::ok;
}

}  // namespace imprint::acr
