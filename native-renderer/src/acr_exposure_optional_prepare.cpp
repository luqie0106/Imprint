#include "acr_exposure_optional_prepare.hpp"

#include <algorithm>
#include <cmath>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace imprint::acr {
namespace {

constexpr double kMaximumChannelBlend = 0x1.fae147ae147aep-1;
constexpr double kSqrtTwo = 0x1.6a09e667f3bcdp+0;

bool clamp_unit_interval(double& value) noexcept {
    value = std::fmin(value, 1.0);
    if (value < -1.0) value = -1.0;
    return std::isfinite(value);
}

bool caller_black_value(const ExposureOptionalControls& controls,
                        double& output) noexcept {
    double z = static_cast<double>(controls.blacks) * -0.01;
    if (!clamp_unit_interval(z)) return false;

    if (controls.caller_gain_active) {
        const double adjustment =
            z <= 0.0 ? 0.25 : (1.0 - z) * 0.25;
        const double profile_gain =
            static_cast<double>(controls.profile_gain_param) * -0.01;
        const double scaled_gain = adjustment * profile_gain;
        z = z + scaled_gain;
        if (!clamp_unit_interval(z)) return false;
    }

    output = z;
    return true;
}

}  // namespace

std::array<double, 18> PreparedExposureOptional::as_array() const noexcept {
    return {qv,
            qraw,
            q,
            z,
            B,
            D,
            cap,
            channel_active ? 1.0 : 0.0,
            post_active ? 1.0 : 0.0,
            a[0],
            b[0],
            a[1],
            b[1],
            a[2],
            b[2],
            h,
            p,
            z_caller};
}

ExposureOptionalStatus prepare_exposure_optional(
    const ExposureOptionalControls& controls,
    PreparedExposureOptional& output) noexcept {
    if (!std::isfinite(controls.k) || !std::isfinite(controls.a8) ||
        controls.a8 < -120.0 || controls.a8 > 120.0) {
        return ExposureOptionalStatus::invalid_argument;
    }
    for (const double base : controls.base_gain) {
        if (!std::isfinite(base) || base < 0.0)
            return ExposureOptionalStatus::invalid_argument;
    }

    ExposureQcapResult qcap{};
    if (prepare_exposure_qcap(controls.context, qcap) !=
        ExposureQcapStatus::ok) {
        return ExposureOptionalStatus::dependency_failure;
    }

    PreparedBlackResponse black{};
    if (prepare_exposure_black_response(
            controls.blacks, controls.profile_gain_param,
            controls.producer_gain_active, controls.mode, controls.k, qcap.q,
            &black) != BlackPrepareStatus::ok) {
        return ExposureOptionalStatus::dependency_failure;
    }

    PreparedExposureOptional prepared{};
    prepared.qv = qcap.anchor_v;
    prepared.qraw = qcap.response;
    prepared.q = qcap.q;
    prepared.z = black.z;
    prepared.B = black.post_strength;
    prepared.D = black.channel_strength;

    const double gain = std::exp2(controls.a8);
    if (!std::isfinite(gain)) return ExposureOptionalStatus::invalid_argument;

    std::array<double, 3> t{};
    for (std::size_t index = 0; index < t.size(); ++index) {
        const double g = gain * controls.base_gain[index];
        if (!std::isfinite(g)) return ExposureOptionalStatus::invalid_argument;

        // Keep the D+g rounding separate from the following fused operation.
        const double sum = prepared.D + g;
        t[index] = std::fmin(std::fma(-prepared.D, g, sum),
                             kMaximumChannelBlend);
    }

    prepared.channel_active =
        t[0] > 0.0 || t[1] > 0.0 || t[2] > 0.0;
    if (prepared.channel_active) {
        const double cap_input =
            std::exp2(controls.context.filtered_p0001) / kSqrtTwo;
        double cap = 0.0;
        if (evaluate_exposure_context_response(controls.context, cap_input,
                                               cap) !=
                ExposureQcapStatus::ok ||
            !std::isfinite(cap)) {
            return ExposureOptionalStatus::dependency_failure;
        }
        prepared.cap = cap;

        for (std::size_t index = 0; index < t.size(); ++index) {
            prepared.a[index] = t[index] < cap ? t[index] : cap;
            const double remainder = t[index] - prepared.a[index];
            prepared.b[index] = std::fmax(remainder, 0.0);
            if (!std::isfinite(prepared.a[index]) ||
                !std::isfinite(prepared.b[index]) ||
                prepared.a[index] < 0.0 || prepared.a[index] >= 1.0 ||
                prepared.b[index] < 0.0 || prepared.b[index] >= 1.0) {
                return ExposureOptionalStatus::invalid_argument;
            }
        }

        double z_caller = 0.0;
        if (!caller_black_value(controls, z_caller))
            return ExposureOptionalStatus::invalid_argument;
        prepared.z_caller = z_caller;
        prepared.h = std::fma(prepared.z_caller, 2.0, 0.0);
        prepared.h = std::fmin(prepared.h, 0.5);
        if (prepared.h < 0.0) prepared.h = 0.0;
        if (!std::isfinite(prepared.h) || prepared.h < 0.0 ||
            prepared.h > 0.5) {
            return ExposureOptionalStatus::invalid_argument;
        }
    }

    prepared.post_active = prepared.B > 0.0;
    if (prepared.post_active) {
        prepared.p = prepared.B * 0.25;
        if (!std::isfinite(prepared.p) || prepared.p < 0.0 ||
            prepared.p > 1.0) {
            return ExposureOptionalStatus::invalid_argument;
        }
    }

    const auto values = prepared.as_array();
    for (const double value : values) {
        if (!std::isfinite(value))
            return ExposureOptionalStatus::invalid_argument;
    }

    output = prepared;
    return ExposureOptionalStatus::ok;
}

}  // namespace imprint::acr
