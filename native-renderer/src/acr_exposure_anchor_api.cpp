#include "imprint_renderer.h"
#include "acr_exposure_rgb_prepare.hpp"
#include "acr_highlight_stat.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
bool range(const void *p, size_t bytes, size_t alignment) noexcept {
    const auto a = reinterpret_cast<uintptr_t>(p);
    return p && a % alignment == 0 && bytes <= std::numeric_limits<uintptr_t>::max() - a;
}
bool overlap(const void *a, size_t an, const void *b, size_t bn) noexcept {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
}

im_status im_native_acr_exposure_from_anchor_rgbf32(
    uint32_t width, uint32_t height, const float *rgb, size_t samples,
    const float *anchor_log, size_t anchor_samples,
    const double *controls, size_t control_count,
    float *output, size_t output_samples, double *diagnostics, size_t diagnostic_count) {
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (!width || !height || pixels > imprint::acr::kMaxHighlightStatSamples ||
        !anchor_samples || anchor_samples > imprint::acr::kMaxHighlightStatSamples ||
        samples != pixels * 3 || output_samples != samples ||
        control_count != 7 || diagnostic_count != 10) return IM_STATUS_INVALID_ARGUMENT;
    const size_t image_bytes = samples * sizeof(float);
    const size_t anchor_bytes = anchor_samples * sizeof(float);
    constexpr size_t control_bytes = 7 * sizeof(double), diagnostic_bytes = 10 * sizeof(double);
    if (!range(rgb, image_bytes, alignof(float)) ||
        !range(output, image_bytes, alignof(float)) ||
        !range(anchor_log, anchor_bytes, alignof(float)) ||
        !range(controls, control_bytes, alignof(double)) ||
        !range(diagnostics, diagnostic_bytes, alignof(double))) return IM_STATUS_INVALID_ARGUMENT;
    if ((rgb != output && overlap(rgb, image_bytes, output, image_bytes)) ||
        overlap(anchor_log, anchor_bytes, output, image_bytes) ||
        overlap(controls, control_bytes, output, image_bytes) ||
        overlap(diagnostics, diagnostic_bytes, output, image_bytes) ||
        overlap(rgb, image_bytes, diagnostics, diagnostic_bytes) ||
        overlap(anchor_log, anchor_bytes, diagnostics, diagnostic_bytes) ||
        overlap(controls, control_bytes, diagnostics, diagnostic_bytes)) return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < 7; ++i)
        if (!std::isfinite(controls[i])) return IM_STATUS_INVALID_ARGUMENT;
    if (controls[5] <= 0 || controls[5] > 0x1p-12 ||
        controls[6] <= 0 || controls[6] > 0x1p-16 ||
        static_cast<float>(controls[6]) == 0) return IM_STATUS_INVALID_ARGUMENT;

    imprint::acr::HighlightStat stat;
    const auto stat_status = imprint::acr::compute_highlight_stat(
        anchor_log, anchor_samples, controls[5], stat);
    if (stat_status == imprint::acr::HighlightStatStatus::allocation_failure)
        return IM_STATUS_RUNTIME_ERROR;
    if (stat_status != imprint::acr::HighlightStatStatus::ok) return IM_STATUS_INVALID_ARGUMENT;

    imprint::acr::ExposureRgbPrepareInputs input;
    input.a_ev = controls[0];
    input.b_ev = controls[1];
    input.amount58 = controls[2];
    input.gray_diff_b8 = controls[3];
    input.whites_ev = controls[4];
    input.anchor690 = stat.h;
    input.post_tone_stat670 = stat.filtered_p1;
    input.post_tone_stat678 = stat.filtered_p99;
    input.post_tone_stats_prepared = true;
    input.pixel_epsilon = static_cast<float>(controls[6]);
    input.sdr_mode = true;
    input.whites_zero = input.whites_ev == 0;
    input.mask_enabled = false;
    input.color_transform_enabled = false;
    imprint::acr::PreparedExposureRgbPlan plan;
    const auto prepare_status = imprint::acr::prepare_exposure_rgb(input, &plan);
    if (prepare_status == imprint::acr::ExposureRgbPrepareStatus::allocation_failure)
        return IM_STATUS_RUNTIME_ERROR;
    if (prepare_status != imprint::acr::ExposureRgbPrepareStatus::ok) return IM_STATUS_INVALID_ARGUMENT;
    const std::array<double, 10> result{
        stat.filtered_p1, stat.filtered_p99, stat.h, static_cast<double>(stat.filtered_count),
        plan.split_a_ev, plan.split_b_ev, plan.positive_lut_ev,
        plan.rational_remainder_ev, plan.positive_gain, plan.white_clip_squared};
    if (!std::all_of(result.begin(), result.end(), [](double x) { return std::isfinite(x); }))
        return IM_STATUS_RUNTIME_ERROR;
    // All diagnostics are ready before applying. The pixel core commits only
    // on success; the following memcpy cannot allocate or fail afterward.
    const auto pixel_status = plan.apply(rgb, output, static_cast<size_t>(pixels));
    if (pixel_status == imprint::acr::ExposureRgbStatus::allocation_failure)
        return IM_STATUS_RUNTIME_ERROR;
    if (pixel_status != imprint::acr::ExposureRgbStatus::ok) return IM_STATUS_INVALID_ARGUMENT;
    std::memcpy(diagnostics, result.data(), sizeof(result));
    return IM_STATUS_OK;
}
