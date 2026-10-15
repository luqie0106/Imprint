#include "imprint_renderer.h"
#include "acr_profile_tone.hpp"
#include "acr_negative_curve.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {
bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
    const auto av = reinterpret_cast<uintptr_t>(a);
    const auto bv = reinterpret_cast<uintptr_t>(b);
    if (an > std::numeric_limits<uintptr_t>::max() - av ||
        bn > std::numeric_limits<uintptr_t>::max() - bv) return true;
    return av < bv + bn && bv < av + an;
}

double negative_input(double value, const void *context) {
    double result;
    const auto status = imprint::acr::evaluate_negative_curve_scalar(
        value, *static_cast<const double *>(context), result);
    if (status != imprint::acr::NegativeCurveStatus::ok)
        throw std::invalid_argument("invalid negative pre-profile curve");
    return result;
}

im_status apply_profile(uint32_t width, uint32_t height, const float *source,
                        size_t source_samples, const double *points_xy, size_t point_count,
                        int modern, double curve_ev18, float *destination,
                        size_t destination_samples) {
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (!width || !height || pixels > (uint64_t{1} << 29) / 3 ||
        !source || !destination || !points_xy || point_count < 2 || point_count > 65536 ||
        (modern != 0 && modern != 1) || !std::isfinite(curve_ev18) ||
        curve_ev18 < -120 || curve_ev18 > 0)
        return IM_STATUS_INVALID_ARGUMENT;
    const size_t samples = static_cast<size_t>(pixels * 3);
    if (source_samples != samples || destination_samples != samples ||
        overlaps(source, samples * sizeof(float), destination, samples * sizeof(float)) ||
        overlaps(points_xy, point_count * 2 * sizeof(double), destination, samples * sizeof(float)))
        return IM_STATUS_INVALID_ARGUMENT;
    try {
        imprint::acr::apply_profile_tone(source, destination, static_cast<size_t>(pixels),
            points_xy, point_count, modern != 0,
            curve_ev18 == 0 ? nullptr : negative_input, &curve_ev18);
        return IM_STATUS_OK;
    } catch (const std::invalid_argument &) {
        return IM_STATUS_INVALID_ARGUMENT;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}
}

im_status im_native_acr_profile_tone_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *points_xy, size_t point_count, int modern,
    float *destination, size_t destination_samples) {
    return apply_profile(width, height, source, source_samples, points_xy, point_count,
                         modern, 0, destination, destination_samples);
}

im_status im_native_acr_profile_tone_exposure_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *points_xy, size_t point_count, int modern, double curve_ev18,
    float *destination, size_t destination_samples) {
    return apply_profile(width, height, source, source_samples, points_xy, point_count,
                         modern, curve_ev18, destination, destination_samples);
}
