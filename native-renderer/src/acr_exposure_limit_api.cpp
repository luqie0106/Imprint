#include "imprint_renderer.h"
#include "acr_exposure_limit.hpp"
#include "acr_highlight_stat.hpp"
#include "acr_flare.hpp"
#include "acr_source_log.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
bool valid_range(const void *pointer, size_t bytes, size_t alignment) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % alignment == 0 &&
           bytes <= std::numeric_limits<uintptr_t>::max() - address;
}
bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
    const auto av = reinterpret_cast<uintptr_t>(a);
    const auto bv = reinterpret_cast<uintptr_t>(b);
    return av <= bv ? bv - av < an : av - bv < bn;
}
}

im_status im_native_acr_exposure_limit_log_rgbf32(
    uint32_t width, uint32_t height, const float *log_rgb, size_t source_samples,
    const double *controls, size_t control_count, double epsilon,
    double *output, size_t output_count) {
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (!width || !height || pixels > imprint::acr::kMaxHighlightStatSamples ||
        source_samples != pixels * 3 || control_count != 17 || output_count != 15 ||
        !std::isfinite(epsilon) || epsilon <= 0 || epsilon > 0x1p-12)
        return IM_STATUS_INVALID_ARGUMENT;
    const size_t source_bytes = source_samples * sizeof(float);
    if (!valid_range(log_rgb, source_bytes, alignof(float)) ||
        !valid_range(controls, 17 * sizeof(double), alignof(double)) ||
        !valid_range(output, 15 * sizeof(double), alignof(double)) ||
        overlaps(log_rgb, source_bytes, output, 15 * sizeof(double)) ||
        overlaps(controls, 17 * sizeof(double), output, 15 * sizeof(double)))
        return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < 17; ++i)
        if (!std::isfinite(controls[i])) return IM_STATUS_INVALID_ARGUMENT;
    std::array<size_t, 3> order{};
    for (size_t i = 0; i < 3; ++i) {
        if (controls[i] < 0 || controls[i] > 2 || controls[i] != std::trunc(controls[i]))
            return IM_STATUS_INVALID_ARGUMENT;
        order[i] = static_cast<size_t>(controls[i]);
    }
    imprint::acr::CameraFloatParams camera{};
    for (size_t i = 3; i < 16; ++i)
        if (std::abs(controls[i]) > std::numeric_limits<float>::max())
            return IM_STATUS_INVALID_ARGUMENT;
    camera.gain_max = static_cast<float>(controls[3]);
    camera.gain_middle = static_cast<float>(controls[4]);
    camera.upper = static_cast<float>(controls[5]);
    camera.scale = static_cast<float>(controls[6]);
    for (size_t i = 0; i < 9; ++i) camera.matrix[i] = static_cast<float>(controls[7 + i]);
    try {
        std::vector<float> plane(static_cast<size_t>(pixels));
        std::array<double, 15> result{};
        std::array<double, 3> high{};
        for (size_t c = 0; c < 3; ++c) {
            for (size_t i = 0; i < pixels; ++i) plane[i] = log_rgb[i * 3 + c];
            imprint::acr::HighlightStat statistic;
            const auto status = imprint::acr::compute_highlight_stat(
                plane.data(), plane.size(), epsilon, statistic);
            if (status == imprint::acr::HighlightStatStatus::allocation_failure)
                return IM_STATUS_RUNTIME_ERROR;
            if (status != imprint::acr::HighlightStatStatus::ok)
                return IM_STATUS_INVALID_ARGUMENT;
            result[c] = high[c] = statistic.h;
            result[3 + c] = static_cast<double>(statistic.filtered_count);
        }
        imprint::acr::ExposureLimitResult limit;
        const auto status = imprint::acr::prepare_acr_exposure_limit(
            high.data(), order.data(), &camera, controls[16], &limit);
        if (status != imprint::acr::ExposureLimitStatus::ok)
            return IM_STATUS_INVALID_ARGUMENT;
        std::copy(limit.linear.begin(), limit.linear.end(), result.begin() + 6);
        std::copy(limit.mapped.begin(), limit.mapped.end(), result.begin() + 9);
        result[12] = limit.limit_ev88;
        result[13] = limit.gray_max_log_diff_b8;
        result[14] = limit.post_scale;
        std::memcpy(output, result.data(), sizeof(result));
        return IM_STATUS_OK;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_native_acr_source_limit_scene_u16_rgb(
    uint32_t width, uint32_t height, const uint16_t *source, size_t source_samples,
    const double *controls, size_t control_count, double *output, size_t output_count) {
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (!width || !height || pixels > (uint64_t{1} << 26) ||
        source_samples != pixels * 3 || control_count != 22 || output_count != 24)
        return IM_STATUS_INVALID_ARGUMENT;
    const size_t source_bytes = source_samples * sizeof(uint16_t);
    if (!valid_range(source, source_bytes, alignof(uint16_t)) ||
        !valid_range(controls, 22 * sizeof(double), alignof(double)) ||
        !valid_range(output, 24 * sizeof(double), alignof(double)) ||
        overlaps(source, source_bytes, output, 24 * sizeof(double)) ||
        overlaps(controls, 22 * sizeof(double), output, 24 * sizeof(double)))
        return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < 22; ++i)
        if (!std::isfinite(controls[i])) return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < 5; ++i)
        if (controls[i] != std::trunc(controls[i])) return IM_STATUS_INVALID_ARGUMENT;
    if (controls[0] < 0 || controls[0] > width || controls[1] < 0 || controls[1] > height ||
        controls[2] < 1 || controls[2] > width || controls[3] < 1 || controls[3] > height ||
        (controls[4] != 0 && controls[4] != 1)) return IM_STATUS_INVALID_ARGUMENT;
    const acr_flare::Crop crop{static_cast<size_t>(controls[0]),
                               static_cast<size_t>(controls[1]),
                               static_cast<size_t>(controls[2]),
                               static_cast<size_t>(controls[3])};
    try {
        acr_flare::Result flare;
        acr_flare::StatisticsLevel statistics;
        const auto flare_status = acr_flare::calculate(
            source, width, height, crop, 0, controls[4] == 1, true,
            &flare, nullptr, 0, &statistics);
        if (flare_status == acr_flare::Status::allocation_failure)
            return IM_STATUS_RUNTIME_ERROR;
        if (flare_status != acr_flare::Status::ok) return IM_STATUS_INVALID_ARGUMENT;
        std::vector<float> linear(statistics.pixels.size());
        // 00a963e8 supplies 65535 to 0133d62c; that kernel computes one float
        // reciprocal, then multiplies each exactly converted uint16 sample.
        constexpr float reciprocal = 1.0f / 65535.0f;
        for (size_t i = 0; i < linear.size(); ++i)
            linear[i] = static_cast<float>(statistics.pixels[i]) * reciprocal;
        std::vector<float> logarithm(linear.size());
        const auto log_status = imprint::acr::apply_acr_source_log(
            linear.data(), logarithm.data(), linear.size(), flare.epsilon, false);
        if (log_status == imprint::acr::SourceLogStatus::allocation_failure)
            return IM_STATUS_RUNTIME_ERROR;
        if (log_status != imprint::acr::SourceLogStatus::ok)
            return IM_STATUS_INVALID_ARGUMENT;
        std::array<double, 24> result{
            flare.mean, flare.flare, flare.epsilon,
            static_cast<double>(flare.last_width), static_cast<double>(flare.last_height),
            static_cast<double>(flare.level_count), static_cast<double>(statistics.level),
            static_cast<double>(statistics.width), static_cast<double>(statistics.height)};
        const auto status = im_native_acr_exposure_limit_log_rgbf32(
            static_cast<uint32_t>(statistics.width), static_cast<uint32_t>(statistics.height),
            logarithm.data(), logarithm.size(), controls + 5, 17, flare.epsilon,
            result.data() + 9, 15);
        if (status != IM_STATUS_OK) return status;
        std::memcpy(output, result.data(), sizeof(result));
        return IM_STATUS_OK;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}
