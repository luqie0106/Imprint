#include "imprint_renderer.h"
#include "acr_gray.hpp"
#include "acr_source_log.hpp"
#include "acr_u16_store.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
bool valid_range(const void *pointer, size_t bytes, size_t alignment) noexcept {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % alignment == 0 &&
           bytes <= std::numeric_limits<uintptr_t>::max() - address;
}
bool overlaps(const void *a, size_t an, const void *b, size_t bn) noexcept {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
}

static im_status gray_log_impl(
    uint32_t width, uint32_t height, const float *rgb, size_t rgb_samples,
    const double *coefficients, size_t coefficient_count, double source_epsilon,
    float *log_gray, size_t log_samples, double *diagnostics, size_t diagnostic_count,
    int storage_mode, int32_t origin_x, int32_t origin_y) {
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (!width || !height || pixels > (uint64_t{1} << 26) ||
        rgb_samples != pixels * 3 || log_samples != pixels ||
        coefficient_count != 3 || diagnostic_count != 6 ||
        !std::isfinite(source_epsilon) || source_epsilon <= 0 ||
        source_epsilon > 0x1p-12 || static_cast<float>(source_epsilon) == 0)
        return IM_STATUS_INVALID_ARGUMENT;
    const size_t rgb_bytes = rgb_samples * sizeof(float);
    const size_t log_bytes = log_samples * sizeof(float);
    constexpr size_t coefficient_bytes = 3 * sizeof(double), diagnostic_bytes = 6 * sizeof(double);
    if (!valid_range(rgb, rgb_bytes, alignof(float)) ||
        !valid_range(coefficients, coefficient_bytes, alignof(double)) ||
        !valid_range(log_gray, log_bytes, alignof(float)) ||
        !valid_range(diagnostics, diagnostic_bytes, alignof(double)) ||
        overlaps(rgb, rgb_bytes, log_gray, log_bytes) ||
        overlaps(coefficients, coefficient_bytes, log_gray, log_bytes) ||
        overlaps(diagnostics, diagnostic_bytes, log_gray, log_bytes) ||
        overlaps(rgb, rgb_bytes, diagnostics, diagnostic_bytes) ||
        overlaps(coefficients, coefficient_bytes, diagnostics, diagnostic_bytes))
        return IM_STATUS_INVALID_ARGUMENT;

    imprint::acr::PreparedRgbToGray prepared{};
    const auto prepare_status = imprint::acr::prepare_rgb_to_gray(coefficients, &prepared);
    if (prepare_status != imprint::acr::GrayStatus::ok) return IM_STATUS_INVALID_ARGUMENT;
    const std::array<double, 6> result{
        static_cast<double>(prepared.lanes_rgb[0]), static_cast<double>(prepared.lanes_rgb[1]),
        static_cast<double>(prepared.lanes_rgb[2]), static_cast<double>(prepared.weights_rgb[0]),
        static_cast<double>(prepared.weights_rgb[1]), static_cast<double>(prepared.weights_rgb[2])};
    try {
        std::vector<float> gray(static_cast<size_t>(pixels));
        const auto gray_status = imprint::acr::apply_rgb_to_gray(rgb, gray.data(), gray.size(), &prepared);
        if (gray_status == imprint::acr::GrayStatus::allocation_failure)
            return IM_STATUS_RUNTIME_ERROR;
        if (gray_status != imprint::acr::GrayStatus::ok) return IM_STATUS_INVALID_ARGUMENT;
        if (storage_mode != 0) {
            std::vector<uint16_t> stored(gray.size());
            const auto store_status = imprint::acr::apply_gray_u16_store(
                gray.data(), stored.data(), gray.size(), width, height,
                origin_x, origin_y, storage_mode == 2);
            if (store_status == imprint::acr::U16StoreStatus::allocation_failure)
                return IM_STATUS_RUNTIME_ERROR;
            if (store_status != imprint::acr::U16StoreStatus::ok)
                return IM_STATUS_INVALID_ARGUMENT;
            const float reciprocal = 1.0f / 65535.0f;
            for (size_t i = 0; i < gray.size(); ++i)
                gray[i] = static_cast<float>(stored[i]) * reciprocal;
        }
        // Source log stages their output before committing. Diagnostics were
        // prepared above, so neither caller output changes on any failure.
        const auto log_status = imprint::acr::apply_acr_source_log(
            gray.data(), log_gray, gray.size(), source_epsilon, false);
        if (log_status == imprint::acr::SourceLogStatus::allocation_failure)
            return IM_STATUS_RUNTIME_ERROR;
        if (log_status != imprint::acr::SourceLogStatus::ok) return IM_STATUS_INVALID_ARGUMENT;
        std::memcpy(diagnostics, result.data(), sizeof(result));
        return IM_STATUS_OK;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_native_acr_gray_log_rgbf32(
    uint32_t width, uint32_t height, const float *rgb, size_t rgb_samples,
    const double *coefficients, size_t coefficient_count, double source_epsilon,
    float *log_gray, size_t log_samples, double *diagnostics, size_t diagnostic_count) {
    return gray_log_impl(width, height, rgb, rgb_samples, coefficients, coefficient_count,
                         source_epsilon, log_gray, log_samples, diagnostics, diagnostic_count,
                         0, 0, 0);
}

im_status im_native_acr_gray_u16_log_rgbf32(
    uint32_t width, uint32_t height, const float *rgb, size_t rgb_samples,
    const double *coefficients, size_t coefficient_count, double source_epsilon,
    int32_t origin_x, int32_t origin_y, uint32_t dither,
    float *log_gray, size_t log_samples, double *diagnostics, size_t diagnostic_count) {
    if (dither > 1) return IM_STATUS_INVALID_ARGUMENT;
    return gray_log_impl(width, height, rgb, rgb_samples, coefficients, coefficient_count,
                         source_epsilon, log_gray, log_samples, diagnostics, diagnostic_count,
                         dither ? 2 : 1, origin_x, origin_y);
}
