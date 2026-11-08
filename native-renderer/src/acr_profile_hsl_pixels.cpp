#include "imprint_renderer.h"
#include "acr_color_kernels.hpp"
#include "acr_profile_hsl_controls.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {
bool valid_span(const void* pointer, std::size_t bytes, std::size_t alignment) noexcept {
    if (pointer == nullptr) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return address % alignment == 0 && bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) noexcept {
    const auto left = reinterpret_cast<std::uintptr_t>(a);
    const auto right = reinterpret_cast<std::uintptr_t>(b);
    return left <= right ? right - left < a_bytes : left - right < b_bytes;
}
}  // namespace

im_status im_native_acr_profile_hsl_sdr_rgbf32(
    uint32_t width, uint32_t height, const float* source, size_t source_samples,
    const double* current, size_t current_count,
    const int32_t* profile_defaults, size_t default_count, double profile_scalar,
    float* destination, size_t destination_samples) {
    const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
    if (width == 0 || height == 0 || pixels > (std::uint64_t{1} << 29) / 3 ||
        current_count != 25 || default_count != 24 || !std::isfinite(profile_scalar) ||
        profile_scalar < 0.0 || profile_scalar > 2.0) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    // Bound the pixel count before multiplying, including on 32-bit callers.
    const auto samples = static_cast<std::size_t>(pixels * 3);
    const auto pixel_bytes = samples * sizeof(float);
    constexpr std::size_t current_bytes = 25 * sizeof(double);
    constexpr std::size_t default_bytes = 24 * sizeof(std::int32_t);
    if (source_samples != samples || destination_samples != samples ||
        !valid_span(source, pixel_bytes, alignof(float)) ||
        !valid_span(destination, pixel_bytes, alignof(float)) ||
        !valid_span(current, current_bytes, alignof(double)) ||
        !valid_span(profile_defaults, default_bytes, alignof(std::int32_t)) ||
        overlaps(source, pixel_bytes, destination, pixel_bytes) ||
        overlaps(current, current_bytes, destination, pixel_bytes) ||
        overlaps(profile_defaults, default_bytes, destination, pixel_bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    try {
        imprint::acr::ProfileHslControlsInput input{};
        input.profile_scalar = profile_scalar;
        // This ABI is explicitly the prepared color SDR subset. It neither
        // selects a profile nor exposes the monochrome/gray branches.
        input.monochrome_profile = false;
        input.convert_to_gray = 0;
        for (std::size_t lane = 0; lane < current_count; ++lane) {
            const double value = current[lane];
            if (!std::isfinite(value) || value < -100.0 || value > 100.0 ||
                (lane < 24 && std::trunc(value) != value)) {
                return IM_STATUS_INVALID_ARGUMENT;
            }
            if (lane < 24) {
                input.current[lane] = static_cast<std::int32_t>(value);
                input.profile_defaults[lane] = profile_defaults[lane];
            }
        }
        imprint::acr::ProfileHslControlsResult controls{};
        if (imprint::acr::process_profile_hsl_controls(input, controls) !=
            imprint::acr::ProfileHslControlsStatus::ok) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        imprint::acr::HslParams params{};
        for (std::size_t lane = 0; lane < 8; ++lane) {
            params.hue[lane] = controls.values[lane];
            params.saturation[lane] = controls.values[8 + lane];
            params.luminance[lane] = controls.values[16 + lane];
        }
        params.vibrance = current[24];
        // apply_hsl validates all controls and the entire source, then builds
        // all three LUTs before its first destination write. Its pixel loop
        // does not allocate or throw, so validation/allocation failures remain
        // atomic without adding a second full-resolution pixel buffer.
        imprint::acr::apply_hsl(source, destination, static_cast<std::size_t>(pixels), params);
        return IM_STATUS_OK;
    } catch (const std::invalid_argument&) {
        return IM_STATUS_INVALID_ARGUMENT;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}
