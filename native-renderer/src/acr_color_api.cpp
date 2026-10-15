#include "imprint_renderer.h"
#include "acr_color_kernels.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {
bool valid_extent(uint32_t width, uint32_t height, const float *source,
                  size_t source_samples, const double *controls, size_t count,
                  size_t expected_controls, float *destination, size_t destination_samples) {
    const uint64_t samples = static_cast<uint64_t>(width) * height * 3;
    if (!width || !height || !source || !destination || !controls ||
        count != expected_controls || samples > (1ull << 29) ||
        samples != source_samples || samples != destination_samples) return false;
    const auto a = reinterpret_cast<uintptr_t>(source);
    const auto b = reinterpret_cast<uintptr_t>(destination);
    const uint64_t bytes = samples * sizeof(float);
    if (bytes > std::numeric_limits<uintptr_t>::max() - a ||
        bytes > std::numeric_limits<uintptr_t>::max() - b) return false;
    return !(a < b + bytes && b < a + bytes);
}
}

im_status im_native_acr_white_temperature(double x, double y, double *output,
                                           size_t output_values) {
    if (!output || output_values != 2) return IM_STATUS_INVALID_ARGUMENT;
    try {
        const auto result = imprint::acr::white_temperature(x, y);
        output[0] = result.temperature;
        output[1] = result.tint;
        return IM_STATUS_OK;
    } catch (const std::invalid_argument &) {
        return IM_STATUS_INVALID_ARGUMENT;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

static im_status hsl_api(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *controls, size_t count, float *destination, size_t destination_samples,
    bool rendered_input) {
    if ((count != 24 && count != 25) ||
        !valid_extent(width, height, source, source_samples, controls, count, count,
                      destination, destination_samples)) return IM_STATUS_INVALID_ARGUMENT;
    try {
        imprint::acr::HslParams params;
        if (count == 25) params.vibrance = controls[24];
        for (size_t i = 0; i < 8; ++i) {
            params.hue[i] = controls[i];
            params.saturation[i] = controls[8+i];
            params.luminance[i] = controls[16+i];
        }
        if (rendered_input) {
            imprint::acr::apply_rendered_hsl(source, destination, source_samples/3, params);
        } else {
            imprint::acr::apply_hsl(source, destination, source_samples/3, params);
        }
        return IM_STATUS_OK;
    } catch (const std::invalid_argument &) {
        return IM_STATUS_INVALID_ARGUMENT;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_native_acr_hsl_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *controls, size_t count, float *destination, size_t destination_samples) {
    return hsl_api(width, height, source, source_samples, controls, count,
                   destination, destination_samples, false);
}

im_status im_native_acr_rendered_hsl_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *controls, size_t count, float *destination, size_t destination_samples) {
    return hsl_api(width, height, source, source_samples, controls, count,
                   destination, destination_samples, true);
}

im_status im_native_acr_grading_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *controls, size_t count, float *destination, size_t destination_samples) {
    if (!valid_extent(width, height, source, source_samples, controls, count, 14,
                      destination, destination_samples)) return IM_STATUS_INVALID_ARGUMENT;
    try {
        imprint::acr::GradingParams params;
        params.shadows = {controls[0], controls[1], controls[2]};
        params.midtones = {controls[3], controls[4], controls[5]};
        params.highlights = {controls[6], controls[7], controls[8]};
        params.global = {controls[9], controls[10], controls[11]};
        params.blend = controls[12];
        params.balance = controls[13];
        imprint::acr::apply_grading(source, destination, source_samples/3, params);
        return IM_STATUS_OK;
    } catch (const std::invalid_argument &) {
        return IM_STATUS_INVALID_ARGUMENT;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}
