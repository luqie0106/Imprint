#include "imprint_renderer.h"
#include "acr_camera_kernels.hpp"
#include "acr_tone_kernels.hpp"
#include "acr_tone_mapping.hpp"
#include "acr_tone_pixels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool byte_range(const void *pointer, size_t bytes, uintptr_t *begin, uintptr_t *end) {
    if (!pointer || !bytes || !begin || !end) return false;
    const uintptr_t first = reinterpret_cast<uintptr_t>(pointer);
    if (bytes > std::numeric_limits<uintptr_t>::max() - first) return false;
    *begin = first;
    *end = first + bytes;
    return true;
}

bool ranges_overlap(const void *left, size_t left_bytes,
                    const void *right, size_t right_bytes, bool *valid) {
    uintptr_t left_begin = 0, left_end = 0, right_begin = 0, right_end = 0;
    const bool ranges_valid = byte_range(left, left_bytes, &left_begin, &left_end) &&
                              byte_range(right, right_bytes, &right_begin, &right_end);
    if (valid) *valid = ranges_valid;
    return ranges_valid && left_begin < right_end && right_begin < left_end;
}

bool checked_samples(uint32_t width, uint32_t height, size_t *samples) {
    if (!width || !height || !samples) return false;
    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    if (w > std::numeric_limits<size_t>::max() / h) return false;
    const size_t pixels = w * h;
    if (pixels > std::numeric_limits<size_t>::max() / 3) return false;
    const size_t count = pixels * 3;
    if (count > (size_t{1} << 29) || count > std::vector<float>().max_size()) return false;
    *samples = count;
    return true;
}

bool float_control(double value, float *result) {
    if (!result || !std::isfinite(value) ||
        value > static_cast<double>(std::numeric_limits<float>::max()) ||
        value < -static_cast<double>(std::numeric_limits<float>::max())) return false;
    const float converted = static_cast<float>(value);
    if (!std::isfinite(converted)) return false;
    *result = converted;
    return true;
}

bool finite_nonnegative(const float *values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]) || values[i] < 0.0f) return false;
    }
    return true;
}

template <typename T>
bool is_aligned(const T *pointer) {
    return reinterpret_cast<uintptr_t>(pointer) % alignof(T) == 0;
}

}  // namespace

im_status im_native_acr_camera_float_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *controls, size_t control_count, float *destination,
    size_t destination_samples) {
    size_t samples = 0;
    if (!checked_samples(width, height, &samples) || !source || !destination ||
        !controls || control_count != 13 || source_samples != samples ||
        destination_samples != samples ||
        samples > std::numeric_limits<size_t>::max() / sizeof(float) ||
        control_count > std::numeric_limits<size_t>::max() / sizeof(double)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    const size_t image_bytes = samples * sizeof(float);
    const size_t control_bytes = control_count * sizeof(double);
    bool ranges_valid = false;
    const bool source_destination_overlap = ranges_overlap(
        source, image_bytes, destination, image_bytes, &ranges_valid);
    if (!ranges_valid || source_destination_overlap) return IM_STATUS_INVALID_ARGUMENT;
    const bool destination_controls_overlap = ranges_overlap(
        destination, image_bytes, controls, control_bytes, &ranges_valid);
    if (!ranges_valid || destination_controls_overlap) return IM_STATUS_INVALID_ARGUMENT;

    imprint::acr::CameraFloatParams parameters{};
    if (!float_control(controls[0], &parameters.gain_max) ||
        !float_control(controls[1], &parameters.gain_middle) ||
        !float_control(controls[2], &parameters.upper) ||
        !float_control(controls[3], &parameters.scale)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < parameters.matrix.size(); ++i) {
        if (!float_control(controls[4 + i], &parameters.matrix[i])) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
    }

    try {
        std::vector<float> temporary(samples);
        imprint::acr::apply_camera_float(source, temporary.data(), samples / 3, parameters);
        std::copy(temporary.begin(), temporary.end(), destination);
        return IM_STATUS_OK;
    } catch (const std::invalid_argument &) {
        return IM_STATUS_INVALID_ARGUMENT;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

namespace {

im_status native_acr_tone_prepared_sdr_impl(
    uint32_t width, uint32_t height, const float *guide_rgb, const float *tone_rgb,
    size_t samples, const double *controls, size_t control_count,
    float *destination, size_t destination_samples, bool mode_aware) {
    size_t expected_samples = 0;
    if (!checked_samples(width, height, &expected_samples) || !guide_rgb || !tone_rgb ||
        !controls || !destination ||
        control_count != (mode_aware ? 9 : 8) ||
        samples != expected_samples ||
        destination_samples != expected_samples ||
        expected_samples > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    if (control_count > std::numeric_limits<size_t>::max() / sizeof(double)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    if (!is_aligned(guide_rgb) || !is_aligned(tone_rgb) ||
        !is_aligned(controls) || !is_aligned(destination)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    const size_t image_bytes = expected_samples * sizeof(float);
    const size_t control_bytes = control_count * sizeof(double);
    bool ranges_valid = false;
    if (ranges_overlap(guide_rgb, image_bytes, destination, image_bytes, &ranges_valid) ||
        !ranges_valid) return IM_STATUS_INVALID_ARGUMENT;
    if (ranges_overlap(tone_rgb, image_bytes, destination, image_bytes, &ranges_valid) ||
        !ranges_valid) return IM_STATUS_INVALID_ARGUMENT;
    if (ranges_overlap(controls, control_bytes, destination, image_bytes, &ranges_valid) ||
        !ranges_valid) return IM_STATUS_INVALID_ARGUMENT;

    std::array<double, 8> values{};
    std::copy(controls, controls + values.size(), values.begin());
    for (double value : values) {
        if (!std::isfinite(value)) return IM_STATUS_INVALID_ARGUMENT;
    }
    const double epsilon = values[0];
    const double ev90 = values[1];
    const double ev98 = values[2];
    const double ev100 = values[3];
    const double shadows = values[4];
    const double highlights = values[5];
    const double clarity = values[6];
    const double sigma_mode = values[7];
    imprint::acr::ToneOutputMode output_mode = imprint::acr::ToneOutputMode::preserve_rgb;
    if (epsilon <= 0.0 || epsilon > 0x1p-12 ||
        static_cast<float>(epsilon) == 0.0f ||
        ev90 < -120.0 || ev90 > 120.0 ||
        ev98 < -120.0 || ev98 > 120.0 ||
        ev100 < -120.0 || ev100 > 120.0 ||
        shadows < -100.0 || shadows > 100.0 ||
        highlights < -100.0 || highlights > 100.0 ||
        clarity < 0.0 || clarity > 100.0 ||
        (sigma_mode != 0.0 && sigma_mode != 1.0 && sigma_mode != 2.0)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    if (mode_aware) {
        const double mode_value = controls[8];
        if (!std::isfinite(mode_value) || (mode_value != 0.0 && mode_value != 1.0)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        output_mode = static_cast<imprint::acr::ToneOutputMode>(
            static_cast<int>(mode_value));
    }

    try {
        if (!finite_nonnegative(guide_rgb, expected_samples) ||
            !finite_nonnegative(tone_rgb, expected_samples)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        const bool no_tone_controls = output_mode == imprint::acr::ToneOutputMode::preserve_rgb &&
            shadows == 0.0 && highlights == 0.0 && clarity == 0.0;
        std::vector<float> temporary(expected_samples);
        if (no_tone_controls) {
            std::copy(tone_rgb, tone_rgb + expected_samples, temporary.begin());
        } else {
            const size_t pixels = expected_samples / 3;
            if (pixels < 2 || width > 65535 || height > 65535 || pixels > (size_t{1} << 26)) {
                return IM_STATUS_INVALID_ARGUMENT;
            }

            const imprint::acr::GrayImage log_guide = imprint::acr::make_gray_log(
                guide_rgb, width, height, ev90, epsilon);
            const imprint::acr::LocalToneMask mask = imprint::acr::build_local_tone_mask(
                log_guide.pixels.data(), width, height, epsilon, static_cast<int>(sigma_mode));
            imprint::acr::ToneMappingParameters mapping_parameters{};
            mapping_parameters.source_stats = mask.source_stats;
            mapping_parameters.mask_aligned_stats = mask.aligned_mask_stats;
            mapping_parameters.ev90 = ev90;
            mapping_parameters.ev98 = ev98;
            mapping_parameters.ev100 = ev100;
            mapping_parameters.tone_708 = static_cast<float>(shadows / 100.0);
            mapping_parameters.tone_70c = static_cast<float>(highlights / -100.0);
            mapping_parameters.tone_710 = static_cast<float>(clarity / 100.0);
            const imprint::acr::ToneMapping mapping =
                imprint::acr::build_tone_mapping(mapping_parameters);
            const imprint::acr::TonePixelTables tables =
                imprint::acr::make_tone_pixel_tables(epsilon, ev90);

            for (size_t i = 0; i < pixels; ++i) {
                const std::array<float, 3> rgb{{tone_rgb[3 * i], tone_rgb[3 * i + 1],
                                                tone_rgb[3 * i + 2]}};
                const std::array<float, 3> result = imprint::acr::tone_rgb_pixel(
                    tables, mapping, rgb, mask.delta.pixels[i], 0.0f, 0.0f, 0.0f,
                    output_mode);
                for (size_t channel = 0; channel < 3; ++channel) {
                    temporary[3 * i + channel] = result[channel];
                }
            }
        }

        if (!std::all_of(temporary.begin(), temporary.end(),
                         [](float value) { return std::isfinite(value); })) {
            return IM_STATUS_RUNTIME_ERROR;
        }
        std::copy(temporary.begin(), temporary.end(), destination);
        return IM_STATUS_OK;
    } catch (const std::invalid_argument &) {
        return IM_STATUS_INVALID_ARGUMENT;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

}  // namespace

im_status im_native_acr_tone_prepared_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *guide_rgb, const float *tone_rgb,
    size_t samples, const double *controls, size_t control_count,
    float *destination, size_t destination_samples) {
    return native_acr_tone_prepared_sdr_impl(
        width, height, guide_rgb, tone_rgb, samples, controls, control_count,
        destination, destination_samples, false);
}

im_status im_native_acr_tone_prepared_sdr_mode_rgbf32(
    uint32_t width, uint32_t height, const float *guide_rgb, const float *tone_rgb,
    size_t samples, const double *controls, size_t control_count,
    float *destination, size_t destination_samples) {
    return native_acr_tone_prepared_sdr_impl(
        width, height, guide_rgb, tone_rgb, samples, controls, control_count,
        destination, destination_samples, true);
}
