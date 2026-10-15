#include "../src/acr_color_kernels.hpp"
#include "../src/acr_tone_kernels.hpp"
#include "../src/acr_tone_mapping.hpp"
#include "../src/acr_camera_kernels.hpp"
#include "../src/acr_tone_pixels.hpp"
#include "imprint_renderer.h"

#include <array>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

extern "C" IMPRINT_API int acr_test_camera_float(const float *source, float *output,
        std::size_t pixels, const float *controls) {
    if (!controls || !source || !output || !pixels || pixels > (1u<<26)) return 1;
    try {
        imprint::acr::CameraFloatParams params{};
        params.gain_max = controls[0]; params.gain_middle = controls[1];
        params.upper = controls[2]; params.scale = controls[3];
        std::copy(controls+4,controls+13,params.matrix.begin());
        imprint::acr::apply_camera_float(source,output,pixels,params);
        return 0;
    } catch (...) { return 1; }
}

extern "C" IMPRINT_API int acr_test_tone_rgb(const double *parameters,
        double epsilon, const float *rgb3, const float *delta_local4,
        std::size_t count, float *output) {
    if (!parameters || !rgb3 || !delta_local4 || !output || !count || count > (1u<<24)) return 1;
    try {
        for (std::size_t i = 0; i < 12; ++i) {
            if (!std::isfinite(parameters[i])) return 1;
            if (i >= 9 && std::abs(parameters[i]) > std::numeric_limits<float>::max()) return 1;
        }
        imprint::acr::ToneMappingParameters params{};
        params.source_stats = {parameters[0],parameters[1],parameters[2],0,0,false};
        params.mask_aligned_stats = {parameters[3],parameters[4],parameters[5],0,0,false};
        params.ev90 = parameters[6]; params.ev98 = parameters[7]; params.ev100 = parameters[8];
        params.tone_708 = static_cast<float>(parameters[9]);
        params.tone_70c = static_cast<float>(parameters[10]);
        params.tone_710 = static_cast<float>(parameters[11]);
        const auto model = imprint::acr::build_tone_mapping(params);
        const auto tables = imprint::acr::make_tone_pixel_tables(epsilon,params.ev90);
        std::vector<float> result(count*3);
        for (std::size_t i = 0; i < count; ++i) {
            const auto *v = delta_local4+4*i;
            const std::array<float,3> rgb{rgb3[3*i],rgb3[3*i+1],rgb3[3*i+2]};
            const auto pixel = imprint::acr::tone_rgb_pixel(tables,model,rgb,v[0],v[1],v[2],v[3]);
            std::copy(pixel.begin(),pixel.end(),result.begin()+3*i);
        }
        std::copy(result.begin(),result.end(),output);
        return 0;
    } catch (...) { return 1; }
}

// Research-only mapping inputs: source C/low/high, aligned-mask C/low/high,
// EV90/98/100 and internal 708/70c/710. Sample lanes: x, delta, local3/4/5.
extern "C" IMPRINT_API int acr_test_tone_mapping(const double *parameters,
        const float *samples5, std::size_t count, float *output, double *metadata) {
    if (!parameters || !samples5 || !output || !metadata || !count || count > (1u<<24)) return 1;
    try {
        for (std::size_t i = 0; i < 12; ++i) {
            if (!std::isfinite(parameters[i])) return 1;
            if (i >= 9 && std::abs(parameters[i]) > std::numeric_limits<float>::max()) return 1;
        }
        imprint::acr::ToneMappingParameters params{};
        params.source_stats = {parameters[0],parameters[1],parameters[2],0,0,false};
        params.mask_aligned_stats = {parameters[3],parameters[4],parameters[5],0,0,false};
        params.ev90 = parameters[6]; params.ev98 = parameters[7]; params.ev100 = parameters[8];
        params.tone_708 = static_cast<float>(parameters[9]);
        params.tone_70c = static_cast<float>(parameters[10]);
        params.tone_710 = static_cast<float>(parameters[11]);
        const auto model = imprint::acr::build_tone_mapping(params);
        std::vector<float> result(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto *v = samples5+5*i;
            result[i] = imprint::acr::evaluate_tone_mapping(model,v[0],v[1],v[2],v[3],v[4]);
        }
        const double fields[] = {model.center,model.low_node,model.high_node,
            model.inner_low,model.inner_high,model.threshold,model.minimum_gate,
            model.width_factor,model.mix,model.gaussian_coefficient};
        std::copy(result.begin(),result.end(),output);
        std::copy(fields,fields+10,metadata);
        return 0;
    } catch (...) { return 1; }
}

extern "C" IMPRINT_API int acr_test_tone_resize(std::uint32_t width, std::uint32_t height,
        const float *input, std::size_t samples, std::uint32_t output_width,
        std::uint32_t output_height, float *output, std::size_t output_samples) {
    if (!output || samples != std::uint64_t{width}*height ||
        output_samples != std::uint64_t{output_width}*output_height) return 1;
    try {
        const auto result = imprint::acr::resize_gray(input, width, height, output_width, output_height);
        std::copy(result.pixels.begin(), result.pixels.end(), output);
        return 0;
    } catch (...) { return 1; }
}

extern "C" IMPRINT_API int acr_test_log_stats(const float *input, std::size_t samples,
        double epsilon, double *output) {
    if (!output) return 1;
    try {
        const auto result = imprint::acr::build_log_scene_stats(input, samples, epsilon);
        const double values[] = {result.center,result.low,result.high,result.median,
                                 static_cast<double>(result.effective_count),result.fallback ? 1.0 : 0.0};
        std::copy(values,values+6,output);
        return 0;
    } catch (...) { return 1; }
}

extern "C" IMPRINT_API int acr_test_gray_log(std::uint32_t width, std::uint32_t height,
        const float *input, std::size_t samples, double gain_ev, double epsilon,
        float *output, std::size_t output_samples) {
    if (!output || samples != std::uint64_t{width}*height*3 ||
        output_samples != std::uint64_t{width}*height) return 1;
    try {
        const auto result = imprint::acr::make_gray_log(input, width, height, gain_ev, epsilon);
        std::copy(result.pixels.begin(), result.pixels.end(), output);
        return 0;
    } catch (...) { return 1; }
}

extern "C" IMPRINT_API int acr_test_flare(const float *input, std::size_t pixels,
        std::uint32_t channels, int source_mode, double *output) {
    if (!output) return 1;
    try {
        const auto stored = imprint::acr::calculate_flare_for_log(input, pixels, channels);
        const auto epsilon = imprint::acr::select_log_epsilon(source_mode, stored);
        output[0] = stored;
        output[1] = epsilon;
        return 0;
    } catch (...) { return 1; }
}

extern "C" IMPRINT_API int acr_test_local_laplacian(std::uint32_t width, std::uint32_t height,
        const float *input, std::size_t samples, const double *controls,
        float *output, std::size_t output_samples) {
    if (!controls || !output || samples != std::uint64_t{width}*height || output_samples != samples) return 1;
    try {
        const auto result = imprint::acr::local_laplacian_gray(input, width, height,
                controls[0], controls[1], controls[2], controls[3], controls[4]);
        std::copy(result.pixels.begin(), result.pixels.end(), output);
        return 0;
    } catch (...) { return 1; }
}

extern "C" IMPRINT_API int acr_test_delta_mask(std::uint32_t width, std::uint32_t height,
        const float *original, const float *collapsed, std::size_t samples,
        double original_center, double mask_center, float *output, std::size_t output_samples) {
    if (!output || samples != std::uint64_t{width}*height || output_samples != samples) return 1;
    try {
        const auto result = imprint::acr::build_delta_mask(original, collapsed, width, height,
                                                          original_center, mask_center);
        std::copy(result.pixels.begin(), result.pixels.end(), output);
        return 0;
    } catch (...) { return 1; }
}

extern "C" IMPRINT_API int acr_test_local_tone_mask(std::uint32_t width, std::uint32_t height,
        const float *input, std::size_t samples, double epsilon, int mode,
        float *output, std::size_t output_samples, double *metadata) {
    if (!metadata || !output || samples != std::uint64_t{width}*height || output_samples != samples) return 1;
    try {
        const auto result = imprint::acr::build_local_tone_mask(input,width,height,epsilon,mode);
        const double fields[] = {result.source_stats.center,result.source_stats.low,result.source_stats.high,
            result.source_stats.median,result.mask_stats.center,result.mask_stats.low,result.mask_stats.high,
            result.mask_stats.median,result.aligned_mask_stats.low,result.aligned_mask_stats.high,
            result.remap_sigma,static_cast<double>(result.statistics_level)};
        std::copy(result.delta.pixels.begin(),result.delta.pixels.end(),output);
        std::copy(fields,fields+12,metadata);
        return 0;
    } catch (...) { return 1; }
}

extern "C" IMPRINT_API int acr_test_saturation_s16(const std::int16_t *input,
        std::int16_t *output, std::size_t n, double ui_saturation) {
    try {
        imprint::acr::apply_saturation_s16(input, output, n, ui_saturation);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" IMPRINT_API int acr_test_hsl(const float *input, float *output, std::size_t n,
                            const double controls24[24]) {
    try {
        if (!controls24) return 1;
        imprint::acr::HslParams params;
        for (std::size_t i = 0; i < 8; ++i) {
            params.hue[i] = controls24[i];
            params.saturation[i] = controls24[8 + i];
            params.luminance[i] = controls24[16 + i];
        }
        imprint::acr::apply_hsl(input, output, n, params);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" IMPRINT_API int acr_test_hsl_vibrance(const float *input, float *output,
                                       std::size_t n, const double controls25[25]) {
    try {
        if (!controls25) return 1;
        imprint::acr::HslParams params;
        for (std::size_t i = 0; i < 8; ++i) {
            params.hue[i] = controls25[i];
            params.saturation[i] = controls25[8 + i];
            params.luminance[i] = controls25[16 + i];
        }
        params.vibrance = controls25[24];
        imprint::acr::apply_hsl(input, output, n, params);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" IMPRINT_API int acr_test_grading(const float *input, float *output, std::size_t n,
                                const double controls14[14]) {
    try {
        if (!controls14) return 1;
        imprint::acr::GradingParams params;
        params.shadows = {controls14[0], controls14[1], controls14[2]};
        params.midtones = {controls14[3], controls14[4], controls14[5]};
        params.highlights = {controls14[6], controls14[7], controls14[8]};
        params.global = {controls14[9], controls14[10], controls14[11]};
        params.blend = controls14[12];
        params.balance = controls14[13];
        imprint::acr::apply_grading(input, output, n, params);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" IMPRINT_API int acr_test_rendered_hsl(const float *input, float *output,
                                                 std::size_t n, const double controls25[25]) {
    try {
        if (!controls25) return 1;
        imprint::acr::HslParams params;
        for (std::size_t i = 0; i < 8; ++i) {
            params.hue[i] = controls25[i];
            params.saturation[i] = controls25[8+i];
            params.luminance[i] = controls25[16+i];
        }
        params.vibrance = controls25[24];
        imprint::acr::apply_rendered_hsl(input, output, n, params);
        return 0;
    } catch (...) {
        return 1;
    }
}
