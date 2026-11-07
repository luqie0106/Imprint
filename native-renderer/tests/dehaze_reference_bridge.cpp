#include "dehaze_reference.hpp"
#include "imprint_renderer.h"

#include <cmath>
#include <cstdint>

extern "C" IMPRINT_API int im_native_dehaze_reference_run(
    uint32_t width, uint32_t height, const uint16_t *source,
    const im_dehaze_params *params, uint16_t *destination, float *stats_out) {
    if (!width || !height || width > 65535 || height > 65535 ||
        static_cast<uint64_t>(width) * height > (1ull << 29) / 3 ||
        !source || !params || !destination || !stats_out) return 1;
    const float values[] = {params->strength, params->naturalness, params->fog_retention,
        params->local_contrast, params->color_recovery, params->color_protection,
        params->highlight_protection, params->shadow_protection, params->brightness_protection};
    for (float value : values) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) return 1;
    }
    try {
        const size_t pixels = static_cast<size_t>(width) * height;
        imprint::ImageStats stats = imprint::dehaze_image_stats(source, pixels);
        stats.brightness_gain = imprint::dehaze_brightness_gain(source, pixels, *params, stats);
        stats_out[0] = stats.air_r;
        stats_out[1] = stats.air_g;
        stats_out[2] = stats.air_b;
        stats_out[3] = stats.haze_level;
        stats_out[4] = stats.brightness_gain;
        imprint::apply_dehaze_reference(source, pixels, *params, stats, destination);
        return 0;
    } catch (...) {
        return 2;
    }
}
