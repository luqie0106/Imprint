#include "dehaze_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace imprint {
namespace {

constexpr float kLumaR = 0.2126f;
constexpr float kLumaG = 0.7152f;
constexpr float kLumaB = 0.0722f;
constexpr float kPeak = 65535.0f;

struct Vec3 {
    float r;
    float g;
    float b;
};

float clamp01(float value) { return std::min(1.0f, std::max(0.0f, value)); }
float luma(Vec3 value) { return kLumaR * value.r + kLumaG * value.g + kLumaB * value.b; }
float smoothstep01(float value) {
    const float t = clamp01(value);
    return t * t * (3.0f - 2.0f * t);
}
float smoothstep(float low, float high, float value) {
    return smoothstep01((value - low) / (high - low));
}

Vec3 read_pixel(const uint16_t *rgb16, size_t pixel) {
    const size_t at = pixel * 3;
    return {rgb16[at] / kPeak, rgb16[at + 1] / kPeak, rgb16[at + 2] / kPeak};
}

float median(std::vector<float> &values) {
    if (values.empty()) return 0.0f;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const float upper = values[middle];
    if (values.size() & 1u) return upper;
    const float lower = *std::max_element(values.begin(), values.begin() + middle);
    return (lower + upper) * 0.5f;
}

Vec3 smooth_chroma_gamut(Vec3 image) {
    const float y = luma(image);
    const Vec3 chroma{image.r - y, image.g - y, image.b - y};
    float limit = std::numeric_limits<float>::infinity();
    const float channels[] = {chroma.r, chroma.g, chroma.b};
    for (float value : channels) {
        if (value > 1e-7f) limit = std::min(limit, (1.0f - y) / value);
        else if (value < -1e-7f) limit = std::min(limit, y / -value);
    }
    if (!std::isfinite(limit)) limit = 1.0f;
    const float delta = 1.0f - limit;
    const float scale = clamp01(0.5f * (1.0f + limit - std::sqrt(delta * delta + 1e-10f)));
    return {y + chroma.r * scale, y + chroma.g * scale, y + chroma.b * scale};
}

Vec3 smooth_chroma_caps(Vec3 image, Vec3 caps) {
    const float y = luma(image);
    const Vec3 chroma{image.r - y, image.g - y, image.b - y};
    float limit = std::numeric_limits<float>::infinity();
    const float values[] = {chroma.r, chroma.g, chroma.b};
    const float bounds[] = {caps.r, caps.g, caps.b};
    for (size_t channel = 0; channel < 3; ++channel) {
        if (values[channel] > 1e-7f) limit = std::min(limit, (bounds[channel] - y) / values[channel]);
        else if (values[channel] < -1e-7f) limit = std::min(limit, y / -values[channel]);
    }
    if (!std::isfinite(limit)) limit = 1.0f;
    const float delta = 1.0f - limit;
    const float scale = clamp01(0.5f * (1.0f + limit - std::sqrt(delta * delta + 1e-10f)));
    return {y + chroma.r * scale, y + chroma.g * scale, y + chroma.b * scale};
}

Vec3 pre_brightness_pixel(Vec3 source, const im_dehaze_params &p, const ImageStats &stats) {
    if (p.strength <= 1e-6f) return source;

    const float source_y = luma(source);
    Vec3 atmosphere{stats.air_r, stats.air_g, stats.air_b};
    const float atmosphere_mean = (atmosphere.r + atmosphere.g + atmosphere.b) / 3.0f;
    const float neutral_mix = 0.45f * p.color_protection * (0.65f + 0.35f * p.naturalness);
    atmosphere = {
        clamp01(atmosphere.r * (1.0f - neutral_mix) + atmosphere_mean * neutral_mix),
        clamp01(atmosphere.g * (1.0f - neutral_mix) + atmosphere_mean * neutral_mix),
        clamp01(atmosphere.b * (1.0f - neutral_mix) + atmosphere_mean * neutral_mix),
    };
    atmosphere.r = std::max(0.35f, atmosphere.r);
    atmosphere.g = std::max(0.35f, atmosphere.g);
    atmosphere.b = std::max(0.35f, atmosphere.b);

    const float omega = p.strength * (0.66f - 0.14f * p.fog_retention) *
                        (0.90f + 0.10f * (1.0f - p.naturalness)) *
                        (0.92f + 0.08f * stats.haze_level);
    const float transmission_floor = 0.27f + 0.21f * p.fog_retention + 0.11f * p.naturalness;
    const float transmission = std::min(1.0f, std::max(transmission_floor, 1.0f - omega));
    const Vec3 recovered{
        clamp01((source.r - atmosphere.r) / transmission + atmosphere.r),
        clamp01((source.g - atmosphere.g) / transmission + atmosphere.g),
        clamp01((source.b - atmosphere.b) / transmission + atmosphere.b),
    };

    const float blend = std::min(0.82f, std::max(0.0f, p.strength * (0.78f - 0.28f * p.fog_retention) *
                                                          (0.90f + 0.10f * (1.0f - p.naturalness))));
    Vec3 natural{
        source.r * (1.0f - blend) + recovered.r * blend,
        source.g * (1.0f - blend) + recovered.g * blend,
        source.b * (1.0f - blend) + recovered.b * blend,
    };

    const float enhanced_y = luma(natural);
    const float luma_ratio = enhanced_y / std::max(source_y, 1e-4f);
    const Vec3 luma_only{clamp01(source.r * luma_ratio), clamp01(source.g * luma_ratio), clamp01(source.b * luma_ratio)};
    const float chroma_mix = p.color_protection * (0.72f + 0.28f * p.naturalness);
    natural = {
        natural.r * (1.0f - chroma_mix) + luma_only.r * chroma_mix,
        natural.g * (1.0f - chroma_mix) + luma_only.g * chroma_mix,
        natural.b * (1.0f - chroma_mix) + luma_only.b * chroma_mix,
    };

    if (p.color_recovery > 1e-6f) {
        const Vec3 source_chroma{source.r - source_y, source.g - source_y, source.b - source_y};
        const float source_chroma_norm = std::sqrt(source_chroma.r * source_chroma.r +
                                                   source_chroma.g * source_chroma.g +
                                                   source_chroma.b * source_chroma.b);
        float confidence = clamp01((source_chroma_norm - 0.006f) / 0.084f);
        confidence = confidence * confidence * (3.0f - 2.0f * confidence);
        const float source_norm_floor = std::max(source_chroma_norm, 1e-6f);
        const Vec3 source_direction{source_chroma.r / source_norm_floor,
                                    source_chroma.g / source_norm_floor,
                                    source_chroma.b / source_norm_floor};

        const float natural_y = luma(natural);
        const Vec3 natural_chroma{natural.r - natural_y, natural.g - natural_y, natural.b - natural_y};
        const float natural_chroma_norm = std::sqrt(natural_chroma.r * natural_chroma.r +
                                                    natural_chroma.g * natural_chroma.g +
                                                    natural_chroma.b * natural_chroma.b);
        const float aligned_chroma = std::max(0.0f, natural_chroma.r * source_direction.r +
                                                       natural_chroma.g * source_direction.g +
                                                       natural_chroma.b * source_direction.b);
        const float recovery_amount = p.color_recovery * p.strength;
        const float source_target_norm = source_chroma_norm * (1.0f + 1.80f * p.color_recovery * confidence);
        const float natural_target_norm = natural_chroma_norm * (1.0f + 0.30f * recovery_amount * confidence) * confidence;
        const float target_chroma_norm = std::max(std::max(aligned_chroma * confidence, natural_target_norm),
                                                  source_target_norm);
        const Vec3 target_chroma{source_direction.r * target_chroma_norm,
                                 source_direction.g * target_chroma_norm,
                                 source_direction.b * target_chroma_norm};

        float highlight_position = smoothstep(0.58f, 0.98f, source_y);
        float shadow_position = smoothstep(0.0f, 0.26f, 0.26f - source_y);
        const float protection = (1.0f - highlight_position * p.highlight_protection) *
                                 (1.0f - shadow_position * p.shadow_protection);
        const float requested_recovery = std::min(0.95f, std::max(0.0f,
            recovery_amount * protection * (0.95f + 0.35f * (1.0f - confidence))));
        const float neutral_guard = p.color_protection * (1.0f - confidence) *
                                    (1.08f + 0.12f * p.naturalness) * protection;
        const float correction_strength = std::min(0.95f, std::max(requested_recovery, neutral_guard));
        natural = {
            natural_y + natural_chroma.r * (1.0f - correction_strength) + target_chroma.r * correction_strength,
            natural_y + natural_chroma.g * (1.0f - correction_strength) + target_chroma.g * correction_strength,
            natural_y + natural_chroma.b * (1.0f - correction_strength) + target_chroma.b * correction_strength,
        };
        natural = smooth_chroma_gamut(natural);
    }

    if (p.local_contrast > 1e-6f) {
        const float y = luma(natural);
        const float contrast_amount = 0.55f * p.local_contrast;
        const float contrast_y = y + contrast_amount * (2.0f * y - 1.0f) * y * (1.0f - y);
        const float ratio = contrast_y / std::max(y, 1e-4f);
        natural = {natural.r * ratio, natural.g * ratio, natural.b * ratio};
    }

    const float highlight_position = smoothstep(0.58f, 0.98f, source_y);
    const float highlight_blend = clamp01(highlight_position * p.highlight_protection);
    natural = {
        natural.r * (1.0f - highlight_blend) + source.r * highlight_blend,
        natural.g * (1.0f - highlight_blend) + source.g * highlight_blend,
        natural.b * (1.0f - highlight_blend) + source.b * highlight_blend,
    };
    const float shadow_position = smoothstep(0.0f, 0.26f, 0.26f - source_y);
    const float shadow_blend = clamp01(shadow_position * p.shadow_protection);
    natural = {
        natural.r * (1.0f - shadow_blend) + source.r * shadow_blend,
        natural.g * (1.0f - shadow_blend) + source.g * shadow_blend,
        natural.b * (1.0f - shadow_blend) + source.b * shadow_blend,
    };
    return natural;
}

Vec3 post_brightness_pixel(Vec3 source, Vec3 image, const im_dehaze_params &p, float gain) {
    if (gain > 1.0f && p.brightness_protection > 1e-6f) {
        const float y = luma(image);
        const float curve_y = y * gain / (1.0f + (gain - 1.0f) * y);
        const float ratio = y > 1e-6f ? curve_y / y : 1.0f;
        image = {clamp01(image.r * ratio), clamp01(image.g * ratio), clamp01(image.b * ratio)};
        image = smooth_chroma_gamut(image);
    }

    constexpr float saturation_threshold = 0.97f;
    const float saturation_limit = std::max(0.0f, saturation_threshold - 1.5f / kPeak);
    const float source_y = luma(source);
    const float final_y = luma(image);
    const float luma_cap = source_y < saturation_threshold ? saturation_limit : 1.0f;
    const float luma_scale = std::min(1.0f, luma_cap / std::max(final_y, 1e-4f));
    image = {image.r * luma_scale, image.g * luma_scale, image.b * luma_scale};
    const Vec3 caps{source.r < saturation_threshold ? saturation_limit : 1.0f,
                    source.g < saturation_threshold ? saturation_limit : 1.0f,
                    source.b < saturation_threshold ? saturation_limit : 1.0f};
    if (p.color_recovery > 1e-6f) {
        image = smooth_chroma_caps(image, caps);
    } else {
        image = {std::min(image.r, caps.r), std::min(image.g, caps.g), std::min(image.b, caps.b)};
    }
    return {std::min(1.0f, std::max(0.0f, image.r)),
            std::min(1.0f, std::max(0.0f, image.g)),
            std::min(1.0f, std::max(0.0f, image.b))};
}

} // namespace

ImageStats dehaze_image_stats(const uint16_t *rgb16, size_t pixels) {
    ImageStats stats{0, 0, 1, 1, 1, 0.12f, 1.0f};
    if (!rgb16 || pixels == 0) return stats;

    std::vector<float> dark(pixels);
    double luma_sum = 0.0;
    float max_luma = 0.0f;
    for (size_t i = 0; i < pixels; ++i) {
        const Vec3 c = read_pixel(rgb16, i);
        dark[i] = std::min(c.r, std::min(c.g, c.b));
        const float y = luma(c);
        luma_sum += y;
        max_luma = std::max(max_luma, y);
    }

    const size_t percentile_position = static_cast<size_t>(0.75 * static_cast<double>(pixels - 1));
    const float fraction = static_cast<float>(0.75 * static_cast<double>(pixels - 1) - percentile_position);
    std::vector<float> percentile_values = dark;
    size_t selected_position = percentile_position;
    if (fraction > 0.0f) ++selected_position;
    std::nth_element(percentile_values.begin(), percentile_values.begin() + selected_position,
                     percentile_values.end());
    const float upper = percentile_values[selected_position];
    const float lower = fraction > 0.0f
                            ? *std::max_element(percentile_values.begin(), percentile_values.begin() + selected_position)
                            : upper;
    const float dark_reference = lower + (upper - lower) * fraction;
    const float haze_level = std::min(0.92f, std::max(0.12f, (dark_reference - 0.03f) / 0.92f));

    const size_t candidate_count = std::min(pixels, std::max<size_t>(16, static_cast<size_t>(pixels * 0.001)));
    std::nth_element(percentile_values.begin(), percentile_values.begin() + (pixels - candidate_count),
                     percentile_values.end());
    const float candidate_threshold = percentile_values[pixels - candidate_count];
    struct Candidate { float y; Vec3 rgb; };
    std::vector<Candidate> candidates;
    candidates.reserve(candidate_count);
    for (size_t index = 0; index < pixels; ++index) {
        if (dark[index] > candidate_threshold) {
            const Vec3 rgb = read_pixel(rgb16, index);
            candidates.push_back({luma(rgb), rgb});
        }
    }
    for (size_t index = 0; index < pixels && candidates.size() < candidate_count; ++index) {
        if (dark[index] == candidate_threshold) {
            const Vec3 rgb = read_pixel(rgb16, index);
            candidates.push_back({luma(rgb), rgb});
        }
    }
    const size_t brightest_count = std::max<size_t>(1, candidate_count / 8);
    if (brightest_count < candidates.size()) {
        std::nth_element(candidates.begin(), candidates.begin() + brightest_count, candidates.end(),
                         [](const Candidate &a, const Candidate &b) { return a.y > b.y; });
        candidates.resize(brightest_count);
    }
    std::vector<float> channels[3];
    for (auto &channel : channels) channel.reserve(candidates.size());
    for (const Candidate &candidate : candidates) {
        channels[0].push_back(candidate.rgb.r);
        channels[1].push_back(candidate.rgb.g);
        channels[2].push_back(candidate.rgb.b);
    }
    stats.mean_luma = static_cast<float>(luma_sum / static_cast<double>(pixels));
    stats.max_luma = max_luma;
    stats.air_r = std::min(1.0f, std::max(0.35f, median(channels[0])));
    stats.air_g = std::min(1.0f, std::max(0.35f, median(channels[1])));
    stats.air_b = std::min(1.0f, std::max(0.35f, median(channels[2])));
    stats.haze_level = haze_level;
    stats.brightness_gain = 1.0f;
    return stats;
}

float dehaze_brightness_gain(const uint16_t *rgb16, size_t pixels,
                             const im_dehaze_params &params, const ImageStats &stats) {
    if (!rgb16 || pixels == 0 || params.strength <= 1e-6f || params.brightness_protection <= 1e-6f) return 1.0f;
    std::vector<float> source_lumas;
    std::vector<float> image_lumas;
    source_lumas.reserve(pixels);
    image_lumas.reserve(pixels);
    for (size_t i = 0; i < pixels; ++i) {
        const Vec3 source = read_pixel(rgb16, i);
        const float source_y = luma(source);
        if (source_y <= 0.08f || source_y >= 0.88f) continue;
        const Vec3 image = pre_brightness_pixel(source, params, stats);
        const float image_y = luma(image);
        if (std::isfinite(source_y) && std::isfinite(image_y)) {
            source_lumas.push_back(source_y);
            image_lumas.push_back(image_y);
        }
    }
    if (source_lumas.size() < 8) return 1.0f;
    const float source_median = median(source_lumas);
    const float image_median = median(image_lumas);
    if (!std::isfinite(source_median) || !std::isfinite(image_median) ||
        source_median <= 1e-5f || image_median <= 1e-5f) return 1.0f;
    const float brightness_drop_ev = std::log2(source_median / image_median);
    if (!std::isfinite(brightness_drop_ev)) return 1.0f;
    const float allowed_drop_ev = 0.08f + 0.22f * clamp01(params.strength);
    const float excess_drop_ev = std::max(0.0f, brightness_drop_ev - allowed_drop_ev);
    const float compensation_ev = std::min(0.40f, excess_drop_ev * clamp01(params.brightness_protection));
    if (!std::isfinite(compensation_ev) || compensation_ev <= 1e-6f) return 1.0f;
    const float gain = std::exp2(compensation_ev);
    return std::isfinite(gain) ? gain : 1.0f;
}

void apply_dehaze_reference(const uint16_t *rgb16, size_t pixels,
                            const im_dehaze_params &params, ImageStats stats,
                            uint16_t *destination) {
    if (!rgb16 || !destination || pixels == 0) return;
    stats.brightness_gain = dehaze_brightness_gain(rgb16, pixels, params, stats);
    for (size_t i = 0; i < pixels; ++i) {
        const Vec3 source = read_pixel(rgb16, i);
        const Vec3 before_brightness = pre_brightness_pixel(source, params, stats);
        const Vec3 result = params.strength <= 1e-6f
                                ? source
                                : post_brightness_pixel(source, before_brightness, params, stats.brightness_gain);
        destination[i * 3] = static_cast<uint16_t>(std::nearbyint(clamp01(result.r) * kPeak));
        destination[i * 3 + 1] = static_cast<uint16_t>(std::nearbyint(clamp01(result.g) * kPeak));
        destination[i * 3 + 2] = static_cast<uint16_t>(std::nearbyint(clamp01(result.b) * kPeak));
    }
}

} // namespace imprint
