#include <metal_stdlib>
using namespace metal;

struct DehazeParams {
    float strength;
    float naturalness;
    float fog_retention;
    float local_contrast;
    float color_recovery;
    float color_protection;
    float highlight_protection;
    float shadow_protection;
    float brightness_protection;
};

struct BasicParams {
    float exposure;
    float contrast;
    float highlights;
    float shadows;
    float whites;
    float blacks;
    float vibrance;
    float saturation;
};

struct FilterParams {
    float curve_mix;
    float lut_mix;
    float exposure_ev;
    float contrast;
    float saturation;
    float warmth;
    float tint;
    float fade;
};

struct ImageStats {
    float mean_luma;
    float max_luma;
    float air_r;
    float air_g;
    float air_b;
    float haze_level;
    float brightness_gain;
};

static float luma(float3 rgb) {
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

static float3 with_saturation(float3 rgb, float amount) {
    float y = luma(rgb);
    return clamp(float3(y) + (rgb - float3(y)) * amount, 0.0, 1.0);
}

static float3 smooth_chroma_gamut(float3 color) {
    float y = luma(color);
    float3 chroma = color - float3(y);
    float limit = 1e30;
    for (uint channel = 0; channel < 3; ++channel) {
        if (chroma[channel] > 1e-7) limit = min(limit, (1.0 - y) / chroma[channel]);
        else if (chroma[channel] < -1e-7) limit = min(limit, y / -chroma[channel]);
    }
    if (limit > 1e20) limit = 1.0;
    float delta = 1.0 - limit;
    float scale = clamp(0.5 * (1.0 + limit - sqrt(delta * delta + 1e-10)), 0.0, 1.0);
    return float3(y) + chroma * scale;
}

static float3 smooth_chroma_caps(float3 color, float3 caps) {
    float y = luma(color);
    float3 chroma = color - float3(y);
    float limit = 1e30;
    for (uint channel = 0; channel < 3; ++channel) {
        if (chroma[channel] > 1e-7) limit = min(limit, (caps[channel] - y) / chroma[channel]);
        else if (chroma[channel] < -1e-7) limit = min(limit, y / -chroma[channel]);
    }
    if (limit > 1e20) limit = 1.0;
    float delta = 1.0 - limit;
    float scale = clamp(0.5 * (1.0 + limit - sqrt(delta * delta + 1e-10)), 0.0, 1.0);
    return float3(y) + chroma * scale;
}

static float3 apply_dehaze(float3 original, constant DehazeParams &p, constant ImageStats &stats) {
    if (p.strength <= 1e-6) return original;
    float y = luma(original);
    float3 air = float3(stats.air_r, stats.air_g, stats.air_b);
    float air_mean = (air.r + air.g + air.b) / 3.0;
    air = mix(air, float3(air_mean), 0.45 * p.color_protection * (0.65 + 0.35 * p.naturalness));
    air = clamp(air, float3(0.35), float3(1.0));
    float omega = p.strength * (0.66 - 0.14 * p.fog_retention) *
                  (0.90 + 0.10 * (1.0 - p.naturalness)) * (0.92 + 0.08 * stats.haze_level);
    float transmission_floor = 0.27 + 0.21 * p.fog_retention + 0.11 * p.naturalness;
    float transmission = clamp(1.0 - omega, transmission_floor, 1.0);
    float3 delta = original - air;
    float3 effective_transmission = transmission + (1.0 - transmission) * max(delta, float3(0.0)) /
                                    max(float3(1.0) - air, float3(1e-4));
    float3 recovered = clamp(air + delta / effective_transmission, 0.0, 1.0);
    float amount = clamp(p.strength * (0.78 - 0.28 * p.fog_retention) *
                         (0.90 + 0.10 * (1.0 - p.naturalness)), 0.0, 0.82);
    float3 natural = mix(original, recovered, amount);
    float3 luma_only = clamp(original * (luma(natural) / max(y, 1e-4)), 0.0, 1.0);
    natural = mix(natural, luma_only, p.color_protection * (0.72 + 0.28 * p.naturalness));
    if (p.color_recovery > 1e-6) {
        float3 source_chroma = original - float3(y);
        float source_chroma_norm = length(source_chroma);
        float confidence = clamp((source_chroma_norm - 0.006) / 0.084, 0.0, 1.0);
        confidence = confidence * confidence * (3.0 - 2.0 * confidence);
        float3 source_direction = source_chroma / max(source_chroma_norm, 1e-6);
        float natural_y = luma(natural);
        float3 natural_chroma = natural - float3(natural_y);
        float natural_chroma_norm = length(natural_chroma);
        float aligned_chroma = max(0.0, dot(natural_chroma, source_direction));
        float recovery_amount = p.color_recovery * p.strength;
        float source_target_norm = source_chroma_norm * (1.0 + 1.80 * p.color_recovery * confidence);
        float natural_target_norm = natural_chroma_norm * (1.0 + 0.30 * recovery_amount * confidence) * confidence;
        float target_chroma_norm = max(max(aligned_chroma * confidence, natural_target_norm), source_target_norm);
        float3 target_chroma = source_direction * target_chroma_norm;
        float highlight_position = smoothstep(0.58, 0.98, y);
        float shadow_position = smoothstep(0.0, 0.26, 0.26 - y);
        float protection = (1.0 - highlight_position * p.highlight_protection) *
                           (1.0 - shadow_position * p.shadow_protection);
        float requested_recovery = clamp(recovery_amount * protection * (0.95 + 0.35 * (1.0 - confidence)),
                                         0.0, 0.95);
        float neutral_guard = p.color_protection * p.strength * (1.0 - confidence) *
                              (1.08 + 0.12 * p.naturalness) * protection;
        float correction_strength = clamp(max(requested_recovery, neutral_guard), 0.0, 0.95);
        natural = float3(natural_y) + natural_chroma * (1.0 - correction_strength) +
                  target_chroma * correction_strength;
        natural = smooth_chroma_gamut(natural);
    }

    if (p.local_contrast > 1e-6) {
        float current_y = luma(natural);
        float contrast_amount = 0.55 * p.local_contrast * p.strength;
        float contrast_y = current_y + contrast_amount * (2.0 * current_y - 1.0) * current_y * (1.0 - current_y);
        natural *= contrast_y / max(current_y, 1e-4);
    }

    float highlight_position = smoothstep(0.58, 0.98, y);
    float highlight_blend = clamp(highlight_position * p.highlight_protection, 0.0, 1.0);
    natural = natural * (1.0 - highlight_blend) + original * highlight_blend;
    float shadow_position = smoothstep(0.0, 0.26, 0.26 - y);
    float shadow_blend = clamp(shadow_position * p.shadow_protection, 0.0, 1.0);
    natural = natural * (1.0 - shadow_blend) + original * shadow_blend;

    if (p.brightness_protection > 1e-6 && stats.brightness_gain > 1.0) {
        float pre_y = luma(natural);
        float curve_y = pre_y * stats.brightness_gain /
                        (1.0 + (stats.brightness_gain - 1.0) * pre_y);
        float scale = pre_y > 1e-6 ? curve_y / pre_y : 1.0;
        natural = clamp(natural * scale, 0.0, 1.0);
        natural = smooth_chroma_gamut(natural);
    }

    float saturation_limit = max(0.0, 0.97 - 1.5 / 65535.0);
    float final_y = luma(natural);
    float luma_cap = y < 0.97 ? saturation_limit : 1.0;
    float luma_scale = min(1.0, luma_cap / max(final_y, 1e-4));
    natural *= luma_scale;
    float3 caps = float3(original.r < 0.97 ? saturation_limit : 1.0,
                         original.g < 0.97 ? saturation_limit : 1.0,
                         original.b < 0.97 ? saturation_limit : 1.0);
    if (p.color_recovery > 1e-6) natural = smooth_chroma_caps(natural, caps);
    else natural = min(natural, caps);
    return clamp(natural, 0.0, 1.0);
}

// Spatial mode mirrors the CPU reference's first pass. It deliberately uses
// only the transmission value for this pixel; sampling neighboring values here
// would introduce edge halos around depth boundaries.
static float3 apply_spatial_dehaze_pre(float3 original, constant DehazeParams &p,
                                      constant ImageStats &stats, float spatial_transmission) {
    if (p.strength <= 1e-6) return original;
    float y = luma(original);
    float3 air = float3(stats.air_r, stats.air_g, stats.air_b);
    float air_mean = (air.r + air.g + air.b) / 3.0;
    air = mix(air, float3(air_mean), 0.45 * p.color_protection * (0.65 + 0.35 * p.naturalness));
    air = clamp(air, float3(0.01), float3(1.0));
    float transmission_floor = 0.20 + 0.16 * p.fog_retention + 0.08 * p.naturalness;
    float transmission = clamp(spatial_transmission, transmission_floor, 1.0);
    float3 delta = original - air;
    float3 effective_transmission = transmission + (1.0 - transmission) * max(delta, float3(0.0)) /
                                    max(float3(1.0) - air, float3(1e-4));
    float3 recovered = clamp(air + delta / effective_transmission, 0.0, 1.0);
    float amount = clamp(p.strength * (0.78 - 0.28 * p.fog_retention) *
                         (0.90 + 0.10 * (1.0 - p.naturalness)), 0.0, 0.82);
    float3 natural = mix(original, recovered, amount);
    float3 luma_only = clamp(original * (luma(natural) / max(y, 1e-4)), 0.0, 1.0);
    natural = mix(natural, luma_only, p.color_protection * (0.72 + 0.28 * p.naturalness));
    if (p.color_recovery > 1e-6) {
        float3 source_chroma = original - float3(y);
        float source_chroma_norm = length(source_chroma);
        float confidence = clamp((source_chroma_norm - 0.006) / 0.084, 0.0, 1.0);
        confidence = confidence * confidence * (3.0 - 2.0 * confidence);
        float3 source_direction = source_chroma / max(source_chroma_norm, 1e-6);
        float natural_y = luma(natural);
        float3 natural_chroma = natural - float3(natural_y);
        float natural_chroma_norm = length(natural_chroma);
        float aligned_chroma = max(0.0, dot(natural_chroma, source_direction));
        float recovery_amount = p.color_recovery * p.strength;
        float source_target_norm = source_chroma_norm * (1.0 + 1.80 * p.color_recovery * confidence);
        float natural_target_norm = natural_chroma_norm *
                                    (1.0 + 0.30 * recovery_amount * confidence) * confidence;
        float target_chroma_norm = max(max(aligned_chroma * confidence, natural_target_norm),
                                       source_target_norm);
        float3 target_chroma = source_direction * target_chroma_norm;
        float highlight_position = smoothstep(0.58, 0.98, y);
        float shadow_position = smoothstep(0.0, 0.26, 0.26 - y);
        float protection = (1.0 - highlight_position * p.highlight_protection) *
                           (1.0 - shadow_position * p.shadow_protection);
        float requested_recovery = clamp(recovery_amount * protection *
                                         (0.95 + 0.35 * (1.0 - confidence)), 0.0, 0.95);
        float neutral_guard = p.color_protection * p.strength * (1.0 - confidence) *
                              (1.08 + 0.12 * p.naturalness) * protection;
        float correction_strength = clamp(max(requested_recovery, neutral_guard), 0.0, 0.95);
        natural = float3(natural_y) + natural_chroma * (1.0 - correction_strength) +
                  target_chroma * correction_strength;
        natural = smooth_chroma_gamut(natural);
    }

    if (p.local_contrast > 1e-6) {
        float current_y = luma(natural);
        float contrast_amount = 0.55 * p.local_contrast * p.strength;
        float contrast_y = current_y + contrast_amount * (2.0 * current_y - 1.0) *
                           current_y * (1.0 - current_y);
        natural *= contrast_y / max(current_y, 1e-4);
    }

    float source_peak = max(original.r, max(original.g, original.b));
    float highlight_position = max(smoothstep(0.58, 0.98, y),
                                   smoothstep(0.35, 0.90, source_peak));
    float highlight_blend = clamp(highlight_position * p.highlight_protection, 0.0, 1.0);
    natural = mix(natural, original, highlight_blend);
    float shadow_position = smoothstep(0.0, 0.26, 0.26 - y);
    float shadow_blend = clamp(shadow_position * p.shadow_protection, 0.0, 1.0);
    return mix(natural, original, shadow_blend);
}

static float3 apply_spatial_caps(float3 source, float3 image, constant DehazeParams &p) {
    float saturation_limit = max(0.0, 0.97 - 1.5 / 65535.0);
    float source_y = luma(source);
    float final_y = luma(image);
    float luma_cap = source_y < 0.97 ? saturation_limit : 1.0;
    image *= min(1.0, luma_cap / max(final_y, 1e-4));
    float3 caps = float3(source.r < 0.97 ? saturation_limit : 1.0,
                         source.g < 0.97 ? saturation_limit : 1.0,
                         source.b < 0.97 ? saturation_limit : 1.0);
    if (p.color_recovery > 1e-6) image = smooth_chroma_caps(image, caps);
    else image = min(image, caps);
    return clamp(image, 0.0, 1.0);
}

static float3 apply_spatial_brightness(float3 source, float3 image,
                                       constant DehazeParams &p, constant ImageStats &stats) {
    if (stats.brightness_gain <= 1.0 + 1e-6 || p.brightness_protection <= 1e-6) return image;
    float y = luma(image);
    float curve_y = y * stats.brightness_gain /
                    (1.0 + (stats.brightness_gain - 1.0) * y);
    float scale = y > 1e-6 ? curve_y / y : 1.0;
    image = clamp(image * scale, 0.0, 1.0);
    image = smooth_chroma_gamut(image);

    // The spatial reference always applies smooth per-channel caps in this
    // second pass, independent of the color-recovery setting.
    float saturation_limit = max(0.0, 0.97 - 1.5 / 65535.0);
    float source_y = luma(source);
    float luma_cap = source_y < 0.97 ? saturation_limit : 1.0;
    float luma_scale = min(1.0, luma_cap / max(luma(image), 1e-4));
    image *= luma_scale;
    float3 caps = float3(source.r < 0.97 ? saturation_limit : 1.0,
                         source.g < 0.97 ? saturation_limit : 1.0,
                         source.b < 0.97 ? saturation_limit : 1.0);
    return clamp(smooth_chroma_caps(image, caps), 0.0, 1.0);
}

static float3 apply_basic(float3 rgb, constant BasicParams &p) {
    rgb *= exp2(p.exposure);
    rgb = (rgb - float3(0.5)) * (1.0 + p.contrast / 100.0) + float3(0.5);
    float y = luma(rgb);
    float shadow_mask = pow(clamp((0.62 - y) / 0.62, 0.0, 1.0), 1.5);
    float highlight_mask = pow(clamp((y - 0.38) / 0.62, 0.0, 1.0), 1.5);
    float tone = (p.shadows * 0.0022 + p.blacks * 0.0008) * shadow_mask
               + (p.highlights * 0.0018 + p.whites * 0.0008) * highlight_mask;
    rgb += float3(tone);
    y = luma(rgb);
    float3 chroma = rgb - float3(y);
    float chroma_level = max(abs(chroma.r), max(abs(chroma.g), abs(chroma.b)));
    float vibrance = 1.0 + p.vibrance / 100.0 * clamp(1.0 - chroma_level, 0.0, 1.0);
    return clamp(float3(y) + chroma * (1.0 + p.saturation / 100.0) * vibrance, 0.0, 1.0);
}

static float3 apply_filter_controls(float3 rgb, constant FilterParams &p) {
    rgb *= exp2(p.exposure_ev);
    rgb = (rgb - float3(0.18)) * p.contrast + float3(0.18);
    rgb.r *= 1.0 + p.warmth * 0.12;
    rgb.b *= 1.0 - p.warmth * 0.12;
    rgb.g *= 1.0 + p.tint * 0.08;
    rgb = with_saturation(clamp(rgb, 0.0, 1.0), p.saturation);
    float y = luma(rgb);
    rgb = clamp(rgb + float3(p.fade * 0.08 * (0.5 - y)), 0.0, 1.0);
    return rgb;
}

static float sample_curve(device const ushort *curve, uint channel, float value) {
    float position = clamp(value, 0.0, 1.0) * 255.0;
    uint lo = uint(floor(position));
    uint hi = min(lo + 1, 255u);
    float fraction = position - float(lo);
    float a = float(curve[channel * 256 + lo]) / 65535.0;
    float b = float(curve[channel * 256 + hi]) / 65535.0;
    return mix(a, b, fraction);
}

static float3 sample_lut(device const ushort *lut, uint edge, float3 color) {
    float3 coordinate = clamp(color, 0.0, 1.0) * float(edge - 1);
    uint3 lo = uint3(floor(coordinate));
    uint3 hi = min(lo + uint3(1), uint3(edge - 1));
    float3 f = coordinate - float3(lo);
    float3 result = float3(0.0);
    for (uint channel = 0; channel < 3; ++channel) {
        float c000 = float(lut[((lo.r * edge + lo.g) * edge + lo.b) * 3 + channel]) / 65535.0;
        float c001 = float(lut[((lo.r * edge + lo.g) * edge + hi.b) * 3 + channel]) / 65535.0;
        float c010 = float(lut[((lo.r * edge + hi.g) * edge + lo.b) * 3 + channel]) / 65535.0;
        float c011 = float(lut[((lo.r * edge + hi.g) * edge + hi.b) * 3 + channel]) / 65535.0;
        float c100 = float(lut[((hi.r * edge + lo.g) * edge + lo.b) * 3 + channel]) / 65535.0;
        float c101 = float(lut[((hi.r * edge + lo.g) * edge + hi.b) * 3 + channel]) / 65535.0;
        float c110 = float(lut[((hi.r * edge + hi.g) * edge + lo.b) * 3 + channel]) / 65535.0;
        float c111 = float(lut[((hi.r * edge + hi.g) * edge + hi.b) * 3 + channel]) / 65535.0;
        float c00 = mix(c000, c001, f.b);
        float c01 = mix(c010, c011, f.b);
        float c10 = mix(c100, c101, f.b);
        float c11 = mix(c110, c111, f.b);
        float c0 = mix(c00, c01, f.g);
        float c1 = mix(c10, c11, f.g);
        result[channel] = mix(c0, c1, f.r);
    }
    return result;
}

kernel void render_kernel(device const ushort *source [[buffer(0)]],
                          device ushort *destination [[buffer(1)]],
                          constant DehazeParams &dehaze [[buffer(2)]],
                          constant BasicParams &basic [[buffer(3)]],
                          constant FilterParams &filter [[buffer(4)]],
                          constant ImageStats &stats [[buffer(5)]],
                          device const ushort *curve [[buffer(6)]],
                          device const ushort *lut [[buffer(7)]],
                          constant uint &lut_edge [[buffer(8)]],
                          constant uint &pixel_count [[buffer(9)]],
                          uint pixel [[thread_position_in_grid]]) {
    if (pixel >= pixel_count) return;
    float3 color = float3(source[pixel * 3], source[pixel * 3 + 1], source[pixel * 3 + 2]) / 65535.0;
    color = apply_dehaze(color, dehaze, stats);
    color = apply_basic(color, basic);
    color = apply_filter_controls(color, filter);
    float3 curved = float3(sample_curve(curve, 0, color.r),
                           sample_curve(curve, 1, color.g),
                           sample_curve(curve, 2, color.b));
    color = mix(color, curved, filter.curve_mix);
    if (lut_edge >= 2) color = mix(color, sample_lut(lut, lut_edge, color), filter.lut_mix);
    color = clamp(color, 0.0, 1.0);
    destination[pixel * 3] = ushort(rint(color.r * 65535.0));
    destination[pixel * 3 + 1] = ushort(rint(color.g * 65535.0));
    destination[pixel * 3 + 2] = ushort(rint(color.b * 65535.0));
}

kernel void render_spatial_first_pass(device const ushort *source [[buffer(0)]],
                                     device ushort *destination [[buffer(1)]],
                                     constant DehazeParams &dehaze [[buffer(2)]],
                                     constant ImageStats &stats [[buffer(3)]],
                                     device const float *transmission [[buffer(4)]],
                                     constant uint &pixel_count [[buffer(5)]],
                                     uint pixel [[thread_position_in_grid]]) {
    if (pixel >= pixel_count) return;
    float3 original = float3(source[pixel * 3], source[pixel * 3 + 1],
                             source[pixel * 3 + 2]) / 65535.0;
    if (dehaze.strength <= 1e-6) {
        destination[pixel * 3] = source[pixel * 3];
        destination[pixel * 3 + 1] = source[pixel * 3 + 1];
        destination[pixel * 3 + 2] = source[pixel * 3 + 2];
        return;
    }
    float3 color = apply_spatial_dehaze_pre(original, dehaze, stats, transmission[pixel]);
    color = apply_spatial_caps(original, color, dehaze);
    destination[pixel * 3] = ushort(rint(color.r * 65535.0));
    destination[pixel * 3 + 1] = ushort(rint(color.g * 65535.0));
    destination[pixel * 3 + 2] = ushort(rint(color.b * 65535.0));
}

kernel void render_spatial_finish_pass(device const ushort *source [[buffer(0)]],
                                      device const ushort *first_pass [[buffer(1)]],
                                      device ushort *destination [[buffer(2)]],
                                      constant DehazeParams &dehaze [[buffer(3)]],
                                      constant ImageStats &stats [[buffer(4)]],
                                      constant uint &pixel_count [[buffer(5)]],
                                      uint pixel [[thread_position_in_grid]]) {
    if (pixel >= pixel_count) return;
    if (dehaze.strength <= 1e-6) {
        destination[pixel * 3] = source[pixel * 3];
        destination[pixel * 3 + 1] = source[pixel * 3 + 1];
        destination[pixel * 3 + 2] = source[pixel * 3 + 2];
        return;
    }
    if (stats.brightness_gain <= 1.0 + 1e-6 || dehaze.brightness_protection <= 1e-6) {
        destination[pixel * 3] = first_pass[pixel * 3];
        destination[pixel * 3 + 1] = first_pass[pixel * 3 + 1];
        destination[pixel * 3 + 2] = first_pass[pixel * 3 + 2];
        return;
    }
    float3 original = float3(source[pixel * 3], source[pixel * 3 + 1],
                             source[pixel * 3 + 2]) / 65535.0;
    float3 color = float3(first_pass[pixel * 3], first_pass[pixel * 3 + 1],
                          first_pass[pixel * 3 + 2]) / 65535.0;
    color = apply_spatial_brightness(original, color, dehaze, stats);
    destination[pixel * 3] = ushort(rint(color.r * 65535.0));
    destination[pixel * 3 + 1] = ushort(rint(color.g * 65535.0));
    destination[pixel * 3 + 2] = ushort(rint(color.b * 65535.0));
}

static float physical_smoothstep(float low, float high, float value) {
    float position = clamp((value - low) / (high - low), 0.0, 1.0);
    return position * position * (3.0 - 2.0 * position);
}

static float physical_inverse_transmission(float transmission, float3 original,
                                            DehazeParams p) {
    float gain = 1.0f + 0.8f * p.strength * (1.0f - 0.35f * p.naturalness);
    float floor = 1.0f / gain;
    float width = 0.015f * (1.0f - floor);
    float t = max(transmission, floor);
    if (width > 0.0f) {
        float softness = max(width - abs(transmission - floor), 0.0f);
        t = t + softness * softness / (4.0f * width);
    }
    float highlight = physical_smoothstep(0.55f, 0.95f,
                                          max(original.r, max(original.g, original.b))) *
                      p.highlight_protection;
    return clamp(t + (1.0f - t) * highlight, 1e-4f, 1.0f);
}

static float physical_luma(float3 rgb) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    float red = rgb.r * 0.2126f;
    float green = rgb.g * 0.7152f;
    float blue = rgb.b * 0.0722f;
    float red_green = red + green;
    return red_green + blue;
}

static float3 physical_gamut(float3 rgb) {
    float y = physical_luma(rgb);
    float3 chroma = rgb - float3(y);
    float3 positive_room = (1.0 - float3(y)) / max(chroma, float3(1e-7));
    float3 negative_room = float3(y) / max(-chroma, float3(1e-7));
    float3 room = float3(chroma.r > 0.0 ? positive_room.r : negative_room.r,
                         chroma.g > 0.0 ? positive_room.g : negative_room.g,
                         chroma.b > 0.0 ? positive_room.b : negative_room.b);
    float scale = clamp(min(room.r, min(room.g, room.b)), 0.0, 1.0);
    return clamp(float3(y) + chroma * scale, 0.0, 1.0);
}

kernel void render_physical_float(device const float *source [[buffer(0)]],
                                  device const float *transmission [[buffer(1)]],
                                  device const float *airlight [[buffer(2)]],
                                  constant DehazeParams &p [[buffer(3)]],
                                  device float *destination [[buffer(4)]],
                                  constant uint &pixel_count [[buffer(5)]],
                                  uint pixel [[thread_position_in_grid]]) {
    if (pixel >= pixel_count) return;
    float3 original = float3(source[pixel * 3], source[pixel * 3 + 1],
                             source[pixel * 3 + 2]);
    float3 air = float3(airlight[0], airlight[1], airlight[2]);
    float y = physical_luma(original);
    float t = physical_inverse_transmission(transmission[pixel], original, p);

    float3 delta = original - air;
    float3 shoulder = float3(t) + (1.0 - t) * max(delta, float3(0.0)) /
                      max(float3(1.0) - air, float3(1e-4));
    float3 positive = air + delta / shoulder;
    float3 deficit = max(-delta, float3(0.0));
    float3 smooth_deficit = deficit * deficit /
                            (deficit + 0.025f * air + float3(1e-8f));
    float3 loss = (1.0f / t - 1.0f) * smooth_deficit;
    float retention = 0.10f + 0.45f * p.brightness_protection +
                      0.10f * p.shadow_protection;
    // Preserve black-level shadows while allowing fog-lifted midtones to
    // leave the toe; match the linear CPU reference at every pixel.
    float3 budget = (1.0f - retention) * original * original /
                    (original + 0.12f * air + float3(1e-8f)) * p.strength;
    float3 fraction = loss / max(budget, float3(1e-8f));
    float3 bounded = select(fraction,
                            1.0f - 0.5f * exp(-2.0f * max(fraction - 0.5f, float3(0.0f))),
                            fraction > float3(0.5f));
    float3 negative = original - budget * bounded;
    float3 recovered = select(positive, negative, delta < float3(0.0f));
    float recovered_y = physical_luma(recovered);
    float3 source_hue = original * (recovered_y / max(y, 1e-6));
    float3 source_chroma = original - float3(y);
    float chroma_confidence = physical_smoothstep(0.005, 0.08, length(source_chroma));
    float physical_colour = (1.0 - p.color_protection) *
                            (0.28 + 0.72 * (1.0 - p.naturalness)) * chroma_confidence;
    float3 result = recovered * physical_colour + source_hue * (1.0 - physical_colour);

    float result_y = physical_luma(result);
    float3 chroma = result - float3(result_y);
    chroma *= 1.0 + 0.25 * p.color_recovery * p.strength * chroma_confidence;
    float contrast_delta = 0.15 * p.local_contrast * p.strength * result_y *
                           (1.0 - result_y) * (2.0 * result_y - 1.0);
    contrast_delta = clamp(contrast_delta, -0.02 * result_y,
                           0.02 * (1.0 - result_y));
    float tone_y = clamp(result_y + contrast_delta, 0.0, 1.0);
    result = physical_gamut(float3(tone_y) + chroma);
    result_y = physical_luma(result);
    result *= min(1.0, y / max(result_y, 1e-6));
    result = clamp(result, 0.0, 1.0);
    destination[pixel * 3] = result.r;
    destination[pixel * 3 + 1] = result.g;
    destination[pixel * 3 + 2] = result.b;
}
