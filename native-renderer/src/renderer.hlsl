cbuffer RenderConstants : register(b0) {
    float4 dehaze0; // strength, naturalness, fog retention, local contrast
    float4 dehaze1; // recovery, color protection, highlight protection, shadow protection
    float4 dehaze2; // brightness protection, padding
    float4 basic0;  // exposure, contrast, highlights, shadows
    float4 basic1;  // whites, blacks, vibrance, saturation
    float4 filter0; // curve mix, LUT mix, exposure EV, contrast
    float4 filter1; // saturation, warmth, tint, fade
    float4 stats0;  // mean luma, max luma, air R, air G
    float4 stats1;  // air B, haze level, brightness gain, guarded dark floor
    uint4 dispatch; // LUT edge, pixel count, identity-copy flag, padding
};

ByteAddressBuffer SourceData : register(t0);
ByteAddressBuffer CurveData : register(t1);
ByteAddressBuffer LutData : register(t2);
ByteAddressBuffer TransmissionData : register(t3);
ByteAddressBuffer SpatialFirstPassData : register(t4);
RWByteAddressBuffer DestinationData : register(u0);

static const float kInvU16 = 1.0 / 65535.0;

uint unpack_u16(uint packed, uint index) {
    return (index & 1) == 0 ? (packed & 0xffff) : (packed >> 16);
}

uint load_source_u16(uint index) {
    return unpack_u16(SourceData.Load((index >> 1) << 2), index);
}

uint load_curve_u16(uint index) {
    return unpack_u16(CurveData.Load((index >> 1) << 2), index);
}

uint load_lut_u16(uint index) {
    return unpack_u16(LutData.Load((index >> 1) << 2), index);
}

uint load_spatial_first_pass_u16(uint index) {
    return SpatialFirstPassData.Load(index * 4);
}

float load_transmission(uint pixel) {
    return asfloat(TransmissionData.Load(pixel * 4));
}

float clamp01(float value) {
    return min(1.0, max(0.0, value));
}

float smooth01(float value) {
    float t = clamp01(value);
    return t * t * (3.0 - 2.0 * t);
}

float smooth_range(float low, float high, float value) {
    return smooth01((value - low) / (high - low));
}

float luma(float3 rgb) {
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

float3 with_saturation(float3 rgb, float amount) {
    float y = luma(rgb);
    return clamp(float3(y, y, y) + (rgb - float3(y, y, y)) * amount, 0.0, 1.0);
}

float3 smooth_chroma_gamut(float3 color) {
    float y = luma(color);
    float3 chroma = color - float3(y, y, y);
    float limit = 1e30;
    [unroll]
    for (uint channel = 0; channel < 3; ++channel) {
        if (chroma[channel] > 1e-7) limit = min(limit, (1.0 - y) / chroma[channel]);
        else if (chroma[channel] < -1e-7) limit = min(limit, y / -chroma[channel]);
    }
    if (limit > 1e20) limit = 1.0;
    float delta = 1.0 - limit;
    float scale = clamp(0.5 * (1.0 + limit - sqrt(delta * delta + 1e-10)), 0.0, 1.0);
    return float3(y, y, y) + chroma * scale;
}

float3 smooth_chroma_caps(float3 color, float3 caps) {
    float y = luma(color);
    float3 chroma = color - float3(y, y, y);
    float limit = 1e30;
    [unroll]
    for (uint channel = 0; channel < 3; ++channel) {
        if (chroma[channel] > 1e-7) limit = min(limit, (caps[channel] - y) / chroma[channel]);
        else if (chroma[channel] < -1e-7) limit = min(limit, y / -chroma[channel]);
    }
    if (limit > 1e20) limit = 1.0;
    float delta = 1.0 - limit;
    float scale = clamp(0.5 * (1.0 + limit - sqrt(delta * delta + 1e-10)), 0.0, 1.0);
    return float3(y, y, y) + chroma * scale;
}

float3 apply_dehaze(float3 original, bool spatial_mode, float spatial_transmission) {
    if (dehaze0.x <= 1e-6) return original;
    float y = luma(original);
    float3 air = float3(stats0.z, stats0.w, stats1.x);
    float air_mean = (air.x + air.y + air.z) / 3.0;
    float air_mix = 0.45 * dehaze1.y * (0.65 + 0.35 * dehaze0.y);
    float atmosphere_floor = spatial_mode ? 0.01 : 0.35;
    air = clamp(air * (1.0 - air_mix) + float3(air_mean, air_mean, air_mean) * air_mix,
                atmosphere_floor, 1.0);
    float transmission_floor = spatial_mode
                                  ? 0.20 + 0.16 * dehaze0.z + 0.08 * dehaze0.y
                                  : 0.27 + 0.21 * dehaze0.z + 0.11 * dehaze0.y;
    float transmission = spatial_transmission;
    if (spatial_mode) {
        transmission = clamp(spatial_transmission, transmission_floor, 1.0);
    } else {
        float omega = dehaze0.x * (0.66 - 0.14 * dehaze0.z) *
                      (0.90 + 0.10 * (1.0 - dehaze0.y)) * (0.92 + 0.08 * stats1.y);
        transmission = clamp(1.0 - omega, transmission_floor, 1.0);
    }
    float3 delta = original - air;
    float3 effective_transmission = transmission + (1.0 - transmission) * max(delta, 0.0) /
                                    max(1.0 - air, 1e-4);
    float3 recovered = clamp(air + delta / effective_transmission, 0.0, 1.0);
    float amount = clamp(dehaze0.x * (0.78 - 0.28 * dehaze0.z) *
                         (0.90 + 0.10 * (1.0 - dehaze0.y)), 0.0, 0.82);
    float3 natural = original * (1.0 - amount) + recovered * amount;
    float3 luma_only = clamp(original * (luma(natural) / max(y, 1e-4)), 0.0, 1.0);
    float chroma_mix = dehaze1.y * (0.72 + 0.28 * dehaze0.y);
    natural = natural * (1.0 - chroma_mix) + luma_only * chroma_mix;

    if (dehaze1.x > 1e-6) {
        float3 source_chroma = original - float3(y, y, y);
        float source_chroma_norm = length(source_chroma);
        float confidence = clamp01((source_chroma_norm - 0.006) / 0.084);
        confidence = confidence * confidence * (3.0 - 2.0 * confidence);
        float3 source_direction = source_chroma / max(source_chroma_norm, 1e-6);
        float natural_y = luma(natural);
        float3 natural_chroma = natural - float3(natural_y, natural_y, natural_y);
        float natural_chroma_norm = length(natural_chroma);
        float aligned_chroma = max(0.0, dot(natural_chroma, source_direction));
        float recovery_amount = dehaze1.x * dehaze0.x;
        float source_target_norm = source_chroma_norm * (1.0 + 1.80 * dehaze1.x * confidence);
        float natural_target_norm = natural_chroma_norm * (1.0 + 0.30 * recovery_amount * confidence) * confidence;
        float target_chroma_norm = max(max(aligned_chroma * confidence, natural_target_norm), source_target_norm);
        float3 target_chroma = source_direction * target_chroma_norm;
        float highlight_position = smooth_range(0.58, 0.98, y);
        float shadow_position = smooth_range(0.0, 0.26, 0.26 - y);
        float protection = (1.0 - highlight_position * dehaze1.z) *
                           (1.0 - shadow_position * dehaze1.w);
        float requested_recovery = clamp(recovery_amount * protection * (0.95 + 0.35 * (1.0 - confidence)),
                                         0.0, 0.95);
        float neutral_guard = dehaze1.y * dehaze0.x * (1.0 - confidence) *
                              (1.08 + 0.12 * dehaze0.y) * protection;
        float correction_strength = clamp(max(requested_recovery, neutral_guard), 0.0, 0.95);
        natural = float3(natural_y, natural_y, natural_y) + natural_chroma * (1.0 - correction_strength) +
                  target_chroma * correction_strength;
        natural = smooth_chroma_gamut(natural);
    }

    if (dehaze0.w > 1e-6) {
        float current_y = luma(natural);
        float contrast_amount = 0.55 * dehaze0.w * dehaze0.x;
        float contrast_y = current_y + contrast_amount * (2.0 * current_y - 1.0) * current_y * (1.0 - current_y);
        natural *= contrast_y / max(current_y, 1e-4);
    }

    float highlight_position = smooth_range(0.58, 0.98, y);
    if (spatial_mode) {
        float source_peak = max(original.r, max(original.g, original.b));
        highlight_position = max(highlight_position, smooth_range(0.35, 0.90, source_peak));
    }
    float highlight_blend = clamp01(highlight_position * dehaze1.z);
    natural = natural * (1.0 - highlight_blend) + original * highlight_blend;
    float shadow_position = smooth_range(0.0, 0.26, 0.26 - y);
    float shadow_blend = clamp01(shadow_position * dehaze1.w);
    natural = natural * (1.0 - shadow_blend) + original * shadow_blend;

    if (!spatial_mode && dehaze2.x > 1e-6 && stats1.z > 1.0) {
        float pre_y = luma(natural);
        float curve_y = pre_y * stats1.z / (1.0 + (stats1.z - 1.0) * pre_y);
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
    if (dehaze1.x > 1e-6) natural = smooth_chroma_caps(natural, caps);
    else natural = min(natural, caps);
    return clamp(natural, 0.0, 1.0);
}

float3 apply_brightness_and_caps(float3 source, float3 image, float gain) {
    if (gain > 1.0 && dehaze2.x > 1e-6) {
        float image_y = luma(image);
        float curve_y = image_y * gain / (1.0 + (gain - 1.0) * image_y);
        float scale = image_y > 1e-6 ? curve_y / image_y : 1.0;
        image = clamp(image * scale, 0.0, 1.0);
        image = smooth_chroma_gamut(image);
    }

    float saturation_limit = max(0.0, 0.97 - 1.5 / 65535.0);
    float source_y = luma(source);
    float final_y = luma(image);
    float luma_cap = source_y < 0.97 ? saturation_limit : 1.0;
    float luma_scale = min(1.0, luma_cap / max(final_y, 1e-4));
    image *= luma_scale;
    float3 caps = float3(source.r < 0.97 ? saturation_limit : 1.0,
                         source.g < 0.97 ? saturation_limit : 1.0,
                         source.b < 0.97 ? saturation_limit : 1.0);
    image = smooth_chroma_caps(image, caps);
    return clamp(image, 0.0, 1.0);
}

float3 apply_basic(float3 rgb) {
    rgb *= exp2(basic0.x);
    rgb = (rgb - float3(0.5, 0.5, 0.5)) * (1.0 + basic0.y / 100.0) + float3(0.5, 0.5, 0.5);
    float y = luma(rgb);
    float shadow_mask = pow(clamp((0.62 - y) / 0.62, 0.0, 1.0), 1.5);
    float highlight_mask = pow(clamp((y - 0.38) / 0.62, 0.0, 1.0), 1.5);
    float tone = (basic0.w * 0.0022 + basic1.y * 0.0008) * shadow_mask
               + (basic0.z * 0.0018 + basic1.x * 0.0008) * highlight_mask;
    rgb += float3(tone, tone, tone);
    y = luma(rgb);
    float3 chroma = rgb - float3(y, y, y);
    float chroma_level = max(abs(chroma.r), max(abs(chroma.g), abs(chroma.b)));
    float vibrance = 1.0 + basic1.z / 100.0 * clamp01(1.0 - chroma_level);
    return clamp(float3(y, y, y) + chroma * (1.0 + basic1.w / 100.0) * vibrance, 0.0, 1.0);
}

float3 apply_filter_controls(float3 rgb) {
    rgb *= exp2(filter0.z);
    rgb = (rgb - float3(0.18, 0.18, 0.18)) * filter0.w + float3(0.18, 0.18, 0.18);
    rgb.r *= 1.0 + filter1.y * 0.12;
    rgb.b *= 1.0 - filter1.y * 0.12;
    rgb.g *= 1.0 + filter1.z * 0.08;
    rgb = with_saturation(clamp(rgb, 0.0, 1.0), filter1.x);
    float y = luma(rgb);
    float fade_offset = filter1.w * 0.08 * (0.5 - y);
    rgb = clamp(rgb + float3(fade_offset, fade_offset, fade_offset), 0.0, 1.0);
    return rgb;
}

float sample_curve(uint channel, float value) {
    float position = clamp01(value) * 255.0;
    uint lo = (uint)floor(position);
    uint hi = min(lo + 1, 255u);
    float fraction = position - (float)lo;
    float a = (float)load_curve_u16(channel * 256 + lo) * kInvU16;
    float b = (float)load_curve_u16(channel * 256 + hi) * kInvU16;
    return a + (b - a) * fraction;
}

float3 sample_lut(float3 color) {
    uint edge = dispatch.x;
    float3 coordinate = clamp(color, 0.0, 1.0) * (float)(edge - 1);
    uint3 lo = (uint3)floor(coordinate);
    uint3 hi = min(lo + uint3(1, 1, 1), uint3(edge - 1, edge - 1, edge - 1));
    float3 f = coordinate - (float3)lo;
    float3 result = float3(0.0, 0.0, 0.0);
    [unroll]
    for (uint channel = 0; channel < 3; ++channel) {
        uint c000i = ((lo.r * edge + lo.g) * edge + lo.b) * 3 + channel;
        uint c001i = ((lo.r * edge + lo.g) * edge + hi.b) * 3 + channel;
        uint c010i = ((lo.r * edge + hi.g) * edge + lo.b) * 3 + channel;
        uint c011i = ((lo.r * edge + hi.g) * edge + hi.b) * 3 + channel;
        uint c100i = ((hi.r * edge + lo.g) * edge + lo.b) * 3 + channel;
        uint c101i = ((hi.r * edge + lo.g) * edge + hi.b) * 3 + channel;
        uint c110i = ((hi.r * edge + hi.g) * edge + lo.b) * 3 + channel;
        uint c111i = ((hi.r * edge + hi.g) * edge + hi.b) * 3 + channel;
        float c000 = (float)load_lut_u16(c000i) * kInvU16;
        float c001 = (float)load_lut_u16(c001i) * kInvU16;
        float c010 = (float)load_lut_u16(c010i) * kInvU16;
        float c011 = (float)load_lut_u16(c011i) * kInvU16;
        float c100 = (float)load_lut_u16(c100i) * kInvU16;
        float c101 = (float)load_lut_u16(c101i) * kInvU16;
        float c110 = (float)load_lut_u16(c110i) * kInvU16;
        float c111 = (float)load_lut_u16(c111i) * kInvU16;
        float c00 = c000 + (c001 - c000) * f.b;
        float c01 = c010 + (c011 - c010) * f.b;
        float c10 = c100 + (c101 - c100) * f.b;
        float c11 = c110 + (c111 - c110) * f.b;
        float c0 = c00 + (c01 - c00) * f.g;
        float c1 = c10 + (c11 - c10) * f.g;
        result[channel] = c0 + (c1 - c0) * f.r;
    }
    return result;
}

uint round_to_even_u16(float value) {
    float scaled = clamp01(value) * 65535.0;
    float floor_value = floor(scaled);
    float fraction = scaled - floor_value;
    uint base = (uint)floor_value;
    if (fraction > 0.5 || (fraction == 0.5 && (base & 1) != 0)) ++base;
    return min(base, 65535u);
}

[numthreads(256, 1, 1)]
void render_kernel(uint3 thread_id : SV_DispatchThreadID) {
    uint pixel = thread_id.x + thread_id.y * dispatch.w;
    if (pixel >= dispatch.y) return;
    uint input_base = pixel * 3;
    uint output_base = input_base * 4;
    if (dispatch.z == 1) {
        DestinationData.Store(output_base, load_source_u16(input_base));
        DestinationData.Store(output_base + 4, load_source_u16(input_base + 1));
        DestinationData.Store(output_base + 8, load_source_u16(input_base + 2));
        return;
    }

    float3 source = float3(load_source_u16(input_base),
                           load_source_u16(input_base + 1),
                           load_source_u16(input_base + 2)) * kInvU16;
    if (dispatch.z == 2) {
        float3 color = apply_dehaze(source, true, load_transmission(pixel));
        DestinationData.Store(output_base, round_to_even_u16(color.r));
        DestinationData.Store(output_base + 4, round_to_even_u16(color.g));
        DestinationData.Store(output_base + 8, round_to_even_u16(color.b));
        return;
    }
    if (dispatch.z == 3) {
        uint first_r = load_spatial_first_pass_u16(input_base);
        uint first_g = load_spatial_first_pass_u16(input_base + 1);
        uint first_b = load_spatial_first_pass_u16(input_base + 2);
        if (stats1.z <= 1.000001 || dehaze2.x <= 1e-6) {
            DestinationData.Store(output_base, first_r);
            DestinationData.Store(output_base + 4, first_g);
            DestinationData.Store(output_base + 8, first_b);
            return;
        }
        float3 first_pass = float3(first_r, first_g, first_b) * kInvU16;
        float3 color = apply_brightness_and_caps(source, first_pass, stats1.z);
        DestinationData.Store(output_base, round_to_even_u16(color.r));
        DestinationData.Store(output_base + 4, round_to_even_u16(color.g));
        DestinationData.Store(output_base + 8, round_to_even_u16(color.b));
        return;
    }

    float3 color = apply_dehaze(source, false, 0.0);
    color = apply_basic(color);
    color = apply_filter_controls(color);
    float3 curved = float3(sample_curve(0, color.r),
                           sample_curve(1, color.g),
                           sample_curve(2, color.b));
    color = color + (curved - color) * filter0.x;
    if (dispatch.x >= 2) {
        float3 mapped = sample_lut(color);
        color = color + (mapped - color) * filter0.y;
    }
    color = clamp(color, 0.0, 1.0);
    DestinationData.Store(output_base, round_to_even_u16(color.r));
    DestinationData.Store(output_base + 4, round_to_even_u16(color.g));
    DestinationData.Store(output_base + 8, round_to_even_u16(color.b));
}

float physical_smoothstep(float low, float high, float value) {
    float position = clamp((value - low) / (high - low), 0.0, 1.0);
    return position * position * (3.0 - 2.0 * position);
}

float physical_inverse_transmission(float transmission, float3 original) {
    float gain = 1.0 + 0.8 * dehaze0.x * (1.0 - 0.35 * dehaze0.y);
    float floor_value = 1.0 / gain;
    float width = 0.015 * (1.0 - floor_value);
    float t = max(transmission, floor_value);
    if (width > 0.0) {
        float softness = max(width - abs(transmission - floor_value), 0.0);
        t = t + softness * softness / (4.0 * width);
    }
    float highlight = physical_smoothstep(
        0.55, 0.95, max(original.r, max(original.g, original.b))) * dehaze1.z;
    return clamp(t + (1.0 - t) * highlight, 1e-4, 1.0);
}

float physical_luma(float3 rgb) {
    precise float red = rgb.r * 0.2126f;
    precise float green = rgb.g * 0.7152f;
    precise float blue = rgb.b * 0.0722f;
    precise float red_green = red + green;
    precise float result = red_green + blue;
    return result;
}

float3 physical_gamut(float3 rgb) {
    float y = physical_luma(rgb);
    float3 chroma = rgb - float3(y, y, y);
    float3 positive_room = (1.0 - float3(y, y, y)) / max(chroma, float3(1e-7, 1e-7, 1e-7));
    float3 negative_room = float3(y, y, y) / max(-chroma, float3(1e-7, 1e-7, 1e-7));
    float3 room = float3(chroma.r > 0.0 ? positive_room.r : negative_room.r,
                         chroma.g > 0.0 ? positive_room.g : negative_room.g,
                         chroma.b > 0.0 ? positive_room.b : negative_room.b);
    float scale = clamp(min(room.r, min(room.g, room.b)), 0.0, 1.0);
    return clamp(float3(y, y, y) + chroma * scale, 0.0, 1.0);
}

float physical_bounded_fraction(float fraction) {
    if (fraction <= 0.5) return fraction;
    return 1.0 - 0.5 * exp(-2.0 * max(fraction - 0.5, 0.0));
}

float3 physical_recovered(float3 positive, float3 negative, float3 delta) {
    return float3(delta.r < 0.0 ? negative.r : positive.r,
                  delta.g < 0.0 ? negative.g : positive.g,
                  delta.b < 0.0 ? negative.b : positive.b);
}

float3 physical_room(float3 lower_room, float3 upper_room, float3 deviation) {
    return float3(deviation.r > 0.0 ? upper_room.r : lower_room.r,
                  deviation.g > 0.0 ? upper_room.g : lower_room.g,
                  deviation.b > 0.0 ? upper_room.b : lower_room.b);
}

float physical_one_minus_exp_negative(float value) {
    if (value < 0.05) {
        precise float value2 = value * value;
        precise float value3 = value2 * value;
        precise float value4 = value3 * value;
        precise float value5 = value4 * value;
        precise float value6 = value5 * value;
        precise float value7 = value6 * value;
        precise float term2 = 0.5 * value2;
        precise float term3 = value3 * 0.16666666666666666;
        precise float term4 = value4 * 0.041666666666666664;
        precise float term5 = value5 * 0.008333333333333333;
        precise float term6 = value6 * 0.001388888888888889;
        precise float term7 = value7 * 0.0001984126984126984;
        precise float first_pair = value - term2;
        precise float first_three = first_pair + term3;
        precise float first_four = first_three - term4;
        precise float first_five = first_four + term5;
        precise float first_six = first_five - term6;
        precise float result = first_six + term7;
        return result;
    }
    precise float exponential = exp(-value);
    precise float result = 1.0 - exponential;
    return result;
}

[numthreads(256, 1, 1)]
void render_physical_float(uint3 thread_id : SV_DispatchThreadID) {
    uint pixel = thread_id.x + thread_id.y * dispatch.w;
    if (pixel >= dispatch.y) return;

    uint input_base = pixel * 12u;
    float3 original = float3(asfloat(SourceData.Load(input_base)),
                             asfloat(SourceData.Load(input_base + 4u)),
                             asfloat(SourceData.Load(input_base + 8u)));
    if (dehaze0.x <= 1e-6) {
        DestinationData.Store(input_base, asuint(original.r));
        DestinationData.Store(input_base + 4u, asuint(original.g));
        DestinationData.Store(input_base + 8u, asuint(original.b));
        return;
    }

    float3 air = float3(stats0.z, stats0.w, stats1.x);
    float y = physical_luma(original);
    float t = physical_inverse_transmission(load_transmission(pixel), original);

    float3 delta = original - air;
    float3 shoulder = float3(t, t, t) + (1.0 - t) * max(delta, float3(0.0, 0.0, 0.0)) /
                      max(float3(1.0, 1.0, 1.0) - air, float3(1e-4, 1e-4, 1e-4));
    float3 positive = air + delta / shoulder;
    float3 deficit = max(-delta, float3(0.0, 0.0, 0.0));
    float3 smooth_deficit = deficit * deficit /
                            (deficit + 0.025 * air + float3(1e-8, 1e-8, 1e-8));
    float3 loss = (1.0 / t - 1.0) * smooth_deficit;
    float retention = 0.10 + 0.45 * dehaze2.x + 0.10 * dehaze1.w;
    float3 budget = (1.0 - retention) * original * original /
                    (original + 0.12 * air + float3(1e-8, 1e-8, 1e-8)) * dehaze0.x;
    float3 fraction = loss / max(budget, float3(1e-8, 1e-8, 1e-8));
    float3 bounded = float3(physical_bounded_fraction(fraction.r),
                            physical_bounded_fraction(fraction.g),
                            physical_bounded_fraction(fraction.b));
    float3 negative = original - budget * bounded;
    float3 recovered = physical_recovered(positive, negative, delta);
    float recovered_y = physical_luma(recovered);
    float3 source_hue = original * (recovered_y / max(y, 1e-6));
    float3 source_chroma = original - float3(y, y, y);
    float chroma_confidence = physical_smoothstep(0.005, 0.08, length(source_chroma));
    float physical_colour = (1.0 - dehaze1.y) *
                            (0.28 + 0.72 * (1.0 - dehaze0.y)) * chroma_confidence;
    float3 result = recovered * physical_colour + source_hue * (1.0 - physical_colour);

    float result_y = physical_luma(result);
    float3 chroma = result - float3(result_y, result_y, result_y);
    chroma *= 1.0 + 0.25 * dehaze1.x * dehaze0.x * chroma_confidence;
    float contrast_delta = 0.15 * dehaze0.w * dehaze0.x * result_y *
                           (1.0 - result_y) * (2.0 * result_y - 1.0);
    contrast_delta = clamp(contrast_delta, -0.02 * result_y,
                           0.02 * (1.0 - result_y));
    float tone_y = clamp(result_y + contrast_delta, 0.0, 1.0);
    result = physical_gamut(float3(tone_y, tone_y, tone_y) + chroma);
    result_y = physical_luma(result);
    result *= min(1.0, y / max(result_y, 1e-6));

    result_y = min(physical_luma(result), y);
    float3 base = original * (result_y / max(y, 1e-20));
    float3 deviation = result - base;
    float3 upper_room = max(original - base, float3(0.0, 0.0, 0.0)) /
                        max(deviation, float3(1e-20, 1e-20, 1e-20));
    float3 lower_room = base / max(-deviation, float3(1e-20, 1e-20, 1e-20));
    float3 room = physical_room(lower_room, upper_room, deviation);
    float colour_scale = clamp(min(room.r, min(room.g, room.b)), 0.0, 1.0);
    result = clamp(base + deviation * colour_scale, 0.0, 1.0);

    if (stats1.w > 1e-8) {
        precise float source_y = y;
        precise float guarded_result_y = physical_luma(result);
        precise float normalized_source_y = source_y / stats1.w;
        precise float envelope_fraction = physical_one_minus_exp_negative(normalized_source_y);
        precise float floor_value = stats1.w * envelope_fraction;
        precise float width = 0.02 * stats1.w;
        precise float delta_y = guarded_result_y - floor_value;
        precise float target_y = max(guarded_result_y, floor_value);
        precise float handoff = max(width - abs(delta_y), 0.0);
        precise float correction = (handoff * handoff) / (4.0 * width);
        target_y = target_y + correction;
        precise float lift = max(target_y - guarded_result_y, 0.0);
        if (lift > 0.0) {
            precise float source_denominator = max(source_y, 1e-20);
            precise float hue_amount = lift / source_denominator;
            precise float3 candidate = result + original * hue_amount;

            precise float candidate_y = min(physical_luma(candidate), source_y);
            precise float3 guard_base = original * (candidate_y / max(source_y, 1e-20));
            precise float3 guard_deviation = candidate - guard_base;
            precise float3 guard_upper_room = max(original - guard_base,
                                                   float3(0.0, 0.0, 0.0)) /
                max(guard_deviation, float3(1e-20, 1e-20, 1e-20));
            precise float3 guard_lower_room = guard_base /
                max(-guard_deviation, float3(1e-20, 1e-20, 1e-20));
            precise float3 guard_room = physical_room(guard_lower_room, guard_upper_room,
                                                       guard_deviation);
            precise float guard_colour_scale = clamp(
                min(guard_room.r, min(guard_room.g, guard_room.b)), 0.0, 1.0);
            result = clamp(guard_base + guard_deviation * guard_colour_scale, 0.0, 1.0);
        }
    }

    DestinationData.Store(input_base, asuint(result.r));
    DestinationData.Store(input_base + 4u, asuint(result.g));
    DestinationData.Store(input_base + 8u, asuint(result.b));
}

cbuffer WarpConstants : register(b2) {
    float4 warp_coefficients[6];
    uint4 warp_metadata; // width, height, pixel count, row stride in pixels
};
ByteAddressBuffer WarpSourceData : register(t8);
RWByteAddressBuffer WarpDestinationData : register(u2);

float warp_constant(uint index) {
    return warp_coefficients[index >> 2][index & 3u];
}

uint warp_load_u16(uint index) {
    uint packed = WarpSourceData.Load((index >> 1) << 2);
    return (index & 1u) == 0u ? (packed & 0xffffu) : (packed >> 16);
}

precise float2 warp_source_position(uint x, uint y, uint channel,
                                    float center_x, float center_y,
                                    float pixel_scale_v, float radius) {
    uint coefficient = channel * 6u;
    precise float dx = float(x) - center_x;
    precise float dy = float(y) - center_y;
    precise float scaled_dy = dy * pixel_scale_v;
    precise float norm_x = dx / radius;
    precise float norm_y = scaled_dy / radius;
    precise float norm_x_squared = norm_x * norm_x;
    precise float norm_y_squared = norm_y * norm_y;
    precise float r2 = min(norm_x_squared + norm_y_squared, 1.0);
    precise float radial = warp_constant(coefficient) + r2 * (warp_constant(coefficient + 1u) +
        r2 * (warp_constant(coefficient + 2u) + r2 * warp_constant(coefficient + 3u)));
    precise float tangential_x = warp_constant(coefficient + 5u) *
        (r2 + 2.0 * norm_x_squared) +
        2.0 * warp_constant(coefficient + 4u) * norm_x * norm_y;
    precise float tangential_y = warp_constant(coefficient + 4u) *
        (r2 + 2.0 * norm_y_squared) +
        2.0 * warp_constant(coefficient + 5u) * norm_x * norm_y;
    precise float source_x = center_x + dx * radial + radius * tangential_x;
    precise float source_y = center_y + dy * radial + radius * tangential_y / pixel_scale_v;
    return float2(source_x, source_y);
}

uint warp_round_u16(float value) {
    value = clamp(value, 0.0, 65535.0);
    float lower_float = floor(value);
    float fraction = value - lower_float;
    uint lower = (uint)lower_float;
    if (fraction > 0.5 || (fraction == 0.5 && (lower & 1u) != 0u)) ++lower;
    return min(lower, 65535u);
}

uint warp_sample_channel(uint width, uint height, uint channel, float2 position) {
    precise float x = clamp(position.x, 0.0, float(width - 1u));
    precise float y = clamp(position.y, 0.0, float(height - 1u));
    uint x0 = (uint)floor(x);
    uint y0 = (uint)floor(y);
    uint x1 = min(x0 + 1u, width - 1u);
    uint y1 = min(y0 + 1u, height - 1u);
    precise float fx = x - float(x0);
    precise float fy = y - float(y0);
    uint at00 = (y0 * width + x0) * 3u + channel;
    uint at01 = (y0 * width + x1) * 3u + channel;
    uint at10 = (y1 * width + x0) * 3u + channel;
    uint at11 = (y1 * width + x1) * 3u + channel;
    precise float top = float(warp_load_u16(at00)) * (1.0 - fx) + float(warp_load_u16(at01)) * fx;
    precise float bottom = float(warp_load_u16(at10)) * (1.0 - fx) + float(warp_load_u16(at11)) * fx;
    precise float value = top * (1.0 - fy) + bottom * fy;
    return warp_round_u16(value);
}

[numthreads(256, 1, 1)]
void warp_rectilinear_rgb16(uint3 thread_id : SV_DispatchThreadID) {
    uint pixel = thread_id.y * warp_metadata.w + thread_id.x;
    if (pixel >= warp_metadata.z) return;
    precise float center_x = float(warp_metadata.x) * warp_constant(18u);
    precise float center_y = float(warp_metadata.y) * warp_constant(19u);
    precise float pixel_scale_v = warp_constant(20u);
    precise float radius = warp_constant(21u);
    uint x = pixel % warp_metadata.x;
    uint y = pixel / warp_metadata.x;
    uint output_at = pixel * 3u;
    [unroll]
    for (uint channel = 0; channel < 3u; ++channel) {
        float2 position = warp_source_position(x, y, channel, center_x, center_y,
                                               pixel_scale_v, radius);
        WarpDestinationData.Store((output_at + channel) * 4u,
                                  warp_sample_channel(warp_metadata.x, warp_metadata.y,
                                                      channel, position));
    }
}

cbuffer CameraProfileConstants : register(b1) {
    float4 camera_profile_constants0;
    float4 camera_profile_constants1;
    float4 camera_profile_constants2;
    float4 camera_profile_constants3;
    float4 camera_profile_constants4;
    float4 camera_profile_constants5;
    uint4 camera_profile_dimensions; // hue, saturation, value, encoding
    uint4 camera_profile_meta;       // pixel count, tone points, row stride, reserved
};

ByteAddressBuffer camera_profile_source : register(t5);
ByteAddressBuffer camera_profile_look : register(t6);
ByteAddressBuffer camera_profile_tone : register(t7);
RWByteAddressBuffer camera_profile_output : register(u1);

float camera_profile_constant(uint index) {
    if (index < 4) return camera_profile_constants0[index];
    if (index < 8) return camera_profile_constants1[index - 4];
    if (index < 12) return camera_profile_constants2[index - 8];
    if (index < 16) return camera_profile_constants3[index - 12];
    if (index < 20) return camera_profile_constants4[index - 16];
    return camera_profile_constants5[index - 20];
}

uint camera_profile_load_u16(uint sample_index) {
    uint byte_offset = sample_index * 2;
    uint packed = camera_profile_source.Load(byte_offset & ~3u);
    return (packed >> ((byte_offset & 2u) * 8u)) & 0xffffu;
}

float camera_profile_load_look(uint index) {
    return asfloat(camera_profile_look.Load(index * 4));
}

float camera_profile_load_tone(uint index) {
    return asfloat(camera_profile_tone.Load(index * 4));
}

float camera_profile_encode_srgb(float value) {
    value = max(value, 0.0);
    if (value <= 0.0031308) return value * 12.92;
    return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

float camera_profile_decode_srgb(float value) {
    if (value <= 0.04045) return value / 12.92;
    return pow((value + 0.055) / 1.055, 2.4);
}

float camera_profile_remainder_360(float value) {
    float result = fmod(value, 360.0);
    if (result < 0.0) result += 360.0;
    return result;
}

struct CameraProfileHsv {
    float h;
    float s;
    float v;
};

CameraProfileHsv camera_profile_rgb_to_hsv(float r, float g, float b) {
    float value = max(r, max(g, b));
    float minimum = min(r, min(g, b));
    float difference = value - minimum;
    float saturation = difference / (abs(value) + 1.1920928955078125e-7);
    float hue = 0.0;
    if (difference > 1.1920928955078125e-7) {
        if (value == r) hue = g - b;
        else if (value == g) hue = (b - r) + 2.0 * difference;
        else hue = (r - g) + 4.0 * difference;
        hue *= 60.0 / difference;
        if (hue < 0.0) hue += 360.0;
    }
    CameraProfileHsv result;
    result.h = hue;
    result.s = saturation;
    result.v = value;
    return result;
}

void camera_profile_hsv_to_rgb(float h, float s, float v,
                               out float r, out float g, out float b) {
    float sector_position = h * (1.0 / 60.0);
    if (sector_position < 0.0) sector_position += 6.0;
    else if (sector_position >= 6.0) sector_position -= 6.0;
    int sector = (int)floor(sector_position);
    float fraction = sector_position - (float)sector;
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * fraction);
    float t = v * (1.0 - s * (1.0 - fraction));
    if (sector == 0) { r = v; g = t; b = p; }
    else if (sector == 1) { r = q; g = v; b = p; }
    else if (sector == 2) { r = p; g = v; b = t; }
    else if (sector == 3) { r = p; g = q; b = v; }
    else if (sector == 4) { r = t; g = p; b = v; }
    else { r = v; g = p; b = q; }
}

float camera_profile_look_sample(uint value_index, uint hue_index,
                                 uint saturation_index, uint channel) {
    uint at = (((value_index * camera_profile_dimensions.x + hue_index) *
                camera_profile_dimensions.y + saturation_index) * 3) + channel;
    return camera_profile_load_look(at);
}

void camera_profile_apply_look(inout float r, inout float g, inout float b) {
    CameraProfileHsv hsv = camera_profile_rgb_to_hsv(r, g, b);
    float encoded_value = camera_profile_dimensions.w == 1
        ? camera_profile_encode_srgb(hsv.v) : hsv.v;
    float hue_coord = camera_profile_remainder_360(hsv.h) *
                      ((float)camera_profile_dimensions.x / 360.0);
    float saturation_coord = clamp(hsv.s, 0.0, 1.0) *
                             (float)(camera_profile_dimensions.y - 1);
    float value_coord = clamp(encoded_value, 0.0, 1.0) *
                        (float)(camera_profile_dimensions.z - 1);
    uint h0 = (uint)floor(hue_coord);
    uint h1 = (h0 + 1u) % camera_profile_dimensions.x;
    uint s_floor = (uint)floor(saturation_coord);
    uint v_floor = (uint)floor(value_coord);
    uint s0 = min(s_floor, camera_profile_dimensions.y - 2u);
    uint v0 = min(v_floor, camera_profile_dimensions.z - 2u);
    float hf = hue_coord - (float)h0;
    float sf = saturation_coord - (float)s0;
    float vf = value_coord - (float)v0;
    float3 mods = 0.0;
    [unroll] for (uint dh = 0; dh < 2; ++dh) {
        uint hi = dh ? h1 : h0;
        float hw = dh ? hf : (1.0 - hf);
        [unroll] for (uint ds = 0; ds < 2; ++ds) {
            uint si = s0 + ds;
            float sw = ds ? sf : (1.0 - sf);
            [unroll] for (uint dv = 0; dv < 2; ++dv) {
                uint vi = v0 + dv;
                float vw = dv ? vf : (1.0 - vf);
                float weight = hw * sw * vw;
                [unroll] for (uint channel = 0; channel < 3; ++channel) {
                    float term = weight * camera_profile_look_sample(vi, hi, si, channel);
                    mods[channel] = mods[channel] + term;
                }
            }
        }
    }
    hsv.h = camera_profile_remainder_360(hsv.h + mods[0]);
    hsv.s = clamp(hsv.s * mods[1], 0.0, 1.0);
    float new_value = clamp(encoded_value * mods[2], 0.0, 1.0);
    hsv.v = camera_profile_dimensions.w == 1
        ? camera_profile_decode_srgb(new_value) : new_value;
    camera_profile_hsv_to_rgb(hsv.h, hsv.s, hsv.v, r, g, b);
}

float camera_profile_interp_curve(float x) {
    uint points = camera_profile_meta.y;
    float first_x = camera_profile_load_tone(0);
    if (x <= first_x) return camera_profile_load_tone(1);
    float last_x = camera_profile_load_tone((points - 1) * 2);
    if (x >= last_x) return camera_profile_load_tone((points - 1) * 2 + 1);
    uint low = 0;
    uint high = points;
    while (low < high) {
        uint middle = low + (high - low) / 2;
        if (camera_profile_load_tone(middle * 2) <= x) low = middle + 1;
        else high = middle;
    }
    uint right = low;
    uint left = right - 1;
    float x0 = camera_profile_load_tone(left * 2);
    float y0 = camera_profile_load_tone(left * 2 + 1);
    float x1 = camera_profile_load_tone(right * 2);
    float y1 = camera_profile_load_tone(right * 2 + 1);
    float slope = (y1 - y0) / (x1 - x0);
    return slope * (x - x0) + y0;
}

float camera_profile_encode_srgb_final(float value) {
    value = max(value, 0.0);
    if (value <= 0.0031308) return value * 12.92;
    return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

uint camera_profile_quantize_rgb8(float value) {
    float scaled = camera_profile_encode_srgb_final(value) * 255.0;
    if (!(scaled > 0.0)) return 0;
    if (scaled >= 255.0) return 255;
    float lower = floor(scaled);
    float fraction = scaled - lower;
    uint rounded = (uint)lower;
    if (fraction > 0.5 || (fraction == 0.5 && (rounded & 1u) != 0u)) ++rounded;
    return min(rounded, 255u);
}

[numthreads(256, 1, 1)]
void render_camera_profile(uint3 thread_id : SV_DispatchThreadID) {
    uint pixel = thread_id.y * camera_profile_meta.z + thread_id.x;
    if (pixel >= camera_profile_meta.x) return;
    uint at = pixel * 3;
    float camera_r = (float)camera_profile_load_u16(at) / 65535.0;
    float camera_g = (float)camera_profile_load_u16(at + 1) / 65535.0;
    float camera_b = (float)camera_profile_load_u16(at + 2) / 65535.0;
    float clipped_r = min(camera_r, camera_profile_constant(9));
    float clipped_g = min(camera_g, camera_profile_constant(10));
    float clipped_b = min(camera_b, camera_profile_constant(11));
    precise float matrix_r = (clipped_r * camera_profile_constant(0) +
                              clipped_g * camera_profile_constant(1)) +
                             clipped_b * camera_profile_constant(2);
    precise float matrix_g = (clipped_r * camera_profile_constant(3) +
                              clipped_g * camera_profile_constant(4)) +
                             clipped_b * camera_profile_constant(5);
    precise float matrix_b = (clipped_r * camera_profile_constant(6) +
                              clipped_g * camera_profile_constant(7)) +
                             clipped_b * camera_profile_constant(8);
    float r = clamp(matrix_r, 0.0, 1.0) * camera_profile_constant(12);
    float g = clamp(matrix_g, 0.0, 1.0) * camera_profile_constant(12);
    float b = clamp(matrix_b, 0.0, 1.0) * camera_profile_constant(12);
    r = clamp(r, 0.0, 1.0);
    g = clamp(g, 0.0, 1.0);
    b = clamp(b, 0.0, 1.0);
    camera_profile_apply_look(r, g, b);

    float low = min(r, min(g, b));
    float high = max(r, max(g, b));
    float span = high - low;
    float new_low = camera_profile_interp_curve(low);
    float new_high = camera_profile_interp_curve(high);
    float3 positions = 0.0;
    if (span > 1.0e-12) positions = (float3(r, g, b) - low) / span;
    float tone_r = new_low + (new_high - new_low) * positions.r;
    float tone_g = new_low + (new_high - new_low) * positions.g;
    float tone_b = new_low + (new_high - new_low) * positions.b;
    precise float projected_r = (tone_r * camera_profile_constant(13) +
                                 tone_g * camera_profile_constant(14)) +
                                tone_b * camera_profile_constant(15);
    precise float projected_g = (tone_r * camera_profile_constant(16) +
                                 tone_g * camera_profile_constant(17)) +
                                tone_b * camera_profile_constant(18);
    precise float projected_b = (tone_r * camera_profile_constant(19) +
                                 tone_g * camera_profile_constant(20)) +
                                tone_b * camera_profile_constant(21);
    camera_profile_output.Store(at * 4, camera_profile_quantize_rgb8(projected_r));
    camera_profile_output.Store((at + 1) * 4, camera_profile_quantize_rgb8(projected_g));
    camera_profile_output.Store((at + 2) * 4, camera_profile_quantize_rgb8(projected_b));
}
