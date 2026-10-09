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
