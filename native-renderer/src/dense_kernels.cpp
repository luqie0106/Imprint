#include "imprint_renderer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

constexpr float kLumaR = 0.2126f;
constexpr float kLumaG = 0.7152f;
constexpr float kLumaB = 0.0722f;
constexpr size_t kParallelThreshold = 65536;
constexpr unsigned int kMaxWorkers = 8;

struct RGB {
    float r;
    float g;
    float b;
};

struct PhysicalConstants {
    float transmission_floor;
    float transmission_knee_width;
    float subtraction_budget_factor;
    float physical_colour_factor;
    float chroma_recovery_factor;
    float contrast_factor;
};

void join_threads(std::vector<std::thread> &threads) {
    for (std::thread &worker : threads) {
        if (worker.joinable()) worker.join();
    }
}

template <typename Function>
bool parallel_for(size_t count, Function &&function) {
    if (count < kParallelThreshold) {
        try {
            for (size_t index = 0; index < count; ++index) function(index);
            return true;
        } catch (...) {
            return false;
        }
    }

    unsigned int workers = std::thread::hardware_concurrency();
    if (workers == 0) workers = 1;
    workers = std::min(workers, kMaxWorkers);
    if (workers < 2) {
        try {
            for (size_t index = 0; index < count; ++index) function(index);
            return true;
        } catch (...) {
            return false;
        }
    }

    const size_t chunk = (count + workers - 1) / workers;
    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;
    try {
        threads.reserve(workers - 1);
        for (unsigned int worker = 1; worker < workers; ++worker) {
            const size_t begin = static_cast<size_t>(worker) * chunk;
            if (begin >= count) break;
            const size_t end = std::min(count, begin + chunk);
            threads.emplace_back([&function, &failed, begin, end] {
                try {
                    for (size_t index = begin; index < end; ++index) function(index);
                } catch (...) {
                    failed.store(true, std::memory_order_relaxed);
                }
            });
        }
        const size_t main_end = std::min(count, chunk);
        try {
            for (size_t index = 0; index < main_end; ++index) function(index);
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
        }
    } catch (...) {
        join_threads(threads);
        return false;
    }
    join_threads(threads);
    return !failed.load(std::memory_order_relaxed);
}

float clamp(float value, float low, float high) {
    return std::min(high, std::max(low, value));
}

float luminance(RGB value) {
    const float rg = value.r * kLumaR + value.g * kLumaG;
    return rg + value.b * kLumaB;
}

float smoothstep(float low, float high, float value) {
    const float position = clamp((value - low) / (high - low), 0.0f, 1.0f);
    return position * position * (3.0f - 2.0f * position);
}

bool valid_dehaze_params(const im_dehaze_params &params) {
    const float values[] = {
        params.strength, params.naturalness, params.fog_retention,
        params.local_contrast, params.color_recovery, params.color_protection,
        params.highlight_protection, params.shadow_protection,
        params.brightness_protection,
    };
    for (float value : values) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) return false;
    }
    return true;
}

bool valid_basic_params(const im_basic_params &params) {
    const float values[] = {
        params.exposure, params.contrast, params.highlights, params.shadows,
        params.whites, params.blacks, params.vibrance, params.saturation,
    };
    for (size_t index = 0; index < sizeof(values) / sizeof(values[0]); ++index) {
        const float low = index == 0 ? -5.0f : -100.0f;
        const float high = index == 0 ? 5.0f : 100.0f;
        if (!std::isfinite(values[index]) || values[index] < low || values[index] > high) {
            return false;
        }
    }
    return true;
}

bool image_extent(uint32_t width, uint32_t height, size_t *pixels, size_t *rgb_values) {
    if (width == 0 || height == 0 || width > 65535 || height > 65535 || !pixels || !rgb_values) return false;
    const uint64_t pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    const uint64_t max_values = static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 3);
    constexpr uint64_t kAbiMaxRgbValues = 1ull << 29;
    if (pixel_count > max_values || pixel_count > kAbiMaxRgbValues / 3) return false;
    *pixels = static_cast<size_t>(pixel_count);
    *rgb_values = *pixels * 3;
    return true;
}

template <typename T>
bool byte_length(size_t count, size_t *bytes) {
    if (!bytes || count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
    *bytes = count * sizeof(T);
    return true;
}

bool ranges_overlap(const void *left, size_t left_bytes, const void *right, size_t right_bytes) {
    const uintptr_t left_start = reinterpret_cast<uintptr_t>(left);
    const uintptr_t right_start = reinterpret_cast<uintptr_t>(right);
    const uintptr_t max_address = std::numeric_limits<uintptr_t>::max();
    if (left_bytes > max_address - left_start || right_bytes > max_address - right_start) return true;
    const uintptr_t left_end = left_start + left_bytes;
    const uintptr_t right_end = right_start + right_bytes;
    return left_start < right_end && right_start < left_end;
}

constexpr size_t kAbiMaxDenseValues = static_cast<size_t>(1ull << 29);

struct RemapAxis {
    uint32_t first;
    uint32_t second;
    double fraction;
};

uint32_t round_nonnegative_ties_even(double value) {
    const double lower_value = std::floor(value);
    const double fraction = value - lower_value;
    uint32_t lower = static_cast<uint32_t>(lower_value);
    if (fraction > 0.5 || (fraction == 0.5 && (lower & 1u) != 0u)) ++lower;
    return lower;
}

uint32_t reflect101_index(int64_t index, uint32_t extent) {
    if (extent <= 1) return 0;
    const int64_t edge = static_cast<int64_t>(extent - 1);
    const int64_t period = 2 * edge;
    int64_t folded = index % period;
    if (folded < 0) folded += period;
    if (folded > edge) folded = period - folded;
    return static_cast<uint32_t>(folded);
}

RemapAxis remap_axis_continuous(float coordinate, uint32_t extent) {
    if (extent <= 1) return {0, 0, 0};

    // Fold in double precision to keep all finite float coordinates safe,
    // including coordinates outside the range of integer conversions. The
    // OpenCV float-map remap path interpolates at the supplied float
    // coordinate directly; it does not quantize fractions to 1/32 here.
    const double edge = static_cast<double>(extent - 1);
    const double period = 2.0 * edge;
    double folded = std::fmod(static_cast<double>(coordinate), period);
    if (folded < 0.0) folded += period;
    if (folded > edge) folded = period - folded;

    const uint32_t first = static_cast<uint32_t>(std::floor(folded));
    const double fraction = folded - static_cast<double>(first);
    uint32_t second = first + 1u;
    if (second >= extent) second = extent - 2u;
    return {first, second, fraction};
}

RemapAxis remap_axis_fixed(float coordinate, uint32_t extent) {
    if (extent <= 1) return {0, 0, 0};

    const double floor_value = std::floor(static_cast<double>(coordinate));
    const float fraction = coordinate - static_cast<float>(floor_value);
    const uint32_t quantized_fraction = round_nonnegative_ties_even(
        static_cast<double>(fraction * 32.0f));
    double base = floor_value;
    uint32_t fraction_bits = quantized_fraction;
    if (fraction_bits == 32u) {
        base += 1.0;
        fraction_bits = 0;
    }

    // OpenCV's float-map conversion stores integer coordinates in signed
    // 16-bit map entries, while fractions remain in the 1/32 table index.
    // Saturating here also avoids unsafe conversions for extreme finite maps.
    constexpr double kMapCoordinateMin = -32768.0;
    constexpr double kMapCoordinateMax = 32767.0;
    const int64_t first_unfolded = static_cast<int64_t>(
        std::max(kMapCoordinateMin, std::min(kMapCoordinateMax, base)));
    return {reflect101_index(first_unfolded, extent),
            reflect101_index(first_unfolded + 1, extent),
            static_cast<double>(fraction_bits) / 32.0};
}

uint16_t remap_sample_channel(const uint16_t *source, uint32_t width,
                              uint32_t height, uint32_t channel,
                              float x_coordinate, float y_coordinate,
                              uint32_t interpolation_mode) {
    const RemapAxis x = interpolation_mode == 0
        ? remap_axis_fixed(x_coordinate, width)
        : remap_axis_continuous(x_coordinate, width);
    const RemapAxis y = interpolation_mode == 0
        ? remap_axis_fixed(y_coordinate, height)
        : remap_axis_continuous(y_coordinate, height);
    const size_t row0 = static_cast<size_t>(y.first) * width;
    const size_t row1 = static_cast<size_t>(y.second) * width;
    const size_t offset00 = (row0 + x.first) * 3u + channel;
    const size_t offset01 = (row0 + x.second) * 3u + channel;
    const size_t offset10 = (row1 + x.first) * 3u + channel;
    const size_t offset11 = (row1 + x.second) * 3u + channel;
    const double fx = x.fraction;
    const double fy = y.fraction;
    const double top = static_cast<double>(source[offset00]) * (1.0 - fx) +
                       static_cast<double>(source[offset01]) * fx;
    const double bottom = static_cast<double>(source[offset10]) * (1.0 - fx) +
                          static_cast<double>(source[offset11]) * fx;
    const double interpolated = top * (1.0 - fy) + bottom * fy;
    const uint32_t rounded = round_nonnegative_ties_even(interpolated);
    return static_cast<uint16_t>(std::min<uint32_t>(rounded, 65535u));
}

bool finite_rgb(RGB value) {
    return std::isfinite(value.r) && std::isfinite(value.g) && std::isfinite(value.b);
}

bool normalized_rgb(RGB value) {
    return finite_rgb(value) && value.r >= 0.0f && value.r <= 1.0f &&
           value.g >= 0.0f && value.g <= 1.0f && value.b >= 0.0f && value.b <= 1.0f;
}

RGB read_rgb(const float *values, size_t pixel) {
    const size_t at = pixel * 3;
    return {values[at], values[at + 1], values[at + 2]};
}

void write_rgb(float *values, size_t pixel, RGB value) {
    const size_t at = pixel * 3;
    values[at] = value.r;
    values[at + 1] = value.g;
    values[at + 2] = value.b;
}

PhysicalConstants physical_constants(const im_dehaze_params &params) {
    const double strength = static_cast<double>(params.strength);
    const double naturalness = static_cast<double>(params.naturalness);
    const double gain = 1.0 + 0.8 * strength * (1.0 - 0.35 * naturalness);
    const double inverse_floor = 1.0 / gain;
    const double retention = 0.10 + 0.45 * static_cast<double>(params.brightness_protection) +
                             0.10 * static_cast<double>(params.shadow_protection);
    const double colour_factor = (1.0 - static_cast<double>(params.color_protection)) *
        (0.28 + 0.72 * (1.0 - naturalness));
    const double recovery_factor = 0.25 * static_cast<double>(params.color_recovery) * strength;
    const double contrast_factor = 0.15 * static_cast<double>(params.local_contrast) * strength;
    return {
        static_cast<float>(inverse_floor),
        static_cast<float>(0.015 * (1.0 - inverse_floor)),
        static_cast<float>(1.0 - retention),
        static_cast<float>(colour_factor),
        static_cast<float>(recovery_factor),
        static_cast<float>(contrast_factor),
    };
}

float inverse_transmission(RGB source, float transmission, const im_dehaze_params &params,
                           const PhysicalConstants &constants) {
    const float floor = constants.transmission_floor;
    const float width = constants.transmission_knee_width;
    float effective = std::max(transmission, floor);
    if (width > 0.0f) {
        const float knee = std::max(width - std::abs(transmission - floor), 0.0f);
        effective = effective + knee * knee / (4.0f * width);
    }
    const float peak = std::max(source.r, std::max(source.g, source.b));
    const float highlight = smoothstep(0.55f, 0.95f, peak) * params.highlight_protection;
    return clamp(effective + (1.0f - effective) * highlight, 1e-4f, 1.0f);
}

RGB gamut(RGB value) {
    const float y = luminance(value);
    const RGB chroma{value.r - y, value.g - y, value.b - y};
    const float rooms[] = {
        chroma.r > 0.0f ? (1.0f - y) / std::max(chroma.r, 1e-7f)
                        : y / std::max(-chroma.r, 1e-7f),
        chroma.g > 0.0f ? (1.0f - y) / std::max(chroma.g, 1e-7f)
                        : y / std::max(-chroma.g, 1e-7f),
        chroma.b > 0.0f ? (1.0f - y) / std::max(chroma.b, 1e-7f)
                        : y / std::max(-chroma.b, 1e-7f),
    };
    const float scale = clamp(std::min(rooms[0], std::min(rooms[1], rooms[2])), 0.0f, 1.0f);
    return {clamp(y + chroma.r * scale, 0.0f, 1.0f),
            clamp(y + chroma.g * scale, 0.0f, 1.0f),
            clamp(y + chroma.b * scale, 0.0f, 1.0f)};
}

RGB source_colour_ceiling(RGB source, RGB result) {
    const float source_y = luminance(source);
    const float result_y = std::min(luminance(result), source_y);
    const float base_scale = result_y / std::max(source_y, 1e-20f);
    const RGB base{source.r * base_scale, source.g * base_scale, source.b * base_scale};
    const RGB deviation{result.r - base.r, result.g - base.g, result.b - base.b};
    const float upper_room[] = {
        std::max(source.r - base.r, 0.0f) / std::max(deviation.r, 1e-20f),
        std::max(source.g - base.g, 0.0f) / std::max(deviation.g, 1e-20f),
        std::max(source.b - base.b, 0.0f) / std::max(deviation.b, 1e-20f),
    };
    const float lower_room[] = {
        base.r / std::max(-deviation.r, 1e-20f),
        base.g / std::max(-deviation.g, 1e-20f),
        base.b / std::max(-deviation.b, 1e-20f),
    };
    const float room_r = deviation.r > 0.0f ? upper_room[0] : lower_room[0];
    const float room_g = deviation.g > 0.0f ? upper_room[1] : lower_room[1];
    const float room_b = deviation.b > 0.0f ? upper_room[2] : lower_room[2];
    const float colour_scale = clamp(std::min(room_r, std::min(room_g, room_b)), 0.0f, 1.0f);
    return {clamp(base.r + deviation.r * colour_scale, 0.0f, 1.0f),
            clamp(base.g + deviation.g * colour_scale, 0.0f, 1.0f),
            clamp(base.b + deviation.b * colour_scale, 0.0f, 1.0f)};
}

RGB physical_pixel(RGB source, float transmission, RGB air, const im_dehaze_params &params,
                   const PhysicalConstants &constants) {
    const float source_y = luminance(source);
    const float effective_t = inverse_transmission(source, transmission, params, constants);
    const RGB delta{source.r - air.r, source.g - air.g, source.b - air.b};
    const float source_channels[] = {source.r, source.g, source.b};
    const float air_channels[] = {air.r, air.g, air.b};
    const float delta_channels[] = {delta.r, delta.g, delta.b};
    float positive_channels[3];
    float negative_channels[3];
    for (size_t channel = 0; channel < 3; ++channel) {
        const float channel_delta = delta_channels[channel];
        const float shoulder = effective_t + (1.0f - effective_t) *
            std::max(channel_delta, 0.0f) / std::max(1.0f - air_channels[channel], 1e-4f);
        positive_channels[channel] = air_channels[channel] + channel_delta / shoulder;

        const float deficit = std::max(-channel_delta, 0.0f);
        const float smooth_deficit = deficit * deficit /
            (deficit + 0.025f * air_channels[channel] + 1e-8f);
        const float loss = (1.0f / effective_t - 1.0f) * smooth_deficit;
        const float source_value = source_channels[channel];
        const float budget = (constants.subtraction_budget_factor * source_value * source_value /
            (source_value + 0.12f * air_channels[channel] + 1e-8f)) * params.strength;
        const float fraction = loss / std::max(budget, 1e-8f);
        float bounded = fraction;
        if (fraction > 0.5f) {
            bounded = 1.0f - 0.5f * std::exp(-2.0f * std::max(fraction - 0.5f, 0.0f));
        }
        negative_channels[channel] = source_value - budget * bounded;
    }

    const RGB recovered{
        delta.r < 0.0f ? negative_channels[0] : positive_channels[0],
        delta.g < 0.0f ? negative_channels[1] : positive_channels[1],
        delta.b < 0.0f ? negative_channels[2] : positive_channels[2],
    };
    const float recovered_y = luminance(recovered);
    const float hue_scale = recovered_y / std::max(source_y, 1e-6f);
    const RGB source_hue{source.r * hue_scale, source.g * hue_scale, source.b * hue_scale};

    const float dr = source.r - source_y;
    const float dg = source.g - source_y;
    const float db = source.b - source_y;
    const float chroma_length = std::sqrt((dr * dr + dg * dg) + db * db);
    const float chroma_confidence = smoothstep(0.005f, 0.08f, chroma_length);
    const float physical_colour = constants.physical_colour_factor * chroma_confidence;
    RGB result{
        recovered.r * physical_colour + source_hue.r * (1.0f - physical_colour),
        recovered.g * physical_colour + source_hue.g * (1.0f - physical_colour),
        recovered.b * physical_colour + source_hue.b * (1.0f - physical_colour),
    };

    const float result_y = luminance(result);
    RGB chroma{result.r - result_y, result.g - result_y, result.b - result_y};
    const float chroma_gain = 1.0f + constants.chroma_recovery_factor * chroma_confidence;
    chroma = {chroma.r * chroma_gain, chroma.g * chroma_gain, chroma.b * chroma_gain};
    float contrast_delta = constants.contrast_factor * result_y *
        (1.0f - result_y) * (2.0f * result_y - 1.0f);
    contrast_delta = clamp(contrast_delta, -0.02f * result_y, 0.02f * (1.0f - result_y));
    const float tone_y = clamp(result_y + contrast_delta, 0.0f, 1.0f);
    result = {tone_y + chroma.r, tone_y + chroma.g, tone_y + chroma.b};
    result = gamut(result);

    const float gamut_y = luminance(result);
    const float solar_scale = std::min(1.0f, source_y / std::max(gamut_y, 1e-6f));
    result = {result.r * solar_scale, result.g * solar_scale, result.b * solar_scale};
    return source_colour_ceiling(source, result);
}

RGB protect_dark_pixel(RGB source, RGB result, float floor_level, float width) {
    if (floor_level <= 1e-8f) return result;
    const float source_y = luminance(source);
    const float result_y = luminance(result);
    const float floor = floor_level * -std::expm1(-source_y / floor_level);
    const float delta = result_y - floor;
    float target_y = std::max(result_y, floor);
    const float knee = std::max(width - std::abs(delta), 0.0f);
    target_y = target_y + knee * knee / (4.0f * width);
    const float lift = std::max(target_y - result_y, 0.0f);
    const float lift_scale = lift / std::max(source_y, 1e-20f);
    const RGB candidate{result.r + source.r * lift_scale,
                        result.g + source.g * lift_scale,
                        result.b + source.b * lift_scale};
    const RGB protected_result = source_colour_ceiling(source, candidate);
    return lift > 0.0f ? protected_result : result;
}

RGB basic_pixel(RGB source, const im_basic_params &params, float maximum) {
    const float exposure_gain = static_cast<float>(std::pow(2.0, static_cast<double>(params.exposure)));
    const float contrast_scale = static_cast<float>(1.0 + static_cast<double>(params.contrast) / 100.0);
    RGB rgb{
        (source.r / maximum) * exposure_gain,
        (source.g / maximum) * exposure_gain,
        (source.b / maximum) * exposure_gain,
    };
    rgb = {(rgb.r - 0.5f) * contrast_scale + 0.5f,
           (rgb.g - 0.5f) * contrast_scale + 0.5f,
           (rgb.b - 0.5f) * contrast_scale + 0.5f};

    const float original_luminance = luminance(rgb);
    const float shadow_base = clamp((0.62f - original_luminance) / 0.62f, 0.0f, 1.0f);
    const float highlight_base = clamp((original_luminance - 0.38f) / 0.62f, 0.0f, 1.0f);
    const float shadow_mask = std::pow(shadow_base, 1.5f);
    const float highlight_mask = std::pow(highlight_base, 1.5f);
    // Python computes each control combination as a scalar double, then NumPy
    // applies it to float32 arrays. Preserve that conversion point here.
    const float shadow_lift = static_cast<float>(
        static_cast<double>(params.shadows) * 0.0022 +
        static_cast<double>(params.blacks) * 0.0008);
    const float highlight_lift = static_cast<float>(
        static_cast<double>(params.highlights) * 0.0018 +
        static_cast<double>(params.whites) * 0.0008);
    rgb = {rgb.r + shadow_lift * shadow_mask + highlight_lift * highlight_mask,
           rgb.g + shadow_lift * shadow_mask + highlight_lift * highlight_mask,
           rgb.b + shadow_lift * shadow_mask + highlight_lift * highlight_mask};

    const float adjusted_luminance = luminance(rgb);
    const RGB chroma{rgb.r - adjusted_luminance,
                     rgb.g - adjusted_luminance,
                     rgb.b - adjusted_luminance};
    const float chroma_level = std::max(std::abs(chroma.r),
                                        std::max(std::abs(chroma.g), std::abs(chroma.b)));
    const float vibrance_amount = static_cast<float>(static_cast<double>(params.vibrance) / 100.0);
    const float vibrance_factor = 1.0f + vibrance_amount * clamp(1.0f - chroma_level, 0.0f, 1.0f);
    const float saturation_factor = static_cast<float>(1.0 + static_cast<double>(params.saturation) / 100.0);
    return {adjusted_luminance + chroma.r * saturation_factor * vibrance_factor,
            adjusted_luminance + chroma.g * saturation_factor * vibrance_factor,
            adjusted_luminance + chroma.b * saturation_factor * vibrance_factor};
}

uint32_t quantize_basic_channel(float value, float maximum) {
    const float scaled = clamp(value, 0.0f, 1.0f) * maximum;
    return round_nonnegative_ties_even(static_cast<double>(scaled));
}

template <typename Sample>
im_status basic_rgb_run(uint32_t width, uint32_t height, const Sample *source,
                        size_t source_samples, const im_basic_params *params,
                        Sample *destination, size_t destination_samples) {
    size_t pixels = 0, rgb_samples = 0;
    if (!source || !params || !destination ||
        !image_extent(width, height, &pixels, &rgb_samples) ||
        source_samples != rgb_samples || destination_samples != rgb_samples ||
        !valid_basic_params(*params)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    size_t source_bytes = 0, destination_bytes = 0, params_bytes = 0;
    if (!byte_length<Sample>(source_samples, &source_bytes) ||
        !byte_length<Sample>(destination_samples, &destination_bytes) ||
        !byte_length<im_basic_params>(1, &params_bytes) ||
        ranges_overlap(source, source_bytes, destination, destination_bytes) ||
        ranges_overlap(params, params_bytes, destination, destination_bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    const im_basic_params controls = *params;
    const float maximum = static_cast<float>(std::numeric_limits<Sample>::max());
    const bool completed = parallel_for(pixels, [&](size_t pixel) {
        const size_t base = pixel * 3u;
        const RGB input{static_cast<float>(source[base]),
                        static_cast<float>(source[base + 1u]),
                        static_cast<float>(source[base + 2u])};
        const RGB output = basic_pixel(input, controls, maximum);
        destination[base] = static_cast<Sample>(quantize_basic_channel(output.r, maximum));
        destination[base + 1u] = static_cast<Sample>(quantize_basic_channel(output.g, maximum));
        destination[base + 2u] = static_cast<Sample>(quantize_basic_channel(output.b, maximum));
    });
    return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
}

bool rgb_has_overlap(const float *left, size_t left_count, const float *right, size_t right_count) {
    size_t left_bytes = 0, right_bytes = 0;
    return !byte_length<float>(left_count, &left_bytes) ||
           !byte_length<float>(right_count, &right_bytes) ||
           ranges_overlap(left, left_bytes, right, right_bytes);
}

bool valid_edge(uint32_t edge) {
    return edge >= 2 && edge <= 129;
}

size_t cube_values(uint32_t edge) {
    const size_t side = static_cast<size_t>(edge);
    return side * side * side;
}

} // namespace

extern "C" IMPRINT_API im_status im_native_lens_remap_rgb16(
    uint32_t width, uint32_t height, const uint16_t *source, size_t source_values,
    const float *maps, size_t map_values, uint32_t strip_height,
    uint32_t map_channels, uint16_t *destination, size_t destination_values,
    uint32_t interpolation_mode) {
    try {
        constexpr uint32_t kMaxRemapDimension = 32766;
        if (!source || !maps || !destination || width == 0 || height == 0 ||
            strip_height == 0 || width > kMaxRemapDimension ||
            height > kMaxRemapDimension || strip_height > height ||
            (map_channels != 1 && map_channels != 3) || interpolation_mode > 1) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        const uint64_t source_pixels_u64 = static_cast<uint64_t>(width) * height;
        const uint64_t strip_pixels_u64 = static_cast<uint64_t>(width) * strip_height;
        const uint64_t source_values_u64 = source_pixels_u64 * 3u;
        const uint64_t output_values_u64 = strip_pixels_u64 * 3u;
        const uint64_t map_values_u64 = strip_pixels_u64 * map_channels * 2u;
        if (source_values_u64 > kAbiMaxDenseValues ||
            output_values_u64 > kAbiMaxDenseValues ||
            map_values_u64 > kAbiMaxDenseValues ||
            source_values != static_cast<size_t>(source_values_u64) ||
            map_values != static_cast<size_t>(map_values_u64) ||
            destination_values < static_cast<size_t>(output_values_u64) ||
            destination_values > kAbiMaxDenseValues) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        size_t source_bytes = 0, maps_bytes = 0, destination_bytes = 0;
        if (!byte_length<uint16_t>(source_values, &source_bytes) ||
            !byte_length<float>(map_values, &maps_bytes) ||
            !byte_length<uint16_t>(destination_values, &destination_bytes) ||
            ranges_overlap(source, source_bytes, destination, destination_bytes) ||
            ranges_overlap(maps, maps_bytes, destination, destination_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        // Validate every map coordinate before any destination sample is
        // written, so malformed maps cannot leave a partial output strip.
        for (size_t index = 0; index < map_values; ++index) {
            if (!std::isfinite(maps[index])) return IM_STATUS_INVALID_ARGUMENT;
        }

        const size_t output_pixels = static_cast<size_t>(strip_pixels_u64);
        const bool completed = parallel_for(output_pixels, [&](size_t pixel) {
            const size_t map_base = pixel * map_channels * 2u;
            const size_t destination_base = pixel * 3u;
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const size_t coordinate_index = map_base +
                    (map_channels == 1 ? 0u : static_cast<size_t>(channel) * 2u);
                destination[destination_base + channel] = remap_sample_channel(
                    source, width, height, channel,
                    maps[coordinate_index], maps[coordinate_index + 1u],
                    interpolation_mode);
            }
        });
        return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_physical_float_run(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *transmission, size_t transmission_count, const float *airlight_rgb,
    const im_dehaze_params *params, float *destination, size_t destination_values) {
    try {
        size_t pixels = 0, rgb_values = 0;
        if (!source || !transmission || !airlight_rgb || !params || !destination ||
            !image_extent(width, height, &pixels, &rgb_values) ||
            source_values != rgb_values || transmission_count != pixels ||
            destination_values < rgb_values || !valid_dehaze_params(*params)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        size_t source_bytes = 0, destination_bytes = 0, transmission_bytes = 0;
        if (!byte_length<float>(source_values, &source_bytes) ||
            !byte_length<float>(destination_values, &destination_bytes) ||
            !byte_length<float>(transmission_count, &transmission_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        size_t airlight_bytes = 0, params_bytes = 0;
        if (!byte_length<float>(3, &airlight_bytes) ||
            !byte_length<im_dehaze_params>(1, &params_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        const bool exact_source_alias = source == destination;
        if ((!exact_source_alias && ranges_overlap(source, source_bytes, destination, destination_bytes)) ||
            ranges_overlap(destination, destination_bytes, transmission, transmission_bytes) ||
            ranges_overlap(destination, destination_bytes, airlight_rgb, airlight_bytes) ||
            ranges_overlap(destination, destination_bytes, params, params_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        const RGB air{airlight_rgb[0], airlight_rgb[1], airlight_rgb[2]};
        if (!normalized_rgb(air)) return IM_STATUS_INVALID_ARGUMENT;
        for (size_t value = 0; value < rgb_values; ++value) {
            if (!std::isfinite(source[value]) || source[value] < 0.0f || source[value] > 1.0f) {
                return IM_STATUS_INVALID_ARGUMENT;
            }
        }
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            if (!std::isfinite(transmission[pixel]) || transmission[pixel] < 0.0f ||
                transmission[pixel] > 1.0f) {
                return IM_STATUS_INVALID_ARGUMENT;
            }
        }

        const PhysicalConstants constants = physical_constants(*params);
        const bool completed = parallel_for(pixels, [&](size_t pixel) {
            const RGB input = read_rgb(source, pixel);
            write_rgb(destination, pixel,
                     physical_pixel(input, transmission[pixel], air, *params, constants));
        });
        return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_dark_guard_run(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *result, size_t result_values, float floor_level,
    float *destination, size_t destination_values) {
    try {
        size_t pixels = 0, rgb_values = 0;
        if (!source || !result || !destination || !image_extent(width, height, &pixels, &rgb_values) ||
            source_values != rgb_values || result_values != rgb_values ||
            destination_values < rgb_values || !std::isfinite(floor_level)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        size_t source_bytes = 0, result_bytes = 0, destination_bytes = 0;
        if (!byte_length<float>(source_values, &source_bytes) ||
            !byte_length<float>(result_values, &result_bytes) ||
            !byte_length<float>(destination_values, &destination_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        if (ranges_overlap(source, source_bytes, destination, destination_bytes) ||
            (result != destination && ranges_overlap(result, result_bytes, destination, destination_bytes))) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            const RGB input = read_rgb(source, pixel);
            const RGB output = read_rgb(result, pixel);
            if (!normalized_rgb(input) || !normalized_rgb(output)) return IM_STATUS_INVALID_ARGUMENT;
        }

        if (floor_level <= 1e-8f) {
            std::memmove(destination, result, rgb_values * sizeof(float));
            return IM_STATUS_OK;
        }
        const float width = static_cast<float>(0.02 * static_cast<double>(floor_level));
        const bool completed = parallel_for(pixels, [&](size_t pixel) {
            const RGB input = read_rgb(source, pixel);
            const RGB output = read_rgb(result, pixel);
            write_rgb(destination, pixel, protect_dark_pixel(input, output, floor_level, width));
        });
        return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_scalar_lut_run(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *lut, size_t lut_values, uint32_t edge,
    float *destination, size_t destination_count) {
    try {
        size_t pixels = 0, rgb_values = 0;
        if (!source || !lut || !destination || !image_extent(width, height, &pixels, &rgb_values) ||
            !valid_edge(edge) || source_values != rgb_values || lut_values != cube_values(edge) ||
            destination_count < pixels) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        size_t source_bytes = 0, lut_bytes = 0, destination_bytes = 0;
        if (!byte_length<float>(source_values, &source_bytes) ||
            !byte_length<float>(lut_values, &lut_bytes) ||
            !byte_length<float>(destination_count, &destination_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        if (ranges_overlap(source, source_bytes, destination, destination_bytes) ||
            ranges_overlap(lut, lut_bytes, destination, destination_bytes) ||
            ranges_overlap(source, source_bytes, lut, lut_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        for (size_t index = 0; index < lut_values; ++index) {
            if (!std::isfinite(lut[index])) return IM_STATUS_INVALID_ARGUMENT;
        }

        const size_t edge_size = static_cast<size_t>(edge);
        const bool completed = parallel_for(pixels, [&](size_t pixel) {
            const RGB input = read_rgb(source, pixel);
            if (!finite_rgb(input)) {
                destination[pixel] = 0.0f;
                return;
            }
            const float colors[] = {
                clamp(input.r, 0.0f, 1.0f) * static_cast<float>(edge - 1),
                clamp(input.g, 0.0f, 1.0f) * static_cast<float>(edge - 1),
                clamp(input.b, 0.0f, 1.0f) * static_cast<float>(edge - 1),
            };
            const size_t lower[] = {
                static_cast<size_t>(std::floor(colors[0])),
                static_cast<size_t>(std::floor(colors[1])),
                static_cast<size_t>(std::floor(colors[2])),
            };
            const float fraction[] = {
                colors[0] - static_cast<float>(lower[0]),
                colors[1] - static_cast<float>(lower[1]),
                colors[2] - static_cast<float>(lower[2]),
            };
            float value = 0.0f;
            for (uint32_t red_bit = 0; red_bit <= 1; ++red_bit) {
                const float red_weight = red_bit ? fraction[0] : 1.0f - fraction[0];
                const size_t red_index = std::min(lower[0] + red_bit, edge_size - 1);
                for (uint32_t green_bit = 0; green_bit <= 1; ++green_bit) {
                    const float green_weight = green_bit ? fraction[1] : 1.0f - fraction[1];
                    const size_t green_index = std::min(lower[1] + green_bit, edge_size - 1);
                    for (uint32_t blue_bit = 0; blue_bit <= 1; ++blue_bit) {
                        const float blue_weight = blue_bit ? fraction[2] : 1.0f - fraction[2];
                        const size_t blue_index = std::min(lower[2] + blue_bit, edge_size - 1);
                        const size_t lut_index = (red_index * edge_size + green_index) * edge_size + blue_index;
                        const float contribution = red_weight * green_weight * blue_weight * lut[lut_index];
                        value = value + contribution;
                    }
                }
            }
            destination[pixel] = value;
        });
        return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_lut_splat_run(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *depth, size_t depth_count, const float *confidence,
    size_t confidence_count, uint32_t edge, float depth_min, float depth_max,
    double *total, double *mass, size_t cube_count) {
    try {
        size_t pixels = 0, rgb_values = 0;
        if (!source || !depth || !confidence || !total || !mass ||
            !image_extent(width, height, &pixels, &rgb_values) || !valid_edge(edge) ||
            source_values != rgb_values || depth_count != pixels || confidence_count != pixels ||
            cube_count != cube_values(edge) || !std::isfinite(depth_min) ||
            !std::isfinite(depth_max) || depth_min > depth_max) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        size_t source_bytes = 0, depth_bytes = 0, confidence_bytes = 0, cube_bytes = 0;
        if (!byte_length<float>(source_values, &source_bytes) ||
            !byte_length<float>(depth_count, &depth_bytes) ||
            !byte_length<float>(confidence_count, &confidence_bytes) ||
            !byte_length<double>(cube_count, &cube_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        if (ranges_overlap(total, cube_bytes, mass, cube_bytes) ||
            ranges_overlap(total, cube_bytes, source, source_bytes) ||
            ranges_overlap(total, cube_bytes, depth, depth_bytes) ||
            ranges_overlap(total, cube_bytes, confidence, confidence_bytes) ||
            ranges_overlap(mass, cube_bytes, source, source_bytes) ||
            ranges_overlap(mass, cube_bytes, depth, depth_bytes) ||
            ranges_overlap(mass, cube_bytes, confidence, confidence_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        std::fill(total, total + cube_count, 0.0);
        std::fill(mass, mass + cube_count, 0.0);

        const size_t edge_size = static_cast<size_t>(edge);
        for (uint32_t red_bit = 0; red_bit <= 1; ++red_bit) {
            for (uint32_t green_bit = 0; green_bit <= 1; ++green_bit) {
                for (uint32_t blue_bit = 0; blue_bit <= 1; ++blue_bit) {
                    for (size_t pixel = 0; pixel < pixels; ++pixel) {
                        const RGB input = read_rgb(source, pixel);
                        const float weight_confidence = confidence[pixel];
                        const float input_depth = depth[pixel];
                        if (!finite_rgb(input) || !std::isfinite(weight_confidence) ||
                            weight_confidence <= 0.0f || !std::isfinite(input_depth)) {
                            continue;
                        }
                        const float colors[] = {
                            clamp(input.r, 0.0f, 1.0f) * static_cast<float>(edge - 1),
                            clamp(input.g, 0.0f, 1.0f) * static_cast<float>(edge - 1),
                            clamp(input.b, 0.0f, 1.0f) * static_cast<float>(edge - 1),
                        };
                        const size_t lower[] = {
                            static_cast<size_t>(std::floor(colors[0])),
                            static_cast<size_t>(std::floor(colors[1])),
                            static_cast<size_t>(std::floor(colors[2])),
                        };
                        const float fraction[] = {
                            colors[0] - static_cast<float>(lower[0]),
                            colors[1] - static_cast<float>(lower[1]),
                            colors[2] - static_cast<float>(lower[2]),
                        };
                        const float red_weight_f = red_bit ? fraction[0] : 1.0f - fraction[0];
                        const float green_weight_f = green_bit ? fraction[1] : 1.0f - fraction[1];
                        const float blue_weight_f = blue_bit ? fraction[2] : 1.0f - fraction[2];
                        const double red_weight = static_cast<double>(red_weight_f);
                        const double green_weight = static_cast<double>(green_weight_f);
                        const double blue_weight = static_cast<double>(blue_weight_f);
                        const size_t red_index = std::min(lower[0] + red_bit, edge_size - 1);
                        const size_t green_index = std::min(lower[1] + green_bit, edge_size - 1);
                        const size_t blue_index = std::min(lower[2] + blue_bit, edge_size - 1);
                        const size_t cube_index =
                            (red_index * edge_size + green_index) * edge_size + blue_index;
                        double weight = static_cast<double>(weight_confidence) * red_weight;
                        weight = weight * green_weight;
                        weight = weight * blue_weight;
                        const float clipped_depth = clamp(input_depth, depth_min, depth_max);
                        total[cube_index] = total[cube_index] +
                            weight * static_cast<double>(clipped_depth);
                        mass[cube_index] = mass[cube_index] + weight;
                    }
                }
            }
        }
        return IM_STATUS_OK;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_basic_rgb8(
    uint32_t width, uint32_t height, const uint8_t *source,
    size_t source_samples, const im_basic_params *params,
    uint8_t *destination, size_t destination_samples) {
    try {
        return basic_rgb_run(width, height, source, source_samples, params,
                             destination, destination_samples);
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_basic_rgb16(
    uint32_t width, uint32_t height, const uint16_t *source,
    size_t source_samples, const im_basic_params *params,
    uint16_t *destination, size_t destination_samples) {
    try {
        return basic_rgb_run(width, height, source, source_samples, params,
                             destination, destination_samples);
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_exposure_float(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    float gain, float *destination, size_t destination_values) {
    try {
        size_t pixels = 0, rgb_values = 0;
        if (!source || !destination || !image_extent(width, height, &pixels, &rgb_values) ||
            source_values != rgb_values || destination_values != rgb_values ||
            !std::isfinite(gain) || gain < 1.0f || gain > 4.0f) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        size_t source_bytes = 0, destination_bytes = 0;
        if (!byte_length<float>(source_values, &source_bytes) ||
            !byte_length<float>(destination_values, &destination_bytes) ||
            ranges_overlap(source, source_bytes, destination, destination_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        for (size_t value = 0; value < rgb_values; ++value) {
            if (!std::isfinite(source[value]) || source[value] < 0.0f || source[value] > 1.0f) {
                return IM_STATUS_INVALID_ARGUMENT;
            }
        }

        const bool completed = parallel_for(rgb_values, [&](size_t value) {
            destination[value] = source[value] * gain;
        });
        return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_rgb_peak(
    uint32_t width, uint32_t height, const float *source,
    size_t source_values, float *peak) {
    try {
        size_t pixels = 0, rgb_values = 0;
        if (!source || !peak || !image_extent(width, height, &pixels, &rgb_values) ||
            source_values != rgb_values) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        size_t source_bytes = 0, peak_bytes = 0;
        if (!byte_length<float>(source_values, &source_bytes) ||
            !byte_length<float>(1, &peak_bytes) ||
            ranges_overlap(source, source_bytes, peak, peak_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        std::atomic<bool> invalid{false};
        std::atomic<float> maximum{-std::numeric_limits<float>::infinity()};
        const bool completed = parallel_for(pixels, [&](size_t pixel) {
            const RGB input = read_rgb(source, pixel);
            if (!normalized_rgb(input)) {
                invalid.store(true, std::memory_order_relaxed);
                return;
            }
            const float pixel_peak = std::fmax(input.r, std::fmax(input.g, input.b));
            float observed = maximum.load(std::memory_order_relaxed);
            while (pixel_peak > observed ||
                   (pixel_peak == observed && pixel_peak == 0.0f &&
                    std::signbit(observed) && !std::signbit(pixel_peak))) {
                if (maximum.compare_exchange_weak(observed, pixel_peak,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
                    break;
                }
            }
        });
        if (!completed) return IM_STATUS_RUNTIME_ERROR;
        if (invalid.load(std::memory_order_relaxed)) return IM_STATUS_INVALID_ARGUMENT;
        *peak = maximum.load(std::memory_order_relaxed);
        return IM_STATUS_OK;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_refine_transmission(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *slope, size_t slope_count, const float *intercept,
    size_t intercept_count, float depth_min, float depth_max,
    float *destination, size_t destination_count) {
    try {
        size_t pixels = 0, rgb_values = 0;
        if (!source || !slope || !intercept || !destination ||
            !image_extent(width, height, &pixels, &rgb_values) ||
            source_values != rgb_values || slope_count != pixels ||
            intercept_count != pixels || destination_count != pixels ||
            !std::isfinite(depth_min) || !std::isfinite(depth_max) ||
            depth_min < 0.0f || depth_max > 10.0f || depth_min > depth_max) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        size_t source_bytes = 0, slope_bytes = 0, intercept_bytes = 0;
        size_t destination_bytes = 0;
        if (!byte_length<float>(source_values, &source_bytes) ||
            !byte_length<float>(slope_count, &slope_bytes) ||
            !byte_length<float>(intercept_count, &intercept_bytes) ||
            !byte_length<float>(destination_count, &destination_bytes) ||
            ranges_overlap(source, source_bytes, destination, destination_bytes) ||
            ranges_overlap(slope, slope_bytes, destination, destination_bytes) ||
            ranges_overlap(intercept, intercept_bytes, destination, destination_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        for (size_t value = 0; value < rgb_values; ++value) {
            if (!std::isfinite(source[value]) || source[value] < 0.0f || source[value] > 1.0f) {
                return IM_STATUS_INVALID_ARGUMENT;
            }
        }
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            if (!std::isfinite(slope[pixel]) || !std::isfinite(intercept[pixel])) {
                return IM_STATUS_INVALID_ARGUMENT;
            }
        }

        const bool completed = parallel_for(pixels, [&](size_t pixel) {
            const float luma = luminance(read_rgb(source, pixel));
            const float refined = clamp(slope[pixel] * luma + intercept[pixel],
                                        depth_min, depth_max);
            destination[pixel] = std::exp(-refined);
        });
        return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_relief_transmission(
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *relief, size_t relief_count, const float *airlight_rgb,
    float base, float knee, float initial_t, float *destination,
    size_t destination_count) {
    try {
        size_t pixels = 0, rgb_values = 0;
        if (!source || !relief || !airlight_rgb || !destination ||
            !image_extent(width, height, &pixels, &rgb_values) ||
            source_values != rgb_values || relief_count != pixels ||
            destination_count != pixels || !std::isfinite(base) ||
            !std::isfinite(knee) || !std::isfinite(initial_t) ||
            base < 0.0f || base > 1.0f || knee < 0.0f || knee > 0.1f ||
            initial_t < 0.0f || initial_t > 1.0f) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        size_t source_bytes = 0, relief_bytes = 0, airlight_bytes = 0;
        size_t destination_bytes = 0;
        if (!byte_length<float>(source_values, &source_bytes) ||
            !byte_length<float>(relief_count, &relief_bytes) ||
            !byte_length<float>(3, &airlight_bytes) ||
            !byte_length<float>(destination_count, &destination_bytes) ||
            ranges_overlap(source, source_bytes, destination, destination_bytes) ||
            ranges_overlap(relief, relief_bytes, destination, destination_bytes) ||
            ranges_overlap(airlight_rgb, airlight_bytes, destination, destination_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        const RGB air = read_rgb(airlight_rgb, 0);
        if (!normalized_rgb(air)) return IM_STATUS_INVALID_ARGUMENT;
        for (size_t value = 0; value < rgb_values; ++value) {
            if (!std::isfinite(source[value]) || source[value] < 0.0f || source[value] > 1.0f) {
                return IM_STATUS_INVALID_ARGUMENT;
            }
        }
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            if (!std::isfinite(relief[pixel]) || relief[pixel] < 0.0f || relief[pixel] > 1.0f) {
                return IM_STATUS_INVALID_ARGUMENT;
            }
        }

        const float air_norm = std::sqrt((air.r * air.r + air.g * air.g) + air.b * air.b);
        const float air_denominator = std::max(air_norm, 1e-6f);
        const bool completed = parallel_for(pixels, [&](size_t pixel) {
            const RGB input = read_rgb(source, pixel);
            const RGB deficit{air.r - input.r, air.g - input.g, air.b - input.b};
            const float radius_norm = std::sqrt(
                (deficit.r * deficit.r + deficit.g * deficit.g) + deficit.b * deficit.b);
            const float radius = radius_norm / air_denominator;
            const float below_r = deficit.r / std::max(air.r, 1e-6f);
            const float below_g = deficit.g / std::max(air.g, 1e-6f);
            const float below_b = deficit.b / std::max(air.b, 1e-6f);
            const float below_air = std::min(below_r, std::min(below_g, below_b));
            const float support = smoothstep(0.10f, 0.25f, radius) *
                                  smoothstep(0.0f, 0.05f, below_air);
            const float target = base - knee + (1.0f - base) * 0.5f *
                                 relief[pixel] * support;
            destination[pixel] = std::max(initial_t, target);
        });
        return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}
