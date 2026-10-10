#include "imprint_renderer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace {

constexpr size_t kParallelThreshold = 65536;
constexpr unsigned int kMaxWorkers = 8;
constexpr size_t kMaxRgbValues = static_cast<size_t>(1ull << 29);
constexpr float kLumaR = 0.2126f;
constexpr float kLumaG = 0.7152f;
constexpr float kLumaB = 0.0722f;
constexpr float kFloatEpsilon = std::numeric_limits<float>::epsilon();

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
        try {
            for (size_t index = 0; index < std::min(count, chunk); ++index) function(index);
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

template <typename T>
bool byte_length(size_t count, size_t *bytes) {
    if (!bytes || count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
    *bytes = count * sizeof(T);
    return true;
}

bool ranges_overlap(const void *left, size_t left_bytes,
                    const void *right, size_t right_bytes) {
    const uintptr_t left_start = reinterpret_cast<uintptr_t>(left);
    const uintptr_t right_start = reinterpret_cast<uintptr_t>(right);
    const uintptr_t max_address = std::numeric_limits<uintptr_t>::max();
    if (left_bytes > max_address - left_start || right_bytes > max_address - right_start) {
        return true;
    }
    const uintptr_t left_end = left_start + left_bytes;
    const uintptr_t right_end = right_start + right_bytes;
    return left_start < right_end && right_start < left_end;
}

bool image_extent(uint32_t width, uint32_t height, size_t *pixels, size_t *rgb_values) {
    if (width == 0 || height == 0 || width > 65535 || height > 65535 ||
        !pixels || !rgb_values) return false;
    const uint64_t pixel_count = static_cast<uint64_t>(width) * height;
    if (pixel_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 3) ||
        pixel_count > kMaxRgbValues / 3) return false;
    *pixels = static_cast<size_t>(pixel_count);
    *rgb_values = *pixels * 3;
    return true;
}

float clamp_float(float value, float low, float high) {
    return std::min(high, std::max(low, value));
}

double interp_curve(double x, const float *curve, size_t points) {
    const double first_x = static_cast<double>(curve[0]);
    if (x <= first_x) return static_cast<double>(curve[1]);
    const double last_x = static_cast<double>(curve[(points - 1) * 2]);
    if (x >= last_x) return static_cast<double>(curve[(points - 1) * 2 + 1]);

    size_t low = 0;
    size_t high = points;
    while (low < high) {
        const size_t middle = low + (high - low) / 2;
        if (static_cast<double>(curve[middle * 2]) <= x) low = middle + 1;
        else high = middle;
    }
    const size_t right = low;
    const size_t left = right - 1;
    const double x0 = static_cast<double>(curve[left * 2]);
    const double y0 = static_cast<double>(curve[left * 2 + 1]);
    const double x1 = static_cast<double>(curve[right * 2]);
    const double y1 = static_cast<double>(curve[right * 2 + 1]);
    const double slope = (y1 - y0) / (x1 - x0);
    return slope * (x - x0) + y0;
}

float encode_srgb_float(float value) {
    value = std::max(value, 0.0f);
    if (value <= 0.0031308f) return value * 12.92f;
    return 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

float decode_srgb_float(float value) {
    if (value <= 0.04045f) return value / 12.92f;
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float remainder_360(float value) {
    float result = std::fmod(value, 360.0f);
    if (result < 0.0f) result += 360.0f;
    return result;
}

struct Hsv {
    float h;
    float s;
    float v;
};

Hsv rgb_to_hsv_cv(float r, float g, float b) {
    const float value = std::max(r, std::max(g, b));
    const float minimum = std::min(r, std::min(g, b));
    const float difference = value - minimum;
    const float saturation = difference / (std::abs(value) + kFloatEpsilon);
    float hue = 0.0f;
    if (difference > kFloatEpsilon) {
        if (value == r) {
            hue = g - b;
        } else if (value == g) {
            hue = (b - r) + 2.0f * difference;
        } else {
            hue = (r - g) + 4.0f * difference;
        }
        hue = hue * (60.0f / difference);
        if (hue < 0.0f) hue += 360.0f;
    }
    return {hue, saturation, value};
}

void hsv_to_rgb_cv(float h, float s, float v, float *r, float *g, float *b) {
    float sector_position = h * (1.0f / 60.0f);
    if (sector_position < 0.0f) sector_position += 6.0f;
    else if (sector_position >= 6.0f) sector_position -= 6.0f;
    const int sector = static_cast<int>(std::floor(sector_position));
    const float fraction = sector_position - static_cast<float>(sector);
    const float p = v * (1.0f - s);
    const float q = v * (1.0f - s * fraction);
    const float t = v * (1.0f - s * (1.0f - fraction));
    switch (sector) {
        case 0: *r = v; *g = t; *b = p; break;
        case 1: *r = q; *g = v; *b = p; break;
        case 2: *r = p; *g = v; *b = t; break;
        case 3: *r = p; *g = q; *b = v; break;
        case 4: *r = t; *g = p; *b = v; break;
        default: *r = v; *g = p; *b = q; break;
    }
}

float look_sample(const float *table, uint32_t hue_count, uint32_t saturation_count,
                  uint32_t value_index, uint32_t hue_index,
                  uint32_t saturation_index, uint32_t channel) {
    const size_t at = (((static_cast<size_t>(value_index) * hue_count + hue_index) *
                        saturation_count + saturation_index) * 3) + channel;
    return table[at];
}

void apply_look(float *r, float *g, float *b, const float *table,
                uint32_t hue_count, uint32_t saturation_count,
                uint32_t value_count, uint32_t encoding) {
    Hsv hsv = rgb_to_hsv_cv(*r, *g, *b);
    const float encoded_value = encoding == 1 ? encode_srgb_float(hsv.v) : hsv.v;
    const float hue_coord = remainder_360(hsv.h) *
        (static_cast<float>(hue_count) / 360.0f);
    const float saturation_coord = clamp_float(hsv.s, 0.0f, 1.0f) *
        static_cast<float>(saturation_count - 1);
    const float value_coord = clamp_float(encoded_value, 0.0f, 1.0f) *
        static_cast<float>(value_count - 1);

    const uint32_t h0 = static_cast<uint32_t>(std::floor(hue_coord));
    const uint32_t h1 = (h0 + 1u) % hue_count;
    const uint32_t s_floor = static_cast<uint32_t>(std::floor(saturation_coord));
    const uint32_t v_floor = static_cast<uint32_t>(std::floor(value_coord));
    const uint32_t s0 = std::min(s_floor, saturation_count - 2u);
    const uint32_t v0 = std::min(v_floor, value_count - 2u);
    // NumPy's mixed float32/int32 subtraction promotes the coordinates to
    // float64 at this point; each weighted table term is rounded back to f32
    // before it is added to the float32 accumulator.
    const double hf = static_cast<double>(hue_coord) - static_cast<double>(h0);
    const double sf = static_cast<double>(saturation_coord) - static_cast<double>(s0);
    const double vf = static_cast<double>(value_coord) - static_cast<double>(v0);
    float mods[3] = {0.0f, 0.0f, 0.0f};
    for (uint32_t dh = 0; dh < 2; ++dh) {
        const uint32_t hi = dh ? h1 : h0;
        const double hw = dh ? hf : (1.0 - hf);
        for (uint32_t ds = 0; ds < 2; ++ds) {
            const uint32_t si = s0 + ds;
            const double sw = ds ? sf : (1.0 - sf);
            for (uint32_t dv = 0; dv < 2; ++dv) {
                const uint32_t vi = v0 + dv;
                const double vw = dv ? vf : (1.0 - vf);
                const double weight = hw * sw * vw;
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    const float term = static_cast<float>(
                        weight * static_cast<double>(look_sample(
                            table, hue_count, saturation_count, vi, hi, si, channel)));
                    mods[channel] = mods[channel] + term;
                }
            }
        }
    }

    hsv.h = remainder_360(hsv.h + mods[0]);
    hsv.s = clamp_float(hsv.s * mods[1], 0.0f, 1.0f);
    const float new_value = clamp_float(encoded_value * mods[2], 0.0f, 1.0f);
    hsv.v = encoding == 1 ? decode_srgb_float(new_value) : new_value;
    hsv_to_rgb_cv(hsv.h, hsv.s, hsv.v, r, g, b);
}

double encode_srgb_double(double value) {
    value = std::max(value, 0.0);
    if (value <= 0.0031308) return value * 12.92;
    return 1.055 * std::pow(value, 1.0 / 2.4) - 0.055;
}

uint8_t quantize_rgb8(double value) {
    const double scaled = encode_srgb_double(value) * 255.0;
    if (!(scaled > 0.0)) return 0;
    if (scaled >= 255.0) return 255;
    const double rounded = std::nearbyint(scaled);
    return static_cast<uint8_t>(std::min(255.0, std::max(0.0, rounded)));
}

bool valid_render_config(const float *constants, const float *look_table,
                         size_t look_values, uint32_t hue_count,
                         uint32_t saturation_count, uint32_t value_count,
                         uint32_t look_encoding, const float *tone_curve,
                         size_t tone_values) {
    if (hue_count < 1 || saturation_count < 2 || value_count < 2 ||
        hue_count > 1000000 || saturation_count > 1000000 || value_count > 1000000 ||
        look_encoding > 1 || tone_values > 4u * 1024u * 1024u) return false;
    const uint64_t entries = static_cast<uint64_t>(hue_count) * saturation_count * value_count;
    if (entries > 1000000 || entries * 3u != look_values) return false;
    for (size_t index = 0; index < 22; ++index) {
        if (!std::isfinite(constants[index])) return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
        if (constants[9 + channel] <= 0.0f || constants[9 + channel] > 1.0f) return false;
    }
    if (constants[12] < 0x1p-40f || constants[12] > 0x1p40f) return false;
    for (size_t index = 0; index < 9; ++index) {
        if (std::abs(constants[index]) > 1.0e6f ||
            std::abs(constants[13 + index]) > 1.0e6f) return false;
    }

    for (size_t entry = 0; entry < static_cast<size_t>(entries); ++entry) {
        const float hue_delta = look_table[entry * 3];
        const float saturation = look_table[entry * 3 + 1];
        const float value = look_table[entry * 3 + 2];
        if (!std::isfinite(hue_delta) || !std::isfinite(saturation) || !std::isfinite(value) ||
            hue_delta < -360.0f || hue_delta > 360.0f ||
            saturation < 0.0f || saturation > 16.0f || value < 0.0f || value > 16.0f) {
            return false;
        }
    }
    if (tone_values < 4 || (tone_values & 1u) != 0u) return false;
    const size_t points = tone_values / 2;
    if (std::abs(tone_curve[0]) > 1.0e-6f ||
        std::abs(tone_curve[(points - 1) * 2] - 1.0f) > 1.0e-6f ||
        std::abs(tone_curve[1]) > 1.0e-6f ||
        std::abs(tone_curve[(points - 1) * 2 + 1] - 1.0f) > 1.0e-6f) return false;
    for (size_t point = 0; point < points; ++point) {
        const float x = tone_curve[point * 2];
        const float y = tone_curve[point * 2 + 1];
        if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0f || x > 1.0f ||
            y < 0.0f || y > 1.0f) return false;
        if (point > 0) {
            const float previous_x = tone_curve[(point - 1) * 2];
            const float previous_y = tone_curve[(point - 1) * 2 + 1];
            if (x <= previous_x || y < previous_y - 1.0e-7f) return false;
        }
    }
    return true;
}

bool disjoint_from_destination(const void *destination, size_t destination_bytes,
                               const void *source, size_t source_bytes) {
    return !ranges_overlap(destination, destination_bytes, source, source_bytes);
}

} // namespace

extern "C" IMPRINT_API im_status im_native_camera_profile_render_rgb16_to_rgb8(
    uint32_t width, uint32_t height, const uint16_t *camera_rgb,
    size_t source_values, const float *constants22, size_t constants_count,
    const float *look_table, size_t look_values, uint32_t hue_count,
    uint32_t saturation_count, uint32_t value_count, uint32_t look_encoding,
    const float *tone_curve, size_t tone_values, uint8_t *destination,
    size_t destination_values) {
    try {
        size_t pixels = 0, rgb_values = 0;
        if (!camera_rgb || !constants22 || !look_table || !tone_curve || !destination ||
            constants_count != 22 || !image_extent(width, height, &pixels, &rgb_values) ||
            source_values != rgb_values || destination_values != rgb_values ||
            !valid_render_config(constants22, look_table, look_values, hue_count,
                                 saturation_count, value_count, look_encoding,
                                 tone_curve, tone_values)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        size_t source_bytes = 0, destination_bytes = 0, constants_bytes = 0;
        size_t look_bytes = 0, tone_bytes = 0;
        if (!byte_length<uint16_t>(source_values, &source_bytes) ||
            !byte_length<uint8_t>(destination_values, &destination_bytes) ||
            !byte_length<float>(constants_count, &constants_bytes) ||
            !byte_length<float>(look_values, &look_bytes) ||
            !byte_length<float>(tone_values, &tone_bytes) ||
            !disjoint_from_destination(destination, destination_bytes, camera_rgb, source_bytes) ||
            !disjoint_from_destination(destination, destination_bytes, constants22, constants_bytes) ||
            !disjoint_from_destination(destination, destination_bytes, look_table, look_bytes) ||
            !disjoint_from_destination(destination, destination_bytes, tone_curve, tone_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }

        const float *camera_matrix = constants22;
        const float *white = constants22 + 9;
        const float gain = constants22[12];
        const float *projection = constants22 + 13;
        const size_t curve_points = tone_values / 2;
        const bool completed = parallel_for(pixels, [&](size_t pixel) {
            const size_t at = pixel * 3;
            const float camera_r = static_cast<float>(camera_rgb[at]) / 65535.0f;
            const float camera_g = static_cast<float>(camera_rgb[at + 1]) / 65535.0f;
            const float camera_b = static_cast<float>(camera_rgb[at + 2]) / 65535.0f;
            const float clipped_r = std::min(camera_r, white[0]);
            const float clipped_g = std::min(camera_g, white[1]);
            const float clipped_b = std::min(camera_b, white[2]);
            const float matrix_r = (clipped_r * camera_matrix[0] +
                                    clipped_g * camera_matrix[1]) + clipped_b * camera_matrix[2];
            const float matrix_g = (clipped_r * camera_matrix[3] +
                                    clipped_g * camera_matrix[4]) + clipped_b * camera_matrix[5];
            const float matrix_b = (clipped_r * camera_matrix[6] +
                                    clipped_g * camera_matrix[7]) + clipped_b * camera_matrix[8];
            float r = clamp_float(matrix_r, 0.0f, 1.0f) * gain;
            float g = clamp_float(matrix_g, 0.0f, 1.0f) * gain;
            float b = clamp_float(matrix_b, 0.0f, 1.0f) * gain;
            r = clamp_float(r, 0.0f, 1.0f);
            g = clamp_float(g, 0.0f, 1.0f);
            b = clamp_float(b, 0.0f, 1.0f);

            apply_look(&r, &g, &b, look_table, hue_count,
                       saturation_count, value_count, look_encoding);

            const float low = std::min(r, std::min(g, b));
            const float high = std::max(r, std::max(g, b));
            const float span = high - low;
            const double new_low = interp_curve(static_cast<double>(low), tone_curve, curve_points);
            const double new_high = interp_curve(static_cast<double>(high), tone_curve, curve_points);
            float positions[3] = {0.0f, 0.0f, 0.0f};
            if (span > 1.0e-12f) {
                positions[0] = (r - low) / span;
                positions[1] = (g - low) / span;
                positions[2] = (b - low) / span;
            }
            const double tone_r = new_low + (new_high - new_low) * static_cast<double>(positions[0]);
            const double tone_g = new_low + (new_high - new_low) * static_cast<double>(positions[1]);
            const double tone_b = new_low + (new_high - new_low) * static_cast<double>(positions[2]);
            const double projected_r = (tone_r * static_cast<double>(projection[0]) +
                                        tone_g * static_cast<double>(projection[1])) +
                                       tone_b * static_cast<double>(projection[2]);
            const double projected_g = (tone_r * static_cast<double>(projection[3]) +
                                        tone_g * static_cast<double>(projection[4])) +
                                       tone_b * static_cast<double>(projection[5]);
            const double projected_b = (tone_r * static_cast<double>(projection[6]) +
                                        tone_g * static_cast<double>(projection[7])) +
                                       tone_b * static_cast<double>(projection[8]);
            destination[at] = quantize_rgb8(projected_r);
            destination[at + 1] = quantize_rgb8(projected_g);
            destination[at + 2] = quantize_rgb8(projected_b);
        });
        return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

extern "C" IMPRINT_API im_status im_native_camera_profile_transfer_rgb16(
    uint32_t width, uint32_t height, const uint16_t *camera_rgb,
    size_t camera_values, const uint16_t *reference_rgb, size_t reference_values,
    const uint16_t *processed_rgb, size_t processed_values,
    const float *inverse_matrix9, size_t inverse_count,
    uint16_t *destination, size_t destination_values) {
    try {
        size_t pixels = 0, rgb_values = 0;
        if (!camera_rgb || !reference_rgb || !processed_rgb || !inverse_matrix9 ||
            !destination || !image_extent(width, height, &pixels, &rgb_values) ||
            camera_values != rgb_values || reference_values != rgb_values ||
            processed_values != rgb_values || destination_values != rgb_values ||
            inverse_count != 9) return IM_STATUS_INVALID_ARGUMENT;

        size_t image_bytes = 0, matrix_bytes = 0;
        if (!byte_length<uint16_t>(rgb_values, &image_bytes) ||
            !byte_length<float>(inverse_count, &matrix_bytes) ||
            !disjoint_from_destination(destination, image_bytes, camera_rgb, image_bytes) ||
            !disjoint_from_destination(destination, image_bytes, reference_rgb, image_bytes) ||
            !disjoint_from_destination(destination, image_bytes, processed_rgb, image_bytes) ||
            !disjoint_from_destination(destination, image_bytes, inverse_matrix9, matrix_bytes)) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        for (size_t index = 0; index < inverse_count; ++index) {
            if (!std::isfinite(inverse_matrix9[index]) ||
                std::abs(inverse_matrix9[index]) > 1.0e6f) {
                return IM_STATUS_INVALID_ARGUMENT;
            }
        }

        bool identical = true;
        for (size_t index = 0; index < rgb_values; ++index) {
            if (reference_rgb[index] != processed_rgb[index]) {
                identical = false;
                break;
            }
        }
        if (identical) {
            std::copy(camera_rgb, camera_rgb + rgb_values, destination);
            return IM_STATUS_OK;
        }

        const bool completed = parallel_for(pixels, [&](size_t pixel) {
            const size_t at = pixel * 3;
            const float pr = static_cast<float>(processed_rgb[at]);
            const float pg = static_cast<float>(processed_rgb[at + 1]);
            const float pb = static_cast<float>(processed_rgb[at + 2]);
            const float rr = static_cast<float>(reference_rgb[at]);
            const float rg = static_cast<float>(reference_rgb[at + 1]);
            const float rb = static_cast<float>(reference_rgb[at + 2]);
            const float processed_luma = std::max((pr * kLumaR + pg * kLumaG) + pb * kLumaB, 0.0f);
            const float reference_luma = std::max((rr * kLumaR + rg * kLumaG) + rb * kLumaB, 0.0f);
            float ratio = processed_luma / std::max(reference_luma, 1.0f);
            float residual_r = pr - rr * ratio;
            float residual_g = pg - rg * ratio;
            float residual_b = pb - rb * ratio;
            if (reference_luma < 1.0f) {
                residual_r = residual_g = residual_b = 0.0f;
                ratio = 0.0f;
            }
            const float camera_r = static_cast<float>(camera_rgb[at]) * ratio;
            const float camera_g = static_cast<float>(camera_rgb[at + 1]) * ratio;
            const float camera_b = static_cast<float>(camera_rgb[at + 2]) * ratio;
            const float transformed_r = (residual_r * inverse_matrix9[0] +
                                         residual_g * inverse_matrix9[3]) +
                                        residual_b * inverse_matrix9[6];
            const float transformed_g = (residual_r * inverse_matrix9[1] +
                                         residual_g * inverse_matrix9[4]) +
                                        residual_b * inverse_matrix9[7];
            const float transformed_b = (residual_r * inverse_matrix9[2] +
                                         residual_g * inverse_matrix9[5]) +
                                        residual_b * inverse_matrix9[8];
            const float out_r = camera_r + transformed_r;
            const float out_g = camera_g + transformed_g;
            const float out_b = camera_b + transformed_b;
            const double rounded_r = std::nearbyint(static_cast<double>(out_r));
            const double rounded_g = std::nearbyint(static_cast<double>(out_g));
            const double rounded_b = std::nearbyint(static_cast<double>(out_b));
            destination[at] = static_cast<uint16_t>(std::min(65535.0, std::max(0.0, rounded_r)));
            destination[at + 1] = static_cast<uint16_t>(std::min(65535.0, std::max(0.0, rounded_g)));
            destination[at + 2] = static_cast<uint16_t>(std::min(65535.0, std::max(0.0, rounded_b)));
        });
        return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}
