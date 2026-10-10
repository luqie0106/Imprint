#include "imprint_renderer.h"
#include "warp_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace {

constexpr size_t kParallelThreshold = 65536;
constexpr unsigned int kMaxWorkers = 8;
constexpr size_t kMaxRgbValues = static_cast<size_t>(1ull << 29);
constexpr float kMaxWarpCoefficient = 1.0e6f;

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
    const uintptr_t address_max = std::numeric_limits<uintptr_t>::max();
    if (left_bytes > address_max - left_start || right_bytes > address_max - right_start) return true;
    const uintptr_t left_end = left_start + left_bytes;
    const uintptr_t right_end = right_start + right_bytes;
    return left_start < right_end && right_start < left_end;
}

template <typename Function>
bool parallel_for(size_t count, Function &&function) {
    if (count < kParallelThreshold) {
        for (size_t index = 0; index < count; ++index) function(index);
        return true;
    }

    unsigned int workers = std::thread::hardware_concurrency();
    if (workers == 0) workers = 1;
    workers = std::min(workers, kMaxWorkers);
    if (workers < 2) {
        for (size_t index = 0; index < count; ++index) function(index);
        return true;
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
        for (std::thread &worker : threads) if (worker.joinable()) worker.join();
        return false;
    }
    for (std::thread &worker : threads) if (worker.joinable()) worker.join();
    return !failed.load(std::memory_order_relaxed);
}

uint16_t round_u16_ties_even(float value) {
    const float lower_value = std::floor(value);
    const float fraction = value - lower_value;
    uint32_t lower = static_cast<uint32_t>(lower_value);
    if (fraction > 0.5f || (fraction == 0.5f && (lower & 1u) != 0u)) ++lower;
    return static_cast<uint16_t>(std::min<uint32_t>(lower, 65535u));
}

float clamp_coordinate(float coordinate, uint32_t extent) {
    return std::min(std::max(coordinate, 0.0f), static_cast<float>(extent - 1));
}

uint16_t sample_channel(const uint16_t *source, uint32_t width, uint32_t height,
                        uint32_t channel, float source_x, float source_y) {
    const float x = clamp_coordinate(source_x, width);
    const float y = clamp_coordinate(source_y, height);
    const uint32_t x0 = static_cast<uint32_t>(std::floor(x));
    const uint32_t y0 = static_cast<uint32_t>(std::floor(y));
    const uint32_t x1 = std::min(x0 + 1u, width - 1u);
    const uint32_t y1 = std::min(y0 + 1u, height - 1u);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const size_t at00 = (static_cast<size_t>(y0) * width + x0) * 3u + channel;
    const size_t at01 = (static_cast<size_t>(y0) * width + x1) * 3u + channel;
    const size_t at10 = (static_cast<size_t>(y1) * width + x0) * 3u + channel;
    const size_t at11 = (static_cast<size_t>(y1) * width + x1) * 3u + channel;
    const float top = static_cast<float>(source[at00]) * (1.0f - fx) +
                      static_cast<float>(source[at01]) * fx;
    const float bottom = static_cast<float>(source[at10]) * (1.0f - fx) +
                         static_cast<float>(source[at11]) * fx;
    const float value = top * (1.0f - fy) + bottom * fy;
    return round_u16_ties_even(std::min(std::max(value, 0.0f), 65535.0f));
}

void source_position(uint32_t x, uint32_t y, uint32_t channel,
                     uint32_t width, uint32_t height, const float *constants,
                     float center_x, float center_y, float pixel_scale_v,
                     float radius, float *source_x, float *source_y) {
    const size_t coefficient = static_cast<size_t>(channel) * 6u;
    const float dx = static_cast<float>(x) - center_x;
    const float dy = static_cast<float>(y) - center_y;
    const float scaled_dy = dy * pixel_scale_v;
    const float norm_x = dx / radius;
    const float norm_y = scaled_dy / radius;
    const float norm_x_squared = norm_x * norm_x;
    const float norm_y_squared = norm_y * norm_y;
    const float r2 = std::min(norm_x_squared + norm_y_squared, 1.0f);
    const float radial = constants[coefficient] + r2 * (constants[coefficient + 1] +
        r2 * (constants[coefficient + 2] + r2 * constants[coefficient + 3]));
    const float tangential_x = constants[coefficient + 5] *
        (r2 + 2.0f * norm_x_squared) +
        2.0f * constants[coefficient + 4] * norm_x * norm_y;
    const float tangential_y = constants[coefficient + 4] *
        (r2 + 2.0f * norm_y_squared) +
        2.0f * constants[coefficient + 5] * norm_x * norm_y;
    *source_x = center_x + dx * radial + radius * tangential_x;
    *source_y = center_y + dy * radial + radius * tangential_y / pixel_scale_v;
    (void)width;
    (void)height;
}

} // namespace

namespace imprint {

bool validate_warp_rectilinear_rgb16(uint32_t width, uint32_t height,
                                     const uint16_t *source, size_t source_values,
                                     const float *constants, size_t constant_values,
                                     uint16_t *destination, size_t destination_values,
                                     size_t *pixel_count, float *radius) {
    if (!source || !constants || !destination || !width || !height ||
        width > 65535 || height > 65535 || constant_values != 21 ||
        !pixel_count || !radius) return false;

    const uint64_t pixels64 = static_cast<uint64_t>(width) * height;
    const uint64_t values64 = pixels64 * 3u;
    if (values64 > kMaxRgbValues || values64 > std::numeric_limits<size_t>::max() ||
        source_values != static_cast<size_t>(values64) ||
        destination_values != static_cast<size_t>(values64)) return false;

    for (size_t index = 0; index < constant_values; ++index) {
        if (!std::isfinite(constants[index])) return false;
    }
    for (size_t index = 0; index < 18; ++index) {
        if (std::abs(constants[index]) > kMaxWarpCoefficient) return false;
    }
    const float center_x = constants[18];
    const float center_y = constants[19];
    const float pixel_scale_v = constants[20];
    if (center_x < 0.0f || center_x > 1.0f || center_y < 0.0f || center_y > 1.0f ||
        pixel_scale_v < 1.0e-4f || pixel_scale_v > 1.0e4f) return false;

    size_t source_bytes = 0, destination_bytes = 0, constants_bytes = 0;
    if (!byte_length<uint16_t>(source_values, &source_bytes) ||
        !byte_length<uint16_t>(destination_values, &destination_bytes) ||
        !byte_length<float>(constant_values, &constants_bytes) ||
        ranges_overlap(source, source_bytes, destination, destination_bytes) ||
        ranges_overlap(constants, constants_bytes, destination, destination_bytes)) return false;

    const float cx = static_cast<float>(width) * center_x;
    const float cy = static_cast<float>(height) * center_y;
    const float left = 0.0f - cx;
    const float right = static_cast<float>(width) - cx;
    const float scaled_height = std::floor(static_cast<float>(height) * pixel_scale_v + 0.5f);
    const float scaled_center_y = scaled_height * center_y;
    const float top = 0.0f - scaled_center_y;
    const float bottom = scaled_height - scaled_center_y;
    const float d00 = std::sqrt(left * left + top * top);
    const float d10 = std::sqrt(right * right + top * top);
    const float d01 = std::sqrt(left * left + bottom * bottom);
    const float d11 = std::sqrt(right * right + bottom * bottom);
    *radius = std::max(std::max(d00, d10), std::max(d01, d11));
    if (!std::isfinite(*radius) || *radius <= 0.0f) return false;
    *pixel_count = static_cast<size_t>(pixels64);
    return true;
}

} // namespace imprint

extern "C" IMPRINT_API im_status im_native_warp_rectilinear_rgb16(
    uint32_t width, uint32_t height, const uint16_t *source, size_t source_values,
    const float *constants, size_t constant_values,
    uint16_t *destination, size_t destination_values) {
    size_t pixels = 0;
    float radius = 0.0f;
    if (!imprint::validate_warp_rectilinear_rgb16(width, height, source, source_values,
                                                  constants, constant_values, destination,
                                                  destination_values, &pixels, &radius)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    const float center_x = static_cast<float>(width) * constants[18];
    const float center_y = static_cast<float>(height) * constants[19];
    const float pixel_scale_v = constants[20];
    try {
        const bool completed = parallel_for(pixels, [&](size_t pixel) {
            const uint32_t x = static_cast<uint32_t>(pixel % width);
            const uint32_t y = static_cast<uint32_t>(pixel / width);
            const size_t output_at = pixel * 3u;
            for (uint32_t channel = 0; channel < 3; ++channel) {
                float source_x = 0.0f, source_y = 0.0f;
                source_position(x, y, channel, width, height, constants, center_x,
                                center_y, pixel_scale_v, radius, &source_x, &source_y);
                destination[output_at + channel] = sample_channel(
                    source, width, height, channel, source_x, source_y);
            }
        });
        return completed ? IM_STATUS_OK : IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}
