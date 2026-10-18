#include "acr_source_pyramid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>

namespace acr_source_pyramid {
namespace {

constexpr std::size_t kMaximumPixels = 100'000'000;
constexpr float kEpsilon = 0x1p-8f;
constexpr float kBias = 0x1p-4f;
constexpr float kForwardScale = 0x1.107fe0p+0f;  // bits 0x3f883ff0
constexpr float kInverseScale = 0x1.e0ffc0p-1f;  // bits 0x3f707fe0

bool checked_multiply(std::size_t left,
                      std::size_t right,
                      std::size_t* product) noexcept {
    if (product == nullptr || (right != 0 && left >
                                                   std::numeric_limits<std::size_t>::max() /
                                                       right)) {
        return false;
    }
    *product = left * right;
    return true;
}

bool address_span_is_valid(const void* pointer, std::size_t bytes) noexcept {
    if (pointer == nullptr || bytes == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return bytes <= std::numeric_limits<std::uintptr_t>::max() - begin;
}

float add32(float left, float right) noexcept {
    volatile float result = left + right;
    return result;
}

float subtract32(float left, float right) noexcept {
    volatile float result = left - right;
    return result;
}

float multiply32(float left, float right) noexcept {
    volatile float result = left * right;
    return result;
}

bool forward_bracket(float value, float* output) noexcept {
    float root_input = 0.0f;
    float scale_sign = 1.0f;
    if (value < 0.0f) {
        root_input = subtract32(kEpsilon, value);
        scale_sign = -1.0f;
    } else {
        // This branch intentionally includes negative zero, matching the
        // recovered scalar comparison and producing positive zero at -0.
        root_input = add32(value, kEpsilon);
    }
    if (!std::isfinite(root_input) || root_input < 0.0f) return false;

    const float root = ::sqrtf(root_input);
    if (!std::isfinite(root)) return false;
    const float centered = subtract32(root, kBias);
    if (!std::isfinite(centered)) return false;
    const float signed_centered = scale_sign < 0.0f ? -centered : centered;
    const float mapped = multiply32(signed_centered, kForwardScale);
    if (!std::isfinite(mapped)) return false;
    *output = mapped;
    return true;
}

bool inverse_bracket(float value, float* output) noexcept {
    float result = 0.0f;
    if (value < 0.0f) {
        const float q = std::fma(-value, kInverseScale, kBias);
        if (!std::isfinite(q)) return false;
        // Scalar FNMADD(q,q,-epsilon) semantics. Keeping this signed form
        // differs from negating FMA(q,q,-epsilon) at exact cancellation.
        result = std::fma(-q, q, kEpsilon);
    } else {
        const float q = std::fma(value, kInverseScale, kBias);
        if (!std::isfinite(q)) return false;
        result = std::fma(q, q, -kEpsilon);
    }
    if (!std::isfinite(result)) return false;
    *output = result;
    return true;
}

std::size_t rounded_level_dimension(std::size_t original,
                                    std::size_t level) noexcept {
    const std::size_t offset = std::size_t{1} << level;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (original > maximum - offset) return 0;
    return std::max<std::size_t>(1, (original + offset) >> (level + 1));
}

std::size_t clamped_coordinate(std::size_t coordinate,
                              std::size_t extent) noexcept {
    return coordinate < extent ? coordinate : extent - 1;
}

bool sample_average(const float* source,
                    std::size_t source_width,
                    std::size_t source_height,
                    std::size_t origin_x,
                    std::size_t origin_y,
                    std::size_t output_x,
                    std::size_t output_y,
                    std::size_t channel,
                    bool nonlinear_enabled,
                    float* output) noexcept {
    const std::size_t x0 = clamped_coordinate(origin_x + 2 * output_x,
                                               source_width);
    const std::size_t x1 = clamped_coordinate(origin_x + 2 * output_x + 1,
                                               source_width);
    const std::size_t y0 = clamped_coordinate(origin_y + 2 * output_y,
                                               source_height);
    const std::size_t y1 = clamped_coordinate(origin_y + 2 * output_y + 1,
                                               source_height);
    const std::size_t row0 = y0 * source_width * 3;
    const std::size_t row1 = y1 * source_width * 3;

    float a = source[row0 + x0 * 3 + channel];
    float b = source[row0 + x1 * 3 + channel];
    float c = source[row1 + x0 * 3 + channel];
    float d = source[row1 + x1 * 3 + channel];
    if (nonlinear_enabled) {
        if (!forward_bracket(a, &a) || !forward_bracket(b, &b) ||
            !forward_bracket(c, &c) || !forward_bracket(d, &d)) {
            return false;
        }
    }

    // Match the recovered scalar kernel: three ordered binary32 additions,
    // with no accumulator initialized from +0, then one binary32 multiply.
    const float t1 = add32(a, b);
    if (!std::isfinite(t1)) return false;
    const float t2 = add32(t1, c);
    if (!std::isfinite(t2)) return false;
    const float t3 = add32(t2, d);
    if (!std::isfinite(t3)) return false;
    const float average = multiply32(t3, 0.25f);
    if (!std::isfinite(average)) return false;

    if (nonlinear_enabled) return inverse_bracket(average, output);
    *output = average;
    return true;
}

}  // namespace

Status build_float_rgb(const float* packed_rgb,
                       std::size_t image_width,
                       std::size_t image_height,
                       acr_flare::Crop crop,
                       bool nonlinear_enabled,
                       std::vector<Level>* out) noexcept {
    if (packed_rgb == nullptr || out == nullptr || image_width == 0 ||
        image_height == 0 || crop.width == 0 || crop.height == 0 ||
        reinterpret_cast<std::uintptr_t>(packed_rgb) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(out) % alignof(std::vector<Level>) != 0) {
        return Status::invalid_argument;
    }

    std::size_t image_pixels = 0;
    if (!checked_multiply(image_width, image_height, &image_pixels)) {
        return Status::size_overflow;
    }
    if (image_pixels > kMaximumPixels) return Status::too_many_pixels;
    if (crop.x > image_width || crop.y > image_height ||
        crop.width > image_width - crop.x ||
        crop.height > image_height - crop.y) {
        return Status::invalid_argument;
    }

    std::size_t input_samples = 0;
    std::size_t input_bytes = 0;
    if (!checked_multiply(image_pixels, 3, &input_samples) ||
        !checked_multiply(input_samples, sizeof(float), &input_bytes) ||
        !address_span_is_valid(packed_rgb, input_bytes)) {
        return Status::size_overflow;
    }

    for (std::size_t index = 0; index < input_samples; ++index) {
        if (!std::isfinite(packed_rgb[index])) return Status::non_finite_input;
    }

    const bool build_pyramid = crop.width > 64 || crop.height > 64;
    try {
        std::vector<Level> candidate;
        if (build_pyramid) candidate.reserve(5);

        for (std::size_t level = 0; level < (build_pyramid ? 5u : 0u); ++level) {
            const std::size_t destination_width =
                rounded_level_dimension(crop.width, level);
            const std::size_t destination_height =
                rounded_level_dimension(crop.height, level);
            if (destination_width == 0 || destination_height == 0) {
                return Status::size_overflow;
            }

            std::size_t destination_pixels = 0;
            std::size_t destination_samples = 0;
            std::size_t destination_bytes = 0;
            if (!checked_multiply(destination_width, destination_height,
                                  &destination_pixels) ||
                !checked_multiply(destination_pixels, 3, &destination_samples) ||
                !checked_multiply(destination_samples, sizeof(float),
                                  &destination_bytes)) {
                return Status::size_overflow;
            }
            (void)destination_bytes;

            candidate.emplace_back();
            Level& destination = candidate.back();
            destination.width = destination_width;
            destination.height = destination_height;
            destination.pixels.resize(destination_samples);

            const bool first_level = level == 0;
            const float* source = first_level
                                      ? packed_rgb
                                      : candidate[level - 1].pixels.data();
            const std::size_t source_width = first_level
                                                 ? image_width
                                                 : candidate[level - 1].width;
            const std::size_t source_height = first_level
                                                  ? image_height
                                                  : candidate[level - 1].height;
            const std::size_t origin_x = first_level ? crop.x : 0;
            const std::size_t origin_y = first_level ? crop.y : 0;
            std::size_t output_index = 0;
            for (std::size_t y = 0; y < destination_height; ++y) {
                for (std::size_t x = 0; x < destination_width; ++x) {
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        float value = 0.0f;
                        if (!sample_average(source,
                                            source_width,
                                            source_height,
                                            origin_x,
                                            origin_y,
                                            x,
                                            y,
                                            channel,
                                            nonlinear_enabled,
                                            &value)) {
                            return Status::non_finite_intermediate;
                        }
                        destination.pixels[output_index++] = value;
                    }
                }
            }
        }

        out->swap(candidate);
        return Status::ok;
    } catch (const std::bad_alloc&) {
        return Status::allocation_failure;
    } catch (...) {
        return Status::allocation_failure;
    }
}

}  // namespace acr_source_pyramid
