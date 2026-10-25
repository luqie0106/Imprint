#include "acr_gray.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumPixels = std::size_t{1} << 26;
constexpr double kCoefficientScale = 16384.0;
constexpr double kUint32Limit = 4294967296.0;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool make_range(const void *pointer, std::size_t byte_count,
                std::size_t alignment, AddressRange *range) noexcept {
    if (range == nullptr || alignment == 0) return false;
    if (byte_count == 0) {
        if (pointer == nullptr) {
            *range = {};
            return true;
        }
        const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
        if (begin % alignment != 0) return false;
        *range = {begin, begin};
        return true;
    }
    if (pointer == nullptr) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0 ||
        byte_count > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }
    *range = {begin, begin + byte_count};
    return true;
}

bool overlaps(const AddressRange &left, const AddressRange &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

GrayStatus quantize_coefficients(const double *coefficients,
                                 double quantized[3]) noexcept {
    double carry = 0.0;
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const double accumulated = carry + coefficients[channel];
        if (!std::isfinite(accumulated)) return GrayStatus::overflow;

        const double scaled = accumulated * kCoefficientScale;
        if (!std::isfinite(scaled)) return GrayStatus::overflow;
        const double bias = scaled > 0.0 ? 0.5 : -0.5;
        const double rounded = scaled + bias;
        if (!std::isfinite(rounded) || rounded <= -2147483649.0 ||
            rounded >= 2147483648.0) {
            return GrayStatus::overflow;
        }

        const auto integer = static_cast<std::int32_t>(rounded);
        const double value = static_cast<double>(integer) / kCoefficientScale;
        const double next_carry = accumulated - value;
        if (!std::isfinite(value) || !std::isfinite(next_carry))
            return GrayStatus::overflow;
        quantized[channel] = value;
        carry = next_carry;
    }
    return GrayStatus::ok;
}

bool prepared_weights_valid(const PreparedRgbToGray &prepared) noexcept {
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const float expected =
            static_cast<float>(prepared.lanes_rgb[channel]) / 16384.0f;
        if (!std::isfinite(prepared.weights_rgb[channel]) ||
            prepared.weights_rgb[channel] != expected) {
            return false;
        }
    }
    return true;
}

}  // namespace

GrayStatus prepare_rgb_to_gray(const double *coefficients_rgb,
                               PreparedRgbToGray *prepared) noexcept {
    AddressRange coefficient_range;
    AddressRange prepared_range;
    if (!make_range(coefficients_rgb, 3 * sizeof(double), alignof(double),
                    &coefficient_range) ||
        !make_range(prepared, sizeof(*prepared), alignof(PreparedRgbToGray),
                    &prepared_range) ||
        overlaps(coefficient_range, prepared_range)) {
        return GrayStatus::invalid_argument;
    }

    for (std::size_t channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(coefficients_rgb[channel]) ||
            coefficients_rgb[channel] < -120.0 ||
            coefficients_rgb[channel] > 120.0) {
            return GrayStatus::invalid_coefficients;
        }
    }

    double quantized[3]{};
    const GrayStatus quantize_status =
        quantize_coefficients(coefficients_rgb, quantized);
    if (quantize_status != GrayStatus::ok) return quantize_status;

    PreparedRgbToGray candidate{};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        double scaled = quantized[channel] * kCoefficientScale + 0.5;
        if (!std::isfinite(scaled)) return GrayStatus::overflow;
        if (scaled < 0.0) scaled = 0.0;
        if (scaled >= kUint32Limit) return GrayStatus::overflow;

        const auto code = static_cast<std::uint32_t>(scaled);
        const auto lane = static_cast<std::uint16_t>(code & 0xffffu);
        candidate.lanes_rgb[channel] = lane;
        candidate.weights_rgb[channel] = static_cast<float>(lane) / 16384.0f;
        if (!std::isfinite(candidate.weights_rgb[channel]))
            return GrayStatus::overflow;
    }

    *prepared = candidate;
    return GrayStatus::ok;
}

GrayStatus apply_rgb_to_gray(const float *input_rgb, float *output_gray,
                             std::size_t pixel_count,
                             const PreparedRgbToGray *prepared) noexcept {
    if (prepared == nullptr || pixel_count > kMaximumPixels ||
        pixel_count > std::numeric_limits<std::size_t>::max() / 3) {
        return GrayStatus::invalid_argument;
    }
    const std::size_t input_count = pixel_count * 3;
    if (input_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        pixel_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return GrayStatus::invalid_argument;
    }

    AddressRange input_range;
    AddressRange output_range;
    AddressRange prepared_range;
    const std::size_t input_bytes = input_count * sizeof(float);
    const std::size_t output_bytes = pixel_count * sizeof(float);
    if (!make_range(input_rgb, input_bytes, alignof(float), &input_range) ||
        !make_range(output_gray, output_bytes, alignof(float), &output_range) ||
        !make_range(prepared, sizeof(*prepared), alignof(PreparedRgbToGray),
                    &prepared_range) ||
        overlaps(input_range, output_range) ||
        overlaps(input_range, prepared_range) ||
        overlaps(output_range, prepared_range)) {
        return GrayStatus::invalid_argument;
    }
    if (!prepared_weights_valid(*prepared)) return GrayStatus::invalid_argument;

    try {
        std::vector<float> candidate(pixel_count);
        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            const float red = input_rgb[pixel * 3];
            const float green = input_rgb[pixel * 3 + 1];
            const float blue = input_rgb[pixel * 3 + 2];
            if (!std::isfinite(red) || !std::isfinite(green) ||
                !std::isfinite(blue)) {
                return GrayStatus::invalid_rgb;
            }

            const float green_term = prepared->weights_rgb[1] * green;
            if (!std::isfinite(green_term)) return GrayStatus::overflow;
            const float red_accumulation =
                std::fma(prepared->weights_rgb[0], red, green_term);
            if (!std::isfinite(red_accumulation)) return GrayStatus::overflow;
            const float gray =
                std::fma(prepared->weights_rgb[2], blue, red_accumulation);
            if (!std::isfinite(gray)) return GrayStatus::overflow;
            candidate[pixel] = gray;
        }

        if (pixel_count != 0)
            std::copy(candidate.begin(), candidate.end(), output_gray);
    } catch (const std::bad_alloc &) {
        return GrayStatus::allocation_failure;
    } catch (...) {
        return GrayStatus::allocation_failure;
    }
    return GrayStatus::ok;
}

}  // namespace imprint::acr
