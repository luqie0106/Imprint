#include "acr_exposure_signed.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace imprint::acr {
namespace {

constexpr std::size_t kChannels = 3;
constexpr std::size_t kMaximumPixels = std::size_t{1} << 26;
constexpr float kPixelMinimum = -1000.0f;
constexpr float kPixelMaximum = 1000.0f;
constexpr float kMaskMinimum = -16.0f;
constexpr float kMaskMaximum = 16.0f;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

enum class RangeStatus { ok, invalid_argument, overflow };

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t *result) noexcept {
    if (result == nullptr || (left != 0 &&
                              right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

RangeStatus make_range(const void *pointer, std::size_t byte_count,
                       std::size_t alignment,
                       AddressRange *range) noexcept {
    if (range == nullptr || alignment == 0) return RangeStatus::invalid_argument;
    if (byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::ptrdiff_t>::max())) {
        return RangeStatus::overflow;
    }
    if (byte_count == 0) {
        if (pointer == nullptr) {
            *range = {};
            return RangeStatus::ok;
        }
        const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
        if (begin % alignment != 0) return RangeStatus::invalid_argument;
        *range = {begin, begin};
        return RangeStatus::ok;
    }
    if (pointer == nullptr) return RangeStatus::invalid_argument;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) return RangeStatus::invalid_argument;
    if (byte_count > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return RangeStatus::overflow;
    }
    *range = {begin, begin + byte_count};
    return RangeStatus::ok;
}

bool overlaps(const AddressRange &left, const AddressRange &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

SignedExposureStatus range_status(RangeStatus status) noexcept {
    switch (status) {
        case RangeStatus::ok:
            return SignedExposureStatus::ok;
        case RangeStatus::overflow:
            return SignedExposureStatus::overflow;
        case RangeStatus::invalid_argument:
            return SignedExposureStatus::invalid_argument;
    }
    return SignedExposureStatus::invalid_argument;
}

bool valid_controls(const PreparedSignedExposure &prepared) noexcept {
    return std::isfinite(prepared.gain) && prepared.gain >= 0.0f &&
           std::isfinite(prepared.compression) && prepared.compression > 0.0f &&
           std::isfinite(prepared.expansion) && prepared.expansion > 0.0f &&
           std::isfinite(prepared.epsilon) && prepared.epsilon > 0.0f;
}

bool channels_enabled(const PreparedSignedExposure &prepared) noexcept {
    return prepared.channel[0].values != nullptr &&
           prepared.channel[1].values != nullptr &&
           prepared.channel[2].values != nullptr;
}

SignedExposureStatus validate_lut(const SignedExposureLut &lut,
                                  AddressRange *range) noexcept {
    if (lut.values == nullptr || range == nullptr ||
        lut.max_index > static_cast<std::uint32_t>(
                            std::numeric_limits<int>::max() - 1) ||
        !std::isfinite(lut.scale) || lut.scale < 0.0f) {
        return SignedExposureStatus::invalid_argument;
    }
    const std::size_t required = static_cast<std::size_t>(lut.max_index) + 2;
    if (lut.capacity < required) return SignedExposureStatus::lut_out_of_bounds;
    std::size_t bytes = 0;
    if (!checked_multiply(lut.capacity, sizeof(float), &bytes)) {
        return SignedExposureStatus::overflow;
    }
    const auto status = make_range(lut.values, bytes, alignof(float), range);
    return range_status(status);
}

bool finite(float value) noexcept { return std::isfinite(value); }

float clamp_pixel(float value) noexcept {
    if (value < kPixelMinimum) return kPixelMinimum;
    if (value > kPixelMaximum) return kPixelMaximum;
    return value;
}

SignedExposureStatus lookup(const SignedExposureLut &lut, float value,
                            float *result) noexcept {
    const float coordinate = value * lut.scale;
    std::int32_t index = 0;
    if (std::isnan(coordinate)) {
        index = 0;  // ARM FCVTZS maps NaN to zero.
    } else if (coordinate >= 0x1p31f) {
        index = std::numeric_limits<std::int32_t>::max();
    } else if (coordinate <= -0x1p31f) {
        index = std::numeric_limits<std::int32_t>::min();
    } else {
        index = static_cast<std::int32_t>(coordinate);
    }
    if (index < 0 || static_cast<std::uint32_t>(index) > lut.max_index ||
        static_cast<std::size_t>(index) + 1 >= lut.capacity) {
        return SignedExposureStatus::lut_out_of_bounds;
    }
    const float first = lut.values[index];
    const float second = lut.values[index + 1];
    if (!finite(first) || !finite(second)) {
        return SignedExposureStatus::invalid_argument;
    }

    const float fraction = coordinate - static_cast<float>(index);
    const float one_minus_fraction = 1.0f - fraction;
    const float next_part = fraction * second;
    const float interpolated = std::fmaf(first, one_minus_fraction, next_part);
    *result = interpolated;
    return SignedExposureStatus::ok;
}

SignedExposureStatus calculate_mask_gain(float mask_value, float gain,
                                         float *local_gain) noexcept {
    if (!finite(mask_value)) return SignedExposureStatus::invalid_argument;
    if (mask_value < kMaskMinimum) mask_value = kMaskMinimum;
    if (mask_value > kMaskMaximum) mask_value = kMaskMaximum;

    float mask_gain = 1.0f;
    if (mask_value != 0.0f) {
        const float scaled_mask = mask_value * 4.0f;
        const float floored = ::floorf(scaled_mask);
        const auto exponent = static_cast<std::int32_t>(floored);
        const float fraction = scaled_mask - static_cast<float>(exponent);

        const std::uint32_t power_bits =
            0x3f800000u + static_cast<std::uint32_t>(exponent) * 0x00800000u;
        float power_of_two = 0.0f;
        std::memcpy(&power_of_two, &power_bits, sizeof(power_of_two));
        constexpr std::uint32_t kFactorBits = 0xbeadf655u;
        float factor = 0.0f;
        std::memcpy(&factor, &kFactorBits, sizeof(factor));
        const float one_minus_fraction = 1.0f - fraction;
        const float p = std::fmaf(one_minus_fraction, factor, 1.0f);
        const float q = std::fmaf(p, fraction, 1.0f);
        mask_gain = q * power_of_two;
    }

    *local_gain = mask_gain * gain;
    return SignedExposureStatus::ok;
}

void adapt_tone(float rgb[3], float *local_gain,
                const PreparedSignedExposure &prepared) noexcept {
    const float green_plus_red = rgb[1] + rgb[0];
    const float double_green_plus_red = rgb[1] + green_plus_red;
    const float sum = rgb[2] + double_green_plus_red;

    if (prepared.compression < 1.0f) {
        const float gray_scale = *local_gain * 0.25f;
        const float gray = sum * gray_scale;
        const float gray_plus_one = gray + 1.0f;
        const float first_quotient = gray / gray_plus_one;
        const float gray_over_compression = gray / prepared.compression;
        const float second_factor = gray_over_compression + 1.0f;
        const float mapped = first_quotient * second_factor;
        const float numerator = mapped + prepared.epsilon;
        const float denominator = gray + prepared.epsilon;
        const float ratio = numerator / denominator;
        *local_gain = *local_gain * ratio;
    } else if (prepared.expansion < 1.0f) {
        const float gray_scale = *local_gain * 0.25f;
        const float gray = sum * gray_scale;
        const float z = gray - 1.0f;
        const float inverse_expansion = 1.0f / prepared.expansion;
        const float gray_times_four = gray * 4.0f;
        const float term = inverse_expansion * gray_times_four;
        const float radicand = std::fmaf(z, z, term);
        const float root = ::sqrtf(radicand);
        const float z_plus_root = z + root;
        const float expansion_half = prepared.expansion * 0.5f;
        const float mapped = expansion_half * z_plus_root;
        const float numerator = mapped + prepared.epsilon;
        const float denominator = gray + prepared.epsilon;
        const float ratio = numerator / denominator;
        *local_gain = *local_gain * ratio;
    }

    for (std::size_t channel = 0; channel < kChannels; ++channel) {
        rgb[channel] = rgb[channel] * *local_gain;
    }
}

}  // namespace

SignedExposureStatus apply_prepared_signed_exposure_rgb(
    const float *input, float *output, std::size_t pixel_count,
    const float *mask, std::size_t mask_count,
    const PreparedSignedExposure *prepared) noexcept {
    if (prepared == nullptr || pixel_count == 0 || pixel_count > kMaximumPixels ||
        (mask == nullptr && mask_count != 0) ||
        (mask != nullptr && mask_count < pixel_count)) {
        return SignedExposureStatus::invalid_argument;
    }

    std::size_t float_count = 0;
    std::size_t input_bytes = 0;
    std::size_t mask_bytes = 0;
    if (!checked_multiply(pixel_count, kChannels, &float_count) ||
        !checked_multiply(float_count, sizeof(float), &input_bytes) ||
        !checked_multiply(mask_count, sizeof(float), &mask_bytes)) {
        return SignedExposureStatus::overflow;
    }

    AddressRange input_range;
    AddressRange output_range;
    AddressRange prepared_range;
    AddressRange mask_range;
    auto status = make_range(input, input_bytes, alignof(float), &input_range);
    if (status != RangeStatus::ok) return range_status(status);
    status = make_range(output, input_bytes, alignof(float), &output_range);
    if (status != RangeStatus::ok) return range_status(status);
    status = make_range(prepared, sizeof(*prepared), alignof(PreparedSignedExposure),
                        &prepared_range);
    if (status != RangeStatus::ok) return range_status(status);
    status = make_range(mask, mask_bytes, alignof(float), &mask_range);
    if (status != RangeStatus::ok) return range_status(status);
    if (!valid_controls(*prepared)) return SignedExposureStatus::invalid_argument;

    const bool use_channels = channels_enabled(*prepared);
    const bool use_gray = prepared->gray.values != nullptr;
    AddressRange channel_ranges[kChannels]{};
    AddressRange gray_range;
    if (use_channels) {
        for (std::size_t channel = 0; channel < kChannels; ++channel) {
            const auto lut_status = validate_lut(prepared->channel[channel],
                                                 &channel_ranges[channel]);
            if (lut_status != SignedExposureStatus::ok) return lut_status;
        }
    }
    if (use_gray) {
        const auto lut_status = validate_lut(prepared->gray, &gray_range);
        if (lut_status != SignedExposureStatus::ok) return lut_status;
    }

    if ((overlaps(input_range, output_range) && input != output) ||
        overlaps(output_range, prepared_range) ||
        overlaps(output_range, mask_range)) {
        return SignedExposureStatus::unsupported_aliasing;
    }
    if (use_channels) {
        for (const auto &range : channel_ranges) {
            if (overlaps(output_range, range)) {
                return SignedExposureStatus::unsupported_aliasing;
            }
        }
    }
    if (use_gray && overlaps(output_range, gray_range)) {
        return SignedExposureStatus::unsupported_aliasing;
    }

    std::unique_ptr<float[]> temporary(new (std::nothrow) float[float_count]);
    if (!temporary) return SignedExposureStatus::allocation_failure;

    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
        const std::size_t offset = pixel * kChannels;
        float rgb[kChannels];
        bool originally_negative[kChannels];
        for (std::size_t channel = 0; channel < kChannels; ++channel) {
            const float source = input[offset + channel];
            if (!finite(source)) return SignedExposureStatus::invalid_argument;
            const float clamped = clamp_pixel(source);
            originally_negative[channel] = clamped < 0.0f;
            rgb[channel] = clamped < 0.0f ? -clamped : clamped;
        }

        float local_gain = prepared->gain;
        if (mask != nullptr) {
            const auto mask_status =
                calculate_mask_gain(mask[pixel], prepared->gain, &local_gain);
            if (mask_status != SignedExposureStatus::ok) return mask_status;
        }

        adapt_tone(rgb, &local_gain, *prepared);

        if (use_channels) {
            for (std::size_t channel = 0; channel < kChannels; ++channel) {
                if (rgb[channel] < 1.0f) {
                    float mapped = 0.0f;
                    const auto lut_status = lookup(prepared->channel[channel],
                                                   rgb[channel], &mapped);
                    if (lut_status != SignedExposureStatus::ok) return lut_status;
                    rgb[channel] = mapped;
                }
            }
        }

        if (use_gray) {
            const float green_plus_red = rgb[1] + rgb[0];
            const float double_green_plus_red = rgb[1] + green_plus_red;
            const float sum = rgb[2] + double_green_plus_red;
            const float gray = sum * 0.25f;
            if (gray < 1.0f) {
                float mapped = 0.0f;
                const auto lut_status = lookup(prepared->gray, gray, &mapped);
                if (lut_status != SignedExposureStatus::ok) return lut_status;
                const float denominator =
                    gray > prepared->epsilon ? gray : prepared->epsilon;
                const float gray_gain = mapped / denominator;
                for (std::size_t channel = 0; channel < kChannels; ++channel) {
                    rgb[channel] = rgb[channel] * gray_gain;
                }
            }
        }

        for (std::size_t channel = 0; channel < kChannels; ++channel) {
            float value = originally_negative[channel] ? -rgb[channel] : rgb[channel];
            value = std::fmin(value, kPixelMaximum);
            if (value < kPixelMinimum) value = kPixelMinimum;
            temporary[offset + channel] = value;
        }
    }

    std::memcpy(output, temporary.get(), input_bytes);
    return SignedExposureStatus::ok;
}

}  // namespace imprint::acr
