#include "acr_exposure_sdr_mask.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kChannels = 3;
constexpr std::size_t kContextCount = 5;
constexpr std::size_t kPreLumaContext = 0;
constexpr std::size_t kRedContext = 1;
constexpr std::size_t kGreenContext = 2;
constexpr std::size_t kBlueContext = 3;
constexpr std::size_t kPostLumaContext = 4;
constexpr std::size_t kMaximumPixels = std::size_t{1} << 26;
constexpr float kMaskMinimum = -16.0f;
constexpr float kMaskMaximum = 16.0f;
constexpr float kUpperClip = 1.0f;
constexpr std::uint32_t kMaskApproximationBits = 0xbeadf655u;
constexpr float kSignedIndexLimit = 0x1p31f;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

enum class RangeStatus { ok, invalid_argument, overflow };

bool checked_product(std::size_t left, std::size_t right,
                     std::size_t *product) noexcept {
    if (product == nullptr ||
        (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

bool required_samples(std::size_t height, std::size_t stride,
                      std::size_t row_samples,
                      std::size_t *required) noexcept {
    std::size_t row_offset = 0;
    if (required == nullptr || height == 0 ||
        !checked_product(height - 1, stride, &row_offset) ||
        row_offset > std::numeric_limits<std::size_t>::max() - row_samples) {
        return false;
    }
    *required = row_offset + row_samples;
    return true;
}

RangeStatus make_range(const void *pointer, std::size_t count,
                       std::size_t element_size, std::size_t alignment,
                       AddressRange *range) noexcept {
    if (range == nullptr || alignment == 0) return RangeStatus::invalid_argument;
    std::size_t byte_count = 0;
    if (!checked_product(count, element_size, &byte_count) ||
        byte_count > static_cast<std::size_t>(
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

ExposureSdrMaskStatus map_range_status(RangeStatus status) noexcept {
    switch (status) {
        case RangeStatus::ok:
            return ExposureSdrMaskStatus::ok;
        case RangeStatus::invalid_argument:
            return ExposureSdrMaskStatus::invalid_argument;
        case RangeStatus::overflow:
            return ExposureSdrMaskStatus::arithmetic_overflow;
    }
    return ExposureSdrMaskStatus::invalid_argument;
}

bool finite(float value) noexcept { return std::isfinite(value); }

ExposureSdrMaskStatus require_finite(float value) noexcept {
    return finite(value) ? ExposureSdrMaskStatus::ok
                         : ExposureSdrMaskStatus::non_finite_intermediate;
}

bool valid_controls(const PreparedExposureSdrMask &prepared) noexcept {
    return finite(prepared.gain) && prepared.gain >= 0.0f &&
           finite(prepared.main_lut_scale) && prepared.main_lut_scale >= 0.0f &&
           finite(prepared.rational_scale) && prepared.rational_scale > 0.0f &&
           finite(prepared.sqrt_amount) &&
           finite(prepared.rational_denominator) &&
           prepared.rational_denominator > 0.0f && finite(prepared.epsilon) &&
           prepared.epsilon > 0.0f;
}

ExposureSdrMaskStatus mask_gain(float value, float *result) noexcept {
    if (!finite(value) || result == nullptr) {
        return ExposureSdrMaskStatus::invalid_argument;
    }

    // The finite-input subset makes FMINNM equivalent to fmin here. Keep the
    // two clamps and the ARM floor/conversion distinction explicit.
    float clipped = std::fmin(value, kMaskMaximum);
    if (clipped < kMaskMinimum) clipped = kMaskMinimum;
    float approximation = 1.0f;
    if (clipped != 0.0f) {
        const float scaled = clipped * 4.0f;
        if (require_finite(scaled) != ExposureSdrMaskStatus::ok) {
            return ExposureSdrMaskStatus::non_finite_intermediate;
        }
        const float floor_value = ::floorf(scaled);  // ARM FRINTM
        const float fraction = scaled - floor_value;
        if (require_finite(floor_value) != ExposureSdrMaskStatus::ok ||
            require_finite(fraction) != ExposureSdrMaskStatus::ok) {
            return ExposureSdrMaskStatus::non_finite_intermediate;
        }
        // The mask clamp bounds this conversion to [-64, 64]. FCVTMS uses
        // round-toward-minus-infinity, matching floor_value rather than a cast
        // that truncates toward zero.
        const auto exponent = static_cast<std::int32_t>(floor_value);
        const std::uint32_t power_bits =
            0x3f800000u + (static_cast<std::uint32_t>(exponent) << 23);
        float power_of_two = 0.0f;
        float factor = 0.0f;
        std::memcpy(&power_of_two, &power_bits, sizeof(power_of_two));
        std::memcpy(&factor, &kMaskApproximationBits, sizeof(factor));

        const float one_minus_fraction = 1.0f - fraction;
        const float first_fma =
            std::fmaf(one_minus_fraction, factor, 1.0f);
        const float second_fma = std::fmaf(first_fma, fraction, 1.0f);
        approximation = second_fma * power_of_two;
        if (require_finite(one_minus_fraction) != ExposureSdrMaskStatus::ok ||
            require_finite(first_fma) != ExposureSdrMaskStatus::ok ||
            require_finite(second_fma) != ExposureSdrMaskStatus::ok ||
            require_finite(approximation) != ExposureSdrMaskStatus::ok) {
            return ExposureSdrMaskStatus::non_finite_intermediate;
        }
    }
    *result = approximation;
    return ExposureSdrMaskStatus::ok;
}

ExposureSdrMaskStatus rational_map(float value,
                                   const PreparedExposureSdrMask &prepared,
                                   float *result) noexcept {
    const float scaled = value * prepared.rational_scale;
    const float first_denominator = scaled + 1.0f;
    const float first_quotient = scaled / first_denominator;
    const float scaled_by_q = scaled / prepared.rational_denominator;
    const float second_factor = scaled_by_q + 1.0f;
    const float product = first_quotient * second_factor;
    const float mapped = product / prepared.rational_scale;
    if (require_finite(scaled) != ExposureSdrMaskStatus::ok ||
        require_finite(first_denominator) != ExposureSdrMaskStatus::ok ||
        require_finite(first_quotient) != ExposureSdrMaskStatus::ok ||
        require_finite(scaled_by_q) != ExposureSdrMaskStatus::ok ||
        require_finite(second_factor) != ExposureSdrMaskStatus::ok ||
        require_finite(product) != ExposureSdrMaskStatus::ok ||
        require_finite(mapped) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }
    *result = mapped;
    return ExposureSdrMaskStatus::ok;
}

ExposureSdrMaskStatus preserve_ratio(float *low, float *middle, float *high,
                                     float epsilon,
                                     float *ratio) noexcept {
    const float numerator = *middle - *low;
    const float difference = *high - *low;
    const float denominator = difference > epsilon ? difference : epsilon;
    const float value = numerator / denominator;
    if (require_finite(numerator) != ExposureSdrMaskStatus::ok ||
        require_finite(difference) != ExposureSdrMaskStatus::ok ||
        require_finite(denominator) != ExposureSdrMaskStatus::ok ||
        require_finite(value) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }
    *ratio = value;
    return ExposureSdrMaskStatus::ok;
}

ExposureSdrMaskStatus reconstruct_middle(float *low, float *middle,
                                         float *high,
                                         float ratio) noexcept {
    const float difference = *high - *low;
    const float value = std::fmaf(ratio, difference, *low);
    if (require_finite(difference) != ExposureSdrMaskStatus::ok ||
        require_finite(value) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }
    *middle = value;
    return ExposureSdrMaskStatus::ok;
}

ExposureSdrMaskStatus curve_lookup(
    float value, const PreparedExposureSdrMask &prepared,
    float *result) noexcept {
    const float coordinate = value * prepared.main_lut_scale;
    if (require_finite(coordinate) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }
    // The supported curve input is nonnegative. Protect FCVTZS from signed
    // conversion overflow before converting and then validate both samples.
    if (coordinate < 0.0f || coordinate >= kSignedIndexLimit) {
        return ExposureSdrMaskStatus::lut_out_of_bounds;
    }
    const auto index = static_cast<std::int32_t>(coordinate);  // FCVTZS
    const float fraction = coordinate - static_cast<float>(index);
    const auto unsigned_index = static_cast<std::size_t>(index);
    if (unsigned_index >= prepared.main_lut_count ||
        prepared.main_lut_count - unsigned_index < 2) {
        return ExposureSdrMaskStatus::lut_out_of_bounds;
    }

    const float first = prepared.main_lut[unsigned_index];
    const float second = prepared.main_lut[unsigned_index + 1];
    const float delta = second - first;
    const float mapped = std::fmaf(fraction, delta, first);
    if (require_finite(fraction) != ExposureSdrMaskStatus::ok ||
        require_finite(first) != ExposureSdrMaskStatus::ok ||
        require_finite(second) != ExposureSdrMaskStatus::ok ||
        require_finite(delta) != ExposureSdrMaskStatus::ok ||
        require_finite(mapped) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }
    *result = mapped;
    return ExposureSdrMaskStatus::ok;
}

ExposureSdrMaskStatus validate_context_lut(
    const ExposureSdrMaskLut &lut, AddressRange *range) noexcept {
    if (lut.values == nullptr || range == nullptr || !finite(lut.scale) ||
        lut.scale < 0.0f ||
        lut.max_index > static_cast<std::uint32_t>(
                            std::numeric_limits<std::int32_t>::max())) {
        return ExposureSdrMaskStatus::invalid_argument;
    }
    // max_index is the largest accepted coordinate, not a promise that every
    // index up to that value will be read. The actual lookup checks its own
    // two-sample guard before indexing.
    if (lut.count < 2) {
        return ExposureSdrMaskStatus::lut_out_of_bounds;
    }
    const auto range_status = make_range(
        lut.values, lut.count, sizeof(float), alignof(float), range);
    return map_range_status(range_status);
}

std::int32_t fcvtzs_i32(float value) noexcept {
    // ARM FCVTZS saturates signed conversion overflow. Keep the boundary
    // checks before the cast so the equivalent C++ conversion is defined.
    if (value >= kSignedIndexLimit) {
        return std::numeric_limits<std::int32_t>::max();
    }
    if (value <= -kSignedIndexLimit) {
        return std::numeric_limits<std::int32_t>::min();
    }
    return static_cast<std::int32_t>(value);
}

ExposureSdrMaskStatus context_lookup(
    float value, const ExposureSdrMaskLut &lut, float *result) noexcept {
    const float coordinate = lut.scale * value;
    if (require_finite(coordinate) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }
    const std::int32_t index = fcvtzs_i32(coordinate);
    if (index < 0 || static_cast<std::uint32_t>(index) > lut.max_index) {
        return ExposureSdrMaskStatus::lut_out_of_bounds;
    }
    const float fraction = coordinate - static_cast<float>(index);
    const std::size_t sample_index = static_cast<std::size_t>(index);
    if (sample_index >= lut.count || lut.count - sample_index < 2) {
        return ExposureSdrMaskStatus::lut_out_of_bounds;
    }
    const float first_sample = lut.values[sample_index];
    const float next_sample = lut.values[sample_index + 1];
    const float complement = 1.0f - fraction;
    const float first = fraction * next_sample;
    // The recovered helper computes fma(firstSample, 1-frac, frac*nextSample),
    // which differs from the main curve's delta-based interpolation.
    const float mapped = std::fmaf(first_sample, complement, first);
    if (require_finite(fraction) != ExposureSdrMaskStatus::ok ||
        require_finite(first_sample) != ExposureSdrMaskStatus::ok ||
        require_finite(next_sample) != ExposureSdrMaskStatus::ok ||
        require_finite(complement) != ExposureSdrMaskStatus::ok ||
        require_finite(first) != ExposureSdrMaskStatus::ok ||
        require_finite(mapped) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }
    *result = mapped;
    return ExposureSdrMaskStatus::ok;
}

ExposureSdrMaskStatus apply_luma_context(
    float *rgb, const ExposureSdrMaskLut &lut, float epsilon) noexcept {
    const float red_plus_green = rgb[0] + rgb[1];
    const float green_plus_first = rgb[1] + red_plus_green;
    const float blue_plus_second = rgb[2] + green_plus_first;
    const float luma = blue_plus_second * 0.25f;
    if (require_finite(red_plus_green) != ExposureSdrMaskStatus::ok ||
        require_finite(green_plus_first) != ExposureSdrMaskStatus::ok ||
        require_finite(blue_plus_second) != ExposureSdrMaskStatus::ok ||
        require_finite(luma) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }

    float mapped_luma = 0.0f;
    auto status = context_lookup(luma, lut, &mapped_luma);
    if (status != ExposureSdrMaskStatus::ok) return status;
    const float denominator = luma > epsilon ? luma : epsilon;
    const float gain = mapped_luma / denominator;
    if (require_finite(denominator) != ExposureSdrMaskStatus::ok ||
        require_finite(gain) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }
    for (std::size_t channel = 0; channel < kChannels; ++channel) {
        const float mapped = rgb[channel] * gain;
        if (require_finite(mapped) != ExposureSdrMaskStatus::ok) {
            return ExposureSdrMaskStatus::non_finite_intermediate;
        }
        rgb[channel] = mapped;
    }
    return ExposureSdrMaskStatus::ok;
}

ExposureSdrMaskStatus apply_pixel(
    const float *source, const float *mask_value, bool has_mask,
    const PreparedExposureSdrMask &prepared, float sqrt_minus_one,
    float *destination) noexcept {
    float local_gain = prepared.gain;
    auto status = ExposureSdrMaskStatus::ok;
    if (has_mask) {
        float approximation = 0.0f;
        status = mask_gain(*mask_value, &approximation);
        if (status != ExposureSdrMaskStatus::ok) return status;
        local_gain = approximation * prepared.gain;
    }
    if (require_finite(local_gain) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }
    float rgb[kChannels];
    for (std::size_t channel = 0; channel < kChannels; ++channel) {
        if (!finite(source[channel]) || source[channel] < 0.0f) {
            return ExposureSdrMaskStatus::invalid_argument;
        }
        rgb[channel] = source[channel] * local_gain;
        if (require_finite(rgb[channel]) != ExposureSdrMaskStatus::ok) {
            return ExposureSdrMaskStatus::non_finite_intermediate;
        }
    }

    // Match the recovered comparison tree, including its tie selection.
    float *low = nullptr;
    float *middle = nullptr;
    float *high = nullptr;
    if (rgb[0] >= rgb[1]) {
        if (rgb[0] >= rgb[2]) {
            high = &rgb[0];
            if (rgb[1] < rgb[2]) {
                low = &rgb[1];
                middle = &rgb[2];
            } else {
                low = &rgb[2];
                middle = &rgb[1];
            }
        } else {
            low = &rgb[1];
            middle = &rgb[0];
            high = &rgb[2];
        }
    } else if (rgb[1] >= rgb[2]) {
        high = &rgb[1];
        if (rgb[0] < rgb[2]) {
            low = &rgb[0];
            middle = &rgb[2];
        } else {
            low = &rgb[2];
            middle = &rgb[0];
        }
    } else {
        low = &rgb[0];
        middle = &rgb[1];
        high = &rgb[2];
    }

    if (prepared.rational_denominator < 1.0f) {
        if (*high == *low) {
            float mapped = 0.0f;
            status = rational_map(*low, prepared, &mapped);
            if (status != ExposureSdrMaskStatus::ok) return status;
            *low = mapped;
            *middle = mapped;
            *high = mapped;
        } else {
            float ratio = 0.0f;
            status = preserve_ratio(low, middle, high, prepared.epsilon, &ratio);
            if (status != ExposureSdrMaskStatus::ok) return status;
            float mapped_low = 0.0f;
            float mapped_high = 0.0f;
            status = rational_map(*low, prepared, &mapped_low);
            if (status != ExposureSdrMaskStatus::ok) return status;
            status = rational_map(*high, prepared, &mapped_high);
            if (status != ExposureSdrMaskStatus::ok) return status;
            *low = mapped_low;
            *high = mapped_high;
            status = reconstruct_middle(low, middle, high, ratio);
            if (status != ExposureSdrMaskStatus::ok) return status;
        }
    }

    // Mask baseline 013729ac always invokes 00dc6288, including when no clip
    // is needed. No-mask baseline 01372428 branches to it only for high>1
    // (013726e0..013726e4), preserving the original median below that limit.
    float ratio = 0.0f;
    if (has_mask || *high > kUpperClip) {
        status = preserve_ratio(low, middle, high, prepared.epsilon, &ratio);
        if (status != ExposureSdrMaskStatus::ok) return status;
        *low = std::fmin(*low, kUpperClip);  // ARM FMINNM for finite operands
        *high = std::fmin(*high, kUpperClip);
        status = reconstruct_middle(low, middle, high, ratio);
        if (status != ExposureSdrMaskStatus::ok) return status;
    }

    if (prepared.rational_scale > 1.0f) {
        ratio = 0.0f;
        if (*high != *low) {
            status = preserve_ratio(low, middle, high, prepared.epsilon, &ratio);
            if (status != ExposureSdrMaskStatus::ok) return status;
        }
        float mapped_low = 0.0f;
        float mapped_high = 0.0f;
        status = curve_lookup(*low, prepared, &mapped_low);
        if (status != ExposureSdrMaskStatus::ok) return status;
        status = curve_lookup(*high, prepared, &mapped_high);
        if (status != ExposureSdrMaskStatus::ok) return status;
        *low = mapped_low;
        *high = mapped_high;
        status = reconstruct_middle(low, middle, high, ratio);
        if (status != ExposureSdrMaskStatus::ok) return status;
    }

    if (prepared.sqrt_amount > 1.0f) {
        ratio = 0.0f;
        if (*high != *low) {
            status = preserve_ratio(low, middle, high, prepared.epsilon, &ratio);
            if (status != ExposureSdrMaskStatus::ok) return status;
        }
        const auto sqrt_map = [&](float value, float *mapped) noexcept {
            const float square = value * value;
            const float numerator = square * prepared.sqrt_amount;
            const float denominator = std::fmaf(sqrt_minus_one, square, 1.0f);
            const float quotient = numerator / denominator;
            const float root = ::sqrtf(quotient);
            if (require_finite(square) != ExposureSdrMaskStatus::ok ||
                require_finite(numerator) != ExposureSdrMaskStatus::ok ||
                require_finite(denominator) != ExposureSdrMaskStatus::ok ||
                require_finite(quotient) != ExposureSdrMaskStatus::ok ||
                require_finite(root) != ExposureSdrMaskStatus::ok) {
                return ExposureSdrMaskStatus::non_finite_intermediate;
            }
            *mapped = root;
            return ExposureSdrMaskStatus::ok;
        };
        float mapped_low = 0.0f;
        float mapped_high = 0.0f;
        status = sqrt_map(*low, &mapped_low);
        if (status != ExposureSdrMaskStatus::ok) return status;
        status = sqrt_map(*high, &mapped_high);
        if (status != ExposureSdrMaskStatus::ok) return status;
        *low = mapped_low;
        *high = mapped_high;
        status = reconstruct_middle(low, middle, high, ratio);
        if (status != ExposureSdrMaskStatus::ok) return status;
    }

    if (prepared.context[kPreLumaContext].values != nullptr) {
        status = apply_luma_context(rgb, prepared.context[kPreLumaContext],
                                    prepared.epsilon);
        if (status != ExposureSdrMaskStatus::ok) return status;
    }

    const bool use_rgb_contexts =
        prepared.context[kRedContext].values != nullptr &&
        prepared.context[kGreenContext].values != nullptr &&
        prepared.context[kBlueContext].values != nullptr;
    if (use_rgb_contexts) {
        for (std::size_t channel = 0; channel < kChannels; ++channel) {
            float mapped = 0.0f;
            status = context_lookup(
                rgb[channel], prepared.context[kRedContext + channel], &mapped);
            if (status != ExposureSdrMaskStatus::ok) return status;
            rgb[channel] = mapped;
        }
    }

    if (prepared.context[kPostLumaContext].values != nullptr) {
        status = apply_luma_context(rgb, prepared.context[kPostLumaContext],
                                    prepared.epsilon);
        if (status != ExposureSdrMaskStatus::ok) return status;
        for (std::size_t channel = 0; channel < kChannels; ++channel) {
            rgb[channel] = std::fmin(rgb[channel], kUpperClip);
            if (require_finite(rgb[channel]) != ExposureSdrMaskStatus::ok) {
                return ExposureSdrMaskStatus::non_finite_intermediate;
            }
        }
    }

    for (std::size_t channel = 0; channel < kChannels; ++channel) {
        if (require_finite(rgb[channel]) != ExposureSdrMaskStatus::ok) {
            return ExposureSdrMaskStatus::non_finite_intermediate;
        }
        destination[channel] = rgb[channel];
    }
    return ExposureSdrMaskStatus::ok;
}

}  // namespace

namespace {

ExposureSdrMaskStatus apply_prepared_exposure_sdr_impl(
    bool has_mask, const float *src, std::size_t src_count,
    std::size_t src_stride, std::size_t height, std::size_t width,
    const float *mask, std::size_t mask_count, std::size_t mask_stride,
    const PreparedExposureSdrMask *prepared, float *dst,
    std::size_t dst_count, std::size_t dst_stride) noexcept {
    if (src == nullptr || (has_mask && mask == nullptr) || prepared == nullptr ||
        dst == nullptr || height == 0 || width == 0) {
        return ExposureSdrMaskStatus::invalid_argument;
    }

    std::size_t pixels = 0;
    if (!checked_product(width, height, &pixels)) {
        return ExposureSdrMaskStatus::arithmetic_overflow;
    }
    if (pixels > kMaximumPixels) return ExposureSdrMaskStatus::resource_limit;

    std::size_t row_samples = 0;
    if (!checked_product(width, kChannels, &row_samples)) {
        return ExposureSdrMaskStatus::arithmetic_overflow;
    }
    if (src_stride < row_samples || dst_stride < row_samples ||
        (has_mask && mask_stride < width)) {
        return ExposureSdrMaskStatus::invalid_argument;
    }

    std::size_t unused_stride_bytes = 0;
    const auto max_ptrdiff = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    if (!checked_product(src_stride, sizeof(float), &unused_stride_bytes) ||
        unused_stride_bytes > max_ptrdiff ||
        (has_mask &&
         (!checked_product(mask_stride, sizeof(float), &unused_stride_bytes) ||
          unused_stride_bytes > max_ptrdiff)) ||
        !checked_product(dst_stride, sizeof(float), &unused_stride_bytes) ||
        unused_stride_bytes > max_ptrdiff) {
        return ExposureSdrMaskStatus::arithmetic_overflow;
    }

    std::size_t required_src = 0;
    std::size_t required_mask = 0;
    std::size_t required_dst = 0;
    if (!required_samples(height, src_stride, row_samples, &required_src) ||
        (has_mask &&
         !required_samples(height, mask_stride, width, &required_mask)) ||
        !required_samples(height, dst_stride, row_samples, &required_dst)) {
        return ExposureSdrMaskStatus::arithmetic_overflow;
    }
    std::size_t required_src_bytes = 0;
    std::size_t required_mask_bytes = 0;
    std::size_t required_dst_bytes = 0;
    if (!checked_product(required_src, sizeof(float), &required_src_bytes) ||
        required_src_bytes > max_ptrdiff ||
        (has_mask &&
         (!checked_product(required_mask, sizeof(float), &required_mask_bytes) ||
          required_mask_bytes > max_ptrdiff)) ||
        !checked_product(required_dst, sizeof(float), &required_dst_bytes) ||
        required_dst_bytes > max_ptrdiff) {
        return ExposureSdrMaskStatus::arithmetic_overflow;
    }

    AddressRange src_range;
    AddressRange mask_range;
    AddressRange prepared_range;
    AddressRange dst_range;
    auto range_status = make_range(src, src_count, sizeof(float), alignof(float),
                                   &src_range);
    if (range_status != RangeStatus::ok) return map_range_status(range_status);
    if (has_mask) {
        range_status = make_range(mask, mask_count, sizeof(float), alignof(float),
                                  &mask_range);
        if (range_status != RangeStatus::ok) {
            return map_range_status(range_status);
        }
    }
    range_status = make_range(prepared, 1, sizeof(*prepared),
                              alignof(PreparedExposureSdrMask), &prepared_range);
    if (range_status != RangeStatus::ok) return map_range_status(range_status);
    range_status = make_range(dst, dst_count, sizeof(float), alignof(float),
                              &dst_range);
    if (range_status != RangeStatus::ok) return map_range_status(range_status);

    if (src_count < required_src || (has_mask && mask_count < required_mask) ||
        dst_count < required_dst) {
        return ExposureSdrMaskStatus::buffer_too_small;
    }

    if (!valid_controls(*prepared)) return ExposureSdrMaskStatus::invalid_argument;

    const bool use_main_lut = prepared->rational_scale > 1.0f;
    AddressRange main_lut_range;
    if (use_main_lut) {
        if (prepared->main_lut == nullptr || prepared->main_lut_count < 2) {
            return ExposureSdrMaskStatus::lut_out_of_bounds;
        }
        range_status = make_range(prepared->main_lut, prepared->main_lut_count,
                                  sizeof(float), alignof(float), &main_lut_range);
        if (range_status != RangeStatus::ok) return map_range_status(range_status);
    }

    bool use_context[kContextCount] = {};
    use_context[kPreLumaContext] =
        prepared->context[kPreLumaContext].values != nullptr;
    use_context[kRedContext] = prepared->context[kRedContext].values != nullptr &&
                               prepared->context[kGreenContext].values != nullptr &&
                               prepared->context[kBlueContext].values != nullptr;
    use_context[kGreenContext] = use_context[kRedContext];
    use_context[kBlueContext] = use_context[kRedContext];
    use_context[kPostLumaContext] =
        prepared->context[kPostLumaContext].values != nullptr;
    AddressRange context_ranges[kContextCount]{};
    for (std::size_t index = 0; index < kContextCount; ++index) {
        if (!use_context[index]) continue;
        const auto status = validate_context_lut(prepared->context[index],
                                                 &context_ranges[index]);
        if (status != ExposureSdrMaskStatus::ok) return status;
    }

    const bool exact_in_place = src == dst && src_count == dst_count &&
                                src_stride == dst_stride;
    if ((overlaps(src_range, dst_range) && !exact_in_place) ||
        (has_mask && overlaps(dst_range, mask_range)) ||
        overlaps(dst_range, prepared_range) ||
        (use_main_lut && overlaps(dst_range, main_lut_range))) {
        return ExposureSdrMaskStatus::unsupported_aliasing;
    }
    for (std::size_t index = 0; index < kContextCount; ++index) {
        if (use_context[index] && overlaps(dst_range, context_ranges[index])) {
            return ExposureSdrMaskStatus::unsupported_aliasing;
        }
    }

    std::size_t candidate_count = 0;
    std::size_t candidate_bytes = 0;
    if (!checked_product(pixels, kChannels, &candidate_count) ||
        !checked_product(candidate_count, sizeof(float), &candidate_bytes) ||
        candidate_bytes > static_cast<std::size_t>(
                              std::numeric_limits<std::ptrdiff_t>::max())) {
        return ExposureSdrMaskStatus::arithmetic_overflow;
    }

    const float sqrt_minus_one = prepared->sqrt_amount + (-1.0f);
    if (require_finite(sqrt_minus_one) != ExposureSdrMaskStatus::ok) {
        return ExposureSdrMaskStatus::non_finite_intermediate;
    }

    try {
        std::vector<float> candidate;
        if (candidate_count > candidate.max_size()) {
            return ExposureSdrMaskStatus::resource_limit;
        }
        candidate.resize(candidate_count);

        for (std::size_t y = 0; y < height; ++y) {
            const std::size_t src_row = y * src_stride;
            const std::size_t mask_row = has_mask ? y * mask_stride : 0;
            const std::size_t candidate_row = y * row_samples;
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t source_offset = src_row + x * kChannels;
                const std::size_t output_offset = candidate_row + x * kChannels;
                const float *pixel_mask =
                    has_mask ? &mask[mask_row + x] : nullptr;
                const auto status = apply_pixel(
                    src + source_offset, pixel_mask, has_mask, *prepared,
                    sqrt_minus_one, candidate.data() + output_offset);
                if (status != ExposureSdrMaskStatus::ok) return status;
            }
        }

        const std::size_t row_bytes = row_samples * sizeof(float);
        for (std::size_t y = 0; y < height; ++y) {
            std::memcpy(dst + y * dst_stride,
                        candidate.data() + y * row_samples, row_bytes);
        }
        return ExposureSdrMaskStatus::ok;
    } catch (const std::bad_alloc &) {
        return ExposureSdrMaskStatus::allocation_failure;
    } catch (const std::length_error &) {
        return ExposureSdrMaskStatus::resource_limit;
    } catch (...) {
        return ExposureSdrMaskStatus::allocation_failure;
    }
}

}  // namespace

ExposureSdrMaskStatus apply_prepared_exposure_sdr_mask(
    const float *src, std::size_t src_count, std::size_t src_stride,
    std::size_t height, std::size_t width, const float *mask,
    std::size_t mask_count, std::size_t mask_stride,
    const PreparedExposureSdrMask *prepared, float *dst,
    std::size_t dst_count, std::size_t dst_stride) noexcept {
    return apply_prepared_exposure_sdr_impl(
        true, src, src_count, src_stride, height, width, mask, mask_count,
        mask_stride, prepared, dst, dst_count, dst_stride);
}

ExposureSdrMaskStatus apply_prepared_exposure_sdr_nomask(
    const float *src, std::size_t src_count, std::size_t src_stride,
    std::size_t height, std::size_t width,
    const PreparedExposureSdrMask *prepared, float *dst,
    std::size_t dst_count, std::size_t dst_stride) noexcept {
    return apply_prepared_exposure_sdr_impl(
        false, src, src_count, src_stride, height, width, nullptr, 0, 0,
        prepared, dst, dst_count, dst_stride);
}

}  // namespace imprint::acr
