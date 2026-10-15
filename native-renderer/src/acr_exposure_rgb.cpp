#include "acr_exposure_rgb.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace imprint::acr {
namespace {

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

bool valid_controls(const PreparedExposureRgb &prepared) noexcept {
    return std::isfinite(prepared.gain) && prepared.gain >= 0.0f &&
           std::isfinite(prepared.lut_scale) && prepared.lut_scale >= 0.0f &&
           std::isfinite(prepared.rational_gain) &&
           prepared.rational_gain >= 0.0f &&
           std::isfinite(prepared.epsilon) && prepared.epsilon > 0.0f &&
           std::isfinite(prepared.positive_gain) && prepared.positive_gain >= 1.0f &&
           std::isfinite(prepared.white_clip_squared) &&
           prepared.white_clip_squared > 0.0f && prepared.white_clip_squared <= 1.0f &&
           std::isfinite(prepared.whites_lut_scale) && prepared.whites_lut_scale >= 0.0f;
}

// The branch structure mirrors the recovered comparisons. In particular,
// equal values retain the same channel identities selected by <= in the
// original routine.
void sort_channels(const float rgb[3], std::size_t *minimum,
                   std::size_t *middle, std::size_t *maximum) noexcept {
    const float red = rgb[0];
    const float green = rgb[1];
    const float blue = rgb[2];
    if (green <= red) {
        if (blue <= red) {
            if (blue <= green) {
                *minimum = 2;
                *middle = 1;
            } else {
                *minimum = 1;
                *middle = 2;
            }
            *maximum = 0;
        } else {
            *minimum = 1;
            *middle = 0;
            *maximum = 2;
        }
    } else if (blue <= green) {
        if (blue <= red) {
            *minimum = 2;
            *middle = 0;
        } else {
            *minimum = 0;
            *middle = 2;
        }
        *maximum = 1;
    } else {
        *minimum = 0;
        *middle = 1;
        *maximum = 2;
    }
}

ExposureRgbStatus hue_fraction(float minimum, float middle, float maximum,
                               float epsilon, float *fraction) noexcept {
    float result = 0.0f;
    if (minimum != maximum) {
        float span = maximum - minimum;
        if (!std::isfinite(span)) return ExposureRgbStatus::overflow;
        if (span <= epsilon) span = epsilon;
        const float offset = middle - minimum;
        if (!std::isfinite(offset)) return ExposureRgbStatus::overflow;
        result = offset / span;
        if (!std::isfinite(result)) return ExposureRgbStatus::overflow;
    }
    *fraction = result;
    return ExposureRgbStatus::ok;
}

ExposureRgbStatus apply_highlight_clip(float epsilon, float *minimum,
                                       float *middle, float *maximum) noexcept {
    const float original_minimum = *minimum;
    const float original_middle = *middle;
    const float original_maximum = *maximum;
    float denominator = original_maximum - original_minimum;
    if (!std::isfinite(denominator)) return ExposureRgbStatus::overflow;
    if (denominator <= epsilon) denominator = epsilon;

    const float clipped_minimum = std::min(original_minimum, 1.0f);
    const float clipped_maximum = std::min(original_maximum, 1.0f);
    const float middle_offset = original_middle - original_minimum;
    if (!std::isfinite(middle_offset)) return ExposureRgbStatus::overflow;
    const float normalized_middle = middle_offset / denominator;
    if (!std::isfinite(normalized_middle)) return ExposureRgbStatus::overflow;
    const float clipped_span = clipped_maximum - clipped_minimum;
    if (!std::isfinite(clipped_span)) return ExposureRgbStatus::overflow;
    const float scaled_middle = clipped_span * normalized_middle;
    if (!std::isfinite(scaled_middle)) return ExposureRgbStatus::overflow;
    const float clipped_middle = std::fma(clipped_span, normalized_middle,
                                          clipped_minimum);  // 00dc62c8
    if (!std::isfinite(clipped_middle)) return ExposureRgbStatus::overflow;

    *minimum = clipped_minimum;
    *middle = clipped_middle;
    *maximum = clipped_maximum;
    return ExposureRgbStatus::ok;
}

ExposureRgbStatus lookup(float value, const PreparedExposureRgb &prepared,
                         float *result) noexcept {
    const float coordinate = value * prepared.lut_scale;
    if (!std::isfinite(coordinate)) return ExposureRgbStatus::overflow;
    if (coordinate < 0.0f || coordinate >= 0x1p31f) {
        return ExposureRgbStatus::lut_out_of_bounds;
    }

    // The recovered code converts the nonnegative coordinate to signed int,
    // truncating toward zero, then reads both index and index + 1.
    const auto index = static_cast<std::uint32_t>(coordinate);
    if (static_cast<std::size_t>(index) >= prepared.lut_size ||
        static_cast<std::size_t>(index) + 1 >= prepared.lut_size) {
        return ExposureRgbStatus::lut_out_of_bounds;
    }
    const float first = prepared.lut[index];
    const float second = prepared.lut[index + 1];
    if (!std::isfinite(first) || !std::isfinite(second) ||
        first < 0.0f || second < 0.0f) {
        return ExposureRgbStatus::invalid_rgb;
    }
    const float delta = second - first;
    if (!std::isfinite(delta)) return ExposureRgbStatus::overflow;
    const float fraction = coordinate - static_cast<float>(index);
    if (!std::isfinite(fraction)) return ExposureRgbStatus::overflow;
    const float scaled_delta = delta * fraction;
    if (!std::isfinite(scaled_delta)) return ExposureRgbStatus::overflow;
    const float interpolated = std::fma(delta, fraction, first);  // 01372388
    if (!std::isfinite(interpolated)) return ExposureRgbStatus::overflow;
    if (interpolated < 0.0f) return ExposureRgbStatus::invalid_rgb;
    *result = interpolated;
    return ExposureRgbStatus::ok;
}

ExposureRgbStatus rational_curve(float value, float gain,
                                 float *result) noexcept {
    const float square = value * value;
    if (!std::isfinite(square)) return ExposureRgbStatus::overflow;
    const float numerator = square * gain;
    if (!std::isfinite(numerator)) return ExposureRgbStatus::overflow;
    const float gain_minus_one = gain - 1.0f;
    if (!std::isfinite(gain_minus_one)) return ExposureRgbStatus::overflow;
    const float denominator_product = square * gain_minus_one;
    if (!std::isfinite(denominator_product)) return ExposureRgbStatus::overflow;
    const float denominator = std::fma(square, gain_minus_one, 1.0f);  // 01372220
    if (!std::isfinite(denominator) || denominator <= 0.0f) {
        return ExposureRgbStatus::overflow;
    }
    const float quotient = numerator / denominator;
    if (!std::isfinite(quotient) || quotient < 0.0f) {
        return ExposureRgbStatus::overflow;
    }
    const float curved = std::sqrt(quotient);
    if (!std::isfinite(curved)) return ExposureRgbStatus::overflow;
    *result = curved;
    return ExposureRgbStatus::ok;
}

ExposureRgbStatus expand_white(float value, const PreparedExposureRgb &prepared,
                               float *output) noexcept {
    const float z = value * prepared.positive_gain;
    const float z_plus_one = z + 1.0f;
    const float first = z / z_plus_one;
    const float second = z / prepared.white_clip_squared + 1.0f;
    const float result = (first * second) / prepared.positive_gain;
    if (!std::isfinite(z) || !std::isfinite(z_plus_one) ||
        !std::isfinite(first) || !std::isfinite(second) || !std::isfinite(result))
        return ExposureRgbStatus::overflow;
    *output = result;
    return ExposureRgbStatus::ok;
}

ExposureRgbStatus apply_whites_gray(float rgb[3],
                                   const PreparedExposureRgb &prepared) noexcept {
    // 00dc61d4..61f0: B + (G + (R + G)), then multiply by 1/4.
    const float rg = rgb[0] + rgb[1];
    const float rgg = rgb[1] + rg;
    const float gray = (rgb[2] + rgg) * 0.25f;
    const float coordinate = gray * prepared.whites_lut_scale;
    if (!std::isfinite(gray) || !std::isfinite(coordinate))
        return ExposureRgbStatus::overflow;
    if (coordinate < 0.0f || coordinate >= 0x1p31f)
        return ExposureRgbStatus::lut_out_of_bounds;
    const auto index = static_cast<std::uint32_t>(coordinate);
    if (static_cast<std::size_t>(index) + 1 >= prepared.whites_lut_size)
        return ExposureRgbStatus::lut_out_of_bounds;
    const float first = prepared.whites_lut[index];
    const float second = prepared.whites_lut[index + 1];
    if (!std::isfinite(first) || !std::isfinite(second) || first < 0 || second < 0)
        return ExposureRgbStatus::invalid_rgb;
    const float fraction = coordinate - static_cast<float>(index);
    const float next_part = fraction * second;
    const float mapped = std::fma(first, 1.0f - fraction, next_part);
    const float denominator = gray > prepared.epsilon ? gray : prepared.epsilon;
    const float gain = mapped / denominator;
    if (!std::isfinite(mapped) || !std::isfinite(gain) || gain < 0)
        return ExposureRgbStatus::overflow;
    for (std::size_t c = 0; c < 3; ++c) {
        rgb[c] *= gain;
        if (!std::isfinite(rgb[c])) return ExposureRgbStatus::overflow;
    }
    return ExposureRgbStatus::ok;
}

ExposureRgbStatus validate_pixels(const float *input, float *output,
                                  std::size_t pixel_count,
                                  const PreparedExposureRgb *prepared,
                                  std::size_t *element_count,
                                  AddressRange *input_range,
                                  AddressRange *output_range,
                                  AddressRange *prepared_range,
                                  AddressRange *lut_range) noexcept {
    if (prepared == nullptr || element_count == nullptr || input_range == nullptr ||
        output_range == nullptr || prepared_range == nullptr || lut_range == nullptr ||
        pixel_count > std::numeric_limits<std::size_t>::max() / 3) {
        return ExposureRgbStatus::invalid_argument;
    }
    *element_count = pixel_count * 3;
    if (*element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return ExposureRgbStatus::invalid_argument;
    }
    const std::size_t pixel_bytes = *element_count * sizeof(float);
    if (!make_range(input, pixel_bytes, alignof(float), input_range) ||
        !make_range(output, pixel_bytes, alignof(float), output_range) ||
        !make_range(prepared, sizeof(*prepared), alignof(PreparedExposureRgb),
                    prepared_range)) {
        return ExposureRgbStatus::invalid_argument;
    }
    if (prepared->lut_size > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return ExposureRgbStatus::invalid_argument;
    }
    const std::size_t lut_bytes = prepared->lut_size * sizeof(float);
    if (!make_range(prepared->lut, lut_bytes, alignof(float), lut_range)) {
        return ExposureRgbStatus::invalid_argument;
    }
    if ((prepared->lut == nullptr) != (prepared->lut_size == 0) ||
        (prepared->lut != nullptr && prepared->lut_size < 2)) {
        return ExposureRgbStatus::invalid_argument;
    }
    if (overlaps(*output_range, *input_range) && output != input) {
        return ExposureRgbStatus::invalid_argument;
    }
    if (overlaps(*output_range, *lut_range) ||
        overlaps(*output_range, *prepared_range)) {
        return ExposureRgbStatus::invalid_argument;
    }
    AddressRange whites_range;
    if (prepared->whites_lut_size > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        !make_range(prepared->whites_lut, prepared->whites_lut_size * sizeof(float),
                    alignof(float), &whites_range) ||
        (prepared->whites_lut == nullptr) != (prepared->whites_lut_size == 0) ||
        (prepared->whites_lut != nullptr && prepared->whites_lut_size < 2) ||
        overlaps(*output_range, whites_range)) return ExposureRgbStatus::invalid_argument;
    return ExposureRgbStatus::ok;
}

ExposureRgbStatus transform_pixel(const float *source, float *destination,
                                  const PreparedExposureRgb &prepared) noexcept {
    float rgb[3]{};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const float value = source[channel];
        if (!std::isfinite(value) || value < 0.0f) {
            return ExposureRgbStatus::invalid_rgb;
        }
        rgb[channel] = value * prepared.gain;
        if (!std::isfinite(rgb[channel])) return ExposureRgbStatus::overflow;
        if (rgb[channel] < 0.0f) return ExposureRgbStatus::invalid_rgb;
    }

    std::size_t minimum_index = 0;
    std::size_t middle_index = 0;
    std::size_t maximum_index = 0;
    sort_channels(rgb, &minimum_index, &middle_index, &maximum_index);
    float minimum = rgb[minimum_index];
    float middle = rgb[middle_index];
    float maximum = rgb[maximum_index];

    if (prepared.white_clip_squared < 1.0f) {
        // The achromatic path assigns all three channels to the same mapped
        // value. The chromatic path preserves the original extrema fraction.
        const bool achromatic = minimum == maximum;
        float hue = 0.0f;
        auto status = hue_fraction(minimum, middle, maximum, prepared.epsilon, &hue);
        if (status != ExposureRgbStatus::ok) return status;
        float mapped_minimum = 0.0f;
        float mapped_maximum = 0.0f;
        status = expand_white(minimum, prepared, &mapped_minimum);
        if (status != ExposureRgbStatus::ok) return status;
        status = expand_white(maximum, prepared, &mapped_maximum);
        if (status != ExposureRgbStatus::ok) return status;
        middle = achromatic ? mapped_minimum
                            : std::fma(hue, mapped_maximum - mapped_minimum,
                                       mapped_minimum);
        minimum = mapped_minimum;
        maximum = mapped_maximum;
        if (!std::isfinite(middle)) return ExposureRgbStatus::overflow;
    }

    if (maximum > 1.0f) {
        const auto status = apply_highlight_clip(prepared.epsilon, &minimum,
                                                 &middle, &maximum);
        if (status != ExposureRgbStatus::ok) return status;
    }

    if (prepared.lut != nullptr) {
        float hue = 0.0f;
        auto status = hue_fraction(minimum, middle, maximum, prepared.epsilon, &hue);
        if (status != ExposureRgbStatus::ok) return status;
        float mapped_minimum = 0.0f;
        float mapped_maximum = 0.0f;
        status = lookup(minimum, prepared, &mapped_minimum);
        if (status != ExposureRgbStatus::ok) return status;
        status = lookup(maximum, prepared, &mapped_maximum);
        if (status != ExposureRgbStatus::ok) return status;
        const float mapped_span = mapped_maximum - mapped_minimum;
        if (!std::isfinite(mapped_span)) return ExposureRgbStatus::overflow;
        const float mapped_offset = mapped_span * hue;
        if (!std::isfinite(mapped_offset)) return ExposureRgbStatus::overflow;
        const float mapped_middle = std::fma(mapped_span, hue, mapped_minimum);
        if (!std::isfinite(mapped_middle)) return ExposureRgbStatus::overflow;
        minimum = mapped_minimum;
        middle = mapped_middle;
        maximum = mapped_maximum;
    }

    if (prepared.rational_gain > 1.0f) {
        float hue = 0.0f;
        auto status = hue_fraction(minimum, middle, maximum, prepared.epsilon, &hue);
        if (status != ExposureRgbStatus::ok) return status;
        float mapped_minimum = 0.0f;
        float mapped_maximum = 0.0f;
        status = rational_curve(minimum, prepared.rational_gain, &mapped_minimum);
        if (status != ExposureRgbStatus::ok) return status;
        status = rational_curve(maximum, prepared.rational_gain, &mapped_maximum);
        if (status != ExposureRgbStatus::ok) return status;
        const float mapped_span = mapped_maximum - mapped_minimum;
        if (!std::isfinite(mapped_span)) return ExposureRgbStatus::overflow;
        const float mapped_offset = mapped_span * hue;
        if (!std::isfinite(mapped_offset)) return ExposureRgbStatus::overflow;
        const float mapped_middle = std::fma(mapped_span, hue, mapped_minimum);
        if (!std::isfinite(mapped_middle)) return ExposureRgbStatus::overflow;
        minimum = mapped_minimum;
        middle = mapped_middle;
        maximum = mapped_maximum;
    }

    if (!std::isfinite(minimum) || !std::isfinite(middle) ||
        !std::isfinite(maximum) || minimum < 0.0f || middle < 0.0f ||
        maximum < 0.0f) {
        return ExposureRgbStatus::invalid_rgb;
    }
    rgb[minimum_index] = minimum;
    rgb[middle_index] = middle;
    rgb[maximum_index] = maximum;
    if (prepared.whites_lut != nullptr) {
        const auto status = apply_whites_gray(rgb, prepared);
        if (status != ExposureRgbStatus::ok) return status;
    }
    destination[0] = rgb[0];
    destination[1] = rgb[1];
    destination[2] = rgb[2];
    return ExposureRgbStatus::ok;
}

}  // namespace

ExposureRgbStatus apply_prepared_exposure_rgb(
    const float *input, float *output, std::size_t pixel_count,
    const PreparedExposureRgb *prepared) noexcept {
    std::size_t element_count = 0;
    AddressRange input_range;
    AddressRange output_range;
    AddressRange prepared_range;
    AddressRange lut_range;
    const auto range_status = validate_pixels(
        input, output, pixel_count, prepared, &element_count, &input_range,
        &output_range, &prepared_range, &lut_range);
    if (range_status != ExposureRgbStatus::ok) return range_status;
    if (!valid_controls(*prepared)) return ExposureRgbStatus::invalid_argument;

    std::vector<float> temporary;
    try {
        temporary.resize(element_count);
    } catch (...) {
        return ExposureRgbStatus::allocation_failure;
    }
    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
        const auto status = transform_pixel(input + pixel * 3,
                                            temporary.data() + pixel * 3,
                                            *prepared);
        if (status != ExposureRgbStatus::ok) return status;
    }
    if (element_count != 0) {
        std::memcpy(output, temporary.data(), element_count * sizeof(float));
    }
    return ExposureRgbStatus::ok;
}

}  // namespace imprint::acr
