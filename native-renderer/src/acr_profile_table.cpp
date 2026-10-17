#include "acr_profile_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumEntries = 1'000'000;
constexpr std::size_t kMaximumPixels = std::size_t{1} << 26;
constexpr float kHueToSectors = 0x1.555556p-3f;  // bits 0x3e2aaaab
constexpr float kHueShiftScale = 0x1.111112p-6f; // bits 0x3c888889
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "the recovered kernel requires IEEE-754 binary32 float");
static_assert(kHueShiftScale == 1.0f / 60.0f,
              "the recovered hue shift scale is the float32 1/60 constant");

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

bool checked_product(std::size_t left, std::size_t right,
                     std::size_t *product) noexcept {
    if (product == nullptr ||
        (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

bool empty_curve(const ProfileCurveView &curve) noexcept {
    return curve.data == nullptr && curve.sample_count == 0 && curve.scale == 0.0f;
}

bool valid_curve(const ProfileCurveView &curve,
                 AddressRange *range) noexcept {
    if (curve.data == nullptr || curve.sample_count < 2 ||
        !std::isfinite(curve.scale) || curve.scale <= 0.0f ||
        !std::isfinite(std::floor(curve.scale)) ||
        std::floor(curve.scale) + 1.0f >=
            static_cast<double>(curve.sample_count)) {
        return false;
    }
    std::size_t bytes = 0;
    if (!checked_product(curve.sample_count, sizeof(float), &bytes) ||
        !make_range(curve.data, bytes, alignof(float), range)) {
        return false;
    }
    for (std::size_t i = 0; i < curve.sample_count; ++i) {
        if (!std::isfinite(curve.data[i]) || curve.data[i] < 0.0f ||
            curve.data[i] > 1.0f) {
            return false;
        }
    }
    return true;
}

bool valid_table(const ProfileTableView &table,
                 AddressRange *range) noexcept {
    if (table.data == nullptr || table.hue_count < 1 ||
        table.saturation_count < 2 || table.value_count < 1) {
        return false;
    }
    std::size_t entries = 0;
    std::size_t product = 0;
    if (!checked_product(table.hue_count, table.saturation_count, &product) ||
        !checked_product(product, table.value_count, &entries) ||
        entries == 0 || entries > kMaximumEntries || entries != table.count) {
        return false;
    }
    std::size_t float_count = 0;
    if (!checked_product(entries, 3, &float_count)) {
        return false;
    }
    std::size_t bytes = 0;
    if (!checked_product(float_count, sizeof(float), &bytes) ||
        !make_range(table.data, bytes, alignof(float), range)) {
        return false;
    }
    for (std::size_t entry = 0; entry < entries; ++entry) {
        const float hue = table.data[entry * 3];
        const float saturation = table.data[entry * 3 + 1];
        const float value = table.data[entry * 3 + 2];
        if (!std::isfinite(hue) || !std::isfinite(saturation) ||
            !std::isfinite(value) || hue < -360.0f || hue > 360.0f ||
            saturation < 0.0f || saturation > 16.0f || value < 0.0f ||
            value > 16.0f) {
            return false;
        }
    }
    return true;
}

float clip01(float value) noexcept {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

// Mirrors the prepared curve lookup used by the caller. The view has already
// been validated over [0,1], but retain the index guard for defensive safety.
bool curve_lookup(float value, const ProfileCurveView &curve,
                  float *result) noexcept {
    const float q = value * curve.scale;
    if (!std::isfinite(q) || q < 0.0f) return false;
    const float truncated = std::trunc(q);
    if (truncated < 0.0f ||
        truncated >= static_cast<float>(curve.sample_count - 1)) {
        return false;
    }
    const auto index = static_cast<std::size_t>(truncated);
    if (index + 1 >= curve.sample_count) return false;
    const float fraction = q - static_cast<float>(index);
    const float upper_term = fraction * curve.data[index + 1];
    *result = std::fma(1.0f - fraction, curve.data[index], upper_term);
    return std::isfinite(*result);
}

float blend(float low, float high, float t, float inverse_t) noexcept {
    const float high_term = t * high;
    return std::fma(inverse_t, low, high_term);
}

std::size_t table_offset(const ProfileTableView &table, std::size_t value,
                         std::size_t hue, std::size_t saturation) noexcept {
    return ((value * table.hue_count + hue) * table.saturation_count +
            saturation) * 3;
}

void hsv_from_rgb(float red, float green, float blue, float *hue,
                  float *saturation, float *value) noexcept {
    const float max_gb = green > blue ? green : blue;
    const float min_gb = green < blue ? green : blue;
    const float maximum = max_gb < red ? red : max_gb;
    const float minimum = min_gb > red ? red : min_gb;
    const float delta = maximum - minimum;
    float h = 0.0f;
    float s = 0.0f;
    if (delta > 0.0f) {
        if (maximum == red) {
            h = (green - blue) / delta;
            if (h < 0.0f) h += 6.0f;
        } else if (maximum == green) {
            h = (blue - red) / delta + 2.0f;
        } else {
            h = (red - green) / delta + 4.0f;
        }
        s = delta / maximum;
    }
    *hue = h;
    *saturation = s;
    *value = maximum;
}

bool sample_modifiers(const ProfileTableView &table, std::size_t h0,
                      std::size_t h1, float hf, float inverse_hf,
                      std::size_t s0, float sf, float inverse_sf,
                      std::size_t v0, std::size_t v1, float vf,
                      float inverse_vf, float modifiers[3]) noexcept {
    for (std::size_t channel = 0; channel < 3; ++channel) {
        auto at = [&](std::size_t value, std::size_t saturation,
                      std::size_t hue) noexcept {
            return table.data[table_offset(table, value, hue, saturation) +
                              channel];
        };

        auto hue_at = [&](std::size_t value,
                          std::size_t saturation) noexcept {
            return blend(at(value, saturation, h0),
                         at(value, saturation, h1), hf, inverse_hf);
        };

        const float low_sat_low_value = hue_at(v0, s0);
        float low_sat = low_sat_low_value;
        if (table.value_count > 1) {
            const float low_sat_high_value = hue_at(v1, s0);
            low_sat = blend(low_sat_low_value, low_sat_high_value, vf,
                            inverse_vf);
        }

        const float high_sat_low_value = hue_at(v0, s0 + 1);
        float high_sat = high_sat_low_value;
        if (table.value_count > 1) {
            const float high_sat_high_value = hue_at(v1, s0 + 1);
            high_sat = blend(high_sat_low_value, high_sat_high_value, vf,
                             inverse_vf);
        }
        modifiers[channel] = blend(low_sat, high_sat, sf, inverse_sf);
        if (!std::isfinite(modifiers[channel])) return false;
    }
    return true;
}

}  // namespace

ProfileTableStatus validate_profile_table_view(const ProfileTableView &table) noexcept {
    AddressRange range;
    return valid_table(table, &range) ? ProfileTableStatus::ok : ProfileTableStatus::invalid_table;
}

ProfileTableStatus apply_profile_table(
    const float *input_rgb, float *output_rgb, std::size_t pixel_count,
    const ProfileTableView &table, const ProfileCurveView &forward_curve,
    const ProfileCurveView &inverse_curve) noexcept {
    if (pixel_count > kMaximumPixels ||
        pixel_count > std::numeric_limits<std::size_t>::max() / 3) {
        return ProfileTableStatus::invalid_argument;
    }

    std::size_t pixel_elements = 0;
    std::size_t pixel_bytes = 0;
    if (!checked_product(pixel_count, 3, &pixel_elements) ||
        !checked_product(pixel_elements, sizeof(float), &pixel_bytes)) {
        return ProfileTableStatus::invalid_argument;
    }
    AddressRange input_range;
    AddressRange output_range;
    AddressRange table_range;
    AddressRange forward_range;
    AddressRange inverse_range;
    if (!make_range(input_rgb, pixel_bytes, alignof(float), &input_range) ||
        !make_range(output_rgb, pixel_bytes, alignof(float), &output_range)) {
        return ProfileTableStatus::invalid_argument;
    }
    if (!valid_table(table, &table_range))
        return ProfileTableStatus::invalid_table;
    if (overlaps(output_range, table_range))
        return ProfileTableStatus::invalid_argument;

    const bool no_forward = empty_curve(forward_curve);
    const bool no_inverse = empty_curve(inverse_curve);
    if (no_forward != no_inverse) return ProfileTableStatus::invalid_curve;
    const bool use_curves = !no_forward;
    if (use_curves &&
        (!valid_curve(forward_curve, &forward_range) ||
         !valid_curve(inverse_curve, &inverse_range))) {
        return ProfileTableStatus::invalid_curve;
    }
    if (use_curves &&
        (overlaps(output_range, forward_range) ||
         overlaps(output_range, inverse_range))) {
        return ProfileTableStatus::invalid_argument;
    }

    if (pixel_count == 0) return ProfileTableStatus::ok;
    if (overlaps(input_range, output_range))
        return ProfileTableStatus::invalid_argument;

    try {
        std::vector<float> candidate(pixel_elements);
        const float hue_scale = table.hue_count > 1
                                    ? static_cast<float>(table.hue_count) *
                                          kHueToSectors
                                    : 0.0f;
        const float saturation_scale =
            static_cast<float>(table.saturation_count - 1);
        const float value_scale = static_cast<float>(table.value_count - 1);

        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            const float red = input_rgb[pixel * 3];
            const float green = input_rgb[pixel * 3 + 1];
            const float blue = input_rgb[pixel * 3 + 2];
            if (!std::isfinite(red) || !std::isfinite(green) ||
                !std::isfinite(blue) || red < 0.0f || red > 1.0f ||
                green < 0.0f || green > 1.0f || blue < 0.0f || blue > 1.0f) {
                return ProfileTableStatus::invalid_rgb;
            }

            float hue = 0.0f;
            float saturation = 0.0f;
            float value = 0.0f;
            hsv_from_rgb(red, green, blue, &hue, &saturation, &value);

            if (use_curves && table.value_count > 1) {
                value = clip01(value);
                if (!curve_lookup(value, forward_curve, &value))
                    return ProfileTableStatus::invalid_curve;
            }

            const float hue_coordinate = hue_scale * hue;
            const float saturation_coordinate = saturation * saturation_scale;
            if (!std::isfinite(hue_coordinate) ||
                !std::isfinite(saturation_coordinate)) {
                return ProfileTableStatus::overflow;
            }
            const float hue_truncated = std::trunc(hue_coordinate);
            const float saturation_truncated =
                std::trunc(saturation_coordinate);
            if (hue_truncated < 0.0f || saturation_truncated < 0.0f ||
                hue_truncated > static_cast<float>(table.hue_count) ||
                saturation_truncated >
                    static_cast<float>(table.saturation_count)) {
                return ProfileTableStatus::overflow;
            }
            const auto hue_floor = static_cast<std::size_t>(hue_truncated);
            const auto saturation_floor =
                static_cast<std::size_t>(saturation_truncated);
            const std::size_t h0 =
                std::min(hue_floor, table.hue_count - 1);
            const std::size_t h1 = hue_floor < table.hue_count - 1
                                       ? hue_floor + 1
                                       : 0;
            const std::size_t s0 = std::min(saturation_floor,
                                            table.saturation_count - 2);
            const float hf = hue_coordinate - static_cast<float>(h0);
            const float sf = saturation_coordinate - static_cast<float>(s0);
            float v_fraction = 0.0f;
            std::size_t v0 = 0;
            std::size_t v1 = 0;
            if (table.value_count > 1) {
                const float value_coordinate = value * value_scale;
                if (!std::isfinite(value_coordinate) ||
                    value_coordinate < 0.0f) {
                    return ProfileTableStatus::overflow;
                }
                const float value_truncated = std::trunc(value_coordinate);
                if (value_truncated < 0.0f ||
                    value_truncated > static_cast<float>(table.value_count)) {
                    return ProfileTableStatus::overflow;
                }
                v0 = std::min(static_cast<std::size_t>(value_truncated),
                              table.value_count - 2);
                v1 = v0 + 1;
                v_fraction = value_coordinate - static_cast<float>(v0);
            }

            const float inverse_hf = 1.0f - hf;
            const float inverse_sf = 1.0f - sf;
            const float inverse_vf = 1.0f - v_fraction;
            float modifiers[3]{};
            if (!sample_modifiers(table, h0, h1, hf, inverse_hf, s0, sf,
                                  inverse_sf, v0, v1, v_fraction, inverse_vf,
                                  modifiers)) {
                return ProfileTableStatus::overflow;
            }

            const float multiplied_value = value * modifiers[2];
            const float mapped_value = clip01(multiplied_value);
            float output_value = mapped_value;
            if (use_curves && !curve_lookup(mapped_value, inverse_curve,
                                             &output_value)) {
                return ProfileTableStatus::invalid_curve;
            }

            float mapped_saturation = saturation * modifiers[1];
            if (mapped_saturation > 1.0f) mapped_saturation = 1.0f;
            if (!std::isfinite(mapped_saturation))
                return ProfileTableStatus::overflow;

            float out_red = output_value;
            float out_green = output_value;
            float out_blue = output_value;
            if (mapped_saturation > 0.0f) {
                const float hue_delta = modifiers[0] * kHueShiftScale;
                const float shifted_hue = hue + hue_delta;
                if (!std::isfinite(shifted_hue))
                    return ProfileTableStatus::overflow;
                float wrapped_hue = std::fmod(shifted_hue, 6.0f);
                if (wrapped_hue < 0.0f) wrapped_hue += 6.0f;
                const float sector_float = std::trunc(wrapped_hue);
                const int sector = static_cast<int>(sector_float);
                const float fraction = wrapped_hue - sector_float;
                const float one_minus_saturation = 1.0f - mapped_saturation;
                const float p = one_minus_saturation * output_value;
                const float q_factor =
                    std::fma(-mapped_saturation, fraction, 1.0f);
                const float q = q_factor * output_value;
                const float one_minus_fraction = 1.0f - fraction;
                const float t_factor =
                    std::fma(-mapped_saturation, one_minus_fraction, 1.0f);
                const float t = t_factor * output_value;
                switch (sector) {
                    case 1:
                        out_red = q;
                        out_green = output_value;
                        out_blue = p;
                        break;
                    case 2:
                        out_red = p;
                        out_green = output_value;
                        out_blue = t;
                        break;
                    case 3:
                        out_red = p;
                        out_green = q;
                        out_blue = output_value;
                        break;
                    case 4:
                        out_red = t;
                        out_green = p;
                        out_blue = output_value;
                        break;
                    case 5:
                        out_red = output_value;
                        out_green = p;
                        out_blue = q;
                        break;
                    case 0:
                    default:  // Sector 6 follows the recovered default path.
                        out_red = output_value;
                        out_green = t;
                        out_blue = p;
                        break;
                }
            }

            if (!std::isfinite(out_red) || !std::isfinite(out_green) ||
                !std::isfinite(out_blue)) {
                return ProfileTableStatus::overflow;
            }
            candidate[pixel * 3] = out_red;
            candidate[pixel * 3 + 1] = out_green;
            candidate[pixel * 3 + 2] = out_blue;
        }
        std::copy(candidate.begin(), candidate.end(), output_rgb);
    } catch (const std::bad_alloc &) {
        return ProfileTableStatus::allocation_failure;
    } catch (...) {
        return ProfileTableStatus::allocation_failure;
    }
    return ProfileTableStatus::ok;
}

}  // namespace imprint::acr
