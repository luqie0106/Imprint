#include "acr_tone_pixels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace imprint::acr {
namespace {

constexpr double kLn2 = 0.6931471805599453;
constexpr double kLowRange = 0x1p-10;
constexpr double kPixelEpsilonMax = 0x1p-12;
constexpr double kMinimumGrayMin = 0x1p-120;
constexpr double kMinimumGrayMax = 0x1p-16;
constexpr double kMaximumAbsoluteEv90 = 120.0;
constexpr double kGrayLogMax = 0x1p120;

void require_finite(double value, const char *message) {
    if (!std::isfinite(value)) throw std::invalid_argument(message);
}

void require_finite(float value, const char *message) {
    if (!std::isfinite(value)) throw std::invalid_argument(message);
}

float positive_float(double value, const char *message) {
    require_finite(value, message);
    if (value > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument(message);
    }
    const float result = static_cast<float>(value);
    if (!std::isfinite(result) || result <= 0.0f) {
        throw std::invalid_argument(message);
    }
    return result;
}

float checked_add(float lhs, float rhs, const char *message) {
    const float result = lhs + rhs;
    require_finite(result, message);
    return result;
}

float checked_mul(float lhs, float rhs, const char *message) {
    const float result = lhs * rhs;
    require_finite(result, message);
    return result;
}

float interpolate(const std::array<float, kTonePixelIntervals + 2> &table,
                  float coordinate) {
    require_finite(coordinate, "non-finite tone pixel LUT coordinate");
    if (coordinate < 0.0f || coordinate > static_cast<float>(kTonePixelIntervals)) {
        throw std::out_of_range("tone pixel LUT coordinate outside recovered endpoints");
    }
    const auto index = static_cast<std::size_t>(coordinate);
    const float current = table[index];
    const float next = table[index + 1];
    require_finite(current, "non-finite tone pixel LUT value");
    require_finite(next, "non-finite tone pixel LUT value");
    const float fraction = coordinate - static_cast<float>(index);
    const float difference = next - current;
    require_finite(difference, "tone pixel LUT difference overflow");
    const float adjustment = difference * fraction;
    require_finite(adjustment, "tone pixel LUT interpolation overflow");
    return checked_add(current, adjustment, "tone pixel LUT result overflow");
}

void validate_tables(const TonePixelTables &tables) {
    require_finite(tables.pixel_epsilon, "non-finite tone pixel epsilon");
    require_finite(tables.log_floor, "non-finite tone pixel log floor");
    require_finite(tables.inverse_log_floor,
                   "non-finite tone pixel inverse log floor");
    require_finite(tables.minimum_gray, "non-finite tone pixel minimum gray");
    if (tables.pixel_epsilon <= 0.0f ||
        tables.pixel_epsilon > static_cast<float>(kPixelEpsilonMax) ||
        tables.log_floor >= 0.0f || tables.inverse_log_floor <= 0.0f ||
        tables.minimum_gray < static_cast<float>(kMinimumGrayMin) ||
        tables.minimum_gray > static_cast<float>(kMinimumGrayMax)) {
        throw std::invalid_argument("invalid tone pixel table parameters");
    }
}

}  // namespace

TonePixelTables make_tone_pixel_tables(double source_epsilon, double ev90) {
    require_finite(source_epsilon, "non-finite selected source epsilon");
    require_finite(ev90, "non-finite EV90 for tone pixel tables");
    if (source_epsilon <= 0.0 || source_epsilon > kPixelEpsilonMax) {
        throw std::invalid_argument("selected source epsilon outside recovered range");
    }
    if (ev90 < -kMaximumAbsoluteEv90 || ev90 > kMaximumAbsoluteEv90) {
        throw std::invalid_argument("EV90 outside recovered tone pixel range");
    }

    const double exposure_scale = std::exp2(-ev90);
    require_finite(exposure_scale, "EV90 exposure scale is not finite");
    const double pixel_epsilon_double = std::clamp(
        source_epsilon * exposure_scale, 0.0, kPixelEpsilonMax);
    const float pixel_epsilon = positive_float(
        pixel_epsilon_double, "pixel epsilon is not representable as positive float32");

    const double base_minimum_gray = std::clamp(
        source_epsilon / 128.0, kMinimumGrayMin, kMinimumGrayMax);
    const double minimum_gray_double = std::clamp(
        base_minimum_gray * exposure_scale, kMinimumGrayMin, kMinimumGrayMax);
    const float minimum_gray = positive_float(
        minimum_gray_double, "minimum gray is not representable as positive float32");

    const double pixel_log = std::log2(pixel_epsilon_double);
    require_finite(pixel_log, "non-finite tone pixel epsilon logarithm");
    const double log_floor_double = std::log(pixel_epsilon_double) / kLn2;
    require_finite(log_floor_double, "non-finite tone pixel log floor");
    const float log_floor = static_cast<float>(log_floor_double);
    require_finite(log_floor, "tone pixel log floor is not representable as float32");
    if (log_floor == 0.0f) {
        throw std::invalid_argument("tone pixel log floor cannot be zero");
    }
    const float inverse_log_floor = -1.0f / log_floor;
    require_finite(inverse_log_floor,
                   "tone pixel inverse log floor is not representable as float32");

    TonePixelTables tables{};
    tables.pixel_epsilon = pixel_epsilon;
    tables.log_floor = log_floor;
    tables.inverse_log_floor = inverse_log_floor;
    tables.minimum_gray = minimum_gray;

    for (std::size_t i = 0; i <= kTonePixelIntervals; ++i) {
        const double fraction = static_cast<double>(i) /
                                static_cast<double>(kTonePixelIntervals);
        const double low_value = std::log2(
            pixel_epsilon_double + fraction * kLowRange);
        const double unit_value = std::log2(
            pixel_epsilon_double + kLowRange + fraction * (1.0 - kLowRange));
        const double inverse_value = std::exp2(
            pixel_log + fraction * (-pixel_log)) - pixel_epsilon_double;
        require_finite(low_value, "non-finite low tone pixel LUT sample");
        require_finite(unit_value, "non-finite unit tone pixel LUT sample");
        require_finite(inverse_value, "non-finite inverse tone pixel LUT sample");
        tables.log_low[i] = static_cast<float>(low_value);
        tables.log_unit[i] = static_cast<float>(unit_value);
        tables.inverse_negative[i] = static_cast<float>(inverse_value);
        require_finite(tables.log_low[i], "low tone pixel LUT sample exceeds float32");
        require_finite(tables.log_unit[i], "unit tone pixel LUT sample exceeds float32");
        require_finite(tables.inverse_negative[i],
                       "inverse tone pixel LUT sample exceeds float32");
    }
    tables.log_low[kTonePixelIntervals + 1] = tables.log_low[kTonePixelIntervals];
    tables.log_unit[kTonePixelIntervals + 1] = tables.log_unit[kTonePixelIntervals];
    tables.inverse_negative[kTonePixelIntervals + 1] =
        tables.inverse_negative[kTonePixelIntervals];
    return tables;
}

std::array<float, 3> tone_rgb_pixel(const TonePixelTables &tables,
                                   const ToneMapping &model,
                                   std::array<float, 3> rgb, float delta,
                                   float local3, float local4, float local5,
                                   ToneOutputMode output_mode) {
    if (output_mode != ToneOutputMode::preserve_rgb &&
        output_mode != ToneOutputMode::normalized_gray) {
        throw std::invalid_argument("invalid tone pixel output mode");
    }
    validate_tables(tables);
    for (const float channel : rgb) {
        require_finite(channel, "non-finite tone pixel RGB input");
        if (channel < 0.0f) {
            throw std::invalid_argument("tone pixel RGB input must be nonnegative");
        }
    }

    // Preserve the recovered scalar evaluation order and do not use fma().
    const float green = checked_mul(rgb[1], 0.59f, "tone pixel gray overflow");
    const float red = checked_mul(rgb[0], 0.3f, "tone pixel gray overflow");
    const float green_red = checked_add(green, red, "tone pixel gray overflow");
    const float blue = checked_mul(rgb[2], 0.110000014f,
                                   "tone pixel gray overflow");
    const float gray = checked_add(green_red, blue, "tone pixel gray overflow");

    float log_gray = 0.0f;
    if (gray <= static_cast<float>(kLowRange)) {
        const float scaled = checked_mul(gray, 1024.0f,
                                         "low tone pixel coordinate overflow");
        const float coordinate = checked_mul(scaled, 4096.0f,
                                              "low tone pixel coordinate overflow");
        log_gray = interpolate(tables.log_low, coordinate);
    } else if (gray <= 1.0f) {
        const float scaled = checked_mul(gray, 1.0009775f,
                                         "unit tone pixel coordinate overflow");
        const float shifted = scaled - 0.0009775171f;
        require_finite(shifted, "unit tone pixel coordinate overflow");
        const float coordinate = checked_mul(shifted, 4096.0f,
                                              "unit tone pixel coordinate overflow");
        log_gray = interpolate(tables.log_unit, coordinate);
    } else {
        const float bounded_gray = std::clamp(gray, 0.0f,
                                              static_cast<float>(kGrayLogMax));
        const float logarithm_input = checked_add(
            bounded_gray, tables.pixel_epsilon, "tone pixel log input overflow");
        log_gray = ::log2f(logarithm_input);
        require_finite(log_gray, "non-finite tone pixel input logarithm");
    }

    const float mapped = evaluate_tone_mapping(model, log_gray, delta,
                                               local3, local4, local5);
    float linear = 0.0f;
    if (mapped <= tables.log_floor) {
        linear = 0.0f;
    } else if (mapped < 0.0f) {
        const float scaled = checked_mul(tables.inverse_log_floor, mapped,
                                         "inverse tone pixel coordinate overflow");
        const float normalized = checked_add(1.0f, scaled,
                                             "inverse tone pixel coordinate overflow");
        const float coordinate = checked_mul(normalized, 4096.0f,
                                              "inverse tone pixel coordinate overflow");
        linear = interpolate(tables.inverse_negative, coordinate);
    } else {
        const float bounded = std::clamp(mapped, -120.0f, 120.0f);
        const float expanded = ::exp2f(bounded);
        require_finite(expanded, "non-finite tone pixel exp2 result");
        linear = expanded - tables.pixel_epsilon;
        require_finite(linear, "tone pixel linear output overflow");
    }

    if (output_mode == ToneOutputMode::normalized_gray) {
        const float gray = std::clamp(linear, 0.0f, 1.0f);
        return {{gray, gray, gray}};
    }

    const float denominator = std::max(gray, tables.minimum_gray);
    const float gain = linear / denominator;
    require_finite(gain, "tone pixel gain overflow");
    std::array<float, 3> output{};
    for (std::size_t i = 0; i < output.size(); ++i) {
        const float channel = checked_mul(rgb[i], gain, "tone pixel channel overflow");
        output[i] = std::max(channel, 0.0f);
    }
    return output;
}

}  // namespace imprint::acr
