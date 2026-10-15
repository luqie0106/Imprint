#include "acr_color_kernels.hpp"
#include "acr_default_tone_tables.hpp"
#include "acr_white_temperature_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::array<double, 8> kHueCenters = {0.0, 30.0, 55.0, 95.0,
                                                160.0, 230.0, 275.0, 330.0};

void require_finite_range(double value, double low, double high,
                          const char *what) {
    if (!std::isfinite(value) || value < low || value > high) {
        throw std::invalid_argument(what);
    }
}

std::size_t rgb_value_count(std::size_t pixel_count) {
    if (pixel_count > std::numeric_limits<std::size_t>::max() / (3 * sizeof(float))) {
        throw std::invalid_argument("pixel count overflows RGB extent");
    }
    return pixel_count * 3;
}

bool ranges_overlap(const void *left, const void *right, std::size_t bytes) {
    const auto a = reinterpret_cast<std::uintptr_t>(left);
    const auto b = reinterpret_cast<std::uintptr_t>(right);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - a ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - b) {
        return true;
    }
    return a < b + bytes && b < a + bytes;
}

void validate_buffers(const float *rgb, float *output, std::size_t pixel_count) {
    const std::size_t values = rgb_value_count(pixel_count);
    if (values == 0) return;
    if (!rgb || !output) throw std::invalid_argument("null RGB buffer");
    const std::size_t bytes = values * sizeof(float);
    if (rgb != output && ranges_overlap(rgb, output, bytes)) {
        throw std::invalid_argument("partially overlapping RGB buffers");
    }
    for (std::size_t i = 0; i < values; ++i) {
        if (!std::isfinite(rgb[i]) || rgb[i] < 0.0f || rgb[i] > 1.0f) {
            throw std::invalid_argument("input RGB must be finite and in [0,1]");
        }
    }
}

void copy_if_distinct(const float *rgb, float *output, std::size_t values) {
    if (values != 0 && rgb != output) {
        std::memcpy(output, rgb, values * sizeof(float));
    }
}

double clamp01(double value) { return std::min(1.0, std::max(0.0, value)); }

double positive_mod(double value, double modulus) {
    double result = std::fmod(value, modulus);
    if (result < 0.0) result += modulus;
    return result;
}

std::vector<double> solve_tridiagonal(const std::vector<double> &lower,
                                      const std::vector<double> &diagonal,
                                      const std::vector<double> &upper,
                                      const std::vector<double> &rhs) {
    const std::size_t count = diagonal.size();
    std::vector<double> c = upper;
    std::vector<double> d = rhs;
    std::vector<double> b = diagonal;
    for (std::size_t i = 1; i < count; ++i) {
        const double factor = lower[i] / b[i - 1];
        b[i] -= factor * c[i - 1];
        d[i] -= factor * d[i - 1];
    }
    std::vector<double> result(count);
    result[count - 1] = d[count - 1] / b[count - 1];
    for (std::size_t i = count - 1; i-- > 0;) {
        result[i] = (d[i] - c[i] * result[i + 1]) / b[i];
    }
    return result;
}

std::array<float, 361> build_hue_table(const std::array<double, 8> &controls,
                                      bool hue_offsets) {
    constexpr std::size_t kKnotCount = 40;
    std::array<double, kKnotCount> x{};
    std::array<double, kKnotCount> y{};
    std::array<double, 8> knot_values{};
    for (std::size_t i = 0; i < 8; ++i) {
        if (!hue_offsets) {
            knot_values[i] = controls[i] / 100.0;
            continue;
        }
        const double current = kHueCenters[i];
        const double following = (i == 7) ? kHueCenters[0] + 360.0
                                           : kHueCenters[i + 1];
        const double previous = (i == 0) ? kHueCenters[7] - 360.0
                                           : kHueCenters[i - 1];
        const double positive_gap = following - current;
        const double negative_gap = current - previous;
        const double level = controls[i];
        // Adobe's sector displacement uses distinct neighboring gaps for the
        // positive and negative directions, with a 30-degree floor.
        const double gap = level >= 0.0 ? positive_gap : negative_gap;
        knot_values[i] = (level / 100.0) * std::max(30.0, 0.9 * gap);
    }

    for (int period = -1; period <= 3; ++period) {
        for (std::size_t band = 0; band < 8; ++band) {
            const std::size_t index = static_cast<std::size_t>(period + 1) * 8 + band;
            x[index] = kHueCenters[band] + static_cast<double>(period) * 360.0;
            y[index] = knot_values[band];
        }
    }

    std::array<double, kKnotCount - 1> dx{};
    std::array<double, kKnotCount - 1> secants{};
    for (std::size_t i = 0; i < kKnotCount - 1; ++i) {
        dx[i] = x[i + 1] - x[i];
        secants[i] = (y[i + 1] - y[i]) / dx[i];
    }
    std::array<double, kKnotCount> initial{};
    for (std::size_t i = 1; i < kKnotCount - 1; ++i) {
        const double dx_previous = dx[i - 1];
        const double dx_next = dx[i];
        initial[i] = (dx_previous * secants[i] + dx_next * secants[i - 1]) /
                     (dx_previous + dx_next);
    }
    initial[0] = 2.0 * secants[0] - initial[1];
    initial[kKnotCount - 1] = 2.0 * secants[kKnotCount - 2] - initial[kKnotCount - 2];

    std::vector<double> lower(kKnotCount, 0.0), diagonal(kKnotCount, 1.0),
        upper(kKnotCount, 0.0), rhs(kKnotCount, 0.0);
    upper[0] = 0.5;
    rhs[0] = 0.75 * (initial[0] + initial[1]);
    lower[kKnotCount - 1] = 0.5;
    rhs[kKnotCount - 1] = 0.75 * (initial[kKnotCount - 2] + initial[kKnotCount - 1]);
    for (std::size_t i = 1; i < kKnotCount - 1; ++i) {
        const double dx_previous = dx[i - 1];
        const double dx_next = dx[i];
        const double denominator = 2.0 * (dx_previous + dx_next);
        lower[i] = dx_next / denominator;
        upper[i] = dx_previous / denominator;
        rhs[i] = 1.5 * initial[i];
    }
    std::vector<double> slopes = solve_tridiagonal(lower, diagonal, upper, rhs);

    for (std::size_t i = 0; i < kKnotCount - 1; ++i) {
        const double secant = secants[i];
        if (secant > 0.0) {
            slopes[i] = std::max(0.0, slopes[i]);
            slopes[i + 1] = std::max(0.0, slopes[i + 1]);
        } else if (secant < 0.0) {
            slopes[i] = std::min(0.0, slopes[i]);
            slopes[i + 1] = std::min(0.0, slopes[i + 1]);
        } else {
            slopes[i] = 0.0;
            slopes[i + 1] = 0.0;
        }
        if (secant != 0.0) {
            const double alpha = slopes[i] / secant;
            const double beta = slopes[i + 1] / secant;
            const double magnitude_squared = alpha * alpha + beta * beta;
            if (magnitude_squared > 9.0) {
                const double factor = 3.0 / std::sqrt(magnitude_squared);
                slopes[i] *= factor;
                slopes[i + 1] *= factor;
            }
        }
    }

    std::array<double, kKnotCount> periodic_slopes{};
    for (std::size_t i = 0; i < kKnotCount; ++i) {
        periodic_slopes[i] = slopes[16 + (i % 8)];
    }

    std::array<float, 361> table{};
    for (int query = 0; query <= 360; ++query) {
        std::size_t right = 0;
        while (right < kKnotCount && x[right] <= static_cast<double>(query)) ++right;
        right = std::max<std::size_t>(1, std::min(kKnotCount - 1, right));
        const std::size_t left = right - 1;
        const double h = x[right] - x[left];
        const double t = (static_cast<double>(query) - x[left]) / h;
        const double u = 1.0 - t;
        double value = u * u * (1.0 + 2.0 * t) * y[left] +
                       u * u * t * h * periodic_slopes[left] +
                       t * t * (3.0 - 2.0 * t) * y[right] -
                       t * t * u * h * periodic_slopes[right];
        table[static_cast<std::size_t>(query)] = static_cast<float>(value);
    }
    table[360] = table[0];
    return table;
}

double lookup_table(const std::array<float, 361> &table, double hue_degrees) {
    const double hue = positive_mod(hue_degrees, 360.0);
    const auto left = static_cast<std::size_t>(std::floor(hue));
    const double fraction = hue - static_cast<double>(left);
    return static_cast<double>(table[left]) + fraction *
           (static_cast<double>(table[left + 1]) - static_cast<double>(table[left]));
}

struct HslExtrema {
    double hue;
    double minimum;
    double maximum;
};

HslExtrema rgb_to_hue_extrema(double red, double green, double blue) {
    const double maximum = std::max(red, std::max(green, blue));
    const double minimum = std::min(red, std::min(green, blue));
    const double delta = maximum - minimum;
    double hue = 0.0;
    if (delta > 1.0e-12) {
        if (red == maximum) {
            hue = positive_mod((green - blue) / delta, 6.0) * 60.0;
        } else if (green == maximum) {
            hue = ((blue - red) / delta + 2.0) * 60.0;
        } else {
            hue = ((red - green) / delta + 4.0) * 60.0;
        }
    }
    if (delta <= 1.0e-12) return {hue, minimum, maximum};

    const double middle = std::max(std::min(red, green),
                                   std::min(std::max(red, green), blue));
    const double t = (middle - minimum) / delta;
    const double modern_maximum = maximum + (middle - maximum) * t * t * 0.5;
    const double inverse_t = 1.0 - t;
    const double modern_minimum = minimum + (middle - minimum) * inverse_t * inverse_t * 0.5;
    return {hue, modern_minimum, modern_maximum};
}

void reconstruct_rgb(double hue, double minimum, double maximum,
                     double *red, double *green, double *blue) {
    const double h = positive_mod(hue, 360.0) / 60.0;
    const int sector = static_cast<int>(std::floor(h)) % 6;
    const double parity = positive_mod(h, 2.0);
    const double t = 1.0 - std::abs(parity - 1.0);
    const double denominator = t * t - t + 2.0;
    const double old_minimum = minimum;
    const double old_maximum = maximum;
    minimum = (old_minimum * (t * t * t - t * t + 2.0) -
               old_maximum * t * (t - 1.0) * (t - 1.0)) / denominator;
    maximum = (old_minimum * (t - 1.0) * t * t -
               old_maximum * (t + t * t * t - 2.0 * t * t - 2.0)) / denominator;
    minimum = clamp01(minimum);
    maximum = clamp01(maximum);
    const double delta = maximum - minimum;
    const double secondary = delta * (1.0 - std::abs(parity - 1.0));
    switch (sector) {
        case 0: *red = delta;     *green = secondary; *blue = 0.0; break;
        case 1: *red = secondary; *green = delta;     *blue = 0.0; break;
        case 2: *red = 0.0;       *green = delta;     *blue = secondary; break;
        case 3: *red = 0.0;       *green = secondary; *blue = delta; break;
        case 4: *red = secondary; *green = 0.0;       *blue = delta; break;
        default:*red = delta;     *green = 0.0;       *blue = secondary; break;
    }
    *red += minimum;
    *green += minimum;
    *blue += minimum;
}

bool hsl_is_neutral(const HslParams &params) {
    if (params.vibrance != 0.0) return false;
    for (std::size_t i = 0; i < 8; ++i) {
        if (params.hue[i] != 0.0 || params.saturation[i] != 0.0 ||
            params.luminance[i] != 0.0) return false;
    }
    return true;
}

void apply_default_rgb_tone(const float *rgb, float *output, std::size_t pixels,
                            const std::array<float, 1025> &table) {
    auto lookup = [&](double x) {
        const double coordinate = x * 1024.0;
        const auto index = std::min<std::size_t>(1023, static_cast<std::size_t>(coordinate));
        const double fraction = coordinate - static_cast<double>(index);
        return (1.0-fraction) * table[index] + fraction * table[index+1];
    };
    for (std::size_t i = 0; i < pixels; ++i) {
        const auto source = rgb_to_hue_extrema(rgb[3*i], rgb[3*i+1], rgb[3*i+2]);
        double red, green, blue;
        reconstruct_rgb(source.hue, lookup(source.minimum), lookup(source.maximum),
                        &red, &green, &blue);
        output[3*i] = static_cast<float>(red);
        output[3*i+1] = static_cast<float>(green);
        output[3*i+2] = static_cast<float>(blue);
    }
}

void validate_hsl_params(const HslParams &params) {
    require_finite_range(params.vibrance, -100.0, 100.0, "invalid HSL vibrance control");
    for (std::size_t i = 0; i < 8; ++i) {
        require_finite_range(params.hue[i], -100.0, 100.0, "invalid HSL hue control");
        require_finite_range(params.saturation[i], -100.0, 100.0,
                             "invalid HSL saturation control");
        require_finite_range(params.luminance[i], -100.0, 100.0,
                             "invalid HSL luminance control");
    }
}

void apply_vibrance(double hue_degrees, double ui_vibrance,
                    double *minimum, double *maximum) {
    const double original_minimum = *minimum;
    const double original_maximum = *maximum;
    if (original_maximum - original_minimum == 0.0 || original_maximum == 0.0) return;

    const double saturation = (original_maximum - original_minimum) / original_maximum;
    const double strength = std::clamp(ui_vibrance / 100.0, -1.0, 1.0);
    double q = std::min(16.0 * original_maximum, 1.0);
    q *= 2.0 - q;
    const double k = saturation * (1.0 - saturation);

    if (strength > 0.0) {
        const double hue_sector = positive_mod(hue_degrees / 60.0, 6.0);
        const double skin_hue = positive_mod(hue_sector + 1.0, 6.0);
        double skin = std::min(clamp01(skin_hue),
                               clamp01(1.0 - 4.0 * (skin_hue - 1.5)));
        skin *= 1.0 - saturation * saturation;
        const double vibrance_gain = ((skin * strength + 1.0) - skin) * strength;
        const double new_maximum = original_maximum *
            (1.0 + std::max(1.0 - original_maximum, 0.0) * k * (2.0 - k) *
             q * vibrance_gain * 0.25);
        // Keep the source's rounded float literals from FUN_01354258.
        const double sat_factor = static_cast<double>(0.8333333f) -
                                  skin * static_cast<double>(0.4047619f);
        const double denominator = 1.0 - (1.0 - saturation) * q *
                                   (1.0 - original_minimum) * sat_factor * vibrance_gain;
        *maximum = new_maximum;
        *minimum = new_maximum - new_maximum * saturation / denominator;
        return;
    }

    if (strength < 0.0) {
        const double new_maximum = original_maximum *
            (1.0 + std::max(1.0 - original_maximum, 0.0) * q * strength * k * (2.0 - k));
        const double new_minimum = new_maximum - new_maximum * (1.0 + strength * 0.25) *
            saturation * (strength + 1.0 - q * saturation * strength *
                          (saturation * 0.5 + 0.5));
        *maximum = new_maximum;
        *minimum = new_minimum;
    }
}

double srgb_decode(double value) {
    if (value > 0.040449936) {
        const double shifted = std::max(value + 0.055, 0.0) * (1.0 / 1.055);
        return std::pow(shifted, 2.4);
    }
    return value / 12.92;
}

double srgb_encode(double value) {
    if (value > 0.0031308) {
        return 1.055 * std::pow(std::max(value, 0.0), 1.0 / 2.4) - 0.055;
    }
    return 12.92 * value;
}

double factor(double midpoint) { return (1.0 - midpoint) / midpoint; }

double redistributive_curve(double value, double a) {
    return a * value / (a * value + 1.0 - value);
}

double inverse_redistributive_curve(double value, double a) {
    const double scaled = value / a;
    return scaled / (scaled + 1.0 - value);
}

double blend_curve(double x, double shadow, double highlight, double blending,
                   double weight_source) {
    const double w1 = 1.0 - std::pow(2.0 * weight_source - 1.0, 2.0);
    const double w2 = w1 * w1;
    const double full = x * (x * x + (shadow + x * (highlight - shadow)) * (1.0 - x));
    const double left = x * (x * x + (shadow + x * (2.0 - shadow)) * (1.0 - x));
    const double right = x * (x * x + (1.0 + x * (highlight - 1.0)) * (1.0 - x));
    double partial = weight_source <= 0.5 ? left : right;
    partial += (x - partial) * w2;
    return partial + (full - partial) * blending;
}

std::array<double, 3> hue_coefficients(double hue, double saturation,
                                       double low, double high, int flag) {
    const double h = positive_mod(hue, 360.0);
    double delta = (high - low) * saturation;
    if (flag != 0) delta = -delta;
    auto base = [=](double weight) {
        double value = 1.0 - weight * delta;
        value = std::min(high, value);
        value = std::max(low, value);
        value = std::min(value, high - delta);
        value = std::max(low - delta, value);
        return value;
    };
    if (h < 60.0) {
        const double t = h / 60.0;
        const double k = t * 0.7119140625 + 0.2880859375;
        const double b = base(k);
        return {delta + b, b + t * delta, b};
    }
    if (h < 120.0) {
        const double t = (120.0 - h) / 60.0;
        const double k = t * 0.2880859375 + 0.7119140625;
        const double b = base(k);
        return {b + t * delta, delta + b, b};
    }
    if (h < 180.0) {
        const double t = (h - 120.0) / 60.0;
        const double b = base(0.7119140625);
        return {b, delta + b, b + t * delta};
    }
    if (h < 240.0) {
        const double t = (240.0 - h) / 60.0;
        const double b = base(t * 0.7119140625);
        return {b, b + t * delta, delta + b};
    }
    if (h < 300.0) {
        const double t = (h - 240.0) / 60.0;
        const double b = base(t * 0.2880859375);
        return {b + t * delta, b, delta + b};
    }
    const double t = (360.0 - h) / 60.0;
    const double b = base(0.2880859375);
    return {delta + b, b, b + t * delta};
}

std::array<double, 3> wheel_parameters(const GradingWheel &wheel,
                                       double saturation_scale,
                                       int kind) {
    const double saturation = wheel.saturation * saturation_scale;
    if (kind == 0) return hue_coefficients(wheel.hue, saturation / 100.0, 0.05, 2.5, 0);
    if (kind == 1) {
        auto result = hue_coefficients(wheel.hue, saturation / 100.0, 0.2, 2.5, 1);
        for (double &component : result) component = 3.0 - component;
        return result;
    }
    auto result = hue_coefficients(wheel.hue, 1.0, -1.0, 1.0, 0);
    for (double &component : result) component *= saturation / 100.0;
    return result;
}

bool all_grading_chroma_neutral(const GradingParams &params) {
    return params.shadows.saturation == 0.0 && params.midtones.saturation == 0.0 &&
           params.highlights.saturation == 0.0 && params.global.saturation == 0.0;
}

bool grading_is_neutral(const GradingParams &params) {
    return all_grading_chroma_neutral(params) &&
           params.shadows.luminance == 0.0 && params.midtones.luminance == 0.0 &&
           params.highlights.luminance == 0.0 && params.global.luminance == 0.0;
}

void validate_wheel(const GradingWheel &wheel) {
    require_finite_range(wheel.hue, 0.0, 360.0, "invalid grading hue");
    require_finite_range(wheel.saturation, 0.0, 100.0, "invalid grading saturation");
    require_finite_range(wheel.luminance, -100.0, 100.0, "invalid grading luminance");
}

void validate_grading_params(const GradingParams &params) {
    validate_wheel(params.shadows);
    validate_wheel(params.midtones);
    validate_wheel(params.highlights);
    validate_wheel(params.global);
    require_finite_range(params.blend, 0.0, 100.0, "invalid grading blend");
    require_finite_range(params.balance, -100.0, 100.0, "invalid grading balance");
}

double apply_grading_channel(double source, const GradingParams &params,
                             const std::array<double, 3> &shadow,
                             const std::array<double, 3> &mid,
                             const std::array<double, 3> &highlight,
                             const std::array<double, 3> &global_color,
                             std::size_t channel,
                             double a, double blending, double balance_damping) {
    double x0 = redistributive_curve(source, a);
    double x = x0;
    double shadow_lum = params.shadows.luminance;
    double highlight_lum = params.highlights.luminance;
    if (shadow_lum > 0.0) {
        shadow_lum *= 0.7;
        if (params.balance > 0.0) shadow_lum *= balance_damping;
    }
    if (highlight_lum < 0.0) {
        highlight_lum *= 0.7;
        if (params.balance < 0.0) highlight_lum *= balance_damping;
    }
    const double lum_shadow = 1.0 + shadow_lum * 0.01;
    const double lum_highlight = 2.0 + highlight_lum * 0.01;
    if (lum_shadow != 1.0 || lum_highlight != 2.0) {
        x = blend_curve(x, lum_shadow, lum_highlight, blending, x);
        double encoded = srgb_encode(inverse_redistributive_curve(x, a));
        if (lum_shadow > 1.0) {
            const double cubic = ((encoded * 0.72338 - 4.95264e-16) * encoded + 0.5) * encoded + 0.16;
            const double shoulder = encoded < 0.48 ? cubic : encoded;
            encoded += (shoulder - encoded) * (lum_shadow - 1.0);
        }
        if (lum_highlight < 2.0) {
            const double cubic = ((encoded * 0.72338 - 2.17014) * encoded + 2.67014) * encoded - 0.38338;
            const double shoulder = encoded > 0.52 ? cubic : encoded;
            encoded += (shoulder - encoded) * std::abs(lum_highlight - 2.0);
        }
        x = redistributive_curve(srgb_decode(encoded), a);
    }

    const double shadow_component = shadow[channel];
    const double mid_component = mid[channel];
    const double highlight_component = highlight[channel];
    const double global_component = global_color[channel];
    double y = x * (x * x + (shadow_component +
                  x * (highlight_component - shadow_component)) * (1.0 - x));
    if (shadow_component != 1.0 || highlight_component != 2.0) {
        // The original pre-grading x0 drives the chromatic blend weight.
        y = blend_curve(x, shadow_component, highlight_component, blending, x0);
    }
    const double w1 = 1.0 - std::pow(2.0 * x0 - 1.0, 2.0);
    const double w2 = w1 * w1;
    const double weight = w2 + (w1 - w2) * blending;
    if (mid_component != 0.0) {
        const double z = redistributive_curve(y, factor(0.5 - 0.14 * mid_component));
        y += (z - y) * weight;
    }
    if (params.midtones.luminance != 0.0) {
        const double z = redistributive_curve(y, factor(0.5 - 0.12 * params.midtones.luminance * 0.01));
        y += (z - y) * weight;
    }
    y = inverse_redistributive_curve(y, a);
    if (global_component != 0.0 || params.global.luminance != 0.0) {
        const double a0 = factor(srgb_decode(0.5));
        y = redistributive_curve(y, a0);
        if (global_component != 0.0) {
            y = redistributive_curve(y, factor(0.5 - 0.14 * global_component));
        }
        if (params.global.luminance != 0.0) {
            y = redistributive_curve(y, factor(0.5 - 0.12 * params.global.luminance * 0.01));
        }
        y = inverse_redistributive_curve(y, a0);
    }
    return y;
}

}  // namespace

WhiteTemperature white_temperature(double x, double y) {
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0 || y <= 0.0 || x+y >= 1.0) {
        throw std::invalid_argument("invalid white chromaticity");
    }
    const double denominator = 1.5 - x + 6.0*y;
    const double u = 2.0*x/denominator, v = 3.0*y/denominator;
    double previous_distance = 0.0, previous_a = 0.0, previous_b = 0.0;
    for (std::size_t j = 1; j < kWhiteTemperatureTable.size(); ++j) {
        const auto &row = kWhiteTemperatureTable[j];
        double a = 1.0/std::sqrt(1.0+row[3]*row[3]);
        double b = row[3]*a;
        const double distance = (v-row[2])*a-b*(u-row[1]);
        if (distance <= 0.0 || j+1 == kWhiteTemperatureTable.size()) {
            const auto &previous = kWhiteTemperatureTable[j-1];
            const double clipped_distance = std::min(distance, 0.0);
            const double f = j == 1 ? 0.0 : -clipped_distance/(previous_distance-clipped_distance);
            const double q = 1.0-f;
            const double temperature = 1.0e6/(q*row[0]+f*previous[0]);
            a = q*a+f*previous_a;
            b = q*b+f*previous_b;
            const double norm = std::hypot(a,b);
            const double tint = -3000.0*(b/norm*(v-(q*row[2]+f*previous[2]))+
                                         a/norm*(u-(q*row[1]+f*previous[1])));
            if (!std::isfinite(temperature) || !std::isfinite(tint) || temperature <= 0.0) {
                throw std::invalid_argument("white projection is not finite");
            }
            return {temperature, tint};
        }
        previous_distance = distance;
        previous_a = a;
        previous_b = b;
    }
    throw std::invalid_argument("white projection failed");
}

void apply_saturation_s16(const std::int16_t *rgb, std::int16_t *output,
                          std::size_t pixel_count, double ui_saturation) {
    require_finite_range(ui_saturation, -100.0, 100.0, "invalid basic saturation");
    const std::size_t values = rgb_value_count(pixel_count);
    if (!values) return;
    if (!rgb || !output || (rgb != output &&
        ranges_overlap(rgb, output, values * sizeof(std::int16_t)))) {
        throw std::invalid_argument("invalid signed-16 RGB buffers");
    }
    if (ui_saturation == 0.0) {
        if (rgb != output) std::memcpy(output, rgb, values * sizeof(std::int16_t));
        return;
    }
    // Floor division reproduces ARM ASR without implementation-defined
    // signed right shifts on a C++17 compiler.
    auto floor_shift = [](std::int64_t n, unsigned bits) {
        const std::int64_t divisor = std::int64_t{1} << bits;
        return n >= 0 ? n / divisor : -((-n + divisor - 1) / divisor);
    };
    auto clip = [](std::int64_t n) {
        return std::clamp(n, std::int64_t{-32768}, std::int64_t{32767});
    };
    const double scale = (ui_saturation + 100.0) * 0.01;
    const auto kr = static_cast<std::int64_t>(scale * 0.7023319615912208 * 4096.0 + 0.5);
    const auto kb = static_cast<std::int64_t>(scale * 0.25 * 4096.0 + 0.5);
    for (std::size_t i = 0; i < values; i += 3) {
        const std::int64_t r = rgb[i], g = rgb[i+1], b = rgb[i+2];
        const auto y = floor_shift(r * 0x127 + g * 0x2d9 + 0x200, 10);
        const auto cr = clip(floor_shift((r-y) * kr + 0x800, 12));
        const auto cb = clip(floor_shift((b-y) * kb + 0x800, 12));
        output[i] = static_cast<std::int16_t>(clip(y + floor_shift(cr * 0x16c8 + 0x800, 12)));
        output[i+1] = static_cast<std::int16_t>(clip(y + floor_shift(cr * -0x938 + 0x800, 12)));
        output[i+2] = static_cast<std::int16_t>(clip(y + cb * 4));
    }
}

void apply_hsl(const float *rgb, float *output, std::size_t pixel_count,
               const HslParams &params) {
    validate_hsl_params(params);
    const std::size_t values = rgb_value_count(pixel_count);
    validate_buffers(rgb, output, pixel_count);
    if (hsl_is_neutral(params)) {
        copy_if_distinct(rgb, output, values);
        return;
    }

    const auto hue_table = build_hue_table(params.hue, true);
    const auto saturation_table = build_hue_table(params.saturation, false);
    const auto luminance_table = build_hue_table(params.luminance, false);

    for (std::size_t i = 0; i < pixel_count; ++i) {
        const double red = rgb[3 * i];
        const double green = rgb[3 * i + 1];
        const double blue = rgb[3 * i + 2];
        const HslExtrema source = rgb_to_hue_extrema(red, green, blue);
        const double hue_delta = lookup_table(hue_table, source.hue);
        const double saturation = lookup_table(saturation_table, source.hue);
        double luminance = lookup_table(luminance_table, source.hue);
        double minimum = source.minimum;
        double maximum = source.maximum;
        const double source_chroma = source.maximum - source.minimum;

        // The recovered SDR path applies Vibrance after modern MMHG conversion
        // and before HSL. Its input chroma remains the original MMHG channel;
        // HSL Luminance below also uses that pre-Vibrance chroma.
        if (params.vibrance != 0.0) {
            apply_vibrance(source.hue, params.vibrance, &minimum, &maximum);
        }

        // Match the modern SDR MMHG channel order: Luminance, Saturation,
        // then Hue. All three lookup tables use the original source Hue.
        double gain = clamp01(source_chroma);
        for (int repeat = 0; repeat < 3; ++repeat) gain *= 2.0 - gain;
        luminance *= gain;
        for (int repeat = 0; repeat < 2; ++repeat) {
            minimum *= 1.0 + luminance * (1.0 - clamp01(minimum));
            maximum *= 1.0 + luminance * (1.0 - clamp01(maximum));
        }

        if (saturation < 0.0) {
            const double midpoint = (minimum + maximum) * 0.5;
            minimum += saturation * (minimum - midpoint);
            maximum += saturation * (maximum - midpoint);
        } else if (saturation > 0.0) {
            const double chroma = maximum == 0.0 ? 0.0 : (maximum - minimum) / maximum;
            const double q = std::min(16.0 * maximum, 1.0);
            const double denominator = 1.0 - (1.0 - chroma) * (5.0 / 6.0) * saturation *
                                       (1.0 - minimum) * q * (2.0 - q);
            minimum = maximum - maximum * chroma / denominator;
        }

        double out_red, out_green, out_blue;
        reconstruct_rgb(source.hue + hue_delta, minimum, maximum,
                        &out_red, &out_green, &out_blue);
        output[3 * i] = static_cast<float>(out_red);
        output[3 * i + 1] = static_cast<float>(out_green);
        output[3 * i + 2] = static_cast<float>(out_blue);
    }
}

void apply_rendered_hsl(const float *rgb, float *output, std::size_t pixel_count,
                        const HslParams &params) {
    validate_hsl_params(params);
    validate_buffers(rgb, output, pixel_count);
    if (hsl_is_neutral(params)) {
        copy_if_distinct(rgb, output, rgb_value_count(pixel_count));
        return;
    }
    // 00dc4278 wraps the source curve in 012ffb20, whose Evaluate invokes the
    // underlying inverse. 00dbfa64 retains its forward in the later main curve.
    // Modern ordinary RGBTone (0137d95c, flag=1) maps MMHG extrema.
    apply_default_rgb_tone(rgb, output, pixel_count, detail::kDefaultToneInverse);
    apply_hsl(output, output, pixel_count, params);
    apply_default_rgb_tone(output, output, pixel_count, detail::kDefaultToneForward);
}

void apply_grading(const float *rgb, float *output, std::size_t pixel_count,
                   const GradingParams &params) {
    validate_grading_params(params);
    const std::size_t values = rgb_value_count(pixel_count);
    validate_buffers(rgb, output, pixel_count);
    if (grading_is_neutral(params)) {
        copy_if_distinct(rgb, output, values);
        return;
    }

    const double saturation_scale = 0.9 + 0.1 * std::max(0.0, (params.blend - 50.0) * 0.02);
    const auto shadow = wheel_parameters(params.shadows, saturation_scale, 0);
    const auto mid = wheel_parameters(params.midtones, 1.0, 2);
    const auto highlight = wheel_parameters(params.highlights, saturation_scale, 1);
    auto global_color = hue_coefficients(params.global.hue, 1.0, -1.0, 1.0, 0);
    for (double &component : global_color) component *= params.global.saturation / 100.0;

    const double beta_input = std::clamp(-params.balance / 100.0, -1.0, 1.0) * 0.4 + 0.5;
    const double beta = srgb_decode(beta_input);
    const double a = factor(beta);
    const double blend_unit = params.blend / 100.0;
    const double blending = (blend_unit * (3.0 / 7.0)) /
                            (blend_unit * (3.0 / 7.0) + 1.0 - blend_unit);
    const double balance_damping = 1.0 - 0.85 * std::abs(params.balance) * 0.01;

    for (std::size_t i = 0; i < pixel_count; ++i) {
        const double red = rgb[3 * i];
        const double green = rgb[3 * i + 1];
        const double blue = rgb[3 * i + 2];
        const double out_red = apply_grading_channel(red, params, shadow, mid, highlight,
                                                      global_color, 0, a, blending,
                                                      balance_damping);
        const double out_green = apply_grading_channel(green, params, shadow, mid, highlight,
                                                        global_color, 1, a, blending,
                                                        balance_damping);
        const double out_blue = apply_grading_channel(blue, params, shadow, mid, highlight,
                                                       global_color, 2, a, blending,
                                                       balance_damping);
        if (!std::isfinite(out_red) || !std::isfinite(out_green) || !std::isfinite(out_blue)) {
            throw std::invalid_argument("grading produced a non-finite RGB value");
        }
        output[3 * i] = static_cast<float>(out_red);
        output[3 * i + 1] = static_cast<float>(out_green);
        output[3 * i + 2] = static_cast<float>(out_blue);
    }
}

}  // namespace imprint::acr
