#include "acr_profile_tone.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

struct NaturalSpline {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> derivative;
};

void require_finite(double value, const char *message) {
    if (!std::isfinite(value)) throw std::invalid_argument(message);
}

void require_finite(float value, const char *message) {
    if (!std::isfinite(value)) throw std::invalid_argument(message);
}

NaturalSpline make_spline(const double *points_xy, std::size_t point_count) {
    if (!points_xy || point_count < 2 || point_count > 65536 ||
        point_count > std::vector<double>().max_size() / 2) {
        throw std::invalid_argument("profile tone requires at least two points");
    }

    NaturalSpline curve;
    curve.x.resize(point_count);
    curve.y.resize(point_count);
    curve.derivative.resize(point_count);
    for (std::size_t i = 0; i < point_count; ++i) {
        const double x = points_xy[2 * i];
        const double y = points_xy[2 * i + 1];
        require_finite(x, "non-finite profile x coordinate");
        require_finite(y, "non-finite profile y coordinate");
        if (x < 0.0 || x > 1.0 || y < 0.0 || y > 1.0) {
            throw std::invalid_argument("profile coordinates must be in [0, 1]");
        }
        if (i != 0 && !(x > curve.x[i - 1])) {
            throw std::invalid_argument("profile x coordinates must increase strictly");
        }
        curve.x[i] = x;
        curve.y[i] = y;
    }

    const std::size_t intervals = point_count - 1;
    std::vector<double> spacing(intervals);
    std::vector<double> secant(intervals);
    for (std::size_t i = 0; i < intervals; ++i) {
        spacing[i] = curve.x[i + 1] - curve.x[i];
        secant[i] = (curve.y[i + 1] - curve.y[i]) / spacing[i];
        require_finite(spacing[i], "invalid profile point spacing");
        require_finite(secant[i], "profile secant is not representable");
    }

    // The two-point branch is explicitly linear. For three or more points,
    // solve the recovered unit-diagonal first-derivative system with Thomas
    // elimination. It permits non-monotone y values.
    if (point_count == 2) {
        curve.derivative[0] = secant[0];
        curve.derivative[1] = secant[0];
        return curve;
    }

    std::vector<double> lower(point_count, 0.0);
    std::vector<double> diagonal(point_count, 1.0);
    std::vector<double> upper(point_count, 0.0);
    std::vector<double> rhs(point_count, 0.0);
    std::vector<double> modified_upper(point_count, 0.0);
    std::vector<double> modified_rhs(point_count, 0.0);

    upper[0] = 0.5;
    rhs[0] = 1.5 * secant[0];
    for (std::size_t i = 1; i + 1 < point_count; ++i) {
        const double previous = spacing[i - 1];
        const double next = spacing[i];
        const double sum = previous + next;
        lower[i] = next / (2.0 * sum);
        upper[i] = previous / (2.0 * sum);
        rhs[i] = 1.5 * (previous * secant[i] + next * secant[i - 1]) / sum;
    }
    lower[point_count - 1] = 0.5;
    rhs[point_count - 1] = 1.5 * secant[intervals - 1];

    modified_upper[0] = upper[0] / diagonal[0];
    modified_rhs[0] = rhs[0] / diagonal[0];
    for (std::size_t i = 1; i < point_count; ++i) {
        const double pivot = diagonal[i] - lower[i] * modified_upper[i - 1];
        if (!(pivot > 0.0) || !std::isfinite(pivot)) {
            throw std::invalid_argument("singular profile spline system");
        }
        modified_upper[i] = upper[i] / pivot;
        modified_rhs[i] = (rhs[i] - lower[i] * modified_rhs[i - 1]) / pivot;
    }
    curve.derivative[point_count - 1] = modified_rhs[point_count - 1];
    for (std::size_t i = point_count - 1; i-- > 0;) {
        curve.derivative[i] = modified_rhs[i] - modified_upper[i] * curve.derivative[i + 1];
    }
    for (double value : curve.derivative) {
        require_finite(value, "profile spline derivative is not representable");
    }
    return curve;
}

double evaluate_spline(const NaturalSpline &curve, double value) {
    if (value <= curve.x.front()) return curve.y.front();
    if (value >= curve.x.back()) return curve.y.back();

    const auto exact = std::lower_bound(curve.x.begin(), curve.x.end(), value);
    if (exact != curve.x.end() && *exact == value) {
        return curve.y[static_cast<std::size_t>(exact - curve.x.begin())];
    }
    const std::size_t right = static_cast<std::size_t>(
        std::upper_bound(curve.x.begin(), curve.x.end(), value) - curve.x.begin());
    const std::size_t left = right - 1;
    const double h = curve.x[right] - curve.x[left];
    const double t = (value - curve.x[left]) / h;
    const double one_minus_t = 1.0 - t;

    // Hermite form in the same grouping as 0127ad8c.
    const double right_slope = one_minus_t * (-(h * curve.derivative[right]));
    const double right_y = (one_minus_t + (2.0 - t)) * curve.y[right];
    const double first = (t * t) * (right_slope + right_y);
    const double left_slope = (h * curve.derivative[left]) * t;
    const double left_y = (t + (2.0 - one_minus_t)) * curve.y[left];
    const double second = (one_minus_t * one_minus_t) * (left_slope + left_y);
    const double result = first + second;
    require_finite(result, "non-finite profile spline result");
    return result;
}

bool ranges_overlap(const void *a, std::size_t a_bytes,
                    const void *b, std::size_t b_bytes) {
    const auto a_start = reinterpret_cast<std::uintptr_t>(a);
    const auto b_start = reinterpret_cast<std::uintptr_t>(b);
    if (a_start <= b_start) return b_start - a_start < a_bytes;
    return a_start - b_start < b_bytes;
}

float lut_sample(const ProfileToneLut &lut, float value) {
    const float scaled = value * static_cast<float>(kProfileToneLutSegments);
    const auto index = static_cast<std::size_t>(scaled);
    const float fraction = scaled - static_cast<float>(index);
    const float upper = lut[index + 1] * fraction;
    const float lower = (1.0f - fraction) * lut[index];
    return upper + lower;
}

struct OrderedChannels {
    float minimum;
    float middle;
    float maximum;
    std::size_t minimum_index;
    std::size_t middle_index;
    std::size_t maximum_index;
};

OrderedChannels sort_channels(const float *rgb) {
    // Preserve the comparison order recovered from 0137d95c, including its
    // deterministic handling of equal channel values.
    float pair_max = rgb[2];
    float pair_min = rgb[1];
    std::size_t pair_max_index = 2;
    std::size_t pair_min_index = 1;
    if (rgb[2] <= rgb[1]) {
        pair_max = rgb[1];
        pair_min = rgb[2];
        pair_max_index = 1;
        pair_min_index = 2;
    }

    float middle = pair_min;
    float minimum = rgb[0];
    std::size_t middle_index = pair_min_index;
    std::size_t minimum_index = 0;
    if (pair_min <= rgb[0]) {
        middle = rgb[0];
        minimum = pair_min;
        middle_index = 0;
        minimum_index = pair_min_index;
    }

    float maximum = pair_max;
    std::size_t maximum_index = pair_max_index;
    if (pair_max <= middle) {
        maximum = middle;
        middle = pair_max;
        maximum_index = middle_index;
        middle_index = pair_max_index;
    }
    return {minimum, middle, maximum,
            minimum_index, middle_index, maximum_index};
}

float clip_unit(float value) {
    if (value > 1.0f) return 1.0f;
    if (value < 0.0f) return 0.0f;
    return value;
}

void apply_pixel(const ProfileToneLut &lut, const float *rgb,
                 float *destination, bool modern) {
    const OrderedChannels ordered = sort_channels(rgb);
    const float range = ordered.maximum - ordered.minimum;
    float hue_gap = 0.0f;
    if (range != 0.0f) {
        hue_gap = (ordered.middle - ordered.minimum) / range;
    }

    float output_maximum;
    float output_minimum;
    if (modern) {
        const float one_minus_hue = 1.0f - hue_gap;
        const float max_delta = ordered.middle - ordered.maximum;
        const float max_shift = ((max_delta * hue_gap) * hue_gap) * 0.5f;
        const float max_input = ordered.maximum + max_shift;
        const float min_delta = ordered.middle - ordered.minimum;
        const float min_shift = ((min_delta * one_minus_hue) * one_minus_hue) * 0.5f;
        const float min_input = ordered.minimum + min_shift;
        const float curve_maximum = lut_sample(lut, max_input);
        const float curve_minimum = lut_sample(lut, min_input);

        const float hue_squared = hue_gap * hue_gap;
        const float hue_minus_one = hue_gap - 1.0f;
        const float denominator = (hue_squared - hue_gap) + 2.0f;
        const float reciprocal = 1.0f / denominator;

        const float coeff_max_1 = hue_gap * hue_squared;
        const float coeff_max_2 = hue_squared * -2.0f;
        const float coeff_max_3 = hue_gap + coeff_max_1;
        const float coeff_max_4 = coeff_max_3 + coeff_max_2;
        const float coeff_max = coeff_max_4 + -2.0f;
        const float weighted_curve_max = curve_maximum * coeff_max;
        const float coeff_min_max = (hue_minus_one * hue_squared) * curve_minimum;
        const float maximum_numerator = -weighted_curve_max + coeff_min_max;
        output_maximum = clip_unit(reciprocal * maximum_numerator);

        const float correction_max = (hue_gap * hue_squared) - hue_squared;
        const float coeff_curve_min = correction_max + 2.0f;
        const float weighted_min = coeff_curve_min * curve_minimum;
        const float minimum_numerator = -(curve_maximum * hue_gap * hue_minus_one * hue_minus_one)
                                      + weighted_min;
        output_minimum = clip_unit(reciprocal * minimum_numerator);
    } else {
        output_maximum = lut_sample(lut, ordered.maximum);
        output_minimum = lut_sample(lut, ordered.minimum);
    }

    const float output_middle = output_minimum + hue_gap * (output_maximum - output_minimum);
    destination[ordered.minimum_index] = output_minimum;
    destination[ordered.middle_index] = output_middle;
    destination[ordered.maximum_index] = output_maximum;
}

}  // namespace

ProfileToneLut build_profile_tone_lut(const double *points_xy,
                                      std::size_t point_count,
                                      bool clip_curve_output,
                                      ProfileToneInputTransform input_transform,
                                      const void *transform_context) {
    const NaturalSpline curve = make_spline(points_xy, point_count);
    ProfileToneLut lut{};
    for (std::size_t i = 0; i <= kProfileToneLutSegments; ++i) {
        const double x = static_cast<double>(i) /
                         static_cast<double>(kProfileToneLutSegments);
        const double transformed = input_transform ? input_transform(x, transform_context) : x;
        require_finite(transformed, "non-finite pre-profile scalar coordinate");
        double value = evaluate_spline(curve, transformed);
        if (clip_curve_output) value = std::clamp(value, 0.0, 1.0);
        const float sample = static_cast<float>(value);
        require_finite(sample, "profile LUT sample is not representable as float32");
        lut[i] = sample;
    }
    // 01300490 reserves the following entry and duplicates the endpoint for
    // interpolation at exactly x=1.
    lut[kProfileToneLutSegments + 1] = lut[kProfileToneLutSegments];
    return lut;
}

void apply_profile_tone(const float *source, float *destination,
                        std::size_t pixel_count, const double *points_xy,
                        std::size_t point_count, bool modern,
                        ProfileToneInputTransform input_transform,
                        const void *transform_context) {
    if (!source || !destination || !points_xy || pixel_count == 0 ||
        pixel_count > std::numeric_limits<std::size_t>::max() / 3 ||
        point_count > std::numeric_limits<std::size_t>::max() / (2 * sizeof(double))) {
        throw std::invalid_argument("invalid profile tone buffer or extent");
    }
    const std::size_t samples = pixel_count * 3;
    if (samples > std::vector<float>().max_size()) {
        throw std::invalid_argument("profile tone image is too large");
    }
    const std::size_t source_bytes = samples * sizeof(float);
    const std::size_t point_bytes = point_count * 2 * sizeof(double);
    if (ranges_overlap(source, source_bytes, destination, source_bytes) ||
        ranges_overlap(points_xy, point_bytes, destination, source_bytes)) {
        throw std::invalid_argument("profile tone destination overlaps an input");
    }
    for (std::size_t i = 0; i < samples; ++i) {
        const float value = source[i];
        require_finite(value, "non-finite profile tone RGB input");
        if (value < 0.0f || value > 1.0f) {
            throw std::invalid_argument("profile tone RGB input must be in [0, 1]");
        }
    }

    const ProfileToneLut lut = build_profile_tone_lut(
        points_xy, point_count, true, input_transform, transform_context);
    std::vector<float> result(samples);
    for (std::size_t i = 0; i < pixel_count; ++i) {
        apply_pixel(lut, source + 3 * i, result.data() + 3 * i, modern);
    }
    std::copy(result.begin(), result.end(), destination);
}

}  // namespace imprint::acr
