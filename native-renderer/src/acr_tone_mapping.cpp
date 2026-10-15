#include "acr_tone_mapping.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace imprint::acr {
namespace {

void finite(double value, const char *message) {
    if (!std::isfinite(value)) throw std::invalid_argument(message);
}

void finite(float value, const char *message) {
    if (!std::isfinite(value)) throw std::invalid_argument(message);
}

double add(double a, double b) {
    const double result = a + b;
    finite(result, "tone mapping double addition overflow");
    return result;
}

double sub(double a, double b) {
    const double result = a - b;
    finite(result, "tone mapping double subtraction overflow");
    return result;
}

double mul(double a, double b) {
    const double result = a * b;
    finite(result, "tone mapping double multiplication overflow");
    return result;
}

double div(double a, double b) {
    if (b == 0.0) throw std::invalid_argument("tone mapping zero double divisor");
    const double result = a / b;
    finite(result, "tone mapping double division overflow");
    return result;
}

float add(float a, float b) {
    const float result = a + b;
    finite(result, "tone mapping float addition overflow");
    return result;
}

float sub(float a, float b) {
    const float result = a - b;
    finite(result, "tone mapping float subtraction overflow");
    return result;
}

float mul(float a, float b) {
    const float result = a * b;
    finite(result, "tone mapping float multiplication overflow");
    return result;
}

float div(float a, float b) {
    if (b == 0.0f) throw std::invalid_argument("tone mapping zero float divisor");
    const float result = a / b;
    finite(result, "tone mapping float division overflow");
    return result;
}

float to_float(double value, const char *message) {
    finite(value, message);
    if (value > static_cast<double>(std::numeric_limits<float>::max()) ||
        value < -static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument(message);
    }
    const float result = static_cast<float>(value);
    finite(result, message);
    return result;
}

void validate_stats(const LogSceneStats &stats) {
    finite(stats.center, "non-finite source tone statistics");
    finite(stats.low, "non-finite source tone statistics");
    finite(stats.high, "non-finite source tone statistics");
    finite(stats.median, "non-finite source tone statistics");
}

struct ShiftedStats {
    double center;
    double low;
    double high;
};

ShiftedStats shifted_stats(const LogSceneStats &stats, double ev90) {
    ShiftedStats result{stats.center, stats.low, stats.high};
    if (ev90 > 0.0) {
        // 009d94dc shifts the first 22 doubles. In this named subset that is
        // center/low/high; median (stats[23]) remains unchanged.
        result.center = sub(result.center, ev90);
        result.low = sub(result.low, ev90);
        result.high = sub(result.high, ev90);
    }
    return result;
}

ToneSpline5 make_spline(const std::array<float, 5> &values) {
    ToneSpline5 curve{};
    std::array<double, 5> initial{};
    std::array<double, 4> secants{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        finite(values[i], "non-finite tone spline node");
        curve.nodes[i] = static_cast<double>(values[i]);
    }
    for (std::size_t i = 0; i < secants.size(); ++i) {
        const double spacing = sub(curve.nodes[i + 1], curve.nodes[i]);
        if (spacing == 0.0) throw std::invalid_argument("zero tone spline node spacing");
        secants[i] = spacing;
    }

    for (std::size_t i = 1; i < 4; ++i) {
        initial[i] = mul(add(secants[i - 1], secants[i]), 0.5);
    }
    initial[0] = sub(mul(secants[0], 2.0), initial[1]);
    initial[4] = sub(mul(secants[3], 2.0), initial[3]);

    std::array<double, 5> rhs{};
    std::array<double, 5> upper{{0.5, 0.25, 0.25, 0.25, 0.0}};
    std::array<double, 5> lower{{0.0, 0.25, 0.25, 0.25, 0.5}};
    rhs[0] = mul(add(initial[0], initial[1]), 0.75);
    for (std::size_t i = 1; i < 4; ++i) rhs[i] = mul(initial[i], 1.5);
    rhs[4] = mul(add(initial[3], initial[4]), 0.75);

    // Thomas elimination for the recovered unit-diagonal five-point system.
    std::array<double, 5> modified_upper{};
    std::array<double, 5> modified_rhs{};
    double denominator = 1.0;
    modified_upper[0] = div(upper[0], denominator);
    modified_rhs[0] = div(rhs[0], denominator);
    for (std::size_t i = 1; i < 5; ++i) {
        denominator = sub(1.0, mul(lower[i], modified_upper[i - 1]));
        if (denominator == 0.0) throw std::invalid_argument("singular tone spline system");
        modified_upper[i] = div(upper[i], denominator);
        modified_rhs[i] = div(sub(rhs[i], mul(lower[i], modified_rhs[i - 1])), denominator);
    }
    curve.derivatives[4] = modified_rhs[4];
    for (std::size_t i = 4; i-- > 0;) {
        curve.derivatives[i] = sub(modified_rhs[i], mul(modified_upper[i], curve.derivatives[i + 1]));
    }
    for (double slope : curve.derivatives) finite(slope, "non-finite tone spline derivative");
    return curve;
}

double spline_value(const ToneSpline5 &curve, double t) {
    finite(t, "non-finite tone spline input");
    if (t <= 0.0) return curve.nodes[0];
    if (t >= 4.0) return curve.nodes[4];

    const std::size_t segment = t <= 1.0 ? 0 : (t <= 2.0 ? 1 : (t <= 3.0 ? 2 : 3));
    const double u = sub(t, static_cast<double>(segment));
    const double v = sub(1.0, u);
    const double term1_inner = add(v, 2.0);
    const double term1_factor = sub(term1_inner, u);
    const double term1_slope = mul(-v, curve.derivatives[segment + 1]);
    const double term1_y = mul(term1_factor, curve.nodes[segment + 1]);
    const double term1 = mul(mul(u, u), add(term1_slope, term1_y));

    const double term2_inner = add(u, 2.0);
    const double term2_factor = sub(term2_inner, v);
    const double term2_slope = mul(u, curve.derivatives[segment]);
    const double term2_y = mul(term2_factor, curve.nodes[segment]);
    const double term2 = mul(mul(v, v), add(term2_slope, term2_y));
    const double result = add(term1, term2);
    finite(result, "non-finite tone spline result");
    return result;
}

double normalized_spline(const ToneSpline5 &curve, double t) {
    const double value = spline_value(curve, t);
    const std::size_t segment = t <= 1.0 ? 0 : (t <= 2.0 ? 1 : (t <= 3.0 ? 2 : 3));
    const double numerator = sub(value, curve.nodes[segment]);
    const double denominator = sub(curve.nodes[segment + 1], curve.nodes[segment]);
    if (denominator == 0.0) throw std::invalid_argument("zero tone spline normalization span");
    const double fraction = div(numerator, denominator);
    const double result = add(fraction, static_cast<double>(segment));
    finite(result, "non-finite normalized tone spline result");
    return result;
}

std::array<float, 4> split4(float z) {
    std::array<float, 4> weights{};
    for (std::size_t i = 0; i < 3; ++i) {
        weights[i] = std::min(z, 1.0f);
        z = sub(z, weights[i]);
    }
    weights[3] = z;
    return weights;
}

float powf_checked(float base, float exponent) {
    const float result = ::powf(base, exponent);
    finite(result, "non-finite tone mapping powf result");
    return result;
}

float expf_checked(float value) {
    const float result = ::expf(value);
    finite(result, "non-finite tone mapping expf result");
    return result;
}

float float_from_bits(std::uint32_t bits) {
    float result = 0.0f;
    static_assert(sizeof(result) == sizeof(bits), "float32 is required");
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

}  // namespace

ToneMapping build_tone_mapping(const ToneMappingParameters &parameters) {
    validate_stats(parameters.source_stats);
    validate_stats(parameters.mask_aligned_stats);
    finite(parameters.ev90, "non-finite EV90");
    finite(parameters.ev98, "non-finite EV98");
    finite(parameters.ev100, "non-finite EV100");
    finite(parameters.tone_708, "non-finite tone[708]");
    finite(parameters.tone_70c, "non-finite tone[70c]");
    finite(parameters.tone_710, "non-finite tone[710]");

    const ShiftedStats source = shifted_stats(parameters.source_stats, parameters.ev90);
    const ShiftedStats mask = shifted_stats(parameters.mask_aligned_stats, parameters.ev90);

    ToneMapping model{};
    model.center = to_float(source.center, "tone center is not representable as float32");
    model.source_low = to_float(source.low, "tone source low is not representable as float32");
    model.source_high = to_float(source.high, "tone source high is not representable as float32");
    model.mask_center = to_float(mask.center, "tone mask center is not representable as float32");
    model.mask_low = to_float(mask.low, "tone mask low is not representable as float32");
    model.mask_high = to_float(mask.high, "tone mask high is not representable as float32");
    model.tone_708 = parameters.tone_708;
    model.tone_70c = parameters.tone_70c;
    model.tone_710 = parameters.tone_710;
    model.ev98_plus_ev100 = to_float(add(parameters.ev98, parameters.ev100),
                                     "EV98 plus EV100 is not representable as float32");

    const float center_minus_quarter = sub(model.center, 0.125f);
    const float center_plus_quarter = add(model.center, 0.125f);
    model.low_node = std::min(model.source_low, center_minus_quarter);
    model.high_node = std::max(model.source_high, center_plus_quarter);
    const float adjusted_mask_low = std::min(model.mask_low, center_minus_quarter);
    const float adjusted_mask_high = std::max(model.mask_high, center_plus_quarter);
    const float inner_low_a = add(adjusted_mask_low, 0.25f);
    const float inner_low_delta = mul(sub(adjusted_mask_low, model.center), 0.1f);
    const float inner_low_b = add(model.center, inner_low_delta);
    model.inner_low = std::min(inner_low_a, inner_low_b);
    const float inner_high_a = sub(adjusted_mask_high, 0.25f);
    const float inner_high_delta = mul(sub(adjusted_mask_high, model.center), 0.1f);
    const float inner_high_b = add(model.center, inner_high_delta);
    model.inner_high = std::max(inner_high_a, inner_high_b);

    std::array<float, 5> left_nodes{{model.low_node, model.inner_low, 0.0f, 0.0f, 0.0f}};
    std::array<float, 5> right_nodes{{model.high_node, model.inner_high, 0.0f, 0.0f, 0.0f}};
    left_nodes[2] = mul(add(model.inner_low, model.center), 0.5f);
    right_nodes[2] = mul(add(model.inner_high, model.center), 0.5f);
    left_nodes[3] = mul(add(left_nodes[2], model.center), 0.5f);
    right_nodes[3] = mul(add(right_nodes[2], model.center), 0.5f);
    left_nodes[4] = mul(add(left_nodes[3], model.center), 0.5f);
    right_nodes[4] = mul(add(right_nodes[3], model.center), 0.5f);
    model.inner_low = left_nodes[1];
    model.inner_high = right_nodes[1];
    model.left_spline = make_spline(left_nodes);
    model.right_spline = make_spline(right_nodes);

    const float left_denominator = sub(model.low_node, model.center);
    const float right_denominator = sub(model.high_node, model.center);
    for (std::size_t i = 0; i < 4; ++i) {
        const float left_value = left_nodes[i + 1];
        model.left_affine_slopes[i] = div(sub(left_value, model.center), left_denominator);
    }
    model.right_affine_slopes[0] = div(sub(model.center, model.inner_high),
                                       sub(model.center, model.high_node));
    for (std::size_t i = 1; i < 4; ++i) {
        const float right_value = right_nodes[i + 1];
        model.right_affine_slopes[i] = div(sub(right_value, model.center), right_denominator);
    }
    const float shadow_denominator = sub(model.source_low, model.center);
    model.positive_shadow_input_slope = shadow_denominator == 0.0f ? 0.0f
        : div(-0.85f, shadow_denominator);

    const double source_range = sub(source.high, source.low);
    const double sigma_unclamped = add(div(sub(source_range, 2.0), 32.0), 0.125);
    const double sigma = std::clamp(sigma_unclamped, 0.125, 0.25);
    model.gaussian_coefficient = to_float(div(-1.0, mul(sigma, sigma)),
                                          "tone gaussian coefficient is not representable");

    const double tone_difference = std::abs(sub(static_cast<double>(model.tone_708),
                                                 static_cast<double>(model.tone_70c)));
    model.mix = to_float(std::clamp(sub(1.25, tone_difference), 0.0, 1.0),
                          "tone mix is not representable as float32");

    const float high_inner_range = sub(model.inner_high, model.high_node);
    const float high_outer = sub(model.high_node, mul(high_inner_range, 2.0f));
    const float low_inner_range = sub(model.inner_low, model.low_node);
    const float low_outer = sub(model.low_node, mul(low_inner_range, 2.0f));
    const float span = sub(high_outer, low_outer);
    if (span >= 12.0f) {
        model.width_factor = 1.0f;
    } else if (span <= 0x1p-15f) {
        model.width_factor = 393216.0f;
    } else {
        model.width_factor = div(12.0f, span);
    }

    const double high_delta = sub(0.0, parameters.ev98);
    const double center_plus_one = add(static_cast<double>(model.center), 1.0);
    const double highlight_anchor = std::max(high_delta, center_plus_one);
    const double highlight_denominator = sub(static_cast<double>(model.center), highlight_anchor);
    model.nonpositive_highlight_slope = highlight_denominator == 0.0
        ? 0.0f
        : to_float(div(0.9, highlight_denominator),
                   "non-positive highlight slope is not representable as float32");

    const double threshold_target = sub(sub(0.0, parameters.ev98), parameters.ev100);
    finite(threshold_target, "tone threshold target overflow");
    if (threshold_target > static_cast<double>(model.inner_high) ||
        threshold_target > model.right_spline.nodes[2]) {
        const double threshold_denominator = std::max(
            sub(static_cast<double>(model.center), threshold_target), 0x1p-16);
        model.minimum_gate = to_float(std::clamp(
            div(sub(static_cast<double>(model.source_high), threshold_target),
                threshold_denominator), 0.0, 1.0),
            "tone minimum gate is not representable as float32");

        const double source_high = static_cast<double>(model.source_high);
        const double center = static_cast<double>(model.center);
        const double right_span = sub(center, static_cast<double>(model.high_node));
        const double first_slope = div(sub(center, static_cast<double>(model.inner_high)),
                                       right_span);
        const double second_slope = div(sub(center, model.right_spline.nodes[2]), right_span);
        const double first_target = add(center, mul(sub(source_high, center), first_slope));
        const double second_target = add(center, mul(sub(source_high, center), second_slope));
        auto callback = [&](double r) {
            const double z = normalized_spline(model.right_spline, mul(2.0, r));
            const double first_fraction = std::min(z, 1.0);
            const double first = add(source_high,
                mul(sub(first_target, source_high), first_fraction));
            const double result = add(first,
                mul(sub(second_target, first), sub(z, first_fraction)));
            finite(result, "non-finite tone threshold callback");
            return result;
        };

        double r0 = 0.0;
        double r1 = 1.0;
        double y0 = callback(r0);
        double y1 = callback(r1);
        for (int iteration = 0; iteration < 30; ++iteration) {
            if (std::abs(sub(y1, y0)) < 1e-10) break;
            const double y_delta = sub(threshold_target, y1);
            const double r_delta = sub(r1, r0);
            const double y_span = sub(y1, y0);
            const double r2_unclamped = add(r1, div(mul(y_delta, r_delta), y_span));
            const double r2 = std::clamp(r2_unclamped, 0.0, 1.0);
            r0 = r1;
            r1 = r2;
            y0 = y1;
            y1 = callback(r1);
        }
        const float root = to_float(r1, "tone threshold root is not representable as float32");
        const float doubled_root = add(root, root);
        model.threshold = std::max(doubled_root, 0.0f);
    }

    finite(model.threshold, "non-finite tone threshold");
    finite(model.minimum_gate, "non-finite tone minimum gate");
    finite(model.width_factor, "non-finite tone width factor");
    return model;
}

float evaluate_tone_mapping(const ToneMapping &model, float x, float delta,
                            float local3, float local4, float local5) {
    finite(x, "non-finite tone mapping input");
    finite(delta, "non-finite tone mapping delta");
    finite(local3, "non-finite local tone lane 3");
    finite(local4, "non-finite local tone lane 4");
    finite(local5, "non-finite local tone lane 5");

    std::array<float, 4> affine{};
    if (x < model.center) {
        for (std::size_t i = 0; i < 4; ++i) {
            const float node = to_float(model.left_spline.nodes[i + 1],
                                        "invalid left tone spline node");
            // The stored affine map is anchor + (x-anchor)*slope, with the
            // node's y-value as the anchor value.
            affine[i] = add(node, mul(sub(x, model.low_node), model.left_affine_slopes[i]));
        }
    } else {
        affine[0] = add(model.center,
            mul(sub(x, model.center), model.right_affine_slopes[0]));
        for (std::size_t i = 1; i < 4; ++i) {
            const float node = to_float(model.right_spline.nodes[i + 1],
                                        "invalid right tone spline node");
            affine[i] = add(node,
                mul(sub(x, model.high_node), model.right_affine_slopes[i]));
        }
    }

    const float x_plus_delta = add(x, delta);
    const float offset = sub(x_plus_delta, affine[0]);
    const float shadow_amount = mul(add(model.tone_708, local3), 2.0f);
    const float highlight_amount = mul(add(model.tone_70c, local4), 2.0f);

    std::array<float, 4> shadow{};
    if (shadow_amount > 0.0f) {
        float p = add(0.15f, mul(sub(x, model.source_low),
                                 model.positive_shadow_input_slope));
        p = std::clamp(p, 0.15f, 1.0f);
        const float shaped = powf_checked(p, float_from_bits(0x3ee8ba2e));
        const float t = mul(mul(shadow_amount, 1.2f), shaped);
        shadow = split4(to_float(normalized_spline(model.left_spline,
                                                   static_cast<double>(t)),
                                 "left tone weight is not representable as float32"));
    } else {
        float ease = std::min(mul(shadow_amount, -0.5f), 1.0f);
        if (ease < 0.0f) ease = 0.0f;
        const float factor = add(1.0f, mul(sub(model.width_factor, 1.0f), ease));
        shadow[0] = mul(factor, shadow_amount);
    }

    std::array<float, 4> highlight{};
    if (highlight_amount <= 0.0f) {
        float p = add(1.0f, mul(sub(x, model.center), model.nonpositive_highlight_slope));
        p = std::clamp(p, 0.0f, 1.0f);
        const float shaped = powf_checked(p, float_from_bits(0x3f36db6e));
        const float q = mul(shaped, highlight_amount);
        float ease = std::min(mul(q, -0.5f), 1.0f);
        if (ease < 0.0f) ease = 0.0f;
        const float factor = add(1.0f, mul(sub(model.width_factor, 1.0f), ease));
        highlight[0] = mul(q, factor);
    } else {
        float t = highlight_amount;
        if (highlight_amount > model.threshold) {
            const float denominator = sub(model.center, model.source_high);
            const float slope = denominator == 0.0f ? 0.0f
                : div(sub(1.0f, model.minimum_gate), denominator);
            float p = add(1.0f, mul(sub(x, model.center), slope));
            p = std::clamp(p, model.minimum_gate, 1.0f);

            float a = std::min(mul(highlight_amount, 0.5f), 1.0f);
            if (a < 0.0f) a = 0.0f;
            a = add(mul(a, 0.34999996f), 0.3f);
            const double a_double = static_cast<double>(a);
            const double numerator = mul(a_double, 0.7999999970197678);
            const double denominator_double = mul(sub(1.0, a_double),
                                                  0.20000000298023224);
            const double k = div(numerator, denominator_double);
            const double p_double = static_cast<double>(p);
            const double kp = mul(k, p_double);
            const double warped_denominator = sub(add(kp, 1.0), p_double);
            const float v = to_float(div(kp, warped_denominator),
                                     "positive highlight warp is not representable as float32");
            t = add(model.threshold, mul(sub(highlight_amount, model.threshold), v));
        }
        highlight = split4(to_float(normalized_spline(model.right_spline,
                                                      static_cast<double>(t)),
                                    "right tone weight is not representable as float32"));
    }

    float clarity = add(model.tone_710, local5);
    float bias = 0.0f;
    if (clarity > 0.0f) {
        float current = x;
        for (std::size_t i = 0; i < 4; ++i) {
            const float target = add(affine[i], offset);
            const float weight = current <= target ? shadow[i] : highlight[i];
            current = add(current, mul(sub(target, current), weight));
        }
        float v = add(add(model.ev98_plus_ev100, current), 8.0f);
        v = sub(mul(v, 0.25f), 1.0f);
        v = std::clamp(v, -1.0f, 1.0f);
        const float v2 = mul(v, v);
        const float v4 = mul(v2, v2);
        const float v8 = mul(v4, v4);
        const float one_minus_v8 = sub(1.0f, v8);
        const float square = mul(one_minus_v8, one_minus_v8);
        const float fourth = mul(square, square);
        const float clarity_weight = mul(clarity, fourth);
        bias = mul(offset, clarity_weight);
    }

    const float initial_bias = bias > 0.0f ? add(bias, bias) : bias;
    float current = add(x, initial_bias);
    const float target0 = add(add(x, delta), bias);
    std::array<float, 4> targets{{target0, 0.0f, 0.0f, 0.0f}};
    for (std::size_t i = 1; i < 4; ++i) {
        targets[i] = add(add(offset, affine[i]), bias);
    }

    for (std::size_t i = 0; i < 4; ++i) {
        if (i != 0 && shadow[i] <= 0.0f && highlight[i] <= 0.0f) continue;
        const float diff = sub(current, targets[i]);
        const float exponent = mul(mul(diff, model.gaussian_coefficient), diff);
        const float e = expf_checked(exponent);
        const float gate = add(model.mix,
            mul(sub(1.0f, model.mix), sub(1.0f, e)));
        const float weight = current <= targets[i] ? shadow[i] : highlight[i];
        const float update = mul(sub(targets[i], current), mul(gate, weight));
        current = add(current, update);
    }

    if (local3 != 0.0f && current <= 0.0f) {
        const float plus = add(mul(local3, 2.0f), -20.0f);
        const float minus = add(mul(local3, -2.0f), -20.0f);
        float slope = 0.0f;
        float base = plus;
        float start = -20.0f;
        if (local3 >= 0.0f) {
            slope = div(plus, -20.0f);
            start = -20.0f;
        } else {
            if (minus != 0.0f) slope = div(-20.0f, minus);
            base = -20.0f;
            start = minus;
        }
        current = add(base, mul(sub(current, start), slope));
    }
    finite(current, "non-finite tone mapping output");
    return current;
}

}  // namespace imprint::acr
