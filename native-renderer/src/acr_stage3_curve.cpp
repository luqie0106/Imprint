#include "acr_stage3_curve.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace imprint::acr {
namespace {

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

bool address_range(const void* pointer, std::size_t size,
                   std::size_t alignment, AddressRange* out) noexcept {
    if (pointer == nullptr || out == nullptr || alignment == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) return false;
    constexpr auto kMax = std::numeric_limits<std::uintptr_t>::max();
    if (size > static_cast<std::size_t>(kMax - begin)) return false;
    out->begin = begin;
    out->end = begin + static_cast<std::uintptr_t>(size);
    return true;
}

bool overlaps(const AddressRange& left, const AddressRange& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

bool finite(double value) noexcept { return std::isfinite(value); }

bool same_value(double left, double right) noexcept {
    if (left != right) return false;
    return left != 0.0 || std::signbit(left) == std::signbit(right);
}

double double_from_bits(std::uint64_t bits) noexcept {
    double value = 0.0;
    static_assert(sizeof(value) == sizeof(bits), "double must be 64-bit");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

double clamp01(double value) noexcept {
    const double upper = std::fmin(value, 1.0);
    return upper < 0.0 ? +0.0 : upper;
}

bool derive_prepared(double pivot, bool cubic,
                     Stage3CurvePrepared* out) noexcept {
    if (!finite(pivot) || !(pivot >= 0.0 && pivot < 1.0)) return false;
    const double log_black = pivot > 0.0 ? 0.25 : +0.0;
    const Stage3CurvePrepared candidate{pivot, log_black, cubic};
    if (!finite(candidate.pivot) || !finite(candidate.log_black)) return false;
    *out = candidate;
    return true;
}

bool valid_prepared(const Stage3CurvePrepared& prepared) noexcept {
    Stage3CurvePrepared expected{};
    if (!derive_prepared(prepared.pivot, prepared.cubic, &expected)) return false;
    return same_value(prepared.pivot, expected.pivot) &&
           same_value(prepared.log_black, expected.log_black) &&
           prepared.cubic == expected.cubic;
}

bool cubic_forward(double sample, double pivot, double* result) noexcept {
    if (!finite(sample) || result == nullptr) return false;
    const double pivot_delta = pivot + -1.0;
    if (!finite(pivot_delta) || pivot_delta == 0.0) return false;
    const double reciprocal = -1.0 / pivot_delta;
    if (!finite(reciprocal)) return false;
    const double centered = sample - pivot;
    if (!finite(centered)) return false;
    const double u = std::fma(reciprocal, centered, +0.0);
    if (!finite(u)) return false;
    double t = u * double_from_bits(0x3fefc00000000000ULL);  // 0.9921875
    if (!finite(t)) return false;
    t = u * t;
    if (!finite(t)) return false;
    t = u * t;
    if (!finite(t)) return false;
    const double v = std::fma(u, double_from_bits(0x3f80000000000000ULL), t);
    if (!finite(v)) return false;
    const double output_delta = -pivot_delta;
    if (!finite(output_delta)) return false;
    const double value = std::fma(output_delta, v, pivot);
    if (!finite(value)) return false;
    *result = value;
    return true;
}

bool cubic_inverse(double target, double pivot, double* result) noexcept {
    if (!finite(target) || result == nullptr) return false;
    double xprev = +0.0;
    double xcur = 1.0;
    double yprev = 0.0;
    double ycur = 0.0;
    if (!cubic_forward(xprev, pivot, &yprev) ||
        !cubic_forward(xcur, pivot, &ycur)) {
        return false;
    }
    const double threshold = double_from_bits(0x3ddb7cdfd9d7bdbbULL);
    for (int iteration = 0; iteration < 30; ++iteration) {
        const double diff = ycur - yprev;
        if (!finite(diff)) return false;
        const double abs_diff = std::abs(diff);
        if (!finite(abs_diff)) return false;
        if (abs_diff < threshold) {
            *result = xcur;
            return finite(xcur);
        }
        const double a = target - ycur;
        if (!finite(a)) return false;
        const double b = xcur - xprev;
        if (!finite(b)) return false;
        double v = a * b;
        if (!finite(v)) return false;
        v = v / diff;
        if (!finite(v)) return false;
        const double next_sum = xcur + v;
        if (!finite(next_sum)) return false;
        const double xnext = clamp01(next_sum);
        if (!finite(xnext)) return false;
        double ynext = 0.0;
        if (!cubic_forward(xnext, pivot, &ynext)) return false;
        xprev = xcur;
        yprev = ycur;
        xcur = xnext;
        ycur = ynext;
    }
    *result = xcur;
    return finite(xcur);
}

bool log_forward(double sample, const Stage3CurvePrepared& prepared,
                 double* result) noexcept {
    if (!finite(sample) || result == nullptr) return false;
    const double pivot_delta = prepared.pivot + -1.0;
    if (!finite(pivot_delta) || pivot_delta == 0.0) return false;
    const double reciprocal = -1.0 / pivot_delta;
    if (!finite(reciprocal)) return false;
    const double centered = sample - prepared.pivot;
    if (!finite(centered)) return false;
    const double z = std::fma(reciprocal, centered, +0.0);
    if (!finite(z)) return false;
    const bool negative = z < 0.0;
    const double absolute = negative ? -z : z;
    if (!finite(absolute)) return false;
    const double inside_log = std::fma(
        absolute, double_from_bits(0x3fefae147ae147aeULL),
        double_from_bits(0x3f847ae147ae147bULL));
    if (!finite(inside_log) || !(inside_log > 0.0)) return false;
    const double logarithm = std::log(inside_log);
    if (!finite(logarithm)) return false;
    const double shifted = logarithm + double_from_bits(0x40126bb1bbb55514ULL);
    if (!finite(shifted)) return false;
    double value = shifted * double_from_bits(0x3fcbcb7b1526e510ULL);
    if (!finite(value)) return false;
    if (negative) value = -value;
    if (!finite(value)) return false;
    const double log_delta = prepared.log_black + -1.0;
    if (!finite(log_delta)) return false;
    const double output_delta = -log_delta;
    if (!finite(output_delta)) return false;
    const double output = std::fma(output_delta, value, prepared.log_black);
    if (!finite(output)) return false;
    const double clamped = clamp01(output);
    if (!finite(clamped)) return false;
    *result = clamped;
    return true;
}

bool log_inverse(double sample, const Stage3CurvePrepared& prepared,
                 double* result) noexcept {
    if (!finite(sample) || result == nullptr) return false;
    const double log_delta = prepared.log_black + -1.0;
    if (!finite(log_delta) || log_delta == 0.0) return false;
    const double reciprocal = -1.0 / log_delta;
    if (!finite(reciprocal)) return false;
    const double centered = sample - prepared.log_black;
    if (!finite(centered)) return false;
    const double z = std::fma(reciprocal, centered, +0.0);
    if (!finite(z)) return false;
    const bool negative = z < 0.0;
    const double absolute = negative ? -z : z;
    if (!finite(absolute)) return false;
    const double exponent = std::fma(
        absolute, double_from_bits(0x40126bb1bbb55514ULL),
        -double_from_bits(0x40126bb1bbb55514ULL));
    if (!finite(exponent)) return false;
    const double exponential = std::exp(exponent);
    if (!finite(exponential)) return false;
    const double shifted = exponential + -double_from_bits(0x3f847ae147ae147bULL);
    if (!finite(shifted)) return false;
    double value = shifted * double_from_bits(0x3ff0295fad40a57fULL);
    if (!finite(value)) return false;
    if (negative) value = -value;
    if (!finite(value)) return false;
    const double pivot_delta = prepared.pivot + -1.0;
    if (!finite(pivot_delta)) return false;
    const double output_delta = -pivot_delta;
    if (!finite(output_delta)) return false;
    const double output = std::fma(output_delta, value, prepared.pivot);
    if (!finite(output)) return false;
    const double clamped = clamp01(output);
    if (!finite(clamped)) return false;
    *result = clamped;
    return true;
}

bool evaluate_valid(double sample, const Stage3CurvePrepared& prepared,
                    bool inverse, double* result) noexcept {
    if (!finite(sample) || result == nullptr) return false;
    if (inverse) {
        double value = 0.0;
        if (!log_inverse(sample, prepared, &value)) return false;
        if (!prepared.cubic) {
            *result = value;
            return true;
        }
        return cubic_inverse(value, prepared.pivot, result);
    }

    double first = sample;
    if (prepared.cubic && !cubic_forward(sample, prepared.pivot, &first)) {
        return false;
    }
    if (!finite(first)) return false;
    const double clamped_first = clamp01(first);
    if (!finite(clamped_first)) return false;
    return log_forward(clamped_first, prepared, result);
}

struct LutContext {
    Stage3CurvePrepared prepared;
    bool inverse;
};

bool evaluate_lut_sample(double sample, const void* opaque,
                         double* result) noexcept {
    if (opaque == nullptr || result == nullptr) return false;
    const auto* context = static_cast<const LutContext*>(opaque);
    return evaluate_valid(sample, context->prepared, context->inverse, result);
}

}  // namespace

Stage3CurveStatus prepare_stage3_curve(
    double pivot, bool cubic, Stage3CurvePrepared* out) noexcept {
    AddressRange out_range{};
    if (!address_range(out, sizeof(Stage3CurvePrepared),
                       alignof(Stage3CurvePrepared), &out_range)) {
        return Stage3CurveStatus::invalid_argument;
    }
    Stage3CurvePrepared candidate{};
    if (!derive_prepared(pivot, cubic, &candidate)) {
        return Stage3CurveStatus::invalid_parameters;
    }
    *out = candidate;
    return Stage3CurveStatus::ok;
}

Stage3CurveStatus evaluate_stage3_curve(
    double sample, const Stage3CurvePrepared& prepared, bool inverse,
    double* out) noexcept {
    AddressRange prepared_range{};
    AddressRange out_range{};
    if (!address_range(&prepared, sizeof(Stage3CurvePrepared),
                       alignof(Stage3CurvePrepared), &prepared_range) ||
        !address_range(out, sizeof(double), alignof(double), &out_range)) {
        return Stage3CurveStatus::invalid_argument;
    }
    if (overlaps(prepared_range, out_range)) return Stage3CurveStatus::overlap;
    if (!finite(sample) || !valid_prepared(prepared)) {
        return Stage3CurveStatus::invalid_parameters;
    }
    double candidate = 0.0;
    if (!evaluate_valid(sample, prepared, inverse, &candidate) ||
        !finite(candidate)) {
        return Stage3CurveStatus::invalid_parameters;
    }
    *out = candidate;
    return Stage3CurveStatus::ok;
}

Stage3CurveStatus build_stage3_curve_lut(
    const Stage3CurvePrepared& prepared, bool inverse,
    FloatLutPrepared* out) noexcept {
    AddressRange prepared_range{};
    AddressRange output_range{};
    if (!address_range(&prepared, sizeof(Stage3CurvePrepared),
                       alignof(Stage3CurvePrepared), &prepared_range) ||
        !address_range(out, sizeof(FloatLutPrepared), alignof(FloatLutPrepared),
                       &output_range)) {
        return Stage3CurveStatus::invalid_argument;
    }
    if (overlaps(prepared_range, output_range)) return Stage3CurveStatus::overlap;
    if (!valid_prepared(prepared)) return Stage3CurveStatus::invalid_parameters;

    const LutContext context{prepared, inverse};
    AddressRange context_range{};
    if (!address_range(&context, sizeof(context), alignof(LutContext),
                       &context_range)) {
        return Stage3CurveStatus::invalid_argument;
    }
    if (overlaps(context_range, output_range)) return Stage3CurveStatus::overlap;

    const auto status = build_float_lut_01300490(
        &evaluate_lut_sample, &context, out);
    switch (status) {
        case FloatLutStatus::ok:
            return Stage3CurveStatus::ok;
        case FloatLutStatus::invalid_argument:
            return Stage3CurveStatus::invalid_argument;
        case FloatLutStatus::invalid_parameters:
            return Stage3CurveStatus::invalid_parameters;
        case FloatLutStatus::overlap:
            return Stage3CurveStatus::overlap;
        case FloatLutStatus::evaluation_failure:
            return Stage3CurveStatus::evaluation_failure;
    }
    return Stage3CurveStatus::evaluation_failure;
}

}  // namespace imprint::acr
