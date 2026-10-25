#include "acr_noise_transform.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

bool address_range(const void* pointer, std::size_t size, std::size_t alignment,
                   AddressRange* out) noexcept {
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

bool finite(double value) noexcept {
    return std::isfinite(value);
}

bool derive_prepared(double slope, double offset, double input_black,
                     double output_black,
                     NoiseTransformPrepared* out) noexcept {
    if (!finite(slope) || !finite(offset) || !finite(input_black) ||
        !finite(output_black) || !(slope > 0.0) || !(offset >= 0.0) ||
        !(input_black >= 0.0 && input_black < 1.0) ||
        !(output_black >= 0.0 && output_black < 1.0)) {
        return false;
    }

    const double baseline_radical = std::fma(slope, +0.0, offset);
    if (!finite(baseline_radical) || baseline_radical < 0.0) return false;
    const double baseline_root = std::sqrt(baseline_radical);
    if (!finite(baseline_root)) return false;
    const double baseline_sum = baseline_root + baseline_root;
    if (!finite(baseline_sum)) return false;
    const double baseline = baseline_sum / slope;
    if (!finite(baseline)) return false;

    const double scale_radical = slope + offset;
    if (!finite(scale_radical) || scale_radical < 0.0) return false;
    const double scale_root = std::sqrt(scale_radical);
    if (!finite(scale_root)) return false;
    const double scale_sum = scale_root + scale_root;
    if (!finite(scale_sum)) return false;
    const double scale_denominator = scale_sum / slope;
    if (!finite(scale_denominator)) return false;
    const double scale_delta = scale_denominator - baseline;
    if (!finite(scale_delta) || scale_delta == 0.0) return false;
    const double scale = 1.0 / scale_delta;
    if (!finite(scale) || !(scale > 0.0)) return false;

    const double input_black_delta = input_black + -1.0;
    if (!finite(input_black_delta) || input_black_delta == 0.0) return false;
    const double input_reciprocal = -1.0 / input_black_delta;
    const double input_range = -input_black_delta;
    if (!finite(input_reciprocal) || !finite(input_range) ||
        !(input_reciprocal > 0.0) || !(input_range > 0.0)) {
        return false;
    }

    const double output_black_delta = output_black + -1.0;
    if (!finite(output_black_delta) || output_black_delta == 0.0) return false;
    const double output_reciprocal = -1.0 / output_black_delta;
    const double output_range = -output_black_delta;
    if (!finite(output_reciprocal) || !finite(output_range) ||
        !(output_reciprocal > 0.0) || !(output_range > 0.0)) {
        return false;
    }

    const NoiseTransformPrepared candidate{
        slope, offset, baseline, scale, input_black, output_black,
        input_reciprocal, input_range, output_reciprocal, output_range};
    if (!finite(candidate.slope) || !finite(candidate.offset) ||
        !finite(candidate.baseline) || !finite(candidate.scale) ||
        !finite(candidate.input_black) || !finite(candidate.output_black) ||
        !finite(candidate.input_reciprocal) || !finite(candidate.input_range) ||
        !finite(candidate.output_reciprocal) || !finite(candidate.output_range)) {
        return false;
    }
    *out = candidate;
    return true;
}

bool same_value(double left, double right) noexcept {
    if (left != right) return false;
    return left != 0.0 || std::signbit(left) == std::signbit(right);
}

bool valid_prepared(const NoiseTransformPrepared& prepared) noexcept {
    NoiseTransformPrepared expected{};
    if (!derive_prepared(prepared.slope, prepared.offset, prepared.input_black,
                         prepared.output_black, &expected)) {
        return false;
    }
    return same_value(prepared.baseline, expected.baseline) &&
           same_value(prepared.scale, expected.scale) &&
           same_value(prepared.input_reciprocal, expected.input_reciprocal) &&
           same_value(prepared.input_range, expected.input_range) &&
           same_value(prepared.output_reciprocal, expected.output_reciprocal) &&
           same_value(prepared.output_range, expected.output_range);
}

double clamp01(double value) noexcept {
    const double upper = std::fmin(value, 1.0);
    return upper < 0.0 ? +0.0 : upper;
}

NoiseTransformStatus validate_apply(double sample,
                                    const NoiseTransformPrepared& prepared,
                                    double* out) noexcept {
    AddressRange prepared_range{};
    AddressRange out_range{};
    if (!address_range(&prepared, sizeof(prepared), alignof(NoiseTransformPrepared),
                       &prepared_range) ||
        !address_range(out, sizeof(double), alignof(double), &out_range)) {
        return NoiseTransformStatus::invalid_argument;
    }
    if (overlaps(prepared_range, out_range)) return NoiseTransformStatus::overlap;
    if (!finite(sample) || !valid_prepared(prepared)) {
        return NoiseTransformStatus::invalid_parameters;
    }
    return NoiseTransformStatus::ok;
}

}  // namespace

NoiseTransformStatus prepare_noise_transform_00b31b40(
    double slope, double offset, double input_black, double output_black,
    NoiseTransformPrepared* out) noexcept {
    AddressRange out_range{};
    if (!address_range(out, sizeof(NoiseTransformPrepared),
                       alignof(NoiseTransformPrepared), &out_range)) {
        return NoiseTransformStatus::invalid_argument;
    }

    NoiseTransformPrepared candidate{};
    if (!derive_prepared(slope, offset, input_black, output_black, &candidate)) {
        return NoiseTransformStatus::invalid_parameters;
    }
    *out = candidate;
    return NoiseTransformStatus::ok;
}

NoiseTransformStatus apply_noise_transform_forward_00b4bb88(
    double sample, const NoiseTransformPrepared& prepared, double* out) noexcept {
    const auto status = validate_apply(sample, prepared, out);
    if (status != NoiseTransformStatus::ok) return status;

    const double input_delta = sample - prepared.input_black;
    if (!finite(input_delta)) return NoiseTransformStatus::invalid_parameters;
    const double z = std::fma(prepared.input_reciprocal, input_delta, +0.0);
    if (!finite(z)) return NoiseTransformStatus::invalid_parameters;

    const double radicand = z >= 0.0
        ? std::fma(prepared.slope, z, prepared.offset)
        : std::fma(-prepared.slope, z, prepared.offset);
    if (!finite(radicand) || radicand < 0.0) {
        return NoiseTransformStatus::invalid_parameters;
    }
    const double radical = std::sqrt(radicand);
    if (!finite(radical)) return NoiseTransformStatus::invalid_parameters;
    const double radical_sum = radical + radical;
    if (!finite(radical_sum)) return NoiseTransformStatus::invalid_parameters;
    const double normalized_radical = radical_sum / prepared.slope;
    if (!finite(normalized_radical)) return NoiseTransformStatus::invalid_parameters;
    const double centered = normalized_radical - prepared.baseline;
    if (!finite(centered)) return NoiseTransformStatus::invalid_parameters;
    const double scaled = prepared.scale * centered;
    if (!finite(scaled)) return NoiseTransformStatus::invalid_parameters;
    double value = clamp01(scaled);
    if (z < 0.0) value = -value;

    const double output = std::fma(prepared.output_range, value,
                                   +prepared.output_black);
    if (!finite(output)) return NoiseTransformStatus::invalid_parameters;
    const double result = clamp01(output);
    if (!finite(result)) return NoiseTransformStatus::invalid_parameters;
    *out = result;
    return NoiseTransformStatus::ok;
}

NoiseTransformStatus apply_noise_transform_inverse_00b4bc38(
    double sample, const NoiseTransformPrepared& prepared, double* out) noexcept {
    const auto status = validate_apply(sample, prepared, out);
    if (status != NoiseTransformStatus::ok) return status;

    const double output_delta = sample - prepared.output_black;
    if (!finite(output_delta)) return NoiseTransformStatus::invalid_parameters;
    const double z = std::fma(prepared.output_reciprocal, output_delta, +0.0);
    if (!finite(z)) return NoiseTransformStatus::invalid_parameters;
    const double d = z / prepared.scale;
    if (!finite(d)) return NoiseTransformStatus::invalid_parameters;

    const double minus_delta = prepared.baseline - d;
    if (!finite(minus_delta)) return NoiseTransformStatus::invalid_parameters;
    const double minus_product = prepared.slope * minus_delta;
    if (!finite(minus_product)) return NoiseTransformStatus::invalid_parameters;
    const double minus = minus_product * 0.5;
    if (!finite(minus)) return NoiseTransformStatus::invalid_parameters;

    const double plus_delta = prepared.baseline + d;
    if (!finite(plus_delta)) return NoiseTransformStatus::invalid_parameters;
    const double plus_product = prepared.slope * plus_delta;
    if (!finite(plus_product)) return NoiseTransformStatus::invalid_parameters;
    const double plus = plus_product * 0.5;
    if (!finite(plus)) return NoiseTransformStatus::invalid_parameters;

    const double low_numerator = std::fma(minus, minus, -prepared.offset);
    if (!finite(low_numerator)) return NoiseTransformStatus::invalid_parameters;
    const double low_ratio = low_numerator / prepared.slope;
    if (!finite(low_ratio)) return NoiseTransformStatus::invalid_parameters;
    const double lo = -clamp01(low_ratio);

    const double high_numerator = std::fma(plus, plus, -prepared.offset);
    if (!finite(high_numerator)) return NoiseTransformStatus::invalid_parameters;
    const double high_ratio = high_numerator / prepared.slope;
    if (!finite(high_ratio)) return NoiseTransformStatus::invalid_parameters;
    const double hi = clamp01(high_ratio);

    const double selected = z < 0.0 ? lo : hi;
    const double input = std::fma(prepared.input_range, selected,
                                  +prepared.input_black);
    if (!finite(input)) return NoiseTransformStatus::invalid_parameters;
    const double result = clamp01(input);
    if (!finite(result)) return NoiseTransformStatus::invalid_parameters;
    *out = result;
    return NoiseTransformStatus::ok;
}

}  // namespace imprint::acr
