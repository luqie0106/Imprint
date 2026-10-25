#include "acr_float_lut.hpp"

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

bool float_bits_equal(float left, float right) noexcept {
    std::uint32_t left_bits = 0;
    std::uint32_t right_bits = 0;
    static_assert(sizeof(left_bits) == sizeof(left), "float must be 32-bit");
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

bool finite(float value) noexcept {
    return std::isfinite(value);
}

bool valid_lut(const FloatLutPrepared& lut) noexcept {
    for (float value : lut.values) {
        if (!finite(value)) return false;
    }
    return float_bits_equal(lut.values[kFloatLutSegments],
                            lut.values[kFloatLutSegments + 1]);
}

bool interpolate(const FloatLutPrepared& lut, float coordinate,
                 float* result) noexcept {
    const float scaled = coordinate * 4096.0f;
    if (!finite(scaled) || !(scaled > -1.0f && scaled < 4097.0f)) return false;
    const std::int32_t signed_index = static_cast<std::int32_t>(scaled);
    const auto index = static_cast<std::size_t>(signed_index);
    const float fraction = scaled - static_cast<float>(signed_index);
    const float product = fraction * lut.values[index + 1];
    const float complement = 1.0f - fraction;
    const float value = std::fma(lut.values[index], complement, product);
    if (!finite(fraction) || !finite(product) || !finite(complement) ||
        !finite(value)) {
        return false;
    }
    *result = value;
    return true;
}

bool evaluate_sample(const FloatLutPrepared& lut, float sample,
                     bool signed_extension, float* result) noexcept {
    if (!finite(sample)) return false;

    float value = 0.0f;
    if (!signed_extension) {
        if (!interpolate(lut, sample, &value)) return false;
        *result = value;
        return true;
    }

    const float absolute = sample < 0.0f ? -sample : sample;
    const float coordinate = std::fmin(absolute, 1.0f);
    float interpolated = 0.0f;
    if (!finite(absolute) || !finite(coordinate) ||
        !interpolate(lut, coordinate, &interpolated)) {
        return false;
    }

    const float extension = absolute > 1.0f
        ? absolute + -1.0f
        : -0.0f;
    const float extended = extension + interpolated;
    if (!finite(extension) || !finite(extended)) return false;

    float zero = 0.0f;
    if (!interpolate(lut, +0.0f, &zero)) return false;
    const float twice_zero = zero + zero;
    if (!finite(twice_zero)) return false;

    value = sample < 0.0f ? twice_zero - extended : extended;
    if (!finite(value)) return false;
    *result = value;
    return true;
}

}  // namespace

FloatLutStatus build_float_lut_01300490(
    FloatLutEvaluator evaluator, const void* context,
    FloatLutPrepared* out) noexcept {
    AddressRange output_range{};
    if (evaluator == nullptr ||
        !address_range(out, sizeof(FloatLutPrepared),
                       alignof(FloatLutPrepared), &output_range)) {
        return FloatLutStatus::invalid_argument;
    }

    FloatLutPrepared candidate{};
    for (std::size_t i = 0; i <= kFloatLutSegments; ++i) {
        const double x = static_cast<double>(i) * (1.0 / 4096.0);
        double evaluated = 0.0;
        if (!evaluator(x, context, &evaluated) || !std::isfinite(evaluated) ||
            std::fabs(evaluated) >
                static_cast<double>(std::numeric_limits<float>::max())) {
            return FloatLutStatus::evaluation_failure;
        }
        const float sample = static_cast<float>(evaluated);
        if (!finite(sample)) return FloatLutStatus::evaluation_failure;
        candidate.values[i] = sample;
    }
    candidate.values[kFloatLutSegments + 1] =
        candidate.values[kFloatLutSegments];
    *out = candidate;
    return FloatLutStatus::ok;
}

FloatLutStatus apply_float_lut_row_00875edc(
    const FloatLutPrepared& lut, const float* source,
    std::size_t source_count, float* destination,
    std::size_t destination_count, std::size_t width,
    bool signed_extension) noexcept {
    if (width == 0 || source_count < width || destination_count < width ||
        width > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        width > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float)) {
        return FloatLutStatus::invalid_argument;
    }

    const std::size_t processed_bytes = width * sizeof(float);
    AddressRange lut_range{};
    AddressRange source_range{};
    AddressRange destination_range{};
    if (!address_range(&lut, sizeof(lut), alignof(FloatLutPrepared), &lut_range) ||
        !address_range(source, processed_bytes, alignof(float), &source_range) ||
        !address_range(destination, processed_bytes, alignof(float),
                       &destination_range)) {
        return FloatLutStatus::invalid_argument;
    }
    if (overlaps(lut_range, destination_range) ||
        (source != destination && overlaps(source_range, destination_range))) {
        return FloatLutStatus::overlap;
    }
    if (!valid_lut(lut)) return FloatLutStatus::invalid_parameters;

    // First validate every sample and result. No destination element is
    // written until the whole processed row has passed these checks.
    for (std::size_t i = 0; i < width; ++i) {
        float ignored = 0.0f;
        if (!evaluate_sample(lut, source[i], signed_extension, &ignored)) {
            return FloatLutStatus::invalid_parameters;
        }
    }
    for (std::size_t i = 0; i < width; ++i) {
        float value = 0.0f;
        (void)evaluate_sample(lut, source[i], signed_extension, &value);
        destination[i] = value;
    }
    return FloatLutStatus::ok;
}

}  // namespace imprint::acr
