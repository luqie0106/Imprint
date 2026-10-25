#include "acr_u16_lut.hpp"

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

bool valid_input(const FloatLutPrepared& input) noexcept {
    for (float value : input.values) {
        if (!std::isfinite(value)) return false;
    }
    return float_bits_equal(input.values[kFloatLutSegments],
                            input.values[kFloatLutSegments + 1]);
}

}  // namespace

FloatLutStatus build_u16_lut_01300600(
    const FloatLutPrepared& input, U16LutPrepared* out) noexcept {
    AddressRange input_range{};
    AddressRange output_range{};
    if (!address_range(input.values.data(), sizeof(input.values), alignof(float),
                       &input_range) ||
        !address_range(out, sizeof(U16LutPrepared), alignof(U16LutPrepared),
                       &output_range)) {
        return FloatLutStatus::invalid_argument;
    }
    if (overlaps(input_range, output_range)) return FloatLutStatus::overlap;
    if (!valid_input(input)) return FloatLutStatus::invalid_parameters;

    constexpr double kScale = 65535.0;
    constexpr double kHalf = 0.5;
    constexpr double kInt32UpperExclusive = 2147483648.0;
    constexpr double kInt32Lower =
        static_cast<double>(std::numeric_limits<std::int32_t>::min());

    const double step = static_cast<double>(kFloatLutSegments) / kScale;
    double left = static_cast<double>(input.values[0]);
    double right = static_cast<double>(input.values[1]);
    double base = std::fma(left, kScale, kHalf);
    double slope = (right - left) * kScale;
    std::size_t index = 1;
    double fraction = +0.0;

    if (!std::isfinite(step) || !std::isfinite(left) ||
        !std::isfinite(right) || !std::isfinite(base) ||
        !std::isfinite(slope) || !std::isfinite(fraction)) {
        return FloatLutStatus::evaluation_failure;
    }

    U16LutPrepared candidate{};
    for (std::size_t i = 0; i < kU16LutEntryCount; ++i) {
        const double result = std::fma(slope, fraction, base);
        if (!std::isfinite(result) || result < kInt32Lower ||
            !(result < kInt32UpperExclusive)) {
            return FloatLutStatus::evaluation_failure;
        }

        const std::int32_t signed_value = static_cast<std::int32_t>(result);
        candidate.values[i] = static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(signed_value));

        fraction = step + fraction;
        if (!std::isfinite(fraction)) return FloatLutStatus::evaluation_failure;
        if (fraction > 1.0) {
            if (index >= kFloatLutEntryCount - 1) {
                return FloatLutStatus::evaluation_failure;
            }
            ++index;
            fraction = fraction + -1.0;
            if (!std::isfinite(fraction)) {
                return FloatLutStatus::evaluation_failure;
            }
            if (index >= kFloatLutEntryCount) {
                return FloatLutStatus::evaluation_failure;
            }
            const double next = static_cast<double>(input.values[index]);
            base = std::fma(right, kScale, kHalf);
            slope = (next - right) * kScale;
            if (!std::isfinite(next) || !std::isfinite(base) ||
                !std::isfinite(slope)) {
                return FloatLutStatus::evaluation_failure;
            }
            right = next;
        }
    }

    *out = candidate;
    return FloatLutStatus::ok;
}

FloatLutStatus apply_u16_lut_row_0132d484(
    const U16LutPrepared& lut, const std::uint16_t* source,
    std::size_t source_count, std::uint16_t* destination,
    std::size_t destination_count, std::size_t width) noexcept {
    static_assert(sizeof(std::uint16_t) == 2, "uint16_t must be 16-bit");
    if (width == 0 || source_count < width || destination_count < width ||
        width > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) ||
        width > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max()) /
                    sizeof(std::uint16_t)) {
        return FloatLutStatus::invalid_argument;
    }

    const std::size_t processed_bytes = width * sizeof(std::uint16_t);
    AddressRange lut_range{};
    AddressRange source_range{};
    AddressRange destination_range{};
    if (!address_range(&lut, sizeof(lut), alignof(U16LutPrepared), &lut_range) ||
        !address_range(source, processed_bytes, alignof(std::uint16_t),
                       &source_range) ||
        !address_range(destination, processed_bytes, alignof(std::uint16_t),
                       &destination_range)) {
        return FloatLutStatus::invalid_argument;
    }
    if (overlaps(lut_range, destination_range) ||
        (source != destination && overlaps(source_range, destination_range))) {
        return FloatLutStatus::overlap;
    }

    for (std::size_t i = 0; i < width; ++i) {
        destination[i] = lut.values[source[i]];
    }
    return FloatLutStatus::ok;
}

}  // namespace imprint::acr
