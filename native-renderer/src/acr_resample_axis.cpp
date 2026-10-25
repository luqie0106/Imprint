#include "acr_resample_axis.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace imprint::acr {
namespace {

static_assert(sizeof(std::int64_t) == sizeof(std::uint64_t),
              "axis arithmetic requires 64-bit integers");
static_assert(sizeof(std::int32_t) == sizeof(std::uint32_t),
              "axis arithmetic requires 32-bit integers");
static_assert(std::numeric_limits<std::int64_t>::min() ==
                  -std::numeric_limits<std::int64_t>::max() - 1,
              "axis arithmetic requires two's-complement int64_t");
static_assert(std::numeric_limits<std::int32_t>::min() ==
                  -std::numeric_limits<std::int32_t>::max() - 1,
              "axis arithmetic requires two's-complement int32_t");

std::uint64_t bits_of(std::int64_t value) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::int64_t int64_from_bits(std::uint64_t bits) noexcept {
    std::int64_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::int32_t int32_from_bits(std::uint32_t bits) noexcept {
    std::int32_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::int64_t fcvtzs(double value) noexcept {
    if (std::isnan(value)) return 0;

    constexpr double kInt64UpperExclusive = 0x1p63;
    if (value >= kInt64UpperExclusive) {
        return std::numeric_limits<std::int64_t>::max();
    }
    if (value <= -kInt64UpperExclusive) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return static_cast<std::int64_t>(value);
}

std::int64_t quantize_q32(double value) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const double scaled = value * 0x1p32;
    const double bias = scaled >= 0.0 ? 0.5 : -0.5;
    const double adjusted = scaled + bias;
    return fcvtzs(adjusted);
}

std::int64_t arithmetic_shift_right_one(std::int64_t value) noexcept {
    const std::uint64_t bits = bits_of(value);
    std::uint64_t shifted = bits >> 1;
    if ((bits & (std::uint64_t{1} << 63)) != 0) {
        shifted |= std::uint64_t{1} << 63;
    }
    return int64_from_bits(shifted);
}

}  // namespace

ResampleAxisStatus prepare_ordinary_resample_axis(
    double translation, double along_scale, double cross_scale,
    std::int32_t bound_start, std::int32_t bound_end,
    std::int32_t half_taps, ResampleAxisPlan &output) noexcept {
    const std::int64_t t = quantize_q32(translation);
    const std::int64_t s = quantize_q32(along_scale);
    const std::int64_t x = quantize_q32(cross_scale);
    if (x != 0) return ResampleAxisStatus::unsupported_skew;

    constexpr std::int64_t kOneQ32 = std::int64_t{1} << 32;
    ResampleAxisPlan staged{};
    if (s == kOneQ32 && x == 0 && t == 0) {
        staged.kind = ResampleAxisKind::identity;
        output = staged;
        return ResampleAxisStatus::ok;
    }

    const std::uint64_t t_bits = bits_of(t);
    if (s == kOneQ32 && (t_bits & 0xffffffffu) == 0) {
        const std::uint32_t high_word =
            static_cast<std::uint32_t>(t_bits >> 32);
        staged.kind = ResampleAxisKind::integer_translation;
        staged.integer_offset = int32_from_bits(std::uint32_t{0} - high_word);
        output = staged;
        return ResampleAxisStatus::ok;
    }

    if (half_taps <= 0) return ResampleAxisStatus::invalid_half_taps;

    const std::uint32_t n = static_cast<std::uint32_t>(half_taps);
    const std::uint32_t start = static_cast<std::uint32_t>(bound_start);
    const std::uint32_t end = static_cast<std::uint32_t>(bound_end);
    const std::uint32_t one_minus_n = std::uint32_t{1} - n;
    const std::uint32_t lower_high = start + one_minus_n;
    const std::uint32_t upper_high = end - n;

    const std::uint64_t origin_bits =
        t_bits + bits_of(arithmetic_shift_right_one(s)) +
        bits_of(arithmetic_shift_right_one(x)) +
        (static_cast<std::uint64_t>(one_minus_n) << 32) -
        std::uint64_t{0x7f000000};

    staged.kind = ResampleAxisKind::resample;
    staged.origin = int64_from_bits(origin_bits);
    staged.step = s;
    staged.lower = int64_from_bits(static_cast<std::uint64_t>(lower_high) << 32);
    staged.upper = int64_from_bits(static_cast<std::uint64_t>(upper_high) << 32);
    output = staged;
    return ResampleAxisStatus::ok;
}

}  // namespace imprint::acr
