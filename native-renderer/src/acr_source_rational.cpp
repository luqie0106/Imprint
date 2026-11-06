#include "acr_source_rational.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

static_assert(sizeof(std::uint32_t) == 4,
              "source rational requires 32-bit words");
static_assert(sizeof(double) == 8 &&
                  std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53 &&
                  std::numeric_limits<double>::max_exponent == 1024,
              "source rational requires IEEE-754 binary64");

constexpr std::uint32_t kU32Max = std::numeric_limits<std::uint32_t>::max();
constexpr double kU32Limit = 4294967296.0;  // 2^32, exclusive conversion bound.
constexpr std::uint32_t kApproxDenominatorForSmallValues = 1U << 30U;
constexpr std::uint32_t kApproxDenominatorForMediumValues = 1U << 15U;
constexpr double kMediumValueThreshold = 32768.0;  // 2^15.

void reduce_common_powers_of_two(SourceRational &value) noexcept {
    while (value.denominator >= 2U && (value.denominator & 1U) == 0U &&
           (value.numerator & 1U) == 0U) {
        value.numerator >>= 1U;
        value.denominator >>= 1U;
    }
}

void reduce_pre_scale_pair(SourceRational &value) noexcept {
    if (value.denominator > 1U &&
        value.numerator % value.denominator == 0U) {
        value.numerator /= value.denominator;
        value.denominator = 1U;
    }

    reduce_common_powers_of_two(value);

    while (value.denominator >= 3U && value.numerator % 3U == 0U &&
           value.denominator % 3U == 0U) {
        value.numerator /= 3U;
        value.denominator /= 3U;
    }

    while (value.denominator >= 5U && value.numerator % 5U == 0U &&
           value.denominator % 5U == 0U) {
        value.numerator /= 5U;
        value.denominator /= 5U;
    }
}

// The instruction sequence adds 0.5, clamps a negative input to zero, checks
// the exclusive 2^32 bound, then uses FCVTZU. Do not substitute std::round.
bool round_nonnegative_to_u32(double value,
                              std::uint32_t &result) noexcept {
    if (value < 0.0) value = 0.0;
    if (!(value < kU32Limit)) return false;
    result = static_cast<std::uint32_t>(value);
    return true;
}

bool approximate_scaled_pair(const SourceRational &value, double factor,
                             SourceRational &result) noexcept {
    if (value.denominator == 0U) {
        result = SourceRational{0U, 1U};
        return true;
    }

    const double numerator = static_cast<double>(value.numerator);
    const double denominator = static_cast<double>(value.denominator);
    const double ratio = numerator / denominator;
    const double scaled_value = ratio * factor;

    if (scaled_value <= 0.0) {
        result = SourceRational{0U, 1U};
        return true;
    }

    std::uint32_t approximation_denominator = 1U;
    if (scaled_value < 1.0) {
        approximation_denominator = kApproxDenominatorForSmallValues;
    } else if (scaled_value < kMediumValueThreshold) {
        approximation_denominator = kApproxDenominatorForMediumValues;
    }

    const double denominator_as_double =
        static_cast<double>(approximation_denominator);
    const double scaled_numerator = scaled_value * denominator_as_double;
    const double rounded_numerator = scaled_numerator + 0.5;
    std::uint32_t numerator_word = 0U;
    if (!round_nonnegative_to_u32(rounded_numerator, numerator_word)) {
        return false;
    }

    result = SourceRational{numerator_word, approximation_denominator};
    reduce_common_powers_of_two(result);
    return true;
}

}  // namespace

SourceRationalStatus scale_source_rational(const SourceRational &input,
                                           double factor,
                                           SourceRational &output) noexcept {
    if (!std::isfinite(factor)) {
        return SourceRationalStatus::non_finite_factor;
    }

    SourceRational candidate = input;
    if (factor <= 0.0) {
        output = SourceRational{0U, 1U};
        return SourceRationalStatus::ok;
    }

    // A positive scale leaves a zero-denominator pair untouched. This check
    // precedes all canonicalization and factor conversion in the source code.
    if (candidate.denominator == 0U) {
        output = candidate;
        return SourceRationalStatus::ok;
    }

    reduce_pre_scale_pair(candidate);

    if (factor > 1.0) {
        const double rounded_factor = factor + 0.5;
        std::uint32_t integer_factor = 0U;
        if (!round_nonnegative_to_u32(rounded_factor, integer_factor)) {
            return SourceRationalStatus::numeric_overflow;
        }

        if (static_cast<double>(integer_factor) == factor) {
            if (candidate.denominator % integer_factor == 0U) {
                candidate.denominator /= integer_factor;
            } else {
                const std::uint64_t product =
                    static_cast<std::uint64_t>(candidate.numerator) *
                    static_cast<std::uint64_t>(integer_factor);
                if (product <= static_cast<std::uint64_t>(kU32Max)) {
                    candidate.numerator =
                        static_cast<std::uint32_t>(product);
                    reduce_common_powers_of_two(candidate);
                    output = candidate;
                    return SourceRationalStatus::ok;
                }

                SourceRational approximation{};
                if (!approximate_scaled_pair(candidate, factor,
                                             approximation)) {
                    return SourceRationalStatus::numeric_overflow;
                }
                output = approximation;
                return SourceRationalStatus::ok;
            }

            reduce_common_powers_of_two(candidate);
            output = candidate;
            return SourceRationalStatus::ok;
        }

        SourceRational approximation{};
        if (!approximate_scaled_pair(candidate, factor, approximation)) {
            return SourceRationalStatus::numeric_overflow;
        }
        output = approximation;
        return SourceRationalStatus::ok;
    }

    if (factor < 1.0) {
        const double reciprocal = 1.0 / factor;
        const double rounded_reciprocal = reciprocal + 0.5;
        std::uint32_t integer_reciprocal = 0U;
        if (!round_nonnegative_to_u32(rounded_reciprocal,
                                      integer_reciprocal)) {
            return SourceRationalStatus::numeric_overflow;
        }

        const double reciprocal_of_integer =
            1.0 / static_cast<double>(integer_reciprocal);
        if (reciprocal_of_integer == factor) {
            if (candidate.numerator % integer_reciprocal == 0U) {
                candidate.numerator /= integer_reciprocal;
                reduce_common_powers_of_two(candidate);
                output = candidate;
                return SourceRationalStatus::ok;
            }

            const std::uint64_t product =
                static_cast<std::uint64_t>(candidate.denominator) *
                static_cast<std::uint64_t>(integer_reciprocal);
            if (product <= static_cast<std::uint64_t>(kU32Max)) {
                candidate.denominator = static_cast<std::uint32_t>(product);
                reduce_common_powers_of_two(candidate);
                output = candidate;
                return SourceRationalStatus::ok;
            }
        }

        SourceRational approximation{};
        if (!approximate_scaled_pair(candidate, factor, approximation)) {
            return SourceRationalStatus::numeric_overflow;
        }
        output = approximation;
        return SourceRationalStatus::ok;
    }

    // factor == 1: pre-scale 2/3/5 handling has already happened.
    reduce_common_powers_of_two(candidate);
    output = candidate;
    return SourceRationalStatus::ok;
}

}  // namespace imprint::acr
