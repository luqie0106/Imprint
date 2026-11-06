#include "acr_profile_gain_scalar.hpp"

#include <cstdint>
#include <cstring>
#include <limits>

namespace imprint::acr {
namespace {

static_assert(sizeof(double) == sizeof(std::uint64_t),
              "profile scalar quantizer requires binary64 storage");
static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53 &&
                  std::numeric_limits<double>::max_exponent == 1024,
              "profile scalar quantizer requires IEEE-754 binary64");

constexpr std::uint64_t kSignMask = 0x8000000000000000ULL;
constexpr std::uint64_t kExponentMask = 0x7ff0000000000000ULL;
constexpr std::uint64_t kFractionMask = 0x000fffffffffffffULL;
constexpr std::uint64_t kQuietNanBit = 0x0008000000000000ULL;
constexpr std::uint64_t kExponentAllOnes = 0x7ff0000000000000ULL;

// Exact binary64 bits emitted by the recovered MOVZ/MOVK sequence.
constexpr double kScaleBack = 0x1.47ae147ae147bp-7;

}  // namespace

ProfileGainScalarStatus quantize_profile_gain_scalar(
    double input, double &output) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &input, sizeof(bits));

    const std::uint64_t exponent = bits & kExponentMask;
    if (exponent == kExponentAllOnes) {
        const std::uint64_t fraction = bits & kFractionMask;
        if (fraction == 0) {
            const double candidate = (bits & kSignMask) != 0 ? 0.0 : 2.0;
            output = candidate;
            return ProfileGainScalarStatus::ok;
        }
        if ((fraction & kQuietNanBit) != 0) {
            output = 2.0;
            return ProfileGainScalarStatus::ok;
        }
        return ProfileGainScalarStatus::signaling_nan;
    }

    const double capped = input < 2.0 ? input : 2.0;
    const double scaled = capped < 0.0 ? 0.0 : capped * 100.0;
    const double shifted = scaled + (scaled > 0.0 ? 0.5 : -0.5);
    const std::int32_t quantized = static_cast<std::int32_t>(shifted);
    const double candidate = static_cast<double>(quantized) * kScaleBack;
    output = candidate;
    return ProfileGainScalarStatus::ok;
}

}  // namespace imprint::acr
