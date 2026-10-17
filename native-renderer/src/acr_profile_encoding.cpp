#include "acr_profile_encoding.hpp"

#include <cmath>

namespace imprint::acr {
namespace {

// Exact binary64 constants recovered from 01305764 / 013057ec.
constexpr double kEncodeThreshold = 0x1.9a5c37387b719p-9;
constexpr double kEncodeExponent = 0x1.aaaaaaaaaaaabp-2;
constexpr double kEncodeSlope = 0x1.9d70a3d70a3d7p+3;
constexpr double kEncodeMultiplier = 0x1.0e147ae147ae1p+0;
constexpr double kEncodeOffset = -0x1.c28f5c28f5c29p-5;
constexpr double kDecodeThreshold = 0x1.4b5daa07d0fa8p-5;
constexpr double kDecodeOffset = 0x1.c28f5c28f5c29p-5;
constexpr double kDecodeMultiplier = 0x1.e54edcd0aeb60p-1;
constexpr double kDecodeExponent = 0x1.3333333333333p+1;
constexpr double kDecodeSlope = 0x1.3d0722149b580p-4;

double encode(double x) noexcept {
    if (x <= kEncodeThreshold) return x * kEncodeSlope;
    return std::fma(std::pow(x, kEncodeExponent),
                    kEncodeMultiplier, kEncodeOffset);
}

double decode(double x) noexcept {
    if (x <= kDecodeThreshold) return x * kDecodeSlope;
    const double shifted = x + kDecodeOffset;
    const double base = shifted * kDecodeMultiplier;
    return std::pow(base, kDecodeExponent);
}

}  // namespace

const ProfileSrgbTables &profile_srgb_tables() noexcept {
    static const ProfileSrgbTables tables = [] {
        ProfileSrgbTables result{};
        const double reciprocal = 1.0 / static_cast<double>(kProfileSrgbIntervals);
        for (std::size_t i = 0; i <= kProfileSrgbIntervals; ++i) {
            const double x = reciprocal * static_cast<double>(i);
            result.forward[i] = static_cast<float>(encode(x));
            result.inverse[i] = static_cast<float>(decode(x));
        }
        result.forward[kProfileSrgbIntervals + 1] = result.forward[kProfileSrgbIntervals];
        result.inverse[kProfileSrgbIntervals + 1] = result.inverse[kProfileSrgbIntervals];
        return result;
    }();
    return tables;
}

}  // namespace imprint::acr
