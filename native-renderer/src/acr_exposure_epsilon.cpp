#include "acr_exposure_epsilon.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

constexpr double kSourceFlareCap = 0x1p-12;
constexpr double kSourceFlareScale = 0x1p-7;
constexpr double kDefaultEpsilon = 0x1p-19;
constexpr double kMinimumEpsilon = 0x1p-120;
constexpr double kMaximumEpsilon = 0x1p-16;

double clamp_epsilon(double value) noexcept {
    value = std::fmin(value, kMaximumEpsilon);
    if (value < kMinimumEpsilon) value = kMinimumEpsilon;
    return value;
}

}  // namespace

ExposureEpsilonStatus prepare_exposure_pixel_epsilon(
    std::uint32_t colorimetric_reference, double source_flare,
    double original_ev90, float *output) noexcept {
    if (output == nullptr) return ExposureEpsilonStatus::invalid_argument;
    const auto output_address = reinterpret_cast<std::uintptr_t>(output);
    if (output_address % alignof(float) != 0 ||
        sizeof(float) >
            std::numeric_limits<std::uintptr_t>::max() - output_address ||
        !std::isfinite(source_flare) || !std::isfinite(original_ev90)) {
        return ExposureEpsilonStatus::invalid_argument;
    }

    double value = kDefaultEpsilon;
    if (colorimetric_reference == 0 && source_flare > 0.0) {
        value = std::fmin(source_flare, kSourceFlareCap) * kSourceFlareScale;
    }
    value = clamp_epsilon(value);

    if (original_ev90 != 0.0) {
        value = std::exp2(-original_ev90) * value;
    }
    value = clamp_epsilon(value);

    *output = static_cast<float>(value);
    return ExposureEpsilonStatus::ok;
}

}  // namespace imprint::acr
