#pragma once

#include <cstddef>
#include <vector>

namespace imprint::acr {

enum class ResampleCoefficientStatus : int {
    ok = 0,
    invalid_input = 1,
    unsupported_tap_count = 2,
    invalid_coefficients = 3,
    allocation_failure = 4,
};

inline constexpr std::size_t kResampleCoefficientPhases = 128;
inline constexpr std::size_t kResampleCoefficientStrideQuantum = 8;
inline constexpr std::size_t kResampleCoefficientMaximumTaps = 4096;

// Ordinary float coefficient rows for the recovered high-quality filter.
// The row stride rounds the tap count up to a multiple of eight. Padding is
// set to zero by this implementation; the saved Adobe evidence does not
// establish the padding contents. The 4096-tap limit is a local allocation
// guard, not an inferred Adobe limit. This helper does not resample pixels.
struct ResampleCoefficientTable {
    double q = 0.0;
    double scale = 0.0;
    double parameter_a = 0.0;
    double parameter_b = 0.0;
    std::size_t tap_count = 0;
    std::size_t stride = 0;
    std::vector<float> coefficients;
};

// On failure, result is left unchanged. q must be finite and positive, and
// the generated tap count must fit this helper's explicit 4096-tap resource
// limit.
ResampleCoefficientStatus build_resample_coefficients(
    double q, ResampleCoefficientTable *result) noexcept;

namespace resample_detail {

// Internal research bridge helper. A and B must come from a valid recovered
// filter constructor. This is not part of the public C ABI.
double evaluate_resample_weight(double parameter_a, double parameter_b,
                                double x) noexcept;

}  // namespace resample_detail
}  // namespace imprint::acr
