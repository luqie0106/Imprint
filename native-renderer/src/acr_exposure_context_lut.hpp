#pragma once

#include <cstddef>
#include <vector>

namespace imprint::acr {

enum class ExposureContextLutStatus {
    ok = 0,
    invalid_argument = 1,
    non_finite_result = 2,
    allocation_failure = 3,
};

constexpr std::size_t kExposureContextLutSize = 4098;

// These evaluators reproduce the offline scalar curves only within the
// explicit finite-input domains: x in [0,1], a/b in [0,1), h in [0,0.5],
// and p in [0,1]. Channel parameters a==1 and b==1 are unsupported. The
// output pointer must be non-null, naturally aligned, and span sizeof(double)
// without uintptr_t overflow. Every non-ok result leaves *output unchanged.
ExposureContextLutStatus evaluate_exposure_channel_curve(
    double x, double a, double b, double h, double* output) noexcept;

ExposureContextLutStatus evaluate_exposure_post_curve(
    double x, double p, double* output) noexcept;

// Build is transactional: output changes only on success. Each table has
// 4097 binary64 evaluations at x=(1/4096)*i for i=0..4096, converted once to
// float; slot 4097 copies slot 4096 as the guard, for 4098 total slots.
ExposureContextLutStatus build_exposure_channel_lut(
    double a, double b, double h, std::vector<float>& output) noexcept;

ExposureContextLutStatus build_exposure_post_lut(
    double p, std::vector<float>& output) noexcept;

}  // namespace imprint::acr
