#pragma once

#include <cstddef>

namespace imprint::acr {

enum class ExposureRgbStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_rgb = 2,
    overflow = 3,
    lut_out_of_bounds = 4,
    allocation_failure = 5,
};

struct PreparedExposureRgb {
    float gain = 1.0f;
    float lut_scale = 1.0f;
    float rational_gain = 1.0f;
    float epsilon = 1.0e-6f;
    const float *lut = nullptr;
    std::size_t lut_size = 0;
    // 01372428's Whites subset. Other optional transforms and masks remain
    // excluded. Defaults preserve the original 01372168 fast branch.
    float positive_gain = 1.0f;       // float(max(exp2(context+A0), 1))
    float white_clip_squared = 1.0f;  // square of float(009224b0 result)
    const float *whites_lut = nullptr;
    std::size_t whites_lut_size = 0;
    float whites_lut_scale = 4096.0f;
};

// Reconstructs 01372168's prepared SDR fast branch and 01372428's Whites
// subset. No local mask or optional color transforms. Surrounding context
// selection remains the caller's responsibility. Finite nonnegative RGB is
// required. Explicit source FMADDs remain fused under otherwise strict FP.
// Output is committed only after every pixel succeeds. Exact input/output
// aliasing is supported; partial aliasing and output/LUT overlap are rejected.
ExposureRgbStatus apply_prepared_exposure_rgb(
    const float *input, float *output, std::size_t pixel_count,
    const PreparedExposureRgb *prepared) noexcept;

}  // namespace imprint::acr
