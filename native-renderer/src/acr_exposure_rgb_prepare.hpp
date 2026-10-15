#pragma once

#include "acr_exposure_rgb.hpp"

#include <cstddef>
#include <vector>

namespace imprint::acr {

enum class ExposureRgbPrepareStatus : int {
    ok = 0,
    invalid_argument = 1,
    unsupported_context = 2,
    non_finite_input = 3,
    input_out_of_range = 4,
    non_finite_intermediate = 5,
    positive_curve_failure = 6,
    overflow = 7,
    allocation_failure = 8,
};

// Only the explicitly supplied SDR, no-mask/no-transform subset
// is supported. Defaults fail closed until the caller supplies that context.
struct ExposureRgbPrepareInputs {
    double a_ev = 0.0;
    double b_ev = 0.0;
    double amount58 = 0.0;
    double gray_diff_b8 = 0.0;
    double anchor690 = 0.0;
    // Final prepared float from FUN_009e02b0, not source flareepsilon.
    float pixel_epsilon = 0.0f;
    double whites_ev = 0.0;          // context+80, UI Whites / 50
    double post_tone_stat670 = 0.0;
    double post_tone_stat678 = 0.0;
    bool post_tone_stats_prepared = false;
    bool sdr_mode = false;
    bool whites_zero = false;
    bool mask_enabled = true;
    bool color_transform_enabled = true;
};

// This owns the optional positive-curve table. The core view is assembled
// inside apply() so moving this object can never leave a pointer into an old
// vector allocation.
struct PreparedExposureRgbPlan {
    double split_a_ev = 0.0;
    double split_b_ev = 0.0;
    double positive_lut_ev = 0.0;
    double rational_remainder_ev = 0.0;
    float gain = 1.0f;
    float rational_gain = 1.0f;
    float pixel_epsilon = 1.0e-6f;
    float lut_scale = 4096.0f;
    float positive_gain = 1.0f;
    float white_clip_squared = 1.0f;
    double whites_ev = 0.0;
    bool lut_enabled = false;
    std::vector<float> positive_lut;
    std::vector<float> negative_whites_lut;

    ExposureRgbStatus apply(const float *input, float *output,
                            std::size_t pixel_count) const noexcept;
};

// `output` is replaced only after every validation and optional LUT build has
// succeeded. The five double inputs and epsilon are finite and bounded to
// [-120, 120]; epsilon must additionally be positive.
ExposureRgbPrepareStatus prepare_exposure_rgb(
    const ExposureRgbPrepareInputs &inputs,
    PreparedExposureRgbPlan *output) noexcept;

}  // namespace imprint::acr
