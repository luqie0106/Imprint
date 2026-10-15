#pragma once

#include <cstdint>

namespace imprint::acr {

// Inputs are already resolved by the caller. This modern-only preparation
// subset does not inspect file extensions or infer camera/source metadata.
struct ExposureInput {
    // The caller must apply source-specific Exposure limits before building
    // this input. This subset enforces only the defensive +/-10 EV range.
    std::int32_t manual_exposure_x100 = 0;
    std::int32_t whites_ui = 0;
    double source_exposure_ev = 0.0;
    double source_scale = 1.0;
    double source_linear_response_limit = 1.0;
    std::uint32_t strongest_gain_q12 = 4096;
    bool three_color_planes = true;
    bool hdr_edit = false;
    double limit_ev88 = 0.0;
    double profile_gain_amount = 0.0;  // prepared ProfileGainTableMap amount, context +0x58
};

struct ExposurePrepared {
    double manual_exposure_ev = 0.0;  // context +0x08
    double whites_ev = 0.0;           // context +0x80
    double source_exposure_ev = 0.0;  // resolved caller input
    double source_scale = 1.0;
    double context_ev50 = 0.0;
    double limit_ev88 = 0.0;
    double profile_gain_amount = 0.0;  // context +0x58
    double first_arg = 0.0;
    double second_arg = 0.0;
    double exposure_gain = 1.0;       // exp2(context_ev50 + second_arg)
    double ev90 = 0.0;
    double ev98 = 0.0;
    double ev100 = 0.0;
    double ev108 = 0.0;
    double ev112 = 0.0;
    double context_ev18_change = 0.0;
    double camera_scale = 1.0;        // reset by this modern path
    double rounded_source_limit = 1.0;
    std::uint32_t source_limit_u16 = 65535;
};

enum class ExposureStatus : std::uint32_t {
    ok = 0,
    manual_exposure_out_of_range,
    whites_out_of_range,
    source_exposure_out_of_range,
    source_scale_invalid,
    source_limit_out_of_range,
    strongest_gain_out_of_range,
    limit_ev88_out_of_range,
    profile_gain_out_of_range,
    result_out_of_range,
    result_not_finite,
};

// The output is assigned only after all inputs and intermediate results pass
// validation. On failure, `prepared` remains unchanged.
ExposureStatus prepare_exposure(const ExposureInput &input,
                                ExposurePrepared &prepared) noexcept;

}  // namespace imprint::acr
