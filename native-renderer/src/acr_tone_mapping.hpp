#pragma once

#include "acr_tone_kernels.hpp"

#include <array>

namespace imprint::acr {

// Inputs are already prepared internal values. tone_708 / tone_70c / tone_710
// are the float values stored at those ACR offsets; they are not UI slider
// units. Current parameter tracing maps them to Shadows2012/100,
// -Highlights2012/100, and the base Clarity2012/100 lane, respectively.
struct ToneMappingParameters {
    LogSceneStats source_stats;
    LogSceneStats mask_aligned_stats;
    double ev90;
    double ev98;
    double ev100;
    float tone_708;
    float tone_70c;
    float tone_710;
};

// Five equally spaced x nodes, the five y values, and the derivative solution
// from the generic four-interval spline builder (without HSL-specific limits).
struct ToneSpline5 {
    std::array<double, 5> nodes{};
    std::array<double, 5> derivatives{};
};

// Internal scalar model recovered from 00db20e0 and 00db2d20. This is an
// inspectable research component, not a UI-facing API or an Adobe table model.
struct ToneMapping {
    float center{};
    float source_low{};
    float source_high{};
    float mask_center{};
    float mask_low{};
    float mask_high{};
    float low_node{};
    float high_node{};
    float inner_low{};
    float inner_high{};
    ToneSpline5 left_spline;
    ToneSpline5 right_spline;
    std::array<float, 4> left_affine_slopes{};
    std::array<float, 4> right_affine_slopes{};

    float tone_708{};
    float tone_70c{};
    float tone_710{};
    float ev98_plus_ev100{};
    float positive_shadow_input_slope{};
    float nonpositive_highlight_slope{};

    float threshold{1000.0f};
    float minimum_gate{1.0f};
    float width_factor{1.0f};
    float mix{};
    float gaussian_coefficient{};
};

ToneMapping build_tone_mapping(const ToneMappingParameters &parameters);

// Evaluate one log-luminance scalar. delta is the already prepared local
// delta-mask value; local3/4/5 are additive internal SH/Clarity lanes whose
// texture source and range are not defined by this component.
float evaluate_tone_mapping(const ToneMapping &model, float x, float delta,
                            float local3 = 0.0f, float local4 = 0.0f,
                            float local5 = 0.0f);

}  // namespace imprint::acr
