#pragma once

#include "acr_tone_mapping.hpp"

#include <array>
#include <cstddef>

namespace imprint::acr {

inline constexpr std::size_t kTonePixelIntervals = 4096;

enum class ToneOutputMode {
    preserve_rgb = 0,
    normalized_gray = 1,
};

// Three recovered scalar RGB endpoint tables. Each has samples 0..4096 and a
// repeated endpoint at 4097 for the lookup's next-sample read.
struct TonePixelTables {
    std::array<float, kTonePixelIntervals + 2> log_low{};
    std::array<float, kTonePixelIntervals + 2> log_unit{};
    std::array<float, kTonePixelIntervals + 2> inverse_negative{};
    float pixel_epsilon{};
    float log_floor{};
    float inverse_log_floor{};
    float minimum_gray{};
};

// source_epsilon is the already selected 009e01e8 epsilon (0 < epsilon <=
// 2^-12); this component does not infer source mode. EV90 is the prepared
// scene value and must be within [-120, 120].
TonePixelTables make_tone_pixel_tables(double source_epsilon, double ev90);

// Map one already prepared, nonnegative finite ProPhoto-domain RGB pixel.
// Values above 1 are preserved through the recovered upper log branch.
// local3/4/5 are the prepared additive lanes consumed by evaluate_tone_mapping.
// The scalar gray-weight accumulation intentionally uses separate multiply
// and add operations. Whether Adobe fuses that boundary on every CPU path is
// not established by the recovered source evidence (AdobeFMA boundary).
std::array<float, 3> tone_rgb_pixel(const TonePixelTables &tables,
                                   const ToneMapping &model,
                                   std::array<float, 3> rgb, float delta,
                                   float local3 = 0.0f,
                                   float local4 = 0.0f,
                                   float local5 = 0.0f,
                                   ToneOutputMode output_mode =
                                       ToneOutputMode::preserve_rgb);

}  // namespace imprint::acr
