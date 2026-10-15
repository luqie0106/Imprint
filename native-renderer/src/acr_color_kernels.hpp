#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace imprint::acr {

struct WhiteTemperature {
    double temperature;
    double tint;
};
// Recovered 31-row xy -> temperature/tint projection. The surrounding API
// restricts inputs to finite physical chromaticities; Adobe's inner function
// itself does not perform these outer input checks.
WhiteTemperature white_temperature(double x, double y);

// Candidate ACR color operators for the modern SDR (HDREditMode=false) branch.
// Input and output are contiguous RGB float triplets in linear ProPhoto, with
// input components constrained to [0, 1]. These operators are not a complete
// Camera Raw pipeline and do not cover HDR processing or TIFF decoloring.
struct HslParams {
    // UI units: -100..100 for each Red, Orange, Yellow, Green, Aqua, Blue,
    // Purple, and Magenta band.
    std::array<double, 8> hue{};
    std::array<double, 8> saturation{};
    std::array<double, 8> luminance{};
    double vibrance = 0.0;  // UI units: -100..100; applied before the HSL bands.
};

struct GradingWheel {
    double hue = 0.0;         // degrees, 0..360
    double saturation = 0.0;  // 0..100
    double luminance = 0.0;   // -100..100
};

struct GradingParams {
    GradingWheel shadows{};
    GradingWheel midtones{};
    GradingWheel highlights{};
    GradingWheel global{};
    double blend = 50.0;      // 0..100
    double balance = 0.0;     // -100..100
};

// Throws std::invalid_argument for invalid pointers, parameters, input values,
// or partially overlapping buffers. Exact in-place operation is supported.
void apply_hsl(const float *rgb, float *output, std::size_t pixel_count,
               const HslParams &params);
// Rendered-input normalization branch: inverse default tone -> HSL/Vibrance
// -> default tone. This is distinct from the scene-linear apply_hsl stage.
void apply_rendered_hsl(const float *rgb, float *output, std::size_t pixel_count,
                        const HslParams &params);
void apply_grading(const float *rgb, float *output, std::size_t pixel_count,
                   const GradingParams &params);

// Recovered signed-16 CPU RGBtoYCC -> YCCtoRGB pair for basic Saturation.
// Interleaved triplets here adapt Adobe's planar storage only. This is an
// internal-stage candidate, NOT uint16 image data or a float ProPhoto API:
// the surrounding representation/encoding is still under investigation.
// Zero UI skips the pair as in the builder's neutral branch. No other YCC
// operations (local color/tone adjustments) may be interposed in this helper.
void apply_saturation_s16(const std::int16_t *rgb, std::int16_t *output,
                          std::size_t pixel_count, double ui_saturation);

}  // namespace imprint::acr
