#pragma once

#include <array>
#include <cstdint>

namespace imprint::acr {

enum class CameraStageStatus : int {
    ok = 0,
    invalid_input = 1,
    non_finite_intermediate = 2,
    invalid_compensation_denominator = 3,
    float_overflow = 4,
};

// Inputs resolved and prepared by the caller. This factory does not inspect
// image metadata, resolve white balance, or select a profile. source_vector is
// the caller-prepared AnalogBalance-derived vector; this stage only computes
// its maximum normalization used by ShadowTint compensation.
struct CameraStageInput {
    // Prepared ColorSpec CameraWhite. SetWhiteXY clamps each value to
    // [0.001, 1.0] after normalization; reciprocal multiply rounding can leave
    // the largest entry one ULP below 1.0, so max must be in [nextafter(1,0),1].
    std::array<double, 3> camera_white{};
    std::array<double, 9> camera_to_pcs{};
    std::array<double, 9> prophoto_inverse{};
    std::array<double, 3> source_vector{};
    // RedHue, RedSaturation, GreenHue, GreenSaturation, BlueHue,
    // BlueSaturation, each in [-100, 100].
    std::array<std::int32_t, 6> calibration{};
    std::int32_t shadow_tint = 0;
};

struct CameraStagePrepared {
    // M = (S * H * P * T) * diag(CameraWhite), row-major; each entry is
    // converted from double to float after the complete double calculation.
    std::array<float, 9> matrix{};
    // Same matrix with columns permuted by sorted_plane_indices for the camera
    // kernel, which consumes input planes in descending quantized gain order.
    std::array<float, 9> sorted_matrix{};
    std::array<std::uint16_t, 3> gain_q12{};
    std::array<float, 3> gain_float{};
    std::array<double, 3> black_compensation{};
    std::array<std::uint8_t, 3> sorted_plane_indices{};
};

// Reconstruct the three-color-plane camera-stage parameter factory. On any
// error, output is left byte-for-byte unchanged. Compile with FP contraction
// disabled to preserve the recovered multiply/add sequence.
CameraStageStatus prepare_camera_stage(const CameraStageInput &input,
                                       CameraStagePrepared &output) noexcept;

}  // namespace imprint::acr
