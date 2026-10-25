#include "acr_camera_prepare.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace imprint::acr {
namespace {

constexpr double kHueScale = 0.0033000000000000004;
constexpr double kSaturationScale = 0.004;
constexpr double kShadowTintScale = -0.0002;
constexpr double kQ12Scale = 4096.0;

bool ranges_overlap(const void *left, std::size_t left_size,
                    const void *right, std::size_t right_size) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_size;
    return left_start - right_start < right_size;
}

bool finite_inputs(const CameraStageInput &input) noexcept {
    double camera_white_max = 0.0;
    for (const double value : input.camera_white) {
        if (!std::isfinite(value) || value < 0.001 || value > 1.0) return false;
        if (value > camera_white_max) camera_white_max = value;
    }
    if (camera_white_max < std::nextafter(1.0, 0.0)) return false;
    for (const double value : input.camera_to_pcs) {
        if (!std::isfinite(value)) return false;
    }
    for (const double value : input.prophoto_inverse) {
        if (!std::isfinite(value)) return false;
    }
    for (const double value : input.source_vector) {
        if (!std::isfinite(value) || value <= 0.0) return false;
    }
    for (const std::int32_t value : input.calibration) {
        if (value < -100 || value > 100) return false;
    }
    return input.shadow_tint >= -100 && input.shadow_tint <= 100;
}

bool checked_add(double left, double right, double &result) noexcept {
    result = left + right;
    return std::isfinite(result);
}

bool multiply_3x3(const std::array<double, 9> &left,
                  const std::array<double, 9> &right,
                  std::array<double, 9> &result) noexcept {
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double accumulator = 0.0;
            for (std::size_t inner = 0; inner < 3; ++inner) {
                // 0131f884 starts with +0 and executes one FMADD per term.
                accumulator = std::fma(left[row * 3 + inner],
                                       right[inner * 3 + column], accumulator);
                if (!std::isfinite(accumulator)) return false;
            }
            result[row * 3 + column] = accumulator;
        }
    }
    return true;
}

bool multiply_3x3_vector(const std::array<double, 9> &matrix,
                        const std::array<double, 3> &vector,
                        std::array<double, 3> &result) noexcept {
    for (std::size_t row = 0; row < 3; ++row) {
        double accumulator = 0.0;
        for (std::size_t column = 0; column < 3; ++column) {
            // The count-three path of 0131fa10 includes the initial +0 FMA.
            accumulator = std::fma(matrix[row * 3 + column], vector[column],
                                   accumulator);
            if (!std::isfinite(accumulator)) return false;
        }
        result[row] = accumulator;
    }
    return true;
}

// Direct translation of FUN_00877fa8's comparison branches. In particular,
// equality handling depends on the original sequence of strict and inclusive
// comparisons and is intentionally not replaced with a generic sort.
std::array<std::uint8_t, 3> sort_gain_indices(
    const std::array<std::uint16_t, 3> &gain) noexcept {
    std::array<std::uint8_t, 3> indices{0, 1, 2};
    std::uint32_t uVar1 = 0;
    std::uint32_t uVar10 = 0;
    std::uint32_t uVar11 = 0;
    std::uint32_t uVar12 = 0;
    std::uint32_t uVar13 = 0;
    std::uint32_t uVar14 = 0;
    std::uint32_t uVar15 = 0;

    uVar11 = gain[0];
    uVar1 = gain[1];
    if (uVar11 < uVar1) {
        uVar10 = 0;
        indices[0] = 1;
        indices[1] = 0;
        uVar14 = 1;
        uVar13 = gain[2];
        uVar12 = 2;
        if (uVar11 < uVar13) goto initial_second_swap;
    } else {
        uVar14 = 0;
        uVar10 = 1;
        uVar13 = gain[2];
        uVar12 = 2;
        uVar11 = uVar1;
        if (uVar1 < uVar13) goto initial_second_swap;
    }
    goto initial_done;

initial_second_swap:
    indices[1] = 2;
    indices[2] = static_cast<std::uint8_t>(uVar10);
    uVar11 = uVar13;
    uVar12 = uVar10;
    uVar10 = 2;

initial_done:
    uVar1 = gain[uVar14];
    if (uVar1 < uVar11) {
        indices[0] = static_cast<std::uint8_t>(uVar10);
        indices[1] = static_cast<std::uint8_t>(uVar14);
        uVar15 = uVar14;
        uVar13 = uVar14;
        uVar14 = uVar10;
        if (gain[uVar12] <= uVar1) goto after_second_stage;
        goto compare_last_pair;
    }

    uVar15 = uVar10;
    uVar13 = uVar10;
    if (uVar11 < gain[uVar12]) goto compare_last_pair;

after_second_stage:
    uVar1 = gain[uVar14];
    uVar11 = gain[uVar13];
    uVar15 = uVar12;
    if (uVar11 <= uVar1) goto compare_final_pair;
    goto set_first_pair;

set_first_pair:
    indices[0] = static_cast<std::uint8_t>(uVar13);
    indices[1] = static_cast<std::uint8_t>(uVar14);
    if (gain[uVar15] <= uVar1) goto finish_order;
    goto compare_final_pair;

compare_last_pair:
    indices[1] = static_cast<std::uint8_t>(uVar12);
    indices[2] = static_cast<std::uint8_t>(uVar15);
    uVar1 = gain[uVar14];
    uVar11 = gain[uVar12];
    uVar13 = uVar12;
    if (uVar1 < uVar11) goto set_first_pair;
    goto compare_final_pair;

compare_final_pair:
    uVar14 = uVar13;
    if (gain[uVar15] <= uVar11) goto finish_order;

    indices[1] = static_cast<std::uint8_t>(uVar15);
    indices[2] = static_cast<std::uint8_t>(uVar14);
    return indices;

finish_order:
    return indices;
}

}  // namespace

CameraStageStatus prepare_camera_stage(const CameraStageInput &input,
                                       CameraStagePrepared &output) noexcept {
    if (ranges_overlap(&input, sizeof(input), &output, sizeof(output))) {
        return CameraStageStatus::invalid_input;
    }
    if (!finite_inputs(input)) return CameraStageStatus::invalid_input;

    // Build H and S in the expression order recovered from FUN_00740ea0.
    const double h_red = static_cast<double>(input.calibration[0]) * kHueScale;
    const double s_red = static_cast<double>(input.calibration[1]) * kSaturationScale;
    const double h_green = static_cast<double>(input.calibration[2]) * kHueScale;
    const double s_green = static_cast<double>(input.calibration[3]) * kSaturationScale;
    const double h_blue = static_cast<double>(input.calibration[4]) * kHueScale;
    const double s_blue = static_cast<double>(input.calibration[5]) * kSaturationScale;

    const std::array<double, 9> hue_matrix{
        (h_green + 1.0) - h_blue, -h_green, h_blue,
        h_red, (h_blue + 1.0) - h_red, -h_blue,
        -h_red, h_green, (h_red + 1.0) - h_green,
    };
    const double one_plus_red_saturation = s_red + 1.0;
    const std::array<double, 9> saturation_matrix{
        (s_green + 1.0) + s_blue, -s_green, -s_blue,
        -s_red, one_plus_red_saturation + s_blue, -s_blue,
        -s_red, -s_green, one_plus_red_saturation + s_green,
    };

    std::array<double, 9> calibration_matrix{};
    std::array<double, 9> calibrated_to_prophoto{};
    std::array<double, 9> camera_matrix{};
    if (!multiply_3x3(saturation_matrix, hue_matrix, calibration_matrix) ||
        !multiply_3x3(calibration_matrix, input.prophoto_inverse,
                      calibrated_to_prophoto) ||
        !multiply_3x3(calibrated_to_prophoto, input.camera_to_pcs,
                      camera_matrix)) {
        return CameraStageStatus::non_finite_intermediate;
    }

    double source_max = input.source_vector[0];
    if (source_max <= input.source_vector[1]) source_max = input.source_vector[1];
    if (source_max <= input.source_vector[2]) source_max = input.source_vector[2];
    std::array<double, 3> normalized_source{};
    std::array<double, 3> tint_vector{};
    const double tint = static_cast<double>(input.shadow_tint) * kShadowTintScale;
    for (std::size_t channel = 0; channel < 3; ++channel) {
        normalized_source[channel] = input.source_vector[channel] / source_max;
        if (!std::isfinite(normalized_source[channel])) {
            return CameraStageStatus::non_finite_intermediate;
        }
    }
    // Preserve diag(n) * [t,t,t], including its off-diagonal +0 FMA terms.
    const std::array<double, 9> source_diagonal{
        normalized_source[0], 0.0, 0.0,
        0.0, normalized_source[1], 0.0,
        0.0, 0.0, normalized_source[2],
    };
    const std::array<double, 3> tint_values{tint, tint, tint};
    if (!multiply_3x3_vector(source_diagonal, tint_values, tint_vector)) {
        return CameraStageStatus::non_finite_intermediate;
    }
    std::array<double, 3> black_vector{};
    if (!multiply_3x3_vector(camera_matrix, tint_vector, black_vector)) {
        return CameraStageStatus::non_finite_intermediate;
    }
    double minimum_black = black_vector[0];
    if (black_vector[1] <= minimum_black) minimum_black = black_vector[1];
    if (black_vector[2] <= minimum_black) minimum_black = black_vector[2];
    const double compensation_denominator = 1.0 - minimum_black;
    if (!std::isfinite(compensation_denominator)) {
        return CameraStageStatus::non_finite_intermediate;
    }
    if (compensation_denominator == 0.0) {
        return CameraStageStatus::invalid_compensation_denominator;
    }

    CameraStagePrepared prepared{};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        double numerator = 0.0;
        if (!checked_add(black_vector[channel], -minimum_black, numerator)) {
            return CameraStageStatus::non_finite_intermediate;
        }
        prepared.black_compensation[channel] = numerator / compensation_denominator;
        if (!std::isfinite(prepared.black_compensation[channel])) {
            return CameraStageStatus::non_finite_intermediate;
        }

        const double rounded_gain = std::floor(kQ12Scale / input.camera_white[channel] + 0.5);
        if (!std::isfinite(rounded_gain)) return CameraStageStatus::non_finite_intermediate;
        const double clamped_gain = std::max(0.0, rounded_gain);
        const double quantized_gain = std::min(65535.0, clamped_gain);
        prepared.gain_q12[channel] = static_cast<std::uint16_t>(quantized_gain);
        prepared.gain_float[channel] =
            static_cast<float>(prepared.gain_q12[channel]) / 4096.0f;
        if (!std::isfinite(prepared.gain_float[channel])) {
            return CameraStageStatus::float_overflow;
        }
    }

    const std::array<double, 9> white_diagonal{
        input.camera_white[0], 0.0, 0.0,
        0.0, input.camera_white[1], 0.0,
        0.0, 0.0, input.camera_white[2],
    };
    std::array<double, 9> white_camera_matrix{};
    if (!multiply_3x3(camera_matrix, white_diagonal, white_camera_matrix)) {
        return CameraStageStatus::non_finite_intermediate;
    }
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            const double value = white_camera_matrix[row * 3 + column];
            if (std::fabs(value) > std::numeric_limits<float>::max()) {
                return CameraStageStatus::float_overflow;
            }
            const float converted = static_cast<float>(value);
            if (!std::isfinite(converted)) return CameraStageStatus::float_overflow;
            prepared.matrix[row * 3 + column] = converted;
        }
    }

    prepared.sorted_plane_indices = sort_gain_indices(prepared.gain_q12);
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            prepared.sorted_matrix[row * 3 + column] =
                prepared.matrix[row * 3 + prepared.sorted_plane_indices[column]];
        }
    }

    output = prepared;
    return CameraStageStatus::ok;
}

}  // namespace imprint::acr
