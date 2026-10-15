#include "acr_camera_prepare.hpp"
#include "imprint_renderer.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr std::size_t kControlCount = 31;
constexpr std::size_t kOutputCount = 30;

bool overlaps(const void *left, std::size_t left_size,
              const void *right, std::size_t right_size) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start > std::numeric_limits<std::uintptr_t>::max() - left_size ||
        right_start > std::numeric_limits<std::uintptr_t>::max() - right_size) return true;
    if (left_start <= right_start) return right_start - left_start < left_size;
    return left_start - right_start < right_size;
}

}  // namespace

im_status im_native_acr_prepare_camera_stage(
    const double *controls, std::size_t control_count,
    double *output, std::size_t output_count) {
    if (controls == nullptr || output == nullptr || control_count != kControlCount ||
        output_count != kOutputCount ||
        overlaps(controls, kControlCount * sizeof(double),
                 output, kOutputCount * sizeof(double))) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    imprint::acr::CameraStageInput input;
    std::memcpy(input.camera_white.data(), controls, 3 * sizeof(double));
    std::memcpy(input.camera_to_pcs.data(), controls + 3, 9 * sizeof(double));
    std::memcpy(input.prophoto_inverse.data(), controls + 12, 9 * sizeof(double));
    std::memcpy(input.source_vector.data(), controls + 21, 3 * sizeof(double));
    for (std::size_t index = 0; index < input.calibration.size(); ++index) {
        const double value = controls[24 + index];
        if (!std::isfinite(value) || std::trunc(value) != value ||
            value < -100.0 || value > 100.0) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
        input.calibration[index] = static_cast<std::int32_t>(value);
    }
    const double shadow_tint = controls[30];
    if (!std::isfinite(shadow_tint) || std::trunc(shadow_tint) != shadow_tint ||
        shadow_tint < -100.0 || shadow_tint > 100.0) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    input.shadow_tint = static_cast<std::int32_t>(shadow_tint);

    imprint::acr::CameraStagePrepared prepared;
    const auto status = imprint::acr::prepare_camera_stage(input, prepared);
    if (status != imprint::acr::CameraStageStatus::ok) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    std::array<double, kOutputCount> result{};
    for (std::size_t index = 0; index < prepared.matrix.size(); ++index) {
        result[index] = static_cast<double>(prepared.matrix[index]);
        result[9 + index] = static_cast<double>(prepared.sorted_matrix[index]);
    }
    for (std::size_t index = 0; index < prepared.gain_q12.size(); ++index) {
        result[18 + index] = static_cast<double>(prepared.gain_q12[index]);
        result[21 + index] = static_cast<double>(prepared.gain_float[index]);
        result[24 + index] = prepared.black_compensation[index];
        result[27 + index] = static_cast<double>(prepared.sorted_plane_indices[index]);
    }
    std::memcpy(output, result.data(), result.size() * sizeof(double));
    return IM_STATUS_OK;
}
