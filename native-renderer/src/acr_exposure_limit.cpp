#include "acr_exposure_limit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace imprint::acr {
namespace {

constexpr double kInputFloor = 0x1p-12;
constexpr double kLogLimit = 0x1p120;
constexpr double kGrayG = 0.5899999737739563;
constexpr double kGrayR = 0.30000001192092896;
constexpr double kGrayB = 0.11000001430511475;

bool valid_range(const void *pointer, std::size_t byte_count,
                 std::size_t alignment) noexcept {
    if (pointer == nullptr || alignment == 0) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    return start % alignment == 0 &&
           byte_count <= std::numeric_limits<std::uintptr_t>::max() - start;
}

bool overlaps(const void *left, std::size_t left_size,
              const void *right, std::size_t right_size) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_size;
    return left_start - right_start < right_size;
}

double log_guarded(double value) noexcept {
    if (value <= 0.0) return -12.0;
    if (value >= kLogLimit) return 120.0;
    return std::log2(value + kInputFloor);
}

}  // namespace

ExposureLimitStatus prepare_acr_exposure_limit(
    const double *high_log_planes, const std::size_t *plane_order,
    const CameraFloatParams *camera, double temp_ev90,
    ExposureLimitResult *output) noexcept {
    constexpr std::size_t kThreeDoubles = 3 * sizeof(double);
    constexpr std::size_t kThreeIndices = 3 * sizeof(std::size_t);
    if (!valid_range(high_log_planes, kThreeDoubles, alignof(double)) ||
        !valid_range(plane_order, kThreeIndices, alignof(std::size_t)) ||
        !valid_range(camera, sizeof(CameraFloatParams), alignof(CameraFloatParams)) ||
        !valid_range(output, sizeof(ExposureLimitResult), alignof(ExposureLimitResult))) {
        return ExposureLimitStatus::invalid_input;
    }
    if (overlaps(high_log_planes, kThreeDoubles, output, sizeof(*output)) ||
        overlaps(plane_order, kThreeIndices, output, sizeof(*output)) ||
        overlaps(camera, sizeof(*camera), output, sizeof(*output)) ||
        !std::isfinite(temp_ev90) || temp_ev90 < -120.0 || temp_ev90 > 120.0) {
        return ExposureLimitStatus::invalid_input;
    }

    std::array<double, 3> logs{};
    std::array<std::size_t, 3> order{};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(high_log_planes[channel])) {
            return ExposureLimitStatus::invalid_input;
        }
        logs[channel] = std::clamp(high_log_planes[channel], -120.0, 120.0);
        order[channel] = plane_order[channel];
        if (order[channel] > 2) return ExposureLimitStatus::invalid_input;
    }
    if (order[0] == order[1] || order[0] == order[2] || order[1] == order[2]) {
        return ExposureLimitStatus::invalid_input;
    }

    const CameraFloatParams camera_copy = *camera;
    ExposureLimitResult result;
    std::array<float, 3> sorted_input{};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const double linear = std::max(std::exp2(logs[channel]) - kInputFloor, 0.0);
        const float linear_float = static_cast<float>(linear);
        if (!std::isfinite(linear_float)) return ExposureLimitStatus::invalid_input;
        result.linear[channel] = linear;
        sorted_input[channel] = static_cast<float>(
            std::max(std::exp2(logs[order[channel]]) - kInputFloor, 0.0));
        if (!std::isfinite(sorted_input[channel])) return ExposureLimitStatus::invalid_input;
    }

    std::array<float, 3> camera_output{};
    try {
        apply_camera_float(sorted_input.data(), camera_output.data(), 1, camera_copy);
    } catch (...) {
        return ExposureLimitStatus::camera_kernel_failure;
    }

    float post_scale = 1.0f;
    if (temp_ev90 != 0.0) {
        post_scale = static_cast<float>(std::exp2(temp_ev90));
        if (!std::isfinite(post_scale) || post_scale <= 0.0f) {
            return ExposureLimitStatus::invalid_input;
        }
        for (float &value : camera_output) {
            value = value * post_scale;
            if (!std::isfinite(value)) return ExposureLimitStatus::camera_kernel_failure;
        }
    }

    result.post_scale = static_cast<double>(post_scale);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t channel = 0; channel < 3; ++channel) {
        result.mapped[channel] = static_cast<double>(camera_output[channel]);
        maximum = std::max(maximum, result.mapped[channel]);
    }

    const double gray =
        (result.mapped[1] * kGrayG + result.mapped[0] * kGrayR) +
        result.mapped[2] * kGrayB;
    if (!std::isfinite(maximum) || !std::isfinite(gray)) {
        return ExposureLimitStatus::camera_kernel_failure;
    }
    result.limit_ev88 = -log_guarded(maximum);
    result.gray_max_log_diff_b8 = log_guarded(gray) - log_guarded(maximum);
    if (!std::isfinite(result.limit_ev88) ||
        !std::isfinite(result.gray_max_log_diff_b8)) {
        return ExposureLimitStatus::camera_kernel_failure;
    }

    *output = result;
    return ExposureLimitStatus::ok;
}

}  // namespace imprint::acr
