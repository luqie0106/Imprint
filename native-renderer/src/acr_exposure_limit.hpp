#pragma once

#include "acr_camera_kernels.hpp"

#include <array>
#include <cstddef>

namespace imprint::acr {

enum class ExposureLimitStatus : int {
    ok = 0,
    invalid_input = 1,
    camera_kernel_failure = 2,
};

struct ExposureLimitResult {
    // Linear values in the original camera-plane order, before the float cast
    // used by the camera kernel.
    std::array<double, 3> linear{};
    // Camera-kernel output after the optional EV90 float32 post-scale.
    std::array<double, 3> mapped{};
    double limit_ev88 = 0.0;
    double gray_max_log_diff_b8 = 0.0;
    double post_scale = 1.0;
};

// Computes the prepared three-plane exposure limit. `high_log_planes` are in
// original camera-plane order; `plane_order` lists those planes in the sorted
// order expected by CameraFloatParams. `output` is written only on success.
// The input/output ranges must be valid, aligned and non-overlapping.
ExposureLimitStatus prepare_acr_exposure_limit(
    const double *high_log_planes, const std::size_t *plane_order,
    const CameraFloatParams *camera, double temp_ev90,
    ExposureLimitResult *output) noexcept;

}  // namespace imprint::acr
