#include "../src/acr_exposure_limit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr std::size_t kControlCount = 20;
constexpr std::size_t kOutputCount = 9;

bool valid_range(const void *pointer, std::size_t byte_count) noexcept {
    if (pointer == nullptr) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    return start % alignof(double) == 0 &&
           byte_count <= std::numeric_limits<std::uintptr_t>::max() - start;
}

bool overlaps(const void *left, std::size_t left_size,
              const void *right, std::size_t right_size) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_size;
    return left_start - right_start < right_size;
}

}  // namespace

// Controls: h[3], order[3], (A, B, U, S, matrix row-major[9]), temp_ev90.
// Output: linear[3], mapped-after-postscale[3], limit_ev88, B8, post_scale.
extern "C" int acr_test_prepared_exposure_limit(
    const double *controls, std::size_t control_count,
    double *output, std::size_t output_count) {
    if (control_count != kControlCount || output_count != kOutputCount ||
        !valid_range(controls, kControlCount * sizeof(double)) ||
        !valid_range(output, kOutputCount * sizeof(double)) ||
        overlaps(controls, kControlCount * sizeof(double),
                 output, kOutputCount * sizeof(double))) {
        return static_cast<int>(imprint::acr::ExposureLimitStatus::invalid_input);
    }

    for (std::size_t i = 0; i < kControlCount; ++i) {
        if (!std::isfinite(controls[i])) {
            return static_cast<int>(imprint::acr::ExposureLimitStatus::invalid_input);
        }
    }

    std::array<double, 3> high_log{};
    std::array<std::size_t, 3> order{};
    std::copy_n(controls, 3, high_log.begin());
    for (std::size_t i = 0; i < 3; ++i) {
        const double value = controls[3 + i];
        if (std::trunc(value) != value || value < 0.0 || value > 2.0) {
            return static_cast<int>(imprint::acr::ExposureLimitStatus::invalid_input);
        }
        order[i] = static_cast<std::size_t>(value);
    }

    imprint::acr::CameraFloatParams camera{};
    const double *camera_controls = controls + 6;
    float *camera_values[] = {&camera.gain_max, &camera.gain_middle,
                              &camera.upper, &camera.scale};
    for (std::size_t i = 0; i < 4; ++i) {
        if (std::fabs(camera_controls[i]) >
            static_cast<double>(std::numeric_limits<float>::max())) {
            return static_cast<int>(imprint::acr::ExposureLimitStatus::invalid_input);
        }
        *camera_values[i] = static_cast<float>(camera_controls[i]);
    }
    for (std::size_t i = 0; i < camera.matrix.size(); ++i) {
        const double value = camera_controls[4 + i];
        if (std::fabs(value) > static_cast<double>(std::numeric_limits<float>::max())) {
            return static_cast<int>(imprint::acr::ExposureLimitStatus::invalid_input);
        }
        camera.matrix[i] = static_cast<float>(value);
    }

    imprint::acr::ExposureLimitResult result;
    const auto status = imprint::acr::prepare_acr_exposure_limit(
        high_log.data(), order.data(), &camera, controls[19], &result);
    if (status != imprint::acr::ExposureLimitStatus::ok) {
        return static_cast<int>(status);
    }

    std::array<double, kOutputCount> values{};
    std::copy(result.linear.begin(), result.linear.end(), values.begin());
    std::copy(result.mapped.begin(), result.mapped.end(), values.begin() + 3);
    values[6] = result.limit_ev88;
    values[7] = result.gray_max_log_diff_b8;
    values[8] = result.post_scale;
    std::memcpy(output, values.data(), values.size() * sizeof(double));
    return 0;
}
