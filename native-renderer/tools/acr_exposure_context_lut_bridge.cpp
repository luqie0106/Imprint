#include "../src/acr_exposure_context_lut.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

bool valid_output_range(const void* pointer, std::size_t size,
                        std::size_t alignment) noexcept {
    if (pointer == nullptr || alignment == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) return false;
    constexpr auto max_address = std::numeric_limits<std::uintptr_t>::max();
    return size <= static_cast<std::size_t>(max_address - begin);
}

int status_code(imprint::acr::ExposureContextLutStatus status) noexcept {
    return static_cast<int>(status);
}

}  // namespace

extern "C" {

int acr_test_evaluate_exposure_channel(double x, double a, double b, double h,
                                       double* output) noexcept {
    if (!valid_output_range(output, sizeof(double), alignof(double))) return 1;
    double value = 0.0;
    const auto status = imprint::acr::evaluate_exposure_channel_curve(
        x, a, b, h, &value);
    if (status != imprint::acr::ExposureContextLutStatus::ok) {
        return status_code(status);
    }
    std::memcpy(output, &value, sizeof(value));
    return status_code(status);
}

int acr_test_evaluate_exposure_post(double x, double p,
                                    double* output) noexcept {
    if (!valid_output_range(output, sizeof(double), alignof(double))) return 1;
    double value = 0.0;
    const auto status = imprint::acr::evaluate_exposure_post_curve(x, p, &value);
    if (status != imprint::acr::ExposureContextLutStatus::ok) {
        return status_code(status);
    }
    std::memcpy(output, &value, sizeof(value));
    return status_code(status);
}

int acr_test_build_exposure_channel(double a, double b, double h, float* output,
                                    std::size_t count) noexcept {
    constexpr std::size_t required = imprint::acr::kExposureContextLutSize;
    if (count < required ||
        required > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        !valid_output_range(output, required * sizeof(float), alignof(float))) {
        return 1;
    }
    std::vector<float> values;
    const auto status =
        imprint::acr::build_exposure_channel_lut(a, b, h, values);
    if (status != imprint::acr::ExposureContextLutStatus::ok) {
        return status_code(status);
    }
    std::memcpy(output, values.data(), required * sizeof(float));
    return status_code(status);
}

int acr_test_build_exposure_post(double p, float* output,
                                 std::size_t count) noexcept {
    constexpr std::size_t required = imprint::acr::kExposureContextLutSize;
    if (count < required ||
        required > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        !valid_output_range(output, required * sizeof(float), alignof(float))) {
        return 1;
    }
    std::vector<float> values;
    const auto status = imprint::acr::build_exposure_post_lut(p, values);
    if (status != imprint::acr::ExposureContextLutStatus::ok) {
        return status_code(status);
    }
    std::memcpy(output, values.data(), required * sizeof(float));
    return status_code(status);
}

}  // extern "C"
