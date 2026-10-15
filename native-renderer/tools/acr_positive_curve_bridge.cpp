#include "../src/acr_negative_curve.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

bool valid_range(const void *pointer, std::size_t bytes) noexcept {
    if (bytes == 0) return true;
    if (pointer == nullptr) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    return bytes <= std::numeric_limits<std::uintptr_t>::max() - start;
}

}  // namespace

extern "C" int acr_test_evaluate_positive_exposure_curve_scalar(
    double x, double exposure_ev, double *output) noexcept {
    if (!valid_range(output, sizeof(double)) || output == nullptr) return 1;

    double result = 0.0;
    const auto status = imprint::acr::evaluate_positive_exposure_curve_scalar(
        x, exposure_ev, result);
    if (status != imprint::acr::PositiveCurveStatus::ok) {
        return 10 + static_cast<int>(status);
    }
    *output = result;
    return 0;
}

extern "C" int acr_test_build_positive_exposure_lut(
    double exposure_ev, float *output, std::size_t output_count) noexcept {
    if (output_count != imprint::acr::kPositiveExposureLutSize ||
        output_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return 1;
    }
    const std::size_t bytes = output_count * sizeof(float);
    if (!valid_range(output, bytes) || output == nullptr) return 1;

    std::vector<float> generated;
    const auto status = imprint::acr::build_positive_exposure_lut(
        exposure_ev, generated);
    if (status != imprint::acr::PositiveCurveStatus::ok) {
        return 10 + static_cast<int>(status);
    }
    std::copy(generated.begin(), generated.end(), output);
    return 0;
}
