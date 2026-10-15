#include "../src/acr_exposure_rgb.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr std::size_t kControlCount = 4;

bool valid_range(const void *pointer, std::size_t count,
                 std::size_t alignment) noexcept {
    if (alignment == 0) return false;
    if (count == 0) {
        return pointer == nullptr ||
               reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
    }
    if (pointer == nullptr) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return begin % alignment == 0 &&
           count <= std::numeric_limits<std::uintptr_t>::max() - begin;
}

bool overlaps(const void *left, std::size_t left_count,
              const void *right, std::size_t right_count) noexcept {
    if (left_count == 0 || right_count == 0) return false;
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    return left_begin < right_begin + right_count &&
           right_begin < left_begin + left_count;
}

}  // namespace

// Controls: gain, LUT scale, rational gain, epsilon. Input/output are
// interleaved RGB float arrays; an empty LUT disables the table stage.
extern "C" int acr_test_prepared_exposure_rgb(
    const float *input, std::size_t input_count,
    const float *controls, std::size_t control_count,
    const float *lut, std::size_t lut_count,
    float *output, std::size_t output_count) noexcept {
    if (control_count != kControlCount || input_count % 3 != 0 ||
        output_count != input_count ||
        input_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        lut_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        !valid_range(controls, kControlCount * sizeof(float), alignof(float)) ||
        !valid_range(output, output_count * sizeof(float), alignof(float))) {
        return static_cast<int>(imprint::acr::ExposureRgbStatus::invalid_argument);
    }
    if (overlaps(controls, kControlCount * sizeof(float), output,
                 output_count * sizeof(float))) {
        return static_cast<int>(imprint::acr::ExposureRgbStatus::invalid_argument);
    }

    imprint::acr::PreparedExposureRgb prepared;
    prepared.gain = controls[0];
    prepared.lut_scale = controls[1];
    prepared.rational_gain = controls[2];
    prepared.epsilon = controls[3];
    prepared.lut = lut;
    prepared.lut_size = lut_count;
    return static_cast<int>(imprint::acr::apply_prepared_exposure_rgb(
        input, output, input_count / 3, &prepared));
}
