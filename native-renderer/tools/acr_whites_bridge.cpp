#include "../src/acr_whites.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

bool valid_range(const void *pointer, std::size_t bytes,
                 std::size_t alignment) noexcept {
    if (bytes == 0) return true;
    if (pointer == nullptr || alignment == 0) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    return start % alignment == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - start;
}

bool overlaps(const void *left, std::size_t left_bytes,
              const void *right, std::size_t right_bytes) noexcept {
    if (left_bytes == 0 || right_bytes == 0) return false;
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_bytes;
    return left_start - right_start < right_bytes;
}

}  // namespace

extern "C" int acr_test_build_negative_whites_lut(
    double prepared_b0, float *output, std::size_t output_count) noexcept {
    if (output_count != imprint::acr::kNegativeWhitesLutSize ||
        !valid_range(output, output_count * sizeof(float), alignof(float))) {
        return static_cast<int>(imprint::acr::WhitesStatus::invalid_input);
    }

    std::vector<float> generated;
    const auto status = imprint::acr::build_negative_whites_lut(prepared_b0, generated);
    if (status != imprint::acr::WhitesStatus::ok) return static_cast<int>(status);
    std::memcpy(output, generated.data(), generated.size() * sizeof(float));
    return static_cast<int>(imprint::acr::WhitesStatus::ok);
}

extern "C" int acr_test_prepare_white_clip(
    const double *inputs, std::size_t input_count,
    double *outputs, std::size_t output_count) noexcept {
    constexpr std::size_t kInputCount = 5;
    constexpr std::size_t kOutputCount = 6;
    if (input_count != kInputCount || output_count != kOutputCount ||
        !valid_range(inputs, kInputCount * sizeof(double), alignof(double)) ||
        !valid_range(outputs, kOutputCount * sizeof(double), alignof(double)) ||
        overlaps(inputs, kInputCount * sizeof(double),
                 outputs, kOutputCount * sizeof(double))) {
        return static_cast<int>(imprint::acr::WhitesStatus::invalid_input);
    }

    imprint::acr::WhiteClipResult result;
    const auto status = imprint::acr::prepare_white_clip_scalar(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], result);
    if (status != imprint::acr::WhitesStatus::ok) return static_cast<int>(status);

    const double staged[kOutputCount] = {
        result.clip,
        static_cast<double>(result.clip_float),
        static_cast<double>(result.clip_squared),
        result.below_one ? 1.0 : 0.0,
        result.anchor,
        result.response,
    };
    std::memcpy(outputs, staged, sizeof(staged));
    return static_cast<int>(imprint::acr::WhitesStatus::ok);
}
