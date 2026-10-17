#include "../src/acr_resample_vertical.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kBridgeInvalidOutput = 8;
constexpr float kSentinelA = 31.25f;
constexpr float kSentinelB = -12.5f;

bool valid_output_span(const float *output, std::size_t count) noexcept {
    if (output == nullptr ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    const auto address = reinterpret_cast<std::uintptr_t>(output);
    const auto bytes = count * sizeof(float);
    return address % alignof(float) == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

}  // namespace

extern "C" int acr_test_resample_vertical(
    const float *source, std::size_t source_count, std::size_t first_offset,
    std::size_t row_stride, const float *weights, std::size_t weight_count,
    std::size_t taps, std::size_t width, float *output,
    std::size_t output_count) noexcept {
    if (output_count != width || !valid_output_span(output, output_count))
        return kBridgeInvalidOutput;

    try {
        std::vector<float> candidate{kSentinelA, kSentinelB};
        const auto status = imprint::acr::process_resample_vertical(
            source, source_count, first_offset, row_stride, weights,
            weight_count, taps, width, candidate);
        if (status != imprint::acr::ResampleVerticalStatus::ok) {
            if (candidate.size() != 2 || candidate[0] != kSentinelA ||
                candidate[1] != kSentinelB)
                return static_cast<int>(
                    imprint::acr::ResampleVerticalStatus::allocation_failure);
            return static_cast<int>(status);
        }
        std::copy(candidate.begin(), candidate.end(), output);
        return static_cast<int>(imprint::acr::ResampleVerticalStatus::ok);
    } catch (...) {
        return static_cast<int>(
            imprint::acr::ResampleVerticalStatus::allocation_failure);
    }
}

extern "C" int acr_test_resample_vertical_result_alias() noexcept {
    try {
        const float weights[] = {0.5f, 0.25f};
        std::vector<float> values{1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f};
        auto status = imprint::acr::process_resample_vertical(
            values.data(), values.size(), 0, 3, weights, 2, 2, 3, values);
        if (status != imprint::acr::ResampleVerticalStatus::ok ||
            values != std::vector<float>{1.5f, 2.25f, 3.0f})
            return 1;

        const auto before = values;
        status = imprint::acr::process_resample_vertical(
            values.data(), values.size(), 0, 3, weights, 2, 2, 3, values);
        return status ==
                       imprint::acr::ResampleVerticalStatus::source_window_out_of_range &&
                   values == before
               ? 0
               : 2;
    } catch (...) {
        return 3;
    }
}
