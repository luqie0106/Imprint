#include "../src/acr_fast_bayer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int acr_test_downsample_aligned_bayer_2x2(
    const std::uint16_t *input,
    std::size_t input_sample_count,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t input_row_stride,
    const std::uint8_t *pattern,
    std::uint16_t *output_rgb,
    std::size_t output_sample_count) noexcept {
    if (pattern == nullptr) {
        return static_cast<int>(imprint::acr::FastBayerStatus::invalid_argument);
    }
    std::uint8_t local_pattern[4] = {0, 0, 0, 0};
    std::memcpy(local_pattern, pattern, sizeof(local_pattern));

    std::vector<std::uint16_t> result;
    const imprint::acr::FastBayerStatus status =
        imprint::acr::downsample_aligned_bayer_2x2(
            input,
            input_sample_count,
            static_cast<std::size_t>(width),
            static_cast<std::size_t>(height),
            static_cast<std::size_t>(input_row_stride),
            local_pattern,
            &result);
    if (status != imprint::acr::FastBayerStatus::ok) {
        return static_cast<int>(status);
    }
    if (output_rgb == nullptr || output_sample_count < result.size()) {
        return static_cast<int>(imprint::acr::FastBayerStatus::buffer_too_small);
    }
    std::memcpy(output_rgb, result.data(), result.size() * sizeof(result[0]));
    return static_cast<int>(status);
}
