#include "acr_fast_bayer.hpp"

#include <limits>
#include <new>
#include <stdexcept>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumOutputSamples = 100'000'000;
constexpr std::size_t kChannels = 3;
constexpr std::size_t kBlockSide = 2;

bool checked_multiply(std::size_t left,
                      std::size_t right,
                      std::size_t *product) noexcept {
    if (product == nullptr) return false;
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }
    *product = left * right;
    return true;
}

bool checked_add(std::size_t left,
                 std::size_t right,
                 std::size_t *sum) noexcept {
    if (sum == nullptr) return false;
    if (left > std::numeric_limits<std::size_t>::max() - right) return false;
    *sum = left + right;
    return true;
}

// The recovered ctor maps each CFA position to the first matching entry of the
// distinct-color list and always allocates three output planes. A pattern that
// omits a plane, repeats one three times, or names a plane above 2 cannot
// describe that state, so it is rejected instead of silently written.
bool pattern_is_supported(const std::uint8_t pattern[4]) noexcept {
    bool present[kChannels] = {false, false, false};
    for (std::size_t index = 0; index < 4; ++index) {
        const std::uint8_t plane = pattern[index];
        if (plane >= kChannels) return false;
        present[plane] = true;
    }
    return present[0] && present[1] && present[2];
}

}  // namespace

FastBayerStatus downsample_aligned_bayer_2x2(
    const std::uint16_t *input,
    std::size_t input_sample_count,
    std::size_t width,
    std::size_t height,
    std::size_t input_row_stride,
    const std::uint8_t pattern[4],
    std::vector<std::uint16_t> *output_rgb) noexcept {
    if (input == nullptr || pattern == nullptr || output_rgb == nullptr) {
        return FastBayerStatus::invalid_argument;
    }
    if (width == 0 || height == 0 || (width % kBlockSide) != 0 ||
        (height % kBlockSide) != 0) {
        return FastBayerStatus::invalid_dimensions;
    }
    if (input_row_stride < width) {
        return FastBayerStatus::invalid_stride;
    }

    // Last readable sample index is (height - 1) * stride + (width - 1), so the
    // required span is one sample longer. Reject an overflow before comparing.
    std::size_t last_row_start = 0;
    if (!checked_multiply(height - 1, input_row_stride, &last_row_start)) {
        return FastBayerStatus::invalid_dimensions;
    }
    std::size_t required_samples = 0;
    if (!checked_add(last_row_start, width, &required_samples)) {
        return FastBayerStatus::invalid_dimensions;
    }
    if (input_sample_count < required_samples) {
        return FastBayerStatus::buffer_too_small;
    }
    if (!pattern_is_supported(pattern)) {
        return FastBayerStatus::invalid_pattern;
    }

    const std::size_t output_width = width / kBlockSide;
    const std::size_t output_height = height / kBlockSide;
    std::size_t output_pixels = 0;
    if (!checked_multiply(output_width, output_height, &output_pixels) ||
        !checked_multiply(output_pixels, kChannels, &output_pixels)) {
        return FastBayerStatus::invalid_dimensions;
    }
    if (output_pixels > kMaximumOutputSamples) {
        return FastBayerStatus::too_many_pixels;
    }

    std::vector<std::uint16_t> candidate;
    try {
        candidate.resize(output_pixels);
    } catch (const std::bad_alloc &) {
        return FastBayerStatus::allocation_failure;
    } catch (const std::length_error &) {
        return FastBayerStatus::allocation_failure;
    }

    for (std::size_t output_y = 0; output_y < output_height; ++output_y) {
        const std::uint16_t *row0 = input + (output_y * kBlockSide) * input_row_stride;
        const std::uint16_t *row1 = row0 + input_row_stride;
        std::uint16_t *destination = candidate.data() + output_y * output_width * kChannels;
        for (std::size_t output_x = 0; output_x < output_width; ++output_x) {
            const std::size_t block_x = output_x * kBlockSide;
            std::uint32_t sum[kChannels] = {0, 0, 0};
            std::uint32_t count[kChannels] = {0, 0, 0};
            const std::uint16_t samples[4] = {
                row0[block_x],
                row0[block_x + 1],
                row1[block_x],
                row1[block_x + 1],
            };
            for (std::size_t index = 0; index < 4; ++index) {
                const std::uint8_t plane = pattern[index];
                sum[plane] += samples[index];
                count[plane] += 1;
            }
            // Each count is at least 1 for a supported pattern, so the
            // recovered "clamp below 2 to 1" guard never changes the divisor.
            for (std::size_t channel = 0; channel < kChannels; ++channel) {
                const std::uint32_t divisor = count[channel];
                const std::uint32_t rounded = sum[channel] + (divisor >> 1);
                destination[channel] = static_cast<std::uint16_t>(rounded / divisor);
            }
            destination += kChannels;
        }
    }

    output_rgb->swap(candidate);
    return FastBayerStatus::ok;
}

}  // namespace imprint::acr
