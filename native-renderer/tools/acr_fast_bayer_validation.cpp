#include "../src/acr_fast_bayer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using imprint::acr::FastBayerStatus;

int failures = 0;

void check(bool condition, const std::string &message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

std::vector<std::uint16_t> reference(
    const std::vector<std::uint16_t> &input,
    std::size_t width,
    std::size_t height,
    std::size_t stride,
    const std::uint8_t pattern[4]) {
    const std::size_t output_width = width / 2;
    const std::size_t output_height = height / 2;
    std::vector<std::uint16_t> output(output_width * output_height * 3);
    for (std::size_t y = 0; y < output_height; ++y) {
        for (std::size_t x = 0; x < output_width; ++x) {
            std::vector<std::uint32_t> values[3];
            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t col = 0; col < 2; ++col) {
                    values[pattern[row * 2 + col]].push_back(
                        input[(2 * y + row) * stride + 2 * x + col]);
                }
            }
            for (std::size_t channel = 0; channel < 3; ++channel) {
                std::uint32_t sum = 0;
                for (const std::uint32_t value : values[channel]) sum += value;
                const std::uint32_t count =
                    static_cast<std::uint32_t>(values[channel].size());
                output[(y * output_width + x) * 3 + channel] =
                    static_cast<std::uint16_t>((sum + (count >> 1)) / count);
            }
        }
    }
    return output;
}

bool equals(const std::vector<std::uint16_t> &left,
            const std::vector<std::uint16_t> &right) {
    return left == right;
}

std::string describe(const std::vector<std::uint16_t> &values) {
    std::string text;
    for (const std::uint16_t value : values) {
        text += std::to_string(value);
        text += ' ';
    }
    return text;
}

void expect_success(const std::vector<std::uint16_t> &input,
                    std::size_t width,
                    std::size_t height,
                    std::size_t stride,
                    const std::uint8_t pattern[4],
                    const std::vector<std::uint16_t> &expected,
                    const std::string &name) {
    std::vector<std::uint16_t> output;
    const FastBayerStatus status = imprint::acr::downsample_aligned_bayer_2x2(
        input.data(), input.size(), width, height, stride, pattern, &output);
    if (status != FastBayerStatus::ok) {
        check(false, name + ": unexpected status " +
                         std::to_string(static_cast<int>(status)));
        return;
    }
    if (!equals(output, expected)) {
        check(false, name + ": got [" + describe(output) + "] expected [" +
                         describe(expected) + "]");
    }
}

void check_hand_cases() {
    // Single 2x2 RGGB block. The green plane sums to 3 + 4 = 7 (odd), so the
    // recovered half-count rounding yields (7 + 1) / 2 = 4, not 3.
    {
        const std::uint8_t pattern[4] = {0, 1, 1, 2};
        const std::vector<std::uint16_t> input = {
            2, 3,
            4, 5,
        };
        const std::vector<std::uint16_t> expected = {2, 4, 5};
        expect_success(input, 2, 2, 2, pattern, expected, "hand 2x2 RGGB");
    }
    // The same samples under the three other Bayer arrangements.
    {
        const std::vector<std::uint16_t> input = {
            2, 3,
            4, 5,
        };
        const std::uint8_t bggr[4] = {2, 1, 1, 0};
        expect_success(input, 2, 2, 2, bggr, {5, 4, 2}, "hand 2x2 BGGR");
        const std::uint8_t grbg[4] = {1, 0, 2, 1};
        expect_success(input, 2, 2, 2, grbg, {3, 4, 4}, "hand 2x2 GRBG");
        const std::uint8_t gbrg[4] = {1, 2, 0, 1};
        expect_success(input, 2, 2, 2, gbrg, {4, 4, 3}, "hand 2x2 GBRG");
    }
    // 4x4 with a padded row stride of 6 and unused sentinel samples.
    {
        const std::uint8_t pattern[4] = {0, 1, 1, 2};
        const std::vector<std::uint16_t> input = {
            10, 20, 11, 21, 0xFFFF, 0xFFFF,
            30, 40, 31, 41, 0xFFFF, 0xFFFF,
            12, 22, 13, 23, 0xFFFF, 0xFFFF,
            32, 42, 33, 43, 0xFFFF, 0xFFFF,
        };
        // Top-left: R=10, G=(20+30)/2=25, B=40.
        // Top-right: R=11, G=(21+31)/2=26, B=41.
        // Bottom-left: R=12, G=(22+32)/2=27, B=42.
        // Bottom-right: R=13, G=(23+33)/2=28, B=43.
        const std::vector<std::uint16_t> expected = {
            10, 25, 40,
            11, 26, 41,
            12, 27, 42,
            13, 28, 43,
        };
        expect_success(input, 4, 4, 6, pattern, expected, "hand 4x4 padded");
    }
    // Extremes: the odd green sum of 65535 and 0 rounds up to 32768.
    {
        const std::uint8_t pattern[4] = {0, 1, 1, 2};
        const std::vector<std::uint16_t> input = {
            65535, 65535,
            0, 0,
        };
        expect_success(input, 2, 2, 2, pattern, {65535, 32768, 0},
                       "extremes high/low");
        const std::vector<std::uint16_t> zeros = {0, 0, 0, 0};
        expect_success(zeros, 2, 2, 2, pattern, {0, 0, 0}, "extremes zero");
        const std::vector<std::uint16_t> full = {
            65535, 65535, 65535, 65535};
        expect_success(full, 2, 2, 2, pattern, {65535, 65535, 65535},
                       "extremes full scale");
    }
}

void check_input_and_alias(const std::uint8_t pattern[4]) {
    std::vector<std::uint16_t> input(8 * 4);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint16_t>((index * 977) % 65536);
    }
    const std::vector<std::uint16_t> expected = reference(input, 8, 4, 8, pattern);

    const std::vector<std::uint16_t> before = input;
    std::vector<std::uint16_t> output;
    const FastBayerStatus status = imprint::acr::downsample_aligned_bayer_2x2(
        input.data(), input.size(), 8, 4, 8, pattern, &output);
    check(status == FastBayerStatus::ok, "input preservation: status");
    check(equals(input, before), "input must not be modified");
    check(equals(output, expected), "input preservation: output");

    // Aliased run: the operator input points into the storage it replaces.
    std::vector<std::uint16_t> aliased = input;
    const FastBayerStatus alias_status =
        imprint::acr::downsample_aligned_bayer_2x2(
            aliased.data(), aliased.size(), 8, 4, 8, pattern, &aliased);
    check(alias_status == FastBayerStatus::ok, "alias: status");
    check(equals(aliased, expected), "alias: output must match non-aliased run");
}

void check_failure_atomicity(const std::uint8_t pattern[4]) {
    const std::vector<std::uint16_t> sentinel(7, 0xABCD);
    const std::vector<std::uint16_t> input(4 * 4, 42);

    auto expect_failure = [&](const char *name,
                              FastBayerStatus expected,
                              const std::uint16_t *data,
                              std::size_t count,
                              std::size_t width,
                              std::size_t height,
                              std::size_t stride,
                              const std::uint8_t *candidate_pattern) {
        std::vector<std::uint16_t> output = sentinel;
        const FastBayerStatus status = imprint::acr::downsample_aligned_bayer_2x2(
            data, count, width, height, stride, candidate_pattern, &output);
        if (status != expected) {
            check(false, std::string(name) + ": status " +
                             std::to_string(static_cast<int>(status)) +
                             " expected " +
                             std::to_string(static_cast<int>(expected)));
        }
        if (!equals(output, sentinel)) {
            check(false, std::string(name) + ": output changed on failure");
        }
    };

    expect_failure("null input", FastBayerStatus::invalid_argument, nullptr, 16,
                   4, 4, 4, pattern);
    expect_failure("zero width", FastBayerStatus::invalid_dimensions, input.data(),
                   input.size(), 0, 4, 4, pattern);
    expect_failure("odd height", FastBayerStatus::invalid_dimensions, input.data(),
                   input.size(), 4, 3, 4, pattern);
    expect_failure("short stride", FastBayerStatus::invalid_stride, input.data(),
                   input.size(), 4, 4, 3, pattern);
    expect_failure("short buffer", FastBayerStatus::buffer_too_small, input.data(),
                   15, 4, 4, 4, pattern);
    const std::uint8_t skipped_plane[4] = {0, 0, 0, 1};
    expect_failure("missing plane", FastBayerStatus::invalid_pattern, input.data(),
                   input.size(), 4, 4, 4, skipped_plane);
    const std::uint8_t out_of_range[4] = {0, 1, 2, 3};
    expect_failure("plane above two", FastBayerStatus::invalid_pattern,
                   input.data(), input.size(), 4, 4, 4, out_of_range);
    const std::uint8_t repeated_three[4] = {0, 1, 1, 1};
    expect_failure("three green samples", FastBayerStatus::invalid_pattern,
                   input.data(), input.size(), 4, 4, 4, repeated_three);
    // Padding by more than the sample count must still be rejected, and the
    // required span check must not overflow into a false pass.
    expect_failure("huge stride", FastBayerStatus::buffer_too_small, input.data(),
                   input.size(), 4, 4, 1u << 30, pattern);

    std::vector<std::uint16_t> output = sentinel;
    const FastBayerStatus null_pattern =
        imprint::acr::downsample_aligned_bayer_2x2(
            input.data(), input.size(), 4, 4, 4, nullptr, &output);
    check(null_pattern == FastBayerStatus::invalid_argument,
          "null pattern: status");
    check(equals(output, sentinel), "null pattern: output unchanged");

    std::vector<std::uint16_t> output_only;
    const FastBayerStatus null_output =
        imprint::acr::downsample_aligned_bayer_2x2(
            input.data(), input.size(), 4, 4, 4, pattern, nullptr);
    check(null_output == FastBayerStatus::invalid_argument,
          "null output: status");
}

void check_random(const std::uint8_t pattern[4], std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> dimension(1, 12);
    std::uniform_int_distribution<int> padding(0, 5);
    std::uniform_int_distribution<int> sample(0, 65535);
    std::uniform_int_distribution<int> edge(0, 9);

    const std::uint8_t arrangements[4][4] = {
        {0, 1, 1, 2},
        {2, 1, 1, 0},
        {1, 0, 2, 1},
        {1, 2, 0, 1},
    };

    std::size_t cases = 0;
    for (int iteration = 0; iteration < 400; ++iteration) {
        const std::size_t width = static_cast<std::size_t>(dimension(generator)) * 2;
        const std::size_t height = static_cast<std::size_t>(dimension(generator)) * 2;
        const std::size_t stride = width + static_cast<std::size_t>(padding(generator));
        std::vector<std::uint16_t> input(stride * height);
        for (std::size_t index = 0; index < input.size(); ++index) {
            const int roll = edge(generator);
            if (roll == 0) {
                input[index] = 0;
            } else if (roll == 1) {
                input[index] = 65535;
            } else {
                input[index] = static_cast<std::uint16_t>(sample(generator));
            }
        }
        const std::uint8_t *chosen = iteration % 5 == 4
                                         ? pattern
                                         : arrangements[iteration % 4];
        const std::vector<std::uint16_t> expected =
            reference(input, width, height, stride, chosen);
        std::vector<std::uint16_t> output;
        const FastBayerStatus status = imprint::acr::downsample_aligned_bayer_2x2(
            input.data(), input.size(), width, height, stride, chosen, &output);
        if (status != FastBayerStatus::ok) {
            check(false, "random: status " +
                             std::to_string(static_cast<int>(status)));
            return;
        }
        if (!equals(output, expected)) {
            check(false, "random: mismatch at case " + std::to_string(cases) +
                             " width=" + std::to_string(width) +
                             " height=" + std::to_string(height) +
                             " stride=" + std::to_string(stride));
            return;
        }
        ++cases;
    }
    std::printf("random cases matched the integer reference: %zu\n", cases);
}

}  // namespace

int main() {
    const std::uint8_t pattern[4] = {0, 1, 1, 2};
    check_hand_cases();
    check_input_and_alias(pattern);
    check_failure_atomicity(pattern);
    check_random(pattern, 20261018u);
    if (failures != 0) {
        std::fprintf(stderr, "%d fast-bayer checks failed\n", failures);
        return 1;
    }
    std::printf("fast-bayer aligned 2x2 checks passed\n");
    return 0;
}
