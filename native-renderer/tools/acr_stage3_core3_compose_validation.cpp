#include "../src/acr_stage3_core3_compose.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using imprint::acr::Core3ComposeControls;
using imprint::acr::Core3ComposeOutput;
using imprint::acr::Core3ComposeStatus;
using imprint::acr::prepare_core3_controls_007f2124;
using imprint::acr::run_core3_prepared_f32_tile;
using imprint::acr::run_core3_prepared_u16_tile;

int failures = 0;

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::uint16_t bayer_sample(std::size_t y, std::size_t x,
                           std::uint32_t row_origin,
                           std::uint32_t column_origin,
                           std::uint16_t red, std::uint16_t green,
                           std::uint16_t blue) {
    const std::uint32_t row = (row_origin + (y & 1u)) & 1u;
    const std::uint32_t column = (column_origin + (x & 1u)) & 1u;
    if (row == 0u && column == 0u) return red;
    if (row == 1u && column == 1u) return blue;
    return green;
}

std::vector<std::uint16_t> constant_tile(std::size_t width,
                                         std::size_t height,
                                         std::size_t stride,
                                         std::uint16_t value) {
    std::vector<std::uint16_t> tile(height * stride, 0xdead);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            tile[y * stride + x] = value;
        }
    }
    return tile;
}

std::vector<std::uint16_t> rgb_bayer_tile(std::size_t width,
                                          std::size_t height,
                                          std::size_t stride,
                                          std::uint32_t row_origin,
                                          std::uint32_t column_origin) {
    std::vector<std::uint16_t> tile(height * stride, 0xdead);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            tile[y * stride + x] = bayer_sample(
                y, x, row_origin, column_origin, 10000u, 20000u, 40000u);
        }
    }
    return tile;
}

Core3ComposeControls controls_for(float mask_amount,
                                  std::uint32_t row_origin,
                                  std::uint32_t column_origin) {
    Core3ComposeControls controls;
    controls.green.t = 0.0f;
    controls.green.k = 0.0f;
    controls.green.m = mask_amount;
    controls.green.q = 0.0f;
    controls.lower = 0.25f;
    controls.upper = 0.25f;
    controls.row_origin_parity = row_origin;
    controls.column_origin_parity = column_origin;
    return controls;
}

void check_compact_output(const Core3ComposeOutput &output,
                          std::size_t width, std::size_t height,
                          std::size_t margin, const char *prefix) {
    check(output.width == width, prefix);
    check(output.height == height, prefix);
    check(output.valid_region.top == margin &&
              output.valid_region.left == margin &&
              output.valid_region.bottom == height - margin &&
              output.valid_region.right == width - margin,
          prefix);
    check(output.output_width == width - 2u * margin &&
              output.output_height == height - 2u * margin,
          prefix);
    const std::size_t expected_count =
        output.output_width * output.output_height * 3u;
    check(output.rgb.size() == expected_count, prefix);
    check(output.rgb_u16.size() == expected_count, prefix);
}

void check_constant_rgb_result(const Core3ComposeOutput &output,
                               std::uint16_t red, std::uint16_t green,
                               std::uint16_t blue, const char *prefix) {
    const std::uint16_t expected[3] = {red, green, blue};
    for (std::size_t i = 0; i < output.rgb_u16.size(); i += 3u) {
        for (std::size_t channel = 0; channel < 3u; ++channel) {
            check(output.rgb_u16[i + channel] == expected[channel], prefix);
            const float expected_normalized =
                static_cast<float>(expected[channel]) *
                0x1.0001p-16f;  // binary32 bits 0x37800080 from the format ABI
            check(std::fabs(output.rgb[i + channel] - expected_normalized) <
                      2.0e-6f,
                  prefix);
        }
    }
}

bool same_output(const Core3ComposeOutput &a, const Core3ComposeOutput &b) {
    return a.width == b.width && a.height == b.height &&
           a.valid_region.top == b.valid_region.top &&
           a.valid_region.left == b.valid_region.left &&
           a.valid_region.bottom == b.valid_region.bottom &&
           a.valid_region.right == b.valid_region.right &&
           a.output_width == b.output_width &&
           a.output_height == b.output_height && a.rgb == b.rgb &&
           a.rgb_u16 == b.rgb_u16;
}

Core3ComposeOutput sentinel_output() {
    Core3ComposeOutput output;
    output.width = 901u;
    output.height = 902u;
    output.valid_region = {11u, 12u, 13u, 14u};
    output.output_width = 77u;
    output.output_height = 88u;
    output.rgb = {1.25f, -2.0f, 3.5f};
    output.rgb_u16 = {0xabcd, 0x1234};
    return output;
}

struct OutputSnapshot {
    Core3ComposeOutput value;
    const float *rgb_data = nullptr;
    std::size_t rgb_capacity = 0;
    const std::uint16_t *rgb_u16_data = nullptr;
    std::size_t rgb_u16_capacity = 0;
};

OutputSnapshot snapshot_output(const Core3ComposeOutput &output) {
    return {output, output.rgb.data(), output.rgb.capacity(),
            output.rgb_u16.data(), output.rgb_u16.capacity()};
}

bool output_matches_snapshot_bytes(const Core3ComposeOutput &output,
                                   const OutputSnapshot &snapshot) {
    const auto &before = snapshot.value;
    if (output.width != before.width || output.height != before.height ||
        output.valid_region.top != before.valid_region.top ||
        output.valid_region.left != before.valid_region.left ||
        output.valid_region.bottom != before.valid_region.bottom ||
        output.valid_region.right != before.valid_region.right ||
        output.output_width != before.output_width ||
        output.output_height != before.output_height ||
        output.rgb.size() != before.rgb.size() ||
        output.rgb_u16.size() != before.rgb_u16.size() ||
        output.rgb.data() != snapshot.rgb_data ||
        output.rgb.capacity() != snapshot.rgb_capacity ||
        output.rgb_u16.data() != snapshot.rgb_u16_data ||
        output.rgb_u16.capacity() != snapshot.rgb_u16_capacity) {
        return false;
    }
    return (output.rgb.empty() ||
            std::memcmp(output.rgb.data(), before.rgb.data(),
                        output.rgb.size() * sizeof(float)) == 0) &&
           (output.rgb_u16.empty() ||
            std::memcmp(output.rgb_u16.data(), before.rgb_u16.data(),
                        output.rgb_u16.size() * sizeof(std::uint16_t)) == 0);
}

std::vector<float> constant_float_tile(std::size_t width,
                                       std::size_t height,
                                       std::size_t stride, float value) {
    std::vector<float> tile(height * stride,
                            std::numeric_limits<float>::quiet_NaN());
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            tile[y * stride + x] = value;
        }
    }
    return tile;
}

void test_constant_grayscale() {
    constexpr std::size_t width = 32u;
    constexpr std::size_t height = 27u;
    constexpr std::size_t stride = 40u;
    auto input = constant_tile(width, height, stride, 12345u);
    const auto original = input;
    Core3ComposeOutput output;
    const auto status = run_core3_prepared_u16_tile(
        input.data(), input.size(), width, height, stride,
        controls_for(0.0f, 0u, 0u), &output);
    check(status == Core3ComposeStatus::ok,
          "constant grayscale: composition succeeds");
    if (status == Core3ComposeStatus::ok) {
        check_compact_output(output, width, height, 8u,
                             "constant grayscale: compact region");
        check_constant_rgb_result(output, 12345u, 12345u, 12345u,
                                  "constant grayscale: preserved value");
    }
    check(input == original,
          "constant grayscale: input samples and row padding unchanged");
}

void test_constant_rgb_all_parities_and_modes() {
    constexpr std::size_t width = 36u;
    constexpr std::size_t height = 33u;
    constexpr std::size_t stride = 40u;
    for (std::uint32_t row_origin = 0; row_origin <= 1u; ++row_origin) {
        for (std::uint32_t column_origin = 0; column_origin <= 1u;
             ++column_origin) {
            for (float mask_amount : {0.0f, 0.7f}) {
                auto input = rgb_bayer_tile(width, height, stride, row_origin,
                                            column_origin);
                const auto original = input;
                Core3ComposeOutput output;
                const auto status = run_core3_prepared_u16_tile(
                    input.data(), input.size(), width, height, stride,
                    controls_for(mask_amount, row_origin, column_origin),
                    &output);
                check(status == Core3ComposeStatus::ok,
                      "constant RGB Bayer: composition succeeds");
                if (status == Core3ComposeStatus::ok) {
                    const std::size_t margin = mask_amount == 0.0f ? 8u : 12u;
                    check_compact_output(output, width, height, margin,
                                         "constant RGB Bayer: compact region");
                    check_constant_rgb_result(
                        output, 10000u, 20000u, 40000u,
                        "constant RGB Bayer: canonical RGB reconstruction");
                }
                check(input == original,
                      "constant RGB Bayer: source and padding unchanged");
            }
        }
    }
}

void test_random_field_and_repeated_result() {
    constexpr std::size_t width = 36u;
    constexpr std::size_t height = 33u;
    constexpr std::size_t stride = 40u;
    std::vector<std::uint16_t> input(height * stride, 0xeeee);
    std::uint32_t state = 0x51a7c3d9u;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            state = state * 1664525u + 1013904223u;
            input[y * stride + x] = static_cast<std::uint16_t>(state >> 16u);
        }
    }
    const auto original = input;
    Core3ComposeOutput first;
    Core3ComposeOutput second;
    Core3ComposeOutput no_refine;
    auto controls = controls_for(0.7f, 1u, 0u);
    controls.green.t = 17.0f;
    controls.green.k = 0.015f;
    controls.green.q = 0.2f;
    controls.lower = 0.1f;
    controls.upper = 0.1f;
    const auto first_status = run_core3_prepared_u16_tile(
        input.data(), input.size(), width, height, stride, controls, &first);
    const auto second_status = run_core3_prepared_u16_tile(
        input.data(), input.size(), width, height, stride, controls, &second);
    auto no_refine_controls = controls;
    no_refine_controls.green.m = 0.0f;
    const auto no_refine_status = run_core3_prepared_u16_tile(
        input.data(), input.size(), width, height, stride, no_refine_controls,
        &no_refine);
    check(first_status == Core3ComposeStatus::ok &&
              second_status == Core3ComposeStatus::ok &&
              no_refine_status == Core3ComposeStatus::ok,
          "random field: both composition branches succeed");
    if (first_status == Core3ComposeStatus::ok &&
        second_status == Core3ComposeStatus::ok &&
        no_refine_status == Core3ComposeStatus::ok) {
        check_compact_output(first, width, height, 12u,
                             "random field: compact region");
        check(same_output(first, second),
              "random field: repeated composition is bit deterministic");
        bool varies = false;
        for (std::size_t i = 3u; i < first.rgb_u16.size(); ++i) {
            if (first.rgb_u16[i] != first.rgb_u16[0]) {
                varies = true;
                break;
            }
        }
        check(varies, "random field: output retains nontrivial variation");
        bool refine_changes_output = false;
        for (std::size_t y = 12u; y < height - 12u; ++y) {
            for (std::size_t x = 12u; x < width - 12u; ++x) {
                const std::size_t refined_index =
                    ((y - 12u) * first.output_width + (x - 12u)) * 3u;
                const std::size_t plain_index =
                    ((y - 8u) * no_refine.output_width + (x - 8u)) * 3u;
                for (std::size_t channel = 0; channel < 3u; ++channel) {
                    if (first.rgb_u16[refined_index + channel] !=
                        no_refine.rgb_u16[plain_index + channel]) {
                        refine_changes_output = true;
                    }
                }
            }
        }
        check(refine_changes_output,
              "random field: nontrivial generated mask changes refined pixels");
        for (float value : first.rgb) {
            check(std::isfinite(value) && value >= 0.0f && value <= 1.0f,
                  "random field: compact float RGB is finite and normalized");
        }
    }
    check(input == original,
          "random field: source and row padding unchanged");
}

void test_minimum_dimensions() {
    {
        constexpr std::size_t width = 20u;
        constexpr std::size_t height = 17u;
        constexpr std::size_t stride = 24u;
        auto input = constant_tile(width, height, stride, 20000u);
        Core3ComposeOutput output;
        const auto status = run_core3_prepared_u16_tile(
            input.data(), input.size(), width, height, stride,
            controls_for(0.0f, 0u, 0u), &output);
        check(status == Core3ComposeStatus::ok,
              "m=0 minimum dimensions are accepted");
        if (status == Core3ComposeStatus::ok) {
            check_compact_output(output, width, height, 8u,
                                 "m=0 minimum output metadata");
        }
    }
    {
        constexpr std::size_t width = 28u;
        constexpr std::size_t height = 25u;
        constexpr std::size_t stride = 32u;
        auto input = rgb_bayer_tile(width, height, stride, 0u, 0u);
        Core3ComposeOutput output;
        const auto status = run_core3_prepared_u16_tile(
            input.data(), input.size(), width, height, stride,
            controls_for(0.7f, 0u, 0u), &output);
        check(status == Core3ComposeStatus::ok,
              "m!=0 minimum dimensions are accepted");
        if (status == Core3ComposeStatus::ok) {
            check_compact_output(output, width, height, 12u,
                                 "m!=0 minimum output metadata");
            check_constant_rgb_result(output, 10000u, 20000u, 40000u,
                                      "m!=0 minimum reconstructs constant RGB");
        }
    }
}

void test_failure_atomicity_and_invalid_inputs() {
    constexpr std::size_t width = 32u;
    constexpr std::size_t height = 27u;
    constexpr std::size_t stride = 40u;
    const auto input = constant_tile(width, height, stride, 12345u);
    const Core3ComposeControls controls = controls_for(0.0f, 0u, 0u);

    Core3ComposeOutput output = sentinel_output();
    const Core3ComposeOutput before = output;
    auto status = run_core3_prepared_u16_tile(
        input.data(), input.size(), width, height, stride + 1u, controls,
        &output);
    check(status == Core3ComposeStatus::invalid_stride,
          "bad stride is rejected");
    check(same_output(output, before), "bad stride leaves old output unchanged");

    status = run_core3_prepared_u16_tile(
        input.data(), (height - 1u) * stride + width - 1u, width, height,
        stride, controls, &output);
    check(status == Core3ComposeStatus::buffer_too_small,
          "short source capacity is rejected");
    check(same_output(output, before), "short capacity leaves old output unchanged");

    status = run_core3_prepared_u16_tile(
        input.data(), input.size(), 16u, height, 24u, controls, &output);
    check(status == Core3ComposeStatus::invalid_dimensions,
          "too-small output width is rejected");
    check(same_output(output, before),
          "bad dimensions leave old output unchanged");

    const std::size_t enormous_width =
        std::numeric_limits<std::size_t>::max() - 11u;
    const std::size_t enormous_stride =
        std::numeric_limits<std::size_t>::max() - 7u;
    status = run_core3_prepared_u16_tile(
        input.data(), std::numeric_limits<std::size_t>::max(), enormous_width,
        height, enormous_stride, controls, &output);
    check(status == Core3ComposeStatus::arithmetic_overflow,
          "overflowing declared dimensions are rejected");
    check(same_output(output, before),
          "overflow rejection leaves old output unchanged");

    Core3ComposeOutput alias = sentinel_output();
    alias.rgb_u16 = input;
    const auto alias_before = alias;
    status = run_core3_prepared_u16_tile(
        alias.rgb_u16.data(), (height - 1u) * stride + width - 1u, width,
        height, stride, controls, &alias);
    check(status == Core3ComposeStatus::buffer_too_small,
          "failed source alias call rejects short capacity");
    check(same_output(alias, alias_before),
          "failed source alias call preserves backing output vector");
}

void test_source_aliases_old_output_on_success() {
    constexpr std::size_t width = 32u;
    constexpr std::size_t height = 27u;
    constexpr std::size_t stride = 40u;
    Core3ComposeOutput output = sentinel_output();
    output.rgb_u16 = rgb_bayer_tile(width, height, stride, 1u, 1u);
    const auto original_input = output.rgb_u16;
    check((reinterpret_cast<std::uintptr_t>(output.rgb_u16.data()) & 0xFu) == 0,
          "source alias fixture has aligned row zero");
    const auto status = run_core3_prepared_u16_tile(
        output.rgb_u16.data(), output.rgb_u16.size(), width, height, stride,
        controls_for(0.0f, 1u, 1u), &output);
    check(status == Core3ComposeStatus::ok,
          "source can alias old output storage on success");
    if (status == Core3ComposeStatus::ok) {
        check_constant_rgb_result(output, 10000u, 20000u, 40000u,
                                  "source alias output is correctly composed");
    }
    check(original_input.size() == height * stride,
          "source alias fixture retained expected capacity before commit");
}

void test_f32_constant_with_nan_padding_and_non_u16_stride() {
    constexpr std::size_t width = 32u;
    constexpr std::size_t height = 27u;
    constexpr std::size_t stride = 36u;  // %4 == 0, %8 != 0
    auto input = constant_float_tile(width, height, stride, 0.25f);
    check((stride % 4u) == 0u && (stride % 8u) != 0u,
          "float stride fixture is four-aligned but not eight-aligned");
    check((reinterpret_cast<std::uintptr_t>(input.data()) & 0xFu) == 0u,
          "float constant fixture has aligned row zero");

    Core3ComposeOutput output;
    const auto status = run_core3_prepared_f32_tile(
        input.data(), input.size(), width, height, stride,
        controls_for(0.0f, 0u, 0u), &output);
    check(status == Core3ComposeStatus::ok,
          "float constant with NaN padding composes successfully");
    if (status == Core3ComposeStatus::ok) {
        check_compact_output(output, width, height, 8u,
                             "float constant: compact output metadata");
        for (std::size_t i = 0; i < output.rgb.size(); ++i) {
            check(output.rgb[i] == 0.25f,
                  "float constant: exact value passes through composition");
            check(output.rgb_u16[i] == 16384u,
                  "float constant: output encoding remains normalized u16");
        }
    }
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = width; x < stride; ++x) {
            check(std::isnan(input[y * stride + x]),
                  "float constant: row padding remains NaN and unread");
        }
    }
}

void test_f32_matches_u16_normalization_for_both_modes() {
    constexpr std::size_t width = 36u;
    constexpr std::size_t height = 33u;
    constexpr std::size_t u16_stride = 40u;
    constexpr std::size_t f32_stride = 44u;  // %4 == 0, %8 != 0
    std::vector<std::uint16_t> input_u16(height * u16_stride, 0xeeeeu);
    std::vector<float> input_f32(
        height * f32_stride, std::numeric_limits<float>::quiet_NaN());
    std::uint32_t state = 0x93ab71c5u;
    constexpr float reciprocal = 0x1.0001p-16f;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            state = state * 1664525u + 1013904223u;
            const auto sample = static_cast<std::uint16_t>(state >> 16u);
            input_u16[y * u16_stride + x] = sample;
            input_f32[y * f32_stride + x] =
                static_cast<float>(sample) * reciprocal;
        }
    }
    check((reinterpret_cast<std::uintptr_t>(input_f32.data()) & 0xFu) == 0u,
          "float normalized fixture has aligned row zero");

    for (float mask_amount : {0.0f, 0.7f}) {
        auto controls = controls_for(mask_amount, 1u, 0u);
        controls.green.t = 17.0f;
        controls.green.k = 0.015f;
        controls.green.q = 0.2f;
        controls.lower = 0.1f;
        controls.upper = 0.1f;
        Core3ComposeOutput output_u16;
        Core3ComposeOutput output_f32;
        const auto status_u16 = run_core3_prepared_u16_tile(
            input_u16.data(), input_u16.size(), width, height, u16_stride,
            controls, &output_u16);
        const auto status_f32 = run_core3_prepared_f32_tile(
            input_f32.data(), input_f32.size(), width, height, f32_stride,
            controls, &output_f32);
        check(status_u16 == Core3ComposeStatus::ok &&
                  status_f32 == Core3ComposeStatus::ok,
              "float/u16 comparison: both modes compose successfully");
        if (status_u16 == Core3ComposeStatus::ok &&
            status_f32 == Core3ComposeStatus::ok) {
            check(same_output(output_u16, output_f32),
                  "float/u16 comparison: metadata and values match");
            check(output_u16.rgb.size() == output_f32.rgb.size() &&
                      std::memcmp(output_u16.rgb.data(), output_f32.rgb.data(),
                                  output_u16.rgb.size() * sizeof(float)) == 0,
                  "float/u16 comparison: RGB floats match bit for bit");
            check(output_u16.rgb_u16.size() == output_f32.rgb_u16.size() &&
                      std::memcmp(output_u16.rgb_u16.data(),
                                  output_f32.rgb_u16.data(),
                                  output_u16.rgb_u16.size() *
                                      sizeof(std::uint16_t)) == 0,
                  "float/u16 comparison: encoded RGB matches bit for bit");
        }
    }
}

void test_f32_failure_atomicity() {
    constexpr std::size_t width = 32u;
    constexpr std::size_t height = 27u;
    constexpr std::size_t stride = 36u;
    auto input = constant_float_tile(width, height, stride, 0.25f);
    Core3ComposeOutput output = sentinel_output();
    const auto snapshot = snapshot_output(output);
    const Core3ComposeControls controls = controls_for(0.0f, 0u, 0u);

    input[5u * stride + 7u] = std::numeric_limits<float>::quiet_NaN();
    auto status = run_core3_prepared_f32_tile(
        input.data(), input.size(), width, height, stride, controls, &output);
    check(status == Core3ComposeStatus::unsupported_domain,
          "float input NaN in valid pixels is rejected");
    check(output_matches_snapshot_bytes(output, snapshot),
          "float input NaN leaves original output vectors byte-for-byte intact");

    input[5u * stride + 7u] = 0.25f;
    const std::size_t short_capacity = (height - 1u) * stride + width - 1u;
    status = run_core3_prepared_f32_tile(
        input.data(), short_capacity, width, height, stride, controls, &output);
    check(status == Core3ComposeStatus::buffer_too_small,
          "float input short capacity is rejected");
    check(output_matches_snapshot_bytes(output, snapshot),
          "float short capacity leaves original output vectors byte-for-byte intact");

    const std::size_t huge_stride =
        std::numeric_limits<std::size_t>::max() - 3u;
    status = run_core3_prepared_f32_tile(
        input.data(), std::numeric_limits<std::size_t>::max(), width, height,
        huge_stride, controls, &output);
    check(status == Core3ComposeStatus::arithmetic_overflow,
          "float input huge stride overflow is rejected");
    check(output_matches_snapshot_bytes(output, snapshot),
          "float huge stride leaves original output vectors byte-for-byte intact");
}

void test_f32_source_aliases_old_rgb_on_success() {
    constexpr std::size_t width = 32u;
    constexpr std::size_t height = 27u;
    constexpr std::size_t stride = 36u;
    Core3ComposeOutput output = sentinel_output();
    output.rgb = constant_float_tile(width, height, stride, 0.25f);
    const float *old_rgb_storage = output.rgb.data();
    check((reinterpret_cast<std::uintptr_t>(old_rgb_storage) & 0xFu) == 0u,
          "float alias fixture has aligned row zero");

    const auto status = run_core3_prepared_f32_tile(
        old_rgb_storage, output.rgb.size(), width, height, stride,
        controls_for(0.0f, 0u, 0u), &output);
    check(status == Core3ComposeStatus::ok,
          "float source can alias old output.rgb storage on success");
    if (status == Core3ComposeStatus::ok) {
        for (std::size_t i = 0; i < output.rgb.size(); ++i) {
            check(output.rgb[i] == 0.25f,
                  "float alias: output is complete after source storage replacement");
            check(output.rgb_u16[i] == 16384u,
                  "float alias: u16 encoding remains correct after commit");
        }
    }
}

void test_prepared_controls() {
    const double amounts[] = {0.0, 1.0, 6.0, 12.0, 13.0};
    const std::uint32_t expected_m[] = {
        0x00000000u, 0x00000000u, 0x3f800000u, 0x80000000u, 0x00000000u};
    const std::uint32_t expected_q[] = {
        0x3f000000u, 0x3f000000u, 0x3e04bda1u, 0x3dccccccu, 0x3dccccccu};
    for (std::size_t i = 0; i < sizeof(amounts) / sizeof(amounts[0]); ++i) {
        Core3ComposeControls controls{};
        const auto status = prepare_core3_controls_007f2124(
            amounts[i], false, 1u, 0u, &controls);
        check(status == Core3ComposeStatus::ok,
              "bounded amount controls prepare successfully");
        if (status == Core3ComposeStatus::ok) {
            check(bits(controls.green.t) == 0x41b40000u &&
                      bits(controls.green.k) == 0x3cb60b61u,
                  "prepared control t and k preserve recovered bits");
            check(bits(controls.green.m) == expected_m[i],
                  "prepared mask preserves recovered selection and signed zero");
            check(bits(controls.green.q) == expected_q[i],
                  "prepared q preserves recovered fused arithmetic result");
            check(bits(controls.lower) == 0x3c3b80bcu &&
                      bits(controls.upper) == 0x3cbb80bcu,
                  "prepared refine limits preserve recovered bits");
            check(controls.row_origin_parity == 1u &&
                      controls.column_origin_parity == 0u,
                  "prepared controls preserve origin parity");
        }

        Core3ComposeControls suppressed{};
        const auto suppressed_status = prepare_core3_controls_007f2124(
            amounts[i], true, 0u, 1u, &suppressed);
        check(suppressed_status == Core3ComposeStatus::ok &&
                  bits(suppressed.green.m) == 0u &&
                  suppressed.row_origin_parity == 0u &&
                  suppressed.column_origin_parity == 1u,
              "host suppression selects positive zero and preserves parities");
        if (suppressed_status == Core3ComposeStatus::ok) {
            check(bits(suppressed.green.q) == expected_q[i],
                  "host suppression leaves q unchanged");
        }
    }

    Core3ComposeControls output{};
    std::memset(&output, 0xa5, sizeof(output));
    unsigned char before[sizeof(output)];
    std::memcpy(before, &output, sizeof(output));
    struct RejectCase {
        double amount;
        std::uint32_t row;
        std::uint32_t column;
        Core3ComposeStatus status;
    };
    const RejectCase cases[] = {
        {std::numeric_limits<double>::quiet_NaN(), 0u, 0u,
         Core3ComposeStatus::invalid_parameters},
        {std::numeric_limits<double>::infinity(), 0u, 0u,
         Core3ComposeStatus::invalid_parameters},
        {static_cast<double>(std::numeric_limits<float>::max()) * 2.0,
         0u, 0u, Core3ComposeStatus::invalid_parameters},
        {0.0, 2u, 0u, Core3ComposeStatus::invalid_phase},
        {0.0, 0u, 2u, Core3ComposeStatus::invalid_phase},
    };
    for (const auto &test : cases) {
        const auto status = prepare_core3_controls_007f2124(
            test.amount, false, test.row, test.column, &output);
        check(status == test.status,
              "invalid controls input returns its specific status");
        check(std::memcmp(before, &output, sizeof(output)) == 0,
              "rejected controls input leaves output bytes unchanged");
    }

    alignas(Core3ComposeControls) unsigned char storage[sizeof(output) + 1u]{};
    auto *misaligned = reinterpret_cast<Core3ComposeControls *>(storage + 1u);
    check(prepare_core3_controls_007f2124(0.0, false, 0u, 0u, misaligned) ==
              Core3ComposeStatus::invalid_argument,
          "misaligned output is rejected");
    const auto near_end = std::numeric_limits<std::uintptr_t>::max() &
        ~(static_cast<std::uintptr_t>(alignof(Core3ComposeControls)) - 1u);
    check(prepare_core3_controls_007f2124(0.0, false, 0u, 0u,
              reinterpret_cast<Core3ComposeControls *>(near_end)) ==
              Core3ComposeStatus::invalid_argument,
          "wrapping output address span is rejected");
    check(prepare_core3_controls_007f2124(0.0, false, 0u, 0u, nullptr) ==
              Core3ComposeStatus::invalid_argument,
          "null output is rejected");
}

}  // namespace

int main() {
    test_constant_grayscale();
    test_constant_rgb_all_parities_and_modes();
    test_random_field_and_repeated_result();
    test_minimum_dimensions();
    test_failure_atomicity_and_invalid_inputs();
    test_source_aliases_old_output_on_success();
    test_f32_constant_with_nan_padding_and_non_u16_stride();
    test_f32_matches_u16_normalization_for_both_modes();
    test_f32_failure_atomicity();
    test_f32_source_aliases_old_rgb_on_success();
    test_prepared_controls();
    if (failures != 0) {
        std::cerr << failures << " composition validation failure(s)\n";
        return 1;
    }
    std::cout << "core3 prepared uint16/float composition validation passed\n";
    return 0;
}
