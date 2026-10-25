#include "acr_stage3_prepared_image.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using imprint::acr::BayerRepeatRect;
using imprint::acr::PreparedImageControls;
using imprint::acr::PreparedImageOutput;
using imprint::acr::PreparedImageStatus;

constexpr std::size_t kWidth = 9;
constexpr std::size_t kHeight = 7;
constexpr std::size_t kStride = 14;
const BayerRepeatRect kRect{-3, -5, 4, 4};

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

std::vector<std::uint16_t> make_source(unsigned pattern) {
    std::vector<std::uint16_t> source(kStride * kHeight, 0xA55Au);
    for (std::size_t y = 0; y < kHeight; ++y) {
        for (std::size_t x = 0; x < kWidth; ++x) {
            std::uint32_t value = 0;
            if (pattern == 0) {
                value = 16384;
            } else if (pattern == 1) {
                value = 3000u + static_cast<std::uint32_t>(y * 1600u + x * 730u);
            } else {
                value = 12000u + static_cast<std::uint32_t>(y * 900u + x * 470u);
                if (x == kWidth / 2 && y == kHeight / 2) value = 52000u;
            }
            source[y * kStride + x] = static_cast<std::uint16_t>(value);
        }
    }
    return source;
}

PreparedImageControls controls_for(bool cubic, double amount,
                                   std::size_t tile_width,
                                   std::size_t tile_height) {
    PreparedImageControls controls{};
    controls.gains[0] = 1.15;
    controls.gains[1] = 1.0;
    controls.gains[2] = 0.92;
    controls.source_floor = 4096;
    controls.cubic = cubic;
    controls.amount = amount;
    controls.suppress_refine = false;
    controls.hot_threshold = 8000;
    controls.cold_threshold = 6000;
    controls.tile_width = tile_width;
    controls.tile_height = tile_height;
    return controls;
}

bool run_and_compare(const std::vector<std::uint16_t>& source,
                     const PreparedImageControls& controls,
                     const std::vector<std::uint16_t>* expected,
                     std::vector<std::uint16_t>* actual) {
    PreparedImageOutput output{};
    const auto status = imprint::acr::run_stage3_prepared_image_u16(
        source.data(), source.size(), kStride, kRect, controls, &output);
    if (status != PreparedImageStatus::ok) {
        std::cerr << "FAIL: render status=" << static_cast<int>(status)
                  << " cubic=" << controls.cubic
                  << " amount=" << controls.amount
                  << " tile=" << controls.tile_width << 'x'
                  << controls.tile_height << '\n';
        return false;
    }
    if (output.width != kWidth || output.height != kHeight ||
        output.rgb.size() != kWidth * kHeight * 3) {
        std::cerr << "FAIL: wrong output shape for tile=" << controls.tile_width
                  << 'x' << controls.tile_height << '\n';
        return false;
    }
    if (expected != nullptr && output.rgb != *expected) {
        std::size_t index = 0;
        while (index < output.rgb.size() && output.rgb[index] == (*expected)[index]) {
            ++index;
        }
        std::cerr << "FAIL: tile result mismatch at sample " << index
                  << " expected=" << (index < expected->size() ? (*expected)[index] : 0)
                  << " actual=" << (index < output.rgb.size() ? output.rgb[index] : 0)
                  << " tile=" << controls.tile_width << 'x'
                  << controls.tile_height << '\n';
        return false;
    }
    *actual = std::move(output.rgb);
    return true;
}

bool test_tiling_and_patterns() {
    constexpr std::array<std::array<std::size_t, 2>, 4> tiles{{
        {{1, 1}}, {{7, 5}}, {{32, 24}}, {{256, 256}},
    }};
    for (unsigned pattern = 0; pattern < 3; ++pattern) {
        const auto source = make_source(pattern);
        const auto original = source;
        for (bool cubic : {false, true}) {
            for (double amount : {1.0, 6.0}) {
                std::vector<std::uint16_t> reference;
                for (std::size_t index = 0; index < tiles.size(); ++index) {
                    auto controls = controls_for(cubic, amount,
                                                 tiles[index][0], tiles[index][1]);
                    std::vector<std::uint16_t> rendered;
                    if (!run_and_compare(source, controls,
                                         index == 0 ? nullptr : &reference,
                                         &rendered)) {
                        std::cerr << "fixture=" << pattern
                                  << " cubic=" << cubic
                                  << " amount=" << amount << '\n';
                        return false;
                    }
                    if (index == 0) reference = std::move(rendered);
                }
                if (!expect(source == original,
                            "input pixels and padded rows remain unchanged")) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_failure_atomicity_and_validation() {
    const auto source = make_source(1);
    const auto original_source = source;
    PreparedImageOutput output{};
    output.width = 55;
    output.height = 66;
    output.rgb = {7, 11, 13, 17};
    const auto original_output = output;
    auto controls = controls_for(false, 1.0, 8, 8);

    const auto invalid_rect = imprint::acr::run_stage3_prepared_image_u16(
        source.data(), source.size(), kStride, {-3, -5, -1, -5}, controls,
        &output);
    if (!expect(invalid_rect == PreparedImageStatus::invalid_rect,
                "empty rectangle rejected") ||
        !expect(output.width == original_output.width &&
                    output.height == original_output.height &&
                    output.rgb == original_output.rgb,
                "empty rectangle leaves output unchanged")) {
        return false;
    }

    const auto short_source = imprint::acr::run_stage3_prepared_image_u16(
        source.data(), kStride * (kHeight - 1), kStride, kRect, controls,
        &output);
    if (!expect(short_source == PreparedImageStatus::buffer_too_small,
                "short source capacity rejected") ||
        !expect(output.rgb == original_output.rgb,
                "short source leaves output unchanged")) {
        return false;
    }

    controls.tile_width = 0;
    const auto bad_tile = imprint::acr::run_stage3_prepared_image_u16(
        source.data(), source.size(), kStride, kRect, controls, &output);
    if (!expect(bad_tile == PreparedImageStatus::invalid_controls,
                "zero tile dimension rejected") ||
        !expect(output.rgb == original_output.rgb,
                "invalid controls leave output unchanged")) {
        return false;
    }
    controls = controls_for(false, 1.0, 8, 8);

    const auto overflowing_capacity =
        imprint::acr::run_stage3_prepared_image_u16(
            source.data(), std::numeric_limits<std::size_t>::max(), kStride,
            kRect, controls, &output);
    if (!expect(overflowing_capacity == PreparedImageStatus::arithmetic_overflow,
                "unrepresentable source capacity is rejected") ||
        !expect(output.rgb == original_output.rgb,
                "source capacity overflow leaves output unchanged")) {
        return false;
    }

    const BayerRepeatRect left_overflow{
        std::numeric_limits<std::int32_t>::min(), -5,
        std::numeric_limits<std::int32_t>::min() + 7, 4};
    const auto bad_halo = imprint::acr::run_stage3_prepared_image_u16(
        source.data(), source.size(), kStride, left_overflow, controls, &output);
    if (!expect(bad_halo == PreparedImageStatus::invalid_rect,
                "expanded halo endpoint outside int32 rejected") ||
        !expect(output.rgb == original_output.rgb,
                "unrepresentable halo leaves output unchanged")) {
        return false;
    }

    const auto original_huge = output.rgb;
    const BayerRepeatRect huge_rect{0, 0, 2, 100000000};
    const auto too_many = imprint::acr::run_stage3_prepared_image_u16(
        source.data(), 0, 100000000, huge_rect, controls, &output);
    if (!expect(too_many == PreparedImageStatus::buffer_too_small,
                "missing source span is rejected before rendering") ||
        !expect(output.rgb == original_huge,
                "large invalid input leaves output unchanged")) {
        return false;
    }
    if (!expect(source == original_source,
                "failed calls do not modify input")) {
        return false;
    }
    return true;
}

bool test_input_aliases_old_output() {
    PreparedImageOutput output{};
    output.width = 123;
    output.height = 456;
    output.rgb.resize(kStride * kHeight);
    for (std::size_t y = 0; y < kHeight; ++y) {
        for (std::size_t x = 0; x < kWidth; ++x) {
            output.rgb[y * kStride + x] = static_cast<std::uint16_t>(
                8000u + y * 1000u + x * 500u);
        }
        for (std::size_t x = kWidth; x < kStride; ++x) {
            output.rgb[y * kStride + x] = 0x5AA5u;
        }
    }
    auto controls = controls_for(false, 1.0, 7, 5);
    const auto status = imprint::acr::run_stage3_prepared_image_u16(
        output.rgb.data(), output.rgb.size(), kStride, kRect, controls, &output);
    return expect(status == PreparedImageStatus::ok,
                  "source may alias previous output rgb storage") &&
           expect(output.width == kWidth && output.height == kHeight &&
                      output.rgb.size() == kWidth * kHeight * 3,
                  "aliased source is replaced only after successful render");
}

}  // namespace

int main() {
    if (!test_tiling_and_patterns() ||
        !test_failure_atomicity_and_validation() ||
        !test_input_aliases_old_output()) {
        return 1;
    }
    std::cout << "prepared Stage3 image validation passed\n";
    return 0;
}
