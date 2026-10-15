#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace acr_flare {

enum class Status : int {
    ok = 0,
    invalid_argument = 1,
    size_overflow = 2,
    too_many_pixels = 3,
    overlapping_buffers = 4,
    allocation_failure = 5,
};

struct Crop {
    std::size_t x;
    std::size_t y;
    std::size_t width;
    std::size_t height;
};

struct Result {
    double mean;
    double flare;
    double epsilon;
    std::size_t last_width;
    std::size_t last_height;
    // Number of generated lower pyramid levels: 0 or 5 (excludes stage3).
    std::uint32_t level_count;
};

struct StatisticsLevel {
    // Level 0 is the full actual input image. Lower levels use the original
    // DCA crop pyramid. Pixels are tightly packed RGB uint16 samples before
    // black normalization.
    std::size_t width = 0;
    std::size_t height = 0;
    std::uint32_t level = 0;
    std::vector<std::uint16_t> pixels;
};

// Bounded ordinary-image transcription of the uint16 packed-RGB flare-source
// path. The input and outputs are caller-owned. If last_level_rgb is supplied,
// it receives the selected full-bounds target after black normalization.
// If statistics_level is supplied, it receives the selected full-actual-image
// pyramid level before black normalization. Every output remains unchanged
// unless Status::ok is returned. Its pixels vector must own its storage and
// must not overlap the input or any other output buffer.
Status calculate(const std::uint16_t* packed_rgb,
                 std::size_t image_width,
                 std::size_t image_height,
                 Crop crop,
                 std::uint16_t black_offset,
                 bool nonlinear_enabled,
                 bool scene_referred,
                 Result* output,
                 std::uint16_t* last_level_rgb = nullptr,
                 std::size_t last_level_capacity_samples = 0,
                 StatisticsLevel* statistics_level = nullptr) noexcept;

}  // namespace acr_flare
