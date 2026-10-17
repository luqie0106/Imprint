#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace imprint::acr {

// Prepared int32 crop/filter geometry only. This helper does not consume RAW
// pixels, infer source sampling, or model the later 00d0684c render-context
// level adjustment, matrices, or resampler. requested_level is the initial
// level selected by the 009e0364 threshold logic.
struct AnchorRectI32 {
    std::int32_t top = 0;
    std::int32_t left = 0;
    std::int32_t bottom = 0;
    std::int32_t right = 0;
};

struct AnchorRectD {
    double top = 0.0;
    double left = 0.0;
    double bottom = 0.0;
    double right = 0.0;
};

// Rects retain their recovered field offsets; no semantic names are inferred.
struct AnchorFilterBounds {
    bool non_null = false;
    AnchorRectI32 offset_58;
    AnchorRectI32 offset_78;
    AnchorRectI32 offset_68;
    AnchorRectI32 offset_88;
};

struct PreparedAnchorGeometry {
    bool geometry_enabled = true;
    AnchorRectI32 actual_level0_bounds;
    std::array<AnchorRectI32, 6> reference_bounds{};
    std::array<bool, 6> level_available{};
    std::vector<AnchorFilterBounds> filters;
};

enum class AnchorGeometryStatus : int {
    ok = 0,
    invalid_input = 1,
    invalid_geometry = 2,
    int32_overflow = 3,
};

struct AnchorGeometryResult {
    bool actual_map_valid = false;
    AnchorRectD actual_map{};
    std::array<bool, 6> reference_map_valid{};
    std::array<AnchorRectD, 6> reference_map{};
    std::array<bool, 6> output_bounds_valid{};
    std::array<AnchorRectD, 6> output_bounds{};

    int requested_level = 0;
    AnchorRectI32 rounded_source_rect{};
    std::int32_t output_width = 0;
    std::int32_t output_height = 0;
    AnchorRectI32 destination_zero_origin_rect{};
};

// Computes normalized anchor maps and the initial requested level from already
// prepared signed-int32 bounds. On failure, result is left unchanged.
AnchorGeometryStatus compute_anchor_geometry(
    const PreparedAnchorGeometry &input,
    AnchorGeometryResult &result) noexcept;

}  // namespace imprint::acr
