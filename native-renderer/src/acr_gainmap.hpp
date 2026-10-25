#pragma once

#include "acr_bayer_border.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

// Explicit input for the recovered OpcodeList GainMap CPU consumer. The
// source covers image_bounds and is interleaved HWC float32; image_bounds is
// also the complete coordinate domain used for map normalization. This does
// not bind Adobe's host, scheduler, storage format, or original ABI.
struct GainMapSpec {
    BayerRepeatRect image_bounds{};
    BayerRepeatRect area{};
    std::uint32_t image_planes = 0;
    std::uint32_t first_plane = 0;
    std::uint32_t plane_count = 0;
    std::uint32_t row_pitch = 0;
    std::uint32_t col_pitch = 0;
    std::uint32_t map_v = 0;
    std::uint32_t map_h = 0;
    std::uint32_t map_planes = 0;
    std::uint32_t list_stage = 0;
    double spacing_v = 1.0;
    double spacing_h = 1.0;
    double origin_v = 0.0;
    double origin_h = 0.0;
    std::uint16_t black_floor = 0;
    std::size_t tile_width = 256;
    std::size_t tile_height = 256;
};

struct GainMapOutput {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t planes = 0;
    // Tightly packed HWC float32. Pixels outside the selected GainMap area
    // retain their source value.
    std::vector<float> pixels;
};

enum class GainMapStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_rect = 2,
    invalid_dimensions = 3,
    invalid_stride = 4,
    buffer_too_small = 5,
    invalid_metadata = 6,
    unsupported_alignment = 7,
    overlap = 8,
    arithmetic_overflow = 9,
    resource_limit = 10,
    non_finite_input = 11,
    unsupported_domain = 12,
    allocation_failure = 13,
    length_error = 14,
};

// Applies a scale-1/1 prepared GainMap to image_bounds. It uses explicit
// CPU tiles of tile_width x tile_height; horizontal segment state restarts at
// each tile. Different tile sizes can therefore differ by float32 rounding.
// Source rows are HWC with strides/capacities measured in float samples.
// Finite source and map samples are required; negative values are accepted.
// The output is committed only after the complete candidate succeeds. Source
// may alias the current output->pixels storage, which stays alive until commit.
GainMapStatus apply_gainmap_prepared_f32(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const GainMapSpec& spec,
    const float* map, std::size_t map_sample_count,
    GainMapOutput* output) noexcept;

// Shared buffer adapter for product and test ABI callers. rects8 is
// {image T,L,B,R, area T,L,B,R}; controls9
// is {imagePlanes, firstPlane, planeCount, rowPitch, colPitch, mapV, mapH,
// mapPlanes, listStage}; geometry4 is {spacingV, spacingH, originV, originH}.
// All declared spans, natural alignments, and metadata/input aliases are
// checked before processing. No destination bytes are written on failure.
GainMapStatus apply_gainmap_prepared_f32_to_buffer(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const std::int32_t rects8[8],
    const std::uint32_t controls9[9], const double geometry4[4],
    const float* map, std::size_t map_sample_count, std::uint16_t black_floor,
    std::size_t tile_width, std::size_t tile_height, float* destination,
    std::size_t destination_sample_count) noexcept;

}  // namespace imprint::acr
