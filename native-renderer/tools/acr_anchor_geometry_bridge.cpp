#include "../src/acr_anchor_geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" {

struct AcrAnchorFilterInput {
    std::int32_t non_null;
    std::int32_t rects[16];  // offset +58, +78, +68, +88; each [t,l,b,r]
};

struct AcrAnchorGeometryOutput {
    std::int32_t requested_level;
    std::int32_t rounded_source_rect[4];
    std::int32_t output_width;
    std::int32_t output_height;
    std::int32_t destination_zero_origin_rect[4];
    std::int32_t actual_map_valid;
    std::int32_t reference_map_valid_mask;
    std::int32_t output_bounds_valid_mask;
    double actual_map[4];
    double reference_map[24];
    double output_bounds[24];
};

int acr_test_compute_anchor_geometry(
    std::int32_t geometry_enabled,
    const std::int32_t actual_level0_bounds[4],
    const std::int32_t reference_bounds[24],
    std::uint32_t available_level_mask,
    const AcrAnchorFilterInput *filters,
    std::size_t filter_count,
    AcrAnchorGeometryOutput *output) {
    using namespace imprint::acr;
    if ((geometry_enabled != 0 && geometry_enabled != 1) ||
        actual_level0_bounds == nullptr || reference_bounds == nullptr ||
        output == nullptr || (filter_count != 0 && filters == nullptr)) {
        return static_cast<int>(AnchorGeometryStatus::invalid_input);
    }

    PreparedAnchorGeometry input{};
    input.geometry_enabled = geometry_enabled != 0;
    input.actual_level0_bounds = {
        actual_level0_bounds[0], actual_level0_bounds[1],
        actual_level0_bounds[2], actual_level0_bounds[3],
    };
    for (int level = 0; level <= 5; ++level) {
        const std::size_t offset = static_cast<std::size_t>(level) * 4;
        input.reference_bounds[level] = {
            reference_bounds[offset], reference_bounds[offset + 1],
            reference_bounds[offset + 2], reference_bounds[offset + 3],
        };
        input.level_available[level] =
            (available_level_mask & (std::uint32_t{1} << level)) != 0;
    }
    input.filters.reserve(filter_count);
    for (std::size_t index = 0; index < filter_count; ++index) {
        const std::int32_t *rects = filters[index].rects;
        input.filters.push_back({
            filters[index].non_null != 0,
            {rects[0], rects[1], rects[2], rects[3]},
            {rects[4], rects[5], rects[6], rects[7]},
            {rects[8], rects[9], rects[10], rects[11]},
            {rects[12], rects[13], rects[14], rects[15]},
        });
    }

    AnchorGeometryResult result{};
    const AnchorGeometryStatus status = compute_anchor_geometry(input, result);
    if (status != AnchorGeometryStatus::ok) return static_cast<int>(status);

    AcrAnchorGeometryOutput candidate{};
    candidate.requested_level = result.requested_level;
    candidate.rounded_source_rect[0] = result.rounded_source_rect.top;
    candidate.rounded_source_rect[1] = result.rounded_source_rect.left;
    candidate.rounded_source_rect[2] = result.rounded_source_rect.bottom;
    candidate.rounded_source_rect[3] = result.rounded_source_rect.right;
    candidate.output_width = result.output_width;
    candidate.output_height = result.output_height;
    candidate.destination_zero_origin_rect[0] =
        result.destination_zero_origin_rect.top;
    candidate.destination_zero_origin_rect[1] =
        result.destination_zero_origin_rect.left;
    candidate.destination_zero_origin_rect[2] =
        result.destination_zero_origin_rect.bottom;
    candidate.destination_zero_origin_rect[3] =
        result.destination_zero_origin_rect.right;
    candidate.actual_map_valid = result.actual_map_valid ? 1 : 0;
    const auto copy_rect = [](const AnchorRectD &rect, double *destination) {
        destination[0] = rect.top;
        destination[1] = rect.left;
        destination[2] = rect.bottom;
        destination[3] = rect.right;
    };
    if (result.actual_map_valid) copy_rect(result.actual_map, candidate.actual_map);
    for (int level = 0; level <= 5; ++level) {
        if (result.reference_map_valid[level]) {
            candidate.reference_map_valid_mask |= 1 << level;
            copy_rect(result.reference_map[level], candidate.reference_map + 4 * level);
        }
        if (result.output_bounds_valid[level]) {
            candidate.output_bounds_valid_mask |= 1 << level;
            copy_rect(result.output_bounds[level], candidate.output_bounds + 4 * level);
        }
    }
    *output = candidate;
    return static_cast<int>(AnchorGeometryStatus::ok);
}

}  // extern "C"
