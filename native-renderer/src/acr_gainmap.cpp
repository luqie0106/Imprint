#include "acr_gainmap.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumImageSamples = 128u * 1024u * 1024u;
constexpr std::size_t kMaximumMapSamples = 1024u * 1024u;

struct AddressSpan {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool checked_add(std::size_t a, std::size_t b, std::size_t* out) noexcept {
    if (out == nullptr || b > std::numeric_limits<std::size_t>::max() - a) {
        return false;
    }
    *out = a + b;
    return true;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t* out) noexcept {
    if (out == nullptr || (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)) {
        return false;
    }
    *out = a * b;
    return true;
}

bool naturally_aligned(const void* pointer, std::size_t alignment) noexcept {
    return pointer != nullptr && alignment != 0 &&
           reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

bool address_span(const void* pointer, std::size_t count, std::size_t item_size,
                  AddressSpan* out) noexcept {
    if (pointer == nullptr || out == nullptr) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    std::size_t bytes = 0;
    if (!checked_mul(count, item_size, &bytes) ||
        bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }
    out->begin = begin;
    out->end = begin + static_cast<std::uintptr_t>(bytes);
    return true;
}

bool overlaps(AddressSpan a, AddressSpan b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}

bool rectangle_dimensions(BayerRepeatRect rect, std::int64_t* height,
                          std::int64_t* width) noexcept {
    if (height == nullptr || width == nullptr) return false;
    *height = static_cast<std::int64_t>(rect.bottom) - rect.top;
    *width = static_cast<std::int64_t>(rect.right) - rect.left;
    return true;
}

bool checked_image_span(std::size_t width, std::size_t height,
                        std::size_t row_stride, std::size_t row_width,
                        std::size_t* span) noexcept {
    std::size_t rows = 0;
    return width != 0 && height != 0 && row_stride >= row_width &&
           checked_mul(height - 1, row_stride, &rows) &&
           checked_add(rows, row_width, span);
}

bool fits_i32(double value) noexcept {
    return std::isfinite(value) &&
           value >= static_cast<double>(std::numeric_limits<std::int32_t>::min()) &&
           value <= static_cast<double>(std::numeric_limits<std::int32_t>::max());
}

struct PreparedGeometry {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t row_width = 0;
    std::size_t output_samples = 0;
    std::size_t map_samples = 0;
    std::size_t row_stride = 0;
    std::uint32_t plane_end = 0;
    std::uint32_t row_pitch = 0;
    std::uint32_t col_pitch = 0;
    double inv_width = 0.0;
    double inv_height = 0.0;
    double spacing_v = 1.0;
    double spacing_h = 1.0;
    double origin_v = 0.0;
    double origin_h = 0.0;
    bool empty_area = false;
    std::int64_t selected_top = 0;
    std::int64_t selected_left = 0;
    std::int64_t selected_bottom = 0;
    std::int64_t selected_right = 0;
};

GainMapStatus validate_spec(const GainMapSpec& spec, std::size_t source_count,
                            std::size_t source_stride,
                            std::size_t map_count,
                            PreparedGeometry* geometry) noexcept {
    if (geometry == nullptr) return GainMapStatus::invalid_argument;
    if (spec.image_planes == 0 || spec.first_plane >
            std::numeric_limits<std::uint32_t>::max() - spec.plane_count ||
        spec.plane_count == 0 || spec.row_pitch == 0 || spec.col_pitch == 0 ||
        spec.map_v == 0 || spec.map_h == 0 || spec.map_planes == 0) {
        return GainMapStatus::invalid_metadata;
    }
    if (spec.tile_width == 0 || spec.tile_height == 0 ||
        spec.tile_width > 1024 || spec.tile_height > 1024) {
        return GainMapStatus::invalid_metadata;
    }
    if (source_count > kMaximumImageSamples || map_count > kMaximumMapSamples) {
        return GainMapStatus::resource_limit;
    }

    std::int64_t image_height = 0;
    std::int64_t image_width = 0;
    if (!rectangle_dimensions(spec.image_bounds, &image_height, &image_width) ||
        image_height <= 0 || image_width <= 0) {
        return GainMapStatus::invalid_rect;
    }
    if (image_height > std::numeric_limits<std::int32_t>::max() ||
        image_width > std::numeric_limits<std::int32_t>::max()) {
        return GainMapStatus::unsupported_domain;
    }
    geometry->width = static_cast<std::size_t>(image_width);
    geometry->height = static_cast<std::size_t>(image_height);
    if (!checked_mul(geometry->width, spec.image_planes, &geometry->row_width) ||
        !checked_mul(geometry->row_width, geometry->height,
                     &geometry->output_samples) ||
        geometry->output_samples > kMaximumImageSamples) {
        return GainMapStatus::resource_limit;
    }
    if (source_stride < geometry->row_width) return GainMapStatus::invalid_stride;
    geometry->row_stride = source_stride;
    std::size_t source_span_samples = 0;
    if (!checked_image_span(geometry->width, geometry->height, source_stride,
                            geometry->row_width, &source_span_samples)) {
        return GainMapStatus::arithmetic_overflow;
    }
    if (source_count < source_span_samples) return GainMapStatus::buffer_too_small;
    if (!checked_mul(spec.map_v, spec.map_h, &geometry->map_samples) ||
        !checked_mul(geometry->map_samples, spec.map_planes,
                     &geometry->map_samples) ||
        geometry->map_samples > kMaximumMapSamples) {
        return GainMapStatus::resource_limit;
    }
    if (map_count < geometry->map_samples) return GainMapStatus::buffer_too_small;
    geometry->plane_end = spec.first_plane + spec.plane_count;
    geometry->row_pitch = spec.row_pitch;
    geometry->col_pitch = spec.col_pitch;

    geometry->spacing_v = spec.spacing_v;
    geometry->spacing_h = spec.spacing_h;
    geometry->origin_v = spec.origin_v;
    geometry->origin_h = spec.origin_h;
    if (spec.map_v == 1) {
        geometry->spacing_v = 1.0;
        geometry->origin_v = 0.0;
    } else if (!std::isfinite(spec.spacing_v) || spec.spacing_v <= 0.0 ||
               !std::isfinite(spec.origin_v)) {
        return GainMapStatus::invalid_metadata;
    }
    if (spec.map_h == 1) {
        geometry->spacing_h = 1.0;
        geometry->origin_h = 0.0;
    } else if (!std::isfinite(spec.spacing_h) || spec.spacing_h <= 0.0 ||
               !std::isfinite(spec.origin_h)) {
        return GainMapStatus::invalid_metadata;
    }
    if (!std::isfinite(geometry->spacing_v) ||
        !std::isfinite(geometry->spacing_h) ||
        !std::isfinite(geometry->origin_v) ||
        !std::isfinite(geometry->origin_h)) {
        return GainMapStatus::invalid_metadata;
    }

    if (spec.list_stage >= 2 && spec.black_floor == 65535) {
        return GainMapStatus::unsupported_domain;
    }

    geometry->empty_area = !(spec.area.top < spec.area.bottom &&
                             spec.area.left < spec.area.right);
    if (geometry->empty_area) {
        geometry->selected_top = spec.image_bounds.top;
        geometry->selected_left = spec.image_bounds.left;
        geometry->selected_bottom = spec.image_bounds.bottom;
        geometry->selected_right = spec.image_bounds.right;
        return GainMapStatus::ok;
    }

    std::int64_t area_height = 0;
    std::int64_t area_width = 0;
    rectangle_dimensions(spec.area, &area_height, &area_width);
    if (area_height <= 0 || area_width <= 0) return GainMapStatus::invalid_rect;
    geometry->row_pitch = static_cast<std::uint32_t>(
        std::min<std::int64_t>(spec.row_pitch, area_height));
    geometry->col_pitch = static_cast<std::uint32_t>(
        std::min<std::int64_t>(spec.col_pitch, area_width));
    if (geometry->row_pitch == 0 || geometry->col_pitch == 0) {
        return GainMapStatus::invalid_metadata;
    }

    geometry->selected_top = std::max<std::int64_t>(spec.area.top,
                                                    spec.image_bounds.top);
    geometry->selected_left = std::max<std::int64_t>(spec.area.left,
                                                     spec.image_bounds.left);
    geometry->selected_bottom = std::min<std::int64_t>(spec.area.bottom,
                                                       spec.image_bounds.bottom);
    geometry->selected_right = std::min<std::int64_t>(spec.area.right,
                                                      spec.image_bounds.right);
    if (geometry->selected_top >= geometry->selected_bottom ||
        geometry->selected_left >= geometry->selected_right) {
        return GainMapStatus::ok;
    }

    const std::int64_t row_delta = geometry->selected_top - spec.area.top;
    const std::int64_t col_delta = geometry->selected_left - spec.area.left;
    const std::int64_t row_pitch = geometry->row_pitch;
    const std::int64_t col_pitch = geometry->col_pitch;
    const std::int64_t row_remainder = row_delta % row_pitch;
    const std::int64_t col_remainder = col_delta % col_pitch;
    if (row_remainder != 0) geometry->selected_top += row_pitch - row_remainder;
    if (col_remainder != 0) geometry->selected_left += col_pitch - col_remainder;
    if (geometry->selected_top >= geometry->selected_bottom ||
        geometry->selected_left >= geometry->selected_right) {
        return GainMapStatus::ok;
    }

    const std::int64_t last_row = geometry->selected_top +
        ((geometry->selected_bottom - 1 - geometry->selected_top) / row_pitch) * row_pitch;
    const std::int64_t last_col = geometry->selected_left +
        ((geometry->selected_right - 1 - geometry->selected_left) / col_pitch) * col_pitch;
    geometry->selected_bottom = last_row + 1;
    geometry->selected_right = last_col + 1;
    if (geometry->selected_bottom > std::numeric_limits<std::int32_t>::max() ||
        geometry->selected_right > std::numeric_limits<std::int32_t>::max()) {
        return GainMapStatus::unsupported_domain;
    }
    return GainMapStatus::ok;
}

struct VerticalPair {
    std::uint32_t upper = 0;
    std::uint32_t lower = 0;
    float fraction = 0.0f;
};

bool vertical_pair(const GainMapSpec& spec, const PreparedGeometry& geometry,
                   std::int64_t row, VerticalPair* pair) noexcept {
    if (pair == nullptr) return false;
    const double centered = (0.5 - static_cast<double>(spec.image_bounds.top)) +
                            static_cast<double>(row);
    const double q = std::fma(geometry.inv_height, centered,
                              -geometry.origin_v) / geometry.spacing_v;
    if (std::isnan(q)) return false;
    if (q <= 0.0) {
        pair->upper = pair->lower = 0;
        pair->fraction = 0.0f;
    } else {
        const std::uint32_t last = spec.map_v - 1;
        if (q >= static_cast<double>(last)) {
            pair->upper = pair->lower = last;
            pair->fraction = 0.0f;
        } else {
            const auto lower = static_cast<std::uint32_t>(q);
            // The recovered consumer names the smaller map row "upper" and
            // weights it by (1-f); the next row is "lower" and uses f.
            pair->upper = lower;
            pair->lower = lower + 1;
            pair->fraction = static_cast<float>(q - static_cast<double>(lower));
        }
    }
    return true;
}

float vertical_sample(const float* map, const GainMapSpec& spec,
                      const PreparedGeometry& geometry,
                      const VerticalPair& pair, std::uint32_t column,
                      std::uint32_t plane) noexcept {
    const std::size_t upper_index =
        (static_cast<std::size_t>(pair.upper) * spec.map_h + column) *
            spec.map_planes + plane;
    const std::size_t lower_index =
        (static_cast<std::size_t>(pair.lower) * spec.map_h + column) *
            spec.map_planes + plane;
    (void)geometry;
    const float upper_weight = 1.0f - pair.fraction;
    const float lower_term = pair.fraction * map[lower_index];
    return std::fma(map[upper_index], upper_weight, lower_term);
}

struct HorizontalSegment {
    float base = 0.0f;
    float slope = 0.0f;
    float offset = 0.0f;
    std::int32_t end = std::numeric_limits<std::int32_t>::max();
};

GainMapStatus set_segment(const float* map, const GainMapSpec& spec,
                          const PreparedGeometry& geometry,
                          const VerticalPair& vertical, std::uint32_t plane,
                          std::int64_t column, HorizontalSegment* segment) noexcept {
    if (segment == nullptr) return GainMapStatus::invalid_argument;
    const double centered = (0.5 - static_cast<double>(spec.image_bounds.left)) +
                            static_cast<double>(column);
    const double q = std::fma(geometry.inv_width, centered,
                              -geometry.origin_h) / geometry.spacing_h;
    if (std::isnan(q)) return GainMapStatus::arithmetic_overflow;

    const std::uint32_t last = spec.map_h - 1;
    std::uint32_t node = 0;
    if (q <= 0.0) {
        node = 0;
        const double transition = geometry.origin_h / geometry.inv_width -
                                  (0.5 - static_cast<double>(spec.image_bounds.left));
        const double rounded = std::ceil(transition);
        if (!fits_i32(rounded)) return GainMapStatus::unsupported_domain;
        segment->base = vertical_sample(map, spec, geometry, vertical, node, plane);
        segment->slope = 0.0f;
        segment->offset = 0.0f;
        segment->end = static_cast<std::int32_t>(rounded);
        return GainMapStatus::ok;
    }
    if (q >= static_cast<double>(last)) {
        node = last;
        segment->base = vertical_sample(map, spec, geometry, vertical, node, plane);
        segment->slope = 0.0f;
        segment->offset = 0.0f;
        segment->end = std::numeric_limits<std::int32_t>::max();
        return GainMapStatus::ok;
    }

    node = static_cast<std::uint32_t>(q);
    const double q_offset = q - static_cast<double>(node);
    const float a = vertical_sample(map, spec, geometry, vertical, node, plane);
    const float b = vertical_sample(map, spec, geometry, vertical, node + 1, plane);
    const double delta = static_cast<double>(b) - static_cast<double>(a);
    segment->base = static_cast<float>(std::fma(delta, q_offset,
                                                static_cast<double>(a)));
    segment->slope = static_cast<float>(
        (geometry.inv_width * delta) / geometry.spacing_h);
    segment->offset = 0.0f;
    const double transition =
        std::fma(static_cast<double>(node + 1), geometry.spacing_h,
                 geometry.origin_h) /
            geometry.inv_width -
        (0.5 - static_cast<double>(spec.image_bounds.left));
    const double rounded = std::ceil(transition);
    if (!fits_i32(rounded)) return GainMapStatus::unsupported_domain;
    segment->end = static_cast<std::int32_t>(rounded);
    return GainMapStatus::ok;
}

GainMapStatus advance_segment(const float* map, const GainMapSpec& spec,
                              const PreparedGeometry& geometry,
                              const VerticalPair& vertical, std::uint32_t plane,
                              std::int64_t next_column,
                              HorizontalSegment* segment) noexcept {
    if (next_column > std::numeric_limits<std::int32_t>::max() ||
        next_column < std::numeric_limits<std::int32_t>::min()) {
        return GainMapStatus::unsupported_domain;
    }
    if (next_column >= segment->end) {
        return set_segment(map, spec, geometry, vertical, plane, next_column,
                           segment);
    }
    segment->offset = segment->offset + 1.0f;
    return GainMapStatus::ok;
}

GainMapStatus apply_rows(const float* source, const float* map,
                         const GainMapSpec& spec,
                         const PreparedGeometry& geometry,
                         std::vector<float>* pixels) noexcept {
    const std::uint32_t first_plane = spec.first_plane;
    const std::uint32_t plane_end = std::min(spec.image_planes,
                                             geometry.plane_end);
    if (first_plane >= plane_end ||
        geometry.selected_top >= geometry.selected_bottom ||
        geometry.selected_left >= geometry.selected_right) {
        return GainMapStatus::ok;
    }

    const float z = static_cast<float>(spec.black_floor) / 65535.0f;
    const float scale = 1.0f - z;
    const float inverse_scale =
        (spec.list_stage >= 2 && spec.black_floor != 0) ? 1.0f / scale : 1.0f;
    const float bias =
        (spec.list_stage >= 2 && spec.black_floor != 0) ? 1.0f - inverse_scale : 0.0f;
    const bool black_wrap = spec.list_stage >= 2 && spec.black_floor != 0;

    const std::int64_t image_top = spec.image_bounds.top;
    const std::int64_t image_left = spec.image_bounds.left;
    const std::int64_t row_pitch = geometry.row_pitch;
    const std::int64_t col_pitch = geometry.col_pitch;
    const std::int64_t tile_height = static_cast<std::int64_t>(spec.tile_height);
    const std::int64_t tile_width = static_cast<std::int64_t>(spec.tile_width);

    for (std::int64_t tile_y = image_top; tile_y < spec.image_bounds.bottom;
         tile_y += tile_height) {
        const std::int64_t tile_bottom = std::min<std::int64_t>(
            spec.image_bounds.bottom, tile_y + tile_height);
        std::int64_t row_begin = std::max(tile_y, geometry.selected_top);
        const std::int64_t row_end = std::min(tile_bottom, geometry.selected_bottom);
        if (row_begin >= row_end) continue;
        if (!geometry.empty_area) {
            const std::int64_t delta = row_begin - geometry.selected_top;
            const std::int64_t remainder = delta % row_pitch;
            if (remainder != 0) row_begin += row_pitch - remainder;
        }
        if (row_begin >= row_end) continue;

        for (std::int64_t row = row_begin; row < row_end; row += row_pitch) {
            VerticalPair vertical{};
            if (!vertical_pair(spec, geometry, row, &vertical)) {
                return GainMapStatus::arithmetic_overflow;
            }
            for (std::int64_t tile_x = image_left;
                 tile_x < spec.image_bounds.right; tile_x += tile_width) {
                const std::int64_t tile_right = std::min<std::int64_t>(
                    spec.image_bounds.right, tile_x + tile_width);
                const std::int64_t col_begin = std::max(tile_x, geometry.selected_left);
                const std::int64_t col_end = std::min(tile_right, geometry.selected_right);
                if (col_begin >= col_end) continue;

                std::int64_t first_col = col_begin;
                if (!geometry.empty_area) {
                    const std::int64_t delta = first_col - geometry.selected_left;
                    const std::int64_t remainder = delta % col_pitch;
                    if (remainder != 0) first_col += col_pitch - remainder;
                }
                if (first_col >= col_end) continue;

                for (std::uint32_t plane = first_plane; plane < plane_end; ++plane) {
                    const std::uint32_t map_plane =
                        std::min(plane, spec.map_planes - 1);
                    HorizontalSegment segment{};
                    auto status = set_segment(map, spec, geometry, vertical,
                                              map_plane, first_col, &segment);
                    if (status != GainMapStatus::ok) return status;

                    for (std::int64_t column = first_col; column < col_end;
                         column += col_pitch) {
                        const std::size_t y = static_cast<std::size_t>(row - image_top);
                        const std::size_t x = static_cast<std::size_t>(column - image_left);
                        const std::size_t source_index =
                            y * geometry.row_stride + x * spec.image_planes + plane;
                        const std::size_t output_index =
                            (y * geometry.width + x) * spec.image_planes + plane;
                        float pixel = source[source_index];
                        if (black_wrap) pixel = std::fma(pixel, inverse_scale, bias);
                        const float gain = std::fma(segment.slope, segment.offset,
                                                     segment.base);
                        float result = std::fmin(gain * pixel, 1.0f);
                        if (black_wrap) result = std::fma(result, scale, z);
                        (*pixels)[output_index] = result;

                        if (column + col_pitch < col_end) {
                            const std::int64_t next_selected = column + col_pitch;
                            for (std::int64_t absolute = column + 1;
                                 absolute <= next_selected; ++absolute) {
                                status = advance_segment(map, spec, geometry,
                                                         vertical, map_plane,
                                                         absolute, &segment);
                                if (status != GainMapStatus::ok) return status;
                            }
                        }
                    }
                }
            }
        }
    }
    return GainMapStatus::ok;
}

}  // namespace

GainMapStatus apply_gainmap_prepared_f32(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const GainMapSpec& spec,
    const float* map, std::size_t map_sample_count,
    GainMapOutput* output) noexcept {
    if (source == nullptr || map == nullptr || output == nullptr) {
        return GainMapStatus::invalid_argument;
    }
    if (!naturally_aligned(source, alignof(float)) ||
        !naturally_aligned(map, alignof(float)) ||
        !naturally_aligned(output, alignof(GainMapOutput)) ||
        !naturally_aligned(&spec, alignof(GainMapSpec))) {
        return GainMapStatus::unsupported_alignment;
    }
    AddressSpan source_address{};
    AddressSpan map_address{};
    AddressSpan spec_address{};
    AddressSpan output_address{};
    if (!address_span(source, source_sample_count, sizeof(float), &source_address) ||
        !address_span(map, map_sample_count, sizeof(float), &map_address) ||
        !address_span(&spec, 1, sizeof(GainMapSpec), &spec_address) ||
        !address_span(output, 1, sizeof(GainMapOutput), &output_address)) {
        return GainMapStatus::arithmetic_overflow;
    }
    if (overlaps(output_address, source_address) ||
        overlaps(output_address, map_address) ||
        overlaps(output_address, spec_address)) {
        return GainMapStatus::overlap;
    }
    PreparedGeometry geometry{};
    GainMapStatus status = validate_spec(spec, source_sample_count,
                                         source_row_stride, map_sample_count,
                                         &geometry);
    if (status != GainMapStatus::ok) return status;

    for (std::size_t y = 0; y < geometry.height; ++y) {
        const float* row = source + y * source_row_stride;
        for (std::size_t x = 0; x < geometry.row_width; ++x) {
            if (!std::isfinite(row[x])) return GainMapStatus::non_finite_input;
        }
    }
    for (std::size_t i = 0; i < geometry.map_samples; ++i) {
        if (!std::isfinite(map[i])) return GainMapStatus::non_finite_input;
    }

    geometry.inv_width = 1.0 / static_cast<double>(geometry.width);
    geometry.inv_height = 1.0 / static_cast<double>(geometry.height);
    try {
        std::vector<float> candidate(geometry.output_samples);
        for (std::size_t y = 0; y < geometry.height; ++y) {
            std::copy_n(source + y * source_row_stride, geometry.row_width,
                        candidate.begin() + y * geometry.row_width);
        }
        status = apply_rows(source, map, spec, geometry, &candidate);
        if (status != GainMapStatus::ok) return status;

        output->width = geometry.width;
        output->height = geometry.height;
        output->planes = spec.image_planes;
        output->pixels = std::move(candidate);
        return GainMapStatus::ok;
    } catch (const std::bad_alloc&) {
        return GainMapStatus::allocation_failure;
    } catch (const std::length_error&) {
        return GainMapStatus::length_error;
    } catch (...) {
        return GainMapStatus::allocation_failure;
    }
}

GainMapStatus apply_gainmap_prepared_f32_to_buffer(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const std::int32_t rects8[8],
    const std::uint32_t controls9[9], const double geometry4[4],
    const float* map, std::size_t map_sample_count, std::uint16_t black_floor,
    std::size_t tile_width, std::size_t tile_height, float* destination,
    std::size_t destination_sample_count) noexcept {
    if (source == nullptr || rects8 == nullptr || controls9 == nullptr ||
        geometry4 == nullptr || map == nullptr || destination == nullptr) {
        return GainMapStatus::invalid_argument;
    }

    AddressSpan spans[6]{};
    if (!naturally_aligned(source, alignof(float)) ||
        !naturally_aligned(rects8, alignof(std::int32_t)) ||
        !naturally_aligned(controls9, alignof(std::uint32_t)) ||
        !naturally_aligned(geometry4, alignof(double)) ||
        !naturally_aligned(map, alignof(float)) ||
        !naturally_aligned(destination, alignof(float))) {
        return GainMapStatus::unsupported_alignment;
    }
    if (!address_span(source, source_sample_count, sizeof(float), &spans[0]) ||
        !address_span(rects8, 8, sizeof(std::int32_t), &spans[1]) ||
        !address_span(controls9, 9, sizeof(std::uint32_t), &spans[2]) ||
        !address_span(geometry4, 4, sizeof(double), &spans[3]) ||
        !address_span(map, map_sample_count, sizeof(float), &spans[4]) ||
        !address_span(destination, destination_sample_count, sizeof(float),
                      &spans[5])) {
        return GainMapStatus::arithmetic_overflow;
    }
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = i + 1; j < 6; ++j) {
            if (overlaps(spans[i], spans[j])) return GainMapStatus::overlap;
        }
    }

    GainMapSpec spec{};
    spec.image_bounds = {rects8[0], rects8[1], rects8[2], rects8[3]};
    spec.area = {rects8[4], rects8[5], rects8[6], rects8[7]};
    spec.image_planes = controls9[0];
    spec.first_plane = controls9[1];
    spec.plane_count = controls9[2];
    spec.row_pitch = controls9[3];
    spec.col_pitch = controls9[4];
    spec.map_v = controls9[5];
    spec.map_h = controls9[6];
    spec.map_planes = controls9[7];
    spec.list_stage = controls9[8];
    spec.spacing_v = geometry4[0];
    spec.spacing_h = geometry4[1];
    spec.origin_v = geometry4[2];
    spec.origin_h = geometry4[3];
    spec.black_floor = black_floor;
    spec.tile_width = tile_width;
    spec.tile_height = tile_height;

    PreparedGeometry geometry{};
    const auto validation = validate_spec(spec, source_sample_count,
                                          source_row_stride, map_sample_count,
                                          &geometry);
    if (validation != GainMapStatus::ok) return validation;
    if (destination_sample_count < geometry.output_samples) {
        return GainMapStatus::buffer_too_small;
    }

    GainMapOutput candidate{};
    const auto status = apply_gainmap_prepared_f32(
        source, source_sample_count, source_row_stride, spec, map,
        map_sample_count, &candidate);
    if (status != GainMapStatus::ok) return status;
    std::memcpy(destination, candidate.pixels.data(),
                candidate.pixels.size() * sizeof(float));
    return GainMapStatus::ok;
}

}  // namespace imprint::acr
