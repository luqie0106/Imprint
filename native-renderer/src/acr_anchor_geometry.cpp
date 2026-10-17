#include "acr_anchor_geometry.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

using Status = AnchorGeometryStatus;

bool same_rect(const AnchorRectI32 &a, const AnchorRectI32 &b) noexcept {
    return a.top == b.top && a.left == b.left && a.bottom == b.bottom &&
           a.right == b.right;
}

AnchorRectD to_double(const AnchorRectI32 &rect) noexcept {
    return {static_cast<double>(rect.top), static_cast<double>(rect.left),
            static_cast<double>(rect.bottom), static_cast<double>(rect.right)};
}

bool finite_rect(const AnchorRectD &rect) noexcept {
    return std::isfinite(rect.top) && std::isfinite(rect.left) &&
           std::isfinite(rect.bottom) && std::isfinite(rect.right);
}

Status relative_rect(const AnchorRectD &a, const AnchorRectD &b,
                     AnchorRectD &out) noexcept {
    const double height = b.bottom - b.top;
    const double width = b.right - b.left;
    if (!std::isfinite(height) || !std::isfinite(width) || height <= 0.0 ||
        width <= 0.0) {
        return Status::invalid_geometry;
    }

    // Match the recovered order: reciprocal first, then subtract and multiply.
    const double inv_height = 1.0 / height;
    const double inv_width = 1.0 / width;
    AnchorRectD candidate{
        (a.top - b.top) * inv_height,
        (a.left - b.left) * inv_width,
        (a.bottom - b.top) * inv_height,
        (a.right - b.left) * inv_width,
    };
    if (!finite_rect(candidate)) return Status::invalid_geometry;
    out = candidate;
    return Status::ok;
}

Status denormalize_rect(const AnchorRectD &fraction,
                        const AnchorRectD &base,
                        AnchorRectD &out) noexcept {
    const double height = base.bottom - base.top;
    const double width = base.right - base.left;
    if (!finite_rect(fraction) || !std::isfinite(height) ||
        !std::isfinite(width) || height <= 0.0 || width <= 0.0) {
        return Status::invalid_geometry;
    }

    // The recovered ARM64 sequence uses four fmadd instructions.
    AnchorRectD candidate{
        std::fma(fraction.top, height, base.top),
        std::fma(fraction.left, width, base.left),
        std::fma(fraction.bottom, height, base.top),
        std::fma(fraction.right, width, base.left),
    };
    if (!finite_rect(candidate)) return Status::invalid_geometry;
    out = candidate;
    return Status::ok;
}

void zero_origin_nonnegative_extent(AnchorRectD &rect) noexcept {
    const double height = rect.bottom - rect.top;
    const double width = rect.right - rect.left;
    rect.top = 0.0;
    rect.left = 0.0;
    rect.bottom = height > 0.0 ? height : 0.0;
    rect.right = width > 0.0 ? width : 0.0;
}

Status compute_actual_map(const PreparedAnchorGeometry &input,
                          AnchorRectD &out) noexcept {
    AnchorRectD current = to_double(input.actual_level0_bounds);
    for (const AnchorFilterBounds &filter : input.filters) {
        if (!filter.non_null || same_rect(filter.offset_58, filter.offset_78))
            continue;
        AnchorRectD relative{};
        Status status = relative_rect(to_double(filter.offset_58),
                                      to_double(filter.offset_78), relative);
        if (status != Status::ok) return status;
        AnchorRectD transformed{};
        status = denormalize_rect(relative, current, transformed);
        if (status != Status::ok) return status;
        current = transformed;
    }
    out = current;
    return Status::ok;
}

Status compute_reference_map(const PreparedAnchorGeometry &input,
                             int level, AnchorRectD &out) noexcept {
    AnchorRectD current = to_double(input.reference_bounds[level]);
    for (const AnchorFilterBounds &filter : input.filters) {
        if (!filter.non_null || same_rect(filter.offset_68, filter.offset_88))
            continue;
        AnchorRectD relative{};
        Status status = relative_rect(to_double(filter.offset_68),
                                      to_double(filter.offset_88), relative);
        if (status != Status::ok) return status;
        AnchorRectD transformed{};
        status = denormalize_rect(relative, current, transformed);
        if (status != Status::ok) return status;
        current = transformed;
        if (level != 0) zero_origin_nonnegative_extent(current);
    }
    if (!finite_rect(current)) return Status::invalid_geometry;
    out = current;
    return Status::ok;
}

Status round_coordinate(double value, std::int32_t &out) noexcept {
    if (!std::isfinite(value)) return Status::int32_overflow;
    const double bias = value > 0.0 ? 0.5 : -0.5;
    const double biased = value + bias;
    if (!std::isfinite(biased) || !(biased > -2147483649.0) ||
        !(biased < 2147483648.0)) {
        return Status::int32_overflow;
    }
    out = static_cast<std::int32_t>(biased);  // truncates toward zero
    return Status::ok;
}

Status round_rect_and_size(const AnchorRectD &rect, AnchorRectI32 &out,
                           std::int32_t &width,
                           std::int32_t &height) noexcept {
    AnchorRectI32 candidate{};
    Status status = round_coordinate(rect.top, candidate.top);
    if (status != Status::ok) return status;
    status = round_coordinate(rect.left, candidate.left);
    if (status != Status::ok) return status;
    status = round_coordinate(rect.bottom, candidate.bottom);
    if (status != Status::ok) return status;
    status = round_coordinate(rect.right, candidate.right);
    if (status != Status::ok) return status;

    const std::int64_t height64 =
        static_cast<std::int64_t>(candidate.bottom) - candidate.top;
    const std::int64_t width64 =
        static_cast<std::int64_t>(candidate.right) - candidate.left;
    if (height64 < 0 || width64 < 0 ||
        height64 > std::numeric_limits<std::int32_t>::max() ||
        width64 > std::numeric_limits<std::int32_t>::max()) {
        return Status::invalid_geometry;
    }
    out = candidate;
    width = static_cast<std::int32_t>(width64);
    height = static_cast<std::int32_t>(height64);
    return Status::ok;
}

}  // namespace

AnchorGeometryStatus compute_anchor_geometry(
    const PreparedAnchorGeometry &input,
    AnchorGeometryResult &result) noexcept {
    if (!input.level_available[0]) return Status::invalid_input;

    AnchorGeometryResult candidate{};
    if (input.geometry_enabled) {
        Status status = compute_actual_map(input, candidate.actual_map);
        if (status != Status::ok) return status;
        candidate.actual_map_valid = true;

        for (int level = 0; level <= 5; ++level) {
            if (!input.level_available[level]) continue;
            status = compute_reference_map(input, level,
                                           candidate.reference_map[level]);
            if (status != Status::ok) return status;
            candidate.reference_map_valid[level] = true;
        }
        if (!candidate.reference_map_valid[0]) return Status::invalid_input;

        for (int level = 0; level <= 5; ++level) {
            if (!input.level_available[level]) continue;
            AnchorRectD bounds{};
            if (level == 0) {
                bounds = candidate.actual_map;
            } else if (input.filters.empty()) {
                bounds = candidate.reference_map[level];
            } else {
                AnchorRectD scale{};
                status = relative_rect(candidate.actual_map,
                                       candidate.reference_map[0], scale);
                if (status != Status::ok) return status;
                status = denormalize_rect(scale,
                                          candidate.reference_map[level], bounds);
                if (status != Status::ok) return status;
            }
            candidate.output_bounds[level] = bounds;
            candidate.output_bounds_valid[level] = true;
        }
    }

    int requested_level = 0;
    for (int level = 1; level <= 5; ++level) {
        if (!input.level_available[level]) break;
        const AnchorRectD probe = input.geometry_enabled
            ? candidate.reference_map[level]
            : to_double(input.reference_bounds[level]);
        AnchorRectI32 rounded{};
        std::int32_t width = 0;
        std::int32_t height = 0;
        const Status status =
            round_rect_and_size(probe, rounded, width, height);
        if (status != Status::ok) return status;
        if (width < 128 && height < 128) break;
        requested_level = level;
    }

    const AnchorRectD selected = input.geometry_enabled
        ? candidate.output_bounds[requested_level]
        : to_double(input.reference_bounds[requested_level]);
    AnchorRectI32 rounded_source{};
    std::int32_t output_width = 0;
    std::int32_t output_height = 0;
    const Status status = round_rect_and_size(
        selected, rounded_source, output_width, output_height);
    if (status != Status::ok) return status;

    candidate.requested_level = requested_level;
    candidate.rounded_source_rect = rounded_source;
    candidate.output_width = output_width;
    candidate.output_height = output_height;
    candidate.destination_zero_origin_rect = {
        0, 0, output_height, output_width,
    };
    result = candidate;
    return Status::ok;
}

}  // namespace imprint::acr
