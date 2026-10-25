#include "acr_warp.hpp"

#include <algorithm>
#include <array>
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
constexpr std::size_t kParameterValuesPerPlane = 6u;
constexpr std::size_t kRadialCoefficients = 15u;
constexpr std::size_t kFractionSteps = 32u;
constexpr std::size_t kKernelSide = 4u;
constexpr std::size_t kKernelRadius = 2u;

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

bool checked_image_span(std::size_t height, std::size_t row_stride,
                        std::size_t row_width, std::size_t* span) noexcept {
    std::size_t rows = 0;
    return height != 0 && row_stride >= row_width &&
           checked_mul(height - 1, row_stride, &rows) &&
           checked_add(rows, row_width, span);
}

bool rectangle_dimensions(BayerRepeatRect rect, std::int64_t* height,
                          std::int64_t* width) noexcept {
    if (height == nullptr || width == nullptr) return false;
    *height = static_cast<std::int64_t>(rect.bottom) -
              static_cast<std::int64_t>(rect.top);
    *width = static_cast<std::int64_t>(rect.right) -
             static_cast<std::int64_t>(rect.left);
    return true;
}

struct PreparedGeometry {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t row_width = 0;
    std::size_t output_samples = 0;
    std::size_t required_source_samples = 0;
    double center_y = 0.0;
    double center_x = 0.0;
    double radius_center_y = 0.0;
    double inv_radius = 0.0;
    double aspect = 1.0;
    bool nop = false;
};

struct PlaneParams {
    std::array<double, 4> radial{};
    double tangent0 = 0.0;
    double tangent1 = 0.0;
};

using WeightTable = std::array<float,
    kFractionSteps * kFractionSteps * kKernelSide * kKernelSide>;

WarpStatus validate_spec(const float* source, std::size_t source_sample_count,
                         std::size_t source_row_stride, const WarpSpec& spec,
                         const double* coefficients, std::size_t coefficient_count,
                         PreparedGeometry* geometry,
                         std::array<PlaneParams, 4>* params) noexcept {
    if (geometry == nullptr || params == nullptr || source == nullptr ||
        coefficients == nullptr) {
        return WarpStatus::invalid_argument;
    }
    if (!naturally_aligned(source, alignof(float)) ||
        !naturally_aligned(coefficients, alignof(double))) {
        return WarpStatus::unsupported_alignment;
    }
    if (source_sample_count > kMaximumImageSamples) {
        return WarpStatus::resource_limit;
    }

    std::int64_t height = 0;
    std::int64_t width = 0;
    rectangle_dimensions(spec.image_bounds, &height, &width);
    if (height <= 0 || width <= 0) return WarpStatus::invalid_rect;
    if (height > std::numeric_limits<std::int32_t>::max() ||
        width > std::numeric_limits<std::int32_t>::max()) {
        return WarpStatus::invalid_dimensions;
    }
    if (spec.image_planes == 0 || spec.image_planes > 4 ||
        spec.param_planes == 0 || spec.param_planes > 4 ||
        (spec.param_planes != 1 && spec.param_planes != spec.image_planes) ||
        spec.tile_width == 0 || spec.tile_height == 0) {
        return WarpStatus::invalid_metadata;
    }
    if (!std::isfinite(spec.center_x) || !std::isfinite(spec.center_y) ||
        !std::isfinite(spec.aspect) || spec.center_x < 0.0 || spec.center_x > 1.0 ||
        spec.center_y < 0.0 || spec.center_y > 1.0 || spec.aspect <= 0.0) {
        return WarpStatus::invalid_metadata;
    }

    std::size_t expected_coefficients = 0;
    if (!checked_mul(spec.param_planes, kParameterValuesPerPlane,
                     &expected_coefficients)) {
        return WarpStatus::arithmetic_overflow;
    }
    if (coefficient_count < expected_coefficients) return WarpStatus::buffer_too_small;
    if (coefficient_count != expected_coefficients) return WarpStatus::invalid_metadata;

    bool all_radial_identity = true;
    for (std::size_t p = 0; p < spec.param_planes; ++p) {
        const double* c = coefficients + p * kParameterValuesPerPlane;
        for (std::size_t i = 0; i < kParameterValuesPerPlane; ++i) {
            if (!std::isfinite(c[i])) return WarpStatus::invalid_metadata;
        }
        if (c[4] != 0.0 || c[5] != 0.0) return WarpStatus::invalid_metadata;
        (*params)[p] = PlaneParams{{c[0], c[1], c[2], c[3]}, c[4], c[5]};
        all_radial_identity = all_radial_identity && c[0] == 1.0 &&
            c[1] == 0.0 && c[2] == 0.0 && c[3] == 0.0;
    }

    geometry->width = static_cast<std::size_t>(width);
    geometry->height = static_cast<std::size_t>(height);
    if (!checked_mul(geometry->width, spec.image_planes, &geometry->row_width) ||
        !checked_mul(geometry->row_width, geometry->height,
                     &geometry->output_samples) ||
        geometry->output_samples > kMaximumImageSamples) {
        return WarpStatus::resource_limit;
    }
    if (source_row_stride < geometry->row_width) return WarpStatus::invalid_stride;
    if (!checked_image_span(geometry->height, source_row_stride,
                            geometry->row_width,
                            &geometry->required_source_samples)) {
        return WarpStatus::arithmetic_overflow;
    }
    if (source_sample_count < geometry->required_source_samples) {
        return WarpStatus::buffer_too_small;
    }

    geometry->aspect = spec.aspect;
    geometry->nop = all_radial_identity;
    // The host checks the parameter object's NOP state before constructing the
    // warp filter. Prepared NOPs therefore copy after basic metadata/source
    // validation without evaluating aspect-scaled geometry.
    if (geometry->nop) return WarpStatus::ok;

    const double scaled_height_unrounded =
        spec.aspect * static_cast<double>(geometry->height);
    if (!std::isfinite(scaled_height_unrounded)) return WarpStatus::arithmetic_overflow;
    // The recovered path adds 0.5 in double, then converts toward zero to
    // int32. Keep that order: rounding the product first differs immediately
    // below a half-integer because the addition itself may round to the next
    // representable integer.
    const double scaled_height_rounded = scaled_height_unrounded + 0.5;
    const double int32_upper_exclusive =
        static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 1.0;
    if (!std::isfinite(scaled_height_rounded) ||
        scaled_height_rounded < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
        scaled_height_rounded >= int32_upper_exclusive) {
        return WarpStatus::arithmetic_overflow;
    }
    const auto scaled_height_i32 = static_cast<std::int32_t>(scaled_height_rounded);
    if (scaled_height_i32 <= 0) return WarpStatus::invalid_dimensions;

    const double top = static_cast<double>(spec.image_bounds.top);
    const double left = static_cast<double>(spec.image_bounds.left);
    const std::int64_t bottom_scaled_i64 =
        static_cast<std::int64_t>(spec.image_bounds.top) + scaled_height_i32;
    if (bottom_scaled_i64 < std::numeric_limits<std::int32_t>::min() ||
        bottom_scaled_i64 > std::numeric_limits<std::int32_t>::max()) {
        return WarpStatus::arithmetic_overflow;
    }
    const double bottom_scaled = static_cast<double>(bottom_scaled_i64);
    const double image_height = static_cast<double>(geometry->height);
    const double right = static_cast<double>(spec.image_bounds.right);
    geometry->center_y = std::fma(spec.center_y,
                                  image_height, top);
    geometry->radius_center_y = std::fma(spec.center_y,
                                         static_cast<double>(scaled_height_i32), top);
    geometry->center_x = std::fma(spec.center_x,
                                  static_cast<double>(geometry->width), left);
    if (!std::isfinite(bottom_scaled) || !std::isfinite(geometry->center_y) ||
        !std::isfinite(geometry->radius_center_y) ||
        !std::isfinite(geometry->center_x)) {
        return WarpStatus::arithmetic_overflow;
    }

    const double left_dx = left - geometry->center_x;
    const double top_dy = top - geometry->radius_center_y;
    const double bottom_dy = bottom_scaled - geometry->radius_center_y;
    const double right_dx = right - geometry->center_x;
    const double left_dx2 = left_dx * left_dx;
    const double top_left_d2 = std::fma(top_dy, top_dy, left_dx2);
    const double bottom_left_d2 = std::fma(bottom_dy, bottom_dy, left_dx2);
    const double right_dx2 = right_dx * right_dx;
    const double bottom_right_d2 = std::fma(bottom_dy, bottom_dy, right_dx2);
    const double top_right_d2 = std::fma(top_dy, top_dy, right_dx2);
    if (!std::isfinite(top_left_d2) || !std::isfinite(bottom_left_d2) ||
        !std::isfinite(bottom_right_d2) || !std::isfinite(top_right_d2)) {
        return WarpStatus::arithmetic_overflow;
    }
    const double radius_squared = std::max(
        std::max(top_left_d2, bottom_left_d2),
        std::max(bottom_right_d2, top_right_d2));
    const double radius = std::sqrt(radius_squared);
    if (!std::isfinite(radius) || radius <= 0.0) return WarpStatus::arithmetic_overflow;
    geometry->inv_radius = 1.0 / radius;
    if (!std::isfinite(geometry->inv_radius) || geometry->inv_radius <= 0.0) {
        return WarpStatus::arithmetic_overflow;
    }
    return WarpStatus::ok;
}

double radial_value(double r2, const PlaneParams& params) noexcept {
    double rho2 = r2;
    if (rho2 < 0.0) rho2 = 0.0;
    if (rho2 > 1.0) rho2 = 1.0;
    const double rho = std::sqrt(rho2);
    std::array<double, kRadialCoefficients> c{};
    c[0] = params.radial[0];
    c[2] = params.radial[1];
    c[4] = params.radial[2];
    c[6] = params.radial[3];
    double value = c[kRadialCoefficients - 1];
    for (std::size_t i = kRadialCoefficients - 1; i-- > 0;) {
        value = std::fma(value, rho, c[i]);
    }
    return value;
}

bool map_coordinate(std::int32_t row, std::int32_t column,
                    const PreparedGeometry& geometry,
                    const PlaneParams& params,
                    double* source_y, double* source_x) noexcept {
    if (source_y == nullptr || source_x == nullptr) return false;
    const double dy = static_cast<double>(row) - geometry.center_y;
    const double dx = static_cast<double>(column) - geometry.center_x;
    const double ny = dy * geometry.inv_radius;
    const double nx = dx * geometry.inv_radius;
    const double v = ny * geometry.aspect;
    const double u = nx;
    const double v2 = v * v;
    const double u2 = u * u;
    const double r2 = std::fmin(v2 + u2, 1.0);
    const double radial = radial_value(r2, params);
    const double mapped_dy = dy * radial;
    const double mapped_dx = dx * radial;
    *source_y = mapped_dy + geometry.center_y;
    *source_x = mapped_dx + geometry.center_x;
    if (!std::isfinite(dy) || !std::isfinite(dx) || !std::isfinite(ny) ||
        !std::isfinite(nx) || !std::isfinite(v) || !std::isfinite(u) ||
        !std::isfinite(r2) || !std::isfinite(radial) ||
        !std::isfinite(*source_y) || !std::isfinite(*source_x)) {
        return false;
    }

    return true;
}

void clamp_source_coordinates(const WarpSpec& spec, double* source_y,
                              double* source_x) noexcept {
    const double min_y = static_cast<double>(spec.image_bounds.top);
    const double max_y = static_cast<double>(spec.image_bounds.bottom - 1);
    const double min_x = static_cast<double>(spec.image_bounds.left);
    const double max_x = static_cast<double>(spec.image_bounds.right - 1);
    if (*source_y < min_y) *source_y = min_y;
    if (*source_y > max_y) *source_y = max_y;
    if (*source_x < min_x) *source_x = min_x;
    if (*source_x > max_x) *source_x = max_x;
}

double cubic(double x) noexcept {
    const double t = std::fabs(x);
    if (t < 1.0) {
        double z = std::fma(t, 1.25, -2.25);
        z = t * z;
        return std::fma(z, t, 1.0);
    }
    if (t < 2.0) {
        return std::fma(std::fma(std::fma(t, -0.75, 3.75), t, -6.0), t, 3.0);
    }
    return 0.0;
}

void build_weight_table(WeightTable* table) noexcept {
    for (std::size_t fy = 0; fy < kFractionSteps; ++fy) {
        for (std::size_t fx = 0; fx < kFractionSteps; ++fx) {
            std::array<float, kKernelSide * kKernelSide> raw{};
            double sum = 0.0;
            for (std::size_t i = 0; i < kKernelSide; ++i) {
                const double y = static_cast<double>(static_cast<int>(i) - 1) -
                                 static_cast<double>(fy) / 32.0;
                const float wy = static_cast<float>(cubic(y));
                for (std::size_t j = 0; j < kKernelSide; ++j) {
                    const double x = static_cast<double>(static_cast<int>(j) - 1) -
                                     static_cast<double>(fx) / 32.0;
                    const float wx = static_cast<float>(cubic(x));
                    const float product = wx * wy;
                    const std::size_t index = i * kKernelSide + j;
                    raw[index] = product;
                    sum += static_cast<double>(product);
                }
            }
            const float norm = static_cast<float>(1.0 / sum);
            for (std::size_t i = 0; i < raw.size(); ++i) {
                const std::size_t base =
                    ((fy * kFractionSteps + fx) * kKernelSide * kKernelSide) + i;
                (*table)[base] = raw[i] * norm;
            }
        }
    }
}

WarpStatus compute_source_area(const WarpSpec& spec,
                               const PreparedGeometry& geometry,
                               const std::array<PlaneParams, 4>& params,
                               std::int32_t tile_top, std::int32_t tile_left,
                               std::int32_t tile_bottom, std::int32_t tile_right,
                               BayerRepeatRect* source_area) noexcept {
    if (source_area == nullptr) return WarpStatus::invalid_argument;
    std::int64_t min_top_y = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_bottom_y = std::numeric_limits<std::int64_t>::min();
    std::int64_t min_left_x = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_right_x = std::numeric_limits<std::int64_t>::min();

    auto mapped = [&](std::int32_t row, std::int32_t column,
                      const PlaneParams& plane,
                      double* source_y, double* source_x) noexcept -> WarpStatus {
        if (!map_coordinate(row, column, geometry, plane,
                            source_y, source_x)) {
            return WarpStatus::arithmetic_overflow;
        }
        return WarpStatus::ok;
    };
    auto checked_floor_i32 = [](double value, std::int64_t* result) noexcept {
        const double rounded = std::floor(value);
        if (!std::isfinite(rounded) ||
            rounded < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
            rounded > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
            return false;
        }
        *result = static_cast<std::int64_t>(rounded);
        return true;
    };
    auto checked_ceil_i32 = [](double value, std::int64_t* result) noexcept {
        const double rounded = std::ceil(value);
        if (!std::isfinite(rounded) ||
            rounded < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
            rounded > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
            return false;
        }
        *result = static_cast<std::int64_t>(rounded);
        return true;
    };

    for (std::uint32_t p = 0; p < spec.image_planes; ++p) {
        const PlaneParams& plane = params[spec.param_planes == 1 ? 0 : p];
        // The recovered SrcArea builder takes only the vertical lower extent
        // from the top edge and the vertical upper extent from the bottom edge.
        double source_y = 0.0;
        double source_x = 0.0;
        for (std::int32_t x = tile_left; x < tile_right; ++x) {
            auto status = mapped(tile_top, x, plane, &source_y, &source_x);
            if (status != WarpStatus::ok) return status;
            std::int64_t value = 0;
            if (!checked_floor_i32(source_y, &value)) return WarpStatus::arithmetic_overflow;
            min_top_y = std::min(min_top_y, value);
        }

        const std::int32_t last_row = tile_bottom - 1;
        for (std::int32_t x = tile_left; x < tile_right; ++x) {
            auto status = mapped(last_row, x, plane, &source_y, &source_x);
            if (status != WarpStatus::ok) return status;
            std::int64_t value = 0;
            if (!checked_ceil_i32(source_y, &value)) return WarpStatus::arithmetic_overflow;
            max_bottom_y = std::max(max_bottom_y, value);
        }

        // The horizontal lower and upper extents likewise come from the left
        // and right edges respectively; each edge is evaluated per image plane.
        for (std::int32_t y = tile_top; y < tile_bottom; ++y) {
            auto status = mapped(y, tile_left, plane, &source_y, &source_x);
            if (status != WarpStatus::ok) return status;
            std::int64_t value = 0;
            if (!checked_floor_i32(source_x, &value)) return WarpStatus::arithmetic_overflow;
            min_left_x = std::min(min_left_x, value);
        }
        const std::int32_t last_column = tile_right - 1;
        for (std::int32_t y = tile_top; y < tile_bottom; ++y) {
            auto status = mapped(y, last_column, plane, &source_y, &source_x);
            if (status != WarpStatus::ok) return status;
            std::int64_t value = 0;
            if (!checked_ceil_i32(source_x, &value)) return WarpStatus::arithmetic_overflow;
            max_right_x = std::max(max_right_x, value);
        }
    }

    const std::int64_t expanded_top = min_top_y - static_cast<std::int64_t>(kKernelRadius);
    const std::int64_t expanded_left = min_left_x - static_cast<std::int64_t>(kKernelRadius);
    const std::int64_t expanded_bottom = max_bottom_y +
        static_cast<std::int64_t>(kKernelRadius) + 1;
    const std::int64_t expanded_right = max_right_x +
        static_cast<std::int64_t>(kKernelRadius) + 1;
    if (expanded_top < std::numeric_limits<std::int32_t>::min() ||
        expanded_left < std::numeric_limits<std::int32_t>::min() ||
        expanded_bottom > std::numeric_limits<std::int32_t>::max() ||
        expanded_right > std::numeric_limits<std::int32_t>::max()) {
        return WarpStatus::arithmetic_overflow;
    }
    const std::int64_t top = std::max<std::int64_t>(spec.image_bounds.top, expanded_top);
    const std::int64_t left = std::max<std::int64_t>(spec.image_bounds.left, expanded_left);
    const std::int64_t bottom = std::min<std::int64_t>(spec.image_bounds.bottom,
                                                       expanded_bottom);
    const std::int64_t right = std::min<std::int64_t>(spec.image_bounds.right,
                                                      expanded_right);
    if (top >= bottom || left >= right ||
        bottom - top < static_cast<std::int64_t>(kKernelSide + 1) ||
        right - left < static_cast<std::int64_t>(kKernelSide + 1)) {
        return WarpStatus::unsupported_domain;
    }
    source_area->top = static_cast<std::int32_t>(top);
    source_area->left = static_cast<std::int32_t>(left);
    source_area->bottom = static_cast<std::int32_t>(bottom);
    source_area->right = static_cast<std::int32_t>(right);
    return WarpStatus::ok;
}

void copy_active_rows(const float* source, std::size_t source_row_stride,
                      std::size_t row_width, std::size_t height,
                      std::vector<float>* destination) noexcept {
    for (std::size_t row = 0; row < height; ++row) {
        std::memcpy(destination->data() + row * row_width,
                    source + row * source_row_stride,
                    row_width * sizeof(float));
    }
}

WarpStatus render(const float* source, std::size_t source_row_stride,
                  const WarpSpec& spec, const std::array<PlaneParams, 4>& params,
                  const PreparedGeometry& geometry,
                  std::vector<float>* pixels) noexcept {
    if (pixels == nullptr) return WarpStatus::invalid_argument;
    if (geometry.nop) {
        copy_active_rows(source, source_row_stride, geometry.row_width,
                         geometry.height, pixels);
        return WarpStatus::ok;
    }

    WeightTable weights{};
    build_weight_table(&weights);
    const std::int64_t top = spec.image_bounds.top;
    const std::int64_t left = spec.image_bounds.left;
    const std::int64_t bottom = spec.image_bounds.bottom;
    const std::int64_t right = spec.image_bounds.right;
    const std::int64_t tile_height = static_cast<std::int64_t>(
        std::min(spec.tile_height, geometry.height));
    const std::int64_t tile_width = static_cast<std::int64_t>(
        std::min(spec.tile_width, geometry.width));
    for (std::int64_t tile_y = top; tile_y < bottom; tile_y += tile_height) {
        const std::int32_t tile_top = static_cast<std::int32_t>(tile_y);
        const std::int32_t tile_bottom = static_cast<std::int32_t>(
            std::min<std::int64_t>(bottom, tile_y + tile_height));
        for (std::int64_t tile_x = left; tile_x < right; tile_x += tile_width) {
            const std::int32_t tile_left = static_cast<std::int32_t>(tile_x);
            const std::int32_t tile_right = static_cast<std::int32_t>(
                std::min<std::int64_t>(right, tile_x + tile_width));
            BayerRepeatRect source_area{};
            const auto area_status = compute_source_area(
                spec, geometry, params, tile_top, tile_left, tile_bottom,
                tile_right, &source_area);
            if (area_status != WarpStatus::ok) return area_status;
            const std::int64_t max_anchor_y =
                static_cast<std::int64_t>(source_area.bottom) - 5;
            const std::int64_t max_anchor_x =
                static_cast<std::int64_t>(source_area.right) - 5;

            for (std::int32_t y = tile_top; y < tile_bottom; ++y) {
                const std::size_t output_row =
                    static_cast<std::size_t>(static_cast<std::int64_t>(y) - top) *
                    geometry.row_width;
                for (std::int32_t x = tile_left; x < tile_right; ++x) {
                    const std::size_t output_pixel = output_row +
                        static_cast<std::size_t>(static_cast<std::int64_t>(x) - left) *
                        spec.image_planes;
                    for (std::uint32_t p = 0; p < spec.image_planes; ++p) {
                        const PlaneParams& plane =
                            params[spec.param_planes == 1 ? 0 : p];
                        double source_y = 0.0;
                        double source_x = 0.0;
                        if (!map_coordinate(y, x, geometry, plane,
                                            &source_y, &source_x)) {
                            return WarpStatus::arithmetic_overflow;
                        }
                        clamp_source_coordinates(spec, &source_y, &source_x);
                        const double floor_y = std::floor(source_y);
                        const double floor_x = std::floor(source_x);
                        const double frac_y = (source_y - floor_y) * 32.0;
                        const double frac_x = (source_x - floor_x) * 32.0;
                        if (!std::isfinite(frac_y) || !std::isfinite(frac_x) ||
                            frac_y < 0.0 || frac_y >= 32.0 ||
                            frac_x < 0.0 || frac_x >= 32.0) {
                            return WarpStatus::arithmetic_overflow;
                        }
                        const std::int64_t iy = static_cast<std::int64_t>(floor_y);
                        const std::int64_t ix = static_cast<std::int64_t>(floor_x);
                        std::int64_t anchor_y = iy + 1 -
                            static_cast<std::int64_t>(kKernelRadius);
                        std::int64_t anchor_x = ix + 1 -
                            static_cast<std::int64_t>(kKernelRadius);
                        int fy = static_cast<int>(frac_y);
                        int fx = static_cast<int>(frac_x);
                        if (anchor_y < source_area.top || anchor_y > max_anchor_y) fy = 0;
                        if (anchor_x < source_area.left || anchor_x > max_anchor_x) fx = 0;
                        anchor_y = std::max<std::int64_t>(source_area.top,
                            std::min(max_anchor_y, anchor_y));
                        anchor_x = std::max<std::int64_t>(source_area.left,
                            std::min(max_anchor_x, anchor_x));

                        const std::size_t weight_base =
                            (static_cast<std::size_t>(fy) * kFractionSteps +
                             static_cast<std::size_t>(fx)) *
                            kKernelSide * kKernelSide;
                        float sum = 0.0f;
                        for (std::size_t ky = 0; ky < kKernelSide; ++ky) {
                            const std::size_t source_y_offset =
                                static_cast<std::size_t>(anchor_y +
                                    static_cast<std::int64_t>(ky) - top) *
                                source_row_stride;
                            for (std::size_t kx = 0; kx < kKernelSide; ++kx) {
                                const std::size_t source_x_offset =
                                    static_cast<std::size_t>(anchor_x +
                                        static_cast<std::int64_t>(kx) - left) *
                                    spec.image_planes;
                                const std::size_t source_index = source_y_offset +
                                    source_x_offset + p;
                                const float sample = source[source_index];
                                const float weight = weights[weight_base +
                                    ky * kKernelSide + kx];
                                sum = std::fma(weight, sample, sum);
                            }
                        }
                        float value = std::fmin(sum, 1.0f);
                        if (value < 0.0f) value = 0.0f;
                        (*pixels)[output_pixel + p] = value;
                    }
                }
            }
        }
    }
    return WarpStatus::ok;
}

bool finite_active_source(const float* source, std::size_t source_row_stride,
                          std::size_t row_width, std::size_t height) noexcept {
    for (std::size_t y = 0; y < height; ++y) {
        const float* row = source + y * source_row_stride;
        for (std::size_t x = 0; x < row_width; ++x) {
            if (!std::isfinite(row[x])) return false;
        }
    }
    return true;
}

}  // namespace

WarpStatus apply_warp_prepared_f32(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const WarpSpec& spec,
    const double* coefficients, std::size_t coefficient_count,
    WarpOutput* output) noexcept {
    if (source == nullptr || coefficients == nullptr || output == nullptr) {
        return WarpStatus::invalid_argument;
    }
    if (!naturally_aligned(output, alignof(WarpOutput))) {
        return WarpStatus::unsupported_alignment;
    }
    AddressSpan source_span{};
    AddressSpan coefficient_span{};
    AddressSpan spec_span{};
    AddressSpan output_span{};
    if (!address_span(source, source_sample_count, sizeof(float), &source_span) ||
        !address_span(coefficients, coefficient_count, sizeof(double), &coefficient_span) ||
        !address_span(&spec, 1, sizeof(WarpSpec), &spec_span) ||
        !address_span(output, 1, sizeof(WarpOutput), &output_span)) {
        return WarpStatus::arithmetic_overflow;
    }
    if (overlaps(source_span, coefficient_span) || overlaps(source_span, spec_span) ||
        overlaps(source_span, output_span) || overlaps(coefficient_span, spec_span) ||
        overlaps(coefficient_span, output_span) || overlaps(spec_span, output_span)) {
        return WarpStatus::overlap;
    }

    try {
        PreparedGeometry geometry{};
        std::array<PlaneParams, 4> params{};
        const auto validation = validate_spec(source, source_sample_count,
            source_row_stride, spec, coefficients, coefficient_count,
            &geometry, &params);
        if (validation != WarpStatus::ok) return validation;
        if (!finite_active_source(source, source_row_stride, geometry.row_width,
                                  geometry.height)) {
            return WarpStatus::non_finite_input;
        }

        std::vector<float> candidate;
        candidate.resize(geometry.output_samples);
        const auto status = render(source, source_row_stride, spec, params,
                                   geometry, &candidate);
        if (status != WarpStatus::ok) return status;

        output->width = geometry.width;
        output->height = geometry.height;
        output->planes = spec.image_planes;
        output->pixels = std::move(candidate);
        return WarpStatus::ok;
    } catch (const std::bad_alloc&) {
        return WarpStatus::allocation_failure;
    } catch (const std::length_error&) {
        return WarpStatus::length_error;
    } catch (...) {
        return WarpStatus::allocation_failure;
    }
}

WarpStatus apply_warp_prepared_f32_to_buffer(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const std::int32_t rect4[4],
    const std::uint32_t controls2[2], const double geometry3[3],
    const double* coefficients, std::size_t coefficient_count,
    std::size_t tile_width, std::size_t tile_height, float* destination,
    std::size_t destination_sample_count) noexcept {
    if (source == nullptr || rect4 == nullptr || controls2 == nullptr ||
        geometry3 == nullptr || coefficients == nullptr || destination == nullptr) {
        return WarpStatus::invalid_argument;
    }
    if (!naturally_aligned(source, alignof(float)) ||
        !naturally_aligned(rect4, alignof(std::int32_t)) ||
        !naturally_aligned(controls2, alignof(std::uint32_t)) ||
        !naturally_aligned(geometry3, alignof(double)) ||
        !naturally_aligned(coefficients, alignof(double)) ||
        !naturally_aligned(destination, alignof(float))) {
        return WarpStatus::unsupported_alignment;
    }

    std::array<AddressSpan, 6> spans{};
    if (!address_span(source, source_sample_count, sizeof(float), &spans[0]) ||
        !address_span(rect4, 4, sizeof(std::int32_t), &spans[1]) ||
        !address_span(controls2, 2, sizeof(std::uint32_t), &spans[2]) ||
        !address_span(geometry3, 3, sizeof(double), &spans[3]) ||
        !address_span(coefficients, coefficient_count, sizeof(double), &spans[4]) ||
        !address_span(destination, destination_sample_count, sizeof(float), &spans[5])) {
        return WarpStatus::arithmetic_overflow;
    }
    for (std::size_t i = 0; i < spans.size(); ++i) {
        for (std::size_t j = i + 1; j < spans.size(); ++j) {
            if (overlaps(spans[i], spans[j])) return WarpStatus::overlap;
        }
    }

    const WarpSpec spec{
        BayerRepeatRect{rect4[0], rect4[1], rect4[2], rect4[3]},
        controls2[0], controls2[1], geometry3[0], geometry3[1], geometry3[2],
        tile_width, tile_height};
    WarpOutput candidate{};
    const auto status = apply_warp_prepared_f32(source, source_sample_count,
        source_row_stride, spec, coefficients, coefficient_count, &candidate);
    if (status != WarpStatus::ok) return status;
    if (destination_sample_count < candidate.pixels.size()) {
        return WarpStatus::buffer_too_small;
    }
    std::memcpy(destination, candidate.pixels.data(),
                candidate.pixels.size() * sizeof(float));
    return WarpStatus::ok;
}

}  // namespace imprint::acr
