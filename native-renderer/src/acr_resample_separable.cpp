#include "acr_resample_separable.hpp"

#include "acr_resample_vertical.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>

namespace imprint::acr {
namespace {

constexpr std::int64_t kFixedOne = std::int64_t{1} << 32;
constexpr std::int64_t kMaximumFraction = 0xfe000000;

bool valid_float_span(const float *data, std::size_t count) noexcept {
    if (!data || count == 0 ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    const auto bytes = count * sizeof(float);
    return address % alignof(float) == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool checked_multiply_size(std::size_t left, std::size_t right,
                           std::size_t &product) noexcept {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
        return false;
    product = left * right;
    return true;
}

bool checked_multiply_i64(std::int64_t left, std::int64_t right,
                          std::int64_t &product) noexcept {
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if (left == 0 || right == 0) {
        product = 0;
        return true;
    }
    if ((left == -1 && right == minimum) ||
        (right == -1 && left == minimum))
        return false;
    if (left > 0) {
        if ((right > 0 && left > maximum / right) ||
            (right < 0 && right < minimum / left))
            return false;
    } else {
        if ((right > 0 && left < minimum / right) ||
            (right < 0 && left < maximum / right))
            return false;
    }
    product = left * right;
    return true;
}

bool checked_add_i64(std::int64_t left, std::int64_t right,
                     std::int64_t &sum) noexcept {
    if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right))
        return false;
    sum = left + right;
    return true;
}

bool coordinate_at(std::int64_t origin, std::int64_t step,
                   std::int64_t absolute_index,
                   std::int64_t &coordinate) noexcept {
    std::int64_t offset = 0;
    return checked_multiply_i64(step, absolute_index, offset) &&
           checked_add_i64(origin, offset, coordinate);
}

std::int32_t integer_part(std::int64_t coordinate) noexcept {
    const auto high = static_cast<std::uint64_t>(coordinate) >> 32;
    const std::int64_t signed_high = high >= 0x80000000ULL
        ? static_cast<std::int64_t>(high) - 0x100000000LL
        : static_cast<std::int64_t>(high);
    return static_cast<std::int32_t>(signed_high);
}

unsigned phase_part(std::int64_t coordinate) noexcept {
    return static_cast<unsigned>((static_cast<std::uint64_t>(coordinate) >> 25) & 127);
}

bool checked_extent_end(std::int32_t start, std::size_t extent,
                        std::int64_t &end) noexcept {
    if (extent == 0 ||
        extent > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        return false;
    end = static_cast<std::int64_t>(start) + static_cast<std::int64_t>(extent);
    return end <= std::numeric_limits<std::int32_t>::max();
}

bool valid_rows_shape(const PreparedResampleRows &rows) noexcept {
    if (!rows.data || rows.tap_count == 0 ||
        rows.tap_count > kResampleRowMaximumTaps)
        return false;
    const std::size_t stride = ((rows.tap_count + 7) / 8) * 8;
    return rows.stride == stride &&
           rows.count == kResampleRowPhaseCount * stride &&
           valid_float_span(rows.data, rows.count);
}

bool rows_finite(const PreparedResampleRows &rows) noexcept {
    for (std::size_t i = 0; i < rows.count; ++i)
        if (!std::isfinite(rows.data[i])) return false;
    return true;
}

ResampleSeparableStatus bounds_for_source(
    std::int32_t source_start, std::int64_t source_end,
    std::size_t tap_count, const PreparedSeparableAxis &axis,
    std::int64_t &lower, std::int64_t &upper) noexcept {
    if (source_end - source_start < static_cast<std::int64_t>(tap_count))
        return ResampleSeparableStatus::source_window_out_of_range;

    std::int64_t source_lower = 0;
    std::int64_t source_upper_integer = 0;
    std::int64_t source_upper = 0;
    if (!checked_multiply_i64(source_start, kFixedOne, source_lower) ||
        !checked_multiply_i64(source_end - static_cast<std::int64_t>(tap_count),
                              kFixedOne, source_upper_integer) ||
        !checked_add_i64(source_upper_integer, kMaximumFraction, source_upper))
        return ResampleSeparableStatus::coordinate_overflow;

    lower = std::max(axis.lower, source_lower);
    upper = std::min(axis.upper, source_upper);
    if (lower > upper)
        return ResampleSeparableStatus::source_window_out_of_range;
    return ResampleSeparableStatus::ok;
}

ResampleSeparableStatus map_vertical_status(
    ResampleVerticalStatus status) noexcept {
    switch (status) {
        case ResampleVerticalStatus::ok:
            return ResampleSeparableStatus::ok;
        case ResampleVerticalStatus::source_window_out_of_range:
            return ResampleSeparableStatus::source_window_out_of_range;
        case ResampleVerticalStatus::non_finite_source:
            return ResampleSeparableStatus::non_finite_source;
        case ResampleVerticalStatus::non_finite_weight:
            return ResampleSeparableStatus::non_finite_coefficients;
        case ResampleVerticalStatus::allocation_failure:
            return ResampleSeparableStatus::allocation_failure;
        default:
            return ResampleSeparableStatus::vertical_stage_failure;
    }
}

ResampleSeparableStatus map_row_status(ResampleRowStatus status) noexcept {
    switch (status) {
        case ResampleRowStatus::ok:
            return ResampleSeparableStatus::ok;
        case ResampleRowStatus::source_window_out_of_range:
            return ResampleSeparableStatus::source_window_out_of_range;
        case ResampleRowStatus::coordinate_overflow:
            return ResampleSeparableStatus::coordinate_overflow;
        case ResampleRowStatus::allocation_failure:
            return ResampleSeparableStatus::allocation_failure;
        case ResampleRowStatus::invalid_rows:
            return ResampleSeparableStatus::invalid_horizontal_rows;
        default:
            return ResampleSeparableStatus::horizontal_stage_failure;
    }
}

}  // namespace

ResampleSeparableStatus process_resample_separable(
    const SourceView &source, const DestinationRect &destination,
    const PreparedSeparableAxis &vertical_axis,
    const PreparedSeparableAxis &horizontal_axis,
    std::vector<float> &result) noexcept {
    if (!source.data || source.count == 0 || source.height == 0 ||
        source.width == 0 || (source.channels != 1 && source.channels != 3) ||
        !valid_float_span(source.data, source.count))
        return ResampleSeparableStatus::invalid_input;

    if (destination.height == 0 || destination.width == 0)
        return ResampleSeparableStatus::invalid_request;
    if (!valid_rows_shape(vertical_axis.coefficients))
        return ResampleSeparableStatus::invalid_vertical_rows;
    if (!valid_rows_shape(horizontal_axis.coefficients))
        return ResampleSeparableStatus::invalid_horizontal_rows;

    std::int64_t source_bottom = 0;
    std::int64_t source_right = 0;
    std::int64_t destination_bottom = 0;
    std::int64_t destination_right = 0;
    if (!checked_extent_end(source.top, source.height, source_bottom) ||
        !checked_extent_end(source.left, source.width, source_right) ||
        !checked_extent_end(destination.top, destination.height,
                            destination_bottom) ||
        !checked_extent_end(destination.left, destination.width,
                            destination_right))
        return ResampleSeparableStatus::invalid_request;

    std::size_t source_row_stride = 0;
    std::size_t expected_source_count = 0;
    std::size_t destination_row_stride = 0;
    std::size_t destination_count = 0;
    std::size_t intermediate_count = 0;
    if (!checked_multiply_size(source.width, source.channels,
                               source_row_stride) ||
        !checked_multiply_size(source.height, source_row_stride,
                               expected_source_count) ||
        !checked_multiply_size(destination.width, source.channels,
                               destination_row_stride) ||
        !checked_multiply_size(destination.height, destination_row_stride,
                               destination_count) ||
        !checked_multiply_size(destination.height, source_row_stride,
                               intermediate_count) ||
        expected_source_count != source.count ||
        source.count > kResampleSeparableMaximumElements ||
        intermediate_count > kResampleSeparableMaximumElements ||
        destination_count > kResampleSeparableMaximumElements ||
        source_row_stride > kResampleSeparableMaximumElements ||
        destination_row_stride > kResampleSeparableMaximumElements)
        return ResampleSeparableStatus::invalid_request;

    if (!rows_finite(vertical_axis.coefficients) ||
        !rows_finite(horizontal_axis.coefficients))
        return ResampleSeparableStatus::non_finite_coefficients;
    for (std::size_t i = 0; i < source.count; ++i)
        if (!std::isfinite(source.data[i]))
            return ResampleSeparableStatus::non_finite_source;

    std::int64_t vertical_lower = 0;
    std::int64_t vertical_upper = 0;
    auto status = bounds_for_source(
        source.top, source_bottom, vertical_axis.coefficients.tap_count,
        vertical_axis, vertical_lower, vertical_upper);
    if (status != ResampleSeparableStatus::ok) return status;

    std::int64_t horizontal_lower = 0;
    std::int64_t horizontal_upper = 0;
    status = bounds_for_source(
        source.left, source_right, horizontal_axis.coefficients.tap_count,
        horizontal_axis, horizontal_lower, horizontal_upper);
    if (status != ResampleSeparableStatus::ok) return status;

    try {
        std::vector<float> intermediate(intermediate_count);
        std::vector<float> vertical_row;
        for (std::size_t y = 0; y < destination.height; ++y) {
            const auto absolute_y = static_cast<std::int64_t>(destination.top) +
                                    static_cast<std::int64_t>(y);
            std::int64_t coordinate = 0;
            if (!coordinate_at(vertical_axis.origin, vertical_axis.step,
                               absolute_y, coordinate))
                return ResampleSeparableStatus::coordinate_overflow;
            coordinate = std::max(vertical_lower,
                                  std::min(vertical_upper, coordinate));
            const std::int32_t row_start = integer_part(coordinate);
            const auto relative_row = static_cast<std::int64_t>(row_start) -
                                      static_cast<std::int64_t>(source.top);
            if (relative_row < 0 ||
                static_cast<std::uint64_t>(relative_row) > source.height ||
                vertical_axis.coefficients.tap_count >
                    source.height - static_cast<std::size_t>(relative_row))
                return ResampleSeparableStatus::source_window_out_of_range;

            std::size_t first_offset = 0;
            if (!checked_multiply_size(static_cast<std::size_t>(relative_row),
                                       source_row_stride, first_offset))
                return ResampleSeparableStatus::invalid_request;
            const unsigned phase = phase_part(coordinate);
            const float *weights = vertical_axis.coefficients.data +
                                   phase * vertical_axis.coefficients.stride;
            const auto vertical_status = process_resample_vertical(
                source.data, source.count, first_offset, source_row_stride,
                weights, vertical_axis.coefficients.tap_count,
                vertical_axis.coefficients.tap_count, source_row_stride,
                vertical_row);
            status = map_vertical_status(vertical_status);
            if (status != ResampleSeparableStatus::ok) return status;
            const auto row_offset = y * source_row_stride;
            std::copy(vertical_row.begin(), vertical_row.end(),
                      intermediate.begin() + static_cast<std::ptrdiff_t>(row_offset));
        }

        std::vector<float> gathered(source.width);
        std::vector<float> horizontal_row;
        std::vector<float> candidate(destination_count);
        const RowRequest horizontal_request_base{
            0,
            horizontal_axis.step,
            horizontal_lower,
            horizontal_upper,
            source.left,
            destination.width,
            ResampleRowMode::matrix_horizontal_baseline,
        };
        for (std::size_t y = 0; y < destination.height; ++y) {
            const auto intermediate_row = y * source_row_stride;
            const auto output_row = y * destination_row_stride;
            for (std::size_t channel = 0; channel < source.channels; ++channel) {
                for (std::size_t x = 0; x < source.width; ++x)
                    gathered[x] = intermediate[
                        intermediate_row + x * source.channels + channel];

                RowRequest request = horizontal_request_base;
                if (!coordinate_at(horizontal_axis.origin, horizontal_axis.step,
                                   destination.left,
                                   request.coordinate_origin))
                    return ResampleSeparableStatus::coordinate_overflow;
                const auto row_status = process_resample_row(
                    gathered.data(), gathered.size(),
                    horizontal_axis.coefficients, request, horizontal_row);
                status = map_row_status(row_status);
                if (status != ResampleSeparableStatus::ok) return status;

                for (std::size_t x = 0; x < destination.width; ++x)
                    candidate[output_row + x * source.channels + channel] =
                        horizontal_row[x];
            }
        }

        result.swap(candidate);
        return ResampleSeparableStatus::ok;
    } catch (const std::bad_alloc &) {
        return ResampleSeparableStatus::allocation_failure;
    } catch (...) {
        return ResampleSeparableStatus::allocation_failure;
    }
}

}  // namespace imprint::acr
