#include "acr_bayer_border.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

struct AddressSpan {
    std::uintptr_t begin;
    std::uintptr_t end;
};

bool checked_multiply(std::size_t left,
                      std::size_t right,
                      std::size_t *product) noexcept {
    if (product == nullptr) return false;
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }
    *product = left * right;
    return true;
}

bool checked_add(std::size_t left,
                 std::size_t right,
                 std::size_t *sum) noexcept {
    if (sum == nullptr) return false;
    if (left > std::numeric_limits<std::size_t>::max() - right) return false;
    *sum = left + right;
    return true;
}

bool rectangle_dimensions(BayerRepeatRect rect,
                          std::int64_t *height,
                          std::int64_t *width) noexcept {
    if (height == nullptr || width == nullptr) return false;
    *height = static_cast<std::int64_t>(rect.bottom) -
              static_cast<std::int64_t>(rect.top);
    *width = static_cast<std::int64_t>(rect.right) -
             static_cast<std::int64_t>(rect.left);
    return true;
}

bool checked_sample_span(std::int64_t width,
                         std::int64_t height,
                         std::size_t row_stride,
                         std::size_t *span) noexcept {
    if (span == nullptr || width <= 0 || height <= 0) return false;
    const std::size_t rows_before_last = static_cast<std::size_t>(height - 1);
    const std::size_t row_width = static_cast<std::size_t>(width);
    std::size_t last_row_start = 0;
    if (!checked_multiply(rows_before_last, row_stride, &last_row_start)) {
        return false;
    }
    return checked_add(last_row_start, row_width, span);
}

bool checked_address_span(const void *pointer,
                          std::size_t sample_span,
                          AddressSpan *address_span) noexcept {
    if (pointer == nullptr || address_span == nullptr) return false;
    std::size_t byte_count = 0;
    if (!checked_multiply(sample_span, sizeof(std::uint16_t), &byte_count) ||
        byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::ptrdiff_t>::max())) {
        return false;
    }

    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    const std::uintptr_t max_address =
        std::numeric_limits<std::uintptr_t>::max();
    if (byte_count > max_address - begin) return false;
    address_span->begin = begin;
    address_span->end = begin + static_cast<std::uintptr_t>(byte_count);
    return true;
}

bool spans_overlap(AddressSpan left, AddressSpan right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

std::int64_t floor_mod2(std::int64_t value) noexcept {
    const std::int64_t remainder = value % 2;
    return remainder < 0 ? remainder + 2 : remainder;
}

std::int64_t map_to_repeat2(std::int64_t coordinate,
                            std::int64_t low,
                            std::int64_t high) noexcept {
    if (coordinate < low) {
        return low + floor_mod2(coordinate - low);
    }
    if (coordinate >= high) {
        return high - 2 + floor_mod2(coordinate - high);
    }
    return coordinate;
}

}  // namespace

BayerBorderStatus copy_bayer_repeat2_u16(
    const std::uint16_t *source,
    std::size_t source_sample_count,
    std::size_t source_row_stride,
    BayerRepeatRect source_rect,
    BayerRepeatRect request_rect,
    std::uint16_t *destination,
    std::size_t destination_sample_count,
    std::size_t destination_row_stride) noexcept {
    if (source == nullptr || destination == nullptr) {
        return BayerBorderStatus::invalid_argument;
    }
    if ((reinterpret_cast<std::uintptr_t>(source) % alignof(std::uint16_t)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(destination) %
         alignof(std::uint16_t)) != 0) {
        return BayerBorderStatus::invalid_argument;
    }

    std::int64_t source_height = 0;
    std::int64_t source_width = 0;
    std::int64_t request_height = 0;
    std::int64_t request_width = 0;
    rectangle_dimensions(source_rect, &source_height, &source_width);
    rectangle_dimensions(request_rect, &request_height, &request_width);

    if (source_width <= 0 || source_height <= 0 || request_width <= 0 ||
        request_height <= 0) {
        return BayerBorderStatus::invalid_parameters;
    }
    if (source_width < 2 || source_height < 2) {
        return BayerBorderStatus::unsupported_domain;
    }
    constexpr std::int64_t kMaximumDimension =
        std::numeric_limits<std::int32_t>::max();
    if (source_width > kMaximumDimension || source_height > kMaximumDimension ||
        request_width > kMaximumDimension ||
        request_height > kMaximumDimension) {
        return BayerBorderStatus::unsupported_domain;
    }

    if (source_row_stride < static_cast<std::size_t>(source_width) ||
        destination_row_stride < static_cast<std::size_t>(request_width)) {
        return BayerBorderStatus::invalid_parameters;
    }

    // Validate the caller-declared capacities independently from the smaller
    // spans this operation will access. This rejects capacities whose byte
    // size or end address cannot be represented, without using their full
    // ranges for the source/destination overlap decision below.
    AddressSpan source_capacity_address_span{};
    AddressSpan destination_capacity_address_span{};
    if (!checked_address_span(source, source_sample_count,
                              &source_capacity_address_span) ||
        !checked_address_span(destination, destination_sample_count,
                              &destination_capacity_address_span)) {
        return BayerBorderStatus::invalid_parameters;
    }

    std::size_t source_span_samples = 0;
    std::size_t destination_span_samples = 0;
    if (!checked_sample_span(source_width, source_height, source_row_stride,
                             &source_span_samples) ||
        !checked_sample_span(request_width, request_height,
                             destination_row_stride,
                             &destination_span_samples)) {
        return BayerBorderStatus::invalid_parameters;
    }
    if (source_sample_count < source_span_samples ||
        destination_sample_count < destination_span_samples) {
        return BayerBorderStatus::invalid_parameters;
    }

    AddressSpan source_address_span{};
    AddressSpan destination_address_span{};
    if (!checked_address_span(source, source_span_samples,
                              &source_address_span) ||
        !checked_address_span(destination, destination_span_samples,
                              &destination_address_span)) {
        return BayerBorderStatus::invalid_parameters;
    }
    if (spans_overlap(source_address_span, destination_address_span)) {
        return BayerBorderStatus::overlap;
    }

    const std::size_t source_stride = source_row_stride;
    const std::size_t destination_stride = destination_row_stride;
    const std::size_t output_width = static_cast<std::size_t>(request_width);
    for (std::size_t y = 0;
         y < static_cast<std::size_t>(request_height);
         ++y) {
        const std::int64_t requested_y =
            static_cast<std::int64_t>(request_rect.top) +
            static_cast<std::int64_t>(y);
        const std::int64_t mapped_y = map_to_repeat2(
            requested_y, source_rect.top, source_rect.bottom);
        const std::size_t source_y = static_cast<std::size_t>(
            mapped_y - static_cast<std::int64_t>(source_rect.top));
        const std::size_t destination_row = y * destination_stride;
        const std::size_t source_row = source_y * source_stride;

        for (std::size_t x = 0; x < output_width; ++x) {
            const std::int64_t requested_x =
                static_cast<std::int64_t>(request_rect.left) +
                static_cast<std::int64_t>(x);
            const std::int64_t mapped_x = map_to_repeat2(
                requested_x, source_rect.left, source_rect.right);
            const std::size_t source_x = static_cast<std::size_t>(
                mapped_x - static_cast<std::int64_t>(source_rect.left));
            destination[destination_row + x] = source[source_row + source_x];
        }
    }

    return BayerBorderStatus::ok;
}

}  // namespace imprint::acr
