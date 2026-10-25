#include "../src/acr_bayer_border.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

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

bool metadata_address_span(const std::int32_t *rect,
                           AddressSpan *span) noexcept {
    if (rect == nullptr || span == nullptr ||
        (reinterpret_cast<std::uintptr_t>(rect) % alignof(std::int32_t)) != 0) {
        return false;
    }
    constexpr std::size_t kRectBytes = 4 * sizeof(std::int32_t);
    if (kRectBytes > static_cast<std::size_t>(
                         std::numeric_limits<std::ptrdiff_t>::max())) {
        return false;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(rect);
    const std::uintptr_t max_address =
        std::numeric_limits<std::uintptr_t>::max();
    if (kRectBytes > max_address - begin) return false;
    span->begin = begin;
    span->end = begin + static_cast<std::uintptr_t>(kRectBytes);
    return true;
}

bool checked_rect_dimensions(const std::int32_t rect[4],
                             std::int64_t *height,
                             std::int64_t *width) noexcept {
    if (rect == nullptr || height == nullptr || width == nullptr) return false;
    *height = static_cast<std::int64_t>(rect[2]) - rect[0];
    *width = static_cast<std::int64_t>(rect[3]) - rect[1];
    return true;
}

bool output_address_span(const std::uint16_t *output,
                         std::int64_t width,
                         std::int64_t height,
                         std::size_t row_stride,
                         AddressSpan *span) noexcept {
    if (output == nullptr || span == nullptr || width <= 0 || height <= 0 ||
        width > std::numeric_limits<std::int32_t>::max() ||
        height > std::numeric_limits<std::int32_t>::max() ||
        row_stride < static_cast<std::size_t>(width) ||
        (reinterpret_cast<std::uintptr_t>(output) % alignof(std::uint16_t)) != 0) {
        return false;
    }

    std::size_t last_row_start = 0;
    if (!checked_multiply(static_cast<std::size_t>(height - 1), row_stride,
                          &last_row_start)) {
        return false;
    }
    std::size_t samples = 0;
    std::size_t bytes = 0;
    if (!checked_add(last_row_start, static_cast<std::size_t>(width), &samples) ||
        !checked_multiply(samples, sizeof(std::uint16_t), &bytes) ||
        bytes > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max())) {
        return false;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(output);
    const std::uintptr_t max_address =
        std::numeric_limits<std::uintptr_t>::max();
    if (bytes > max_address - begin) return false;
    span->begin = begin;
    span->end = begin + static_cast<std::uintptr_t>(bytes);
    return true;
}

bool spans_overlap(AddressSpan left, AddressSpan right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

}  // namespace

extern "C" int acr_test_bayer_repeat2(
    const std::uint16_t *source,
    std::size_t source_sample_count,
    std::size_t source_row_stride,
    const std::int32_t *source_rect4,
    const std::int32_t *request_rect4,
    std::uint16_t *destination,
    std::size_t destination_sample_count,
    std::size_t destination_row_stride) noexcept {
    AddressSpan source_rect_span{};
    AddressSpan request_rect_span{};
    if (!metadata_address_span(source_rect4, &source_rect_span) ||
        !metadata_address_span(request_rect4, &request_rect_span)) {
        return static_cast<int>(imprint::acr::BayerBorderStatus::invalid_argument);
    }

    // Copy metadata before any output checks or writes so callers can safely
    // reuse their input storage after this function returns.
    std::int32_t source_values[4] = {0, 0, 0, 0};
    std::int32_t request_values[4] = {0, 0, 0, 0};
    std::memcpy(source_values, source_rect4, sizeof(source_values));
    std::memcpy(request_values, request_rect4, sizeof(request_values));

    std::int64_t request_height = 0;
    std::int64_t request_width = 0;
    checked_rect_dimensions(request_values, &request_height, &request_width);
    AddressSpan destination_span{};
    if (output_address_span(destination, request_width, request_height,
                            destination_row_stride, &destination_span) &&
        (spans_overlap(destination_span, source_rect_span) ||
         spans_overlap(destination_span, request_rect_span))) {
        return static_cast<int>(imprint::acr::BayerBorderStatus::overlap);
    }

    const imprint::acr::BayerBorderStatus status =
        imprint::acr::copy_bayer_repeat2_u16(
            source, source_sample_count, source_row_stride,
            {source_values[0], source_values[1], source_values[2],
             source_values[3]},
            {request_values[0], request_values[1], request_values[2],
             request_values[3]},
            destination, destination_sample_count, destination_row_stride);
    return static_cast<int>(status);
}
