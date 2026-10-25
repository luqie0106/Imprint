#include "acr_buffer_format.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "ACR buffer conversion requires IEEE-754 binary32 float");
static_assert(sizeof(std::uint16_t) == 2,
              "ACR buffer conversion requires 16-bit uint16_t");

constexpr std::size_t kMaxActiveSamples = 128u * 1024u * 1024u;

struct AddressSpan {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool checked_mul(std::size_t left, std::size_t right,
                 std::size_t &result) noexcept {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }
    result = left * right;
    return true;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t &result) noexcept {
    if (left > std::numeric_limits<std::size_t>::max() - right) return false;
    result = left + right;
    return true;
}

bool make_declared_span(const void *pointer, std::size_t count,
                        std::size_t item_size,
                        AddressSpan &span) noexcept {
    const auto ptrdiff_limit = static_cast<std::uintmax_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    const auto count_u = static_cast<std::uintmax_t>(count);
    const auto size_u = static_cast<std::uintmax_t>(item_size);
    if (count_u > ptrdiff_limit / size_u ||
        count_u > static_cast<std::uintmax_t>(
                      std::numeric_limits<std::size_t>::max()) /
                      size_u) {
        return false;
    }

    const auto bytes = static_cast<std::size_t>(count * item_size);
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    const auto address_limit = std::numeric_limits<std::uintptr_t>::max();
    if (bytes > address_limit - begin) return false;

    span = {begin, begin + static_cast<std::uintptr_t>(bytes)};
    return true;
}

bool overlaps(const AddressSpan &left, const AddressSpan &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

template <typename Src, typename Dst>
BufferFormatStatus validate(const Src *src, std::size_t src_count,
                            std::size_t src_row_stride, std::size_t height,
                            std::size_t row_samples, Dst *dst,
                            std::size_t dst_count,
                            std::size_t dst_row_stride) noexcept {
    if (src == nullptr || dst == nullptr) {
        return BufferFormatStatus::invalid_argument;
    }
    if (height == 0 || row_samples == 0) {
        return BufferFormatStatus::invalid_dimensions;
    }

    std::size_t active_samples = 0;
    if (!checked_mul(height, row_samples, active_samples)) {
        return BufferFormatStatus::arithmetic_overflow;
    }
    if (active_samples > kMaxActiveSamples) {
        return BufferFormatStatus::resource_limit;
    }
    if (src_row_stride < row_samples || dst_row_stride < row_samples) {
        return BufferFormatStatus::invalid_stride;
    }

    std::size_t src_last_row_offset = 0;
    std::size_t dst_last_row_offset = 0;
    std::size_t src_required = 0;
    std::size_t dst_required = 0;
    if (!checked_mul(height - 1, src_row_stride, src_last_row_offset) ||
        !checked_mul(height - 1, dst_row_stride, dst_last_row_offset) ||
        !checked_add(src_last_row_offset, row_samples, src_required) ||
        !checked_add(dst_last_row_offset, row_samples, dst_required)) {
        return BufferFormatStatus::arithmetic_overflow;
    }
    if (src_count < src_required || dst_count < dst_required) {
        return BufferFormatStatus::buffer_too_small;
    }

    if ((reinterpret_cast<std::uintptr_t>(src) % alignof(Src)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(dst) % alignof(Dst)) != 0) {
        return BufferFormatStatus::unsupported_alignment;
    }

    AddressSpan src_span;
    AddressSpan dst_span;
    if (!make_declared_span(src, src_count, sizeof(Src), src_span) ||
        !make_declared_span(dst, dst_count, sizeof(Dst), dst_span)) {
        return BufferFormatStatus::arithmetic_overflow;
    }
    if (overlaps(src_span, dst_span)) return BufferFormatStatus::overlap;
    return BufferFormatStatus::ok;
}

std::uint16_t acr_float_to_u16(float pixel) noexcept {
    // Match 012eaeb8's FCMP/FCSEL sequence, including its unordered (NaN)
    // behavior, before the fused multiply-add and toward-zero integer convert.
    const float fallback = pixel > 0.5f ? 1.0f : 0.0f;
    const float bounded = (pixel <= 1.0f && pixel > 0.0f) ? pixel : fallback;
    const float scaled = std::fma(bounded, 65535.0f, 0.5f);
    const auto integer = static_cast<std::uint32_t>(scaled);
    return static_cast<std::uint16_t>(integer);
}

}  // namespace

BufferFormatStatus copy_buffer_f32_to_u16_012eaeb8(
    const float *src, std::size_t src_count, std::size_t src_row_stride,
    std::size_t height, std::size_t row_samples, std::uint16_t *dst,
    std::size_t dst_count, std::size_t dst_row_stride) noexcept {
    const auto status = validate(src, src_count, src_row_stride, height,
                                 row_samples, dst, dst_count, dst_row_stride);
    if (status != BufferFormatStatus::ok) return status;

    for (std::size_t row = 0; row < height; ++row) {
        const float *src_row = src + row * src_row_stride;
        std::uint16_t *dst_row = dst + row * dst_row_stride;
        for (std::size_t column = 0; column < row_samples; ++column) {
            dst_row[column] = acr_float_to_u16(src_row[column]);
        }
    }
    return BufferFormatStatus::ok;
}

BufferFormatStatus copy_buffer_u16_to_f32_012eac64(
    const std::uint16_t *src, std::size_t src_count,
    std::size_t src_row_stride, std::size_t height, std::size_t row_samples,
    float *dst, std::size_t dst_count,
    std::size_t dst_row_stride) noexcept {
    const auto status = validate(src, src_count, src_row_stride, height,
                                 row_samples, dst, dst_count, dst_row_stride);
    if (status != BufferFormatStatus::ok) return status;

    constexpr float reciprocal = 1.0f / 65535.0f;
    for (std::size_t row = 0; row < height; ++row) {
        const std::uint16_t *src_row = src + row * src_row_stride;
        float *dst_row = dst + row * dst_row_stride;
        for (std::size_t column = 0; column < row_samples; ++column) {
            const float integer = static_cast<float>(src_row[column]);
            dst_row[column] = reciprocal * integer;
        }
    }
    return BufferFormatStatus::ok;
}

}  // namespace imprint::acr
