#include "acr_stage3_core3_format.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace imprint::acr {
namespace {

static_assert(sizeof(float) == sizeof(std::uint32_t) &&
                  std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "core3 format conversion requires IEEE-754 binary32 float");
static_assert(sizeof(std::uint16_t) == 2,
              "core3 format conversion requires 16-bit uint16_t");

struct AddressSpan {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool make_span(const void *pointer, std::size_t width, std::size_t item_size,
               AddressSpan &span) noexcept {
    const auto ptrdiff_limit = static_cast<std::uintmax_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    const auto size_limit = static_cast<std::uintmax_t>(
        std::numeric_limits<std::size_t>::max());
    const auto count = static_cast<std::uintmax_t>(width);
    const auto item_bytes = static_cast<std::uintmax_t>(item_size);
    if (count > ptrdiff_limit / item_bytes ||
        count > size_limit / item_bytes) {
        return false;
    }

    const auto bytes = static_cast<std::size_t>(width * item_size);
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    const auto address_limit = std::numeric_limits<std::uintptr_t>::max();
    if (bytes > address_limit - begin) return false;

    span = {begin, begin + static_cast<std::uintptr_t>(bytes)};
    return true;
}

bool overlaps(const AddressSpan &left, const AddressSpan &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

bool aligned_16(const void *pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0xFu) == 0;
}

Core3FormatStatus validate(const void *src, std::size_t src_sample_count,
                           std::size_t src_item_size, const void *dst,
                           std::size_t dst_sample_count,
                           std::size_t dst_item_size, std::size_t width,
                           AddressSpan &src_span,
                           AddressSpan &dst_span) noexcept {
    if (src == nullptr || dst == nullptr) {
        return Core3FormatStatus::invalid_argument;
    }
    if (width == 0 || (width & 3u) != 0) {
        return Core3FormatStatus::invalid_dimensions;
    }
    if (src_sample_count < width || dst_sample_count < width) {
        return Core3FormatStatus::insufficient_capacity;
    }
    if (!aligned_16(src) || !aligned_16(dst)) {
        return Core3FormatStatus::unsupported_alignment;
    }
    if (!make_span(src, width, src_item_size, src_span) ||
        !make_span(dst, width, dst_item_size, dst_span)) {
        return Core3FormatStatus::arithmetic_overflow;
    }
    if (overlaps(src_span, dst_span)) {
        return Core3FormatStatus::overlapping_buffers;
    }
    return Core3FormatStatus::ok;
}

float float_from_bits(std::uint32_t bits) noexcept {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float mul_f32(float left, float right) noexcept {
    volatile float rounded = left * right;
    return rounded;
}

std::uint16_t round_to_nearest_even_u16(float value) noexcept {
    // `floor` has fixed downward semantics independent of the active floating
    // environment. The binary32 input converts exactly to binary64, and the
    // supported product is bounded by 65535.
    const double exact = static_cast<double>(value);
    const double lower = std::floor(exact);
    const auto integer = static_cast<std::uint32_t>(lower);
    const double fraction = exact - lower;
    std::uint32_t rounded = integer;
    if (fraction > 0.5 ||
        (fraction == 0.5 && (integer & 1u) != 0u)) {
        ++rounded;
    }
    return static_cast<std::uint16_t>(rounded);
}

}  // namespace

Core3FormatStatus core3_u16_to_f32_prepared_row_008096e4(
    const std::uint16_t *src, std::size_t src_sample_count, float *dst,
    std::size_t dst_sample_count, std::size_t width) noexcept {
    AddressSpan src_span;
    AddressSpan dst_span;
    const Core3FormatStatus status =
        validate(src, src_sample_count, sizeof(*src), dst,
                 dst_sample_count, sizeof(*dst), width, src_span, dst_span);
    if (status != Core3FormatStatus::ok) return status;

    // The selected CPU pipe uses the exact binary32 reciprocal 0x37800080,
    // then multiplies the exactly representable uint16 value by that scale.
    constexpr std::uint32_t reciprocal_bits = 0x37800080u;
    const float reciprocal = float_from_bits(reciprocal_bits);
    for (std::size_t index = 0; index < width; ++index) {
        const float sample = static_cast<float>(src[index]);
        dst[index] = mul_f32(sample, reciprocal);
    }
    return Core3FormatStatus::ok;
}

Core3FormatStatus core3_f32_to_u16_prepared_row_008103dc(
    const float *src, std::size_t src_sample_count, std::uint16_t *dst,
    std::size_t dst_sample_count, std::size_t width) noexcept {
    AddressSpan src_span;
    AddressSpan dst_span;
    const Core3FormatStatus status =
        validate(src, src_sample_count, sizeof(*src), dst,
                 dst_sample_count, sizeof(*dst), width, src_span, dst_span);
    if (status != Core3FormatStatus::ok) return status;

    // Preflight the complete row before writing anything. This API deliberately
    // covers only finite normalized samples; the original narrowing kernel's
    // behavior outside that domain is not part of this prepared subdomain.
    for (std::size_t index = 0; index < width; ++index) {
        const float value = src[index];
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            return Core3FormatStatus::unsupported_domain;
        }
    }

    constexpr float scale = 65535.0f;
    for (std::size_t index = 0; index < width; ++index) {
        const float product = mul_f32(src[index], scale);
        dst[index] = round_to_nearest_even_u16(product);
    }
    return Core3FormatStatus::ok;
}

}  // namespace imprint::acr
