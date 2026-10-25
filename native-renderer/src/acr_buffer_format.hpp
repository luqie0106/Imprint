#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class BufferFormatStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    buffer_too_small = 4,
    unsupported_alignment = 5,
    overlap = 6,
    arithmetic_overflow = 7,
    resource_limit = 8,
    allocation_failure = 9,
    length_error = 10,
};

// Generic active-row conversion for the ACR CopyArea pair
// 012eaeb8 (float -> uint16) and 012eac64 (uint16 -> float). Counts and
// strides are in samples, not bytes. Only `row_samples` values per row are
// accessed; row padding is left untouched. The declared count spans are used
// for alignment, overflow, and overlap validation.
BufferFormatStatus copy_buffer_f32_to_u16_012eaeb8(
    const float *src, std::size_t src_count, std::size_t src_row_stride,
    std::size_t height, std::size_t row_samples, std::uint16_t *dst,
    std::size_t dst_count, std::size_t dst_row_stride) noexcept;

BufferFormatStatus copy_buffer_u16_to_f32_012eac64(
    const std::uint16_t *src, std::size_t src_count,
    std::size_t src_row_stride, std::size_t height, std::size_t row_samples,
    float *dst, std::size_t dst_count,
    std::size_t dst_row_stride) noexcept;

}  // namespace imprint::acr
