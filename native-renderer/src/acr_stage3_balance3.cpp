#include "acr_stage3_balance3.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

struct AddressSpan {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;  // exclusive
};

bool checked_bytes(std::size_t count, std::size_t *bytes) noexcept {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return false;
    }
    const std::size_t result = count * sizeof(float);
    if (result > static_cast<std::uintmax_t>(
                     std::numeric_limits<std::ptrdiff_t>::max())) {
        return false;
    }
    *bytes = result;
    return true;
}

bool checked_span(const void *pointer, std::size_t count,
                 AddressSpan *span) noexcept {
    std::size_t bytes = 0;
    if (pointer == nullptr || count == 0 || !checked_bytes(count, &bytes)) {
        return false;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }
    span->begin = begin;
    span->end = begin + bytes;
    return true;
}

bool checked_source_span(const float *center, std::size_t backward,
                        std::size_t forward, AddressSpan *span) noexcept {
    std::size_t backward_bytes = 0;
    std::size_t forward_count = 0;
    std::size_t forward_bytes = 0;
    std::size_t total_count = 0;
    std::size_t total_bytes = 0;
    if (!checked_bytes(backward, &backward_bytes) ||
        forward == std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    forward_count = forward + 1u;
    if (!checked_bytes(forward_count, &forward_bytes) ||
        backward > std::numeric_limits<std::size_t>::max() - forward_count) {
        return false;
    }
    total_count = backward + forward_count;
    if (!checked_bytes(total_count, &total_bytes)) return false;

    const auto address = reinterpret_cast<std::uintptr_t>(center);
    if (backward_bytes > address ||
        forward_bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
        return false;
    }
    span->begin = address - backward_bytes;
    span->end = address + forward_bytes;
    return span->end >= span->begin &&
           span->end - span->begin == total_bytes;
}

bool overlaps(const AddressSpan &a, const AddressSpan &b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}

bool aligned_16(const void *pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0xFu) == 0u;
}

bool valid_width(std::size_t width) noexcept {
    return width != 0u && (width & 3u) == 0u;
}

bool valid_stride(std::size_t stride) noexcept {
    return stride != 0u && (stride & 3u) == 0u;
}

bool guarded_stride_overflows(std::size_t width,
                              std::size_t radius) noexcept {
    return radius > (std::numeric_limits<std::size_t>::max() - width) / 2u;
}

std::size_t guarded_stride_minimum(std::size_t width,
                                  std::size_t radius) noexcept {
    return width + radius * 2u;
}

bool checked_guarded_stride(std::size_t width, std::size_t radius,
                            std::size_t stride) noexcept {
    return stride >= guarded_stride_minimum(width, radius);
}

Balance3KernelStatus validate_buffers(
    const float *center, std::size_t backward, std::size_t forward,
    std::size_t width, const float *coeff, std::size_t coeff_count,
    float *dst) noexcept {
    AddressSpan source_span;
    AddressSpan destination_span;
    AddressSpan coefficient_span;
    if (!checked_source_span(center, backward, forward, &source_span) ||
        !checked_span(dst, width, &destination_span) ||
        (coeff != nullptr &&
         !checked_span(coeff, coeff_count, &coefficient_span))) {
        return Balance3KernelStatus::arithmetic_overflow;
    }
    if (!aligned_16(center) || !aligned_16(dst)) {
        return Balance3KernelStatus::unsupported_alignment;
    }
    if (overlaps(source_span, destination_span) ||
        (coeff != nullptr && overlaps(coefficient_span, destination_span))) {
        return Balance3KernelStatus::overlapping_buffers;
    }
    return Balance3KernelStatus::ok;
}

// The explicit volatile store forces each recovered f32 intermediate to be
// rounded before it is consumed by the next operation, including targets
// whose evaluation format is wider than binary32.
float rounded_f32(float value) noexcept {
    volatile float rounded = value;
    return rounded;
}

Balance3KernelStatus validate_diagonal_arguments(
    const float *center, std::size_t stride, std::size_t width,
    const float *coeff, std::size_t radius, float *dst, bool plus_direction,
    std::size_t *diagonal_distance, std::size_t *backward,
    std::size_t *forward) noexcept {
    if (center == nullptr || coeff == nullptr || dst == nullptr) {
        return Balance3KernelStatus::invalid_argument;
    }
    if (!valid_width(width)) {
        return Balance3KernelStatus::invalid_dimensions;
    }
    if (!valid_stride(stride)) return Balance3KernelStatus::invalid_stride;
    if (radius > 32u) return Balance3KernelStatus::invalid_parameters;
    if (guarded_stride_overflows(width, radius)) {
        return Balance3KernelStatus::arithmetic_overflow;
    }
    if (!checked_guarded_stride(width, radius, stride)) {
        return Balance3KernelStatus::invalid_stride;
    }

    if (plus_direction && radius != 0u &&
        stride == std::numeric_limits<std::size_t>::max()) {
        return Balance3KernelStatus::arithmetic_overflow;
    }
    const std::size_t distance =
        radius == 0u ? stride : (plus_direction ? stride + 1u : stride - 1u);
    if (stride > static_cast<std::uintmax_t>(
                     std::numeric_limits<std::ptrdiff_t>::max()) ||
        width > static_cast<std::uintmax_t>(
                    std::numeric_limits<std::ptrdiff_t>::max()) ||
        distance > static_cast<std::uintmax_t>(
                       std::numeric_limits<std::ptrdiff_t>::max())) {
        return Balance3KernelStatus::arithmetic_overflow;
    }
    if (radius != 0u && distance >
                            std::numeric_limits<std::size_t>::max() / radius) {
        return Balance3KernelStatus::arithmetic_overflow;
    }
    *diagonal_distance = distance;
    *backward = radius * distance;
    if (width - 1u > std::numeric_limits<std::size_t>::max() - *backward) {
        return Balance3KernelStatus::arithmetic_overflow;
    }
    *forward = *backward + width - 1u;

    const std::size_t coeff_count = radius + 1u;
    AddressSpan coefficient_span;
    if (!checked_span(coeff, coeff_count, &coefficient_span)) {
        return Balance3KernelStatus::arithmetic_overflow;
    }
    if (reinterpret_cast<std::uintptr_t>(coeff) % alignof(float) != 0u) {
        return Balance3KernelStatus::unsupported_alignment;
    }
    (void)coefficient_span;
    for (std::size_t i = 0; i < coeff_count; ++i) {
        if (!std::isfinite(coeff[i])) {
            return Balance3KernelStatus::invalid_parameters;
        }
    }
    return Balance3KernelStatus::ok;
}

Balance3KernelStatus run_diagonal_filter(
    const float *center, std::size_t stride, std::size_t width,
    const float *coeff, std::size_t radius, float *dst,
    bool subtract, bool plus_direction) noexcept {
    std::size_t distance = 0;
    std::size_t backward = 0;
    std::size_t forward = 0;
    const Balance3KernelStatus args_status = validate_diagonal_arguments(
        center, stride, width, coeff, radius, dst, plus_direction, &distance,
        &backward, &forward);
    if (args_status != Balance3KernelStatus::ok) return args_status;

    const Balance3KernelStatus buffer_status = validate_buffers(
        center, backward, forward, width, coeff, radius + 1u, dst);
    if (buffer_status != Balance3KernelStatus::ok) return buffer_status;

    for (std::size_t j = 0; j < width; ++j) {
        float acc = rounded_f32(coeff[0] * center[j]);
        for (std::size_t i = 1; i <= radius; ++i) {
            const std::size_t offset = i * distance;
            const float pair = rounded_f32(
                (center - static_cast<std::ptrdiff_t>(offset))[j] +
                (center + static_cast<std::ptrdiff_t>(offset))[j]);
            acc = std::fma(coeff[i], pair, acc);
        }
        if (subtract) {
            dst[j] = rounded_f32(dst[j] - acc);
        } else {
            dst[j] = acc;
        }
    }
    return Balance3KernelStatus::ok;
}

}  // namespace

Balance3KernelStatus balance3_difference_prepared_row_00813870(
    const float *center, std::size_t stride, std::size_t width, int reverse,
    float *dst) noexcept {
    if (center == nullptr || dst == nullptr) {
        return Balance3KernelStatus::invalid_argument;
    }
    if (!valid_width(width)) {
        return Balance3KernelStatus::invalid_dimensions;
    }
    if (!valid_stride(stride)) return Balance3KernelStatus::invalid_stride;
    if (reverse != 0 && reverse != 1) {
        return Balance3KernelStatus::invalid_parameters;
    }
    if (width > std::numeric_limits<std::size_t>::max() - 2u) {
        return Balance3KernelStatus::arithmetic_overflow;
    }
    if (stride < width + 2u) return Balance3KernelStatus::invalid_stride;
    if (stride > std::numeric_limits<std::size_t>::max() - width) {
        return Balance3KernelStatus::arithmetic_overflow;
    }
    if (stride > static_cast<std::uintmax_t>(
                     std::numeric_limits<std::ptrdiff_t>::max()) ||
        width > static_cast<std::uintmax_t>(
                    std::numeric_limits<std::ptrdiff_t>::max())) {
        return Balance3KernelStatus::arithmetic_overflow;
    }

    const std::size_t backward = stride + 1u;
    const std::size_t forward = stride + width;
    const Balance3KernelStatus buffer_status = validate_buffers(
        center, backward, forward, width, nullptr, 0u, dst);
    if (buffer_status != Balance3KernelStatus::ok) return buffer_status;

    for (std::size_t j = 0; j < width; ++j) {
        const float north = rounded_f32(
            center[-static_cast<std::ptrdiff_t>(stride) +
                   static_cast<std::ptrdiff_t>(j) - 1] +
            center[-static_cast<std::ptrdiff_t>(stride) +
                   static_cast<std::ptrdiff_t>(j) + 1]);
        const float with_south_west = rounded_f32(
            north + center[static_cast<std::ptrdiff_t>(stride) +
                           static_cast<std::ptrdiff_t>(j) - 1]);
        const float sum = rounded_f32(
            with_south_west +
            center[static_cast<std::ptrdiff_t>(stride) +
                   static_cast<std::ptrdiff_t>(j) + 1]);
        const float avg = rounded_f32(sum * 0.25f);
        dst[j] = reverse == 0 ? rounded_f32(center[j] - avg)
                              : rounded_f32(avg - center[j]);
    }
    return Balance3KernelStatus::ok;
}

Balance3KernelStatus balance3_diagonal_filter_prepared_row_008139cc(
    const float *center, std::size_t stride, std::size_t width,
    const float *coeff, std::size_t radius, float *dst) noexcept {
    return run_diagonal_filter(center, stride, width, coeff, radius, dst,
                               false, true);
}

Balance3KernelStatus balance3_diagonal_subtract_prepared_row_00813cf8(
    const float *center, std::size_t stride, std::size_t width,
    const float *coeff, std::size_t radius, float *dst) noexcept {
    return run_diagonal_filter(center, stride, width, coeff, radius, dst,
                               true, false);
}

}  // namespace imprint::acr
