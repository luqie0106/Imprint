#include "acr_stage3_balance3_finish.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

static_assert(sizeof(float) == sizeof(std::uint32_t) &&
                  std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "balance3 finish requires IEEE-754 binary32 float");

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
    span->end = begin + static_cast<std::uintptr_t>(bytes);
    return true;
}

bool checked_source_span(const float *source, std::size_t stride,
                         std::size_t width, AddressSpan *span) noexcept {
    // The read neighborhood is source[-stride-1] through
    // source[stride+width], inclusive. This conservative contiguous span
    // also covers the center samples between the guarded rows.
    if (stride == std::numeric_limits<std::size_t>::max() ||
        width == std::numeric_limits<std::size_t>::max() ||
        stride > std::numeric_limits<std::size_t>::max() - width) {
        return false;
    }
    const std::size_t backward = stride + 1u;
    const std::size_t forward = stride + width;
    if (forward == std::numeric_limits<std::size_t>::max()) return false;
    const std::size_t forward_count = forward + 1u;
    if (backward > std::numeric_limits<std::size_t>::max() - forward_count) {
        return false;
    }

    std::size_t backward_bytes = 0;
    std::size_t forward_bytes = 0;
    std::size_t total_bytes = 0;
    const std::size_t total_count = backward + forward_count;
    if (!checked_bytes(backward, &backward_bytes) ||
        !checked_bytes(forward_count, &forward_bytes) ||
        !checked_bytes(total_count, &total_bytes)) {
        return false;
    }

    const auto address = reinterpret_cast<std::uintptr_t>(source);
    if (backward_bytes > address ||
        forward_bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
        return false;
    }
    span->begin = address - static_cast<std::uintptr_t>(backward_bytes);
    span->end = address + static_cast<std::uintptr_t>(forward_bytes);
    return span->end >= span->begin &&
           span->end - span->begin == total_bytes;
}

bool overlaps(const AddressSpan &left, const AddressSpan &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
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

bool valid_limits(const Balance3FinishLimits &limits) noexcept {
    return std::isfinite(limits.lower_even) &&
           std::isfinite(limits.upper_even) &&
           std::isfinite(limits.lower_odd) &&
           std::isfinite(limits.upper_odd) &&
           limits.lower_even <= limits.upper_even &&
           limits.lower_odd <= limits.upper_odd;
}

// Force every non-fused intermediate back to binary32 even on targets that
// permit excess precision. std::fma below preserves the explicitly fused
// instructions recovered from both finish paths.
float rounded_f32(float value) noexcept {
    volatile float rounded = value;
    return rounded;
}

float ieee_min(float left, float right) noexcept {
    if (left < right) return left;
    if (right < left) return right;
    if (left == 0.0f && right == 0.0f) {
        return (std::signbit(left) || std::signbit(right)) ? -0.0f : 0.0f;
    }
    return left;
}

float ieee_max(float left, float right) noexcept {
    if (left > right) return left;
    if (right > left) return right;
    if (left == 0.0f && right == 0.0f) {
        return (std::signbit(left) && std::signbit(right)) ? -0.0f : 0.0f;
    }
    return left;
}

Balance3FinishStatus validate(
    const float *source, std::size_t stride, const float *diff,
    std::size_t diff_count, std::size_t width, int phase, int row_even,
    const Balance3FinishLimits &provided_limits, float *dst,
    std::size_t dst_count, Balance3FinishLimits *copied_limits) noexcept {
    if (source == nullptr || diff == nullptr || dst == nullptr) {
        return Balance3FinishStatus::invalid_argument;
    }

    // Limits can be stored in destination memory. Snapshot all fields before
    // the first possible write and use only this local copy afterwards.
    *copied_limits = {provided_limits.lower_even, provided_limits.upper_even,
                      provided_limits.lower_odd, provided_limits.upper_odd};

    if (!valid_width(width)) {
        return Balance3FinishStatus::invalid_dimensions;
    }
    if (width > std::numeric_limits<std::size_t>::max() - 2u) {
        return Balance3FinishStatus::arithmetic_overflow;
    }
    if (!valid_stride(stride) || stride < width + 2u) {
        return Balance3FinishStatus::invalid_stride;
    }
    if (width > static_cast<std::uintmax_t>(
                    std::numeric_limits<std::ptrdiff_t>::max()) ||
        stride > static_cast<std::uintmax_t>(
                     std::numeric_limits<std::ptrdiff_t>::max())) {
        return Balance3FinishStatus::arithmetic_overflow;
    }
    if (phase != 0 && phase != 1) {
        return Balance3FinishStatus::invalid_parameters;
    }
    if (row_even != 0 && row_even != 1) {
        return Balance3FinishStatus::invalid_parameters;
    }
    if (!valid_limits(*copied_limits)) {
        return Balance3FinishStatus::invalid_parameters;
    }
    if (diff_count < width || dst_count < width) {
        return Balance3FinishStatus::buffer_too_small;
    }
    if (!aligned_16(source) || !aligned_16(diff) || !aligned_16(dst)) {
        return Balance3FinishStatus::unsupported_alignment;
    }

    AddressSpan source_span;
    AddressSpan diff_span;
    AddressSpan destination_span;
    if (!checked_source_span(source, stride, width, &source_span) ||
        !checked_span(diff, width, &diff_span) ||
        !checked_span(dst, width, &destination_span)) {
        return Balance3FinishStatus::arithmetic_overflow;
    }
    if (overlaps(source_span, destination_span) ||
        overlaps(diff_span, destination_span)) {
        return Balance3FinishStatus::overlapping_buffers;
    }

    const auto stride_offset = static_cast<std::ptrdiff_t>(stride);
    for (std::size_t lane = 0; lane < width; ++lane) {
        const auto column = static_cast<std::ptrdiff_t>(lane);
        const float samples[] = {
            source[-stride_offset + column - 1],
            source[-stride_offset + column + 1],
            source[column],
            source[stride_offset + column - 1],
            source[stride_offset + column + 1],
            diff[lane],
        };
        for (const float sample : samples) {
            if (!std::isfinite(sample)) {
                return Balance3FinishStatus::unsupported_domain;
            }
        }
    }
    return Balance3FinishStatus::ok;
}

float compute_accumulator(const float *source, std::size_t stride,
                    const float *diff, std::size_t lane,
                    float sign) noexcept {
    const auto stride_offset = static_cast<std::ptrdiff_t>(stride);
    const auto column = static_cast<std::ptrdiff_t>(lane);
    float sum = rounded_f32(
        source[-stride_offset + column - 1] +
        source[-stride_offset + column + 1]);
    sum = rounded_f32(sum + source[stride_offset + column - 1]);
    sum = rounded_f32(sum + source[stride_offset + column + 1]);
    float acc = rounded_f32(sum * 0.125f);
    acc = std::fma(source[lane], 0.5f, acc);
    acc = std::fma(diff[lane], sign, acc);
    return acc;
}

Balance3FinishStatus run_finish(
    const float *source, std::size_t stride, const float *diff,
    std::size_t diff_count, std::size_t width, int phase, int row_even,
    const Balance3FinishLimits &limits, float *dst, std::size_t dst_count,
    bool optimized) noexcept {
    Balance3FinishLimits copied_limits;
    const Balance3FinishStatus status = validate(
        source, stride, diff, diff_count, width, phase, row_even, limits, dst,
        dst_count, &copied_limits);
    if (status != Balance3FinishStatus::ok) return status;

    const float lower = row_even != 0 ? copied_limits.lower_even
                                     : copied_limits.lower_odd;
    const float upper = row_even != 0 ? copied_limits.upper_even
                                     : copied_limits.upper_odd;
    const float sign = row_even != 0 ? -0.5f : 0.5f;
    for (std::size_t lane = 0; lane < width; ++lane) {
        const bool selected = optimized
                                  ? ((lane & 1u) ==
                                     static_cast<std::size_t>(phase))
                                  : ((lane & 1u) !=
                                     static_cast<std::size_t>(phase));
        if (!selected) {
            if (optimized) dst[lane] = source[lane];
            continue;
        }

        float delta = rounded_f32(
            compute_accumulator(source, stride, diff, lane, sign) - source[lane]);
        if (optimized) {
            delta = ieee_max(lower, ieee_min(delta, upper));
            const float value = rounded_f32(source[lane] + delta);
            dst[lane] = ieee_max(0.0f, ieee_min(value, 1.0f));
        } else {
            if (!(delta < upper)) delta = upper;
            if (lower > delta) delta = lower;
            float value = rounded_f32(source[lane] + delta);
            value = ieee_min(value, 1.0f);  // FMINNM(value, 1)
            if (value < 0.0f) value = 0.0f;
            dst[lane] = value;
        }
    }
    return Balance3FinishStatus::ok;
}

}  // namespace

Balance3FinishStatus balance3_finish_prepared_row_0081403c(
    const float *source, std::size_t stride, const float *diff,
    std::size_t diff_count, std::size_t width, int phase, int row_even,
    const Balance3FinishLimits &limits, float *dst,
    std::size_t dst_count) noexcept {
    return run_finish(source, stride, diff, diff_count, width, phase, row_even,
                      limits, dst, dst_count, true);
}

Balance3FinishStatus balance3_finish_prepared_row_013312fc(
    const float *source, std::size_t stride, const float *diff,
    std::size_t diff_count, std::size_t width, int phase, int row_even,
    const Balance3FinishLimits &limits, float *dst,
    std::size_t dst_count) noexcept {
    return run_finish(source, stride, diff, diff_count, width, phase, row_even,
                      limits, dst, dst_count, false);
}

Balance3FinishStatus balance3_basic_prepared_row_007ed860(
    const float *source, std::size_t stride, const float *diff,
    std::size_t diff_count, std::size_t width, int phase, int row_even,
    float *dst, std::size_t dst_count) noexcept {
    const Balance3FinishLimits unused_limits{};
    Balance3FinishLimits copied_limits;
    const auto status = validate(source, stride, diff, diff_count, width,
                                 phase, row_even, unused_limits, dst,
                                 dst_count, &copied_limits);
    if (status != Balance3FinishStatus::ok) return status;
    const float sign = row_even != 0 ? -0.5f : 0.5f;
    for (std::size_t lane = 0; lane < width; ++lane) {
        if ((lane & 1u) == static_cast<std::size_t>(phase)) continue;
        float value = compute_accumulator(source, stride, diff, lane, sign);
        // FMINNM with numeric 1; finite sums may overflow to infinity.
        // The subsequent FCSEL MI preserves -0.
        value = std::fmin(value, 1.0f);
        if (value < 0.0f) value = 0.0f;
        dst[lane] = value;
    }
    return Balance3FinishStatus::ok;
}

}  // namespace imprint::acr
