#include "acr_resample_vertical.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "the recovered sampler requires IEEE-754 binary32");

bool valid_span(const float *data, std::size_t count) noexcept {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    if (count == 0) return data == nullptr ||
        reinterpret_cast<std::uintptr_t>(data) % alignof(float) == 0;
    if (data == nullptr) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    const auto byte_count = count * sizeof(float);
    return address % alignof(float) == 0 &&
           byte_count <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t &product) noexcept {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
        return false;
    product = left * right;
    return true;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t &sum) noexcept {
    if (left > std::numeric_limits<std::size_t>::max() - right) return false;
    sum = left + right;
    return true;
}

}  // namespace

ResampleVerticalStatus process_resample_vertical(
    const float *source, std::size_t source_count,
    std::size_t first_offset, std::size_t row_stride,
    const float *weights, std::size_t weight_count,
    std::size_t tap_count, std::size_t output_width,
    std::vector<float> &result) noexcept {
    if (!valid_span(source, source_count) ||
        !valid_span(weights, weight_count))
        return ResampleVerticalStatus::invalid_input;
    if (tap_count == 0 || tap_count > kResampleVerticalMaximumTaps ||
        weight_count != tap_count || output_width == 0 ||
        output_width > kResampleVerticalMaximumOutputWidth ||
        row_stride < output_width ||
        output_width > static_cast<std::size_t>(
                           std::numeric_limits<std::ptrdiff_t>::max()) ||
        tap_count > static_cast<std::size_t>(
                        std::numeric_limits<std::ptrdiff_t>::max()) ||
        source_count > static_cast<std::size_t>(
                           std::numeric_limits<std::ptrdiff_t>::max()))
        return ResampleVerticalStatus::invalid_request;

    bool needs_source = false;
    for (std::size_t tap = 0; tap < tap_count; ++tap) {
        const float weight = weights[tap];
        if (!std::isfinite(weight))
            return ResampleVerticalStatus::non_finite_weight;
        if (weight == 0.0f) continue;
        needs_source = true;

        std::size_t tap_offset = 0;
        std::size_t window_start = 0;
        if (!checked_multiply(tap, row_stride, tap_offset) ||
            !checked_add(first_offset, tap_offset, window_start) ||
            window_start > source_count ||
            output_width > source_count - window_start)
            return ResampleVerticalStatus::source_window_out_of_range;
        if (window_start > static_cast<std::size_t>(
                               std::numeric_limits<std::ptrdiff_t>::max()) ||
            output_width - 1 > static_cast<std::size_t>(
                                   std::numeric_limits<std::ptrdiff_t>::max()) -
                                   window_start)
            return ResampleVerticalStatus::source_window_out_of_range;
    }
    if (needs_source && source == nullptr)
        return ResampleVerticalStatus::invalid_input;

    try {
        std::vector<float> candidate(output_width, 0.0f);
        for (std::size_t tap = 0; tap < tap_count; ++tap) {
            const float weight = weights[tap];
            if (weight == 0.0f) continue;

            const std::size_t window_start = first_offset + tap * row_stride;
            for (std::size_t x = 0; x < output_width; ++x) {
                const float sample = source[window_start + x];
                if (!std::isfinite(sample))
                    return ResampleVerticalStatus::non_finite_source;
                candidate[x] = std::fma(sample, weight, candidate[x]);
            }
        }
        for (const float value : candidate)
            if (!std::isfinite(value))
                return ResampleVerticalStatus::non_finite_output;

        result.swap(candidate);
        return ResampleVerticalStatus::ok;
    } catch (...) {
        return ResampleVerticalStatus::allocation_failure;
    }
}

}  // namespace imprint::acr
