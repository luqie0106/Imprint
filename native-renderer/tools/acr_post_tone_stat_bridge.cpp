#include "../src/acr_highlight_stat.hpp"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

bool overlaps(const void *left, std::size_t left_size,
              const void *right, std::size_t right_size) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_size;
    return left_start - right_start < right_size;
}

bool address_range_overflows(const void *start, std::size_t size) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(start);
    return size > std::numeric_limits<std::uintptr_t>::max() - address;
}

}  // namespace

extern "C" int acr_test_compute_post_tone_stats(const float *log_pixels,
                                                 std::uint64_t count,
                                                 double epsilon,
                                                 double *output) {
    using imprint::acr::HighlightStat;
    using imprint::acr::HighlightStatStatus;

    if (log_pixels == nullptr || output == nullptr) {
        return static_cast<int>(HighlightStatStatus::invalid_argument);
    }
    if (reinterpret_cast<std::uintptr_t>(log_pixels) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(output) % alignof(double) != 0) {
        return static_cast<int>(HighlightStatStatus::invalid_argument);
    }
    if (count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()) ||
        count > imprint::acr::kMaxHighlightStatSamples) {
        return static_cast<int>(HighlightStatStatus::count_out_of_range);
    }
    const auto sample_count = static_cast<std::size_t>(count);
    if (sample_count == 0) {
        return static_cast<int>(HighlightStatStatus::count_out_of_range);
    }

    constexpr std::size_t output_bytes = 4 * sizeof(double);
    const auto input_bytes = sample_count * sizeof(float);
    if (address_range_overflows(log_pixels, input_bytes) ||
        address_range_overflows(output, output_bytes)) {
        return static_cast<int>(HighlightStatStatus::invalid_address_range);
    }
    if (overlaps(log_pixels, input_bytes, output, output_bytes)) {
        return static_cast<int>(HighlightStatStatus::overlapping_buffers);
    }

    HighlightStat computed;
    const auto status = imprint::acr::compute_highlight_stat(
        log_pixels, sample_count, epsilon, computed);
    if (status != HighlightStatStatus::ok) return static_cast<int>(status);

    const double committed[4]{
        computed.filtered_p1,
        computed.filtered_p99,
        computed.h,
        static_cast<double>(computed.filtered_count),
    };
    for (const double value : committed) {
        if (!std::isfinite(value)) {
            return static_cast<int>(HighlightStatStatus::nonfinite_input);
        }
    }
    std::memcpy(output, committed, output_bytes);
    return 0;
}

extern "C" int acr_test_compute_post_tone_stats_with_black(
    const float *log_pixels, std::uint64_t count, double epsilon,
    double *output, std::size_t out_count) {
    using imprint::acr::HighlightStat;
    using imprint::acr::HighlightStatStatus;

    if (log_pixels == nullptr || output == nullptr) {
        return static_cast<int>(HighlightStatStatus::invalid_argument);
    }
    if (reinterpret_cast<std::uintptr_t>(log_pixels) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(output) % alignof(double) != 0) {
        return static_cast<int>(HighlightStatStatus::invalid_argument);
    }
    if (count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()) ||
        count > imprint::acr::kMaxHighlightStatSamples) {
        return static_cast<int>(HighlightStatStatus::count_out_of_range);
    }
    const auto sample_count = static_cast<std::size_t>(count);
    if (sample_count == 0) {
        return static_cast<int>(HighlightStatStatus::count_out_of_range);
    }
    if (out_count < 5) {
        return static_cast<int>(HighlightStatStatus::invalid_argument);
    }
    if (out_count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        return static_cast<int>(HighlightStatStatus::invalid_address_range);
    }

    constexpr std::size_t output_value_count = 5;
    const std::size_t output_bytes = out_count * sizeof(double);
    const auto input_bytes = sample_count * sizeof(float);
    if (address_range_overflows(log_pixels, input_bytes) ||
        address_range_overflows(output, output_bytes)) {
        return static_cast<int>(HighlightStatStatus::invalid_address_range);
    }
    if (overlaps(log_pixels, input_bytes, output, output_bytes)) {
        return static_cast<int>(HighlightStatStatus::overlapping_buffers);
    }

    HighlightStat computed;
    const auto status = imprint::acr::compute_highlight_stat(
        log_pixels, sample_count, epsilon, computed);
    if (status != HighlightStatStatus::ok) return static_cast<int>(status);

    const double committed[output_value_count]{
        computed.filtered_p0001,
        computed.filtered_p1,
        computed.filtered_p99,
        computed.h,
        static_cast<double>(computed.filtered_count),
    };
    for (const double value : committed) {
        if (!std::isfinite(value)) {
            return static_cast<int>(HighlightStatStatus::nonfinite_input);
        }
    }
    std::memcpy(output, committed, sizeof(committed));
    return 0;
}
