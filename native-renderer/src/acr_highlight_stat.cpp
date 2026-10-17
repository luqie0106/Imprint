#include "acr_highlight_stat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

constexpr double kLn2 = 0.693147180559945309417232121458176568;
constexpr double kMaxEpsilon = 1.0 / 4096.0;

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

template <typename Callback>
void visit_filtered(const std::vector<float> &sorted, double low,
                    double threshold, Callback callback) noexcept {
    double previous = -1000.0;
    for (const float sample : sorted) {
        const double value = static_cast<double>(sample);
        const bool discard = value < low ||
                             (value == previous && value >= threshold);
        if (!discard) callback(value);
        // The source loop updates its previous-value state for every original
        // sample, including values removed by the low gate.
        previous = value;
    }
}

}  // namespace

HighlightStatStatus compute_highlight_stat(const float *log_pixels,
                                           std::size_t count,
                                           double epsilon,
                                           HighlightStat &result) noexcept {
    if (log_pixels == nullptr) {
        return HighlightStatStatus::invalid_argument;
    }
    if (count == 0 || count > kMaxHighlightStatSamples) {
        return HighlightStatStatus::count_out_of_range;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0 || epsilon > kMaxEpsilon) {
        return HighlightStatStatus::epsilon_out_of_range;
    }
    const std::size_t input_bytes = count * sizeof(float);
    if (address_range_overflows(log_pixels, input_bytes)) {
        return HighlightStatStatus::invalid_address_range;
    }
    if (overlaps(log_pixels, input_bytes, &result, sizeof(result))) {
        return HighlightStatStatus::overlapping_buffers;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(log_pixels[i])) {
            return HighlightStatStatus::nonfinite_input;
        }
    }

    try {
        std::vector<float> sorted(log_pixels, log_pixels + count);
        std::sort(sorted.begin(), sorted.end());

        const double last_index = static_cast<double>(count - 1);
        const auto p1_index = static_cast<std::size_t>(std::floor(last_index * 0.01));
        const auto p99_index = static_cast<std::size_t>(std::floor(last_index * 0.99));
        const double p1 = static_cast<double>(sorted[p1_index]);
        const double p99 = static_cast<double>(sorted[p99_index]);
        const double threshold = std::max(
            p99 - 0.5, p1 + (p99 - p1) * 0.5);
        const double log_epsilon = std::log(epsilon) / kLn2;
        const double low = log_epsilon + 1.0;

        std::size_t filtered_count = 0;
        visit_filtered(sorted, low, threshold,
                       [&filtered_count](double) noexcept { ++filtered_count; });

        double h = log_epsilon;
        double filtered_p1 = log_epsilon;
        double filtered_p99 = log_epsilon;
        if (filtered_count == 0) {
            // The source creates one fallback element when filtering leaves
            // no samples, so the resulting order-statistic population is 1.
            filtered_count = 1;
        } else {
            const auto filtered_p1_target = std::min(
                static_cast<std::size_t>(std::floor(
                    static_cast<double>(filtered_count) * 0.01)),
                filtered_count - 1);
            const auto filtered_p99_target = std::min(
                static_cast<std::size_t>(std::floor(
                    static_cast<double>(filtered_count) * 0.99)),
                filtered_count - 1);
            const auto h_target = std::min(
                static_cast<std::size_t>(std::floor(
                    static_cast<double>(filtered_count) * 0.9999)),
                filtered_count - 1);
            std::size_t accepted_index = 0;
            visit_filtered(sorted, low, threshold,
                           [&filtered_p1, &filtered_p99, &h,
                            filtered_p1_target, filtered_p99_target, h_target,
                            &accepted_index](double value) noexcept {
                               if (accepted_index == filtered_p1_target) {
                                   filtered_p1 = value;
                               }
                               if (accepted_index == filtered_p99_target) {
                                   filtered_p99 = value;
                               }
                               if (accepted_index == h_target) h = value;
                               ++accepted_index;
                           });
        }

        const HighlightStat computed{p1, p99, threshold, low, filtered_count, h,
                                     filtered_p1, filtered_p99};
        result = computed;
        return HighlightStatStatus::ok;
    } catch (const std::bad_alloc &) {
        return HighlightStatStatus::allocation_failure;
    } catch (const std::length_error &) {
        return HighlightStatStatus::allocation_failure;
    }
}

}  // namespace imprint::acr
