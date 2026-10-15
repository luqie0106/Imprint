#pragma once

#include <cstddef>

namespace imprint::acr {

struct HighlightStat {
    double p1 = 0.0;
    double p99 = 0.0;
    double threshold = 0.0;
    double low = 0.0;
    std::size_t filtered_count = 0;
    double h = 0.0;
};

enum class HighlightStatStatus : int {
    ok = 0,
    invalid_argument = 100,
    count_out_of_range = 101,
    epsilon_out_of_range = 102,
    nonfinite_input = 103,
    overlapping_buffers = 104,
    allocation_failure = 105,
    invalid_address_range = 106,
};

// Defensive bound for this standalone kernel. The function sorts one float
// copy and holds no second per-sample buffer.
constexpr std::size_t kMaxHighlightStatSamples = std::size_t{1} << 26;

// Recover only the high-end statistic used by limitEV88 from one channel of
// float32 log pixels. `result` is assigned only on success.
HighlightStatStatus compute_highlight_stat(const float *log_pixels,
                                           std::size_t count,
                                           double epsilon,
                                           HighlightStat &result) noexcept;

}  // namespace imprint::acr
