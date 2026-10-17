#pragma once

#include <cstddef>
#include <vector>

namespace imprint::acr {

enum class ResampleVerticalStatus : int {
    ok = 0,
    invalid_input = 1,
    invalid_request = 2,
    source_window_out_of_range = 3,
    non_finite_weight = 4,
    non_finite_source = 5,
    non_finite_output = 6,
    allocation_failure = 7,
};

inline constexpr std::size_t kResampleVerticalMaximumTaps = 4096;
inline constexpr std::size_t kResampleVerticalMaximumOutputWidth =
    std::size_t{1} << 26;

// Apply prepared vertical weights to a bounded source span. first_offset and
// row_stride are measured in floats; weight_count must equal tap_count.
// Zero taps and zero output width are outside this bounded interface.
// A zero weight skips its source tap, including window validation and reads.
// On failure, result is unchanged. source may point into result's storage.
// This models the scalar tap accumulation only; it does not reproduce the
// recovered routine's aligned vector padding reads or writes.
ResampleVerticalStatus process_resample_vertical(
    const float *source, std::size_t source_count,
    std::size_t first_offset, std::size_t row_stride,
    const float *weights, std::size_t weight_count,
    std::size_t tap_count, std::size_t output_width,
    std::vector<float> &result) noexcept;

}  // namespace imprint::acr
