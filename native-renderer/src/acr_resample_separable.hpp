#pragma once

#include "acr_resample_row.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

enum class ResampleSeparableStatus : int {
    ok = 0,
    invalid_input = 1,
    invalid_request = 2,
    invalid_vertical_rows = 3,
    invalid_horizontal_rows = 4,
    source_window_out_of_range = 5,
    coordinate_overflow = 6,
    non_finite_source = 7,
    non_finite_coefficients = 8,
    vertical_stage_failure = 9,
    horizontal_stage_failure = 10,
    allocation_failure = 11,
};

inline constexpr std::size_t kResampleSeparableMaximumElements =
    std::size_t{1} << 26;

// Borrowed tightly packed HWC pixels. top and left identify the absolute
// integer coordinate of data[0]. This interface accepts one or three channels.
struct SourceView {
    const float *data = nullptr;
    std::size_t count = 0;
    std::int32_t top = 0;
    std::int32_t left = 0;
    std::size_t height = 0;
    std::size_t width = 0;
    std::size_t channels = 0;
};

struct DestinationRect {
    std::int32_t top = 0;
    std::int32_t left = 0;
    std::size_t height = 0;
    std::size_t width = 0;
};

// Borrowed prepared coordinate geometry and one 128-phase coefficient table.
// origin already includes the constructor bias, and step has no cross-axis/skew
// term. This interface accepts these values as prepared; it does not derive or
// verify their relationship to a source image or to each other.
struct PreparedSeparableAxis {
    std::int64_t origin = 0;
    std::int64_t step = 0;
    std::int64_t lower = 0;
    std::int64_t upper = 0;
    PreparedResampleRows coefficients;
};

// Run the prepared ordinary vertical 0x48 stage followed by the horizontal
// 0x47 baseline over logical pixels only. The intermediate keeps the complete
// source width in tightly packed HWC order; it is not Adobe tile memory layout.
// Source, coefficient rows, and produced intermediate values must be finite for
// success. This does not prepare source EV, matrix/skew geometry, Adobe padding,
// identity/integer-copy paths, or layer selection. Pixels are neither pinned
// nor clamped to a value range. On failure result is unchanged. source.data may
// alias result's storage.
ResampleSeparableStatus process_resample_separable(
    const SourceView &source, const DestinationRect &destination,
    const PreparedSeparableAxis &vertical_axis,
    const PreparedSeparableAxis &horizontal_axis,
    std::vector<float> &result) noexcept;

}  // namespace imprint::acr
