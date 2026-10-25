#pragma once

#include "acr_resample_axis.hpp"
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
    unsupported_axis_kind = 12,
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

// Private C++ handoff for an axis already classified by the ordinary-axis
// preparation stage. Coefficients are consumed only for resample geometry;
// identity and integer-translation axes do not inspect them.
struct PreparedOrdinaryAxis {
    ResampleAxisPlan geometry;
    PreparedResampleRows coefficients;
};

// Run the prepared ordinary vertical 0x48 stage followed by the horizontal
// 0x47 baseline over logical pixels only. The intermediate keeps the complete
// source width in tightly packed HWC order; it is not Adobe tile memory layout.
// Source, coefficient rows, and produced intermediate values must be finite for
// success. This does not prepare source EV, matrix/skew geometry, Adobe padding,
// identity/integer-copy paths, or layer selection. Use
// process_ordinary_resample for classified ordinary axes. Pixels are neither
// pinned nor clamped to a value range. On failure result is unchanged.
// source.data may alias result's storage.
ResampleSeparableStatus process_resample_separable(
    const SourceView &source, const DestinationRect &destination,
    const PreparedSeparableAxis &vertical_axis,
    const PreparedSeparableAxis &horizontal_axis,
    std::vector<float> &result) noexcept;

// Execute ordinary separable geometry in vertical-then-horizontal order.
// General resample axes use their prepared coefficients, while identity axes
// copy pixels at the same absolute logical coordinates. Integer-translation
// axes materialize the relocated logical rectangle by reading source
// coordinates destination minus the stored integer offset; this is not a claim
// about raw descriptor-memory memcpy. Rectangle extents use a non-wrapping
// int32 coordinate domain, so an integer relocation outside that domain
// returns coordinate_overflow. Identity and integer-translation axes do not
// inspect coefficients. Source/coefficient finiteness, packed HWC layout,
// atomic result replacement, and aliasing behavior match
// process_resample_separable.
ResampleSeparableStatus process_ordinary_resample(
    const SourceView &source, const DestinationRect &destination,
    const PreparedOrdinaryAxis &vertical_axis,
    const PreparedOrdinaryAxis &horizontal_axis,
    std::vector<float> &result) noexcept;

}  // namespace imprint::acr
