#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

enum class ResampleRowMode : int {
    // Variable-coordinate Process32 scalar path; uses the 4..24 even-tap
    // special and otherwise the generic scalar accumulation order.
    scalar = 0,
    // Existing four-lane dot-product alternative, sharing Process32's
    // coordinate preparation and clamp; used for rows with at most eight taps.
    optimized = 1,
    // Explicit model of the ordinary 0x47 non-skew matrix-horizontal float
    // baseline (FUN_0134147c), sharing the row coordinate preparation/clamp.
    // Only 4/6/8 taps use tap1-product, tap0-FMA, then tap2..N-1-FMA;
    // other tap counts use the generic scalar accumulation order. This mode
    // does not infer or model Adobe's runtime dispatch selection.
    matrix_horizontal_baseline = 2,
};

enum class ResampleRowStatus : int {
    ok = 0,
    invalid_input = 1,
    invalid_rows = 2,
    invalid_request = 3,
    source_window_out_of_range = 4,
    coordinate_overflow = 5,
    non_finite_output = 6,
    allocation_failure = 7,
};

inline constexpr std::size_t kResampleRowPhaseCount = 128;
inline constexpr std::size_t kResampleRowStrideQuantum = 8;
inline constexpr std::size_t kResampleRowMaximumTaps = 4096;
inline constexpr std::size_t kResampleRowMaximumElements =
    std::size_t{1} << 26;

// Borrowed prepared float rows. The complete allocation contains 128 rows,
// each with stride floats; every coefficient including padding must be finite.
struct PreparedResampleRows {
    const float *data = nullptr;
    std::size_t count = 0;
    std::size_t tap_count = 0;
    std::size_t stride = 0;
};

struct RowRequest {
    std::int64_t coordinate_origin = 0;
    std::int64_t coordinate_step = 0;
    std::int64_t lower_bound = 0;
    std::int64_t upper_bound = 0;
    std::int32_t source_first_index = 0;
    std::size_t output_count = 0;
    ResampleRowMode mode = ResampleRowMode::scalar;
};

// Apply one prepared horizontal row to a source span. Source element zero has
// absolute integer index request.source_first_index. Pixels are not clamped.
// On every failure, result remains unchanged. source may point into result's
// current storage; it is read to completion before result is replaced.
ResampleRowStatus process_resample_row(
    const float *source, std::size_t source_count,
    const PreparedResampleRows &rows, const RowRequest &request,
    std::vector<float> &result) noexcept;

}  // namespace imprint::acr
