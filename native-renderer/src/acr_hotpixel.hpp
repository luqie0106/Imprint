#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class HotPixelStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_parameters = 2,
    overlap = 3,
};

// Scalar prepared-row implementation of Camera Raw routine 0132d7c8.
// `rows` and `row_counts` each contain five entries. The center of rows[2]
// starts at `left_guard`; every row must declare at least
// left_guard + width + 2 readable samples. `dst_count` is the writable
// capacity, while only dst[0..width) is written. Rows may alias one another,
// but the output span may not overlap any declared input span or either
// metadata array. All failure paths leave dst unchanged.
HotPixelStatus hotpixel_u16_row_0132d7c8(
    const std::uint16_t* rows[5], const std::size_t row_counts[5],
    std::size_t left_guard, std::uint16_t* dst, std::size_t dst_count,
    std::size_t width, std::uint32_t phase,
    std::uint16_t hot_threshold, std::uint16_t cold_threshold) noexcept;

}  // namespace imprint::acr
