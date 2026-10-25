#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class Core3FormatStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    insufficient_capacity = 3,
    unsupported_alignment = 4,
    overlapping_buffers = 5,
    arithmetic_overflow = 6,
    unsupported_domain = 7,
};

// Converts one tightly packed row of normalized uint16 samples to prepared
// binary32. This covers only the aligned prepared-row value conversion at
// FUN_008096e4; it does not implement the original ABI's alignment backup,
// slack, tail, multi-plane, or arbitrary-stride behavior.
//
// `src_sample_count` and `dst_sample_count` are capacities in samples. Only
// `width` samples are read or written. Width must be positive and divisible
// by four, capacities must cover width, both pointers must be 16-byte aligned,
// and the declared width spans must not overlap. Rejected calls write nothing.
Core3FormatStatus core3_u16_to_f32_prepared_row_008096e4(
    const std::uint16_t *src, std::size_t src_sample_count, float *dst,
    std::size_t dst_sample_count, std::size_t width) noexcept;

// Converts one tightly packed prepared binary32 row to normalized uint16 at
// FUN_008103dc. Only finite values in the closed interval [0, 1] are in the
// supported mathematical subdomain; unsupported values reject the whole row
// before any output is written. This does not reproduce out-of-domain
// narrowing behavior or the original ABI's alignment backup, slack, tail,
// multi-plane, or arbitrary-stride behavior.
//
// Counts are capacities in samples; only `width` samples are accessed. Width
// and alignment requirements match the forward conversion.
Core3FormatStatus core3_f32_to_u16_prepared_row_008103dc(
    const float *src, std::size_t src_sample_count, std::uint16_t *dst,
    std::size_t dst_sample_count, std::size_t width) noexcept;

}  // namespace imprint::acr
