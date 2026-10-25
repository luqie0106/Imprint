#pragma once

#include "acr_profile_table.hpp"

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

// Result codes for the explicit SDR/HSM header-0 uint16 profile wrapper.
// pixel_type 3 uses unsigned uint16 samples; pixel_type 8 uses the recovered
// signed-word encoding with the 0x8000 bias. This function does not select a
// runtime selector, encoding byte, host binding, or Adobe dispatch path.
enum class ProfileU16Status : int {
    ok = 0,
    invalid_argument = 1,
    invalid_table = 2,
    buffer_too_small = 3,
    unsupported_aliasing = 4,
    arithmetic_overflow = 5,
    resource_limit = 6,
    allocation_failure = 7,
    kernel_failure = 8,
};

// Applies the recovered generic HSM/bulk profile SDR mode-1 operation to
// interleaved RGB uint16 buffers. Counts and row strides are in uint16
// samples; width and height are in pixels. Only pixel_type 3 and 8 are
// supported. The table is the prepared SDR table consumed by
// apply_profile_stage_sdr. The output is committed only after the full image
// succeeds; row padding is left untouched. Exact in-place operation is
// supported when the source and destination pointers, counts, and strides
// match. Other overlapping ranges are rejected.
ProfileU16Status apply_profile_stage_sdr_u16(
    const std::uint16_t *src, std::size_t src_count, std::size_t src_stride,
    std::size_t height, std::size_t width, const ProfileTableView &table,
    unsigned encoding, double lookup_gain, double amount,
    std::uint32_t pixel_type, std::uint16_t *dst, std::size_t dst_count,
    std::size_t dst_stride) noexcept;

}  // namespace imprint::acr
