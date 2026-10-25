#pragma once

#include "acr_float_lut.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace imprint::acr {

constexpr std::size_t kU16LutEntryCount = 65536;

struct U16LutPrepared {
    std::array<std::uint16_t, kU16LutEntryCount> values;
};

// Recovered 01300600 fixed-domain unsigned 16-bit LUT builder. The source
// float LUT must contain finite values and bit-identical guards at 4096/4097.
// The destination and source spans must be naturally aligned and disjoint.
// Every failure leaves `out` unchanged.
FloatLutStatus build_u16_lut_01300600(
    const FloatLutPrepared& input, U16LutPrepared* out) noexcept;

// Recovered 0132d484 scalar row lookup. Only the first `width` samples are
// accessed. In-place operation is allowed; partial aliases are rejected. The
// prepared table must not overlap the destination span.
FloatLutStatus apply_u16_lut_row_0132d484(
    const U16LutPrepared& lut, const std::uint16_t* source,
    std::size_t source_count, std::uint16_t* destination,
    std::size_t destination_count, std::size_t width) noexcept;

}  // namespace imprint::acr
