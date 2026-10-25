#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class U16StoreStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_gray = 2,
    allocation_failure = 3,
};

using U16DitherTile = std::array<std::uint16_t, 128 * 128>;

// Returns the deterministic, row-major tile recovered from the ACR resource
// initializer. The storage is static and read-only after initialization.
const U16DitherTile &get_u16_dither_tile() noexcept;

// Quantizes a prepared float32 grayscale plane to uint16. Every input sample
// must be finite; signed and overrange finite values are accepted, while NaN
// and infinity inputs are rejected before any output is written. The fused
// scale-and-offset may overflow to infinity; its ARM FCVTZS-compatible
// conversion saturates that result to int32 before storage.
//
// With dither=false, the int32 code's low 16 bits are stored without clamping.
// With dither=true, after conversion, uint32(code) >= 65536 selects 0 for a
// negative code or 65535 otherwise; the resulting low 16 bits are stored.
// Dither tile generation and origin-based phase are unchanged. On any error,
// output is left untouched. Input and output must not overlap.
// This is an operator-local contract; it does not establish that every
// source-to-gray-to-writer path or Pin dispatch reaches this store.
U16StoreStatus apply_gray_u16_store(const float *gray,
                                    std::uint16_t *output,
                                    std::size_t pixel_count,
                                    std::uint32_t width,
                                    std::uint32_t height,
                                    std::int32_t origin_x,
                                    std::int32_t origin_y,
                                    bool dither) noexcept;

}  // namespace imprint::acr
