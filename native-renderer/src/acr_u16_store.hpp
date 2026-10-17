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

// Quantizes a prepared float32 grayscale plane to uint16. The input must be
// finite and lie in [0, 1]. On any error, output is left untouched. Input and
// output must not overlap.
U16StoreStatus apply_gray_u16_store(const float *gray,
                                    std::uint16_t *output,
                                    std::size_t pixel_count,
                                    std::uint32_t width,
                                    std::uint32_t height,
                                    std::int32_t origin_x,
                                    std::int32_t origin_y,
                                    bool dither) noexcept;

}  // namespace imprint::acr
