#pragma once

#include <array>
#include <cstdint>

namespace imprint::acr {

// Prepared numeric snapshot, not an opaque Adobe params object. Header byte i
// represents P+0x4+i; the current slots are Basic IDs 0..131 (P+0x18 onward).
struct BasicFlagSlotsInput {
    std::array<std::int32_t, 132> current{};
    std::array<std::uint8_t, 16> header{};
};

struct BasicFlagSlotsResult {
    std::array<std::int32_t, 132> values{};
    // Bit ID records execution of the store, including an unchanged value.
    // Bits 132..159 are always zero.
    std::array<std::uint32_t, 5> reset_mask{};
};

// Only b69664's b69684..b696c8 Basic-slot pre-loop: a mapped, exactly-zero
// activation byte resets that ID to b5df18(ID, 2). Unmapped IDs always retain
// their full int32 input, including sentinels. No later curve/vector resets,
// header writes, b757d0 sentinel normalization, Look merge, or pixel work.
void normalize_basic_flag_slots(const BasicFlagSlotsInput& input,
                                BasicFlagSlotsResult& result) noexcept;

}  // namespace imprint::acr
