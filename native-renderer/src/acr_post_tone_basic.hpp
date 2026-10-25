#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace imprint::acr {

inline constexpr std::size_t kPostToneBasicSlotCount = 132;
using PostToneBasicSlots = std::array<std::int32_t, kPostToneBasicSlotCount>;

// Applies only the recovered 132-slot basic-control loop to values already
// normalized at its loop entry. source_defaults is the separately produced D
// object used for slots 0x19..0x22 on the legacy process-version branch.
void normalize_post_tone_basic(std::uint32_t process_version,
                               const PostToneBasicSlots& input,
                               const PostToneBasicSlots& source_defaults,
                               PostToneBasicSlots& output) noexcept;

}  // namespace imprint::acr
