#pragma once

#include <array>
#include <cstdint>

namespace imprint::acr {

// Prepared inputs for only the ID 43..66 indexed profile-write loop. The
// tuples are {-100, 0, 100}, mode-2 defaults are zero, and b67ed0 returns 2.
// Lane i corresponds to Basic ID 43+i; no other Basic IDs are supported.
struct ProfileHslControlsInput {
    std::array<std::int32_t, 24> current{};
    std::array<std::int32_t, 24> profile_defaults{};
    double profile_scalar = 1.0;
    // Resolved 78abb0(profile+0x370) predicate, not a metadata parser.
    bool monochrome_profile = false;
    // Full profile+0x124 byte. Only exactly 1 disables these indexed writes.
    std::uint8_t convert_to_gray = 0;
};

struct ProfileHslControlsResult {
    std::array<std::int32_t, 24> values{};
    // dd974c results before caller rounding/clamping; skipped lanes hold C.
    std::array<double, 24> blended{};
    // Bit i means the indexed loop wrote ID 43+i, even if its value is equal.
    std::uint32_t wrote_mask = 0;
};

enum class ProfileHslControlsStatus : int {
    ok = 0,
    non_finite_scalar = 1,
};

// Finite-scalar CPU numerical subdomain of b65564 -> dd974c. The two gates
// describe b67ef0(profile+0x120, ID, 0, 0) at this loop point only. No profile
// version gate, real profile extraction, other Prepare writers, or pixel HSL
// processing is modeled. Non-finite scalars leave the entire result unchanged;
// successful evaluation commits a complete candidate after all lanes finish.
// Compile with strict FP and implicit contraction disabled; explicit fma sites
// preserve the saved instruction ordering.
ProfileHslControlsStatus process_profile_hsl_controls(
    const ProfileHslControlsInput &input,
    ProfileHslControlsResult &result) noexcept;

}  // namespace imprint::acr
