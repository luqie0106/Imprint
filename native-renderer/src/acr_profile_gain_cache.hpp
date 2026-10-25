#pragma once

#include "acr_profile_gain_map.hpp"

#include <cstdint>

namespace imprint::acr {

// Owned ProfileGain table generated for the profile cache. The map metadata
// follows the selected source map, while its active values are newly built.
struct ProfileGainCacheMap {
    ProfileGainMapData map;
    float exponent = 0.0f;
    bool negative_remap = false;
};

// Builds the datatype-2 cache table from a source map validated in the
// serialized reader domain. `amount` must be positive and is capped at 200.
// The negative remap is enabled exactly when lookup_flag is zero and bit 0
// of params_predicate is set. Values use CPU log2f, float fma, and exp2f;
// cross-libm bitwise parity is not promised. `source` may alias
// `out.map`. Failure leaves every field, pointer, and capacity in `out`
// unchanged.
ProfileGainMapStatus build_profile_gain_cache_map(
    const ProfileGainMapData &source, std::int32_t amount,
    std::uint8_t lookup_flag, std::uint32_t params_predicate,
    ProfileGainCacheMap &out) noexcept;

}  // namespace imprint::acr
