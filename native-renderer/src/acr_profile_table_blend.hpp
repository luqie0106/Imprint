#pragma once

#include "acr_profile_table.hpp"

#include <vector>

namespace imprint::acr {

// Owned output for the prepared two-illuminant HueSatMap table blend.
// Values use the same interleaved (hue shift, saturation multiplier, value
// multiplier) float32 layout as ProfileTableView.
struct ProfileTableBlendResult {
    std::vector<float> values;
    std::size_t hue_count = 0;
    std::size_t saturation_count = 0;
    std::size_t value_count = 0;

    ProfileTableView view() const noexcept {
        return {values.empty() ? nullptr : values.data(), values.size() / 3,
                hue_count, saturation_count, value_count};
    }
};

// Blend two already prepared HueSatMap tables using the recovered float32
// schedule. Endpoint weights validate and copy only the selected table.
// Results are committed only after validation and the complete blend succeed.
ProfileTableStatus blend_profile_tables(
    const ProfileTableView &first, const ProfileTableView &second,
    double first_weight, ProfileTableBlendResult &result) noexcept;

}  // namespace imprint::acr
