#pragma once

#include "acr_profile_table_blend.hpp"

#include <array>
#include <cstdint>

namespace imprint::acr {

// Prepares the active one- or two-illuminant HueSatMap resource. Metadata is
// [projected temperature, projected tint, first-table weight]. The caller's
// outputs are updated only when the selected table copy or blend succeeds.
ProfileTableStatus prepare_huesatmap_table(
    const ProfileTableView &first, const ProfileTableView *second,
    const std::array<double, 2> &white_xy,
    double endpoint_kelvin1, double endpoint_kelvin2,
    ProfileTableBlendResult &result,
    std::array<double, 3> &metadata) noexcept;

// Derives the source and inverse lookup gains selected by the recovered
// process-version branch. Output is updated only when both gains are finite
// and positive.
ProfileTableStatus prepare_huesatmap_gain(
    std::uint32_t process_version, double source_scale,
    double context_scale20, double context_ev28,
    std::array<double, 2> &output) noexcept;

}  // namespace imprint::acr
