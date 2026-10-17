#pragma once

#include <cstddef>

namespace imprint::acr {

enum class ProfileTableStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_table = 2,
    invalid_curve = 3,
    invalid_rgb = 4,
    overflow = 5,
    allocation_failure = 6,
};

// Interleaved [value, hue, saturation, channel] float32 table. Each entry is
// (hue shift in degrees, saturation multiplier, value multiplier).
struct ProfileTableView {
    const float *data = nullptr;
    // Number of [hue, saturation, value] modifier triples.
    std::size_t count = 0;
    std::size_t hue_count = 0;
    std::size_t saturation_count = 0;
    std::size_t value_count = 0;
};

// A prepared forward or inverse 1D curve. The caller supplies the already
// prepared samples and scale; this kernel does not build or resample curves.
struct ProfileCurveView {
    const float *data = nullptr;
    std::size_t sample_count = 0;
    float scale = 0.0f;
};

// Shared validation for prepared map resources; no pixels are processed.
ProfileTableStatus validate_profile_table_view(const ProfileTableView &table) noexcept;

// Applies the recovered prepared ordinary-SDR profile/LookTable pixel kernel
// to interleaved float32 RGB. Curves must either both be valid or both be the
// canonical empty view {nullptr, 0, 0}. Output is committed only after all
// pixels succeed. This API does not claim Adobe rendered-image equivalence.
ProfileTableStatus apply_profile_table(
    const float *input_rgb, float *output_rgb, std::size_t pixel_count,
    const ProfileTableView &table, const ProfileCurveView &forward_curve,
    const ProfileCurveView &inverse_curve) noexcept;

}  // namespace imprint::acr
