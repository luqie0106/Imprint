#pragma once

#include <cstddef>
#include <vector>

namespace imprint::acr {

enum class WhitesStatus : int {
    ok = 0,
    invalid_input = 1,
    non_finite_intermediate = 2,
    allocation_failure = 3,
    curve_failure = 4,
};

inline constexpr std::size_t kNegativeWhitesLutSize = 4098;

struct WhiteClipResult {
    double anchor = 0.0;
    double response = 0.0;
    double clip = 0.0;
    float clip_float = 0.0f;
    float clip_squared = 0.0f;
    bool below_one = false;
};

// Build the prepared negative-Whites LUT. `prepared_b0` is in [0, 2]. The
// caller's vector is replaced only after all 4097 samples have been evaluated.
WhitesStatus build_negative_whites_lut(
    double prepared_b0, std::vector<float> &output) noexcept;

// Prepare the scalar white-clip value from the already-split stage inputs.
// split_a98 and split_ba0 are bounded to [-120, 120], WhitesEV80 to [-2, 2],
// and stat670/stat678 must be finite. `output` is written only on success.
WhitesStatus prepare_white_clip_scalar(
    double split_a98, double split_ba0, double whites_ev80,
    double stat670, double stat678, WhiteClipResult &output) noexcept;

}  // namespace imprint::acr
