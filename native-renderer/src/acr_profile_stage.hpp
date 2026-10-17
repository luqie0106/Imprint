#pragma once

#include "acr_profile_table.hpp"

namespace imprint::acr {

// Recovered baseline bulk CPU stage, with explicit SDR/no-overrange selection.
// 00885b38 prepares sector-scaled four-lane entries with a duplicate hue row;
// 0135f574 / 0135f8f8 consume them. Encoding 1 uses their fast float curves,
// not the 012ebae4 helper's sampled resources. Single-value tables ignore
// encoding and lookup_gain. This does not select a live Adobe SIMD dispatch.
// Gain is [0,65536], amount [0,2], RGB [0,1]; at most 2^26 pixels.
// No output aliases with input/map. Output is unchanged on failure.
ProfileTableStatus apply_profile_stage_sdr(
    const float *input, float *output, std::size_t pixels,
    const ProfileTableView &table, unsigned encoding,
    double lookup_gain, double amount) noexcept;

}  // namespace imprint::acr
