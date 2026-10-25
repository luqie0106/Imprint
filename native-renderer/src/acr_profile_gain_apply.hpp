#pragma once

#include "acr_profile_gain.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace imprint::acr {

// Selects the scalar clip behavior or the CPU-vector FMIN behavior recovered
// for this numerical consumer. These values are not the source's lane ABI.
enum class ProfileGainApplyMode : std::uint32_t {
    scalar = 0,
    cpu_vector = 1,
};

using ProfileGainApplyStatus = ProfileGainStatus;

// Consumes caller-prepared p0/p1/p2 planes and a gain plane. It does not
// construct or infer any upstream ProfileGainTableMap state.
//
// Dimensions must be positive; both strides must be >= inner_count and <=
// INT32_MAX. Counts and strides are float elements. Each declared span must
// fit size_t, PTRDIFF_MAX, and uintptr_t arithmetic, and pointers must be
// naturally aligned. Only active inner_count values per row are read or
// written; output row padding is preserved. The total number of active output
// values (3 * outer_count * inner_count) is limited to 2^26.
//
// Output declared spans may not overlap each other, the gain span, or either
// view-array object. An output may overlap only its matching p input, and only
// when pointer and declared count are identical. Other read spans may overlap.
// All three planes are staged before commit, so failures leave output bytes
// unchanged. Pixel and gain NaN/Inf values follow IEEE arithmetic; finite
// results are bit-sensitive to the ordered FP32 operations below. NaN payloads
// and floating-point exception flags are not part of this API contract.
//
// The finite double pivot must be within finite float range. Its float value
// is a; d = FP32(a + -1); r = FP32(-1/d) is computed even when a == 1, after
// which r is selected as +0 for that case. factor = -d, bias = fma(-r,a,+0),
// and offset = fma(d,+0,a). Per channel, inner = fma(p,r,bias), mult =
// FP32(inner * gain), and y = fma(mult,factor,offset). No algebraic
// reassociation is permitted. With clipping enabled, scalar uses FMINNM(y,1)
// (std::fmin); cpu_vector uses NaN-propagating FMIN(y,1), with no lower clip.
ProfileGainApplyStatus process_profile_gain_apply_f32(
    const std::array<ProfileGainReadView, 4> &inputs,
    const std::array<ProfileGainWriteView, 3> &outputs,
    std::size_t outer_count, std::size_t inner_count,
    std::size_t value_row_stride, std::size_t gain_row_stride, double pivot,
    ProfileGainApplyMode mode, bool clip) noexcept;

}  // namespace imprint::acr
