#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace imprint::acr {

// Caller-prepared numeric state for the recovered CPU ProfileGain evaluator.
// This describes only the supplied table evaluator; it does not construct or
// identify an Adobe ProfileGainTableMap object, select its stage, or claim the
// surrounding normalized request ABI.
// coefficients map to evaluator +0x58/+0x5c/+0x60/+0x64/+0x68;
// their terms are p0, p1, p2, min(p0,p1,p2), max(p0,p1,p2).
// transform_u is (+0x8c,+0x94), transform_v is (+0x90,+0x98), each
// [scale,bias]. U bounds are (+0x6c,+0x74); V bounds are (+0x70,+0x78).
// Table offsets, in float elements, are z + u*stride_u + v*stride_v.
struct ProfileGainPrepared {
    const float *table = nullptr;
    std::size_t table_count = 0;
    std::array<float, 5> coefficients{};
    std::array<float, 2> transform_u{};
    std::array<float, 2> transform_v{};
    float lower_u = 0.0f;
    float upper_u = 0.0f;
    float lower_v = 0.0f;
    float upper_v = 0.0f;
    float prelookup_scale = 0.0f;
    float index_scale = 0.0f;
    float transfer_scale = 0.0f;
    std::uint32_t stride_v = 0;
    std::uint32_t stride_u = 0;
    std::uint32_t max_u = 0;
    std::uint32_t max_v = 0;
    std::uint32_t max_z = 0;
};

struct ProfileGainReadView {
    const float *data = nullptr;
    std::size_t count = 0;
};

struct ProfileGainWriteView {
    float *data = nullptr;
    std::size_t count = 0;
};

enum class ProfileGainStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    buffer_too_small = 4,
    unsupported_alignment = 5,
    overlap = 6,
    arithmetic_overflow = 7,
    resource_limit = 8,
    invalid_prepared = 9,
    non_finite_input = 10,
    non_finite_intermediate = 11,
    allocation_failure = 12,
};

// Evaluates one scalar sample per (outer, inner) location. Inputs are ordered
// p0, p1, p2, coord_v, coord_u. p0/p1/p2 use value_row_stride;
// coord_v/coord_u and output use coordinate_row_stride. Counts and strides
// are float elements. Only active inner samples are read/written; row padding
// is kept.
//
// This local API accepts only positive dimensions and positive row strides no
// larger than INT32_MAX, with each stride >= inner_count. Declared spans must
// fit size_t, PTRDIFF_MAX and uintptr_t arithmetic and be naturally aligned.
// The destination's full declared span must not overlap any input span, the
// prepared table span, or the ProfileGainPrepared object; read spans may
// overlap each other. Every active input and every declared table value must
// be finite. The active output sample count plus table_count is limited to
// 2^26. The table must cover the prepared maximum indices without signed
// 32-bit address-index overflow. All prepared scalars must be finite and
// index_scale must be nonnegative; finite prelookup_scale and transfer_scale
// may be positive, zero, or negative. The raw powf transform is always
// evaluated. A non-finite selected result or a numeric value outside the
// signed int32-safe conversion domain returns non_finite_intermediate.
// Also, 0 <= lower <= upper < max+1 must hold independently for U and V;
// upper must also remain below 2^31 for signed int32-safe conversion.
// Each maximum index must be <= INT32_MAX-1. Negative row strides and wrapped
// table addresses are outside this local contract even though the raw callback
// does not validate those cases.
// The destination is updated only after the
// whole evaluation succeeds, so failures leave it and row padding unchanged.
// A numeric-path failure (status non_finite_intermediate) also covers a
// finite transformed u/v or q value outside the signed int32-safe conversion
// domain.
//
// This is a bounded numeric evaluator for caller-supplied prepared fields and
// a table. It is not the full Adobe ABI, does not construct the upstream map,
// and does not prove stage eligibility or normalized-request integration.
ProfileGainStatus process_profile_gain_prepared_f32(
    const ProfileGainPrepared &prepared,
    const std::array<ProfileGainReadView, 5> &inputs,
    ProfileGainWriteView output, std::size_t outer_count,
    std::size_t inner_count, std::size_t value_row_stride,
    std::size_t coordinate_row_stride) noexcept;

}  // namespace imprint::acr
