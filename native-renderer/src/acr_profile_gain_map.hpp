#pragma once

#include "acr_profile_gain.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

enum class ProfileGainMapStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_header = 2,
    unsupported_data_type = 3,
    invalid_length = 4,
    invalid_value = 5,
    arithmetic_overflow = 6,
    resource_limit = 7,
    allocation_failure = 8,
    buffer_too_small = 9,
    overlap = 10,
    unsupported_alignment = 11,
    pointer_overflow = 12,
    invalid_geometry = 13,
    numeric_failure = 14,
};

enum class ProfileGainMapByteOrder : std::uint32_t {
    little = 0,
    big = 1,
};

// Owned decoded representation of a serialized ProfileGain map payload.
// `values` is a tight rows/columns/points_n table, with points_n varying
// fastest, then columns, then rows. This is a local representation; it does
// not recreate Adobe's private map allocation or stream object ABI.
struct ProfileGainMapData {
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t points_n = 0;
    std::uint32_t data_type = 3;
    double spacing_v = 1.0;
    double spacing_h = 1.0;
    double origin_v = 0.0;
    double origin_h = 0.0;
    std::array<float, 5> weights{};
    float gamma = 1.0f;
    float gain_min = 1.0f;
    float gain_max = 1.0f;
    std::vector<float> values;
};

// Parses exactly one local tag payload (52525 legacy or 52544 extended).
// `data` may be unaligned. Failure leaves every field, vector capacity, and
// vector allocation in `out` unchanged. The parser does not claim that this
// payload length or entrypoint describes the surrounding Adobe stream ABI.
ProfileGainMapStatus parse_profile_gain_map_payload(
    const std::uint8_t *data, std::size_t bytes, std::uint32_t tag,
    ProfileGainMapByteOrder byte_order, ProfileGainMapData &out) noexcept;

// Builds the numeric fields consumed by the local ProfileGain evaluator.
// `source_rect` is [top, left, bottom, right]. The returned table borrows
// `map.values.data()`; `map` must remain alive and its values must not be
// changed or reallocated for the lifetime of the prepared state.
// This uses the caller-supplied source rectangle; it does not infer Adobe's
// selected source/context bounds from a DNG ActiveArea or select a map/cache.
ProfileGainMapStatus prepare_profile_gain_map(
    const ProfileGainMapData &map,
    const std::array<std::int32_t, 4> &source_rect, double scalar,
    ProfileGainPrepared &out) noexcept;

// Validates an owned source map against the serialized reader domain.
ProfileGainMapStatus validate_profile_gain_source_map(
    const ProfileGainMapData &map) noexcept;

// Preparation for a generated cache table: requires datatype 2 and finite,
// strictly positive active values, which may exceed the serialized gain range.
// Header metadata retains the source reader validation. Geometry, borrowing,
// and failure atomicity are identical to prepare_profile_gain_map above.
ProfileGainMapStatus prepare_cached_profile_gain_map(
    const ProfileGainMapData &map,
    const std::array<std::int32_t, 4> &source_rect, double scalar,
    ProfileGainPrepared &out) noexcept;

}  // namespace imprint::acr
