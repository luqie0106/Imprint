#include "acr_profile_gain_cache.hpp"

#include <cmath>
#include <new>
#include <stdexcept>

namespace imprint::acr {
namespace {

// Exact binary64 constant with bit pattern 0x3f847ae147ae147b.
constexpr double kHundredth = 0x1.47ae147ae147bp-7;

void copy_metadata(const ProfileGainMapData &source,
                   ProfileGainMapData &destination) noexcept {
    destination.rows = source.rows;
    destination.columns = source.columns;
    destination.points_n = source.points_n;
    destination.data_type = 2;
    destination.spacing_v = source.spacing_v;
    destination.spacing_h = source.spacing_h;
    destination.origin_v = source.origin_v;
    destination.origin_h = source.origin_h;
    destination.weights = source.weights;
    destination.gamma = source.gamma;
    destination.gain_min = source.gain_min;
    destination.gain_max = source.gain_max;
}

}  // namespace

ProfileGainMapStatus build_profile_gain_cache_map(
    const ProfileGainMapData &source, std::int32_t amount,
    std::uint8_t lookup_flag, std::uint32_t params_predicate,
    ProfileGainCacheMap &out) noexcept {
    const ProfileGainMapStatus source_status =
        validate_profile_gain_source_map(source);
    if (source_status != ProfileGainMapStatus::ok) return source_status;
    if (amount <= 0) return ProfileGainMapStatus::invalid_argument;

    const std::int32_t clamped_amount = amount < 200 ? amount : 200;
    const double amount_d = static_cast<double>(clamped_amount);
    const double d = (amount_d - 100.0) / 33.0;
    const double expanded = std::fma(d, 15.0, 100.0);
    const double a0 = clamped_amount > 100 ? expanded : amount_d;
    const float exponent = static_cast<float>(a0 * kHundredth);
    const bool negative_remap =
        lookup_flag == 0 && (params_predicate & 1u) != 0;

    ProfileGainCacheMap candidate;
    copy_metadata(source, candidate.map);
    try {
        candidate.map.values.resize(source.values.size());
    } catch (const std::bad_alloc &) {
        return ProfileGainMapStatus::allocation_failure;
    } catch (const std::length_error &) {
        return ProfileGainMapStatus::allocation_failure;
    }

    for (std::size_t i = 0; i < source.values.size(); ++i) {
        const float log_value = ::log2f(source.values[i]);
        const float u = std::fma(exponent, log_value, 0.0f);
        float transformed = u;
        if (negative_remap && u < 0.0f) {
            transformed = ::fminf(1.0f + u, 0.0f);
        }
        const float value = ::exp2f(transformed);
        if (!(std::isfinite(value) && value > 0.0f)) {
            return ProfileGainMapStatus::numeric_failure;
        }
        candidate.map.values[i] = value;
    }

    candidate.exponent = exponent;
    candidate.negative_remap = negative_remap;

    // Commit after all reads and all potentially throwing work. This is safe
    // when `source` aliases `out.map` because no source fields are read after
    // the swap, and the remaining stores are non-throwing scalar assignments.
    out.map.values.swap(candidate.map.values);
    out.map.rows = candidate.map.rows;
    out.map.columns = candidate.map.columns;
    out.map.points_n = candidate.map.points_n;
    out.map.data_type = candidate.map.data_type;
    out.map.spacing_v = candidate.map.spacing_v;
    out.map.spacing_h = candidate.map.spacing_h;
    out.map.origin_v = candidate.map.origin_v;
    out.map.origin_h = candidate.map.origin_h;
    out.map.weights = candidate.map.weights;
    out.map.gamma = candidate.map.gamma;
    out.map.gain_min = candidate.map.gain_min;
    out.map.gain_max = candidate.map.gain_max;
    out.exponent = candidate.exponent;
    out.negative_remap = candidate.negative_remap;
    return ProfileGainMapStatus::ok;
}

}  // namespace imprint::acr
