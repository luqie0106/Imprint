#include "acr_huesatmap_prepare.hpp"

#include "acr_color_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <new>

namespace imprint::acr {

ProfileTableStatus prepare_huesatmap_table(
    const ProfileTableView &first, const ProfileTableView *second,
    const std::array<double, 2> &white_xy,
    double endpoint_kelvin1, double endpoint_kelvin2,
    ProfileTableBlendResult &result,
    std::array<double, 3> &metadata) noexcept {
    if (!std::isfinite(white_xy[0]) || !std::isfinite(white_xy[1]) ||
        !std::isfinite(endpoint_kelvin1) ||
        !std::isfinite(endpoint_kelvin2)) {
        return ProfileTableStatus::invalid_argument;
    }

    ProfileTableBlendResult candidate;
    std::array<double, 3> candidate_metadata{0.0, 0.0, 1.0};
    ProfileTableStatus status = ProfileTableStatus::invalid_argument;

    if (second == nullptr || endpoint_kelvin1 <= 0.0 ||
        endpoint_kelvin2 <= 0.0 || endpoint_kelvin1 == endpoint_kelvin2) {
        // Endpoint selection validates and copies only the selected first map.
        status = blend_profile_tables(first, ProfileTableView{}, 1.0, candidate);
    } else {
        WhiteTemperature projection{};
        try {
            projection = white_temperature(white_xy[0], white_xy[1]);
        } catch (const std::bad_alloc &) {
            return ProfileTableStatus::allocation_failure;
        } catch (...) {
            return ProfileTableStatus::invalid_argument;
        }

        const double minimum_kelvin =
            std::min(endpoint_kelvin1, endpoint_kelvin2);
        const double maximum_kelvin =
            std::max(endpoint_kelvin1, endpoint_kelvin2);
        double lower_endpoint_weight = 0.0;
        if (projection.temperature <= minimum_kelvin) {
            lower_endpoint_weight = 1.0;
        } else if (projection.temperature >= maximum_kelvin) {
            lower_endpoint_weight = 0.0;
        } else {
            // Keep Adobe's reciprocal-temperature arithmetic order intact.
            const double reciprocal_temperature = 1.0 / projection.temperature;
            const double reciprocal_maximum = 1.0 / maximum_kelvin;
            const double numerator =
                reciprocal_temperature - reciprocal_maximum;
            const double reciprocal_minimum = 1.0 / minimum_kelvin;
            const double denominator =
                reciprocal_minimum - reciprocal_maximum;
            lower_endpoint_weight = numerator / denominator;
        }

        const double first_weight = endpoint_kelvin1 <= endpoint_kelvin2
                                        ? lower_endpoint_weight
                                        : 1.0 - lower_endpoint_weight;
        status = blend_profile_tables(first, *second, first_weight, candidate);
        if (status == ProfileTableStatus::ok) {
            candidate_metadata = {projection.temperature, projection.tint,
                                  first_weight};
        }
    }

    if (status != ProfileTableStatus::ok) return status;

    result.values.swap(candidate.values);
    result.hue_count = candidate.hue_count;
    result.saturation_count = candidate.saturation_count;
    result.value_count = candidate.value_count;
    metadata = candidate_metadata;
    return ProfileTableStatus::ok;
}

ProfileTableStatus prepare_huesatmap_gain(
    std::uint32_t process_version, double source_scale,
    double context_scale20, double context_ev28,
    std::array<double, 2> &output) noexcept {
    if (!std::isfinite(source_scale) || !std::isfinite(context_scale20) ||
        !std::isfinite(context_ev28) || source_scale <= 0.0) {
        return ProfileTableStatus::invalid_argument;
    }

    // Preserve the recovered uint32 wraparound predicate verbatim.
    const bool modern_branch =
        (process_version + 0xfaf8ffffU) < 0xfaf8fffeU;
    double source_gain = 0.0;
    if (modern_branch) {
        const double ev_scale = std::exp2(-context_ev28);
        if (!std::isfinite(ev_scale) || ev_scale <= 0.0) {
            return ProfileTableStatus::overflow;
        }
        source_gain = source_scale * ev_scale;
    } else {
        if (context_scale20 <= 0.0) {
            return ProfileTableStatus::invalid_argument;
        }
        source_gain = source_scale * context_scale20;
    }

    if (!std::isfinite(source_gain) || source_gain <= 0.0) {
        return ProfileTableStatus::overflow;
    }
    const double lookup_gain = 1.0 / source_gain;
    if (!std::isfinite(lookup_gain) || lookup_gain <= 0.0) {
        return ProfileTableStatus::overflow;
    }

    output = {source_gain, lookup_gain};
    return ProfileTableStatus::ok;
}

}  // namespace imprint::acr
