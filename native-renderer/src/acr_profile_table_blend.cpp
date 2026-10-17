#include "acr_profile_table_blend.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <new>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumEntries = 1'000'000;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "the recovered blend requires IEEE-754 binary32 float");

bool checked_product(std::size_t left, std::size_t right,
                     std::size_t *product) noexcept {
    if (product == nullptr ||
        (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

ProfileTableStatus validate_table(const ProfileTableView &table,
                                  std::size_t *float_count) noexcept {
    if (float_count == nullptr || table.hue_count < 1 ||
        table.saturation_count < 2 || table.value_count < 1 ||
        table.data == nullptr) {
        return ProfileTableStatus::invalid_table;
    }

    std::size_t hue_saturation = 0;
    std::size_t entry_count = 0;
    if (!checked_product(table.hue_count, table.saturation_count,
                         &hue_saturation) ||
        !checked_product(hue_saturation, table.value_count, &entry_count)) {
        return ProfileTableStatus::overflow;
    }
    if (entry_count == 0 || entry_count > kMaximumEntries ||
        entry_count != table.count) {
        return ProfileTableStatus::invalid_table;
    }

    if (!checked_product(entry_count, std::size_t{3}, float_count)) {
        return ProfileTableStatus::overflow;
    }
    std::size_t byte_count = 0;
    if (!checked_product(*float_count, sizeof(float), &byte_count)) {
        return ProfileTableStatus::overflow;
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(table.data);
    if (begin % alignof(float) != 0) {
        return ProfileTableStatus::invalid_table;
    }
    if (byte_count > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return ProfileTableStatus::overflow;
    }

    for (std::size_t index = 0; index < *float_count; index += 3) {
        const float hue = table.data[index];
        const float saturation = table.data[index + 1];
        const float value = table.data[index + 2];
        if (!std::isfinite(hue) || !std::isfinite(saturation) ||
            !std::isfinite(value) || hue < -360.0f || hue > 360.0f ||
            saturation < 0.0f || saturation > 16.0f || value < 0.0f ||
            value > 16.0f) {
            return ProfileTableStatus::invalid_table;
        }
    }
    return ProfileTableStatus::ok;
}

void commit_result(ProfileTableBlendResult &result,
                   std::vector<float> &candidate,
                   std::size_t hue_count, std::size_t saturation_count,
                   std::size_t value_count) noexcept {
    result.values.swap(candidate);
    result.hue_count = hue_count;
    result.saturation_count = saturation_count;
    result.value_count = value_count;
}

}  // namespace

ProfileTableStatus blend_profile_tables(
    const ProfileTableView &first, const ProfileTableView &second,
    double first_weight, ProfileTableBlendResult &result) noexcept {
    if (!std::isfinite(first_weight)) {
        return ProfileTableStatus::invalid_argument;
    }

    const ProfileTableView *selected = nullptr;
    if (first_weight >= 1.0) {
        selected = &first;
    } else if (first_weight <= 0.0) {
        selected = &second;
    }

    try {
        std::vector<float> candidate;
        if (selected != nullptr) {
            std::size_t float_count = 0;
            const ProfileTableStatus status =
                validate_table(*selected, &float_count);
            if (status != ProfileTableStatus::ok) return status;
            candidate.resize(float_count);
            for (std::size_t index = 0; index < float_count; ++index) {
                candidate[index] = selected->data[index];
            }
            commit_result(result, candidate, selected->hue_count,
                          selected->saturation_count,
                          selected->value_count);
            return ProfileTableStatus::ok;
        }

        std::size_t first_float_count = 0;
        ProfileTableStatus status = validate_table(first, &first_float_count);
        if (status != ProfileTableStatus::ok) return status;
        std::size_t second_float_count = 0;
        status = validate_table(second, &second_float_count);
        if (status != ProfileTableStatus::ok) return status;
        if (first.hue_count != second.hue_count ||
            first.saturation_count != second.saturation_count ||
            first.value_count != second.value_count ||
            first_float_count != second_float_count) {
            return ProfileTableStatus::invalid_table;
        }

        candidate.resize(first_float_count);
        const float weight = static_cast<float>(first_weight);
        const float inverse_weight = 1.0f - weight;
        for (std::size_t index = 0; index < first_float_count; ++index) {
            const float second_term = second.data[index] * inverse_weight;
            candidate[index] = std::fma(weight, first.data[index],
                                         second_term);
        }
        commit_result(result, candidate, first.hue_count,
                      first.saturation_count, first.value_count);
        return ProfileTableStatus::ok;
    } catch (const std::bad_alloc &) {
        return ProfileTableStatus::allocation_failure;
    } catch (...) {
        return ProfileTableStatus::allocation_failure;
    }
}

}  // namespace imprint::acr
