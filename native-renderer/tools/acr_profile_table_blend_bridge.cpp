#include "../src/acr_profile_table_blend.hpp"

#include <cstddef>
#include <new>

using imprint::acr::ProfileTableBlendResult;
using imprint::acr::ProfileTableStatus;
using imprint::acr::ProfileTableView;

extern "C" void *acr_test_profile_table_blend_result_create() noexcept {
    try {
        return new ProfileTableBlendResult();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void acr_test_profile_table_blend_result_destroy(
    void *opaque) noexcept {
    delete static_cast<ProfileTableBlendResult *>(opaque);
}

extern "C" const float *acr_test_profile_table_blend_result_data(
    const void *opaque) noexcept {
    if (opaque == nullptr) return nullptr;
    const auto &result = *static_cast<const ProfileTableBlendResult *>(opaque);
    return result.values.empty() ? nullptr : result.values.data();
}

extern "C" std::size_t acr_test_profile_table_blend_result_size(
    const void *opaque) noexcept {
    if (opaque == nullptr) return 0;
    return static_cast<const ProfileTableBlendResult *>(opaque)->values.size();
}

extern "C" std::size_t acr_test_profile_table_blend_result_hue_count(
    const void *opaque) noexcept {
    if (opaque == nullptr) return 0;
    return static_cast<const ProfileTableBlendResult *>(opaque)->hue_count;
}

extern "C" std::size_t acr_test_profile_table_blend_result_saturation_count(
    const void *opaque) noexcept {
    if (opaque == nullptr) return 0;
    return static_cast<const ProfileTableBlendResult *>(opaque)->saturation_count;
}

extern "C" std::size_t acr_test_profile_table_blend_result_value_count(
    const void *opaque) noexcept {
    if (opaque == nullptr) return 0;
    return static_cast<const ProfileTableBlendResult *>(opaque)->value_count;
}

extern "C" int acr_test_blend_profile_tables(
    const float *first_data, std::size_t first_count,
    std::size_t first_hue_count, std::size_t first_saturation_count,
    std::size_t first_value_count, const float *second_data,
    std::size_t second_count, std::size_t second_hue_count,
    std::size_t second_saturation_count, std::size_t second_value_count,
    double first_weight, void *opaque_result) noexcept {
    if (opaque_result == nullptr) {
        return static_cast<int>(ProfileTableStatus::invalid_argument);
    }
    const ProfileTableView first{first_data, first_count, first_hue_count,
                                 first_saturation_count, first_value_count};
    const ProfileTableView second{second_data, second_count, second_hue_count,
                                  second_saturation_count, second_value_count};
    auto &result = *static_cast<ProfileTableBlendResult *>(opaque_result);
    return static_cast<int>(imprint::acr::blend_profile_tables(
        first, second, first_weight, result));
}
