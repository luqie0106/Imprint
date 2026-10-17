#include "acr_huesatmap_prepare.hpp"
#include "imprint_renderer.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr std::size_t kMaximumEntries = 1'000'000;
constexpr std::size_t kControlCount = 4;
constexpr std::size_t kMetadataCount = 3;

bool checked_product(std::size_t left, std::size_t right,
                     std::size_t *product) noexcept {
    if (product == nullptr ||
        (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

bool valid_range(const void *pointer, std::size_t bytes,
                 std::size_t alignment) noexcept {
    if (pointer == nullptr || alignment == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return begin % alignment == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - begin;
}

bool overlaps(const void *left, std::size_t left_bytes,
              const void *right, std::size_t right_bytes) noexcept {
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    return left_begin <= right_begin
               ? right_begin - left_begin < left_bytes
               : left_begin - right_begin < right_bytes;
}

im_status map_status(imprint::acr::ProfileTableStatus status) noexcept {
    using imprint::acr::ProfileTableStatus;
    switch (status) {
        case ProfileTableStatus::ok:
            return IM_STATUS_OK;
        case ProfileTableStatus::allocation_failure:
            return IM_STATUS_RUNTIME_ERROR;
        case ProfileTableStatus::invalid_argument:
        case ProfileTableStatus::invalid_table:
        case ProfileTableStatus::invalid_curve:
        case ProfileTableStatus::invalid_rgb:
        case ProfileTableStatus::overflow:
            return IM_STATUS_INVALID_ARGUMENT;
    }
    return IM_STATUS_RUNTIME_ERROR;
}

}  // namespace

extern "C" IMPRINT_API im_status im_native_acr_prepare_huesatmap(
    const float *first, std::size_t first_values,
    const float *second, std::size_t second_values,
    std::uint32_t hue_count, std::uint32_t saturation_count,
    std::uint32_t value_count, const double *controls,
    std::size_t control_count, float *output, std::size_t output_values,
    double *metadata, std::size_t metadata_count) {
    if (first == nullptr || controls == nullptr || output == nullptr ||
        metadata == nullptr || control_count != kControlCount ||
        metadata_count != kMetadataCount || hue_count < 1 ||
        saturation_count < 2 || value_count < 1) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    std::size_t entries = static_cast<std::size_t>(hue_count);
    if (!checked_product(entries, static_cast<std::size_t>(saturation_count),
                         &entries) ||
        !checked_product(entries, static_cast<std::size_t>(value_count),
                         &entries) ||
        entries == 0 || entries > kMaximumEntries) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    std::size_t expected_values = 0;
    if (!checked_product(entries, std::size_t{3}, &expected_values) ||
        first_values != expected_values || output_values != expected_values) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    const bool has_second = second != nullptr || second_values != 0;
    if (has_second && (second == nullptr || second_values != expected_values)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    if (!has_second && (second != nullptr || second_values != 0)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    const std::size_t table_bytes = expected_values * sizeof(float);
    constexpr std::size_t control_bytes = kControlCount * sizeof(double);
    constexpr std::size_t metadata_bytes = kMetadataCount * sizeof(double);
    if (!valid_range(first, table_bytes, alignof(float)) ||
        (has_second && !valid_range(second, table_bytes, alignof(float))) ||
        !valid_range(controls, control_bytes, alignof(double)) ||
        !valid_range(output, table_bytes, alignof(float)) ||
        !valid_range(metadata, metadata_bytes, alignof(double))) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    if (overlaps(first, table_bytes, output, table_bytes) ||
        overlaps(first, table_bytes, metadata, metadata_bytes) ||
        overlaps(controls, control_bytes, output, table_bytes) ||
        overlaps(controls, control_bytes, metadata, metadata_bytes) ||
        overlaps(output, table_bytes, metadata, metadata_bytes) ||
        (has_second &&
         (overlaps(second, table_bytes, output, table_bytes) ||
          overlaps(second, table_bytes, metadata, metadata_bytes)))) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    std::array<double, kControlCount> checked_controls{};
    for (std::size_t index = 0; index < checked_controls.size(); ++index) {
        if (!std::isfinite(controls[index])) return IM_STATUS_INVALID_ARGUMENT;
        checked_controls[index] = controls[index];
    }

    const imprint::acr::ProfileTableView first_view{
        first, entries, static_cast<std::size_t>(hue_count),
        static_cast<std::size_t>(saturation_count),
        static_cast<std::size_t>(value_count)};
    const imprint::acr::ProfileTableView second_view{
        second, has_second ? entries : 0,
        static_cast<std::size_t>(hue_count),
        static_cast<std::size_t>(saturation_count),
        static_cast<std::size_t>(value_count)};
    imprint::acr::ProfileTableBlendResult prepared;
    std::array<double, kMetadataCount> prepared_metadata{};
    const auto prepare_status = imprint::acr::prepare_huesatmap_table(
        first_view, has_second ? &second_view : nullptr,
        {checked_controls[0], checked_controls[1]}, checked_controls[2],
        checked_controls[3], prepared, prepared_metadata);
    if (prepare_status != imprint::acr::ProfileTableStatus::ok) {
        return map_status(prepare_status);
    }
    if (prepared.values.size() != expected_values ||
        prepared.hue_count != hue_count ||
        prepared.saturation_count != saturation_count ||
        prepared.value_count != value_count) {
        return IM_STATUS_RUNTIME_ERROR;
    }

    std::memcpy(output, prepared.values.data(), table_bytes);
    std::memcpy(metadata, prepared_metadata.data(), metadata_bytes);
    return IM_STATUS_OK;
}
