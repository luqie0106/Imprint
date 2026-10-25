#include "acr_profile_gain.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumActiveAndTableValues = std::size_t{1} << 26;
constexpr std::uint32_t kMaximumSignedIndex =
    static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max() - 1);
constexpr float kInt32UpperExclusive = 2147483648.0f;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool checked_product(std::size_t left, std::size_t right,
                     std::size_t *product) noexcept {
    if (product == nullptr ||
        (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

ProfileGainStatus make_range(const void *pointer, std::size_t count,
                             std::size_t element_size,
                             std::size_t alignment,
                             AddressRange *range) noexcept {
    if (range == nullptr || alignment == 0) {
        return ProfileGainStatus::invalid_argument;
    }
    if (pointer == nullptr) return ProfileGainStatus::invalid_argument;

    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) {
        return ProfileGainStatus::unsupported_alignment;
    }

    std::size_t byte_count = 0;
    if (!checked_product(count, element_size, &byte_count) ||
        byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::ptrdiff_t>::max()) ||
        byte_count > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    *range = {begin, begin + byte_count};
    return ProfileGainStatus::ok;
}

bool overlaps(const AddressRange &left, const AddressRange &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

bool required_samples(std::size_t outer_count, std::size_t row_stride,
                      std::size_t inner_count,
                      std::size_t *required) noexcept {
    if (required == nullptr || outer_count == 0) return false;
    std::size_t row_offset = 0;
    if (!checked_product(outer_count - 1, row_stride, &row_offset) ||
        row_offset > std::numeric_limits<std::size_t>::max() - inner_count) {
        return false;
    }
    *required = row_offset + inner_count;
    return true;
}

bool all_finite(const ProfileGainPrepared &prepared) noexcept {
    for (const float value : prepared.coefficients) {
        if (!std::isfinite(value)) return false;
    }
    for (const float value : prepared.transform_u) {
        if (!std::isfinite(value)) return false;
    }
    for (const float value : prepared.transform_v) {
        if (!std::isfinite(value)) return false;
    }
    return std::isfinite(prepared.lower_u) &&
           std::isfinite(prepared.upper_u) &&
           std::isfinite(prepared.lower_v) &&
           std::isfinite(prepared.upper_v) &&
           std::isfinite(prepared.prelookup_scale) &&
           std::isfinite(prepared.index_scale) &&
           std::isfinite(prepared.transfer_scale);
}

bool checked_table_extent(const ProfileGainPrepared &prepared,
                           std::size_t *maximum_offset) noexcept {
    std::size_t u_offset = 0;
    std::size_t v_offset = 0;
    if (maximum_offset == nullptr ||
        !checked_product(static_cast<std::size_t>(prepared.max_u),
                         static_cast<std::size_t>(prepared.stride_u),
                         &u_offset) ||
        !checked_product(static_cast<std::size_t>(prepared.max_v),
                         static_cast<std::size_t>(prepared.stride_v),
                         &v_offset) ||
        u_offset > std::numeric_limits<std::size_t>::max() - v_offset) {
        return false;
    }
    const std::size_t uv_offset = u_offset + v_offset;
    if (uv_offset > std::numeric_limits<std::size_t>::max() -
                        static_cast<std::size_t>(prepared.max_z)) {
        return false;
    }
    *maximum_offset = uv_offset + static_cast<std::size_t>(prepared.max_z);
    return true;
}

ProfileGainStatus validate_prepared(const ProfileGainPrepared &prepared,
                                    std::size_t *maximum_offset) noexcept {
    if (!all_finite(prepared) || prepared.table == nullptr ||
        prepared.table_count == 0 || prepared.max_u > kMaximumSignedIndex ||
        prepared.max_v > kMaximumSignedIndex ||
        prepared.max_z > kMaximumSignedIndex ||
        prepared.index_scale < 0.0f ||
        prepared.lower_u < 0.0f || prepared.lower_v < 0.0f ||
        prepared.lower_u > prepared.upper_u ||
        prepared.lower_v > prepared.upper_v ||
        static_cast<double>(prepared.upper_u) >=
            static_cast<double>(prepared.max_u) + 1.0 ||
        static_cast<double>(prepared.upper_v) >=
            static_cast<double>(prepared.max_v) + 1.0 ||
        static_cast<double>(prepared.upper_u) >=
            static_cast<double>(kInt32UpperExclusive) ||
        static_cast<double>(prepared.upper_v) >=
            static_cast<double>(kInt32UpperExclusive) ||
        prepared.table_count > static_cast<std::size_t>(
                                   std::numeric_limits<std::int32_t>::max())) {
        return ProfileGainStatus::invalid_prepared;
    }

    if (!checked_table_extent(prepared, maximum_offset)) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    if (*maximum_offset > static_cast<std::size_t>(
                             std::numeric_limits<std::int32_t>::max()) ||
        *maximum_offset >= prepared.table_count) {
        return ProfileGainStatus::invalid_prepared;
    }
    return ProfileGainStatus::ok;
}

bool compute_table_index(std::int32_t z, std::int32_t u, std::int32_t v,
                         const ProfileGainPrepared &prepared,
                         std::size_t *index) noexcept {
    std::size_t u_offset = 0;
    std::size_t v_offset = 0;
    if (index == nullptr || z < 0 || u < 0 || v < 0 ||
        !checked_product(static_cast<std::size_t>(u),
                         static_cast<std::size_t>(prepared.stride_u),
                         &u_offset) ||
        !checked_product(static_cast<std::size_t>(v),
                         static_cast<std::size_t>(prepared.stride_v),
                         &v_offset) ||
        u_offset > std::numeric_limits<std::size_t>::max() - v_offset) {
        return false;
    }
    const std::size_t uv_offset = u_offset + v_offset;
    if (uv_offset > std::numeric_limits<std::size_t>::max() -
                        static_cast<std::size_t>(z)) {
        return false;
    }
    *index = uv_offset + static_cast<std::size_t>(z);
    return *index < prepared.table_count;
}

bool finite_intermediate(float value) noexcept {
    return std::isfinite(value);
}

}  // namespace

ProfileGainStatus process_profile_gain_prepared_f32(
    const ProfileGainPrepared &prepared,
    const std::array<ProfileGainReadView, 5> &inputs,
    ProfileGainWriteView output, std::size_t outer_count,
    std::size_t inner_count, std::size_t value_row_stride,
    std::size_t coordinate_row_stride) noexcept {
    if (outer_count == 0 || inner_count == 0) {
        return ProfileGainStatus::invalid_dimensions;
    }
    if (value_row_stride < inner_count || coordinate_row_stride < inner_count ||
        value_row_stride > static_cast<std::size_t>(
                               std::numeric_limits<std::int32_t>::max()) ||
        coordinate_row_stride > static_cast<std::size_t>(
                                    std::numeric_limits<std::int32_t>::max())) {
        return ProfileGainStatus::invalid_stride;
    }

    std::size_t active_count = 0;
    if (!checked_product(outer_count, inner_count, &active_count)) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    std::size_t maximum_table_offset = 0;
    const auto metadata_status =
        validate_prepared(prepared, &maximum_table_offset);
    if (metadata_status != ProfileGainStatus::ok) return metadata_status;
    if (prepared.table_count > kMaximumActiveAndTableValues) {
        return ProfileGainStatus::resource_limit;
    }
    std::size_t total_active_and_table = 0;
    if (active_count > std::numeric_limits<std::size_t>::max() -
                           prepared.table_count) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    total_active_and_table = active_count + prepared.table_count;
    if (active_count > kMaximumActiveAndTableValues ||
        total_active_and_table > kMaximumActiveAndTableValues) {
        return ProfileGainStatus::resource_limit;
    }

    std::size_t required_value_samples = 0;
    std::size_t required_coordinate_samples = 0;
    if (!required_samples(outer_count, value_row_stride, inner_count,
                         &required_value_samples) ||
        !required_samples(outer_count, coordinate_row_stride, inner_count,
                         &required_coordinate_samples)) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    if (output.data == nullptr) return ProfileGainStatus::invalid_argument;
    if (output.count < required_coordinate_samples) {
        return ProfileGainStatus::buffer_too_small;
    }

    std::array<AddressRange, 5> input_ranges{};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].data == nullptr) return ProfileGainStatus::invalid_argument;
        const std::size_t required = i < 3 ? required_value_samples
                                           : required_coordinate_samples;
        if (inputs[i].count < required) {
            return ProfileGainStatus::buffer_too_small;
        }
        const auto input_status = make_range(
            inputs[i].data, inputs[i].count, sizeof(float), alignof(float),
            &input_ranges[i]);
        if (input_status != ProfileGainStatus::ok) return input_status;
    }

    AddressRange output_range;
    const auto output_range_status = make_range(
        output.data, output.count, sizeof(float), alignof(float),
        &output_range);
    if (output_range_status != ProfileGainStatus::ok) {
        return output_range_status;
    }

    AddressRange prepared_range;
    const auto prepared_range_status = make_range(
        &prepared, sizeof(prepared), 1, alignof(ProfileGainPrepared),
        &prepared_range);
    if (prepared_range_status != ProfileGainStatus::ok) {
        return prepared_range_status;
    }

    AddressRange table_range;
    const auto table_range_status = make_range(
        prepared.table, prepared.table_count, sizeof(float), alignof(float),
        &table_range);
    if (table_range_status != ProfileGainStatus::ok) {
        return table_range_status;
    }
    for (std::size_t i = 0; i < prepared.table_count; ++i) {
        if (!std::isfinite(prepared.table[i])) {
            return ProfileGainStatus::invalid_prepared;
        }
    }

    for (const AddressRange &input_range : input_ranges) {
        if (overlaps(output_range, input_range)) return ProfileGainStatus::overlap;
    }
    if (overlaps(output_range, table_range) ||
        overlaps(output_range, prepared_range)) {
        return ProfileGainStatus::overlap;
    }

    try {
        std::vector<float> candidate(active_count);

        for (std::size_t outer = 0; outer < outer_count; ++outer) {
            const std::size_t value_row = outer * value_row_stride;
            const std::size_t coordinate_row = outer * coordinate_row_stride;
            const std::size_t candidate_row = outer * inner_count;
            for (std::size_t inner = 0; inner < inner_count; ++inner) {
                const float p0 = inputs[0].data[value_row + inner];
                const float p1 = inputs[1].data[value_row + inner];
                const float p2 = inputs[2].data[value_row + inner];
                const float coord_v = inputs[3].data[coordinate_row + inner];
                const float coord_u = inputs[4].data[coordinate_row + inner];
                if (!std::isfinite(p0) || !std::isfinite(p1) ||
                    !std::isfinite(p2) || !std::isfinite(coord_v) ||
                    !std::isfinite(coord_u)) {
                    return ProfileGainStatus::non_finite_input;
                }

                const float shifted_u = coord_u + 0.5f;
                const float shifted_v = coord_v + 0.5f;
                if (!finite_intermediate(shifted_u) ||
                    !finite_intermediate(shifted_v)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                float u = std::fma(prepared.transform_u[0], shifted_u,
                                   prepared.transform_u[1]);
                float v = std::fma(prepared.transform_v[0], shifted_v,
                                   prepared.transform_v[1]);
                if (!finite_intermediate(u) || !finite_intermediate(v)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                u = u < prepared.upper_u ? u : prepared.upper_u;
                u = prepared.lower_u > u ? prepared.lower_u : u;
                v = v < prepared.upper_v ? v : prepared.upper_v;
                v = prepared.lower_v > v ? prepared.lower_v : v;
                if (u < 0.0f || u >= kInt32UpperExclusive || v < 0.0f ||
                    v >= kInt32UpperExclusive) {
                    return ProfileGainStatus::non_finite_intermediate;
                }

                const std::int32_t u_base = static_cast<std::int32_t>(u);
                const std::int32_t v_base = static_cast<std::int32_t>(v);
                const float u_fraction = u - static_cast<float>(u_base);
                const float v_fraction = v - static_cast<float>(v_base);
                if (!finite_intermediate(u_fraction) ||
                    !finite_intermediate(v_fraction)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                const std::int32_t u_neighbor_raw = u_base + 1;
                const std::int32_t v_neighbor_raw = v_base + 1;
                const std::int32_t u_neighbor =
                    u_neighbor_raw < static_cast<std::int32_t>(prepared.max_u)
                        ? u_neighbor_raw
                        : static_cast<std::int32_t>(prepared.max_u);
                const std::int32_t v_neighbor =
                    v_neighbor_raw < static_cast<std::int32_t>(prepared.max_v)
                        ? v_neighbor_raw
                        : static_cast<std::int32_t>(prepared.max_v);

                const float minimum12 = p1 < p2 ? p1 : p2;
                const float maximum12 = p1 > p2 ? p1 : p2;
                const float minimum = p0 < minimum12 ? p0 : minimum12;
                const float maximum = p0 > maximum12 ? p0 : maximum12;

                float z = prepared.coefficients[1] * p1;
                if (!finite_intermediate(z)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                z = std::fma(prepared.coefficients[0], p0, z);
                if (!finite_intermediate(z)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                z = std::fma(prepared.coefficients[2], p2, z);
                if (!finite_intermediate(z)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                z = std::fma(prepared.coefficients[3], minimum, z);
                if (!finite_intermediate(z)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                z = std::fma(prepared.coefficients[4], maximum, z);
                if (!finite_intermediate(z)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                z = z * prepared.prelookup_scale;
                if (!finite_intermediate(z)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                z = std::fmin(z, 1.0f);
                if (z < 0.0f) z = 0.0f;

                const float powered = ::powf(z, prepared.transfer_scale);
                const float selected =
                    prepared.transfer_scale == 1.0f ? z : powered;
                if (!finite_intermediate(selected)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                const float q = selected * prepared.index_scale;
                if (!finite_intermediate(q)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                if (q < 0.0f || q >= kInt32UpperExclusive) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                const std::int32_t z_base_raw = static_cast<std::int32_t>(q);
                const std::int32_t z_neighbor_raw = z_base_raw + 1;
                const std::int32_t z_limit =
                    static_cast<std::int32_t>(prepared.max_z);
                const std::int32_t z_base =
                    z_base_raw < z_limit ? z_base_raw : z_limit;
                const std::int32_t z_neighbor =
                    z_neighbor_raw < z_limit ? z_neighbor_raw : z_limit;
                const float z_fraction = q - static_cast<float>(z_base);
                if (!finite_intermediate(z_fraction)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }

                std::size_t index000 = 0;
                std::size_t index100 = 0;
                std::size_t index010 = 0;
                std::size_t index110 = 0;
                std::size_t index001 = 0;
                std::size_t index101 = 0;
                std::size_t index011 = 0;
                std::size_t index111 = 0;
                if (!compute_table_index(z_base, u_base, v_base, prepared,
                                         &index000) ||
                    !compute_table_index(z_neighbor, u_base, v_base, prepared,
                                         &index100) ||
                    !compute_table_index(z_base, u_neighbor, v_base, prepared,
                                         &index010) ||
                    !compute_table_index(z_neighbor, u_neighbor, v_base,
                                         prepared, &index110) ||
                    !compute_table_index(z_base, u_base, v_neighbor, prepared,
                                         &index001) ||
                    !compute_table_index(z_neighbor, u_base, v_neighbor,
                                         prepared, &index101) ||
                    !compute_table_index(z_base, u_neighbor, v_neighbor,
                                         prepared, &index011) ||
                    !compute_table_index(z_neighbor, u_neighbor, v_neighbor,
                                         prepared, &index111)) {
                    return ProfileGainStatus::invalid_prepared;
                }

                const float value000 = prepared.table[index000];
                const float value100 = prepared.table[index100];
                const float value010 = prepared.table[index010];
                const float value110 = prepared.table[index110];
                const float value001 = prepared.table[index001];
                const float value101 = prepared.table[index101];
                const float value011 = prepared.table[index011];
                const float value111 = prepared.table[index111];

                const float delta_z00 = value100 - value000;
                const float delta_z10 = value110 - value010;
                const float delta_z01 = value101 - value001;
                const float delta_z11 = value111 - value011;
                if (!finite_intermediate(delta_z00) ||
                    !finite_intermediate(delta_z10) ||
                    !finite_intermediate(delta_z01) ||
                    !finite_intermediate(delta_z11)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                const float z00 = std::fma(z_fraction, delta_z00, value000);
                const float z10 = std::fma(z_fraction, delta_z10, value010);
                const float z01 = std::fma(z_fraction, delta_z01, value001);
                const float z11 = std::fma(z_fraction, delta_z11, value011);
                if (!finite_intermediate(z00) || !finite_intermediate(z10) ||
                    !finite_intermediate(z01) || !finite_intermediate(z11)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }

                const float delta_u0 = z10 - z00;
                const float delta_u1 = z11 - z01;
                if (!finite_intermediate(delta_u0) ||
                    !finite_intermediate(delta_u1)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                const float u0 = std::fma(u_fraction, delta_u0, z00);
                const float u1 = std::fma(u_fraction, delta_u1, z01);
                if (!finite_intermediate(u0) || !finite_intermediate(u1)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }

                const float delta_v = u1 - u0;
                if (!finite_intermediate(delta_v)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                const float result = std::fma(v_fraction, delta_v, u0);
                if (!finite_intermediate(result)) {
                    return ProfileGainStatus::non_finite_intermediate;
                }
                candidate[candidate_row + inner] = result;
            }
        }

        const std::size_t row_bytes = inner_count * sizeof(float);
        for (std::size_t outer = 0; outer < outer_count; ++outer) {
            std::memcpy(output.data + outer * coordinate_row_stride,
                        candidate.data() + outer * inner_count, row_bytes);
        }
        return ProfileGainStatus::ok;
    } catch (const std::bad_alloc &) {
        return ProfileGainStatus::allocation_failure;
    } catch (const std::length_error &) {
        return ProfileGainStatus::allocation_failure;
    } catch (...) {
        return ProfileGainStatus::allocation_failure;
    }
}

}  // namespace imprint::acr
