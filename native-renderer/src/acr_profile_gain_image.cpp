#include "acr_profile_gain_image.hpp"

#include "acr_profile_gain_apply.hpp"
#include "acr_profile_gain_cache.hpp"
#include "acr_profile_gain_coordinates.hpp"
#include "acr_profile_gain_map.hpp"
#include "acr_resample_matrix.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

struct AddressSpan {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool overlaps(const AddressSpan &left, const AddressSpan &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

template <typename T>
ProfileGainImageStatus make_span(const T *pointer, std::size_t count,
                                 AddressSpan *span) noexcept {
    using Status = ProfileGainImageStatus;
    if (span == nullptr || pointer == nullptr) return Status::invalid_argument;
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignof(T) != 0) return Status::unsupported_alignment;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return Status::arithmetic_overflow;
    }
    const std::size_t byte_count = count * sizeof(T);
    if (byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::ptrdiff_t>::max()) ||
        byte_count > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return Status::arithmetic_overflow;
    }
    *span = {begin, begin + byte_count};
    return Status::ok;
}

bool checked_mul(std::size_t left, std::size_t right,
                 std::size_t *result) noexcept {
    if (result == nullptr ||
        (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t *result) noexcept {
    if (result == nullptr ||
        left > std::numeric_limits<std::size_t>::max() - right) {
        return false;
    }
    *result = left + right;
    return true;
}

ProfileGainImageStatus map_map_status(ProfileGainMapStatus status) noexcept {
    using ImageStatus = ProfileGainImageStatus;
    switch (status) {
        case ProfileGainMapStatus::ok:
            return ImageStatus::ok;
        case ProfileGainMapStatus::invalid_argument:
        case ProfileGainMapStatus::invalid_header:
        case ProfileGainMapStatus::unsupported_data_type:
        case ProfileGainMapStatus::invalid_length:
        case ProfileGainMapStatus::invalid_value:
            return ImageStatus::invalid_map;
        case ProfileGainMapStatus::arithmetic_overflow:
        case ProfileGainMapStatus::pointer_overflow:
            return ImageStatus::arithmetic_overflow;
        case ProfileGainMapStatus::resource_limit:
            return ImageStatus::resource_limit;
        case ProfileGainMapStatus::allocation_failure:
            return ImageStatus::allocation_failure;
        case ProfileGainMapStatus::buffer_too_small:
            return ImageStatus::buffer_too_small;
        case ProfileGainMapStatus::overlap:
            return ImageStatus::overlap;
        case ProfileGainMapStatus::unsupported_alignment:
            return ImageStatus::unsupported_alignment;
        case ProfileGainMapStatus::invalid_geometry:
            return ImageStatus::invalid_prepared;
        case ProfileGainMapStatus::numeric_failure:
            return ImageStatus::non_finite_intermediate;
    }
    return ImageStatus::invalid_map;
}

ProfileGainImageStatus from_gain_status(ProfileGainStatus status) noexcept {
    return static_cast<ProfileGainImageStatus>(static_cast<int>(status));
}

ProfileGainImageStatus from_coordinate_status(
    ProfileGainCoordinateStatus status) noexcept {
    return static_cast<ProfileGainImageStatus>(static_cast<int>(status));
}

}  // namespace

ProfileGainImageStatus process_profile_gain_image_rgbf32(
    const float *src, std::size_t srcCount, std::size_t srcStride,
    float *dst, std::size_t dstCount, std::size_t dstStride,
    const std::uint8_t *payload, std::size_t bytes, std::uint32_t tag,
    std::uint32_t byte_order, const std::int32_t *rects,
    std::size_t rectCount, const double *matrix50,
    std::size_t matrixCount, double scalar, std::int32_t amount,
    std::uint32_t lookup_flag, std::uint32_t predicate,
    std::uint32_t has_task) noexcept {
    using Status = ProfileGainImageStatus;
    constexpr std::size_t kResourceLimit = std::size_t{1} << 26;
    constexpr std::size_t kRectCount = 16;
    constexpr std::size_t kMatrixCount = 9;
    constexpr std::size_t kRectSize = 4;
    constexpr std::int64_t kMaxDimension =
        std::numeric_limits<std::int32_t>::max();

    if (amount <= 0 || lookup_flag > std::numeric_limits<std::uint8_t>::max() ||
        has_task > 1u) {
        return Status::invalid_argument;
    }

    AddressSpan src_span;
    AddressSpan dst_span;
    AddressSpan payload_span;
    AddressSpan rect_span;
    AddressSpan matrix_span;
    Status status = make_span(src, srcCount, &src_span);
    if (status != Status::ok) return status;
    status = make_span(dst, dstCount, &dst_span);
    if (status != Status::ok) return status;
    status = make_span(payload, bytes, &payload_span);
    if (status != Status::ok) return status;
    status = make_span(rects, rectCount, &rect_span);
    if (status != Status::ok) return status;
    status = make_span(matrix50, matrixCount, &matrix_span);
    if (status != Status::ok) return status;

    if (overlaps(dst_span, payload_span) || overlaps(dst_span, rect_span) ||
        overlaps(dst_span, matrix_span)) {
        return Status::overlap;
    }
    if (overlaps(src_span, dst_span) &&
        !(src == dst && srcCount == dstCount && srcStride == dstStride)) {
        return Status::overlap;
    }

    if (rectCount < kRectCount || matrixCount < kMatrixCount) {
        return Status::buffer_too_small;
    }
    if (rectCount != kRectCount || matrixCount != kMatrixCount) {
        return Status::invalid_argument;
    }

    std::array<std::int32_t, kRectCount> rect_values{};
    std::array<ResampleMatrix3::value_type, kMatrixCount> matrix_values{};
    for (std::size_t i = 0; i < rect_values.size(); ++i) {
        rect_values[i] = rects[i];
    }
    for (std::size_t i = 0; i < matrix_values.size(); ++i) {
        matrix_values[i] = matrix50[i];
    }

    const std::array<std::int32_t, kRectSize> map_rect = {
        rect_values[0], rect_values[1], rect_values[2], rect_values[3]};
    const std::array<std::int32_t, kRectSize> reference_zero = {
        rect_values[4], rect_values[5], rect_values[6], rect_values[7]};
    const std::array<std::int32_t, kRectSize> state_rect = {
        rect_values[8], rect_values[9], rect_values[10], rect_values[11]};
    const std::array<std::int32_t, kRectSize> tile_rect = {
        rect_values[12], rect_values[13], rect_values[14], rect_values[15]};

    const std::int64_t height_wide =
        static_cast<std::int64_t>(tile_rect[2]) -
        static_cast<std::int64_t>(tile_rect[0]);
    const std::int64_t width_wide =
        static_cast<std::int64_t>(tile_rect[3]) -
        static_cast<std::int64_t>(tile_rect[1]);
    if (height_wide <= 0 || width_wide <= 0 ||
        height_wide > kMaxDimension || width_wide > kMaxDimension) {
        return Status::invalid_dimensions;
    }
    const std::size_t height = static_cast<std::size_t>(height_wide);
    const std::size_t width = static_cast<std::size_t>(width_wide);

    std::size_t active_row_width = 0;
    if (!checked_mul(width, 3, &active_row_width)) {
        return Status::arithmetic_overflow;
    }
    if (srcStride < active_row_width || dstStride < active_row_width ||
        srcStride > static_cast<std::size_t>(kMaxDimension) ||
        dstStride > static_cast<std::size_t>(kMaxDimension) ||
        active_row_width > static_cast<std::size_t>(kMaxDimension)) {
        return Status::invalid_stride;
    }

    std::size_t source_rows = 0;
    std::size_t destination_rows = 0;
    std::size_t source_required = 0;
    std::size_t destination_required = 0;
    if (!checked_mul(height - 1, srcStride, &source_rows) ||
        !checked_add(source_rows, active_row_width, &source_required) ||
        !checked_mul(height - 1, dstStride, &destination_rows) ||
        !checked_add(destination_rows, active_row_width,
                     &destination_required)) {
        return Status::arithmetic_overflow;
    }
    if (srcCount < source_required || dstCount < destination_required) {
        return Status::buffer_too_small;
    }

    std::size_t sample_count = 0;
    std::size_t source_plane_count = 0;
    if (!checked_mul(height, width, &sample_count) ||
        !checked_mul(sample_count, 3, &source_plane_count)) {
        return Status::arithmetic_overflow;
    }

    if (has_task != 0) return Status::unsupported_transform;

    // These stage limits are known before parsing or allocating the map.
    if (sample_count > kResourceLimit / 2 ||
        sample_count > kResourceLimit / 3) {
        return Status::resource_limit;
    }

    ProfileGainMapData source_map;
    status = map_map_status(parse_profile_gain_map_payload(
        payload, bytes, tag, static_cast<ProfileGainMapByteOrder>(byte_order),
        source_map));
    if (status != Status::ok) return status;

    ProfileGainCacheMap cache;
    ProfileGainPrepared gain_prepared;
    if (amount == 100 && (predicate & 1u) == 0) {
        status = map_map_status(prepare_profile_gain_map(
            source_map, map_rect, scalar, gain_prepared));
    } else {
        status = map_map_status(build_profile_gain_cache_map(
            source_map, amount, static_cast<std::uint8_t>(lookup_flag),
            predicate, cache));
        if (status != Status::ok) return status;
        status = map_map_status(prepare_cached_profile_gain_map(
            cache.map, map_rect, scalar, gain_prepared));
    }
    if (status != Status::ok) return status;
    if (gain_prepared.table_count > kResourceLimit ||
        sample_count > kResourceLimit - gain_prepared.table_count) {
        return Status::resource_limit;
    }

    ResampleMatrix3 matrix{};
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = matrix_values[i];
    }
    ProfileGainCoordinatePrepared coordinate_prepared;
    const auto coordinate_prepare_status =
        prepare_profile_gain_coordinate_fields(
            reference_zero, state_rect, matrix, has_task != 0,
            coordinate_prepared);
    status = from_coordinate_status(coordinate_prepare_status);
    if (status != Status::ok) return status;

    try {
        std::vector<float> source_planes(source_plane_count);
        std::vector<float> coord_v(sample_count);
        std::vector<float> coord_u(sample_count);
        std::vector<float> gain(sample_count);

        for (std::size_t row = 0; row < height; ++row) {
            const float *source_row = src + row * srcStride;
            for (std::size_t column = 0; column < width; ++column) {
                const std::size_t pixel = row * width + column;
                source_planes[pixel] = source_row[column * 3];
                source_planes[sample_count + pixel] = source_row[column * 3 + 1];
                source_planes[2 * sample_count + pixel] =
                    source_row[column * 3 + 2];
            }
        }

        const std::array<ProfileGainWriteView, 2> coordinate_outputs = {
            ProfileGainWriteView{coord_v.data(), coord_v.size()},
            ProfileGainWriteView{coord_u.data(), coord_u.size()},
        };
        const auto coordinate_status = process_profile_gain_coordinates_f32(
            coordinate_prepared, tile_rect, coordinate_outputs, width);
        status = from_coordinate_status(coordinate_status);
        if (status != Status::ok) return status;

        const std::array<ProfileGainReadView, 5> evaluator_inputs = {
            ProfileGainReadView{source_planes.data(), sample_count},
            ProfileGainReadView{source_planes.data() + sample_count,
                               sample_count},
            ProfileGainReadView{source_planes.data() + 2 * sample_count,
                               sample_count},
            ProfileGainReadView{coord_v.data(), coord_v.size()},
            ProfileGainReadView{coord_u.data(), coord_u.size()},
        };
        const auto evaluator_status = process_profile_gain_prepared_f32(
            gain_prepared, evaluator_inputs,
            ProfileGainWriteView{gain.data(), gain.size()}, height, width,
            width, width);
        status = from_gain_status(evaluator_status);
        if (status != Status::ok) return status;

        const std::array<ProfileGainReadView, 4> apply_inputs = {
            ProfileGainReadView{source_planes.data(), sample_count},
            ProfileGainReadView{source_planes.data() + sample_count,
                               sample_count},
            ProfileGainReadView{source_planes.data() + 2 * sample_count,
                               sample_count},
            ProfileGainReadView{gain.data(), gain.size()},
        };
        const std::array<ProfileGainWriteView, 3> apply_outputs = {
            ProfileGainWriteView{source_planes.data(), sample_count},
            ProfileGainWriteView{source_planes.data() + sample_count,
                                 sample_count},
            ProfileGainWriteView{source_planes.data() + 2 * sample_count,
                                 sample_count},
        };
        const auto apply_status = process_profile_gain_apply_f32(
            apply_inputs, apply_outputs, height, width, width, width, 0.0,
            ProfileGainApplyMode::scalar, false);
        status = from_gain_status(apply_status);
        if (status != Status::ok) return status;

        // No operation below can fail. Public output is committed only after
        // every parser, preparation, coordinate, evaluation, and apply step.
        for (std::size_t row = 0; row < height; ++row) {
            float *destination_row = dst + row * dstStride;
            for (std::size_t column = 0; column < width; ++column) {
                const std::size_t pixel = row * width + column;
                destination_row[column * 3] = source_planes[pixel];
                destination_row[column * 3 + 1] =
                    source_planes[sample_count + pixel];
                destination_row[column * 3 + 2] =
                    source_planes[2 * sample_count + pixel];
            }
        }
        return Status::ok;
    } catch (const std::bad_alloc &) {
        return Status::allocation_failure;
    } catch (const std::length_error &) {
        return Status::allocation_failure;
    }
}

}  // namespace imprint::acr
