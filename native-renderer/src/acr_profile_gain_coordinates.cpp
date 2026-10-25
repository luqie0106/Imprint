#include "acr_profile_gain_coordinates.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
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

ProfileGainCoordinateStatus make_float_span(float *pointer,
                                            std::size_t count,
                                            AddressSpan *span) noexcept {
    using Status = ProfileGainCoordinateStatus;
    if (span == nullptr || pointer == nullptr) return Status::invalid_argument;
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignof(float) != 0) {
        return Status::unsupported_alignment;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return Status::arithmetic_overflow;
    }
    const std::size_t bytes = count * sizeof(float);
    if (bytes > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max())) {
        return Status::arithmetic_overflow;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return Status::arithmetic_overflow;
    }
    *span = {begin, begin + bytes};
    return Status::ok;
}

ProfileGainCoordinateStatus make_object_span(const void *pointer,
                                             std::size_t bytes,
                                             AddressSpan *span) noexcept {
    using Status = ProfileGainCoordinateStatus;
    if (span == nullptr || pointer == nullptr) return Status::invalid_argument;
    if (bytes > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max())) {
        return Status::arithmetic_overflow;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return Status::arithmetic_overflow;
    }
    *span = {begin, begin + bytes};
    return Status::ok;
}

// A volatile store forces every explicitly staged arithmetic result back to
// binary32 before it participates in the next operation.
float round_to_f32(float value) noexcept {
    volatile float rounded = value;
    return rounded;
}

ProfileGainCoordinateStatus checked_rect_spans(
    const std::array<std::int32_t, 4> &rect, std::uint64_t *height,
    std::uint64_t *width) noexcept {
    using Status = ProfileGainCoordinateStatus;
    if (height == nullptr || width == nullptr) return Status::invalid_argument;
    const std::int64_t height_wide =
        static_cast<std::int64_t>(rect[2]) -
        static_cast<std::int64_t>(rect[0]);
    const std::int64_t width_wide =
        static_cast<std::int64_t>(rect[3]) -
        static_cast<std::int64_t>(rect[1]);
    constexpr std::int64_t kMaxDimension =
        std::numeric_limits<std::int32_t>::max();
    if (height_wide <= 0 || width_wide <= 0 ||
        height_wide > kMaxDimension || width_wide > kMaxDimension) {
        return Status::invalid_dimensions;
    }
    *height = static_cast<std::uint64_t>(height_wide);
    *width = static_cast<std::uint64_t>(width_wide);
    return Status::ok;
}

std::int32_t wrap_negate_i32(std::int32_t value) noexcept {
    if (value == std::numeric_limits<std::int32_t>::min()) return value;
    return -value;
}

bool multiply_finite_3x3(const ResampleMatrix3 &lhs,
                         const ResampleMatrix3 &rhs,
                         ResampleMatrix3 &product) noexcept {
    ResampleMatrix3 candidate{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double accumulator = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                accumulator = std::fma(lhs[row * 3 + k],
                                       rhs[k * 3 + column], accumulator);
                if (!std::isfinite(accumulator)) return false;
            }
            candidate[row * 3 + column] = accumulator;
        }
    }
    product = candidate;
    return true;
}

}  // namespace

ProfileGainCoordinateStatus prepare_profile_gain_coordinate_fields(
    const std::array<std::int32_t, 4> &reference_zero,
    const std::array<std::int32_t, 4> &state_rect,
    const ResampleMatrix3 &matrix_50, bool has_coordinate_task,
    ProfileGainCoordinatePrepared &out) noexcept {
    using Status = ProfileGainCoordinateStatus;
    if (has_coordinate_task) return Status::unsupported_transform;

    std::uint64_t reference_height = 0;
    std::uint64_t reference_width = 0;
    std::uint64_t state_height = 0;
    std::uint64_t state_width = 0;
    Status status = checked_rect_spans(reference_zero, &reference_height,
                                       &reference_width);
    if (status != Status::ok) return status;
    status = checked_rect_spans(state_rect, &state_height, &state_width);
    if (status != Status::ok) return status;

    ResampleMatrix3 inverse{};
    const ResampleMatrixStatus inverse_status =
        invert_prepared_resample_matrix(matrix_50, inverse);
    switch (inverse_status) {
        case ResampleMatrixStatus::ok:
            break;
        case ResampleMatrixStatus::non_finite_input:
        case ResampleMatrixStatus::singular_matrix:
            return Status::invalid_prepared;
        case ResampleMatrixStatus::non_finite_result:
            return Status::non_finite_intermediate;
    }

    const ResampleMatrix3 translate_reference = {
        1.0, 0.0, static_cast<double>(reference_zero[0]),
        0.0, 1.0, static_cast<double>(reference_zero[1]),
        0.0, 0.0, 1.0,
    };
    const double row_scale = static_cast<double>(reference_height) /
                             static_cast<double>(state_height);
    const double column_scale = static_cast<double>(reference_width) /
                                static_cast<double>(state_width);
    if (!(std::isfinite(row_scale) && std::isfinite(column_scale))) {
        return Status::non_finite_intermediate;
    }
    const ResampleMatrix3 scale = {
        row_scale, 0.0, 0.0,
        0.0, column_scale, 0.0,
        0.0, 0.0, 1.0,
    };
    const ResampleMatrix3 translate_state = {
        1.0, 0.0,
        static_cast<double>(wrap_negate_i32(state_rect[0])),
        0.0, 1.0,
        static_cast<double>(wrap_negate_i32(state_rect[1])),
        0.0, 0.0, 1.0,
    };

    ResampleMatrix3 translate_scale{};
    ResampleMatrix3 translate_scale_state{};
    ResampleMatrix3 final_matrix{};
    if (!multiply_finite_3x3(translate_reference, scale,
                             translate_scale) ||
        !multiply_finite_3x3(translate_scale, translate_state,
                             translate_scale_state) ||
        !multiply_finite_3x3(translate_scale_state, inverse, final_matrix)) {
        return Status::non_finite_intermediate;
    }

    ProfileGainCoordinatePrepared candidate{};
    for (std::size_t i = 0; i < candidate.coefficients.size(); ++i) {
        const double value = final_matrix[i];
        if (!std::isfinite(value) ||
            value > static_cast<double>(std::numeric_limits<float>::max()) ||
            value < -static_cast<double>(std::numeric_limits<float>::max())) {
            return Status::non_finite_intermediate;
        }
        const float narrowed = static_cast<float>(value);
        if (!std::isfinite(narrowed)) {
            return Status::non_finite_intermediate;
        }
        candidate.coefficients[i] = narrowed;
    }
    candidate.has_coordinate_task = false;
    out = candidate;
    return Status::ok;
}

ProfileGainCoordinateStatus process_profile_gain_coordinates_f32(
    const ProfileGainCoordinatePrepared &prepared,
    const std::array<std::int32_t, 4> &rect,
    const std::array<ProfileGainWriteView, 2> &outputs,
    std::size_t row_stride) noexcept {
    using Status = ProfileGainCoordinateStatus;

    if (prepared.has_coordinate_task) return Status::unsupported_transform;

    std::array<AddressSpan, 2> output_spans{};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const Status span_status = make_float_span(
            outputs[i].data, outputs[i].count, &output_spans[i]);
        if (span_status != Status::ok) return span_status;
    }
    if (overlaps(output_spans[0], output_spans[1])) return Status::overlap;

    AddressSpan prepared_span;
    AddressSpan rect_span;
    AddressSpan outputs_span;
    Status status = make_object_span(&prepared, sizeof(prepared),
                                     &prepared_span);
    if (status != Status::ok) return status;
    status = make_object_span(&rect, sizeof(rect), &rect_span);
    if (status != Status::ok) return status;
    status = make_object_span(&outputs, sizeof(outputs), &outputs_span);
    if (status != Status::ok) return status;
    for (const AddressSpan &output_span : output_spans) {
        if (overlaps(output_span, prepared_span) ||
            overlaps(output_span, rect_span) ||
            overlaps(output_span, outputs_span)) {
            return Status::overlap;
        }
    }

    std::uint64_t height = 0;
    std::uint64_t width = 0;
    status = checked_rect_spans(rect, &height, &width);
    if (status != Status::ok) return status;
    constexpr std::int64_t kMaxDimension =
        std::numeric_limits<std::int32_t>::max();
    if (row_stride < static_cast<std::size_t>(width) ||
        row_stride > static_cast<std::size_t>(kMaxDimension)) {
        return Status::invalid_stride;
    }

    const std::uint64_t active_per_plane = height * width;
    const std::uint64_t active_total = active_per_plane * 2u;
    constexpr std::uint64_t kMaxActiveSamples = std::uint64_t{1} << 26;
    if (active_total > kMaxActiveSamples) return Status::resource_limit;

    const std::uint64_t required_count_wide =
        (height - 1u) * static_cast<std::uint64_t>(row_stride) + width;
    if (required_count_wide > std::numeric_limits<std::size_t>::max()) {
        return Status::arithmetic_overflow;
    }
    const std::size_t required_count =
        static_cast<std::size_t>(required_count_wide);
    for (const ProfileGainWriteView &output : outputs) {
        if (output.count < required_count) return Status::buffer_too_small;
    }

    std::vector<float> staged;
    try {
        staged.resize(static_cast<std::size_t>(active_total));
    } catch (const std::bad_alloc &) {
        return Status::allocation_failure;
    } catch (...) {
        return Status::allocation_failure;
    }

    const float coefficient_a = prepared.coefficients[0];
    const float coefficient_b = prepared.coefficients[1];
    const float coefficient_c = prepared.coefficients[2];
    const float coefficient_d = prepared.coefficients[3];
    const float coefficient_e = prepared.coefficients[4];
    const float coefficient_f = prepared.coefficients[5];
    const float half_b = round_to_f32(coefficient_b * 0.5f);
    const float half_e = round_to_f32(coefficient_e * 0.5f);
    const std::size_t active_plane_size =
        static_cast<std::size_t>(active_per_plane);
    std::size_t active_index = 0;

    for (std::int64_t row = rect[0]; row < rect[2]; ++row) {
        const float row_f32 = round_to_f32(static_cast<float>(row));
        const float row_half = round_to_f32(row_f32 + 0.5f);
        const float t0 = std::fma(coefficient_a, row_half, half_b);
        const float c_plus_t0 = round_to_f32(coefficient_c + t0);
        const float b0 = round_to_f32(c_plus_t0 + (-0.5f));
        const float t1 = std::fma(coefficient_d, row_half, half_e);
        const float f_plus_t1 = round_to_f32(coefficient_f + t1);
        const float b1 = round_to_f32(f_plus_t1 + (-0.5f));

        float column = round_to_f32(static_cast<float>(rect[1]));
        for (std::uint64_t col_index = 0; col_index < width; ++col_index) {
            staged[active_index] = std::fma(coefficient_b, column, b0);
            staged[active_plane_size + active_index] =
                std::fma(coefficient_e, column, b1);
            ++active_index;
            column = round_to_f32(column + 1.0f);
        }
    }

    const std::size_t output_width = static_cast<std::size_t>(width);
    const std::size_t output_stride = row_stride;
    for (std::size_t row_index = 0;
         row_index < static_cast<std::size_t>(height); ++row_index) {
        const std::size_t destination_offset = row_index * output_stride;
        const std::size_t staged_offset = row_index * output_width;
        for (std::size_t column_index = 0; column_index < output_width;
             ++column_index) {
            outputs[0].data[destination_offset + column_index] =
                staged[staged_offset + column_index];
            outputs[1].data[destination_offset + column_index] =
                staged[active_plane_size + staged_offset + column_index];
        }
    }
    return Status::ok;
}

}  // namespace imprint::acr
