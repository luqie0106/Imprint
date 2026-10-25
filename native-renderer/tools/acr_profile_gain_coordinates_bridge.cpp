#include "../src/acr_profile_gain_coordinates.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

struct AddressSpan {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool overlaps(const AddressSpan &left, const AddressSpan &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

template <typename T>
imprint::acr::ProfileGainCoordinateStatus make_span(
    const T *pointer, std::size_t count, AddressSpan *span) noexcept {
    using Status = imprint::acr::ProfileGainCoordinateStatus;
    if (span == nullptr || pointer == nullptr) return Status::invalid_argument;
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) != 0) {
        return Status::unsupported_alignment;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return Status::arithmetic_overflow;
    }
    const std::size_t bytes = count * sizeof(T);
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

}  // namespace

extern "C" int acr_profile_gain_coordinates_bridge(
    const float *coefficients, std::size_t coefficient_count,
    const std::int32_t *rect, std::size_t rect_count,
    std::uint32_t coordinate_task, float *out_v, std::size_t v_count,
    float *out_u, std::size_t u_count, std::size_t row_stride) noexcept {
    using imprint::acr::ProfileGainCoordinatePrepared;
    using imprint::acr::ProfileGainCoordinateStatus;
    using imprint::acr::ProfileGainWriteView;

    AddressSpan coefficient_span;
    AddressSpan rect_span;
    AddressSpan output_v_span;
    AddressSpan output_u_span;
    ProfileGainCoordinateStatus status =
        make_span(coefficients, coefficient_count, &coefficient_span);
    if (status != ProfileGainCoordinateStatus::ok) {
        return static_cast<int>(status);
    }
    status = make_span(rect, rect_count, &rect_span);
    if (status != ProfileGainCoordinateStatus::ok) {
        return static_cast<int>(status);
    }
    status = make_span(out_v, v_count, &output_v_span);
    if (status != ProfileGainCoordinateStatus::ok) {
        return static_cast<int>(status);
    }
    status = make_span(out_u, u_count, &output_u_span);
    if (status != ProfileGainCoordinateStatus::ok) {
        return static_cast<int>(status);
    }

    if (overlaps(output_v_span, output_u_span) ||
        overlaps(output_v_span, coefficient_span) ||
        overlaps(output_v_span, rect_span) ||
        overlaps(output_u_span, coefficient_span) ||
        overlaps(output_u_span, rect_span)) {
        return static_cast<int>(ProfileGainCoordinateStatus::overlap);
    }
    if (coordinate_task > 1u) {
        return static_cast<int>(ProfileGainCoordinateStatus::invalid_argument);
    }
    if (coordinate_task == 1u) {
        return static_cast<int>(
            ProfileGainCoordinateStatus::unsupported_transform);
    }
    if (coefficient_count < 6 || rect_count < 4) {
        return static_cast<int>(ProfileGainCoordinateStatus::buffer_too_small);
    }

    ProfileGainCoordinatePrepared prepared;
    std::memcpy(prepared.coefficients.data(), coefficients,
                prepared.coefficients.size() * sizeof(float));
    std::array<std::int32_t, 4> rect_copy{};
    std::memcpy(rect_copy.data(), rect, sizeof(rect_copy));
    const std::array<ProfileGainWriteView, 2> outputs = {
        ProfileGainWriteView{out_v, v_count},
        ProfileGainWriteView{out_u, u_count},
    };
    status = imprint::acr::process_profile_gain_coordinates_f32(
        prepared, rect_copy, outputs, row_stride);
    return static_cast<int>(status);
}

extern "C" int acr_profile_gain_coordinate_prepare_bridge(
    const std::int32_t *zero_rect, std::size_t zero_count,
    const std::int32_t *state_rect, std::size_t state_count,
    const double *matrix_50, std::size_t matrix_count,
    std::uint32_t coordinate_task, float *coefficients,
    std::size_t coefficient_count) noexcept {
    using imprint::acr::ProfileGainCoordinatePrepared;
    using imprint::acr::ProfileGainCoordinateStatus;
    using imprint::acr::ResampleMatrix3;

    AddressSpan zero_span;
    AddressSpan state_span;
    AddressSpan matrix_span;
    AddressSpan coefficients_span;
    ProfileGainCoordinateStatus status =
        make_span(zero_rect, zero_count, &zero_span);
    if (status != ProfileGainCoordinateStatus::ok) {
        return static_cast<int>(status);
    }
    status = make_span(state_rect, state_count, &state_span);
    if (status != ProfileGainCoordinateStatus::ok) {
        return static_cast<int>(status);
    }
    status = make_span(matrix_50, matrix_count, &matrix_span);
    if (status != ProfileGainCoordinateStatus::ok) {
        return static_cast<int>(status);
    }
    status = make_span(coefficients, coefficient_count, &coefficients_span);
    if (status != ProfileGainCoordinateStatus::ok) {
        return static_cast<int>(status);
    }
    if (overlaps(coefficients_span, zero_span) ||
        overlaps(coefficients_span, state_span) ||
        overlaps(coefficients_span, matrix_span)) {
        return static_cast<int>(ProfileGainCoordinateStatus::overlap);
    }
    if (coordinate_task > 1u) {
        return static_cast<int>(ProfileGainCoordinateStatus::invalid_argument);
    }
    if (coordinate_task == 1u) {
        return static_cast<int>(
            ProfileGainCoordinateStatus::unsupported_transform);
    }
    if (zero_count < 4 || state_count < 4 || matrix_count < 9 ||
        coefficient_count < 6) {
        return static_cast<int>(ProfileGainCoordinateStatus::buffer_too_small);
    }

    std::array<std::int32_t, 4> zero_copy{};
    std::array<std::int32_t, 4> state_copy{};
    ResampleMatrix3 matrix_copy{};
    std::memcpy(zero_copy.data(), zero_rect, sizeof(zero_copy));
    std::memcpy(state_copy.data(), state_rect, sizeof(state_copy));
    std::memcpy(matrix_copy.data(), matrix_50, sizeof(matrix_copy));

    ProfileGainCoordinatePrepared prepared{};
    status = imprint::acr::prepare_profile_gain_coordinate_fields(
        zero_copy, state_copy, matrix_copy, false, prepared);
    if (status != ProfileGainCoordinateStatus::ok) {
        return static_cast<int>(status);
    }
    std::memcpy(coefficients, prepared.coefficients.data(),
                prepared.coefficients.size() * sizeof(float));
    return static_cast<int>(ProfileGainCoordinateStatus::ok);
}
