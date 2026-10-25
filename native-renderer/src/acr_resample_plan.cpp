#include "acr_resample_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace imprint::acr {
namespace {

ResamplePlanStatus prepare_axis(
    double translation, double scale, double cross,
    std::int32_t bound_start, std::int32_t bound_end,
    ResampleAxisPlan &geometry,
    ResampleCoefficientTable &coefficients) noexcept {
    ResampleAxisPlan classified{};
    const ResampleAxisStatus first_status = prepare_ordinary_resample_axis(
        translation, scale, cross, bound_start, bound_end, 1, classified);
    if (first_status == ResampleAxisStatus::unsupported_skew)
        return ResamplePlanStatus::unsupported_skew;
    if (first_status != ResampleAxisStatus::ok)
        return ResamplePlanStatus::axis_failure;

    if (classified.kind != ResampleAxisKind::resample) {
        geometry = classified;
        return ResamplePlanStatus::ok;
    }

    if (classified.step <= 0) return ResamplePlanStatus::unsupported_scale;

    // Match SCVTF -> FMUL(2^-32) -> FDIV(1, scale) using the quantized step.
    const double quantized_scale =
        static_cast<double>(classified.step) * 0x1p-32;
    const double q = 1.0 / quantized_scale;

    ResampleCoefficientTable staged_coefficients;
    if (build_resample_coefficients(q, &staged_coefficients) !=
        ResampleCoefficientStatus::ok)
        return ResamplePlanStatus::coefficient_failure;

    if (staged_coefficients.tap_count == 0 ||
        (staged_coefficients.tap_count & std::size_t{1}) != 0 ||
        staged_coefficients.tap_count / 2 >
            static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max()))
        return ResamplePlanStatus::coefficient_failure;

    const auto half_taps = static_cast<std::int32_t>(
        staged_coefficients.tap_count / 2);
    ResampleAxisPlan staged_geometry{};
    const ResampleAxisStatus second_status = prepare_ordinary_resample_axis(
        translation, scale, cross, bound_start, bound_end, half_taps,
        staged_geometry);
    if (second_status == ResampleAxisStatus::unsupported_skew)
        return ResamplePlanStatus::unsupported_skew;
    if (second_status != ResampleAxisStatus::ok ||
        staged_geometry.kind != ResampleAxisKind::resample)
        return ResamplePlanStatus::axis_failure;

    geometry = staged_geometry;
    coefficients = std::move(staged_coefficients);
    return ResamplePlanStatus::ok;
}

PreparedResampleRows rows_for(
    const ResampleCoefficientTable &table) noexcept {
    return {table.coefficients.data(), table.coefficients.size(),
            table.tap_count, table.stride};
}

}  // namespace

ResamplePlanStatus prepare_ordinary_resample_plan(
    const ResampleMatrix3 &matrix,
    const std::array<std::int32_t, 4> &bounds,
    OrdinaryResamplePlan &output) noexcept {
    const std::int64_t vertical_span =
        static_cast<std::int64_t>(bounds[2]) -
        static_cast<std::int64_t>(bounds[0]);
    const std::int64_t horizontal_span =
        static_cast<std::int64_t>(bounds[3]) -
        static_cast<std::int64_t>(bounds[1]);
    if (vertical_span <= 0 || horizontal_span <= 0 ||
        vertical_span > std::numeric_limits<std::int32_t>::max() ||
        horizontal_span > std::numeric_limits<std::int32_t>::max())
        return ResamplePlanStatus::invalid_bounds;

    OrdinaryAxisTuples tuples{};
    if (decompose_ordinary_resample_matrix(matrix, tuples) !=
        ResampleMatrixStatus::ok)
        return ResamplePlanStatus::matrix_failure;

    OrdinaryResamplePlan staged;
    const ResamplePlanStatus vertical_status = prepare_axis(
        tuples.vertical_translation, tuples.vertical_scale,
        tuples.vertical_cross, bounds[0], bounds[2], staged.vertical,
        staged.vertical_coefficients);
    if (vertical_status != ResamplePlanStatus::ok) return vertical_status;

    const ResamplePlanStatus horizontal_status = prepare_axis(
        tuples.horizontal_translation, tuples.horizontal_scale,
        tuples.horizontal_cross, bounds[1], bounds[3], staged.horizontal,
        staged.horizontal_coefficients);
    if (horizontal_status != ResamplePlanStatus::ok) return horizontal_status;

    output = std::move(staged);
    return ResamplePlanStatus::ok;
}

ResampleSeparableStatus process_ordinary_resample(
    const SourceView &source, const DestinationRect &destination,
    const OrdinaryResamplePlan &plan,
    std::vector<float> &result) noexcept {
    PreparedOrdinaryAxis vertical{plan.vertical, {}};
    PreparedOrdinaryAxis horizontal{plan.horizontal, {}};
    if (plan.vertical.kind == ResampleAxisKind::resample)
        vertical.coefficients = rows_for(plan.vertical_coefficients);
    if (plan.horizontal.kind == ResampleAxisKind::resample)
        horizontal.coefficients = rows_for(plan.horizontal_coefficients);
    return process_ordinary_resample(source, destination, vertical, horizontal,
                                     result);
}

}  // namespace imprint::acr
