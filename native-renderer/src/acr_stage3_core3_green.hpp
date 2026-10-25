#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class Core3GreenStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    unsupported_alignment = 3,
    invalid_phase = 4,
    invalid_parameters = 5,
    arithmetic_overflow = 6,
};

struct Core3GreenParameters {
    float t = 0.0f;
    float k = 0.0f;
    float m = 0.0f;
    float q = 0.0f;
};

// One prepared-float row of the recovered FUN_0080f2b8 green stage.
// All center pointers refer to column zero of the output row, with the
// required caller-owned guards around the readable regions. The source needs
// two guard rows and two guard columns; workspace planes 0/1/2/5/7/8 need
// three guard rows and three guard columns. Strides are measured in float
// samples. The caller guarantees buffer capacities and non-overlap between
// inputs and outputs.
//
// This API supports aligned pointers, positive width divisible by four, and
// strides divisible by four and at least width. Phase 0 copies source samples
// on odd columns; phase 1 copies them on even columns. Rejected calls leave
// both outputs untouched. When parameters.m is zero, out_aux may be null and
// the auxiliary output is not written. All four control parameters must be
// finite, and q must be nonnegative.
Core3GreenStatus core3_green_prepared_float_row_0080f2b8(
    const float *source_center,
    const float *plane0_center,
    const float *plane1_center,
    const float *plane2_center,
    const float *plane5_center,
    const float *plane7_center,
    const float *plane8_center,
    std::size_t source_stride,
    std::size_t workspace_stride,
    std::size_t width,
    std::uint32_t phase,
    Core3GreenParameters parameters,
    float *out_green,
    float *out_aux) noexcept;

}  // namespace imprint::acr
