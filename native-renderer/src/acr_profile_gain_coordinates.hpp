#pragma once

#include "acr_profile_gain.hpp"
#include "acr_resample_matrix.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace imprint::acr {

// Caller-supplied coefficients for the pure, row/column coordinate transform
// recovered from the early ProfileGain CPU path. Matrix preparation and the
// source/meaning of these coefficients remain unresolved: this API does not
// construct or infer an upstream matrix, select a map, or dispatch a task.
struct ProfileGainCoordinatePrepared {
    std::array<float, 6> coefficients{};
    bool has_coordinate_task = false;
};

enum class ProfileGainCoordinateStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    buffer_too_small = 4,
    unsupported_alignment = 5,
    overlap = 6,
    arithmetic_overflow = 7,
    resource_limit = 8,
    invalid_prepared = 9,
    non_finite_input = 10,
    non_finite_intermediate = 11,
    allocation_failure = 12,
    unsupported_transform = 13,
};

// Generates two coordinate planes for rect=[top,left,bottom,right), with
// outputs ordered as coord_v then coord_u (the recovered scratch planes, not
// RGB channels). The top/left edges are inclusive and bottom/right exclusive.
// Height and width must be positive and <= INT32_MAX. row_stride must be at
// least width and <= INT32_MAX. Each declared output view must cover at least
// (height-1)*row_stride+width float elements. The two declared spans must be
// valid under size_t, PTRDIFF_MAX, and uintptr_t arithmetic; float-aligned;
// disjoint from each other; and disjoint from the prepared, rect, and outputs
// objects. Only active samples are written; row padding is preserved.
// The sum of active samples in both planes is limited to 2^26.
//
// Coefficients are [A,B,C,D,E,F]. For each integer row R, evaluate in strict
// binary32 steps:
//   r=FP32(R); rh=FP32(r+0.5f)
//   t0=fma(A,rh,FP32(B*0.5f)); b0=FP32(FP32(C+t0)+(-0.5f))
//   t1=fma(D,rh,FP32(E*0.5f)); b1=FP32(FP32(F+t1)+(-0.5f))
// Start each row with col=FP32(left). For each active column, write
//   coord_v=fma(B,col,b0); coord_u=fma(E,col,b1)
// then update col=FP32(col+1.0f). The recurrence is intentional; columns are
// not recomputed as FP32(left+j), so large float origins can stop incrementing.
// Coefficient NaN/Inf values flow through these operations; they are not
// classified as input errors. NaN payloads and FP exception flags are not
// guaranteed.
//
// The call is atomic with respect to output: both planes are staged before any
// active output element is changed. When has_coordinate_task is true, the call
// returns unsupported_transform (13) before writing either plane.
ProfileGainCoordinateStatus process_profile_gain_coordinates_f32(
    const ProfileGainCoordinatePrepared &prepared,
    const std::array<std::int32_t, 4> &rect,
    const std::array<ProfileGainWriteView, 2> &outputs,
    std::size_t row_stride) noexcept;

// Builds the six FP32 transform coefficients from caller-supplied rectangles
// and matrix_50. The rectangle values are consumed as supplied; this function
// does not infer ActiveArea semantics or select the runtime map/context.
// reference_zero and state_rect use [top,left,bottom,right), with positive
// height/width spans <= INT32_MAX. The generated double matrix is
// ((T*S)*U)*inverse(matrix_50), where T translates by reference_zero top/left,
// S scales row by reference_height/state_height and column by
// reference_width/state_width, and U translates by the two's-complement
// int32-wrapped negatives of state_rect top/left. Each complete 3x3 product
// uses ascending-k FP64 FMA accumulation beginning at +0; identity terms are
// not skipped. The first two rows are narrowed to coefficients in row-major
// order. A singular/non-finite input matrix returns invalid_prepared (9), a
// non-finite inverse/product or out-of-range FP32 narrowing returns
// non_finite_intermediate (11), and every failure leaves `out` unchanged.
// has_coordinate_task=true returns unsupported_transform (13).
ProfileGainCoordinateStatus prepare_profile_gain_coordinate_fields(
    const std::array<std::int32_t, 4> &reference_zero,
    const std::array<std::int32_t, 4> &state_rect,
    const ResampleMatrix3 &matrix_50, bool has_coordinate_task,
    ProfileGainCoordinatePrepared &out) noexcept;

}  // namespace imprint::acr
