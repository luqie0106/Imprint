#pragma once

#include <array>

namespace imprint::acr {

using ResampleMatrix3 = std::array<double, 9>;

struct OrdinaryAxisTuples {
    double vertical_translation;
    double vertical_scale;
    double vertical_cross;
    double horizontal_translation;
    double horizontal_scale;
    double horizontal_cross;
};

enum class ResampleMatrixStatus : int {
    ok = 0,
    non_finite_input = 1,
    singular_matrix = 2,
    non_finite_result = 3,
};

// Recovered 3x3 inverse used by prepared geometry constructors. Retains the
// ordered cofactors, determinant threshold and finite-intermediate domain of
// decompose_ordinary_resample_matrix. Failure leaves inverse unchanged.
ResampleMatrixStatus invert_prepared_resample_matrix(
    const ResampleMatrix3 &matrix, ResampleMatrix3 &inverse) noexcept;

// Decompose a caller-supplied finite, explicitly prepared 3x3 matrix when each
// inverse step has |det| >= 1e-10. The matrix is consumed as given; affine form
// is not checked and skew is not removed. This support domain does not
// establish Adobe runtime dispatch, the original ABI, or rendered-pixel
// equivalence. Every intermediate scalar must remain finite: this includes
// separately rounded negative products, cofactors, determinant terms and
// accumulators, inverse entries, and matrix-multiply FMA accumulators. A
// non-finite intermediate returns non_finite_result, even if later arithmetic
// could produce a finite final matrix. On failure, tuples is left unchanged.
ResampleMatrixStatus decompose_ordinary_resample_matrix(
    const ResampleMatrix3 &matrix, OrdinaryAxisTuples &tuples) noexcept;

}  // namespace imprint::acr
