#include "acr_resample_matrix.hpp"

#include <cmath>
#include <cstddef>

namespace imprint::acr {
namespace {

using Matrix3 = ResampleMatrix3;

bool finite_matrix(const Matrix3 &matrix) noexcept {
    for (const double value : matrix)
        if (!std::isfinite(value)) return false;
    return true;
}

bool rounded_negative_product(double lhs, double rhs,
                              double &result) noexcept {
    volatile double rounded_product = lhs * rhs;
    result = -rounded_product;
    return std::isfinite(result);
}

bool cofactor(double positive_lhs, double positive_rhs,
              double negative_lhs, double negative_rhs,
              double &result) noexcept {
    double negative_product = 0.0;
    if (!rounded_negative_product(negative_lhs, negative_rhs,
                                  negative_product))
        return false;
    result = std::fma(positive_lhs, positive_rhs, negative_product);
    return std::isfinite(result);
}

ResampleMatrixStatus inverse_3x3(const Matrix3 &matrix,
                                 Matrix3 &inverse) noexcept {
    const double m00 = matrix[0];
    const double m01 = matrix[1];
    const double m02 = matrix[2];
    const double m10 = matrix[3];
    const double m11 = matrix[4];
    const double m12 = matrix[5];
    const double m20 = matrix[6];
    const double m21 = matrix[7];
    const double m22 = matrix[8];

    double c00 = 0.0;
    double c10 = 0.0;
    double c20 = 0.0;
    if (!cofactor(m11, m22, m12, m21, c00) ||
        !cofactor(m12, m20, m10, m22, c10) ||
        !cofactor(m10, m21, m20, m11, c20))
        return ResampleMatrixStatus::non_finite_result;

    volatile double initial_product = m01 * c10;
    double determinant_accumulator = initial_product;
    if (!std::isfinite(determinant_accumulator))
        return ResampleMatrixStatus::non_finite_result;
    determinant_accumulator =
        std::fma(m00, c00, determinant_accumulator);
    if (!std::isfinite(determinant_accumulator))
        return ResampleMatrixStatus::non_finite_result;
    const double determinant =
        std::fma(m02, c20, determinant_accumulator);
    if (!std::isfinite(determinant))
        return ResampleMatrixStatus::non_finite_result;

    const double absolute_determinant = std::fabs(determinant);
    if (!std::isfinite(absolute_determinant))
        return ResampleMatrixStatus::non_finite_result;
    if (absolute_determinant < 1e-10)
        return ResampleMatrixStatus::singular_matrix;

    double c22 = 0.0;
    double c21 = 0.0;
    double c12 = 0.0;
    double c11 = 0.0;
    double c02 = 0.0;
    double c01 = 0.0;
    if (!cofactor(m00, m11, m01, m10, c22) ||
        !cofactor(m20, m01, m00, m21, c21) ||
        !cofactor(m10, m02, m00, m12, c12) ||
        !cofactor(m00, m22, m02, m20, c11) ||
        !cofactor(m01, m12, m02, m11, c02) ||
        !cofactor(m02, m21, m01, m22, c01))
        return ResampleMatrixStatus::non_finite_result;

    const Matrix3 adjugate = {c00, c01, c02,
                              c10, c11, c12,
                              c20, c21, c22};
    Matrix3 candidate{};
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        candidate[i] = adjugate[i] / determinant;
        if (!std::isfinite(candidate[i]))
            return ResampleMatrixStatus::non_finite_result;
    }
    inverse = candidate;
    return ResampleMatrixStatus::ok;
}

ResampleMatrixStatus multiply_3x3(const Matrix3 &lhs, const Matrix3 &rhs,
                                  Matrix3 &product) noexcept {
    Matrix3 candidate{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double accumulator = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                accumulator = std::fma(lhs[row * 3 + k],
                                       rhs[k * 3 + column], accumulator);
                if (!std::isfinite(accumulator))
                    return ResampleMatrixStatus::non_finite_result;
            }
            candidate[row * 3 + column] = accumulator;
        }
    }
    product = candidate;
    return ResampleMatrixStatus::ok;
}

}  // namespace

ResampleMatrixStatus invert_prepared_resample_matrix(
    const ResampleMatrix3 &matrix, ResampleMatrix3 &inverse) noexcept {
    if (!finite_matrix(matrix)) return ResampleMatrixStatus::non_finite_input;
    return inverse_3x3(matrix, inverse);
}

ResampleMatrixStatus decompose_ordinary_resample_matrix(
    const ResampleMatrix3 &matrix, OrdinaryAxisTuples &tuples) noexcept {
    if (!finite_matrix(matrix))
        return ResampleMatrixStatus::non_finite_input;

    Matrix3 inverse_matrix{};
    const auto inverse_status = inverse_3x3(matrix, inverse_matrix);
    if (inverse_status != ResampleMatrixStatus::ok) return inverse_status;
    if (!finite_matrix(inverse_matrix))
        return ResampleMatrixStatus::non_finite_result;

    const Matrix3 horizontal = {1.0, 0.0, 0.0,
                                inverse_matrix[3], inverse_matrix[4],
                                inverse_matrix[5],
                                0.0, 0.0, 1.0};
    if (!finite_matrix(horizontal))
        return ResampleMatrixStatus::non_finite_result;

    Matrix3 inverse_horizontal{};
    const auto horizontal_inverse_status =
        inverse_3x3(horizontal, inverse_horizontal);
    if (horizontal_inverse_status != ResampleMatrixStatus::ok)
        return horizontal_inverse_status;
    if (!finite_matrix(inverse_horizontal))
        return ResampleMatrixStatus::non_finite_result;

    Matrix3 prepared{};
    const auto multiply_status =
        multiply_3x3(inverse_matrix, inverse_horizontal, prepared);
    if (multiply_status != ResampleMatrixStatus::ok) return multiply_status;
    if (!finite_matrix(prepared))
        return ResampleMatrixStatus::non_finite_result;

    const OrdinaryAxisTuples candidate = {
        prepared[2], prepared[0], prepared[1],
        horizontal[5], horizontal[4], horizontal[3]};
    tuples = candidate;
    return ResampleMatrixStatus::ok;
}

}  // namespace imprint::acr
