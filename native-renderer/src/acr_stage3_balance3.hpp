#pragma once

#include <cstddef>

namespace imprint::acr {

enum class Balance3KernelStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    unsupported_alignment = 4,
    invalid_parameters = 5,
    arithmetic_overflow = 6,
    overlapping_buffers = 7,
};

// These APIs cover only three recovered balance3 consumers for one prepared
// binary32 row. `center` addresses column zero of the center row; stride is in
// float samples. The caller provides readable guard storage around the center
// row as required by each kernel. Only `width` destination samples are
// written. Rejected calls leave destination unchanged. Source/destination
// overlap is rejected using the conservative source span declared by each
// kernel; diagonal coefficient/destination overlap is also rejected.
// Arithmetic parity assumes round-to-nearest-even and finite inputs.
// Diagonal coefficient arrays need natural float alignment but not 16-byte
// alignment. This is not a full Stage3 or original ABI implementation.
Balance3KernelStatus balance3_difference_prepared_row_00813870(
    const float *center, std::size_t stride, std::size_t width, int reverse,
    float *dst) noexcept;

Balance3KernelStatus balance3_diagonal_filter_prepared_row_008139cc(
    const float *center, std::size_t stride, std::size_t width,
    const float *coeff, std::size_t radius, float *dst) noexcept;

Balance3KernelStatus balance3_diagonal_subtract_prepared_row_00813cf8(
    const float *center, std::size_t stride, std::size_t width,
    const float *coeff, std::size_t radius, float *dst) noexcept;

}  // namespace imprint::acr
