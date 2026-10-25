#include "acr_stage3_core3.hpp"

#include <cmath>
#include <cstdint>

namespace imprint::acr {
namespace {

bool aligned_16(const void *pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0xF) == 0;
}

bool valid_width(std::size_t width) noexcept {
    // The recovered body processes whole 4-float groups.
    return width != 0 && (width & 3) == 0;
}

bool valid_source_shape(std::size_t width, std::size_t stride) noexcept {
    // A stride that is a multiple of four keeps derived rows
    // (center +/- stride, center +/- 2*stride) 16-byte aligned.
    return valid_width(width) && stride >= width && (stride & 3) == 0;
}

bool valid_output_shape(std::size_t width, std::size_t stride) noexcept {
    // Output rows are independently aligned; only capacity matters here.
    return valid_width(width) && stride >= width;
}

float five_tap_energy(float minus2, float minus1, float center, float plus1,
                      float plus2) noexcept {
    // FUN_00813368: the mean adds the five samples left to right and scales by
    // 0.2f; the squared deviations are summed in +2, +1, 0, -2, -1 order.
    const float mean = (minus2 + minus1 + center + plus1 + plus2) * 0.2f;
    const float d_plus2 = plus2 - mean;
    const float d_plus1 = plus1 - mean;
    const float d_center = center - mean;
    const float d_minus2 = minus2 - mean;
    const float d_minus1 = minus1 - mean;
    return std::sqrt(d_plus2 * d_plus2 + d_plus1 * d_plus1 +
                     d_center * d_center + d_minus2 * d_minus2 +
                     d_minus1 * d_minus1);
}

}  // namespace

Core3Status core3_gradient_rows_0081376c(
    const float *center,
    std::size_t source_stride,
    std::size_t width,
    float *const out[6],
    std::size_t output_stride) noexcept {
    if (center == nullptr || out == nullptr) {
        return Core3Status::invalid_argument;
    }
    for (int index = 0; index < 6; ++index) {
        if (out[index] == nullptr) return Core3Status::invalid_argument;
    }
    if (!valid_source_shape(width, source_stride) ||
        !valid_output_shape(width, output_stride)) {
        return Core3Status::invalid_dimensions;
    }
    if (!aligned_16(center)) {
        return Core3Status::unsupported_alignment;
    }
    for (int index = 0; index < 6; ++index) {
        if (!aligned_16(out[index])) {
            return Core3Status::unsupported_alignment;
        }
    }
    // The recovered kernel reads rows center-2*stride .. center+2*stride and
    // columns j-2 .. j+2. The caller owns the guard samples.
    const float *minus2 = center - 2 * source_stride;
    const float *minus1 = center - source_stride;
    const float *plus1 = center + source_stride;
    const float *plus2 = center + 2 * source_stride;
    for (std::size_t column = 0; column < width; ++column) {
        const std::ptrdiff_t j = static_cast<std::ptrdiff_t>(column);
        const float c = center[j];
        const float c_minus2 = center[j - 2];
        const float c_minus1 = center[j - 1];
        const float c_plus1 = center[j + 1];
        const float c_plus2 = center[j + 2];
        const float twice_c = c + c;

        // 0081376c, aligned 4-float group body:
        //   q0 = C[j..j+3], q2 = C[j-2..j+1], q3 = C[j+2..j+5]
        //   q1 = Vminus2[j..j+3], q4 = Vplus2[j..j+3]
        //   out0 = |C - C[j+2]|
        //   out1 = |C - C[j-2]| + |C - C[j+2]|
        //   out2 = |C - Vplus2|
        //   out3 = |C - Vminus2| + |C - Vplus2|
        out[0][column] = std::fabs(c - c_plus2);
        out[1][column] = std::fabs(c - c_minus2) + std::fabs(c - c_plus2);
        out[2][column] = std::fabs(c - plus2[j]);
        out[3][column] =
            std::fabs(c - minus2[j]) + std::fabs(c - plus2[j]);
        out[4][column] = std::fabs((c_minus1 + c_plus1) - twice_c);
        out[5][column] = std::fabs((minus1[j] + plus1[j]) - twice_c);
    }
    return Core3Status::ok;
}

Core3Status core3_local_energy_rows_00813368(
    const float *horizontal_row,
    const float *vertical_center,
    std::size_t row_stride,
    std::size_t width,
    float *out_horizontal,
    float *out_vertical) noexcept {
    if (horizontal_row == nullptr || vertical_center == nullptr ||
        out_horizontal == nullptr || out_vertical == nullptr) {
        return Core3Status::invalid_argument;
    }
    if (!valid_source_shape(width, row_stride)) {
        return Core3Status::invalid_dimensions;
    }
    if (!aligned_16(horizontal_row) || !aligned_16(vertical_center) ||
        !aligned_16(out_horizontal) || !aligned_16(out_vertical)) {
        return Core3Status::unsupported_alignment;
    }
    const float *vertical_minus2 = vertical_center - 2 * row_stride;
    const float *vertical_minus1 = vertical_center - row_stride;
    const float *vertical_plus1 = vertical_center + row_stride;
    const float *vertical_plus2 = vertical_center + 2 * row_stride;
    for (std::size_t column = 0; column < width; ++column) {
        const std::ptrdiff_t j = static_cast<std::ptrdiff_t>(column);
        out_horizontal[column] =
            five_tap_energy(horizontal_row[j - 2], horizontal_row[j - 1],
                            horizontal_row[j], horizontal_row[j + 1],
                            horizontal_row[j + 2]);
        out_vertical[column] = five_tap_energy(
            vertical_minus2[column], vertical_minus1[column],
            vertical_center[column], vertical_plus1[column],
            vertical_plus2[column]);
    }
    return Core3Status::ok;
}

}  // namespace imprint::acr
