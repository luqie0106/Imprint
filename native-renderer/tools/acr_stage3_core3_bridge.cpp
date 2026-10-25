#include "../src/acr_stage3_core3.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int acr_test_core3_gradient(
    const float *center,
    std::uint64_t source_stride,
    std::uint64_t width,
    float *out0,
    float *out1,
    float *out2,
    float *out3,
    float *out4,
    float *out5,
    std::uint64_t output_stride) noexcept {
    float *rows[6] = {out0, out1, out2, out3, out4, out5};
    return static_cast<int>(imprint::acr::core3_gradient_rows_0081376c(
        center, static_cast<std::size_t>(source_stride),
        static_cast<std::size_t>(width), rows,
        static_cast<std::size_t>(output_stride)));
}

extern "C" int acr_test_core3_energy(
    const float *horizontal_row,
    const float *vertical_center,
    std::uint64_t row_stride,
    std::uint64_t width,
    float *out_horizontal,
    float *out_vertical) noexcept {
    return static_cast<int>(imprint::acr::core3_local_energy_rows_00813368(
        horizontal_row, vertical_center,
        static_cast<std::size_t>(row_stride), static_cast<std::size_t>(width),
        out_horizontal, out_vertical));
}
