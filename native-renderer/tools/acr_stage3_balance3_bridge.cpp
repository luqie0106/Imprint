// Validation seam only; all arithmetic remains in the balance3 row kernels.
#include "../src/acr_stage3_balance3.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int acr_test_balance3_difference(
    const float *center, std::uint64_t stride, std::uint64_t width,
    int reverse, float *destination) noexcept {
    return static_cast<int>(imprint::acr::balance3_difference_prepared_row_00813870(
        center, static_cast<std::size_t>(stride), static_cast<std::size_t>(width),
        reverse, destination));
}

extern "C" int acr_test_balance3_filter(
    const float *center, std::uint64_t stride, std::uint64_t width,
    const float *coeff, std::uint64_t radius, float *destination) noexcept {
    return static_cast<int>(imprint::acr::balance3_diagonal_filter_prepared_row_008139cc(
        center, static_cast<std::size_t>(stride), static_cast<std::size_t>(width),
        coeff, static_cast<std::size_t>(radius), destination));
}

extern "C" int acr_test_balance3_subtract(
    const float *center, std::uint64_t stride, std::uint64_t width,
    const float *coeff, std::uint64_t radius, float *destination) noexcept {
    return static_cast<int>(imprint::acr::balance3_diagonal_subtract_prepared_row_00813cf8(
        center, static_cast<std::size_t>(stride), static_cast<std::size_t>(width),
        coeff, static_cast<std::size_t>(radius), destination));
}
