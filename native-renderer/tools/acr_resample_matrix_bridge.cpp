#include "../src/acr_resample_matrix.hpp"

#include <array>
#include <cstdint>
#include <cstring>

static_assert(sizeof(imprint::acr::ResampleMatrix3) == 9 * sizeof(double));
static_assert(sizeof(imprint::acr::OrdinaryAxisTuples) == 6 * sizeof(double));

extern "C" int acr_test_decompose_ordinary_resample_matrix(
    const double *input, std::uint64_t *output_bits) noexcept {
    if (input == nullptr || output_bits == nullptr) return 98;

    imprint::acr::ResampleMatrix3 matrix{};
    std::memcpy(matrix.data(), input, sizeof(matrix));

    constexpr std::array<std::uint64_t, 6> sentinel_bits = {
        0x3ff0000000000001ULL, 0x8000000000000000ULL,
        0x7ff8000000000042ULL, 0x0000000000000000ULL,
        0xffffffffffffffffULL, 0x0123456789abcdefULL};
    imprint::acr::OrdinaryAxisTuples tuples{};
    std::memcpy(&tuples, sentinel_bits.data(), sizeof(tuples));
    std::array<unsigned char, sizeof(tuples)> before{};
    std::memcpy(before.data(), &tuples, sizeof(tuples));

    const auto status = imprint::acr::decompose_ordinary_resample_matrix(
        matrix, tuples);
    if (status != imprint::acr::ResampleMatrixStatus::ok) {
        if (std::memcmp(before.data(), &tuples, sizeof(tuples)) != 0) return 99;
        return static_cast<int>(status);
    }

    std::memcpy(output_bits, &tuples, sizeof(tuples));
    return static_cast<int>(status);
}
