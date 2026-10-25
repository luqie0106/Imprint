#include "../src/acr_render_geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

template <typename T>
void put_bits(T &target, std::uint64_t bits) noexcept {
    static_assert(sizeof(T) == sizeof(bits));
    std::memcpy(&target, &bits, sizeof(bits));
}

void initialize_sentinel(imprint::acr::RenderGeometryResult &result) noexcept {
    result = {};
    result.level = 0xdeadbeefU;
    result.image = reinterpret_cast<const void *>(
        static_cast<std::uintptr_t>(0x123456789abcdef0ULL));
    result.reference_bounds = {-101, 202, -303, 404};
    result.actual_bounds = {-505, 606, -707, 808};
    result.output_bounds = {-909, 1010, -1111, 1212};

    constexpr std::array<std::uint64_t, 9> matrix_bits = {
        0x3ff0000000000001ULL, 0x8000000000000000ULL,
        0x7ff8000000000042ULL, 0x0000000000000000ULL,
        0xffffffffffffffffULL, 0x0123456789abcdefULL,
        0xfff0000000000000ULL, 0x400921fb54442d18ULL,
        0x3fd5555555555555ULL};
    for (std::size_t index = 0; index < matrix_bits.size(); ++index) {
        put_bits(result.matrix_e0[index], matrix_bits[index]);
        put_bits(result.matrix_170[index], matrix_bits[(index + 3) % 9]);
        put_bits(result.matrix_200[index], matrix_bits[(index + 6) % 9]);
    }

    result.context_after.field_00 = 0x89abcdefU;
    result.context_after.field_7c = 0x76543210U;
    result.context_after.field_8c = 0x13579bdfU;
    result.context_after.field_98 = 0x2468ace0U;
    result.context_after.field_911 = 0xfedcba98U;
    result.context_after.field_846 = 1;
    result.context_after.field_847 = 2;
    result.context_after.field_820 = 3;
    result.context_after.field_84c = -0x1234567;
    result.context_after.field_8a8 = {
        0x0123456789abcdefULL, 0xfedcba9876543210ULL,
        0x7ff8000000000021ULL, 0x8000000000000000ULL,
        0x3ff0000000000000ULL, 0x4000000000000000ULL,
        0x3fe0000000000000ULL};
    result.flag_0c = true;
    result.mask_290 = true;
    result.iterations = 0xcafebabeU;
    put_bits(result.mask_first_value, 0x7ff8000000000043ULL);
}

}  // namespace

extern "C" int acr_test_prepare_render_geometry(
    const imprint::acr::RenderGeometryInput *input,
    std::uint8_t *output_bytes,
    std::size_t output_size) noexcept {
    if (input == nullptr || output_bytes == nullptr
        || output_size != sizeof(imprint::acr::RenderGeometryResult)) {
        return 98;
    }

    std::array<unsigned char, sizeof(imprint::acr::RenderGeometryInput)>
        input_before{};
    std::memcpy(input_before.data(), input, input_before.size());

    imprint::acr::RenderGeometryResult result{};
    initialize_sentinel(result);
    std::array<unsigned char, sizeof(result)> result_before{};
    std::memcpy(result_before.data(), &result, result_before.size());

    const auto status = imprint::acr::prepare_render_geometry(*input, result);

    if (std::memcmp(input_before.data(), input, input_before.size()) != 0) {
        return 99;
    }
    if (status != imprint::acr::RenderGeometryStatus::ok) {
        if (std::memcmp(result_before.data(), &result, result_before.size()) != 0) {
            return 99;
        }
        return static_cast<int>(status);
    }

    std::memcpy(output_bytes, &result, sizeof(result));
    return static_cast<int>(status);
}

