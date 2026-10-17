#include "acr_illuminant.hpp"
#include "imprint_renderer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr std::size_t kCodeCount = 2;
constexpr std::size_t kOutputCount = 2;

bool valid_range(const void *pointer, std::size_t bytes,
                 std::size_t alignment) noexcept {
    if (pointer == nullptr || alignment == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return begin % alignment == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - begin;
}

bool overlaps(const void *left, std::size_t left_bytes,
              const void *right, std::size_t right_bytes) noexcept {
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    return left_begin <= right_begin
               ? right_begin - left_begin < left_bytes
               : left_begin - right_begin < right_bytes;
}

}  // namespace

extern "C" IMPRINT_API im_status im_native_acr_illuminant_temperatures(
    const std::uint32_t *codes, std::size_t count, double *output,
    std::size_t output_count) {
    if (codes == nullptr || output == nullptr || count != kCodeCount ||
        output_count != kOutputCount) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    constexpr std::size_t code_bytes = kCodeCount * sizeof(std::uint32_t);
    constexpr std::size_t output_bytes = kOutputCount * sizeof(double);
    if (!valid_range(codes, code_bytes, alignof(std::uint32_t)) ||
        !valid_range(output, output_bytes, alignof(double)) ||
        overlaps(codes, code_bytes, output, output_bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    std::array<std::uint32_t, kCodeCount> checked_codes{};
    std::memcpy(checked_codes.data(), codes, code_bytes);
    constexpr std::uint32_t kMaximumCode = 65535;
    if (checked_codes[0] > kMaximumCode || checked_codes[1] > kMaximumCode ||
        checked_codes[0] == 255 || checked_codes[1] == 255) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    const std::array<double, kOutputCount> result{
        imprint::acr::illuminant_temperature_kelvin(checked_codes[0]),
        imprint::acr::illuminant_temperature_kelvin(checked_codes[1]),
    };
    std::memcpy(output, result.data(), output_bytes);
    return IM_STATUS_OK;
}
