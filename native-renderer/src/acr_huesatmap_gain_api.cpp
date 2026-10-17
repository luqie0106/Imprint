#include "acr_huesatmap_prepare.hpp"
#include "imprint_renderer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr std::size_t kControlCount = 3;
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

extern "C" IMPRINT_API im_status im_native_acr_prepare_huesatmap_gain(
    std::uint32_t process_version, const double *controls,
    std::size_t count, double *output, std::size_t output_count) {
    if (controls == nullptr || output == nullptr || count != kControlCount ||
        output_count != kOutputCount) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    constexpr std::size_t control_bytes = kControlCount * sizeof(double);
    constexpr std::size_t output_bytes = kOutputCount * sizeof(double);
    if (!valid_range(controls, control_bytes, alignof(double)) ||
        !valid_range(output, output_bytes, alignof(double)) ||
        overlaps(controls, control_bytes, output, output_bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    std::array<double, kControlCount> checked_controls{};
    std::memcpy(checked_controls.data(), controls, control_bytes);

    std::array<double, kOutputCount> prepared{};
    const auto status = imprint::acr::prepare_huesatmap_gain(
        process_version, checked_controls[0], checked_controls[1],
        checked_controls[2], prepared);
    if (status != imprint::acr::ProfileTableStatus::ok) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    std::memcpy(output, prepared.data(), output_bytes);
    return IM_STATUS_OK;
}
