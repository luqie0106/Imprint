#include "imprint_renderer.h"
#include "acr_profile_hsl_controls.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
bool valid_span(const void* pointer, std::size_t bytes, std::size_t alignment) noexcept {
    if (pointer == nullptr) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return address % alignment == 0 && bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) noexcept {
    const auto left = reinterpret_cast<std::uintptr_t>(a);
    const auto right = reinterpret_cast<std::uintptr_t>(b);
    return left <= right ? right - left < a_bytes : left - right < b_bytes;
}
}  // namespace

im_status im_native_acr_profile_hsl_controls(
    const int32_t* current, size_t current_count,
    const int32_t* profile_defaults, size_t default_count,
    double profile_scalar, uint32_t monochrome_profile, uint32_t convert_to_gray,
    im_acr_profile_hsl_controls_result* output) {
    constexpr std::size_t bytes = 24 * sizeof(std::int32_t);
    if (current_count != 24 || default_count != 24 || monochrome_profile > 1 || convert_to_gray > 255 ||
        !valid_span(current, bytes, alignof(std::int32_t)) ||
        !valid_span(profile_defaults, bytes, alignof(std::int32_t)) ||
        !valid_span(output, sizeof(*output), alignof(im_acr_profile_hsl_controls_result)) ||
        overlaps(current, bytes, output, sizeof(*output)) ||
        overlaps(profile_defaults, bytes, output, sizeof(*output))) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    imprint::acr::ProfileHslControlsInput input{};
    std::copy_n(current, 24, input.current.begin());
    std::copy_n(profile_defaults, 24, input.profile_defaults.begin());
    input.profile_scalar = profile_scalar;
    input.monochrome_profile = monochrome_profile != 0;
    input.convert_to_gray = static_cast<std::uint8_t>(convert_to_gray);
    imprint::acr::ProfileHslControlsResult result{};
    if (imprint::acr::process_profile_hsl_controls(input, result) != imprint::acr::ProfileHslControlsStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;
    im_acr_profile_hsl_controls_result candidate{};
    std::copy(result.values.begin(), result.values.end(), candidate.values);
    std::copy(result.blended.begin(), result.blended.end(), candidate.blended);
    candidate.wrote_mask = result.wrote_mask;
    *output = candidate;
    return IM_STATUS_OK;
}
