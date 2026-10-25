#include "acr_post_tone_basic.hpp"
#include "imprint_renderer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr std::size_t kSlotCount = imprint::acr::kPostToneBasicSlotCount;
constexpr std::size_t kSlotBytes = kSlotCount * sizeof(std::int32_t);

bool valid_span(const void* pointer, std::size_t bytes,
                std::size_t alignment) noexcept {
    if (pointer == nullptr) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    return start % alignment == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - start;
}

bool overlaps(const void* left, std::size_t left_bytes,
              const void* right, std::size_t right_bytes) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_bytes;
    return left_start - right_start < right_bytes;
}

bool alias_is_supported(const void* output, std::size_t output_count,
                        const void* input, std::size_t input_bytes,
                        std::size_t output_bytes) noexcept {
    if (!overlaps(output, output_bytes, input, input_bytes)) return true;
    return output_count == kSlotCount && output == input;
}

}  // namespace

extern "C" IMPRINT_API im_status im_native_acr_normalize_post_tone_basic_i32(
    std::uint32_t process_version, const std::int32_t* input,
    std::size_t count, const std::int32_t* source_defaults,
    std::size_t source_count, std::int32_t* output,
    std::size_t output_count) {
    if (input == nullptr || source_defaults == nullptr || output == nullptr ||
        count != kSlotCount || source_count != kSlotCount) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    if (output_count < kSlotCount) return IM_STATUS_BUFFER_TOO_SMALL;

    if (output_count > std::numeric_limits<std::size_t>::max() /
                           sizeof(std::int32_t) ||
        output_count > static_cast<std::size_t>(
                           std::numeric_limits<std::ptrdiff_t>::max()) /
                           sizeof(std::int32_t)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    const std::size_t output_bytes = output_count * sizeof(std::int32_t);

    if (!valid_span(input, kSlotBytes, alignof(std::int32_t)) ||
        !valid_span(source_defaults, kSlotBytes, alignof(std::int32_t)) ||
        !valid_span(output, output_bytes, alignof(std::int32_t))) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    if (!alias_is_supported(output, output_count, input, kSlotBytes,
                            output_bytes) ||
        !alias_is_supported(output, output_count, source_defaults, kSlotBytes,
                            output_bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    imprint::acr::PostToneBasicSlots input_copy{};
    imprint::acr::PostToneBasicSlots defaults_copy{};
    std::memcpy(input_copy.data(), input, kSlotBytes);
    std::memcpy(defaults_copy.data(), source_defaults, kSlotBytes);

    imprint::acr::PostToneBasicSlots candidate{};
    imprint::acr::normalize_post_tone_basic(
        process_version, input_copy, defaults_copy, candidate);
    std::memcpy(output, candidate.data(), kSlotBytes);
    return IM_STATUS_OK;
}
