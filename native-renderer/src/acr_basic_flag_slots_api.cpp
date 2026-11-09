#include "imprint_renderer.h"
#include "acr_basic_flag_slots.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

struct BufferSpan {
    const void* pointer;
    std::size_t bytes;
    std::size_t alignment;
};

bool valid_span(const BufferSpan& span) noexcept {
    if (span.pointer == nullptr) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(span.pointer);
    return address % span.alignment == 0 &&
           span.bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool overlaps(const BufferSpan& a, const BufferSpan& b) noexcept {
    const auto left = reinterpret_cast<std::uintptr_t>(a.pointer);
    const auto right = reinterpret_cast<std::uintptr_t>(b.pointer);
    return left <= right ? right - left < a.bytes : left - right < b.bytes;
}

}  // namespace

im_status im_native_acr_basic_flag_slots(
    const int32_t* current, size_t current_count,
    const uint8_t* header, size_t header_count,
    int32_t* destination, size_t destination_count,
    uint32_t* reset_mask, size_t mask_count) {
    if (current_count != 132 || header_count != 16 ||
        destination_count != 132 || mask_count != 5) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    const std::array<BufferSpan, 4> spans{{
        {current, 132 * sizeof(std::int32_t), alignof(std::int32_t)},
        {header, 16 * sizeof(std::uint8_t), alignof(std::uint8_t)},
        {destination, 132 * sizeof(std::int32_t), alignof(std::int32_t)},
        {reset_mask, 5 * sizeof(std::uint32_t), alignof(std::uint32_t)},
    }};
    // Validate every span and all six pairs before reading either input.
    for (const auto& span : spans) {
        if (!valid_span(span)) return IM_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t i = 0; i < spans.size(); ++i) {
        for (std::size_t j = i + 1; j < spans.size(); ++j) {
            if (overlaps(spans[i], spans[j])) return IM_STATUS_INVALID_ARGUMENT;
        }
    }

    imprint::acr::BasicFlagSlotsInput input{};
    std::copy_n(current, input.current.size(), input.current.begin());
    std::copy_n(header, input.header.size(), input.header.begin());
    imprint::acr::BasicFlagSlotsResult candidate{};
    imprint::acr::normalize_basic_flag_slots(input, candidate);
    // No fallible operations remain: commit both caller-owned outputs.
    std::copy(candidate.values.begin(), candidate.values.end(), destination);
    std::copy(candidate.reset_mask.begin(), candidate.reset_mask.end(), reset_mask);
    return IM_STATUS_OK;
}
