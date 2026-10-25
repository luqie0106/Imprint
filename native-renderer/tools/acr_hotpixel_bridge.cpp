#include "acr_hotpixel.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

using imprint::acr::HotPixelStatus;

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

enum class RangeResult { ok, invalid_pointer, invalid_size, misaligned };

RangeResult make_range(const void* pointer, std::size_t count,
                       std::size_t element_size, std::size_t alignment,
                       AddressRange* out) noexcept {
    if (pointer == nullptr || out == nullptr) return RangeResult::invalid_pointer;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) return RangeResult::misaligned;
    if (element_size != 0 &&
        count > std::numeric_limits<std::size_t>::max() / element_size) {
        return RangeResult::invalid_size;
    }
    const std::size_t bytes = count * element_size;
    if (static_cast<std::uintmax_t>(bytes) >
        static_cast<std::uintmax_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return RangeResult::invalid_size;
    }
    if (static_cast<std::uintmax_t>(bytes) >
        static_cast<std::uintmax_t>(std::numeric_limits<std::uintptr_t>::max())) {
        return RangeResult::invalid_size;
    }
    const auto byte_count = static_cast<std::uintptr_t>(bytes);
    if (byte_count > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return RangeResult::invalid_size;
    }
    out->begin = begin;
    out->end = begin + byte_count;
    return RangeResult::ok;
}

bool overlaps(const AddressRange& a, const AddressRange& b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}

HotPixelStatus map_range_result(RangeResult result) noexcept {
    switch (result) {
        case RangeResult::ok: return HotPixelStatus::ok;
        case RangeResult::invalid_pointer:
        case RangeResult::misaligned: return HotPixelStatus::invalid_argument;
        case RangeResult::invalid_size: return HotPixelStatus::invalid_parameters;
    }
    return HotPixelStatus::invalid_parameters;
}

}  // namespace

extern "C" int acr_test_hotpixel(
    const std::uint16_t* src, std::size_t count, std::size_t stride,
    std::size_t width, std::uint32_t phase, std::uint16_t hot_threshold,
    std::uint16_t cold_threshold, std::uint16_t* dst,
    std::size_t dst_count) {
    using namespace imprint::acr;
    if (src == nullptr || dst == nullptr)
        return static_cast<int>(HotPixelStatus::invalid_argument);
    if (width == 0 || phase > 1 || width >
            std::numeric_limits<std::size_t>::max() - 4 ||
        stride < width + 4 || stride >
            (std::numeric_limits<std::size_t>::max() - width - 4) / 4) {
        return static_cast<int>(HotPixelStatus::invalid_parameters);
    }

    const std::size_t required_count = 4 * stride + width + 4;
    if (count < required_count || dst_count < width)
        return static_cast<int>(HotPixelStatus::invalid_parameters);

    // Validate full declared capacities before forming src + row * stride.
    AddressRange source_range{};
    AddressRange dst_capacity_range{};
    AddressRange dst_write_range{};
    RangeResult range_result = make_range(src, count, sizeof(*src),
                                          alignof(std::uint16_t),
                                          &source_range);
    if (range_result != RangeResult::ok)
        return static_cast<int>(map_range_result(range_result));
    range_result = make_range(dst, dst_count, sizeof(*dst),
                              alignof(std::uint16_t),
                              &dst_capacity_range);
    if (range_result != RangeResult::ok)
        return static_cast<int>(map_range_result(range_result));
    (void)dst_capacity_range;
    range_result = make_range(dst, width, sizeof(*dst),
                              alignof(std::uint16_t),
                              &dst_write_range);
    if (range_result != RangeResult::ok)
        return static_cast<int>(map_range_result(range_result));
    if (overlaps(source_range, dst_write_range))
        return static_cast<int>(HotPixelStatus::overlap);

    const std::uint16_t* rows[5];
    std::size_t row_counts[5];
    for (std::size_t row = 0; row < 5; ++row) {
        const std::size_t offset = row * stride;
        rows[row] = src + offset;
        row_counts[row] = count - offset;
    }
    return static_cast<int>(hotpixel_u16_row_0132d7c8(
        rows, row_counts, 2, dst, dst_count, width, phase,
        hot_threshold, cold_threshold));
}
