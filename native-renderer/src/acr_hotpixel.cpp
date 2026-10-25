#include "acr_hotpixel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

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

bool overlaps(const AddressRange& left, const AddressRange& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
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

std::uint16_t median9(std::uint16_t values[9]) noexcept {
    std::sort(values, values + 9);
    return values[4];
}

std::uint16_t upper_middle_average6(std::uint16_t values[6]) noexcept {
    std::sort(values, values + 6);
    const std::uint32_t sum = static_cast<std::uint32_t>(values[2]) + values[3];
    return static_cast<std::uint16_t>((sum + 1u) >> 1u);
}

std::int32_t floor_divide_by_four(std::int32_t value) noexcept {
    std::int32_t quotient = value / 4;
    if (value < 0 && value % 4 != 0) --quotient;
    return quotient;
}

}  // namespace

HotPixelStatus hotpixel_u16_row_0132d7c8(
    const std::uint16_t* rows[5], const std::size_t row_counts[5],
    std::size_t left_guard, std::uint16_t* dst, std::size_t dst_count,
    std::size_t width, std::uint32_t phase,
    std::uint16_t hot_threshold, std::uint16_t cold_threshold) noexcept {
    if (rows == nullptr || row_counts == nullptr || dst == nullptr)
        return HotPixelStatus::invalid_argument;
    if (width == 0 || phase > 1 || left_guard < 2 || dst_count < width)
        return HotPixelStatus::invalid_parameters;
    if (width > std::numeric_limits<std::size_t>::max() - 2 ||
        left_guard > std::numeric_limits<std::size_t>::max() - width - 2) {
        return HotPixelStatus::invalid_parameters;
    }
    const std::size_t required_row_count = left_guard + width + 2;

    AddressRange rows_array_range{};
    AddressRange counts_array_range{};
    AddressRange dst_capacity_range{};
    AddressRange dst_write_range{};
    RangeResult range_result = make_range(
        rows, 5, sizeof(rows[0]), alignof(const std::uint16_t*),
        &rows_array_range);
    if (range_result != RangeResult::ok) return map_range_result(range_result);
    range_result = make_range(row_counts, 5, sizeof(row_counts[0]),
                              alignof(std::size_t), &counts_array_range);
    if (range_result != RangeResult::ok) return map_range_result(range_result);
    range_result = make_range(dst, dst_count, sizeof(*dst), alignof(std::uint16_t),
                              &dst_capacity_range);
    if (range_result != RangeResult::ok) return map_range_result(range_result);
    range_result = make_range(dst, width, sizeof(*dst), alignof(std::uint16_t),
                              &dst_write_range);
    if (range_result != RangeResult::ok) return map_range_result(range_result);
    (void)dst_capacity_range;

    const std::uint16_t* input_rows[5];
    std::size_t input_counts[5];
    AddressRange input_ranges[5]{};
    for (std::size_t row = 0; row < 5; ++row) {
        input_rows[row] = rows[row];
        input_counts[row] = row_counts[row];
        if (input_rows[row] == nullptr)
            return HotPixelStatus::invalid_argument;
        if (input_counts[row] < required_row_count)
            return HotPixelStatus::invalid_parameters;
        range_result = make_range(input_rows[row], input_counts[row],
                                  sizeof(*input_rows[row]),
                                  alignof(std::uint16_t), &input_ranges[row]);
        if (range_result != RangeResult::ok) return map_range_result(range_result);
    }

    if (overlaps(dst_write_range, rows_array_range) ||
        overlaps(dst_write_range, counts_array_range)) {
        return HotPixelStatus::overlap;
    }
    for (const AddressRange& input_range : input_ranges) {
        if (overlaps(dst_write_range, input_range)) return HotPixelStatus::overlap;
    }

    const std::uint16_t* const a = input_rows[0];
    const std::uint16_t* const b = input_rows[1];
    const std::uint16_t* const c = input_rows[2];
    const std::uint16_t* const d = input_rows[3];
    const std::uint16_t* const e = input_rows[4];

    for (std::size_t x = 0; x < width; ++x) {
        const std::size_t center_index = left_guard + x;
        const std::uint16_t center = c[center_index];
        const std::uint32_t pixel_phase = phase ^ static_cast<std::uint32_t>(x & 1u);
        std::uint16_t neighborhood[9];
        std::uint16_t median_samples[9];
        std::uint16_t vertical6[6];
        std::uint16_t horizontal6[6];
        std::uint16_t replace4[4];

        if (pixel_phase == 0) {
            neighborhood[0] = a[center_index - 2];
            neighborhood[1] = a[center_index];
            neighborhood[2] = a[center_index + 2];
            neighborhood[3] = c[center_index - 2];
            neighborhood[4] = center;
            neighborhood[5] = c[center_index + 2];
            neighborhood[6] = e[center_index - 2];
            neighborhood[7] = e[center_index];
            neighborhood[8] = e[center_index + 2];
            replace4[0] = a[center_index];
            replace4[1] = e[center_index];
            replace4[2] = c[center_index - 2];
            replace4[3] = c[center_index + 2];
        } else {
            neighborhood[0] = a[center_index];
            neighborhood[1] = b[center_index - 1];
            neighborhood[2] = b[center_index + 1];
            neighborhood[3] = c[center_index - 2];
            neighborhood[4] = center;
            neighborhood[5] = c[center_index + 2];
            neighborhood[6] = d[center_index - 1];
            neighborhood[7] = d[center_index + 1];
            neighborhood[8] = e[center_index];
            replace4[0] = b[center_index - 1];
            replace4[1] = b[center_index + 1];
            replace4[2] = d[center_index - 1];
            replace4[3] = d[center_index + 1];
        }

        vertical6[0] = b[center_index - 2];
        vertical6[1] = b[center_index];
        vertical6[2] = b[center_index + 2];
        vertical6[3] = d[center_index - 2];
        vertical6[4] = d[center_index];
        vertical6[5] = d[center_index + 2];
        horizontal6[0] = a[center_index - 1];
        horizontal6[1] = c[center_index - 1];
        horizontal6[2] = e[center_index - 1];
        horizontal6[3] = a[center_index + 1];
        horizontal6[4] = c[center_index + 1];
        horizontal6[5] = e[center_index + 1];

        for (std::size_t i = 0; i < 9; ++i) median_samples[i] = neighborhood[i];
        const std::uint16_t median = median9(median_samples);
        const std::uint16_t vertical_average = upper_middle_average6(vertical6);
        const std::uint16_t horizontal_average = upper_middle_average6(horizontal6);
        const std::int32_t residual =
            static_cast<std::int32_t>(b[center_index]) +
            static_cast<std::int32_t>(d[center_index]) +
            static_cast<std::int32_t>(c[center_index - 1]) +
            static_cast<std::int32_t>(c[center_index + 1]) -
            2 * (static_cast<std::int32_t>(vertical_average) +
                 static_cast<std::int32_t>(horizontal_average));
        const std::int32_t threshold = floor_divide_by_four(
            (static_cast<std::int32_t>(center) -
             static_cast<std::int32_t>(median)) * 3 + 2);

        bool hot_gate = false;
        if (center > hot_threshold) {
            const std::uint32_t upper_limit =
                static_cast<std::uint32_t>(center) - hot_threshold;
            hot_gate = residual < threshold;
            for (std::size_t i = 0; i < 9 && hot_gate; ++i) {
                if (i != 4 &&
                    static_cast<std::uint32_t>(neighborhood[i]) >= upper_limit) {
                    hot_gate = false;
                }
            }
        }

        bool cold_gate = false;
        const std::uint32_t lower_limit =
            static_cast<std::uint32_t>(center) + cold_threshold;
        cold_gate = residual > threshold;
        for (std::size_t i = 0; i < 9 && cold_gate; ++i) {
            if (i != 4 &&
                static_cast<std::uint32_t>(neighborhood[i]) <= lower_limit) {
                cold_gate = false;
            }
        }

        if (hot_gate) {
            dst[x] = *std::max_element(replace4, replace4 + 4);
        } else if (cold_gate) {
            dst[x] = *std::min_element(replace4, replace4 + 4);
        } else {
            dst[x] = center;
        }
    }

    return HotPixelStatus::ok;
}

}  // namespace imprint::acr
