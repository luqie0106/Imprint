#include "acr_u16_lut.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

using imprint::acr::FloatLutPrepared;
using imprint::acr::FloatLutStatus;
using imprint::acr::U16LutPrepared;

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

bool address_range(const void* pointer, std::size_t size,
                   std::size_t alignment, AddressRange* out) noexcept {
    if (pointer == nullptr || out == nullptr || alignment == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) return false;
    constexpr auto kMax = std::numeric_limits<std::uintptr_t>::max();
    if (size > static_cast<std::size_t>(kMax - begin)) return false;
    out->begin = begin;
    out->end = begin + static_cast<std::uintptr_t>(size);
    return true;
}

bool overlaps(const AddressRange& left, const AddressRange& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

}  // namespace

extern "C" {

int acr_test_u16_lut_build(const float* table, std::size_t table_count,
                           std::uint16_t* out, std::size_t capacity) {
    constexpr std::size_t kInputBytes =
        imprint::acr::kFloatLutEntryCount * sizeof(float);
    constexpr std::size_t kOutputBytes =
        imprint::acr::kU16LutEntryCount * sizeof(std::uint16_t);
    if (table_count < imprint::acr::kFloatLutEntryCount ||
        capacity < imprint::acr::kU16LutEntryCount) {
        return static_cast<int>(FloatLutStatus::invalid_argument);
    }

    AddressRange table_range{};
    AddressRange output_range{};
    if (!address_range(table, kInputBytes, alignof(float), &table_range) ||
        !address_range(out, kOutputBytes, alignof(std::uint16_t),
                       &output_range)) {
        return static_cast<int>(FloatLutStatus::invalid_argument);
    }
    if (overlaps(table_range, output_range)) {
        return static_cast<int>(FloatLutStatus::overlap);
    }

    FloatLutPrepared input{};
    std::memcpy(input.values.data(), table, kInputBytes);
    U16LutPrepared prepared{};
    const FloatLutStatus status =
        imprint::acr::build_u16_lut_01300600(input, &prepared);
    if (status != FloatLutStatus::ok) return static_cast<int>(status);
    std::memcpy(out, prepared.values.data(), kOutputBytes);
    return static_cast<int>(FloatLutStatus::ok);
}

int acr_test_u16_lut_apply(const std::uint16_t* table,
                           std::size_t table_count,
                           const std::uint16_t* source,
                           std::size_t source_count,
                           std::uint16_t* destination,
                           std::size_t destination_count,
                           std::size_t width) {
    constexpr std::size_t kTableBytes =
        imprint::acr::kU16LutEntryCount * sizeof(std::uint16_t);
    if (table_count < imprint::acr::kU16LutEntryCount || width == 0 ||
        source_count < width || destination_count < width ||
        width > std::numeric_limits<std::size_t>::max() /
                    sizeof(std::uint16_t) ||
        width > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max()) /
                    sizeof(std::uint16_t)) {
        return static_cast<int>(FloatLutStatus::invalid_argument);
    }

    const std::size_t processed_bytes = width * sizeof(std::uint16_t);
    AddressRange table_range{};
    AddressRange destination_range{};
    if (!address_range(table, kTableBytes, alignof(std::uint16_t),
                       &table_range) ||
        !address_range(destination, processed_bytes, alignof(std::uint16_t),
                       &destination_range)) {
        return static_cast<int>(FloatLutStatus::invalid_argument);
    }
    if (overlaps(table_range, destination_range)) {
        return static_cast<int>(FloatLutStatus::overlap);
    }

    U16LutPrepared prepared{};
    std::memcpy(prepared.values.data(), table, kTableBytes);
    return static_cast<int>(imprint::acr::apply_u16_lut_row_0132d484(
        prepared, source, source_count, destination, destination_count, width));
}

}  // extern "C"
