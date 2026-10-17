#include "acr_u16_store.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumPixels = std::size_t{1} << 26;
constexpr std::uint32_t kTileMask = 127;
constexpr std::int64_t kModulus = 2147483647;
constexpr std::int64_t kMultiplier = 16807;
constexpr std::int64_t kSchrageQuotient = 127773;
constexpr std::int64_t kSchrageRemainder = 2836;
constexpr std::int64_t kMinimumAcceptedLowWord = 255;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool make_range(const void *pointer, std::size_t byte_count,
                std::size_t alignment, AddressRange *range) noexcept {
    if (pointer == nullptr || range == nullptr || alignment == 0 ||
        byte_count == 0) {
        return false;
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0 ||
        byte_count > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }

    *range = {begin, begin + byte_count};
    return true;
}

bool overlaps(const AddressRange &left, const AddressRange &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

std::int32_t next_park_miller(std::int32_t seed) noexcept {
    const std::int64_t state = seed;
    const std::int64_t quotient = state / kSchrageQuotient;
    const std::int64_t remainder = state - quotient * kSchrageQuotient;
    std::int64_t next = kMultiplier * remainder - kSchrageRemainder * quotient;
    if (next < 0) next += kModulus;
    return static_cast<std::int32_t>(next);
}

U16DitherTile make_dither_tile() noexcept {
    U16DitherTile tile{};
    std::int32_t seed = 1;
    for (std::size_t index = 0; index < tile.size(); ++index) {
        do {
            seed = next_park_miller(seed);
        } while ((static_cast<std::uint32_t>(seed) & 0xffffu) <
                 static_cast<std::uint32_t>(kMinimumAcceptedLowWord));
        tile[index] = static_cast<std::uint16_t>(seed & 0xffff);
    }
    return tile;
}

U16StoreStatus validate_layout(const float *gray, std::uint16_t *output,
                               std::size_t pixel_count, std::uint32_t width,
                               std::uint32_t height,
                               AddressRange *input_range,
                               AddressRange *output_range) noexcept {
    if (gray == nullptr || output == nullptr || pixel_count == 0 || width == 0 ||
        height == 0 || pixel_count > kMaximumPixels ||
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) !=
            static_cast<std::uint64_t>(pixel_count) ||
        pixel_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        pixel_count >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        return U16StoreStatus::invalid_argument;
    }

    const std::size_t input_bytes = pixel_count * sizeof(float);
    const std::size_t output_bytes = pixel_count * sizeof(std::uint16_t);
    if (!make_range(gray, input_bytes, alignof(float), input_range) ||
        !make_range(output, output_bytes, alignof(std::uint16_t), output_range) ||
        overlaps(*input_range, *output_range)) {
        return U16StoreStatus::invalid_argument;
    }
    return U16StoreStatus::ok;
}

std::uint16_t quantize_sample(float gray, std::uint32_t x, std::uint32_t y,
                              std::int32_t origin_x, std::int32_t origin_y,
                              bool dither,
                              const U16DitherTile *tile) noexcept {
    float addend = 0.5f;
    if (dither) {
        const std::uint32_t column =
            (x + static_cast<std::uint32_t>(origin_x)) & kTileMask;
        const std::uint32_t row =
            (y + static_cast<std::uint32_t>(origin_y)) & kTileMask;
        const auto noise =
            (*tile)[static_cast<std::size_t>(row) * 128 + column];
        addend = static_cast<float>(noise) * 0x1p-16f;
    }

    const float scaled = std::fma(gray, 65535.0f, addend);
    // The validated input and nonnegative offset keep this value in the
    // representable int32 range. Conversion therefore matches FCVTZS.
    std::int32_t code = static_cast<std::int32_t>(scaled);
    if (code < 0) code = 0;
    if (code > 65535) code = 65535;
    return static_cast<std::uint16_t>(code);
}

}  // namespace

const U16DitherTile &get_u16_dither_tile() noexcept {
    static const U16DitherTile tile = make_dither_tile();
    return tile;
}

U16StoreStatus apply_gray_u16_store(const float *gray,
                                    std::uint16_t *output,
                                    std::size_t pixel_count,
                                    std::uint32_t width,
                                    std::uint32_t height,
                                    std::int32_t origin_x,
                                    std::int32_t origin_y,
                                    bool dither) noexcept {
    AddressRange input_range;
    AddressRange output_range;
    const auto layout_status = validate_layout(
        gray, output, pixel_count, width, height, &input_range, &output_range);
    if (layout_status != U16StoreStatus::ok) return layout_status;

    for (std::size_t index = 0; index < pixel_count; ++index) {
        const float sample = gray[index];
        if (!std::isfinite(sample) || sample < 0.0f || sample > 1.0f)
            return U16StoreStatus::invalid_gray;
    }

    const U16DitherTile *tile = dither ? &get_u16_dither_tile() : nullptr;
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::size_t row_offset = static_cast<std::size_t>(y) * width;
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t index = row_offset + x;
            output[index] = quantize_sample(gray[index], x, y, origin_x,
                                            origin_y, dither, tile);
        }
    }
    return U16StoreStatus::ok;
}

}  // namespace imprint::acr
