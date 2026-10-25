#include "acr_profile_u16.hpp"

#include "acr_profile_stage.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumPixels = std::size_t{1} << 26;
constexpr std::uint16_t kSignedWordBias = 0x8000u;
constexpr float kU16Scale = 1.0f / 65535.0f;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool checked_product(std::size_t left, std::size_t right,
                     std::size_t *product) noexcept {
    if (product == nullptr ||
        (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

bool make_range(const void *pointer, std::size_t count,
                std::size_t element_size, std::size_t alignment,
                AddressRange *range) noexcept {
    if (range == nullptr || alignment == 0) return false;
    std::size_t byte_count = 0;
    if (!checked_product(count, element_size, &byte_count) ||
        byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::ptrdiff_t>::max())) {
        return false;
    }
    if (byte_count == 0) {
        if (pointer == nullptr) {
            *range = {};
            return true;
        }
        const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
        if (begin % alignment != 0) return false;
        *range = {begin, begin};
        return true;
    }
    if (pointer == nullptr) return false;
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

bool required_samples(std::size_t height, std::size_t stride,
                      std::size_t row_samples,
                      std::size_t *required) noexcept {
    std::size_t row_offset = 0;
    if (height == 0 ||
        !checked_product(height - 1, stride, &row_offset) ||
        row_offset > std::numeric_limits<std::size_t>::max() - row_samples) {
        return false;
    }
    *required = row_offset + row_samples;
    return true;
}

std::uint16_t encode_sample(float value, bool signed_word) noexcept {
    const float scaled = std::fma(value, 65535.0f, 0.5f);
    std::int32_t converted = 0;
    if (std::isnan(scaled)) {
        converted = 0;
    } else if (scaled >= 2147483648.0f) {
        converted = std::numeric_limits<std::int32_t>::max();
    } else if (scaled <= -2147483648.0f) {
        converted = std::numeric_limits<std::int32_t>::min();
    } else {
        converted = static_cast<std::int32_t>(scaled);
    }
    auto word = static_cast<std::uint16_t>(
        static_cast<std::uint32_t>(converted) & 0xffffu);
    if (signed_word) word ^= kSignedWordBias;
    return word;
}

ProfileU16Status map_stage_status(ProfileTableStatus status) noexcept {
    switch (status) {
        case ProfileTableStatus::ok:
            return ProfileU16Status::ok;
        case ProfileTableStatus::invalid_argument:
            return ProfileU16Status::invalid_argument;
        case ProfileTableStatus::invalid_table:
            return ProfileU16Status::invalid_table;
        case ProfileTableStatus::allocation_failure:
            return ProfileU16Status::allocation_failure;
        case ProfileTableStatus::invalid_curve:
        case ProfileTableStatus::invalid_rgb:
        case ProfileTableStatus::overflow:
            return ProfileU16Status::kernel_failure;
    }
    return ProfileU16Status::kernel_failure;
}

}  // namespace

ProfileU16Status apply_profile_stage_sdr_u16(
    const std::uint16_t *src, std::size_t src_count, std::size_t src_stride,
    std::size_t height, std::size_t width, const ProfileTableView &table,
    unsigned encoding, double lookup_gain, double amount,
    std::uint32_t pixel_type, std::uint16_t *dst, std::size_t dst_count,
    std::size_t dst_stride) noexcept {
    if (validate_profile_table_view(table) != ProfileTableStatus::ok) {
        return ProfileU16Status::invalid_table;
    }

    if (height == 0 || width == 0 || (pixel_type != 3 && pixel_type != 8)) {
        return ProfileU16Status::invalid_argument;
    }
    if (width > std::numeric_limits<std::size_t>::max() / 3) {
        return ProfileU16Status::arithmetic_overflow;
    }
    const std::size_t row_samples = width * 3;
    if (src_stride < row_samples || dst_stride < row_samples) {
        return ProfileU16Status::invalid_argument;
    }
    std::size_t src_stride_bytes = 0;
    std::size_t dst_stride_bytes = 0;
    if (!checked_product(src_stride, sizeof(std::uint16_t),
                         &src_stride_bytes) ||
        !checked_product(dst_stride, sizeof(std::uint16_t),
                         &dst_stride_bytes) ||
        src_stride_bytes > static_cast<std::size_t>(
                               std::numeric_limits<std::ptrdiff_t>::max()) ||
        dst_stride_bytes > static_cast<std::size_t>(
                               std::numeric_limits<std::ptrdiff_t>::max())) {
        return ProfileU16Status::arithmetic_overflow;
    }
    if (height > std::numeric_limits<std::size_t>::max() / width) {
        return ProfileU16Status::arithmetic_overflow;
    }
    const std::size_t pixels = width * height;
    if (pixels > kMaximumPixels) return ProfileU16Status::resource_limit;

    std::size_t required_src = 0;
    std::size_t required_dst = 0;
    if (!required_samples(height, src_stride, row_samples, &required_src) ||
        !required_samples(height, dst_stride, row_samples, &required_dst)) {
        return ProfileU16Status::arithmetic_overflow;
    }
    if (src == nullptr || dst == nullptr ||
        reinterpret_cast<std::uintptr_t>(src) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(dst) % alignof(std::uint16_t) != 0) {
        return ProfileU16Status::invalid_argument;
    }
    if (src_count < required_src || dst_count < required_dst) {
        return ProfileU16Status::buffer_too_small;
    }

    AddressRange src_range;
    AddressRange dst_range;
    if (!make_range(src, src_count, sizeof(std::uint16_t),
                    alignof(std::uint16_t), &src_range) ||
        !make_range(dst, dst_count, sizeof(std::uint16_t),
                    alignof(std::uint16_t), &dst_range)) {
        return ProfileU16Status::arithmetic_overflow;
    }

    const bool exact_in_place = src == dst && src_count == dst_count &&
                                src_stride == dst_stride;
    if (overlaps(src_range, dst_range) && !exact_in_place) {
        return ProfileU16Status::unsupported_aliasing;
    }

    std::size_t table_values = 0;
    AddressRange table_data_range;
    AddressRange table_object_range;
    if (!checked_product(table.count, 3, &table_values) ||
        !make_range(table.data, table_values, sizeof(float), alignof(float),
                    &table_data_range) ||
        !make_range(&table, sizeof(ProfileTableView), 1,
                    alignof(ProfileTableView), &table_object_range)) {
        return ProfileU16Status::arithmetic_overflow;
    }
    if (overlaps(dst_range, table_data_range) ||
        overlaps(dst_range, table_object_range)) {
        return ProfileU16Status::unsupported_aliasing;
    }

    // Keep the float stage's accepted parameter domain exactly intact.
    if (encoding > 1 || !std::isfinite(lookup_gain) || lookup_gain < 0.0 ||
        lookup_gain > 65536.0 || !std::isfinite(amount) || amount < 0.0 ||
        amount > 2.0) {
        return ProfileU16Status::invalid_argument;
    }

    std::size_t active_samples = 0;
    std::size_t active_bytes = 0;
    if (!checked_product(pixels, 3, &active_samples) ||
        !checked_product(active_samples, sizeof(std::uint16_t), &active_bytes) ||
        active_bytes > static_cast<std::size_t>(
                           std::numeric_limits<std::ptrdiff_t>::max())) {
        return ProfileU16Status::arithmetic_overflow;
    }

    try {
        std::vector<float> float_input(active_samples);
        std::vector<float> float_output(active_samples);
        std::vector<std::uint16_t> candidate(active_samples);
        const bool signed_word = pixel_type == 8;

        for (std::size_t y = 0; y < height; ++y) {
            const std::size_t src_row = y * src_stride;
            const std::size_t compact_row = y * row_samples;
            for (std::size_t x = 0; x < row_samples; ++x) {
                std::uint16_t sample = src[src_row + x];
                if (signed_word) sample ^= kSignedWordBias;
                float_input[compact_row + x] =
                    static_cast<float>(sample) * kU16Scale;
            }
        }

        const auto stage_status = apply_profile_stage_sdr(
            float_input.data(), float_output.data(), pixels, table, encoding,
            lookup_gain, amount);
        const auto wrapper_status = map_stage_status(stage_status);
        if (wrapper_status != ProfileU16Status::ok) return wrapper_status;

        for (std::size_t i = 0; i < active_samples; ++i) {
            candidate[i] = encode_sample(float_output[i], signed_word);
        }

        const std::size_t row_bytes = row_samples * sizeof(std::uint16_t);
        for (std::size_t y = 0; y < height; ++y) {
            std::memcpy(dst + y * dst_stride,
                        candidate.data() + y * row_samples, row_bytes);
        }
        return ProfileU16Status::ok;
    } catch (const std::bad_alloc &) {
        return ProfileU16Status::allocation_failure;
    } catch (const std::length_error &) {
        return ProfileU16Status::resource_limit;
    } catch (...) {
        return ProfileU16Status::allocation_failure;
    }
}

}  // namespace imprint::acr
