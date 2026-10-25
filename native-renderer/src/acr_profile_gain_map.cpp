#include "acr_profile_gain_map.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace imprint::acr {
namespace {

constexpr std::uint32_t kLegacyTag = 52525;
constexpr std::uint32_t kExtendedTag = 52544;
constexpr std::size_t kLegacyHeaderBytes = 64;
constexpr std::size_t kExtendedHeaderBytes = 80;
constexpr std::uint32_t kMaxTableCount = (1u << 24) - 1u;
constexpr float kMinGain = 0x1p-12f;
constexpr float kMaxGain = 4096.0f;

std::uint32_t read_u32(const std::uint8_t *data,
                       ProfileGainMapByteOrder byte_order) noexcept {
    if (byte_order == ProfileGainMapByteOrder::little) {
        return static_cast<std::uint32_t>(data[0]) |
               (static_cast<std::uint32_t>(data[1]) << 8) |
               (static_cast<std::uint32_t>(data[2]) << 16) |
               (static_cast<std::uint32_t>(data[3]) << 24);
    }
    return (static_cast<std::uint32_t>(data[0]) << 24) |
           (static_cast<std::uint32_t>(data[1]) << 16) |
           (static_cast<std::uint32_t>(data[2]) << 8) |
           static_cast<std::uint32_t>(data[3]);
}

std::uint64_t read_u64(const std::uint8_t *data,
                       ProfileGainMapByteOrder byte_order) noexcept {
    std::uint64_t value = 0;
    if (byte_order == ProfileGainMapByteOrder::little) {
        for (unsigned int i = 0; i < 8; ++i) {
            value |= static_cast<std::uint64_t>(data[i]) << (i * 8);
        }
    } else {
        for (unsigned int i = 0; i < 8; ++i) {
            value = (value << 8) | static_cast<std::uint64_t>(data[i]);
        }
    }
    return value;
}

float bits_to_float(std::uint32_t bits) noexcept {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

double bits_to_double(std::uint64_t bits) noexcept {
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float read_f32(const std::uint8_t *data,
               ProfileGainMapByteOrder byte_order) noexcept {
    return bits_to_float(read_u32(data, byte_order));
}

double read_f64(const std::uint8_t *data,
                ProfileGainMapByteOrder byte_order) noexcept {
    return bits_to_double(read_u64(data, byte_order));
}

float half_to_float(std::uint16_t half) noexcept {
    const std::uint32_t sign =
        static_cast<std::uint32_t>(half & 0x8000u) << 16;
    const std::uint32_t exponent = (half >> 10) & 0x1fu;
    std::uint32_t mantissa = half & 0x03ffu;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int unbiased_exponent = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --unbiased_exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign |
                   (static_cast<std::uint32_t>(unbiased_exponent + 127)
                    << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + (127u - 15u)) << 23) |
               (mantissa << 13);
    }

    return bits_to_float(bits);
}

bool valid_source_span(const std::uint8_t *data, std::size_t bytes) noexcept {
    if (bytes > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max())) {
        return false;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
    return bytes <= std::numeric_limits<std::uintptr_t>::max() - begin;
}

bool finite_weights(const std::array<float, 5> &weights) noexcept {
    for (float value : weights) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

ProfileGainMapStatus checked_table_count(std::uint32_t rows,
                                         std::uint32_t columns,
                                         std::uint32_t points_n,
                                         std::uint32_t *count) noexcept {
    if (count == nullptr) return ProfileGainMapStatus::invalid_argument;
    const std::uint64_t first = static_cast<std::uint64_t>(rows) * columns;
    if (first > std::numeric_limits<std::uint32_t>::max() ||
        (points_n != 0 &&
         first > std::numeric_limits<std::uint32_t>::max() / points_n)) {
        return ProfileGainMapStatus::arithmetic_overflow;
    }
    const std::uint32_t product =
        static_cast<std::uint32_t>(first) * points_n;
    if (product >= (1u << 24)) return ProfileGainMapStatus::resource_limit;
    *count = product;
    return ProfileGainMapStatus::ok;
}

ProfileGainMapStatus validate_map_header(const ProfileGainMapData &map,
                                         std::uint32_t *count,
                                         bool cached = false) noexcept {
    if (count == nullptr) return ProfileGainMapStatus::invalid_argument;
    if (map.rows == 0 || map.columns == 0 || map.points_n == 0 ||
        map.rows > static_cast<std::uint32_t>(
                       std::numeric_limits<std::int32_t>::max()) ||
        map.columns > static_cast<std::uint32_t>(
                          std::numeric_limits<std::int32_t>::max())) {
        return ProfileGainMapStatus::invalid_header;
    }
    if (map.data_type > 3 || (cached && map.data_type != 2)) {
        return ProfileGainMapStatus::unsupported_data_type;
    }
    if (!(std::isfinite(map.spacing_v) && map.spacing_v > 0.0 &&
          std::isfinite(map.spacing_h) && map.spacing_h > 0.0 &&
          std::isfinite(map.origin_v) && std::isfinite(map.origin_h) &&
          finite_weights(map.weights))) {
        return ProfileGainMapStatus::invalid_header;
    }
    if (!(std::isfinite(map.gamma) && map.gamma >= 0.125f &&
          map.gamma <= 8.0f && std::isfinite(map.gain_min) &&
          map.gain_min >= kMinGain && std::isfinite(map.gain_max) &&
          map.gain_max <= kMaxGain)) {
        return ProfileGainMapStatus::invalid_value;
    }

    const ProfileGainMapStatus count_status = checked_table_count(
        map.rows, map.columns, map.points_n, count);
    if (count_status != ProfileGainMapStatus::ok) return count_status;
    if (map.values.size() != static_cast<std::size_t>(*count)) {
        return ProfileGainMapStatus::invalid_length;
    }
    for (float value : map.values) {
        if (!std::isfinite(value) ||
            (cached ? value <= 0.0f : (value < kMinGain || value > kMaxGain))) {
            return ProfileGainMapStatus::invalid_value;
        }
    }
    return ProfileGainMapStatus::ok;
}

std::int32_t wrapped_negate(std::int32_t value) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t negated = 0u - bits;
    std::int32_t result;
    std::memcpy(&result, &negated, sizeof(result));
    return result;
}

bool narrow_f32(double value, float *out) noexcept {
    if (out == nullptr || !std::isfinite(value) ||
        value > static_cast<double>(std::numeric_limits<float>::max()) ||
        value < -static_cast<double>(std::numeric_limits<float>::max())) {
        return false;
    }
    const float narrowed = static_cast<float>(value);
    if (!std::isfinite(narrowed)) return false;
    *out = narrowed;
    return true;
}

}  // namespace

ProfileGainMapStatus parse_profile_gain_map_payload(
    const std::uint8_t *data, std::size_t bytes, std::uint32_t tag,
    ProfileGainMapByteOrder byte_order, ProfileGainMapData &out) noexcept {
    if (data == nullptr || (tag != kLegacyTag && tag != kExtendedTag) ||
        (byte_order != ProfileGainMapByteOrder::little &&
         byte_order != ProfileGainMapByteOrder::big)) {
        return ProfileGainMapStatus::invalid_argument;
    }
    if (!valid_source_span(data, bytes)) {
        return ProfileGainMapStatus::pointer_overflow;
    }

    const bool extended = tag == kExtendedTag;
    const std::size_t header_bytes =
        extended ? kExtendedHeaderBytes : kLegacyHeaderBytes;
    if (bytes < header_bytes) return ProfileGainMapStatus::invalid_length;

    ProfileGainMapData candidate;
    candidate.rows = read_u32(data, byte_order);
    candidate.columns = read_u32(data + 4, byte_order);
    candidate.spacing_v = read_f64(data + 8, byte_order);
    candidate.spacing_h = read_f64(data + 16, byte_order);
    candidate.origin_v = read_f64(data + 24, byte_order);
    candidate.origin_h = read_f64(data + 32, byte_order);
    candidate.points_n = read_u32(data + 40, byte_order);
    for (std::size_t i = 0; i < candidate.weights.size(); ++i) {
        candidate.weights[i] = read_f32(data + 44 + i * 4, byte_order);
    }

    if (candidate.rows == 0 || candidate.columns == 0 ||
        candidate.points_n == 0 ||
        candidate.rows > static_cast<std::uint32_t>(
                             std::numeric_limits<std::int32_t>::max()) ||
        candidate.columns > static_cast<std::uint32_t>(
                                std::numeric_limits<std::int32_t>::max())) {
        return ProfileGainMapStatus::invalid_header;
    }

    // A singleton axis is canonicalized before checking encoded spacing and
    // origin, matching the reader's order even when ignored bytes are NaN.
    if (candidate.rows == 1) {
        candidate.spacing_v = 1.0;
        candidate.origin_v = 0.0;
    }
    if (candidate.columns == 1) {
        candidate.spacing_h = 1.0;
        candidate.origin_h = 0.0;
    }
    if (!(std::isfinite(candidate.spacing_v) && candidate.spacing_v > 0.0 &&
          std::isfinite(candidate.spacing_h) && candidate.spacing_h > 0.0 &&
          std::isfinite(candidate.origin_v) &&
          std::isfinite(candidate.origin_h) &&
          finite_weights(candidate.weights))) {
        return ProfileGainMapStatus::invalid_header;
    }

    if (extended) {
        candidate.data_type = read_u32(data + 64, byte_order);
        candidate.gamma = read_f32(data + 68, byte_order);
        candidate.gain_min = read_f32(data + 72, byte_order);
        candidate.gain_max = read_f32(data + 76, byte_order);
        if (candidate.data_type > 3) {
            return ProfileGainMapStatus::unsupported_data_type;
        }
        if (!(std::isfinite(candidate.gamma) && candidate.gamma >= 0.125f &&
              candidate.gamma <= 8.0f && std::isfinite(candidate.gain_min) &&
              candidate.gain_min >= kMinGain &&
              std::isfinite(candidate.gain_max) &&
              candidate.gain_max <= kMaxGain)) {
            return ProfileGainMapStatus::invalid_value;
        }
    } else {
        candidate.data_type = 3;
        candidate.gamma = 1.0f;
        candidate.gain_min = 1.0f;
        candidate.gain_max = 1.0f;
    }

    std::uint32_t count = 0;
    const ProfileGainMapStatus count_status = checked_table_count(
        candidate.rows, candidate.columns, candidate.points_n, &count);
    if (count_status != ProfileGainMapStatus::ok) return count_status;

    const std::size_t bytes_per_entry =
        candidate.data_type == 0 ? 1u
        : candidate.data_type == 1 || candidate.data_type == 2 ? 2u
                                                                : 4u;
    const std::size_t count_size = static_cast<std::size_t>(count);
    if (count_size >
        (std::numeric_limits<std::size_t>::max() - header_bytes) /
            bytes_per_entry) {
        return ProfileGainMapStatus::arithmetic_overflow;
    }
    const std::size_t expected_bytes =
        header_bytes + count_size * bytes_per_entry;
    if (bytes != expected_bytes) return ProfileGainMapStatus::invalid_length;

    try {
        candidate.values.resize(count_size);
    } catch (const std::bad_alloc &) {
        return ProfileGainMapStatus::allocation_failure;
    } catch (const std::length_error &) {
        return ProfileGainMapStatus::allocation_failure;
    }

    const std::uint8_t *entry = data + header_bytes;
    float reciprocal = 0.0f;
    float delta = candidate.gain_max - candidate.gain_min;
    if (candidate.data_type == 0) {
        reciprocal = bits_to_float(0x3b808081u);
    } else if (candidate.data_type == 1) {
        reciprocal = bits_to_float(0x37800080u);
    }

    for (std::size_t i = 0; i < count_size; ++i) {
        float value = 0.0f;
        switch (candidate.data_type) {
            case 0: {
                const std::uint8_t raw = entry[0];
                const float normalized = static_cast<float>(raw) * reciprocal;
                value = std::fma(normalized, delta, candidate.gain_min);
                entry += 1;
                break;
            }
            case 1: {
                const std::uint16_t raw = static_cast<std::uint16_t>(
                    byte_order == ProfileGainMapByteOrder::little
                        ? (static_cast<std::uint16_t>(entry[0]) |
                           (static_cast<std::uint16_t>(entry[1]) << 8))
                        : ((static_cast<std::uint16_t>(entry[0]) << 8) |
                           static_cast<std::uint16_t>(entry[1])));
                const float normalized = static_cast<float>(raw) * reciprocal;
                value = std::fma(normalized, delta, candidate.gain_min);
                entry += 2;
                break;
            }
            case 2: {
                const std::uint16_t raw = static_cast<std::uint16_t>(
                    byte_order == ProfileGainMapByteOrder::little
                        ? (static_cast<std::uint16_t>(entry[0]) |
                           (static_cast<std::uint16_t>(entry[1]) << 8))
                        : ((static_cast<std::uint16_t>(entry[0]) << 8) |
                           static_cast<std::uint16_t>(entry[1])));
                value = half_to_float(raw);
                entry += 2;
                break;
            }
            case 3:
                value = read_f32(entry, byte_order);
                entry += 4;
                break;
            default:
                return ProfileGainMapStatus::unsupported_data_type;
        }

        if (!std::isfinite(value) || value < kMinGain || value > kMaxGain) {
            return ProfileGainMapStatus::invalid_value;
        }
        candidate.values[i] = value;
    }

    // Commit only after all source bytes have been consumed. This also makes
    // parsing safe when `data` aliases the old `out.values` allocation.
    out.values.swap(candidate.values);
    out.rows = candidate.rows;
    out.columns = candidate.columns;
    out.points_n = candidate.points_n;
    out.data_type = candidate.data_type;
    out.spacing_v = candidate.spacing_v;
    out.spacing_h = candidate.spacing_h;
    out.origin_v = candidate.origin_v;
    out.origin_h = candidate.origin_h;
    out.weights = candidate.weights;
    out.gamma = candidate.gamma;
    out.gain_min = candidate.gain_min;
    out.gain_max = candidate.gain_max;
    return ProfileGainMapStatus::ok;
}

static ProfileGainMapStatus prepare_profile_gain_map_impl(
    const ProfileGainMapData &map,
    const std::array<std::int32_t, 4> &source_rect, double scalar,
    ProfileGainPrepared &out, bool cached) noexcept {
    std::uint32_t count = 0;
    const ProfileGainMapStatus map_status = validate_map_header(map, &count, cached);
    if (map_status != ProfileGainMapStatus::ok) return map_status;

    const std::int64_t top = source_rect[0];
    const std::int64_t left = source_rect[1];
    const std::int64_t bottom = source_rect[2];
    const std::int64_t right = source_rect[3];
    const std::int64_t height_i64 = bottom - top;
    const std::int64_t width_i64 = right - left;
    if (height_i64 < 1 || width_i64 < 1 ||
        height_i64 > std::numeric_limits<std::int32_t>::max() ||
        width_i64 > std::numeric_limits<std::int32_t>::max()) {
        return ProfileGainMapStatus::invalid_geometry;
    }

    const double height = static_cast<double>(height_i64);
    const double width = static_cast<double>(width_i64);
    const double a[3][3] = {
        {1.0 / map.spacing_h, 0.0, (-map.origin_h) / map.spacing_h},
        {0.0, 1.0 / map.spacing_v, (-map.origin_v) / map.spacing_v},
        {0.0, 0.0, 1.0},
    };
    const double b[3][3] = {
        {1.0 / width, 0.0,
         static_cast<double>(wrapped_negate(source_rect[1])) / width},
        {0.0, 1.0 / height,
         static_cast<double>(wrapped_negate(source_rect[0])) / height},
        {0.0, 0.0, 1.0},
    };
    double c[3][3]{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                sum = std::fma(a[row][k], b[k][column], sum);
                if (!std::isfinite(sum)) {
                    return ProfileGainMapStatus::numeric_failure;
                }
            }
            c[row][column] = sum;
        }
    }
    for (const auto &row : a) {
        for (double value : row) {
            if (!std::isfinite(value)) {
                return ProfileGainMapStatus::numeric_failure;
            }
        }
    }
    for (const auto &row : b) {
        for (double value : row) {
            if (!std::isfinite(value)) {
                return ProfileGainMapStatus::numeric_failure;
            }
        }
    }

    ProfileGainPrepared candidate;
    if (!narrow_f32(c[0][0], &candidate.transform_u[0]) ||
        !narrow_f32(c[0][2], &candidate.transform_u[1]) ||
        !narrow_f32(c[1][1], &candidate.transform_v[0]) ||
        !narrow_f32(c[1][2], &candidate.transform_v[1]) ||
        !std::isfinite(scalar) || !narrow_f32(scalar, &candidate.prelookup_scale)) {
        return ProfileGainMapStatus::numeric_failure;
    }

    candidate.table = map.values.data();
    candidate.table_count = count;
    candidate.coefficients = map.weights;
    candidate.lower_u = 0.0f;
    candidate.lower_v = 0.0f;
    const float columns_f32 = static_cast<float>(map.columns);
    const float rows_f32 = static_cast<float>(map.rows);
    candidate.upper_u = columns_f32 + (-1.0f);
    candidate.upper_v = rows_f32 + (-1.0f);
    candidate.index_scale = static_cast<float>(map.points_n);
    candidate.transfer_scale = map.gamma;
    candidate.stride_v = map.columns * map.points_n;
    candidate.stride_u = map.points_n;
    candidate.max_u = map.columns - 1u;
    candidate.max_v = map.rows - 1u;
    candidate.max_z = map.points_n - 1u;

    if (!std::isfinite(candidate.upper_u) ||
        !std::isfinite(candidate.upper_v) ||
        !std::isfinite(candidate.index_scale) ||
        !std::isfinite(candidate.transfer_scale) ||
        !std::isfinite(candidate.prelookup_scale)) {
        return ProfileGainMapStatus::numeric_failure;
    }

    out = candidate;
    return ProfileGainMapStatus::ok;
}

ProfileGainMapStatus validate_profile_gain_source_map(
    const ProfileGainMapData &map) noexcept {
    std::uint32_t count = 0;
    return validate_map_header(map, &count);
}

ProfileGainMapStatus prepare_profile_gain_map(
    const ProfileGainMapData &map,
    const std::array<std::int32_t, 4> &source_rect, double scalar,
    ProfileGainPrepared &out) noexcept {
    return prepare_profile_gain_map_impl(map, source_rect, scalar, out, false);
}

ProfileGainMapStatus prepare_cached_profile_gain_map(
    const ProfileGainMapData &map,
    const std::array<std::int32_t, 4> &source_rect, double scalar,
    ProfileGainPrepared &out) noexcept {
    return prepare_profile_gain_map_impl(map, source_rect, scalar, out, true);
}

}  // namespace imprint::acr
