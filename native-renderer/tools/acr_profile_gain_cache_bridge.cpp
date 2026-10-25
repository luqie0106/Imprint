#include "../src/acr_profile_gain_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

struct BridgeSpan {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool overlaps(const BridgeSpan &left, const BridgeSpan &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

template <typename T>
imprint::acr::ProfileGainMapStatus make_span(const T *pointer,
                                             std::size_t count,
                                             BridgeSpan *span) noexcept {
    using imprint::acr::ProfileGainMapStatus;
    if (span == nullptr || pointer == nullptr) {
        return ProfileGainMapStatus::invalid_argument;
    }
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) != 0) {
        return ProfileGainMapStatus::unsupported_alignment;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return ProfileGainMapStatus::pointer_overflow;
    }
    const std::size_t bytes = count * sizeof(T);
    if (bytes > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max())) {
        return ProfileGainMapStatus::pointer_overflow;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return ProfileGainMapStatus::pointer_overflow;
    }
    *span = {begin, begin + bytes};
    return ProfileGainMapStatus::ok;
}

imprint::acr::ProfileGainMapStatus make_byte_span(const std::uint8_t *pointer,
                                                  std::size_t bytes,
                                                  BridgeSpan *span) noexcept {
    using imprint::acr::ProfileGainMapStatus;
    if (span == nullptr || pointer == nullptr) {
        return ProfileGainMapStatus::invalid_argument;
    }
    if (bytes > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max())) {
        return ProfileGainMapStatus::pointer_overflow;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return ProfileGainMapStatus::pointer_overflow;
    }
    *span = {begin, begin + bytes};
    return ProfileGainMapStatus::ok;
}

template <typename T>
imprint::acr::ProfileGainMapStatus require_span(const T *pointer,
                                                std::size_t count,
                                                BridgeSpan *span) noexcept {
    return make_span(pointer, count, span);
}

}  // namespace

extern "C" int acr_profile_gain_cache_bridge(
    const std::uint8_t *data, std::size_t bytes, std::uint32_t tag,
    std::uint32_t byteorder, std::int32_t amount, std::uint32_t lookup_flag,
    std::uint32_t params_predicate, const std::int32_t *rect, double scalar,
    double *meta, std::size_t meta_count, float *fields,
    std::size_t field_count, std::uint32_t *geometry,
    std::size_t geometry_count, float *values, std::size_t value_capacity,
    std::size_t *values_written) noexcept {
    using imprint::acr::ProfileGainCacheMap;
    using imprint::acr::ProfileGainMapByteOrder;
    using imprint::acr::ProfileGainMapData;
    using imprint::acr::ProfileGainMapStatus;

    if (lookup_flag > std::numeric_limits<std::uint8_t>::max()) {
        return static_cast<int>(ProfileGainMapStatus::invalid_argument);
    }

    BridgeSpan input_data_span;
    BridgeSpan rect_span;
    BridgeSpan meta_span;
    BridgeSpan fields_span;
    BridgeSpan geometry_span;
    BridgeSpan values_span;
    BridgeSpan written_span;

    ProfileGainMapStatus status = make_byte_span(data, bytes, &input_data_span);
    if (status != ProfileGainMapStatus::ok) return static_cast<int>(status);
    status = require_span(rect, 4, &rect_span);
    if (status != ProfileGainMapStatus::ok) return static_cast<int>(status);
    status = require_span(meta, meta_count, &meta_span);
    if (status != ProfileGainMapStatus::ok) return static_cast<int>(status);
    status = require_span(fields, field_count, &fields_span);
    if (status != ProfileGainMapStatus::ok) return static_cast<int>(status);
    status = require_span(geometry, geometry_count, &geometry_span);
    if (status != ProfileGainMapStatus::ok) return static_cast<int>(status);
    status = require_span(values, value_capacity, &values_span);
    if (status != ProfileGainMapStatus::ok) return static_cast<int>(status);
    status = require_span(values_written, 1, &written_span);
    if (status != ProfileGainMapStatus::ok) return static_cast<int>(status);

    const std::array<BridgeSpan, 4> output_spans = {
        meta_span, fields_span, geometry_span, values_span};
    for (std::size_t i = 0; i < output_spans.size(); ++i) {
        if (overlaps(output_spans[i], written_span) ||
            overlaps(output_spans[i], input_data_span) ||
            overlaps(output_spans[i], rect_span)) {
            return static_cast<int>(ProfileGainMapStatus::overlap);
        }
        for (std::size_t j = i + 1; j < output_spans.size(); ++j) {
            if (overlaps(output_spans[i], output_spans[j])) {
                return static_cast<int>(ProfileGainMapStatus::overlap);
            }
        }
    }
    if (overlaps(written_span, input_data_span) ||
        overlaps(written_span, rect_span)) {
        return static_cast<int>(ProfileGainMapStatus::overlap);
    }

    if (meta_count < 21 || field_count < 16 || geometry_count < 5) {
        return static_cast<int>(ProfileGainMapStatus::buffer_too_small);
    }

    std::array<std::int32_t, 4> source_rect{};
    std::memcpy(source_rect.data(), rect, sizeof(source_rect));
    ProfileGainMapData source;
    const auto byte_order = static_cast<ProfileGainMapByteOrder>(byteorder);
    status = imprint::acr::parse_profile_gain_map_payload(
        data, bytes, tag, byte_order, source);
    if (status != ProfileGainMapStatus::ok) return static_cast<int>(status);

    ProfileGainCacheMap cache;
    status = imprint::acr::build_profile_gain_cache_map(
        source, amount, static_cast<std::uint8_t>(lookup_flag),
        params_predicate, cache);
    if (status != ProfileGainMapStatus::ok) return static_cast<int>(status);

    imprint::acr::ProfileGainPrepared prepared;
    status = imprint::acr::prepare_cached_profile_gain_map(
        cache.map, source_rect, scalar, prepared);
    if (status != ProfileGainMapStatus::ok) return static_cast<int>(status);
    if (value_capacity < prepared.table_count) {
        return static_cast<int>(ProfileGainMapStatus::buffer_too_small);
    }

    std::array<double, 21> meta_candidate{};
    meta_candidate[0] = static_cast<double>(cache.map.rows);
    meta_candidate[1] = static_cast<double>(cache.map.columns);
    meta_candidate[2] = cache.map.spacing_v;
    meta_candidate[3] = cache.map.spacing_h;
    meta_candidate[4] = cache.map.origin_v;
    meta_candidate[5] = cache.map.origin_h;
    meta_candidate[6] = static_cast<double>(cache.map.points_n);
    for (std::size_t i = 0; i < cache.map.weights.size(); ++i) {
        meta_candidate[7 + i] = static_cast<double>(cache.map.weights[i]);
    }
    meta_candidate[12] = static_cast<double>(cache.map.data_type);
    meta_candidate[13] = static_cast<double>(cache.map.gamma);
    meta_candidate[14] = static_cast<double>(cache.map.gain_min);
    meta_candidate[15] = static_cast<double>(cache.map.gain_max);
    meta_candidate[16] = static_cast<double>(prepared.stride_v);
    meta_candidate[17] = static_cast<double>(prepared.stride_u);
    meta_candidate[18] = static_cast<double>(prepared.table_count);
    meta_candidate[19] = static_cast<double>(cache.exponent);
    meta_candidate[20] = cache.negative_remap ? 1.0 : 0.0;

    std::array<float, 16> fields_candidate{};
    for (std::size_t i = 0; i < prepared.coefficients.size(); ++i) {
        fields_candidate[i] = prepared.coefficients[i];
    }
    fields_candidate[5] = prepared.transform_u[0];
    fields_candidate[6] = prepared.transform_u[1];
    fields_candidate[7] = prepared.transform_v[0];
    fields_candidate[8] = prepared.transform_v[1];
    fields_candidate[9] = prepared.lower_u;
    fields_candidate[10] = prepared.upper_u;
    fields_candidate[11] = prepared.lower_v;
    fields_candidate[12] = prepared.upper_v;
    fields_candidate[13] = prepared.prelookup_scale;
    fields_candidate[14] = prepared.index_scale;
    fields_candidate[15] = prepared.transfer_scale;

    const std::array<std::uint32_t, 5> geometry_candidate = {
        prepared.stride_v, prepared.stride_u, prepared.max_u,
        prepared.max_v, prepared.max_z};

    std::memcpy(meta, meta_candidate.data(), sizeof(meta_candidate));
    std::memcpy(fields, fields_candidate.data(), sizeof(fields_candidate));
    std::memcpy(geometry, geometry_candidate.data(), sizeof(geometry_candidate));
    std::memcpy(values, prepared.table,
                prepared.table_count * sizeof(float));
    *values_written = prepared.table_count;
    return static_cast<int>(ProfileGainMapStatus::ok);
}
