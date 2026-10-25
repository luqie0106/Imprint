#include "../src/acr_profile_gain.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
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
imprint::acr::ProfileGainStatus validate_bridge_span(
    const T *pointer, std::size_t count, BridgeSpan *span) noexcept {
    using imprint::acr::ProfileGainStatus;
    if (span == nullptr) return ProfileGainStatus::invalid_argument;
    if (pointer == nullptr) return ProfileGainStatus::invalid_argument;
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) != 0) {
        return ProfileGainStatus::unsupported_alignment;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    const std::size_t bytes = count * sizeof(T);
    if (bytes > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max())) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    *span = {begin, begin + bytes};
    return ProfileGainStatus::ok;
}

}  // namespace

extern "C" int acr_profile_gain_bridge(
    const float *fields, std::size_t fields_count,
    const std::uint32_t *geometry, std::size_t geometry_count,
    const float *table, std::size_t table_count,
    const float *const *inputs, const std::size_t *input_counts, float *dest,
    std::size_t dest_count, std::size_t outer, std::size_t inner,
    std::size_t value_stride, std::size_t coord_stride) noexcept {
    using imprint::acr::ProfileGainStatus;
    if (fields == nullptr || geometry == nullptr || inputs == nullptr ||
        input_counts == nullptr) {
        return static_cast<int>(ProfileGainStatus::invalid_argument);
    }
    if (fields_count != 16 || geometry_count != 5) {
        return static_cast<int>(ProfileGainStatus::invalid_argument);
    }
    if (dest == nullptr) {
        return static_cast<int>(ProfileGainStatus::invalid_argument);
    }

    BridgeSpan dest_span;
    const auto dest_span_status =
        validate_bridge_span(dest, dest_count, &dest_span);
    if (dest_span_status != ProfileGainStatus::ok) {
        return static_cast<int>(dest_span_status);
    }

    std::array<BridgeSpan, 4> control_spans{};
    const auto fields_span_status =
        validate_bridge_span(fields, fields_count, &control_spans[0]);
    if (fields_span_status != ProfileGainStatus::ok) {
        return static_cast<int>(fields_span_status);
    }
    const auto geometry_span_status =
        validate_bridge_span(geometry, geometry_count, &control_spans[1]);
    if (geometry_span_status != ProfileGainStatus::ok) {
        return static_cast<int>(geometry_span_status);
    }
    const auto inputs_span_status =
        validate_bridge_span(inputs, 5, &control_spans[2]);
    if (inputs_span_status != ProfileGainStatus::ok) {
        return static_cast<int>(inputs_span_status);
    }
    const auto counts_span_status =
        validate_bridge_span(input_counts, 5, &control_spans[3]);
    if (counts_span_status != ProfileGainStatus::ok) {
        return static_cast<int>(counts_span_status);
    }
    // The bridge consumes these declared control spans before calling core;
    // keep the core's eventual output commit from overwriting them.
    for (const BridgeSpan &control_span : control_spans) {
        if (overlaps(dest_span, control_span)) {
            return static_cast<int>(ProfileGainStatus::overlap);
        }
    }

    imprint::acr::ProfileGainPrepared prepared;
    prepared.table = table;
    prepared.table_count = table_count;
    for (std::size_t i = 0; i < prepared.coefficients.size(); ++i) {
        prepared.coefficients[i] = fields[i];
    }
    prepared.transform_u = {fields[5], fields[6]};
    prepared.transform_v = {fields[7], fields[8]};
    prepared.lower_u = fields[9];
    prepared.upper_u = fields[10];
    prepared.lower_v = fields[11];
    prepared.upper_v = fields[12];
    prepared.prelookup_scale = fields[13];
    prepared.index_scale = fields[14];
    prepared.transfer_scale = fields[15];
    prepared.stride_v = geometry[0];
    prepared.stride_u = geometry[1];
    prepared.max_u = geometry[2];
    prepared.max_v = geometry[3];
    prepared.max_z = geometry[4];

    std::array<imprint::acr::ProfileGainReadView, 5> views{};
    for (std::size_t i = 0; i < views.size(); ++i) {
        views[i] = {inputs[i], input_counts[i]};
    }
    const auto status = imprint::acr::process_profile_gain_prepared_f32(
        prepared, views, {dest, dest_count}, outer, inner, value_stride,
        coord_stride);
    return static_cast<int>(status);
}
