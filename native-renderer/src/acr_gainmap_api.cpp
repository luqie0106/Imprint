#include "imprint_renderer.h"
#include "acr_gainmap.hpp"
#include "acr_stage3_core3_format.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

struct Span { std::uintptr_t begin, end; };

bool span(const void* pointer, size_t count, size_t item_size,
          size_t alignment, Span* result) noexcept {
    if (!pointer || !item_size || !alignment || !result) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    if (start % alignment || count > std::numeric_limits<size_t>::max()/item_size ||
        count > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())/item_size) return false;
    const size_t bytes = count*item_size;
    if (bytes > std::numeric_limits<std::uintptr_t>::max()-start) return false;
    *result = {start, start+bytes};
    return true;
}

bool overlap(Span a, Span b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}

bool multiply(size_t a, size_t b, size_t* result) noexcept {
    if (b && a > std::numeric_limits<size_t>::max()/b) return false;
    *result = a*b;
    return true;
}

im_status mapped(imprint::acr::GainMapStatus status) noexcept {
    using imprint::acr::GainMapStatus;
    switch (status) {
        case GainMapStatus::ok: return IM_STATUS_OK;
        case GainMapStatus::buffer_too_small: return IM_STATUS_BUFFER_TOO_SMALL;
        case GainMapStatus::allocation_failure:
        case GainMapStatus::length_error:
        case GainMapStatus::resource_limit: return IM_STATUS_RUNTIME_ERROR;
        default: return IM_STATUS_INVALID_ARGUMENT;
    }
}

template<class T> class Row {
public:
    explicit Row(size_t count) : storage_(count + 16/sizeof(T)) {
        const auto start = reinterpret_cast<std::uintptr_t>(storage_.data());
        if (start > std::numeric_limits<std::uintptr_t>::max()-15)
            throw std::length_error("aligned GainMap row address overflow");
        const auto aligned = (start+15)&~std::uintptr_t(15);
        pointer_ = storage_.data()+(aligned-start)/sizeof(T);
    }
    T* data() noexcept { return pointer_; }
private:
    std::vector<T> storage_;
    T* pointer_ = nullptr;
};

} // namespace

extern "C" IMPRINT_API im_status im_native_acr_gainmap_prepared_f32(
    const float* source, size_t source_samples, size_t source_row_stride,
    const int32_t rects8[8], const uint32_t controls9[9],
    const double geometry4[4], const float* map, size_t map_samples,
    uint16_t black_floor, size_t tile_width, size_t tile_height,
    float* destination, size_t destination_samples) {
    return mapped(imprint::acr::apply_gainmap_prepared_f32_to_buffer(
        source, source_samples, source_row_stride, rects8, controls9, geometry4,
        map, map_samples, black_floor, tile_width, tile_height,
        destination, destination_samples));
}

extern "C" IMPRINT_API im_status im_native_acr_gainmap_prepared_u16(
    const uint16_t* source, size_t source_samples, size_t source_row_stride,
    const int32_t rects8[8], const uint32_t controls9[9],
    const double geometry4[4], const float* map, size_t map_samples,
    uint16_t black_floor, size_t tile_width, size_t tile_height,
    uint16_t* destination, size_t destination_samples) {
    using namespace imprint::acr;
    Span src{}, dst{}, rectangles{}, controls{}, geometry{}, table{};
    if (!span(source, source_samples, sizeof(*source), alignof(uint16_t), &src) ||
        !span(destination, destination_samples, sizeof(*destination), alignof(uint16_t), &dst) ||
        !span(rects8, 8, sizeof(*rects8), alignof(int32_t), &rectangles) ||
        !span(controls9, 9, sizeof(*controls9), alignof(uint32_t), &controls) ||
        !span(geometry4, 4, sizeof(*geometry4), alignof(double), &geometry) ||
        !span(map, map_samples, sizeof(*map), alignof(float), &table) ||
        overlap(dst, src) || overlap(dst, rectangles) || overlap(dst, controls) ||
        overlap(dst, geometry) || overlap(dst, table)) return IM_STATUS_INVALID_ARGUMENT;

    const int64_t h = int64_t(rects8[2])-int64_t(rects8[0]);
    const int64_t w = int64_t(rects8[3])-int64_t(rects8[1]);
    if (h <= 0 || w <= 0 || h > INT32_MAX || w > INT32_MAX ||
        !controls9[0] || map_samples > 1000000 ||
        !tile_width || !tile_height || tile_width > 1024 || tile_height > 1024)
        return IM_STATUS_INVALID_ARGUMENT;
    size_t row_samples = 0, samples = 0, preceding = 0;
    if (!multiply(static_cast<size_t>(w), controls9[0], &row_samples) ||
        !multiply(row_samples, static_cast<size_t>(h), &samples) ||
        source_row_stride < row_samples || samples > 128u*1024u*1024u ||
        !multiply(static_cast<size_t>(h)-1, source_row_stride, &preceding) ||
        preceding > std::numeric_limits<size_t>::max()-row_samples)
        return IM_STATUS_INVALID_ARGUMENT;
    if (source_samples < preceding+row_samples || destination_samples < samples)
        return IM_STATUS_BUFFER_TOO_SMALL;
    // The total-sample cap also makes row+7 and the aligned row allocations safe.
    const size_t padded = (row_samples+7)/8*8;
    GainMapSpec spec{};
    spec.image_bounds = {rects8[0], rects8[1], rects8[2], rects8[3]};
    spec.area = {rects8[4], rects8[5], rects8[6], rects8[7]};
    spec.image_planes = controls9[0];
    spec.first_plane = controls9[1];
    spec.plane_count = controls9[2];
    spec.row_pitch = controls9[3];
    spec.col_pitch = controls9[4];
    spec.map_v = controls9[5];
    spec.map_h = controls9[6];
    spec.map_planes = controls9[7];
    spec.list_stage = controls9[8];
    spec.spacing_v = geometry4[0];
    spec.spacing_h = geometry4[1];
    spec.origin_v = geometry4[2];
    spec.origin_h = geometry4[3];
    spec.black_floor = black_floor;
    spec.tile_width = tile_width;
    spec.tile_height = tile_height;
    try {
        Row<uint16_t> integer_row(padded);
        Row<float> float_row(padded);
        GainMapOutput result{};
        result.pixels.resize(samples);
        for (size_t y = 0; y < static_cast<size_t>(h); ++y) {
            std::fill_n(integer_row.data(), padded, uint16_t(0));
            std::copy_n(source+y*source_row_stride, row_samples, integer_row.data());
            if (core3_u16_to_f32_prepared_row_008096e4(
                    integer_row.data(), padded, float_row.data(), padded, padded)
                != Core3FormatStatus::ok) return IM_STATUS_INVALID_ARGUMENT;
            std::copy_n(float_row.data(), row_samples, result.pixels.data()+y*row_samples);
        }
        const auto status = apply_gainmap_prepared_f32(
            result.pixels.data(), samples, row_samples, spec, map, map_samples, &result);
        if (status != GainMapStatus::ok) return mapped(status);
        if (result.pixels.size() != samples) return IM_STATUS_RUNTIME_ERROR;
        std::vector<uint16_t> candidate(samples);
        for (size_t y = 0; y < static_cast<size_t>(h); ++y) {
            std::fill_n(float_row.data(), padded, 0.0f);
            std::copy_n(result.pixels.data()+y*row_samples, row_samples, float_row.data());
            if (core3_f32_to_u16_prepared_row_008103dc(
                    float_row.data(), padded, integer_row.data(), padded, padded)
                != Core3FormatStatus::ok) return IM_STATUS_INVALID_ARGUMENT;
            std::copy_n(integer_row.data(), row_samples, candidate.data()+y*row_samples);
        }
        std::copy(candidate.begin(), candidate.end(), destination);
        return IM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::length_error&) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}
