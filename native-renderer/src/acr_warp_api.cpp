#include "imprint_renderer.h"
#include "acr_warp.hpp"
#include "acr_stage3_core3_format.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
struct Span { std::uintptr_t begin, end; };
bool span(const void* pointer, size_t count, size_t size, size_t alignment,
          Span* output) noexcept {
    if (!pointer || !output) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    if (start % alignment || count > size_t(PTRDIFF_MAX)/size ||
        count > std::numeric_limits<size_t>::max()/size) return false;
    const size_t bytes = count*size;
    if (bytes > UINTPTR_MAX-start) return false;
    *output = {start, start+bytes};
    return true;
}
bool overlap(Span a, Span b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}
im_status mapped(imprint::acr::WarpStatus status) noexcept {
    using imprint::acr::WarpStatus;
    switch (status) {
        case WarpStatus::ok: return IM_STATUS_OK;
        case WarpStatus::buffer_too_small: return IM_STATUS_BUFFER_TOO_SMALL;
        case WarpStatus::resource_limit:
        case WarpStatus::allocation_failure:
        case WarpStatus::length_error: return IM_STATUS_RUNTIME_ERROR;
        default: return IM_STATUS_INVALID_ARGUMENT;
    }
}
template<class T> class AlignedRow {
public:
    explicit AlignedRow(size_t n) : storage_(n+16/sizeof(T)) {
        auto address = reinterpret_cast<std::uintptr_t>(storage_.data());
        pointer_ = storage_.data()+((16-address%16)%16)/sizeof(T);
    }
    T* data() noexcept { return pointer_; }
private:
    std::vector<T> storage_;
    T* pointer_ = nullptr;
};
} // namespace

extern "C" IMPRINT_API im_status im_native_acr_warp_prepared_f32(
    const float* source, size_t source_samples, size_t source_row_stride,
    const int32_t rect4[4], const uint32_t controls2[2],
    const double geometry3[3], const double* coefficients, size_t coefficient_count,
    size_t tile_width, size_t tile_height, float* destination, size_t destination_samples) {
    return mapped(imprint::acr::apply_warp_prepared_f32_to_buffer(
        source, source_samples, source_row_stride, rect4, controls2, geometry3,
        coefficients, coefficient_count, tile_width, tile_height,
        destination, destination_samples));
}

extern "C" IMPRINT_API im_status im_native_acr_warp_prepared_u16_f32(
    const uint16_t* source, size_t source_samples, size_t source_row_stride,
    const int32_t rect4[4], const uint32_t controls2[2],
    const double geometry3[3], const double* coefficients, size_t coefficient_count,
    size_t tile_width, size_t tile_height, float* destination, size_t destination_samples) {
    // Explicit normalized-format adapter. The recovered Warp consumer writes
    // float tiles; this entry makes no claim about Adobe's final u16 writeback.
    Span ranges[6]{};
    if (!span(source, source_samples, sizeof(uint16_t), alignof(uint16_t), &ranges[0]) ||
        !span(rect4, 4, sizeof(int32_t), alignof(int32_t), &ranges[1]) ||
        !span(controls2, 2, sizeof(uint32_t), alignof(uint32_t), &ranges[2]) ||
        !span(geometry3, 3, sizeof(double), alignof(double), &ranges[3]) ||
        !span(coefficients, coefficient_count, sizeof(double), alignof(double), &ranges[4]) ||
        !span(destination, destination_samples, sizeof(float), alignof(float), &ranges[5]))
        return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i=0; i<6; ++i)
        for (size_t j=i+1; j<6; ++j)
            if (overlap(ranges[i], ranges[j])) return IM_STATUS_INVALID_ARGUMENT;
    const int64_t h = int64_t(rect4[2])-int64_t(rect4[0]);
    const int64_t w = int64_t(rect4[3])-int64_t(rect4[1]);
    if (h<=0 || w<=0 || h>INT32_MAX || w>INT32_MAX ||
        controls2[0]<1 || controls2[0]>4 ||
        !(controls2[1]==1 || controls2[1]==controls2[0]) ||
        coefficient_count!=size_t(controls2[1])*6 ||
        !tile_width || !tile_height || tile_width>1024 || tile_height>1024)
        return IM_STATUS_INVALID_ARGUMENT;
    const size_t row = size_t(w)*controls2[0];
    if (row>128u*1024u*1024u || size_t(h)>128u*1024u*1024u/row ||
        source_row_stride<row ||
        (size_t(h)-1 && source_row_stride>(SIZE_MAX-row)/(size_t(h)-1)))
        return IM_STATUS_INVALID_ARGUMENT;
    const size_t count = size_t(h)*row;
    if (source_samples<(size_t(h)-1)*source_row_stride+row || destination_samples<count)
        return IM_STATUS_BUFFER_TOO_SMALL;
    try {
        const size_t padded = (row+3)/4*4;
        AlignedRow<uint16_t> integers(padded);
        AlignedRow<float> floats(padded);
        std::vector<float> normalized(count);
        for (size_t y=0; y<size_t(h); ++y) {
            std::fill_n(integers.data(), padded, uint16_t(0));
            std::copy_n(source+y*source_row_stride, row, integers.data());
            if (imprint::acr::core3_u16_to_f32_prepared_row_008096e4(
                    integers.data(), padded, floats.data(), padded, padded)
                != imprint::acr::Core3FormatStatus::ok) return IM_STATUS_RUNTIME_ERROR;
            std::copy_n(floats.data(), row, normalized.data()+y*row);
        }
        return mapped(imprint::acr::apply_warp_prepared_f32_to_buffer(
            normalized.data(), count, row, rect4, controls2, geometry3,
            coefficients, coefficient_count, tile_width, tile_height,
            destination, destination_samples));
    } catch (const std::bad_alloc&) { return IM_STATUS_RUNTIME_ERROR; }
      catch (const std::length_error&) { return IM_STATUS_RUNTIME_ERROR; }
}
