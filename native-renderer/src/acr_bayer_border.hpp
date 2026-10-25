#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

struct BayerRepeatRect {
    std::int32_t top;
    std::int32_t left;
    std::int32_t bottom;
    std::int32_t right;
};

enum class BayerBorderStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_parameters = 2,
    overlap = 3,
    unsupported_domain = 4,
};

// Copies a Bayer mosaic region using the recovered two-sample edge period.
// `source` points at source_rect's top-left sample; strides and counts are in
// uint16 samples. The request may extend by any amount beyond source_rect.
// Source rectangles smaller than 2x2 and dimensions above INT32_MAX are
// rejected rather than approximated with a clamp or a one-sample fallback.
// Each declared sample capacity must convert to at most PTRDIFF_MAX bytes and
// its base-plus-capacity byte address must not wrap uintptr_t. This capacity
// representability check does not widen overlap detection: overlap is tested
// only across each required access span (through the last row's active width),
// including padding between rows.
// The source and destination spans, including inter-row padding, must not
// overlap. Only request_rect's width samples are written on each destination
// row; destination row padding is preserved. Every failure leaves destination
// unchanged.
BayerBorderStatus copy_bayer_repeat2_u16(
    const std::uint16_t *source,
    std::size_t source_sample_count,
    std::size_t source_row_stride,
    BayerRepeatRect source_rect,
    BayerRepeatRect request_rect,
    std::uint16_t *destination,
    std::size_t destination_sample_count,
    std::size_t destination_row_stride) noexcept;

}  // namespace imprint::acr
