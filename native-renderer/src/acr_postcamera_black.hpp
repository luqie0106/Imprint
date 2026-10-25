#pragma once

#include "acr_float_lut.hpp"
#include "acr_u16_lut.hpp"

#include <cstddef>

namespace imprint::acr {

enum class PostcameraBlackStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    buffer_too_small = 4,
    unsupported_alignment = 5,
    overlap = 6,
    arithmetic_overflow = 7,
    resource_limit = 8,
    non_finite_input = 9,
    invalid_black = 10,
    out_of_domain = 11,
    non_finite_intermediate = 12,
    allocation_failure = 13,
    length_error = 14,
};

// Builds the explicit float32 post-camera black LUT for finite 0 <= b < 1.
// The builder uses the recovered fixed 4096-segment sampler and endpoint
// guard from acr_float_lut. This is a mode2 component only; it does not claim
// the actual RAW-pipeline format binding or model implicit context state.
PostcameraBlackStatus build_postcamera_black_lut(
    double b, FloatLutPrepared* out) noexcept;

// Applies the explicit unsigned float32 LUT to active rows. Counts and strides
// are measured in float samples. All active inputs must be finite and in
// [0,1]. b==0 is a bit-preserving bypass. Exact in-place operation requires
// identical base pointer, count, and row stride; all other declared-span
// overlaps are rejected. Only active samples are written, so row padding is
// preserved. Validation and compact candidate computation finish before any
// destination write. This API does not select a format for actual RAW renders.
PostcameraBlackStatus apply_postcamera_black_prepared_f32(
    const float* src, std::size_t srcCount, std::size_t srcRowStride,
    std::size_t height, std::size_t rowSamples, double b, float* dst,
    std::size_t dstCount, std::size_t dstRowStride) noexcept;

// Builds the explicit mode1 unsigned uint16 table. It first builds the
// recovered 4098-entry float table, then converts it with 01300600 into the
// complete 65536-entry LUT. The type8 provider and dynamic-provider paths are
// not implemented here.
PostcameraBlackStatus build_postcamera_black_u16_lut(
    double b, U16LutPrepared* out) noexcept;

// Applies the explicit mode1 unsigned uint16 LUT to active rows. Counts and
// strides are measured in uint16 samples. Every uint16 input is valid. b==0
// copies active bytes unchanged; otherwise the complete 65536-entry table is
// used. Exact in-place operation requires identical base pointer, count, and
// row stride; other declared-span overlaps are rejected. Padding is preserved
// and destination writes start only after the full compact candidate is ready.
// This component does not claim actual RAW request policy or adjacent
// conversion selection.
PostcameraBlackStatus apply_postcamera_black_prepared_u16(
    const std::uint16_t* src, std::size_t srcCount,
    std::size_t srcRowStride, std::size_t height, std::size_t rowSamples,
    double b, std::uint16_t* dst, std::size_t dstCount,
    std::size_t dstRowStride) noexcept;

}  // namespace imprint::acr
