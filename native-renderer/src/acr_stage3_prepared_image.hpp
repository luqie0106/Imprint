#pragma once

#include "acr_bayer_border.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

// CPU harness for composing the recovered Stage3 prepared sub-stages over an
// entire input rectangle. These are explicit prepared values: this API does
// not infer camera metadata, invoke Adobe's scheduler/host binding, or enable
// the full-resolution Stage3 dispatcher.
struct PreparedImageControls {
    double gains[3] = {1.0, 1.0, 1.0};
    std::uint16_t source_floor = 0;
    bool cubic = false;
    double amount = 1.0;
    bool suppress_refine = false;
    std::uint16_t hot_threshold = 8000;
    std::uint16_t cold_threshold = 6000;
    std::size_t tile_width = 256;
    std::size_t tile_height = 256;
};

struct PreparedImageOutput {
    std::size_t width = 0;
    std::size_t height = 0;
    // Tightly packed HWC RGB, three uint16 samples per output pixel.
    std::vector<std::uint16_t> rgb;
};

enum class PreparedImageStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_rect = 2,
    invalid_dimensions = 3,
    invalid_stride = 4,
    buffer_too_small = 5,
    invalid_controls = 6,
    unsupported_alignment = 7,
    overlap = 8,
    arithmetic_overflow = 9,
    resource_limit = 10,
    allocation_failure = 11,
    length_error = 12,
    prepare_gain_failure = 20,
    prepare_curve_failure = 21,
    build_curve_lut_failure = 22,
    build_u16_lut_failure = 23,
    prepare_core3_controls_failure = 24,
    copy_bayer_failure = 25,
    u16_to_f32_failure = 26,
    cfa_gain_failure = 27,
    f32_to_u16_failure = 28,
    forward_lut_failure = 29,
    hotpixel_failure = 30,
    core3_failure = 31,
    inverse_lut_failure = 32,
    rgb_gain_failure = 33,
    final_conversion_failure = 34,
};

// Runs the explicit prepared CPU composition over source_rect. source points
// to source_rect's top-left uint16 sample; stride and capacity are samples.
// The source rectangle dimensions must be at least 2x2 and at most INT32_MAX.
// A rectangle may have any representable origin, including negative odd
// coordinates. Scratch rows are padded and 16-byte aligned as required by the
// recovered row kernels. Each tile samples its halo from the full source_rect.
//
// The output is committed only after every tile succeeds. Source may refer to
// the current output->rgb storage; that storage remains alive until the new
// image has been completely rendered. This API is not the Adobe scheduler or
// host binding and does not change the dispatcher's full-resolution rejection.
PreparedImageStatus run_stage3_prepared_image_u16(
    const std::uint16_t* source, std::size_t source_sample_count,
    std::size_t source_row_stride, BayerRepeatRect source_rect,
    const PreparedImageControls& controls,
    PreparedImageOutput* output) noexcept;

// Internal to-buffer counterpart used by test/product ABI adapters. Metadata
// arrays use rect4={top,left,bottom,right} and gains3 contains three doubles.
// The complete source and destination capacities and metadata address spans
// are validated before rendering; destination may not overlap any input span.
// Pixels are copied to destination only after the vector composition succeeds.
PreparedImageStatus run_stage3_prepared_image_u16_to_buffer(
    const std::uint16_t* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const std::int32_t rect4[4],
    const double gains3[3], std::uint16_t source_floor, int cubic,
    double amount, int suppress_refine, std::uint16_t hot_threshold,
    std::uint16_t cold_threshold, std::size_t tile_width,
    std::size_t tile_height, std::uint16_t* destination,
    std::size_t destination_sample_count) noexcept;

}  // namespace imprint::acr
