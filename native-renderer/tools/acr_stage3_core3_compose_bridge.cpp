// Validation seam only; all image computation remains in the native kernels.
#include "../src/acr_stage3_core3_compose.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {
template <typename Sample>
int compose_bridge(
    const Sample *source,
    std::size_t source_sample_count,
    std::size_t width,
    std::size_t height,
    std::size_t input_row_stride,
    const float *green_parameters,
    const float *refine_parameters,
    std::uint32_t row_origin_parity,
    std::uint32_t column_origin_parity,
    float *output_rgb,
    std::size_t output_float_capacity,
    std::uint16_t *output_rgb_u16,
    std::size_t output_u16_capacity,
    std::size_t *metadata,
    std::size_t metadata_capacity) noexcept {
    using namespace imprint::acr;
    if (green_parameters == nullptr || refine_parameters == nullptr ||
        output_rgb == nullptr || output_rgb_u16 == nullptr || metadata == nullptr) {
        return static_cast<int>(Core3ComposeStatus::invalid_argument);
    }

    Core3ComposeControls controls{};
    controls.green = {green_parameters[0], green_parameters[1],
                      green_parameters[2], green_parameters[3]};
    controls.lower = refine_parameters[0];
    controls.upper = refine_parameters[1];
    controls.row_origin_parity = row_origin_parity;
    controls.column_origin_parity = column_origin_parity;

    Core3ComposeOutput result{};
    Core3ComposeStatus status;
    if constexpr (std::is_same_v<Sample, float>) {
        status = run_core3_prepared_f32_tile(
            source, source_sample_count, width, height, input_row_stride,
            controls, &result);
    } else {
        status = run_core3_prepared_u16_tile(
            source, source_sample_count, width, height, input_row_stride,
            controls, &result);
    }
    if (status != Core3ComposeStatus::ok) return static_cast<int>(status);

    const std::size_t sample_count = result.rgb.size();
    constexpr std::size_t kMetadataCount = 9;
    if (output_float_capacity < sample_count ||
        output_u16_capacity < result.rgb_u16.size() ||
        metadata_capacity < kMetadataCount) {
        return static_cast<int>(Core3ComposeStatus::buffer_too_small);
    }

    std::copy(result.rgb.begin(), result.rgb.end(), output_rgb);
    std::copy(result.rgb_u16.begin(), result.rgb_u16.end(), output_rgb_u16);
    const std::size_t values[kMetadataCount] = {
        result.width,
        result.height,
        result.valid_region.top,
        result.valid_region.left,
        result.valid_region.bottom,
        result.valid_region.right,
        result.output_width,
        result.output_height,
        sample_count,
    };
    std::copy(values, values + kMetadataCount, metadata);
    return static_cast<int>(Core3ComposeStatus::ok);
}
}  // namespace

extern "C" int acr_test_core3_compose(
    const std::uint16_t *source, std::size_t source_sample_count,
    std::size_t width, std::size_t height, std::size_t input_row_stride,
    const float *green_parameters, const float *refine_parameters,
    std::uint32_t row_origin_parity, std::uint32_t column_origin_parity,
    float *output_rgb, std::size_t output_float_capacity,
    std::uint16_t *output_rgb_u16, std::size_t output_u16_capacity,
    std::size_t *metadata, std::size_t metadata_capacity) noexcept {
    return compose_bridge(source, source_sample_count, width, height,
        input_row_stride, green_parameters, refine_parameters,
        row_origin_parity, column_origin_parity, output_rgb,
        output_float_capacity, output_rgb_u16, output_u16_capacity,
        metadata, metadata_capacity);
}

extern "C" int acr_test_core3_compose_f32(
    const float *source, std::size_t source_sample_count,
    std::size_t width, std::size_t height, std::size_t input_row_stride,
    const float *green_parameters, const float *refine_parameters,
    std::uint32_t row_origin_parity, std::uint32_t column_origin_parity,
    float *output_rgb, std::size_t output_float_capacity,
    std::uint16_t *output_rgb_u16, std::size_t output_u16_capacity,
    std::size_t *metadata, std::size_t metadata_capacity) noexcept {
    return compose_bridge(source, source_sample_count, width, height,
        input_row_stride, green_parameters, refine_parameters,
        row_origin_parity, column_origin_parity, output_rgb,
        output_float_capacity, output_rgb_u16, output_u16_capacity,
        metadata, metadata_capacity);
}

extern "C" int acr_test_core3_controls(
    double amount, int suppress_refine,
    std::uint32_t row_origin_parity,
    std::uint32_t column_origin_parity,
    float *output, std::size_t output_capacity) noexcept {
    using imprint::acr::Core3ComposeControls;
    using imprint::acr::Core3ComposeStatus;
    if (output == nullptr || output_capacity < 6u ||
        reinterpret_cast<std::uintptr_t>(output) % alignof(float) != 0u ||
        6u * sizeof(float) > std::numeric_limits<std::uintptr_t>::max() -
                                 reinterpret_cast<std::uintptr_t>(output) ||
        (suppress_refine != 0 && suppress_refine != 1)) {
        return static_cast<int>(Core3ComposeStatus::invalid_argument);
    }

    Core3ComposeControls controls{};
    const Core3ComposeStatus status = prepare_core3_controls_007f2124(
        amount, suppress_refine != 0, row_origin_parity,
        column_origin_parity, &controls);
    if (status != Core3ComposeStatus::ok) return static_cast<int>(status);

    const float values[6] = {controls.green.t, controls.green.k,
        controls.green.m, controls.green.q, controls.lower, controls.upper};
    std::copy(values, values + 6, output);
    return static_cast<int>(Core3ComposeStatus::ok);
}
