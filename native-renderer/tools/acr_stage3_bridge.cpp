#include "../src/acr_stage3_dispatch.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

imprint::acr::Stage3Prepared make_prepared(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t input_stride,
    std::uint32_t rect_top,
    std::uint32_t rect_left,
    std::uint32_t rect_bottom,
    std::uint32_t rect_right,
    const std::uint8_t *pattern,
    std::uint32_t plane_count,
    const std::uint8_t *plane_map,
    std::int32_t pair_x,
    std::int32_t pair_y,
    std::uint32_t mode,
    std::uint32_t mosaic_7c,
    std::int32_t offset_80,
    std::int32_t offset_84,
    int task_transform) {
    imprint::acr::Stage3Prepared prepared;
    prepared.width = width;
    prepared.height = height;
    prepared.input_stride = input_stride;
    prepared.rect_top = static_cast<std::int32_t>(rect_top);
    prepared.rect_left = static_cast<std::int32_t>(rect_left);
    prepared.rect_bottom = static_cast<std::int32_t>(rect_bottom);
    prepared.rect_right = static_cast<std::int32_t>(rect_right);
    prepared.plane_count = static_cast<std::uint8_t>(plane_count);
    prepared.pair_x = pair_x;
    prepared.pair_y = pair_y;
    prepared.mode = mode;
    prepared.mosaic_7c = mosaic_7c;
    prepared.offset_80 = offset_80;
    prepared.offset_84 = offset_84;
    prepared.task_transform = task_transform != 0;
    if (pattern != nullptr) {
        std::memcpy(prepared.cfa_pattern, pattern, 4);
    }
    if (plane_map != nullptr) {
        std::memcpy(prepared.plane_map, plane_map, 3);
    }
    return prepared;
}

}  // namespace

extern "C" int acr_test_stage3_dispatch(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t input_stride,
    std::uint32_t rect_top,
    std::uint32_t rect_left,
    std::uint32_t rect_bottom,
    std::uint32_t rect_right,
    const std::uint8_t *pattern,
    std::uint32_t plane_count,
    const std::uint8_t *plane_map,
    std::int32_t pair_x,
    std::int32_t pair_y,
    std::uint32_t mode,
    std::uint32_t mosaic_7c,
    std::int32_t offset_80,
    std::int32_t offset_84,
    int task_transform,
    std::int32_t *out_branch,
    std::int32_t *out_kernel,
    std::uint64_t *out_width,
    std::uint64_t *out_height,
    std::uint64_t *out_samples,
    std::int32_t *out_half_pixel) noexcept {
    if (out_branch == nullptr || out_kernel == nullptr || out_width == nullptr ||
        out_height == nullptr || out_samples == nullptr ||
        out_half_pixel == nullptr) {
        return static_cast<int>(imprint::acr::Stage3Status::invalid_argument);
    }
    const imprint::acr::Stage3Prepared prepared = make_prepared(
        width, height, input_stride, rect_top, rect_left, rect_bottom,
        rect_right, pattern, plane_count, plane_map, pair_x, pair_y, mode,
        mosaic_7c, offset_80, offset_84, task_transform);
    const imprint::acr::Stage3Dispatch dispatch =
        imprint::acr::dispatch_stage3(prepared);
    *out_branch = static_cast<std::int32_t>(dispatch.branch);
    *out_kernel = static_cast<std::int32_t>(dispatch.kernel);
    *out_width = dispatch.output_width;
    *out_height = dispatch.output_height;
    *out_samples = dispatch.output_samples;
    *out_half_pixel = dispatch.half_pixel_shift ? 1 : 0;
    return static_cast<int>(dispatch.status);
}

extern "C" int acr_test_stage3_run_2x2(
    const std::uint16_t *input,
    std::uint64_t input_sample_count,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t input_stride,
    std::uint32_t rect_top,
    std::uint32_t rect_left,
    std::uint32_t rect_bottom,
    std::uint32_t rect_right,
    const std::uint8_t *pattern,
    std::uint32_t plane_count,
    const std::uint8_t *plane_map,
    std::int32_t pair_x,
    std::int32_t pair_y,
    std::uint32_t mode,
    std::uint32_t mosaic_7c,
    std::int32_t offset_80,
    std::int32_t offset_84,
    int task_transform,
    std::uint16_t *output,
    std::uint64_t output_sample_count) noexcept {
    if (input == nullptr || output == nullptr) {
        return static_cast<int>(imprint::acr::Stage3Status::invalid_argument);
    }
    const imprint::acr::Stage3Prepared prepared = make_prepared(
        width, height, input_stride, rect_top, rect_left, rect_bottom,
        rect_right, pattern, plane_count, plane_map, pair_x, pair_y, mode,
        mosaic_7c, offset_80, offset_84, task_transform);
    std::vector<std::uint16_t> result;
    const imprint::acr::Stage3Status status =
        imprint::acr::run_stage3_bayer_2x2(
            input, static_cast<std::size_t>(input_sample_count), prepared,
            &result);
    if (status != imprint::acr::Stage3Status::ok) {
        return static_cast<int>(status);
    }
    if (output_sample_count < result.size()) {
        return static_cast<int>(imprint::acr::Stage3Status::buffer_too_small);
    }
    std::memcpy(output, result.data(), result.size() * sizeof(result[0]));
    return static_cast<int>(status);
}
