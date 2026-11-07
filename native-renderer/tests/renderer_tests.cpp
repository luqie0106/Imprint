#include "imprint_renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace {

int fail(const char *message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

#define REQUIRE(condition, message) do { if (!(condition)) return fail(message); } while (0)

int cpu_contracts() {
    const std::array<uint8_t, 6> source8{0, 127, 255, 1, 128, 254};
    const auto original8 = source8;
    std::array<uint16_t, 7> expanded{};
    expanded.back() = 12345;
    REQUIRE(im_native_expand_rgb8_to_rgb16(2, 1, source8.data(), 6,
                expanded.data(), 6) == IM_STATUS_OK, "RGB8 expansion succeeds");
    for (size_t index = 0; index < 6; ++index) {
        REQUIRE(expanded[index] == source8[index] * 257u, "RGB8 expansion is exact");
    }
    REQUIRE(expanded.back() == 12345 && source8 == original8,
            "conversion preserves source and destination canary");
    std::array<uint8_t, 6> compacted{};
    REQUIRE(im_native_compact_rgb16_to_rgb8(2, 1, expanded.data(), 6,
                compacted.data(), 6) == IM_STATUS_OK && compacted == source8,
            "RGB8 expansion and compaction round trip");
    std::array<float, 6> linear{};
    REQUIRE(im_native_rgb16_to_linear_float(2, 1, expanded.data(), 6,
                linear.data(), 6) == IM_STATUS_OK, "RGB16 normalization succeeds");
    REQUIRE(linear[0] == 0.0f && linear[2] == 1.0f, "normalization retains range endpoints");
    std::array<uint16_t, 6> quantized{};
    REQUIRE(im_native_linear_float_to_rgb16(2, 1, linear.data(), 6,
                quantized.data(), 6) == IM_STATUS_OK &&
                std::equal(quantized.begin(), quantized.end(), expanded.begin()),
            "RGB16 normalization and quantization round trip");
    quantized.fill(42);
    REQUIRE(im_native_linear_float_to_rgb16(2, 1, linear.data(), 6,
                quantized.data(), 5) == IM_STATUS_INVALID_ARGUMENT,
            "conversion rejects undersized destination");
    REQUIRE(std::all_of(quantized.begin(), quantized.end(), [](uint16_t v) { return v == 42; }),
            "rejected conversion does not write destination");
    REQUIRE(im_native_float_range_valid(linear.data(), 6, 0, 1) == 1,
            "finite normalized values pass validation");
    linear[5] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(im_native_float_range_valid(linear.data(), 6, 0, 1) == 0,
            "NaN fails range validation");
    linear[5] = std::numeric_limits<float>::infinity();
    REQUIRE(im_native_linear_float_to_rgb16(2, 1, linear.data(), 6,
                quantized.data(), 6) == IM_STATUS_INVALID_ARGUMENT,
            "non-finite quantization source is rejected");

    const im_basic_params neutral{};
    REQUIRE(im_native_basic_v2_rgb8(2, 1, source8.data(), 6, &neutral,
                compacted.data(), 6) == IM_STATUS_OK && compacted == source8,
            "neutral basic RGB8 operator is an identity");
    REQUIRE(im_native_basic_v2_rgb16(2, 1, expanded.data(), 6, &neutral,
                quantized.data(), 6) == IM_STATUS_OK &&
                std::equal(quantized.begin(), quantized.end(), expanded.begin()),
            "neutral basic RGB16 operator is an identity");
    im_basic_params invalid = neutral;
    invalid.exposure = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(im_native_basic_v2_rgb8(2, 1, source8.data(), 6, &invalid,
                compacted.data(), 6) == IM_STATUS_INVALID_ARGUMENT,
            "basic operator rejects non-finite controls");

    const std::array<float, 6> source{0, 0.2f, 1, 0.1f, 0.5f, 0.9f};
    const auto original = source;
    const float transmission[]{0.5f, 1.0f};
    const float atmosphere[]{0.7f, 0.8f, 0.9f};
    const im_dehaze_params no_dehaze{};
    std::array<float, 7> physical{};
    physical.back() = 42;
    REQUIRE(im_native_physical_float_run(2, 1, source.data(), 6, transmission, 2,
                atmosphere, &no_dehaze, physical.data(), 7) == IM_STATUS_OK,
            "zero-strength physical dehaze succeeds");
    for (size_t index = 0; index < source.size(); ++index) {
        // 浮点内核会经过亮度/色度重组，允许一次 float32 运算的舍入误差。
        REQUIRE(std::abs(source[index] - physical[index]) <= 1e-7f,
                "zero-strength float kernel stays within float32 rounding error");
    }
    REQUIRE(physical.back() == 42 && source == original,
            "physical dehaze preserves source and destination boundaries");
    REQUIRE(im_native_dehaze_spatial_run(expanded.data(), 2, 1, transmission, 2,
                atmosphere, &no_dehaze, quantized.data(), 6) == IM_STATUS_OK &&
                std::equal(quantized.begin(), quantized.end(), expanded.begin()),
            "zero-strength RGB16 dehaze is an exact identity");
    const float invalid_transmission[]{0.5f, std::numeric_limits<float>::infinity()};
    REQUIRE(im_native_physical_float_run(2, 1, source.data(), 6, invalid_transmission, 2,
                atmosphere, &no_dehaze, physical.data(), 7) == IM_STATUS_INVALID_ARGUMENT,
            "physical dehaze rejects non-finite transmission");
    REQUIRE(im_native_physical_float_run(2, 1, source.data(), 6, transmission, 1,
                atmosphere, &no_dehaze, physical.data(), 7) == IM_STATUS_INVALID_ARGUMENT,
            "physical dehaze validates transmission shape");

    const float lut[]{0, 0, 0, 0, 1, 1, 1, 1};
    const float corners[]{0, 0, 0, 1, 1, 1};
    float sampled[]{-1, -1, 42};
    REQUIRE(im_native_scalar_lut_run(2, 1, corners, 6, lut, 8, 2, sampled, 3) == IM_STATUS_OK &&
                sampled[0] == 0 && sampled[1] == 1 && sampled[2] == 42,
            "scalar LUT samples corners without overrunning destination");
    const float identity[]{1, 0, 0, 0, 1, 0, 0, 0, 1};
    REQUIRE(im_native_camera_profile_transfer_rgb16(2, 1, expanded.data(), 6,
                expanded.data(), 6, expanded.data(), 6, identity, 9,
                quantized.data(), 6) == IM_STATUS_OK &&
                std::equal(quantized.begin(), quantized.end(), expanded.begin()),
            "unchanged reference transfer preserves camera samples");
    REQUIRE(im_renderer_create(IM_BACKEND_AUTO, nullptr) == IM_STATUS_INVALID_ARGUMENT,
            "renderer creation rejects a null output handle");
    REQUIRE(im_renderer_upload_preview_image(nullptr, 2, 1, expanded.data(), 6) ==
                IM_STATUS_INVALID_ARGUMENT, "preview upload rejects a null renderer");
    std::puts("PASS: CPU conversions, basic controls, physical dehaze, LUT and transfer ABI");
    return 0;
}

int gpu_contracts() {
    im_renderer *raw = nullptr;
    const im_status status = im_renderer_create(IM_BACKEND_AUTO, &raw);
    if (status == IM_STATUS_BACKEND_UNAVAILABLE) {
        std::printf("SKIP: GPU unavailable: %s\n", im_renderer_last_error(nullptr));
        return 77;
    }
    REQUIRE(status == IM_STATUS_OK && raw, "GPU renderer creation succeeds");
    std::unique_ptr<im_renderer, decltype(&im_renderer_destroy)> renderer(raw, im_renderer_destroy);
    const im_dehaze_params dehaze{};
    const im_basic_params basic{};
    const im_filter_params filter{0, 0, 0, 1, 1, 0, 0, 0};
    REQUIRE(im_renderer_upload_filter(raw, &filter, nullptr, 0, nullptr, 0, 0) == IM_STATUS_OK,
            "neutral GPU filter upload succeeds");
    REQUIRE(im_renderer_render(raw, IM_RENDER_L0, &dehaze, &basic) == IM_STATUS_NOT_READY,
            "GPU render requires an uploaded source");
    std::vector<uint16_t> source(17 * 9 * 3);
    for (size_t index = 0; index < source.size(); ++index) source[index] = (index * 379) % 65536;
    const auto original = source;
    REQUIRE(im_renderer_upload_preview_image(raw, 17, 9, source.data(), source.size() - 1) ==
                IM_STATUS_INVALID_ARGUMENT, "GPU upload validates exact sample count");
    REQUIRE(im_renderer_upload_preview_image(raw, 17, 9, source.data(), source.size()) == IM_STATUS_OK,
            "GPU preview upload succeeds");
    const im_render_level levels[]{IM_RENDER_L0, IM_RENDER_L1, IM_RENDER_L2};
    const uint32_t widths[]{17, 9, 5};
    const uint32_t heights[]{9, 5, 3};
    for (size_t index = 0; index < 3; ++index) {
        REQUIRE(im_renderer_render(raw, levels[index], &dehaze, &basic) == IM_STATUS_OK,
                "GPU cached pyramid level renders");
        uint32_t width = 0, height = 0;
        size_t samples = 0;
        REQUIRE(im_renderer_get_output_size(raw, &width, &height, &samples) == IM_STATUS_OK &&
                    width == widths[index] && height == heights[index] && samples == width * height * 3,
                "GPU pyramid output dimensions are correct");
        std::vector<uint16_t> output(samples + 1, 12345);
        REQUIRE(im_renderer_copy_output(raw, output.data(), samples - 1) == IM_STATUS_BUFFER_TOO_SMALL,
                "GPU readback validates destination capacity");
        REQUIRE(im_renderer_copy_output(raw, output.data(), samples) == IM_STATUS_OK &&
                    output.back() == 12345, "GPU readback preserves destination canary");
        if (index == 0) REQUIRE(std::equal(source.begin(), source.end(), output.begin()),
                               "neutral GPU rendering is an exact identity");
    }
    REQUIRE(source == original, "GPU rendering never modifies source samples");
    std::printf("PASS: GPU preview pyramid and readback; backend=%s\n", im_renderer_backend_name(raw));
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    const int status = cpu_contracts();
    if (status != 0 || (argc == 2 && std::strcmp(argv[1], "--abi-only") == 0)) return status;
    return gpu_contracts();
}
