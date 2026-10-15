#include "imprint_renderer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

int fail(const char *message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

#define REQUIRE(condition, message) do { if (!(condition)) return fail(message); } while (0)

im_shared_buffer *make_buffer(im_renderer *renderer, size_t bytes) {
    im_shared_buffer *buffer = nullptr;
    if (im_renderer_create_shared_buffer(renderer, bytes, &buffer) != IM_STATUS_OK) return nullptr;
    return buffer;
}

void fill_rgb16(uint16_t *pixels, size_t values, unsigned seed) {
    for (size_t i = 0; i < values; ++i) {
        pixels[i] = static_cast<uint16_t>((i * 977u + seed * 7919u + 113u) & 0xffffu);
    }
}

} // namespace

int main() {
    im_renderer *renderer = nullptr;
    const im_status create_status = im_renderer_create(IM_BACKEND_METAL, &renderer);
    if (create_status == IM_STATUS_BACKEND_UNAVAILABLE) return 77;
    REQUIRE(create_status == IM_STATUS_OK && renderer, "create Metal renderer");
    REQUIRE(im_renderer_supports_camera_profile_render(renderer) == 1,
            "Metal profile pipeline is available");
    REQUIRE(im_renderer_supports_camera_profile_transfer(renderer) == 1,
            "Metal transfer pipeline is available");
    im_shared_buffer *invalid_owner = reinterpret_cast<im_shared_buffer *>(uintptr_t{1});
    REQUIRE(im_renderer_create_shared_buffer(renderer, 0, &invalid_owner) == IM_STATUS_INVALID_ARGUMENT &&
                invalid_owner == nullptr,
            "zero-size allocation is rejected and leaves its output null");
    invalid_owner = reinterpret_cast<im_shared_buffer *>(uintptr_t{1});
    REQUIRE(im_renderer_create_shared_buffer(renderer, (size_t{1} << 30) + 1, &invalid_owner) ==
                IM_STATUS_INVALID_ARGUMENT && invalid_owner == nullptr,
            "oversized allocation is rejected and leaves its output null");

    constexpr uint32_t width = 19, height = 13;
    const size_t values = static_cast<size_t>(width) * height * 3;
    const size_t rgb16_bytes = values * sizeof(uint16_t);
    im_shared_buffer *processed = make_buffer(renderer, rgb16_bytes);
    im_shared_buffer *transfer_a = make_buffer(renderer, rgb16_bytes);
    im_shared_buffer *transfer_b = make_buffer(renderer, rgb16_bytes);
    im_shared_buffer *wrong_transfer = make_buffer(renderer, rgb16_bytes - sizeof(uint16_t));
    REQUIRE(processed && transfer_a && transfer_b && wrong_transfer,
            "allocate shared RGB16 transfer buffers");
    auto *processed_pixels = static_cast<uint16_t *>(im_shared_buffer_data(processed));
    auto *transfer_a_pixels = static_cast<uint16_t *>(im_shared_buffer_data(transfer_a));
    auto *transfer_b_pixels = static_cast<uint16_t *>(im_shared_buffer_data(transfer_b));
    REQUIRE(processed_pixels && transfer_a_pixels && transfer_b_pixels,
            "shared RGB16 buffer data is CPU accessible");
    REQUIRE(im_shared_buffer_size(processed) == rgb16_bytes,
            "shared buffer size reports bytes");

    std::vector<uint16_t> camera(values), reference(values), copy_processed(values), copy_output(values);
    fill_rgb16(camera.data(), values, 1);
    fill_rgb16(reference.data(), values, 2);
    fill_rgb16(copy_processed.data(), values, 3);
    std::memcpy(processed_pixels, copy_processed.data(), rgb16_bytes);
    REQUIRE(im_renderer_set_camera_profile_transfer_source(
                renderer, width, height, camera.data(), reference.data(), values) == IM_STATUS_OK,
            "upload transfer calibration sources");
    const float matrix[9] = {1.02f, 0.01f, -0.01f, -0.02f, 0.99f, 0.03f, 0.01f, -0.02f, 1.01f};
    REQUIRE(im_renderer_render_camera_profile_transfer(
                renderer, copy_processed.data(), values, matrix, 9,
                copy_output.data(), values) == IM_STATUS_OK,
            "copy ABI transfer baseline succeeds");
    REQUIRE(im_renderer_render_camera_profile_transfer_shared(
                renderer, processed, matrix, 9, transfer_a) == IM_STATUS_OK,
            "shared transfer succeeds");
    REQUIRE(std::memcmp(copy_output.data(), transfer_a_pixels, rgb16_bytes) == 0,
            "shared transfer is pixel-identical to copy ABI");
    std::vector<uint16_t> first_result(transfer_a_pixels, transfer_a_pixels + values);
    REQUIRE(im_renderer_render_camera_profile_transfer_shared(
                renderer, processed, matrix, 9, transfer_a) == IM_STATUS_OK,
            "re-render into the same explicitly supplied owner succeeds");
    REQUIRE(std::memcmp(first_result.data(), transfer_a_pixels, rgb16_bytes) == 0,
            "a repeated identical render preserves the shared output");
    fill_rgb16(processed_pixels, values, 8);
    REQUIRE(im_renderer_render_camera_profile_transfer_shared(
                renderer, processed, matrix, 9, transfer_b) == IM_STATUS_OK,
            "subsequent render into another owner succeeds");
    REQUIRE(std::memcmp(first_result.data(), transfer_a_pixels, rgb16_bytes) == 0,
            "a later render does not mutate the prior output owner");
    REQUIRE(im_renderer_render_camera_profile_transfer_shared(
                renderer, processed, matrix, 9, wrong_transfer) == IM_STATUS_INVALID_ARGUMENT,
            "transfer rejects a wrong-sized shared destination");
    REQUIRE(im_renderer_render_camera_profile_transfer_shared(
                renderer, processed, matrix, 9, processed) == IM_STATUS_INVALID_ARGUMENT,
            "transfer rejects input/output aliasing");
    float invalid_matrix[9];
    std::memcpy(invalid_matrix, matrix, sizeof(matrix));
    invalid_matrix[4] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(im_renderer_render_camera_profile_transfer_shared(
                renderer, processed, invalid_matrix, 9, transfer_b) == IM_STATUS_INVALID_ARGUMENT,
            "transfer rejects a non-finite matrix");

    constexpr uint32_t hue_count = 24, saturation_count = 6, value_count = 5;
    const size_t look_values = static_cast<size_t>(hue_count) * saturation_count * value_count * 3;
    float constants[22] = {
        1.62f, -0.34f, -0.11f, -0.12f, 1.24f, -0.08f, -0.05f, -0.29f, 1.44f,
        0.92f, 0.98f, 1.0f, 1.0f,
        1.45f, -0.22f, -0.10f, -0.14f, 1.32f, -0.12f, -0.03f, -0.18f, 1.21f};
    std::vector<float> look(look_values, 0.0f);
    for (size_t entry = 0; entry < look_values / 3; ++entry) {
        look[entry * 3 + 1] = 1.0f;
        look[entry * 3 + 2] = 1.0f;
    }
    const std::vector<float> tone = {0.0f, 0.0f, 0.18f, 0.12f, 0.43f, 0.39f,
                                     0.72f, 0.78f, 1.0f, 1.0f};
    im_shared_buffer *profile_source = make_buffer(renderer, rgb16_bytes);
    im_shared_buffer *profile_output = make_buffer(renderer, values);
    im_shared_buffer *wrong_profile_output = make_buffer(renderer, values - 1);
    im_shared_buffer *wrong_profile_source = make_buffer(renderer, rgb16_bytes - sizeof(uint16_t));
    REQUIRE(profile_source && profile_output && wrong_profile_output && wrong_profile_source,
            "allocate shared camera-profile buffers");
    REQUIRE(im_renderer_set_camera_profile_source_shared(
                renderer, width, height, wrong_profile_source) == IM_STATUS_INVALID_ARGUMENT,
            "profile source upload rejects a wrong-sized owner");
    fill_rgb16(static_cast<uint16_t *>(im_shared_buffer_data(profile_source)), values, 11);
    REQUIRE(im_renderer_set_camera_profile_source_shared(
                renderer, width, height, profile_source) == IM_STATUS_OK,
            "retain shared profile source");
    im_shared_buffer_destroy(profile_source);
    profile_source = nullptr;
    std::vector<uint8_t> profile_copy(values);
    REQUIRE(im_renderer_render_camera_profile(
                renderer, constants, 22, look.data(), look.size(), hue_count,
                saturation_count, value_count, 0, tone.data(), tone.size(),
                profile_copy.data(), profile_copy.size()) == IM_STATUS_OK,
            "copy profile ABI renders retained shared source");
    REQUIRE(im_renderer_render_camera_profile_shared(
                renderer, constants, 22, look.data(), look.size(), hue_count,
                saturation_count, value_count, 0, tone.data(), tone.size(),
                profile_output) == IM_STATUS_OK,
            "shared profile render succeeds");
    REQUIRE(std::memcmp(profile_copy.data(), im_shared_buffer_data(profile_output), values) == 0,
            "shared profile output is pixel-identical to copy ABI");
    REQUIRE(im_renderer_render_camera_profile_shared(
                renderer, constants, 22, look.data(), look.size(), hue_count,
                saturation_count, value_count, 0, tone.data(), tone.size(),
                wrong_profile_output) == IM_STATUS_INVALID_ARGUMENT,
            "profile render rejects a wrong-sized shared destination");
    std::memcpy(im_shared_buffer_data(profile_output), constants, sizeof(constants));
    REQUIRE(im_renderer_render_camera_profile_shared(
                renderer, static_cast<float *>(im_shared_buffer_data(profile_output)), 22,
                look.data(), look.size(), hue_count, saturation_count, value_count, 0,
                tone.data(), tone.size(), profile_output) == IM_STATUS_INVALID_ARGUMENT,
            "profile render rejects destination/parameter aliasing");

    im_shared_buffer *survivor = make_buffer(renderer, 32);
    REQUIRE(survivor, "allocate renderer-independent owner");
    auto *survivor_bytes = static_cast<uint8_t *>(im_shared_buffer_data(survivor));
    for (size_t i = 0; i < 32; ++i) survivor_bytes[i] = static_cast<uint8_t>(i * 7u);
    im_renderer_destroy(renderer);
    renderer = nullptr;
    for (size_t i = 0; i < 32; ++i) {
        REQUIRE(static_cast<const uint8_t *>(im_shared_buffer_data(survivor))[i] ==
                    static_cast<uint8_t>(i * 7u),
                "shared owner remains CPU-readable after renderer destruction");
    }

    im_shared_buffer_destroy(survivor);
    im_shared_buffer_destroy(processed);
    im_shared_buffer_destroy(transfer_a);
    im_shared_buffer_destroy(transfer_b);
    im_shared_buffer_destroy(wrong_transfer);
    im_shared_buffer_destroy(profile_output);
    im_shared_buffer_destroy(wrong_profile_output);
    im_shared_buffer_destroy(wrong_profile_source);
    std::puts("PASS: Metal shared buffer ABI contract");
    return 0;
}
