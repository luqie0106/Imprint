#include "imprint_renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

int fail(const char *message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

#define REQUIRE(condition, message) do { if (!(condition)) return fail(message); } while (0)

bool read_output(im_renderer *renderer, std::vector<uint16_t> &output,
                 uint32_t &width, uint32_t &height) {
    size_t count = 0;
    if (im_renderer_get_output_size(renderer, &width, &height, &count) != IM_STATUS_OK) return false;
    output.resize(count);
    return im_renderer_copy_output(renderer, output.data(), output.size()) == IM_STATUS_OK;
}

bool any_difference(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b) {
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return true;
    return false;
}

std::vector<uint16_t> make_pixels(uint32_t width, uint32_t height) {
    std::vector<uint16_t> pixels(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
        pixels[i * 3] = static_cast<uint16_t>(5000 + (i * 379) % 50000);
        pixels[i * 3 + 1] = static_cast<uint16_t>(7000 + (i * 719) % 45000);
        pixels[i * 3 + 2] = static_cast<uint16_t>(2000 + (i * 911) % 55000);
    }
    return pixels;
}

int run_camera_profile_contract(im_renderer *renderer) {
    REQUIRE(renderer, "camera profile test has a renderer");
    constexpr uint32_t width = 127, height = 83;
    const size_t values = static_cast<size_t>(width) * height * 3;
    REQUIRE(im_renderer_supports_camera_profile_render(renderer) == 1,
            "Metal and D3D12 expose the camera profile GPU operator");

    std::vector<uint16_t> source(values);
    std::mt19937 random(0xC4A3E17u);
    std::uniform_int_distribution<uint32_t> sample(0, 65535);
    for (uint16_t &value : source) value = static_cast<uint16_t>(sample(random));
    source[0] = 0;
    source[1] = 65535;
    source[2] = 32768;

    float constants[22] = {
        1.62f, -0.34f, -0.11f, -0.12f, 1.24f, -0.08f, -0.05f, -0.29f, 1.44f,
        0.92f, 0.98f, 1.0f, 1.0f,
        1.45f, -0.22f, -0.10f, -0.14f, 1.32f, -0.12f, -0.03f, -0.18f, 1.21f};
    const uint32_t hue_count = 24, saturation_count = 6, value_count = 5;
    const size_t look_values = static_cast<size_t>(hue_count) * saturation_count * value_count * 3;
    std::vector<float> look(look_values, 0.0f);
    for (size_t entry = 0; entry < look_values / 3; ++entry) {
        look[entry * 3 + 1] = 1.0f;
        look[entry * 3 + 2] = 1.0f;
    }
    std::vector<float> tone = {0.0f, 0.0f, 0.18f, 0.12f, 0.43f, 0.39f,
                               0.72f, 0.78f, 1.0f, 1.0f};
    std::vector<uint8_t> reference(values), gpu(values);

    REQUIRE(im_renderer_render_camera_profile(renderer, constants, 22, look.data(), look.size(),
                                               hue_count, saturation_count, value_count, 0,
                                               tone.data(), tone.size(), gpu.data(), gpu.size()) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile render requires an uploaded source");
    REQUIRE(im_renderer_set_camera_profile_source(renderer, width, height, source.data(), values - 1) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile source count must be exact");
    REQUIRE(im_renderer_set_camera_profile_source(renderer, width, height, source.data(), values) ==
                IM_STATUS_OK,
            "upload the camera profile RGB16 source once");
    REQUIRE(im_renderer_render_camera_profile(renderer, constants, 21, look.data(), look.size(),
                                               hue_count, saturation_count, value_count, 0,
                                               tone.data(), tone.size(), gpu.data(), gpu.size()) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile constants count must be exact");
    REQUIRE(im_renderer_render_camera_profile(renderer, constants, 22, look.data(), look.size() - 1,
                                               hue_count, saturation_count, value_count, 0,
                                               tone.data(), tone.size(), gpu.data(), gpu.size()) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile look table count must be exact");
    REQUIRE(im_renderer_render_camera_profile(renderer, constants, 22, look.data(), look.size(),
                                               hue_count, saturation_count, value_count, 2,
                                               tone.data(), tone.size(), gpu.data(), gpu.size()) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile look encoding is bounded");
    REQUIRE(im_renderer_render_camera_profile(renderer, constants, 22, look.data(), look.size(),
                                               hue_count, saturation_count, value_count, 0,
                                               tone.data(), tone.size(), gpu.data(), gpu.size() - 1) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile destination count must be exact");
    REQUIRE(im_renderer_render_camera_profile(renderer, constants, 22, look.data(), look.size(),
                                               hue_count, saturation_count, value_count, 0,
                                               tone.data(), tone.size(),
                                               reinterpret_cast<uint8_t *>(constants), values) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile destination must not alias input parameters");
    const float previous_constant = constants[0];
    constants[0] = std::numeric_limits<float>::infinity();
    REQUIRE(im_renderer_render_camera_profile(renderer, constants, 22, look.data(), look.size(),
                                               hue_count, saturation_count, value_count, 0,
                                               tone.data(), tone.size(), gpu.data(), gpu.size()) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile constants reject non-finite values");
    constants[0] = previous_constant;
    const float previous_look = look[0];
    look[0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(im_renderer_render_camera_profile(renderer, constants, 22, look.data(), look.size(),
                                               hue_count, saturation_count, value_count, 0,
                                               tone.data(), tone.size(), gpu.data(), gpu.size()) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile look table rejects non-finite values");
    look[0] = previous_look;
    const float previous_tone_x = tone[2];
    tone[2] = tone[0];
    REQUIRE(im_renderer_render_camera_profile(renderer, constants, 22, look.data(), look.size(),
                                               hue_count, saturation_count, value_count, 0,
                                               tone.data(), tone.size(), gpu.data(), gpu.size()) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile tone curve requires strictly increasing x coordinates");
    tone[2] = previous_tone_x;

    auto compare_case = [&](uint32_t encoding) -> int {
        REQUIRE(im_native_camera_profile_render_rgb16_to_rgb8(
                    width, height, source.data(), source.size(), constants, 22,
                    look.data(), look.size(), hue_count, saturation_count, value_count,
                    encoding, tone.data(), tone.size(), reference.data(), reference.size()) == IM_STATUS_OK,
                "CPU camera profile reference succeeds");
        REQUIRE(im_renderer_render_camera_profile(
                    renderer, constants, 22, look.data(), look.size(), hue_count,
                    saturation_count, value_count, encoding, tone.data(), tone.size(),
                    gpu.data(), gpu.size()) == IM_STATUS_OK,
                "GPU camera profile render succeeds");
        int max_difference = 0;
        for (size_t index = 0; index < values; ++index) {
            const int difference = std::abs(static_cast<int>(gpu[index]) -
                                            static_cast<int>(reference[index]));
            max_difference = std::max(max_difference, difference);
        }
        REQUIRE(max_difference <= 1, "GPU camera profile stays within one RGB8 code of CPU");
        return 0;
    };

    REQUIRE(compare_case(0) == 0, "linear-value identity look has RGB8 parity");
    for (unsigned iteration = 0; iteration < 8; ++iteration) {
        std::uniform_real_distribution<float> hue(-45.0f, 45.0f);
        std::uniform_real_distribution<float> modifier(0.65f, 1.4f);
        for (size_t entry = 0; entry < look_values / 3; ++entry) {
            look[entry * 3] = hue(random);
            look[entry * 3 + 1] = modifier(random);
            look[entry * 3 + 2] = modifier(random);
        }
        const float ev = -2.0f + static_cast<float>(iteration) * 0.5f;
        constants[12] = std::exp2(ev);
        if ((iteration & 1u) != 0u) {
            for (size_t i = 1; i + 1 < tone.size() / 2; ++i) {
                tone[i * 2 + 1] = std::min(1.0f, std::max(tone[(i - 1) * 2 + 1] + 0.025f,
                    tone[i * 2 + 1] + (i & 1u ? 0.03f : -0.02f)));
            }
        }
        REQUIRE(compare_case(iteration & 1u) == 0,
                "random look, tone, encoding, and exposure case has RGB8 parity");
    }

    std::fill(look.begin(), look.end(), 0.0f);
    for (size_t entry = 0; entry < look_values / 3; ++entry) {
        look[entry * 3] = entry % 2 ? -360.0f : 360.0f;
        look[entry * 3 + 1] = 0.0f;
        look[entry * 3 + 2] = entry % 3 ? 16.0f : 0.0f;
    }
    constants[12] = 4.0f;
    REQUIRE(compare_case(1) == 0, "extreme legal look modifiers remain within RGB8 parity");
    REQUIRE(im_renderer_clear_camera_profile_source(renderer) == IM_STATUS_OK,
            "clear releases camera profile source and cached GPU resources");
    REQUIRE(im_renderer_render_camera_profile(renderer, constants, 22, look.data(), look.size(),
                                               hue_count, saturation_count, value_count, 1,
                                               tone.data(), tone.size(), gpu.data(), gpu.size()) ==
                IM_STATUS_INVALID_ARGUMENT,
            "camera profile render after clear reports no uploaded source");
    return 0;
}

int run_null_handle_checks() {
    const im_dehaze_params no_dehaze{};
    const im_basic_params no_basic{};
    const im_filter_params no_filter{0, 0, 0, 1, 1, 0, 0, 0};
    REQUIRE(im_renderer_upload_preview_image(nullptr, 1, 1, nullptr, 3) == IM_STATUS_INVALID_ARGUMENT,
            "null renderer/image handles are rejected");
    REQUIRE(im_renderer_upload_filter(nullptr, &no_filter, nullptr, 0, nullptr, 0, 0) == IM_STATUS_INVALID_ARGUMENT,
            "null renderer filter uploads are rejected");
    REQUIRE(im_renderer_render(nullptr, IM_RENDER_L0, &no_dehaze, &no_basic) == IM_STATUS_INVALID_ARGUMENT,
            "null renderer render calls are rejected");
    REQUIRE(im_renderer_set_camera_profile_source(nullptr, 1, 1, nullptr, 3) == IM_STATUS_INVALID_ARGUMENT,
            "camera profile source rejects a null renderer");
    REQUIRE(im_renderer_render_camera_profile(nullptr, nullptr, 0, nullptr, 0, 0, 0, 0, 0,
                                               nullptr, 0, nullptr, 0) == IM_STATUS_INVALID_ARGUMENT,
            "camera profile render rejects a null renderer");
    REQUIRE(im_renderer_clear_camera_profile_source(nullptr) == IM_STATUS_INVALID_ARGUMENT,
            "camera profile clear rejects a null renderer");
    REQUIRE(im_renderer_warp_rectilinear_rgb16(nullptr, 1, 1, nullptr, 3,
                                                nullptr, 21, nullptr, 3) == IM_STATUS_INVALID_ARGUMENT,
            "GPU WarpRectilinear rejects a null renderer");
    REQUIRE(im_renderer_supports_warp_rectilinear(nullptr) == 0,
            "null renderer does not report WarpRectilinear support");

    const uint16_t pixels[] = {10000, 12000, 9000, 10000, 12000, 9000};
    const float transmission[] = {0.27f, 1.0f};
    const float atmosphere[] = {0.08f, 0.10f, 0.12f};
    uint16_t output[] = {0xaaaa, 0xbbbb, 0xcccc, 0xdddd, 0xeeee, 0xffff, 0x1234};
    im_dehaze_params spatial_params{};
    REQUIRE(im_native_dehaze_spatial_run(nullptr, 2, 1, transmission, 2, atmosphere,
                                          &spatial_params, output, 6) == IM_STATUS_INVALID_ARGUMENT,
            "spatial operator rejects a null image");
    REQUIRE(im_renderer_render_spatial_full(nullptr, 2, 1, pixels, 6, transmission, 2,
                                            atmosphere, &spatial_params, output, 6) == IM_STATUS_INVALID_ARGUMENT,
            "GPU spatial operator rejects a null renderer");
    REQUIRE(im_native_dehaze_spatial_run(pixels, 0, 1, transmission, 0, atmosphere,
                                          &spatial_params, output, 6) == IM_STATUS_INVALID_ARGUMENT,
            "spatial operator rejects empty dimensions");
    REQUIRE(im_native_dehaze_spatial_run(pixels, 2, 1, transmission, 1, atmosphere,
                                          &spatial_params, output, 6) == IM_STATUS_INVALID_ARGUMENT,
            "spatial operator requires one transmission per pixel");
    REQUIRE(im_native_dehaze_spatial_run(pixels, 2, 1, transmission, 2, atmosphere,
                                          &spatial_params, output, 5) == IM_STATUS_INVALID_ARGUMENT,
            "spatial operator validates destination capacity");
    const float invalid_transmission[] = {0.27f, std::numeric_limits<float>::infinity()};
    REQUIRE(im_native_dehaze_spatial_run(pixels, 2, 1, invalid_transmission, 2, atmosphere,
                                          &spatial_params, output, 6) == IM_STATUS_INVALID_ARGUMENT,
            "spatial operator rejects non-finite transmission values");
    const float invalid_atmosphere[] = {0.08f, std::numeric_limits<float>::quiet_NaN(), 0.12f};
    REQUIRE(im_native_dehaze_spatial_run(pixels, 2, 1, transmission, 2, invalid_atmosphere,
                                          &spatial_params, output, 6) == IM_STATUS_INVALID_ARGUMENT,
            "spatial operator rejects non-finite airlight values");
    spatial_params.local_contrast = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(im_native_dehaze_spatial_run(pixels, 2, 1, transmission, 2, atmosphere,
                                          &spatial_params, output, 6) == IM_STATUS_INVALID_ARGUMENT,
            "spatial operator rejects non-finite parameters");

    spatial_params = {};
    const uint16_t original[] = {10000, 12000, 9000, 65535, 0, 32768};
    uint16_t copied[6]{};
    REQUIRE(im_native_dehaze_spatial_run(original, 2, 1, transmission, 2, atmosphere,
                                          &spatial_params, copied, 6) == IM_STATUS_OK,
            "zero-strength spatial call succeeds");
    REQUIRE(std::equal(std::begin(original), std::end(original), std::begin(copied)),
            "zero-strength spatial call is an exact RGB16 copy");

    spatial_params.strength = 1.0f;
    uint16_t spatial_output[7] = {};
    spatial_output[6] = 0x4567;
    REQUIRE(im_native_dehaze_spatial_run(pixels, 2, 1, transmission, 2, atmosphere,
                                          &spatial_params, spatial_output, 7) == IM_STATUS_OK,
            "spatial operator processes a valid transmission map");
    REQUIRE(std::equal(pixels, pixels + 3, spatial_output) == false,
            "spatial dehaze changes the low-transmission pixel");
    REQUIRE(std::equal(pixels + 3, pixels + 6, spatial_output + 3),
            "unit transmission preserves the second uniform pixel");
    REQUIRE(spatial_output[6] == 0x4567, "spatial operator leaves excess destination capacity untouched");
    return 0;
}

int run_warp_cpu_contract() {
    constexpr uint32_t width = 5, height = 4;
    const size_t values = static_cast<size_t>(width) * height * 3;
    std::vector<uint16_t> source(values);
    for (size_t pixel = 0; pixel < values / 3; ++pixel) {
        const uint16_t value = static_cast<uint16_t>(pixel * 1703u);
        source[pixel * 3] = value;
        source[pixel * 3 + 1] = value;
        source[pixel * 3 + 2] = value;
    }
    std::vector<uint16_t> output(values, 0xaaaa);
    float identity[21]{};
    for (size_t channel = 0; channel < 3; ++channel) identity[channel * 6] = 1.0f;
    identity[18] = 0.5f;
    identity[19] = 0.5f;
    identity[20] = 1.0f;
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values,
                                              identity, 21, output.data(), values) == IM_STATUS_OK,
            "identity WarpRectilinear succeeds for equal RGB planes");
    REQUIRE(output == source, "identity WarpRectilinear preserves the RGB16 image exactly");

    float boundary[21]{};
    for (size_t channel = 0; channel < 3; ++channel) boundary[channel * 6] = 2.0f;
    boundary[18] = 0.5f;
    boundary[19] = 0.5f;
    boundary[20] = 1.0f;
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values,
                                              boundary, 21, output.data(), values) == IM_STATUS_OK,
            "outward WarpRectilinear succeeds");
    REQUIRE(output[(values / 3 - 1) * 3] == source[(values / 3 - 1) * 3] &&
            output[(values / 3 - 1) * 3 + 1] == source[(values / 3 - 1) * 3 + 1] &&
            output[(values / 3 - 1) * 3 + 2] == source[(values / 3 - 1) * 3 + 2],
            "WarpRectilinear clamps all channels at the bottom-right edge");

    float channel_warp[21]{};
    channel_warp[0] = 1.0f;
    channel_warp[6] = 2.0f;
    channel_warp[12] = 0.5f;
    channel_warp[18] = 0.5f;
    channel_warp[19] = 0.5f;
    channel_warp[20] = 1.0f;
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values,
                                              channel_warp, 21, output.data(), values) == IM_STATUS_OK,
            "per-channel WarpRectilinear succeeds");
    REQUIRE(output[3 * 3] != output[3 * 3 + 2],
            "WarpRectilinear applies independent radial calibration to RGB planes");

    const uint16_t ties_source[] = {0, 1, 4, 1, 2, 6, 2, 3, 8};
    uint16_t ties_output[9]{};
    float ties_constants[21]{};
    ties_constants[0] = ties_constants[6] = ties_constants[12] = 0.75f;
    ties_constants[19] = 0.0f;
    ties_constants[18] = 0.0f;
    ties_constants[20] = 1.0f;
    REQUIRE(im_native_warp_rectilinear_rgb16(3, 1, ties_source, 9, ties_constants, 21,
                                              ties_output, 9) == IM_STATUS_OK,
            "fractional WarpRectilinear sample succeeds");
    REQUIRE(ties_output[6] == 2 && ties_output[7] == 2,
            "RGB16 WarpRectilinear uses nearest-even rounding at half values");

    REQUIRE(im_native_warp_rectilinear_rgb16(0, height, source.data(), values,
                                              identity, 21, output.data(), values) == IM_STATUS_INVALID_ARGUMENT,
            "WarpRectilinear rejects empty dimensions");
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values - 1,
                                              identity, 21, output.data(), values) == IM_STATUS_INVALID_ARGUMENT,
            "WarpRectilinear validates exact source size");
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values,
                                              identity, 20, output.data(), values) == IM_STATUS_INVALID_ARGUMENT,
            "WarpRectilinear requires exactly 21 constants");
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values,
                                              identity, 21, output.data(), values - 1) == IM_STATUS_INVALID_ARGUMENT,
            "WarpRectilinear validates exact destination size");
    float invalid[21];
    std::copy(std::begin(identity), std::end(identity), std::begin(invalid));
    invalid[4] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values,
                                              invalid, 21, output.data(), values) == IM_STATUS_INVALID_ARGUMENT,
            "WarpRectilinear rejects non-finite coefficients");
    std::copy(std::begin(identity), std::end(identity), std::begin(invalid));
    invalid[18] = 1.01f;
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values,
                                              invalid, 21, output.data(), values) == IM_STATUS_INVALID_ARGUMENT,
            "WarpRectilinear rejects centers outside normalized bounds");
    std::copy(std::begin(identity), std::end(identity), std::begin(invalid));
    invalid[20] = 0.0f;
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values,
                                              invalid, 21, output.data(), values) == IM_STATUS_INVALID_ARGUMENT,
            "WarpRectilinear rejects invalid pixel aspect scale");
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values,
                                              identity, 21, source.data(), values) == IM_STATUS_INVALID_ARGUMENT,
            "WarpRectilinear rejects overlapping source and destination");
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, nullptr, values,
                                              identity, 21, output.data(), values) == IM_STATUS_INVALID_ARGUMENT,
            "WarpRectilinear rejects a null source");
    return 0;
}

int run_warp_gpu_contract(im_renderer *renderer) {
    REQUIRE(renderer, "GPU WarpRectilinear test has a renderer");
    REQUIRE(im_renderer_supports_warp_rectilinear(renderer) == 1,
            "Metal and D3D12 expose the WarpRectilinear GPU operator");
    constexpr uint32_t width = 127, height = 83;
    const size_t values = static_cast<size_t>(width) * height * 3;
    std::vector<uint16_t> source(values);
    std::mt19937 random(0xA4D0B13u);
    std::uniform_int_distribution<uint32_t> sample(0, 65535);
    for (uint16_t &value : source) value = static_cast<uint16_t>(sample(random));
    source[0] = 0;
    source[1] = 65535;
    source[2] = 32768;
    float constants[21] = {
        1.0f, 0.018f, -0.002f, 0.0001f, 0.0007f, -0.0004f,
        1.0f, 0.016f, -0.001f, 0.0001f, -0.0005f, 0.0003f,
        1.0f, 0.020f, -0.003f, 0.0002f, 0.0004f, 0.0006f,
        0.48f, 0.52f, 0.83f};
    std::vector<uint16_t> cpu(values), gpu(values);
    REQUIRE(im_native_warp_rectilinear_rgb16(width, height, source.data(), values,
                                              constants, 21, cpu.data(), values) == IM_STATUS_OK,
            "CPU reference WarpRectilinear succeeds");
    REQUIRE(im_renderer_warp_rectilinear_rgb16(renderer, width, height, source.data(), values,
                                                constants, 21, gpu.data(), values) == IM_STATUS_OK,
            "GPU WarpRectilinear succeeds");
    uint32_t max_difference = 0;
    for (size_t index = 0; index < values; ++index) {
        const uint32_t difference = cpu[index] > gpu[index]
            ? static_cast<uint32_t>(cpu[index] - gpu[index])
            : static_cast<uint32_t>(gpu[index] - cpu[index]);
        max_difference = std::max(max_difference, difference);
    }
    REQUIRE(max_difference <= 1, "GPU WarpRectilinear matches the CPU reference within one RGB16 code");
    REQUIRE(im_renderer_warp_rectilinear_rgb16(renderer, width, height, source.data(), values,
                                                constants, 20, gpu.data(), values) == IM_STATUS_INVALID_ARGUMENT,
            "GPU WarpRectilinear validates its constants count");
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    const int null_checks = run_null_handle_checks();
    if (null_checks != 0) return null_checks;
    REQUIRE(run_warp_cpu_contract() == 0, "CPU WarpRectilinear contract passes");
    const bool abi_only = argc > 1 && std::strcmp(argv[1], "--abi-only") == 0;
    const bool require_camera_profile_gpu = argc > 1 &&
        std::strcmp(argv[1], "--require-camera-profile-gpu") == 0;
    if (abi_only) {
        std::puts("PASS: C ABI argument validation, spatial dehaze, and WarpRectilinear checks");
        return 0;
    }

    im_renderer *renderer = nullptr;
    const im_status created = im_renderer_create(IM_BACKEND_AUTO, &renderer);
    if (created == IM_STATUS_BACKEND_UNAVAILABLE) {
        if (require_camera_profile_gpu) return fail("required Metal/D3D12 camera profile GPU is unavailable");
        std::printf("SKIP: native GPU backend unavailable: %s\n", im_renderer_last_error(nullptr));
        return 77;
    }
    REQUIRE(created == IM_STATUS_OK && renderer, "create renderer");
    REQUIRE(im_renderer_backend_name(renderer)[0] != '\0', "backend name is available");
    const char *selected_backend = im_renderer_backend_name(renderer);
    const bool is_gpu_profile_backend = std::strcmp(selected_backend, "Metal") == 0 ||
                                       std::strcmp(selected_backend, "D3D12") == 0;
    if (require_camera_profile_gpu) {
        REQUIRE(im_renderer_supports_camera_profile_render(renderer) == 1,
                "required GPU must provide the camera profile operator");
    }
    if (is_gpu_profile_backend) {
        REQUIRE(run_camera_profile_contract(renderer) == 0, "GPU camera profile contract passes");
        REQUIRE(run_warp_gpu_contract(renderer) == 0, "GPU WarpRectilinear contract passes");
    } else {
        const uint16_t one_pixel[] = {1000, 2000, 3000};
        REQUIRE(im_renderer_supports_camera_profile_render(renderer) == 0,
                "CUDA does not claim camera profile GPU support");
        REQUIRE(im_renderer_set_camera_profile_source(renderer, 1, 1, one_pixel, 3) ==
                    IM_STATUS_BACKEND_UNAVAILABLE,
                "unsupported GPU backends do not fall back to CPU rendering");
        REQUIRE(im_renderer_supports_warp_rectilinear(renderer) == 0,
                "unsupported GPU backends do not claim WarpRectilinear support");
        float warp_constants[21]{};
        warp_constants[0] = warp_constants[6] = warp_constants[12] = 1.0f;
        warp_constants[18] = warp_constants[19] = 0.5f;
        warp_constants[20] = 1.0f;
        uint16_t warp_output[3]{};
        REQUIRE(im_renderer_warp_rectilinear_rgb16(renderer, 1, 1, one_pixel, 3,
                                                    warp_constants, 21, warp_output, 3) ==
                    IM_STATUS_BACKEND_UNAVAILABLE,
                "unsupported GPU backends do not fall back to CPU WarpRectilinear");
    }

    constexpr uint32_t spatial_width = 64, spatial_height = 48;
    std::vector<uint16_t> spatial_input(static_cast<size_t>(spatial_width) * spatial_height * 3);
    std::vector<float> spatial_map(static_cast<size_t>(spatial_width) * spatial_height);
    for (uint32_t y = 0; y < spatial_height; ++y) {
        for (uint32_t x = 0; x < spatial_width; ++x) {
            const size_t pixel = static_cast<size_t>(y) * spatial_width + x;
            const bool building = y >= 18 && x >= 16 && x < 45;
            const uint16_t base = building ? 11000 : 44000;
            spatial_input[pixel * 3] = base;
            spatial_input[pixel * 3 + 1] = base + 1800;
            spatial_input[pixel * 3 + 2] = base + 3200;
            spatial_map[pixel] = building ? 0.48f : 0.78f;
        }
    }
    const float spatial_air[] = {0.68f, 0.71f, 0.74f};
    const im_dehaze_params spatial_params{0.9f, 0.7f, 0.55f, 0.25f, 0.35f,
                                           0.8f, 0.75f, 0.75f, 0.7f};
    std::vector<uint16_t> spatial_reference(spatial_input.size());
    std::vector<uint16_t> spatial_gpu(spatial_input.size());
    REQUIRE(im_native_dehaze_spatial_run(spatial_input.data(), spatial_width, spatial_height,
                                          spatial_map.data(), spatial_map.size(), spatial_air,
                                          &spatial_params, spatial_reference.data(), spatial_reference.size()) == IM_STATUS_OK,
            "CPU spatial reference renders skyline scene");
    REQUIRE(im_renderer_render_spatial_full(renderer, spatial_width, spatial_height,
                                            spatial_input.data(), spatial_input.size(),
                                            spatial_map.data(), spatial_map.size(), spatial_air,
                                            &spatial_params, spatial_gpu.data(), spatial_gpu.size()) == IM_STATUS_OK,
            "GPU spatial renderer renders skyline scene");
    for (size_t i = 0; i < spatial_gpu.size(); ++i) {
        const int difference = static_cast<int>(spatial_gpu[i]) - static_cast<int>(spatial_reference[i]);
        REQUIRE(difference >= -256 && difference <= 256,
                "GPU spatial output stays close to CPU reference");
    }

    const im_dehaze_params dehaze{};
    const im_basic_params basic{};
    const im_filter_params neutral_filter{0, 0, 0, 1, 1, 0, 0, 0};
    REQUIRE(im_renderer_upload_filter(renderer, &neutral_filter, nullptr, 0, nullptr, 0, 0) == IM_STATUS_OK,
            "upload neutral filter resources");
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L0, &dehaze, &basic) == IM_STATUS_NOT_READY,
            "render before preview upload is rejected");

    constexpr uint32_t preview_width = 17, preview_height = 9;
    const std::vector<uint16_t> preview = make_pixels(preview_width, preview_height);
    REQUIRE(im_renderer_upload_preview_image(renderer, preview_width, preview_height,
                                             preview.data(), preview.size() - 1) == IM_STATUS_INVALID_ARGUMENT,
            "RGB16 input value count is validated");
    REQUIRE(im_renderer_upload_preview_image(renderer, preview_width, preview_height,
                                             preview.data(), preview.size()) == IM_STATUS_OK,
            "upload preview image");

    std::vector<uint16_t> output;
    uint32_t width = 0, height = 0;
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L0, &dehaze, &basic) == IM_STATUS_OK,
            "render L0");
    REQUIRE(read_output(renderer, output, width, height), "read L0 output");
    REQUIRE(width == preview_width && height == preview_height, "L0 dimensions match preview base");
    REQUIRE(output == preview, "zero-strength and neutral settings are an exact identity");
    REQUIRE(im_renderer_copy_output(renderer, output.data(), output.size() - 1) == IM_STATUS_BUFFER_TOO_SMALL,
            "output buffer capacity is validated");
    im_dehaze_params invalid_dehaze{};
    invalid_dehaze.strength = 1.01f;
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L0, &invalid_dehaze, &basic) == IM_STATUS_INVALID_ARGUMENT,
            "dehaze parameter bounds are validated");
    im_basic_params invalid_basic{};
    invalid_basic.exposure = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L0, &dehaze, &invalid_basic) == IM_STATUS_INVALID_ARGUMENT,
            "non-finite basic parameters are rejected");
    REQUIRE(im_renderer_upload_filter(renderer, &neutral_filter, nullptr, 0, preview.data(), 0, 0) == IM_STATUS_INVALID_ARGUMENT,
            "empty LUTs must not carry a pointer");

    REQUIRE(im_renderer_render(renderer, IM_RENDER_L1, &dehaze, &basic) == IM_STATUS_OK, "render L1");
    REQUIRE(read_output(renderer, output, width, height), "read L1 output");
    REQUIRE(width == 9 && height == 5, "L1 is approximately half the preview base");
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L2, &dehaze, &basic) == IM_STATUS_OK, "render L2");
    REQUIRE(read_output(renderer, output, width, height), "read L2 output");
    REQUIRE(width == 5 && height == 3, "L2 is approximately one quarter of the preview base");

    im_basic_params brighter{};
    brighter.exposure = 1.0f;
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L0, &dehaze, &brighter) == IM_STATUS_OK,
            "render with updated parameters");
    std::vector<uint16_t> bright_output;
    REQUIRE(read_output(renderer, bright_output, width, height), "read updated render");
    REQUIRE(any_difference(bright_output, preview), "latest render parameters change output");
    const im_dehaze_params stronger_dehaze{1.0f, 0.70f, 0.55f, 0.25f, 0.35f, 0.80f, 0.75f, 0.75f, 0.70f};
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L0, &stronger_dehaze, &basic) == IM_STATUS_OK,
            "render nonzero dehaze settings");
    std::vector<uint16_t> dehazed_output;
    REQUIRE(read_output(renderer, dehazed_output, width, height), "read nonzero dehaze output");
    REQUIRE(any_difference(dehazed_output, preview), "nonzero dehaze changes image values");
    const auto dehazed_range = std::minmax_element(dehazed_output.begin(), dehazed_output.end());
    REQUIRE(*dehazed_range.first < *dehazed_range.second && *dehazed_range.second <= UINT16_MAX,
            "dehaze output retains a non-degenerate RGB16 range");
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L0, &dehaze, &basic) == IM_STATUS_OK,
            "render again with neutral parameters");
    REQUIRE(read_output(renderer, output, width, height) && output == preview,
            "a later render replaces the prior parameter result");

    std::vector<uint16_t> curve(3 * 256);
    for (size_t i = 0; i < 256; ++i) {
        const uint16_t value = static_cast<uint16_t>(i * 257u);
        curve[i] = static_cast<uint16_t>(65535u - value);
        curve[256 + i] = curve[512 + i] = value;
    }
    im_filter_params curve_filter{1, 0, 0, 1, 1, 0, 0, 0};
    REQUIRE(im_renderer_upload_filter(renderer, &curve_filter, curve.data(), curve.size(), nullptr, 0, 0) == IM_STATUS_OK,
            "upload filter curve");
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L0, &dehaze, &basic) == IM_STATUS_OK, "render curve");
    std::vector<uint16_t> curve_output;
    REQUIRE(read_output(renderer, curve_output, width, height), "read curve output");
    REQUIRE(any_difference(curve_output, preview), "curve data changes rendered pixels");

    constexpr uint32_t lut_edge = 2;
    std::vector<uint16_t> lut(lut_edge * lut_edge * lut_edge * 3);
    for (uint32_t r = 0; r < lut_edge; ++r) {
        for (uint32_t g = 0; g < lut_edge; ++g) {
            for (uint32_t b = 0; b < lut_edge; ++b) {
                const size_t at = ((r * lut_edge + g) * lut_edge + b) * 3;
                lut[at] = static_cast<uint16_t>((1 - r) * 65535u);
                lut[at + 1] = static_cast<uint16_t>(g * 65535u);
                lut[at + 2] = static_cast<uint16_t>(b * 65535u);
            }
        }
    }
    im_filter_params lut_filter{0, 1, 0, 1, 1, 0, 0, 0};
    REQUIRE(im_renderer_upload_filter(renderer, &lut_filter, nullptr, 0,
                                      lut.data(), lut.size(), lut_edge) == IM_STATUS_OK,
            "upload filter LUT");
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L0, &dehaze, &basic) == IM_STATUS_OK, "render LUT");
    std::vector<uint16_t> lut_output;
    REQUIRE(read_output(renderer, lut_output, width, height), "read LUT output");
    REQUIRE(any_difference(lut_output, preview), "3D LUT data changes rendered pixels");

    constexpr uint32_t full_width = 21, full_height = 11;
    const std::vector<uint16_t> full_input = make_pixels(full_width, full_height);
    const std::vector<uint16_t> full_original = full_input;
    REQUIRE(im_renderer_render_full(renderer, full_width, full_height, full_input.data(), full_input.size(),
                                    &dehaze, &basic) == IM_STATUS_OK,
            "full render accepts newly decoded full-resolution pixels");
    REQUIRE(read_output(renderer, output, width, height), "read full-resolution output");
    REQUIRE(width == full_width && height == full_height, "full render returns full input dimensions");
    REQUIRE(full_input == full_original, "render does not modify input pixels");
    REQUIRE(im_renderer_render(renderer, IM_RENDER_L0, &dehaze, &basic) == IM_STATUS_OK,
            "preview pyramid remains available after full render");
    REQUIRE(read_output(renderer, output, width, height) && width == preview_width && height == preview_height,
            "full render does not replace cached preview dimensions");
    REQUIRE(preview == make_pixels(preview_width, preview_height), "preview input remains unchanged");

    constexpr uint32_t ricoh_lut_edge = 2;
    std::vector<uint16_t> ricoh_lut(ricoh_lut_edge * ricoh_lut_edge * ricoh_lut_edge * 3);
    for (uint32_t r = 0; r < ricoh_lut_edge; ++r) {
        for (uint32_t g = 0; g < ricoh_lut_edge; ++g) {
            for (uint32_t b = 0; b < ricoh_lut_edge; ++b) {
                const size_t at = ((r * ricoh_lut_edge + g) * ricoh_lut_edge + b) * 3;
                ricoh_lut[at] = static_cast<uint16_t>((1 - r) * 65535u);
                ricoh_lut[at + 1] = static_cast<uint16_t>(g * 65535u);
                ricoh_lut[at + 2] = static_cast<uint16_t>(b * 65535u);
            }
        }
    }
    REQUIRE(im_renderer_render_ricoh_full(renderer, full_width, full_height,
                                          full_input.data(), full_input.size(),
                                          ricoh_lut.data(), ricoh_lut.size() - 1,
                                          ricoh_lut_edge) == IM_STATUS_INVALID_ARGUMENT,
            "Ricoh full render validates LUT value count");
    REQUIRE(im_renderer_render_ricoh_full(renderer, full_width, full_height,
                                          full_input.data(), full_input.size(),
                                          ricoh_lut.data(), ricoh_lut.size(),
                                          ricoh_lut_edge) == IM_STATUS_OK,
            "Ricoh full render applies a supplied 3D LUT");
    std::vector<uint16_t> ricoh_output;
    REQUIRE(read_output(renderer, ricoh_output, width, height), "read Ricoh output");
    REQUIRE(width == full_width && height == full_height, "Ricoh output keeps full input dimensions");
    REQUIRE(any_difference(ricoh_output, full_input), "Ricoh LUT changes rendered pixels");
    REQUIRE(full_input == full_original, "Ricoh render does not modify input pixels");

    const char *backend = im_renderer_backend_name(renderer);
    std::printf("PASS: native renderer C ABI, cached levels, full-resolution path and filter resources; backend=%s\n",
                backend ? backend : "unknown");
    im_renderer_destroy(renderer);
    return 0;
}
