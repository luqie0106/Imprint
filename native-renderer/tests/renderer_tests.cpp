#include "imprint_renderer.h"

#include <algorithm>
#include <cstdio>
#include <limits>
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
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    const int null_checks = run_null_handle_checks();
    if (null_checks != 0) return null_checks;
    if (argc > 1) {
        std::puts("PASS: C ABI null-handle validation");
        return 0;
    }

    im_renderer *renderer = nullptr;
    const im_status created = im_renderer_create(IM_BACKEND_AUTO, &renderer);
    if (created == IM_STATUS_BACKEND_UNAVAILABLE) {
        std::printf("SKIP: native GPU backend unavailable: %s\n", im_renderer_last_error(nullptr));
        return 77;
    }
    REQUIRE(created == IM_STATUS_OK && renderer, "create renderer");
    REQUIRE(im_renderer_backend_name(renderer)[0] != '\0', "backend name is available");

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
