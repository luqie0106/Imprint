#include "imprint_renderer.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
    const uint32_t width = argc > 1 ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 1920;
    const uint32_t height = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 1080;
    const unsigned iterations = argc > 3 ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 100;
    if (!width || !height || !iterations) {
        std::fprintf(stderr, "Usage: %s [preview_width preview_height iterations]\n", argv[0]);
        return 2;
    }

    im_renderer *renderer = nullptr;
    const im_status status = im_renderer_create(IM_BACKEND_AUTO, &renderer);
    if (status == IM_STATUS_BACKEND_UNAVAILABLE) {
        std::puts("No GPU backend is available for the local render benchmark.");
        return 77;
    }
    if (status != IM_STATUS_OK) {
        std::fprintf(stderr, "Could not create renderer (status %d)\n", status);
        return 1;
    }

    std::vector<uint16_t> image(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
        image[i * 3] = static_cast<uint16_t>((i * 37) & 0xffff);
        image[i * 3 + 1] = static_cast<uint16_t>((i * 71 + 10000) & 0xffff);
        image[i * 3 + 2] = static_cast<uint16_t>((i * 113 + 20000) & 0xffff);
    }
    const im_filter_params filter{0, 0, 0, 1, 1, 0, 0, 0};
    const im_dehaze_params dehaze{};
    const im_basic_params basic{};
    if (im_renderer_upload_preview_image(renderer, width, height, image.data(), image.size()) != IM_STATUS_OK ||
        im_renderer_upload_filter(renderer, &filter, nullptr, 0, nullptr, 0, 0) != IM_STATUS_OK) {
        std::fprintf(stderr, "Could not prepare benchmark input: %s\n", im_renderer_last_error(renderer));
        im_renderer_destroy(renderer);
        return 1;
    }
    for (unsigned i = 0; i < 10; ++i) {
        if (im_renderer_render(renderer, IM_RENDER_L2, &dehaze, &basic) != IM_STATUS_OK) {
            std::fprintf(stderr, "Warmup render failed: %s\n", im_renderer_last_error(renderer));
            im_renderer_destroy(renderer);
            return 1;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) {
        if (im_renderer_render(renderer, IM_RENDER_L2, &dehaze, &basic) != IM_STATUS_OK) {
            std::fprintf(stderr, "Render failed: %s\n", im_renderer_last_error(renderer));
            im_renderer_destroy(renderer);
            return 1;
        }
    }
    const auto finish = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(finish - start).count();
    uint32_t output_width = 0, output_height = 0;
    size_t output_values = 0;
    if (im_renderer_get_output_size(renderer, &output_width, &output_height, &output_values) != IM_STATUS_OK) {
        std::fprintf(stderr, "Benchmark output was not available\n");
        im_renderer_destroy(renderer);
        return 1;
    }
    const double ms_per_frame = seconds * 1000.0 / iterations;
    const double megapixels_per_second = (static_cast<double>(output_width) * output_height * iterations) / seconds / 1e6;
    std::printf("backend=%s level=L2 input=%ux%u output=%ux%u iterations=%u total=%.4fs ms/frame=%.3f MPix/s=%.2f\n",
                im_renderer_backend_name(renderer), width, height, output_width, output_height,
                iterations, seconds, ms_per_frame, megapixels_per_second);
    (void)output_values;
    im_renderer_destroy(renderer);
    return 0;
}
