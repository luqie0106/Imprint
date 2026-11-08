// Test-only bridge; commit compact output only after successful processing.
#include "../src/acr_tone_kernels.hpp"
#include <algorithm>
#include <cstdint>

extern "C" int acr_test_guide_resize(const float* input, std::uint32_t width,
    std::uint32_t height, std::uint32_t output_width, std::uint32_t output_height,
    float* output) noexcept {
    if (!output) return 1;
    try {
        const auto image=imprint::acr::resize_gray(input,width,height,output_width,output_height);
        std::copy(image.pixels.begin(),image.pixels.end(),output);
        return 0;
    } catch (...) { return 1; }
}
