// Test-only adapter for prepared mask controls and the existing local remap.
#include "../src/acr_tone_kernels.hpp"
#include <algorithm>
#include <cstdint>

extern "C" int acr_test_mask_sampling(double low, double high, double* output) noexcept {
    try {
        const auto p=imprint::acr::prepare_local_remap_sampling(low,high);
        const double result[]{p.low,p.high,static_cast<double>(p.intervals),p.step,p.inverse_step};
        std::copy_n(result,5,output);
        return 0;
    } catch (...) { return 1; }
}

extern "C" int acr_test_mask_sigma(double low, double high, double epsilon,
                                    int mode, double* output) noexcept {
    try {
        const double result=imprint::acr::prepare_local_mask_sigma(low,high,epsilon,mode);
        *output=result;
        return 0;
    } catch (...) { return 1; }
}

extern "C" int acr_test_mask_remap(const float* input, std::uint32_t width,
    std::uint32_t height, double low, double high, double sigma, float* output) noexcept {
    try {
        const auto image=imprint::acr::local_laplacian_gray(input,width,height,low,high,1.0,0.0,sigma);
        std::copy(image.pixels.begin(),image.pixels.end(),output);
        return 0;
    } catch (...) { return 1; }
}
