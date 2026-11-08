// Test-only prepared log statistics adapter; failed calls leave output intact.
#include "../src/acr_tone_kernels.hpp"
#include <algorithm>
#include <cstddef>

extern "C" int acr_test_log_scene_stats(const float* pixels, std::size_t count,
                                       double epsilon, double* output) noexcept {
    if (!output) return 1;
    try {
        const auto stats = imprint::acr::build_log_scene_stats(pixels,count,epsilon);
        const double fields[]{stats.center,stats.low,stats.high,stats.median,
            static_cast<double>(stats.effective_count),stats.fallback ? 1.0 : 0.0};
        std::copy_n(fields,6,output);
        return 0;
    } catch (...) { return 1; }
}
