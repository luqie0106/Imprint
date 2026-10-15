#include "../src/acr_profile_tone.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
bool overlaps(const void *a, std::size_t a_bytes,
              const void *b, std::size_t b_bytes) {
    const auto a_start = reinterpret_cast<std::uintptr_t>(a);
    const auto b_start = reinterpret_cast<std::uintptr_t>(b);
    if (a_start <= b_start) return b_start - a_start < a_bytes;
    return a_start - b_start < b_bytes;
}
}  // namespace

extern "C" int acr_test_profile_tone_lut(const double *points_xy,
        std::size_t point_count, int clip_curve_output, float *output,
        std::size_t output_entries) {
    if (!points_xy || !output || (clip_curve_output != 0 && clip_curve_output != 1) ||
        output_entries != imprint::acr::kProfileToneLutEntries ||
        point_count > std::numeric_limits<std::size_t>::max() / (2 * sizeof(double)) ||
        overlaps(points_xy, point_count * 2 * sizeof(double),
                 output, output_entries * sizeof(float))) {
        return 1;
    }
    try {
        const auto lut = imprint::acr::build_profile_tone_lut(
            points_xy, point_count, clip_curve_output != 0);
        std::copy(lut.begin(), lut.end(), output);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int acr_test_profile_tone_rgb(const float *source,
        std::size_t pixel_count, const double *points_xy,
        std::size_t point_count, int modern, float *output) {
    if (!source || !points_xy || !output || (modern != 0 && modern != 1) ||
        pixel_count > std::numeric_limits<std::size_t>::max() / 3 ||
        point_count > std::numeric_limits<std::size_t>::max() / (2 * sizeof(double))) {
        return 1;
    }
    const std::size_t samples = pixel_count * 3;
    if (samples > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        overlaps(source, samples * sizeof(float), output, samples * sizeof(float)) ||
        overlaps(points_xy, point_count * 2 * sizeof(double),
                 output, samples * sizeof(float))) {
        return 1;
    }
    try {
        imprint::acr::apply_profile_tone(source, output, pixel_count,
            points_xy, point_count, modern != 0);
        return 0;
    } catch (...) {
        return 1;
    }
}
