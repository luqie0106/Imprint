#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

struct GrayImage {
    std::uint32_t width;
    std::uint32_t height;
    std::vector<float> pixels;
};

// Named subset of flags=0 BuildStats needed for local-mask anchors and tone
// nodes. This is not the full 33-double Adobe stats object (moments omitted).
struct LogSceneStats {
    double center;
    double low;      // effective quantile .0001 (stats[13])
    double high;     // effective quantile .9999 (stats[20])
    double median;   // effective midpoint median (stats[23])
    std::size_t effective_count;
    bool fallback;
};
LogSceneStats build_log_scene_stats(const float *pixels, std::size_t count, double epsilon);

// Recovered CPU float, zero-origin, unskewed Gaussian resize. Coordinates
// clamp before phase selection; edge/column/corner samples repeat nearest.
// This is a tone-pyramid component, not a complete Highlights operator.
GrayImage resize_gray(const float *pixels, std::uint32_t width,
                      std::uint32_t height, std::uint32_t output_width,
                      std::uint32_t output_height);

// Input is the already normalized image entering CalculateFlareForLog;
// caller is responsible for the preceding black/precision stages. Returns
// the stored flare field, which may be zero. It is not yet log's epsilon.
double calculate_flare_for_log(const float *pixels, std::size_t pixel_count,
                                std::uint32_t channels);
double select_log_epsilon(int source_mode, double stored_flare);

// Exact recovered terminal stages: weighted RGB -> nonnegative gray ->
// exp2 gain -> log2(gray + epsilon), with the 2^120 upper protection.
// Input is the pre-gray working domain, not automatically linear sRGB.
GrayImage make_gray_log(const float *rgb, std::uint32_t width,
                        std::uint32_t height, double gain_ev, double epsilon);

// Recovered full-image local Laplacian remap component. low/high must already
// be derived from the effective log statistics (not raw percentiles). Samples
// are processed one at a time to bound memory; the coarsest guide is retained.
// This returns the collapsed log mask, before BuildDeltaMask/SH mapping.
GrayImage local_laplacian_gray(const float *pixels, std::uint32_t width,
                              std::uint32_t height, double low, double high,
                              double alpha, double beta, double sigma,
                              LogSceneStats *collapse_stats = nullptr,
                              std::uint32_t statistics_level = 0,
                              double epsilon = 0x1p-12);

// 00dae238: align the two representative log centers, subtract the original
// log guide, and cap only the upper delta at +4. There is no lower clamp.
GrayImage build_delta_mask(const float *original, const float *collapsed,
                          std::uint32_t width, std::uint32_t height,
                          double original_center, double mask_center);

struct LocalToneMask {
    GrayImage delta;
    LogSceneStats source_stats;
    LogSceneStats mask_stats; // Before 009d94dc's center shift.
    LogSceneStats aligned_mask_stats; // center/low/high shifted; median unchanged.
    double remap_sigma;
    std::uint32_t statistics_level;
};
// Native chain from an already prepared log guide: Gaussian/BuildStats,
// adaptive sigma, incremental remap/collapse callback, BuildDeltaMask. This
// still excludes source image/color preparation and the final SH mapping.
LocalToneMask build_local_tone_mask(const float *log_gray, std::uint32_t width,
                                    std::uint32_t height, double epsilon, int sigma_mode);

}  // namespace imprint::acr
