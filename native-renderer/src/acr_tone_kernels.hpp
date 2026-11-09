#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace imprint::acr {

struct GrayImage {
    std::uint32_t width;
    std::uint32_t height;
    std::vector<float> pixels;
};

struct LocalToneLevelExtent {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};
struct LocalToneGeometry {
    std::array<LocalToneLevelExtent,16> levels{};
    std::uint32_t level_count = 0;
    std::uint32_t statistics_level = 0;
};
// Complete zero-origin image only: <=16 ceil-halved levels (0095b478/688),
// source statistics and collapse callback use the same index (006fbbc8 /
// 00a089f0). This describes local pyramid levels, not source/Stage4 levels.
LocalToneGeometry prepare_local_tone_geometry(std::uint32_t width, std::uint32_t height);

// Named subset of flags=0 BuildStats needed for local-mask anchors and tone
// nodes. This is not the full 33-double Adobe stats object (moments omitted).
struct LogSceneStats {
    double center;   // source center (stats[2], record byte offset +0x10)
    double low;      // effective quantile .0001 (stats[13])
    double high;     // effective quantile .9999 (stats[20])
    double median;   // effective midpoint median (stats[23])
    std::size_t effective_count;
    bool fallback;
};
// Saved 009d6fac flags=0 selected fields: asymmetric raw floor/ceil quantiles,
// effective repeat filtering and grouped/FMA representative center. Retains
// the saved upper-rank post-decrement scan, not a replacement quantile mean.
// The remaining 33-double statistics, loader and runtime dispatch are omitted.
LogSceneStats build_log_scene_stats(const float *pixels, std::size_t count, double epsilon);

// Recovered CPU float, zero-origin, unskewed Gaussian resize. Coordinates
// clamp before phase selection; edge/column/corner samples repeat nearest.
// Float weights with double normalization sum; explicit 01341a30 vertical
// and 0134147c horizontal baseline arithmetic, V then H. Does not infer
// optimized dispatch or reuse the ordinary cubic coefficient generator.
// This is a tone-pyramid component, not a complete Highlights operator.
GrayImage resize_gray(const float *pixels, std::uint32_t width,
                      std::uint32_t height, std::uint32_t output_width,
                      std::uint32_t output_height);

// Prepared zero-origin mode4 components (00a08cc8/00a08e8c, 00a09264).
// Swap rotates each row left by ceil(width/2), or floor(width/2) for inverse;
// odd-width forward is not self-inverse. Copies preserve the input float bits.
// Blend uses the recovered float/double smooth horizontal weight and explicit
// FMA ordering. Both require width >=2 and finite, complete compact images.
GrayImage swap_left_right_gray(const float *pixels, std::uint32_t width,
                               std::uint32_t height, bool inverse = false);
GrayImage blend_horizontal_gray(const float *first, const float *second,
                                std::uint32_t width, std::uint32_t height);

// Input is the already normalized image entering CalculateFlareForLog;
// caller is responsible for the preceding black/precision stages. Returns
// the stored flare field, which may be zero. It is not yet log's epsilon.
double calculate_flare_for_log(const float *pixels, std::size_t pixel_count,
                                std::uint32_t channels);
double select_log_epsilon(int source_mode, double stored_flare);

// Exact recovered terminal stages: weighted RGB -> nonnegative gray ->
// exp2 gain -> log2(gray + epsilon), with the 2^120 upper protection.
// Input is the pre-gray working domain, not automatically linear sRGB.
// Uses the fixed 00889340 float constants and G-product/R-FMA/B-FMA order,
// not acr_gray's Q14 coefficient preparation. EV90==+/-0 omits the gain stage;
// otherwise gain is float(exp2(double EV90)). No post-gain [0,1] clamp.
GrayImage make_gray_log(const float *rgb, std::uint32_t width,
                        std::uint32_t height, double gain_ev, double epsilon);

struct LocalRemapSampling {
    float low;
    float high;
    std::uint32_t intervals;
    float step;
    float inverse_step;
};
// Endpoints are already clamped against the statistics median +/-20 by the
// caller. 00a0c25c / 00a0b958 use FCVTPS (ceil), not truncation. The native
// supported range remains <=40.001; invalid/unrepresentable inputs throw.
LocalRemapSampling prepare_local_remap_sampling(double low, double high);
// 00dafa20 + 012fff5c: explicit prepared low/high statistics and sigma mode.
// Preserves both observed binary64 FMADD sites. Does not compute statistics.
double prepare_local_mask_sigma(double low, double high, double epsilon, int sigma_mode);

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
// Complete zero-origin images only. Boundary2 is the ordinary single pass;
// boundary4 with width >=2 adds a swapped second pass and horizontal blend.
// Delta/alignment retain the second pass's source/collapse statistics, not
// statistics rebuilt from the blend. Width1 boundary4 uses the ordinary pass.
// Other boundary modes are rejected; sigma mode is independent of boundary.
LocalToneMask build_local_tone_mask(const float *log_gray, std::uint32_t width,
                                    std::uint32_t height, double epsilon, int sigma_mode,
                                    std::uint32_t boundary_mode = 2);

}  // namespace imprint::acr
