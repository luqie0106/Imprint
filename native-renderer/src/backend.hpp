#pragma once

#include "imprint_renderer.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imprint {

struct ImageStats {
    float mean_luma;
    float max_luma;
    float air_r;
    float air_g;
    float air_b;
    float haze_level;
    float brightness_gain;
};

struct ImageLevel {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint16_t> pixels;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char *name() const = 0;
    virtual bool supports_physical_float() const { return false; }
    virtual bool set_images(const std::array<ImageLevel, 3> &levels, std::string &error) = 0;
    virtual bool set_filter(const im_filter_params &params,
                            const std::vector<uint16_t> &curve,
                            const std::vector<uint16_t> &lut,
                            uint32_t lut_edge,
                            std::string &error) = 0;
    virtual bool render_cached(unsigned level,
                               const im_dehaze_params &dehaze,
                               const im_basic_params &basic,
                               const ImageStats &stats,
                               uint16_t *destination,
                               size_t destination_values,
                               std::string &error) = 0;
    virtual bool render_full(const ImageLevel &source,
                             const ImageStats &stats,
                             const im_dehaze_params &dehaze,
                             const im_basic_params &basic,
                             uint16_t *destination,
                             size_t destination_values,
                             std::string &error) = 0;
    virtual bool render_spatial_full(const ImageLevel &source,
                                     const float *transmission,
                                     const ImageStats &stats,
                                     const im_dehaze_params &dehaze,
                                     uint16_t *destination,
                                     size_t destination_values,
                                     std::string &error) {
        (void)source; (void)transmission; (void)stats; (void)dehaze;
        (void)destination; (void)destination_values;
        error = "Spatial dehaze is unavailable on this GPU backend";
        return false;
    }
    virtual bool render_physical_float(uint32_t width,
                                       uint32_t height,
                                       const float *source,
                                       const float *transmission,
                                       const float *airlight_rgb,
                                       const im_dehaze_params &params,
                                       float *destination,
                                       size_t destination_values,
                                       std::string &error) {
        (void)width; (void)height; (void)source; (void)transmission;
        (void)airlight_rgb; (void)params; (void)destination; (void)destination_values;
        error = "Physical float dehaze is unavailable on this GPU backend";
        return false;
    }
};

std::unique_ptr<Backend> create_backend(im_backend_kind kind, std::string &error);

} // namespace imprint
