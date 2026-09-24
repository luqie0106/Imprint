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
};

std::unique_ptr<Backend> create_backend(im_backend_kind kind, std::string &error);

} // namespace imprint
