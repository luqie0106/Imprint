#ifndef IMPRINT_GEO_HPP
#define IMPRINT_GEO_HPP

#include "imprint_geo.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace imprint::geo {

enum class FeatureKind { Road, Waterway, Coastline, Place };
enum class Style { Auto = 0, Urban = 1, Coastal = 2, Terrain = 3 };
enum class ProviderKind { Mock = 0, Overpass = 1 };

struct MetricPoint {
    double east_m = 0.0;
    double north_m = 0.0;
};

/* Local azimuthal-equidistant projection: radial distance is in true metres. */
class LocalAzimuthalEquidistant {
public:
    LocalAzimuthalEquidistant(double latitude_deg, double longitude_deg);
    bool valid() const noexcept;
    bool forward(double latitude_deg, double longitude_deg, MetricPoint *out) const noexcept;
    bool inverse(const MetricPoint &point, double *latitude_deg, double *longitude_deg) const noexcept;
private:
    double latitude_rad_;
    double longitude_rad_;
    bool valid_;
};

struct GeoQuery {
    double latitude = 0.0;
    double longitude = 0.0;
    double radius_m = 0.0;
};

struct GeoFeature {
    FeatureKind kind = FeatureKind::Road;
    std::string label;
    std::vector<std::pair<double, double>> points_deg; // latitude, longitude
};

struct GeoDataset {
    std::vector<GeoFeature> features;
    bool mock_data = false;
};

class IGeoDataProvider {
public:
    virtual ~IGeoDataProvider() = default;
    virtual bool load(const GeoQuery &query, GeoDataset *out, std::string *error) const = 0;
};

class MockGeoDataProvider final : public IGeoDataProvider {
public:
    bool load(const GeoQuery &query, GeoDataset *out, std::string *error) const override;
};

class OverpassGeoDataProvider final : public IGeoDataProvider {
public:
    bool load(const GeoQuery &query, GeoDataset *out, std::string *error) const override;
};

/* Replaceable sky segmentation interface. The default implementation is a
 * conservative top-connected RGB/edge mask; it does not run a learned model. */
struct SkyMask {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels;
    /* Three units per pixel (chamfer 3/4 distance), saturated at uint16 max. */
    std::vector<uint16_t> distance_inside;
    std::vector<int32_t> bottom_by_column;
    int32_t min_x = 0, min_y = 0, max_x = -1, max_y = -1;
    size_t area = 0;
};

class ISkyMaskProvider {
public:
    virtual ~ISkyMaskProvider() = default;
    virtual bool build(const uint8_t *rgb, uint32_t width, uint32_t height,
                       SkyMask *out, std::string *error) const = 0;
};

class TopConnectedRgbSkyMaskProvider final : public ISkyMaskProvider {
public:
    bool build(const uint8_t *rgb, uint32_t width, uint32_t height,
               SkyMask *out, std::string *error) const override;
};

/* C++ entry point for injecting a replaceable sky segmenter. The C ABI uses
 * TopConnectedRgbSkyMaskProvider by default. This core does not cache map data;
 * the default sky provider keeps a one-entry thread-local exact-input cache. */
bool render_with_sky_provider(const uint8_t *rgb, uint32_t width, uint32_t height,
                              const GeoConfig &config, uint8_t *overlay_rgba,
                              const ISkyMaskProvider &sky_provider, std::string *error);

bool render_with_providers(const uint8_t *rgb, uint32_t width, uint32_t height,
                           const GeoConfig &config, uint8_t *overlay_rgba,
                           const IGeoDataProvider &data_provider,
                           const ISkyMaskProvider &sky_provider, std::string *error);

} // namespace imprint::geo

#endif /* IMPRINT_GEO_HPP */
