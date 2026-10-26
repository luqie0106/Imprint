#ifndef IMPRINT_GEO_H
#define IMPRINT_GEO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stable C ABI. RGB values in color_rgb use 0xRRGGBB; zero selects Auto. */
typedef struct GeoConfig {
    double latitude;
    double longitude;
    double opacity;
    double radius_km;
    double offset_x;
    double offset_y;
    double scale;
    double rotation;
    double line_width;
    double feather;
    double coverage;
    double manual_horizon;
    uint32_t color_rgb;
    int32_t enabled;
    int32_t has_location;
    int32_t style;
    int32_t auto_sky_fade;
    int32_t labels;
    int32_t north_arrow;
    int32_t scale_bar;
    int32_t photo_marker;
    int32_t adaptive;
    int32_t provider;
} GeoConfig;

/* Geographic feature input for a real dataset. Coordinates are WGS84 degrees.
 * kind: 0=road, 1=waterway, 2=coastline, 3=place (one point). Labels are
 * optional UTF-8 NUL-terminated strings limited to 256 bytes. */
typedef struct GeoVertex {
    double latitude;
    double longitude;
} GeoVertex;

typedef struct GeoFeatureView {
    uint32_t kind;
    uint32_t point_count;
    const GeoVertex *points;
    const char *label;
} GeoFeatureView;

/* Input is tightly packed RGB8 sRGB. Output is straight-alpha RGBA8 sRGB. */
int32_t imprint_geo_render(const uint8_t *rgb,
                           uint32_t width,
                           uint32_t height,
                           const GeoConfig *config,
                           uint8_t *overlay_rgba,
                           char *error,
                           uint32_t error_capacity);

/* Render caller-supplied real geographic features (provider must be 1).
 * This function performs no network request and never falls back to Mock.
 * Limits: 10,000 features, 500,000 points total, 256 UTF-8 label bytes. */
int32_t imprint_geo_render_features(const uint8_t *rgb,
                                    uint32_t width,
                                    uint32_t height,
                                    const GeoConfig *config,
                                    const GeoFeatureView *features,
                                    uint32_t feature_count,
                                    uint8_t *overlay_rgba,
                                    char *error,
                                    uint32_t error_capacity);

/* Straight-alpha source-over in sRGB code values. Alpha zero is byte-exact identity. */
int32_t imprint_geo_composite(const uint8_t *rgb,
                              const uint8_t *overlay_rgba,
                              uint32_t width,
                              uint32_t height,
                              uint8_t *out_rgb);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IMPRINT_GEO_H */
