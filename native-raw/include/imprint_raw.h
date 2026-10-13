#ifndef IMPRINT_RAW_H
#define IMPRINT_RAW_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(IMPRINT_RAW_BUILD)
#    define IMPRINT_RAW_API __declspec(dllexport)
#  else
#    define IMPRINT_RAW_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__)
#  define IMPRINT_RAW_API __attribute__((visibility("default")))
#else
#  define IMPRINT_RAW_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct imprint_raw_sizes {
    uint32_t raw_height;
    uint32_t raw_width;
    uint32_t height;
    uint32_t width;
    uint32_t top_margin;
    uint32_t left_margin;
    uint32_t iheight;
    uint32_t iwidth;
    double pixel_aspect;
    int32_t flip;
    uint32_t crop_left_margin;
    uint32_t crop_top_margin;
    uint32_t crop_width;
    uint32_t crop_height;
    uint32_t raw_pitch_bytes;
} imprint_raw_sizes;

typedef struct imprint_raw_info {
    imprint_raw_sizes sizes;
    uint32_t compression;
    uint32_t filters;
    uint32_t num_colors;
    uint32_t raw_pattern_size;
    uint8_t raw_pattern[36];
    char color_desc[5];
    char make[64];
    char model[64];
    char lens_make[128];
    char lens_model[128];
    char artist[128];
    float camera_whitebalance[4];
    float daylight_whitebalance[4];
    float cam_xyz[12];
    uint32_t black_level_per_channel[4];
    uint32_t white_level;
    int32_t has_camera_white_level_per_channel;
    uint32_t camera_white_level_per_channel[4];
    double iso_speed;
    double shutter_speed;
    double aperture;
    double focal_length;
    int64_t timestamp;
    int32_t shot_order;
    double lens_min_focal;
    double lens_max_focal;
} imprint_raw_info;

typedef struct imprint_raw_postprocess_params {
    int32_t half_size;
    int32_t use_camera_wb;
    int32_t no_auto_bright;
    int32_t output_bps;
    int32_t output_color; /* 0 = camera/raw RGB, 1 = sRGB */
    double gamma[2];
    int32_t has_user_wb;
    float user_wb[4];
    int32_t has_user_flip;
    int32_t user_flip;
} imprint_raw_postprocess_params;

/*
 * Open only Nikon NEFs selected by LibRaw's nikon_he_load_raw decoder and
 * tagged with Nikon NEFCompression 13 or 14. A successful open includes unpack().
 * Every function returns zero on success and writes a diagnostic on failure.
 */
IMPRINT_RAW_API int32_t imprint_raw_probe_nikon_he(
    const char *path,
    int32_t *out_is_nikon_he,
    char *error,
    size_t error_capacity);

IMPRINT_RAW_API int32_t imprint_raw_open(
    const char *path,
    void **out_handle,
    imprint_raw_info *out_info,
    char *error,
    size_t error_capacity);

IMPRINT_RAW_API void imprint_raw_close(void *handle);

/* Copies the visible single-channel CFA into caller-owned uint16 storage. */
IMPRINT_RAW_API int32_t imprint_raw_copy_visible(
    void *handle,
    uint16_t *destination,
    size_t destination_elements,
    char *error,
    size_t error_capacity);

/*
 * Produces a three-channel RGB memory image. The returned pointer remains
 * owned by the handle and is valid until the next postprocess/thumbnail call
 * or close. Callers must serialize access to a handle and copy the pixels.
 */
IMPRINT_RAW_API int32_t imprint_raw_postprocess(
    void *handle,
    const imprint_raw_postprocess_params *params,
    const uint8_t **out_data,
    size_t *out_size,
    uint32_t *out_width,
    uint32_t *out_height,
    uint32_t *out_channels,
    uint32_t *out_bits,
    char *error,
    size_t error_capacity);

/* Returns raw JPEG/BITMAP thumbnail bytes, never a decoded RAW image. */
IMPRINT_RAW_API int32_t imprint_raw_extract_thumb(
    void *handle,
    const uint8_t **out_data,
    size_t *out_size,
    int32_t *out_format, /* 1 = JPEG, 2 = RGB bitmap */
    uint32_t *out_width,
    uint32_t *out_height,
    uint32_t *out_channels,
    uint32_t *out_bits,
    char *error,
    size_t error_capacity);

/* Returns the codec's CPU/Metal diagnostic captured during unpack(). */
IMPRINT_RAW_API const char *imprint_raw_decoder_backend(const void *handle);

#ifdef __cplusplus
}
#endif

#endif /* IMPRINT_RAW_H */
