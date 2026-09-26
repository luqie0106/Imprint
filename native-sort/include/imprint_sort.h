#ifndef IMPRINT_SORT_H
#define IMPRINT_SORT_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(IMPRINT_SORT_BUILD)
#    define IMPRINT_SORT_API __declspec(dllexport)
#  else
#    define IMPRINT_SORT_API __declspec(dllimport)
#  endif
#else
#  define IMPRINT_SORT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct im_sort_roi {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
} im_sort_roi;

/* Returns 0 on success, 1 for invalid arguments, or 2 for an internal error. */
IMPRINT_SORT_API int im_sort_region_sharpness(
    const uint8_t *gray,
    uint32_t width,
    uint32_t height,
    uint32_t x,
    uint32_t y,
    uint32_t roi_width,
    uint32_t roi_height,
    double *score_out);

/* Scores a list of ROIs against one shared grayscale buffer. */
IMPRINT_SORT_API int im_sort_region_sharpness_batch(
    const uint8_t *gray,
    uint32_t width,
    uint32_t height,
    const im_sort_roi *rois,
    uint32_t roi_count,
    double *scores_out);

/* Returns 0 on success, 1 for invalid arguments, or 2 for an internal error. */
IMPRINT_SORT_API int im_sort_exposure_score(
    const uint8_t *gray,
    uint32_t width,
    uint32_t height,
    double *score_out);

#ifdef __cplusplus
}
#endif

#endif
