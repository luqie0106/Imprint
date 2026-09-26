#include "imprint_sort.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

bool image_size_valid(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return false;
    const size_t max_size = std::numeric_limits<size_t>::max();
    return static_cast<size_t>(width) <= max_size / static_cast<size_t>(height);
}

uint32_t reflect101(int64_t coordinate, uint32_t length) {
    if (length <= 1) return 0;
    if (coordinate < 0) return static_cast<uint32_t>(-coordinate);
    if (coordinate >= static_cast<int64_t>(length)) {
        return static_cast<uint32_t>(2 * static_cast<int64_t>(length) - coordinate - 2);
    }
    return static_cast<uint32_t>(coordinate);
}

inline int pixel(const uint8_t *gray, uint32_t image_width,
                 uint32_t x, uint32_t y) {
    return static_cast<int>(gray[static_cast<size_t>(y) * image_width + x]);
}

} // namespace

extern "C" IMPRINT_SORT_API int im_sort_region_sharpness(
    const uint8_t *gray,
    uint32_t width,
    uint32_t height,
    uint32_t x,
    uint32_t y,
    uint32_t roi_width,
    uint32_t roi_height,
    double *score_out) {
    if (!gray || !score_out || !image_size_valid(width, height) ||
        roi_width == 0 || roi_height == 0 || x >= width || y >= height ||
        roi_width > width - x || roi_height > height - y) {
        return 1;
    }

    const size_t roi_pixels = static_cast<size_t>(roi_width) * roi_height;
    std::vector<int32_t> laplacian;
    try {
        laplacian.resize(roi_pixels);
    } catch (...) {
        return 2;
    }

    double lap_sum = 0.0;
    double gradient_energy_sum = 0.0;
    size_t index = 0;
    for (uint32_t ry = 0; ry < roi_height; ++ry) {
        const uint32_t top_y = reflect101(static_cast<int64_t>(ry) - 1, roi_height);
        const uint32_t bottom_y = reflect101(static_cast<int64_t>(ry) + 1, roi_height);
        for (uint32_t rx = 0; rx < roi_width; ++rx, ++index) {
            const uint32_t left_x = reflect101(static_cast<int64_t>(rx) - 1, roi_width);
            const uint32_t right_x = reflect101(static_cast<int64_t>(rx) + 1, roi_width);

            const int center = pixel(gray, width, x + rx, y + ry);
            const int top = pixel(gray, width, x + rx, y + top_y);
            const int bottom = pixel(gray, width, x + rx, y + bottom_y);
            const int left = pixel(gray, width, x + left_x, y + ry);
            const int right = pixel(gray, width, x + right_x, y + ry);
            const int lap = 4 * center - top - bottom - left - right;
            laplacian[index] = lap;
            lap_sum += static_cast<double>(lap);

            const int top_left = pixel(gray, width, x + left_x, y + top_y);
            const int top_right = pixel(gray, width, x + right_x, y + top_y);
            const int bottom_left = pixel(gray, width, x + left_x, y + bottom_y);
            const int bottom_right = pixel(gray, width, x + right_x, y + bottom_y);
            const int sx = (top_right + 2 * right + bottom_right) -
                           (top_left + 2 * left + bottom_left);
            const int sy = (bottom_left + 2 * bottom + bottom_right) -
                           (top_left + 2 * top + top_right);
            gradient_energy_sum += static_cast<double>(sx * sx + sy * sy);
        }
    }

    const double pixel_count = static_cast<double>(roi_pixels);
    const double lap_mean = lap_sum / pixel_count;
    double lap_variance_sum = 0.0;
    for (int32_t value : laplacian) {
        const double centered = static_cast<double>(value) - lap_mean;
        lap_variance_sum += centered * centered;
    }
    const double score = lap_variance_sum / pixel_count + gradient_energy_sum / pixel_count;
    if (!std::isfinite(score)) return 2;
    *score_out = score;
    return 0;
}

extern "C" IMPRINT_SORT_API int im_sort_region_sharpness_batch(
    const uint8_t *gray,
    uint32_t width,
    uint32_t height,
    const im_sort_roi *rois,
    uint32_t roi_count,
    double *scores_out) {
    if (!gray || !rois || !scores_out || roi_count == 0 ||
        !image_size_valid(width, height)) {
        return 1;
    }
    for (uint32_t i = 0; i < roi_count; ++i) {
        const im_sort_roi &roi = rois[i];
        const int status = im_sort_region_sharpness(
            gray, width, height, roi.x, roi.y, roi.width, roi.height, &scores_out[i]);
        if (status != 0) return status;
    }
    return 0;
}

extern "C" IMPRINT_SORT_API int im_sort_exposure_score(
    const uint8_t *gray,
    uint32_t width,
    uint32_t height,
    double *score_out) {
    if (!gray || !score_out || !image_size_valid(width, height)) return 1;

    const size_t total = static_cast<size_t>(width) * height;
    size_t black_count = 0;
    size_t white_count = 0;
    for (size_t i = 0; i < total; ++i) {
        black_count += gray[i] <= 5;
        white_count += gray[i] >= 250;
    }
    const double pct_black = static_cast<double>(black_count) / static_cast<double>(total);
    const double pct_white = static_cast<double>(white_count) / static_cast<double>(total);
    *score_out = std::clamp(1.0 - pct_white * 2.0 - pct_black * 0.5, 0.0, 1.0);
    return 0;
}
