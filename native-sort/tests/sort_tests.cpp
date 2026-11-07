#include "imprint_sort.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {
int fail(const char *message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}
#define REQUIRE(condition, message) do { if (!(condition)) return fail(message); } while (0)
}

int main() {
    double score = -1;
    const uint8_t constant[]{127, 127, 127, 127};
    REQUIRE(im_sort_region_sharpness(nullptr, 2, 2, 0, 0, 2, 2, &score) == 1 && score == -1,
            "null image is rejected without changing output");
    REQUIRE(im_sort_region_sharpness(constant, 2, 2, 0, 0, 2, 2, nullptr) == 1,
            "null score destination is rejected");
    REQUIRE(im_sort_region_sharpness(constant, 2, 2, 1, 0, 2, 2, &score) == 1,
            "out-of-bounds ROI is rejected");
    REQUIRE(im_sort_region_sharpness(constant, 2, 2, 0, 0,
                std::numeric_limits<uint32_t>::max(), 2, &score) == 1,
            "overflow-sized ROI is rejected");
    REQUIRE(im_sort_region_sharpness(constant, 2, 2, 0, 0, 2, 2, &score) == 0 && score == 0,
            "constant image has zero sharpness");
    const uint8_t single[]{255};
    REQUIRE(im_sort_region_sharpness(single, 1, 1, 0, 0, 1, 1, &score) == 0 && score == 0,
            "single-pixel reflect borders are valid");
    const std::array<uint8_t, 9> textured{0, 0, 255, 0, 255, 255, 0, 0, 255};
    const auto original = textured;
    const im_sort_roi regions[]{{0, 0, 3, 3}, {1, 0, 2, 3}};
    double batch[]{-1, -1, 12345};
    REQUIRE(im_sort_region_sharpness_batch(textured.data(), 3, 3, regions, 2, batch) == 0,
            "batch scoring succeeds");
    for (size_t index = 0; index < 2; ++index) {
        const auto &roi = regions[index];
        REQUIRE(im_sort_region_sharpness(textured.data(), 3, 3, roi.x, roi.y,
                    roi.width, roi.height, &score) == 0 && score == batch[index] && score > 0,
                "batch sharpness equals individual textured ROI score");
    }
    REQUIRE(batch[2] == 12345 && textured == original,
            "batch scoring preserves input and destination canary");
    REQUIRE(im_sort_region_sharpness_batch(constant, 2, 2, regions, 0, batch) == 1,
            "empty batch is rejected");
    const uint8_t exposure[]{0, 5, 6, 249, 250, 255, 100, 200};
    REQUIRE(im_sort_exposure_score(exposure, 4, 2, &score) == 0 &&
                std::abs(score - 0.375) < 1e-15, "exposure thresholds and weights are stable");
    const uint8_t white[]{255};
    REQUIRE(im_sort_exposure_score(white, 1, 1, &score) == 0 && score == 0,
            "overexposure score is clamped to zero");
    REQUIRE(im_sort_exposure_score(exposure, 0, 2, &score) == 1,
            "empty image is rejected");
    std::puts("PASS: native sort validation, ROI, batch and exposure contracts");
    return 0;
}
