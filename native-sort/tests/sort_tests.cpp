#include "imprint_sort.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int fail(const char *message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

#define REQUIRE(condition, message) do { if (!(condition)) return fail(message); } while (0)

} // namespace

int main() {
    double score = -1.0;
    REQUIRE(im_sort_region_sharpness(nullptr, 1, 1, 0, 0, 1, 1, &score) == 1,
            "null input is rejected");
    const uint8_t constant[] = {127, 127, 127, 127};
    REQUIRE(im_sort_region_sharpness(constant, 2, 2, 0, 0, 2, 2, &score) == 0,
            "constant image is scored");
    REQUIRE(score == 0.0, "constant image has zero sharpness");
    REQUIRE(im_sort_region_sharpness(constant, 2, 2, 1, 0, 2, 2, &score) == 1,
            "out-of-bounds ROI is rejected");
    const im_sort_roi regions[] = {{0, 0, 2, 1}, {0, 1, 2, 1}};
    double batch_scores[2] = {-1.0, -1.0};
    REQUIRE(im_sort_region_sharpness_batch(
                constant, 2, 2, regions, 2, batch_scores) == 0,
            "batch regions are scored against one image");
    REQUIRE(batch_scores[0] == 0.0 && batch_scores[1] == 0.0,
            "batch scores match the constant image result");

    const uint8_t exposure[] = {0, 5, 6, 249, 250, 255, 100, 200};
    REQUIRE(im_sort_exposure_score(exposure, 4, 2, &score) == 0,
            "exposure image is scored");
    const double expected = 1.0 - (2.0 / 8.0) * 2.0 - (2.0 / 8.0) * 0.5;
    REQUIRE(std::abs(score - expected) < 1e-15, "exposure thresholds and weights match");
    REQUIRE(im_sort_exposure_score(exposure, 0, 2, &score) == 1,
            "zero-width image is rejected");

    std::puts("PASS: native sort C ABI contract");
    return 0;
}
