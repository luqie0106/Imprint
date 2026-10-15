#include "imprint_renderer.h"
#include "acr_flare.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
bool valid_range(const void *pointer, size_t bytes) {
    return pointer && bytes && reinterpret_cast<uintptr_t>(pointer) <=
        std::numeric_limits<uintptr_t>::max() - bytes;
}
bool overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const auto x = reinterpret_cast<uintptr_t>(a);
    const auto y = reinterpret_cast<uintptr_t>(b);
    return x < y + b_bytes && y < x + a_bytes;
}
}

im_status im_native_acr_flare_u16_rgb(
    uint32_t width, uint32_t height, const uint16_t *source, size_t source_samples,
    const double *controls, size_t control_count, double *output, size_t output_count) {
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (!width || !height || pixels > (uint64_t{1} << 26) ||
        source_samples != pixels * 3 || control_count != 7 || output_count != 6)
        return IM_STATUS_INVALID_ARGUMENT;
    const size_t source_bytes = source_samples * sizeof(uint16_t);
    if (!valid_range(source, source_bytes) || !valid_range(controls, 7 * sizeof(double)) ||
        !valid_range(output, 6 * sizeof(double)) ||
        overlaps(source, source_bytes, output, 6 * sizeof(double)) ||
        overlaps(controls, 7 * sizeof(double), output, 6 * sizeof(double)))
        return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < 7; ++i)
        if (!std::isfinite(controls[i]) || controls[i] != std::trunc(controls[i]))
            return IM_STATUS_INVALID_ARGUMENT;
    if (controls[0] < 0 || controls[0] > width || controls[1] < 0 || controls[1] > height ||
        controls[2] < 1 || controls[2] > width || controls[3] < 1 || controls[3] > height ||
        controls[4] < 0 || controls[4] > 65534 ||
        (controls[5] != 0 && controls[5] != 1) || (controls[6] != 0 && controls[6] != 1))
        return IM_STATUS_INVALID_ARGUMENT;
    const acr_flare::Crop crop{static_cast<size_t>(controls[0]), static_cast<size_t>(controls[1]),
                               static_cast<size_t>(controls[2]), static_cast<size_t>(controls[3])};
    acr_flare::Result result;
    const auto status = acr_flare::calculate(source, width, height, crop,
        static_cast<uint16_t>(controls[4]), controls[5] == 1, controls[6] == 1, &result);
    if (status == acr_flare::Status::allocation_failure) return IM_STATUS_RUNTIME_ERROR;
    if (status != acr_flare::Status::ok) return IM_STATUS_INVALID_ARGUMENT;
    const double values[6] = {result.mean, result.flare, result.epsilon,
                             static_cast<double>(result.last_width),
                             static_cast<double>(result.last_height),
                             static_cast<double>(result.level_count)};
    std::memcpy(output, values, sizeof(values));
    return IM_STATUS_OK;
}
