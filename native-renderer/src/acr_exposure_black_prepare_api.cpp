#include "imprint_renderer.h"
#include "acr_exposure_black_prepare.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

extern "C" IMPRINT_API im_status im_native_acr_prepare_black_response(
    const double* controls, size_t control_count,
    double* output, size_t output_count) {
    if (control_count != 6 || controls == nullptr || output == nullptr)
        return IM_STATUS_INVALID_ARGUMENT;
    if (output_count < 3) return IM_STATUS_BUFFER_TOO_SMALL;
    const auto in = reinterpret_cast<std::uintptr_t>(controls);
    const auto out = reinterpret_cast<std::uintptr_t>(output);
    constexpr size_t input_bytes = 6 * sizeof(double);
    if (in % alignof(double) != 0 || out % alignof(double) != 0 ||
        input_bytes > std::numeric_limits<std::uintptr_t>::max() - in ||
        output_count > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double))
        return IM_STATUS_INVALID_ARGUMENT;
    const size_t output_bytes = output_count * sizeof(double);
    if (output_bytes > std::numeric_limits<std::uintptr_t>::max() - out ||
        (in < out + output_bytes && out < in + input_bytes))
        return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < 6; ++i)
        if (!std::isfinite(controls[i])) return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < 2; ++i)
        if (controls[i] < -2147483648.0 || controls[i] > 2147483647.0 ||
            std::trunc(controls[i]) != controls[i])
            return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 2; i < 4; ++i)
        if (controls[i] != 0.0 && controls[i] != 1.0)
            return IM_STATUS_INVALID_ARGUMENT;

    imprint::acr::PreparedBlackResponse candidate;
    const auto status = imprint::acr::prepare_exposure_black_response(
        static_cast<std::int32_t>(controls[0]), static_cast<std::int32_t>(controls[1]),
        controls[2] != 0.0, controls[3] != 0.0, controls[4], controls[5], &candidate);
    if (status != imprint::acr::BlackPrepareStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;
    const double result[] = {candidate.z, candidate.post_strength, candidate.channel_strength};
    std::memcpy(output, result, sizeof(result));
    return IM_STATUS_OK;
}
