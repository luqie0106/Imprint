#include "imprint_renderer.h"
#include "acr_exposure.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

im_status im_native_acr_prepare_exposure(const double *controls, size_t count,
                                        double *output, size_t output_count) {
    if (!controls || !output || count != 10 || output_count != 19)
        return IM_STATUS_INVALID_ARGUMENT;
    const auto a = reinterpret_cast<uintptr_t>(controls);
    const auto b = reinterpret_cast<uintptr_t>(output);
    if (a > std::numeric_limits<uintptr_t>::max() - 10 * sizeof(double) ||
        b > std::numeric_limits<uintptr_t>::max() - 19 * sizeof(double) ||
        (a < b + 19 * sizeof(double) && b < a + 10 * sizeof(double)))
        return IM_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < 10; ++i) {
        if (!std::isfinite(controls[i])) return IM_STATUS_INVALID_ARGUMENT;
    }
    if (controls[0] != std::trunc(controls[0]) || controls[0] < -1000 || controls[0] > 1000 ||
        controls[1] != std::trunc(controls[1]) || controls[1] < -100 || controls[1] > 100 ||
        controls[5] != std::trunc(controls[5]) || controls[5] < 4096 || controls[5] > 65535 ||
        (controls[6] != 0 && controls[6] != 1) ||
        (controls[7] != 0 && controls[7] != 1)) return IM_STATUS_INVALID_ARGUMENT;
    const imprint::acr::ExposureInput input{
        static_cast<int32_t>(controls[0]), static_cast<int32_t>(controls[1]),
        controls[2], controls[3], controls[4], static_cast<uint32_t>(controls[5]),
        controls[6] != 0, controls[7] != 0, controls[8], controls[9],
    };
    imprint::acr::ExposurePrepared p;
    if (imprint::acr::prepare_exposure(input, p) != imprint::acr::ExposureStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;
    const double values[19] = {
        p.manual_exposure_ev, p.whites_ev, p.source_exposure_ev, p.source_scale,
        p.context_ev50, p.limit_ev88, p.profile_gain_amount, p.first_arg, p.second_arg,
        p.exposure_gain, p.ev90, p.ev98, p.ev100, p.ev108, p.ev112,
        p.context_ev18_change, p.camera_scale, p.rounded_source_limit,
        static_cast<double>(p.source_limit_u16),
    };
    for (size_t i = 0; i < 19; ++i) output[i] = values[i];
    return IM_STATUS_OK;
}
