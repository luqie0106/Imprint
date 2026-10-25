#include "imprint_renderer.h"
#include "acr_exposure_epsilon.hpp"

extern "C" IMPRINT_API im_status im_native_acr_prepare_exposure_pixel_epsilon(
    uint32_t colorimetric_reference, double source_flare, double original_ev90,
    float* output, size_t output_count) {
    if (output_count < 1) return IM_STATUS_BUFFER_TOO_SMALL;
    const auto status = imprint::acr::prepare_exposure_pixel_epsilon(
        colorimetric_reference, source_flare, original_ev90, output);
    return status == imprint::acr::ExposureEpsilonStatus::ok
        ? IM_STATUS_OK : IM_STATUS_INVALID_ARGUMENT;
}
