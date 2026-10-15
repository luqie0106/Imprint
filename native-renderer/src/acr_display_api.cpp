#include "imprint_renderer.h"
#include "acr_display.hpp"

namespace {
im_status display_status(imprint::acr::DisplayStatus status) {
    switch (status) {
    case imprint::acr::DisplayStatus::ok: return IM_STATUS_OK;
    case imprint::acr::DisplayStatus::allocation_failure: return IM_STATUS_RUNTIME_ERROR;
    default: return IM_STATUS_INVALID_ARGUMENT;
    }
}
}

im_status im_native_acr_project_srgb_rgb8(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    const double *normalized_prophoto_to_pcs, size_t matrix_count,
    uint8_t *destination, size_t destination_samples) {
    return display_status(imprint::acr::project_display_srgb8(
        width, height, source, source_samples, normalized_prophoto_to_pcs,
        matrix_count, destination, destination_samples));
}

im_status im_native_acr_scale_sdr_rgbf32(
    uint32_t width, uint32_t height, const float *source, size_t source_samples,
    float gain, float *destination, size_t destination_samples) {
    return display_status(imprint::acr::scale_sdr_rgb(
        width, height, source, source_samples, gain, destination, destination_samples));
}
