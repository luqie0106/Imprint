#include "imprint_renderer.h"
#include "acr_exposure_signed.hpp"

#include <cstdint>
#include <limits>

namespace {
bool valid_range(const void* pointer, size_t bytes, size_t alignment) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return pointer != nullptr && address % alignment == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) noexcept {
    const auto av = reinterpret_cast<std::uintptr_t>(a);
    const auto bv = reinterpret_cast<std::uintptr_t>(b);
    return av <= bv ? bv - av < an : av - bv < bn;
}
}  // namespace

extern "C" IMPRINT_API im_status im_native_acr_signed_exposure_prepared_f32(
    const float* source, size_t source_samples, size_t pixel_count,
    const float* controls, size_t control_count,
    const im_acr_exposure_lut_f32* tables, size_t table_count,
    const float* mask, size_t mask_count,
    float* destination, size_t destination_samples) {
    if (pixel_count == 0 || pixel_count > (size_t{1} << 26) ||
        source_samples != pixel_count * 3 || control_count != 4 || table_count != 4)
        return IM_STATUS_INVALID_ARGUMENT;
    if (destination_samples < source_samples) return IM_STATUS_BUFFER_TOO_SMALL;
    if (destination_samples != source_samples) return IM_STATUS_INVALID_ARGUMENT;
    const size_t bytes = source_samples * sizeof(float);
    if (!valid_range(destination, bytes, alignof(float)) ||
        !valid_range(controls, 4 * sizeof(float), alignof(float)) ||
        !valid_range(tables, 4 * sizeof(*tables), alignof(im_acr_exposure_lut_f32)) ||
        overlap(destination, bytes, controls, 4 * sizeof(float)) ||
        overlap(destination, bytes, tables, 4 * sizeof(*tables)))
        return IM_STATUS_INVALID_ARGUMENT;
    imprint::acr::PreparedSignedExposure prepared;
    prepared.gain = controls[0];
    prepared.compression = controls[1];
    prepared.expansion = controls[2];
    prepared.epsilon = controls[3];
    for (size_t i = 0; i < 4; ++i) {
        auto& target = i < 3 ? prepared.channel[i] : prepared.gray;
        target.values = tables[i].values;
        target.capacity = tables[i].count;
        target.max_index = tables[i].max_index;
        target.scale = tables[i].scale;
    }
    using imprint::acr::SignedExposureStatus;
    const auto status = imprint::acr::apply_prepared_signed_exposure_rgb(
        source, destination, pixel_count, mask, mask_count, &prepared);
    switch (status) {
        case SignedExposureStatus::ok: return IM_STATUS_OK;
        case SignedExposureStatus::allocation_failure: return IM_STATUS_RUNTIME_ERROR;
        default: return IM_STATUS_INVALID_ARGUMENT;
    }
}
