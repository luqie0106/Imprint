#include "imprint_renderer.h"
#include "acr_exposure_context_lut.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

extern "C" IMPRINT_API im_status im_native_acr_prepare_exposure_context_lut(
    uint32_t kind, const double* controls, size_t control_count,
    float* output, size_t output_count) {
    const size_t expected_controls = kind == 1 ? 3 : kind == 2 ? 1 : 0;
    if (expected_controls == 0 || control_count != expected_controls ||
        controls == nullptr || output == nullptr)
        return IM_STATUS_INVALID_ARGUMENT;
    if (output_count < imprint::acr::kExposureContextLutSize)
        return IM_STATUS_BUFFER_TOO_SMALL;
    const auto input_address = reinterpret_cast<std::uintptr_t>(controls);
    const auto output_address = reinterpret_cast<std::uintptr_t>(output);
    const size_t input_bytes = expected_controls * sizeof(double);
    if (input_address % alignof(double) != 0 ||
        output_address % alignof(float) != 0 ||
        input_bytes > std::numeric_limits<std::uintptr_t>::max() - input_address ||
        output_count > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float))
        return IM_STATUS_INVALID_ARGUMENT;
    const size_t output_bytes = output_count * sizeof(float);
    if (output_bytes > std::numeric_limits<std::uintptr_t>::max() - output_address ||
        (input_address < output_address + output_bytes &&
         output_address < input_address + input_bytes))
        return IM_STATUS_INVALID_ARGUMENT;

    std::vector<float> table;
    const auto status = kind == 1
        ? imprint::acr::build_exposure_channel_lut(controls[0], controls[1], controls[2], table)
        : imprint::acr::build_exposure_post_lut(controls[0], table);
    using imprint::acr::ExposureContextLutStatus;
    if (status == ExposureContextLutStatus::allocation_failure)
        return IM_STATUS_RUNTIME_ERROR;
    if (status != ExposureContextLutStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;
    if (table.size() != imprint::acr::kExposureContextLutSize)
        return IM_STATUS_RUNTIME_ERROR;
    std::memcpy(output, table.data(), table.size() * sizeof(float));
    return IM_STATUS_OK;
}
