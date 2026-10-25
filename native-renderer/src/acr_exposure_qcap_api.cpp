#include "imprint_renderer.h"
#include "acr_exposure_qcap.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

bool valid_double_spans(const double* input, size_t input_count,
                        const double* output, size_t output_count) noexcept {
    if (input == nullptr || output == nullptr ||
        input_count > std::numeric_limits<size_t>::max() / sizeof(double) ||
        output_count > std::numeric_limits<size_t>::max() / sizeof(double) ||
        output_count > static_cast<size_t>(
                           std::numeric_limits<std::ptrdiff_t>::max()) /
                           sizeof(double)) {
        return false;
    }

    const auto input_address = reinterpret_cast<std::uintptr_t>(input);
    const auto output_address = reinterpret_cast<std::uintptr_t>(output);
    if (input_address % alignof(double) != 0 ||
        output_address % alignof(double) != 0) {
        return false;
    }

    const size_t input_bytes = input_count * sizeof(double);
    const size_t output_bytes = output_count * sizeof(double);
    constexpr auto max_address = std::numeric_limits<std::uintptr_t>::max();
    if (input_bytes > max_address - input_address ||
        output_bytes > max_address - output_address) {
        return false;
    }

    const auto input_end = input_address + input_bytes;
    const auto output_end = output_address + output_bytes;
    return !(input_address < output_end && output_address < input_end);
}

imprint::acr::ExposureQcapContext make_context(const double* values) noexcept {
    return imprint::acr::ExposureQcapContext{
        values[0], values[1], values[2], values[3],
        values[4], values[5], values[6], values[7]};
}

}  // namespace

extern "C" IMPRINT_API im_status im_native_acr_prepare_exposure_qcap(
    const double* controls, size_t control_count,
    double* output, size_t output_count) {
    constexpr size_t expected_controls = 8;
    constexpr size_t required_outputs = 3;
    if (control_count != expected_controls || controls == nullptr ||
        output == nullptr) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    if (output_count < required_outputs) return IM_STATUS_BUFFER_TOO_SMALL;
    if (!valid_double_spans(controls, expected_controls, output, output_count)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    double values[expected_controls];
    std::memcpy(values, controls, sizeof(values));
    for (const double value : values) {
        if (!std::isfinite(value)) return IM_STATUS_INVALID_ARGUMENT;
    }

    imprint::acr::ExposureQcapResult candidate{};
    const auto status =
        imprint::acr::prepare_exposure_qcap(make_context(values), candidate);
    if (status != imprint::acr::ExposureQcapStatus::ok ||
        !std::isfinite(candidate.anchor_v) ||
        !std::isfinite(candidate.response) || !std::isfinite(candidate.q)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    const double committed[required_outputs]{
        candidate.anchor_v, candidate.response, candidate.q};
    std::memcpy(output, committed, sizeof(committed));
    return IM_STATUS_OK;
}
