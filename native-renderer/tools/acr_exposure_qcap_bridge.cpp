#include "../src/acr_exposure_qcap.hpp"
#include "imprint_renderer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

extern "C" im_status im_native_acr_prepare_exposure_qcap(
    const double* controls, size_t control_count,
    double* output, size_t output_count);

namespace {

bool valid_double_spans(const double* input, std::size_t input_count,
                        const double* output, std::size_t output_count) noexcept {
    if (input == nullptr || output == nullptr ||
        input_count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        output_count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        output_count > static_cast<std::size_t>(
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

    const std::size_t input_bytes = input_count * sizeof(double);
    const std::size_t output_bytes = output_count * sizeof(double);
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

extern "C" {

int acr_test_prepare_exposure_qcap(const double* controls,
                                   std::size_t control_count,
                                   double* output,
                                   std::size_t output_count) noexcept {
    return static_cast<int>(im_native_acr_prepare_exposure_qcap(
        controls, control_count, output, output_count));
}

int acr_test_exposure_context_response(const double* controls,
                                       std::size_t control_count,
                                       double input, double* output,
                                       std::size_t output_count) noexcept {
    constexpr std::size_t expected_controls = 8;
    constexpr std::size_t required_outputs = 1;
    if (control_count != expected_controls || controls == nullptr ||
        output == nullptr) {
        return static_cast<int>(imprint::acr::ExposureQcapStatus::invalid_argument);
    }
    if (output_count < required_outputs ||
        !valid_double_spans(controls, expected_controls, output, output_count)) {
        return static_cast<int>(imprint::acr::ExposureQcapStatus::invalid_argument);
    }

    double values[expected_controls];
    std::memcpy(values, controls, sizeof(values));
    imprint::acr::ExposureQcapContext context = make_context(values);
    double candidate = 0.0;
    const auto status = imprint::acr::evaluate_exposure_context_response(
        context, input, candidate);
    if (status != imprint::acr::ExposureQcapStatus::ok) {
        return static_cast<int>(status);
    }

    std::memcpy(output, &candidate, sizeof(candidate));
    return static_cast<int>(status);
}

}  // extern "C"
