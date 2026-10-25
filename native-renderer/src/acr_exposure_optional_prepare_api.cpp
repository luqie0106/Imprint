#include "imprint_renderer.h"
#include "acr_exposure_optional_prepare.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr std::size_t kControlCount = 18;
constexpr std::size_t kOutputCount = 18;

bool valid_spans(const double* controls, double* output,
                 std::size_t output_count) noexcept {
    if (controls == nullptr || output == nullptr ||
        output_count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        output_count > static_cast<std::size_t>(
                           std::numeric_limits<std::ptrdiff_t>::max()) /
                           sizeof(double)) {
        return false;
    }

    const auto input_address = reinterpret_cast<std::uintptr_t>(controls);
    const auto output_address = reinterpret_cast<std::uintptr_t>(output);
    if (input_address % alignof(double) != 0 ||
        output_address % alignof(double) != 0) {
        return false;
    }

    constexpr auto max_address = std::numeric_limits<std::uintptr_t>::max();
    constexpr std::size_t input_bytes = kControlCount * sizeof(double);
    const std::size_t output_bytes = output_count * sizeof(double);
    if (input_bytes > max_address - input_address ||
        output_bytes > max_address - output_address) {
        return false;
    }

    const auto input_end = input_address + input_bytes;
    const auto output_end = output_address + output_bytes;
    return !(input_address < output_end && output_address < input_end);
}

bool exact_int32(double value) noexcept {
    return value >= -2147483648.0 && value <= 2147483647.0 &&
           std::trunc(value) == value;
}

bool exact_flag(double value) noexcept {
    return value == 0.0 || value == 1.0;
}

}  // namespace

extern "C" IMPRINT_API im_status im_native_acr_prepare_exposure_optional(
    const double* controls, size_t control_count,
    double* output, size_t output_count) {
    if (control_count != kControlCount || controls == nullptr ||
        output == nullptr) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    if (output_count < kOutputCount) return IM_STATUS_BUFFER_TOO_SMALL;
    if (!valid_spans(controls, output, output_count))
        return IM_STATUS_INVALID_ARGUMENT;

    double values[kControlCount];
    std::memcpy(values, controls, sizeof(values));
    for (const double value : values) {
        if (!std::isfinite(value)) return IM_STATUS_INVALID_ARGUMENT;
    }
    if (!exact_int32(values[0]) || !exact_int32(values[1]) ||
        !exact_flag(values[2]) || !exact_flag(values[3]) ||
        values[13] < -120.0 || values[13] > 120.0 ||
        values[14] < 0.0 || values[15] < 0.0 || values[16] < 0.0 ||
        !exact_flag(values[17])) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    const imprint::acr::ExposureOptionalControls parsed{
        static_cast<std::int32_t>(values[0]),
        static_cast<std::int32_t>(values[1]),
        values[2] != 0.0,
        values[3] != 0.0,
        values[4],
        imprint::acr::ExposureQcapContext{
            values[5], values[6], values[7], values[8],
            values[9], values[10], values[11], values[12]},
        values[13],
        {values[14], values[15], values[16]},
        values[17] != 0.0};

    imprint::acr::PreparedExposureOptional prepared{};
    const auto status =
        imprint::acr::prepare_exposure_optional(parsed, prepared);
    if (status != imprint::acr::ExposureOptionalStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;

    const auto committed = prepared.as_array();
    std::memcpy(output, committed.data(), sizeof(committed));
    return IM_STATUS_OK;
}
