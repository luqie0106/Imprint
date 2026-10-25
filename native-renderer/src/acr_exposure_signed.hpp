#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class SignedExposureStatus : int {
    ok = 0,
    invalid_argument = 1,
    overflow = 2,
    lut_out_of_bounds = 3,
    allocation_failure = 4,
    unsupported_aliasing = 5,
};

struct SignedExposureLut {
    const float *values = nullptr;
    std::size_t capacity = 0;
    std::uint32_t max_index = 0;
    float scale = 1.0f;
};

struct PreparedSignedExposure {
    float gain = 1.0f;
    float compression = 1.0f;
    float expansion = 1.0f;
    float epsilon = 1.0e-6f;
    SignedExposureLut channel[3]{};
    SignedExposureLut gray{};
};

// Applies the recovered signed-exposure path to finite compact interleaved
// RGB float pixels. Input and output may be exactly aliased; all results are
// staged so an error leaves output byte-for-byte unchanged.
SignedExposureStatus apply_prepared_signed_exposure_rgb(
    const float *input, float *output, std::size_t pixel_count,
    const float *mask, std::size_t mask_count,
    const PreparedSignedExposure *prepared) noexcept;

}  // namespace imprint::acr
