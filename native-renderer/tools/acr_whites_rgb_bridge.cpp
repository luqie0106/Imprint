#include "../src/acr_exposure_rgb.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>

extern "C" int acr_test_whites_rgb(
    const float *input, std::size_t samples,
    const float *controls, std::size_t control_count,
    const float *lut, std::size_t lut_count,
    const float *whites_lut, std::size_t whites_count,
    float *output, std::size_t output_count) noexcept {
    const auto maximum = std::numeric_limits<std::uintptr_t>::max();
    const auto ca = reinterpret_cast<std::uintptr_t>(controls);
    const auto oa = reinterpret_cast<std::uintptr_t>(output);
    if (!controls || ca % alignof(float) || ca > maximum - 7 * sizeof(float) ||
        control_count != 7 || samples % 3 || output_count != samples ||
        samples > std::numeric_limits<std::size_t>::max() / sizeof(float)) return 1;
    const auto output_bytes = samples * sizeof(float);
    if (output_bytes && (!output || oa % alignof(float) ||
                         output_bytes > maximum - oa)) return 1;
    if (output_bytes && ca < oa + output_bytes && oa < ca + 7 * sizeof(float)) return 1;
    imprint::acr::PreparedExposureRgb prepared;
    prepared.gain = controls[0];
    prepared.lut_scale = controls[1];
    prepared.rational_gain = controls[2];
    prepared.epsilon = controls[3];
    prepared.positive_gain = controls[4];
    prepared.white_clip_squared = controls[5];
    prepared.whites_lut_scale = controls[6];
    prepared.lut = lut;
    prepared.lut_size = lut_count;
    prepared.whites_lut = whites_lut;
    prepared.whites_lut_size = whites_count;
    return static_cast<int>(imprint::acr::apply_prepared_exposure_rgb(
        input, output, samples / 3, &prepared));
}
