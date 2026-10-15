#include "../src/acr_colorspec.hpp"
#include "../src/acr_display.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

bool range_nonwrapping(const void *pointer, std::size_t byte_length) noexcept {
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    return byte_length <= std::numeric_limits<std::uintptr_t>::max() - start;
}

}  // namespace

extern "C" int acr_test_project_display_srgb8(
    std::uint32_t width, std::uint32_t height,
    const float *source_rgb, std::size_t source_values,
    const double *normalized_prophoto_to_pcs_q, std::size_t q_values,
    std::uint8_t *output_rgb8, std::size_t output_values) noexcept {
    return static_cast<int>(imprint::acr::project_display_srgb8(
        width, height, source_rgb, source_values,
        normalized_prophoto_to_pcs_q, q_values, output_rgb8, output_values));
}

extern "C" int acr_test_scale_sdr_rgb(
    std::uint32_t width, std::uint32_t height,
    const float *source_rgb, std::size_t source_values, float gain,
    float *destination_rgb, std::size_t destination_values) noexcept {
    return static_cast<int>(imprint::acr::scale_sdr_rgb(
        width, height, source_rgb, source_values, gain,
        destination_rgb, destination_values));
}

// The probe obtains Q through the same native constructor used by the caller;
// production display projection still accepts Q explicitly from its owner.
extern "C" int acr_test_normalized_prophoto_to_pcs_q(
    double *output_q, std::size_t q_values) noexcept {
    if (output_q == nullptr || q_values != 9 ||
        !range_nonwrapping(output_q, q_values * sizeof(double))) {
        return 1;
    }
    std::array<double, 9> inverse{};
    std::array<double, 9> candidate{};
    const auto status = imprint::acr::prepare_normalized_prophoto_inverse(
        inverse, &candidate);
    if (status != imprint::acr::ColorSpecStatus::ok) {
        return static_cast<int>(status);
    }
    std::memcpy(output_q, candidate.data(), candidate.size() * sizeof(double));
    return 0;
}
