// Validation seam only; all image computation belongs to the native module.
#include "acr_stage3_core3_prefix.hpp"

#include <cstddef>
#include <cstring>

extern "C" int acr_test_core3_prefix(
    const float* input, std::size_t input_samples,
    std::size_t width, std::size_t height, std::size_t input_stride,
    float* output, std::size_t output_samples) noexcept {
    using namespace imprint::acr;
    if (output == nullptr) return static_cast<int>(Core3PrefixStatus::invalid_argument);
    Core3PrefixWorkspace candidate;
    const auto status = run_core3_gradient_energy_prefix(
        input, input_samples, width, height, input_stride, &candidate);
    if (status != Core3PrefixStatus::ok) return static_cast<int>(status);
    if (output_samples < candidate.samples.size()) {
        return static_cast<int>(Core3PrefixStatus::buffer_too_small);
    }
    std::memcpy(output, candidate.samples.data(), candidate.samples.size() * sizeof(float));
    return 0;
}
