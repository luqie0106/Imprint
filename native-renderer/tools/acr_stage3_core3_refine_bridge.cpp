// Validation seam only; all image computation remains in the native kernel.
#include "acr_stage3_core3_refine.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int core3_refine_bridge(
    std::uint32_t kernel,
    const float* guide,
    const float* channel_a,
    const float* channel_b,
    const float* mask,
    std::size_t stride,
    std::size_t width,
    std::uint32_t phase,
    const float* controls,
    float* output_guide,
    float* output_a,
    float* output_b) noexcept {
    using namespace imprint::acr;
    if (controls == nullptr) {
        return static_cast<int>(Core3RefineStatus::invalid_argument);
    }

    const RefineInputs inputs{
        guide, channel_a, channel_b, mask, stride, width, phase,
        controls[0], controls[1]};
    const RefineOutputs outputs{output_guide, output_a, output_b};
    switch (kernel) {
        case 0:
            return static_cast<int>(
                core3_refine_prepared_float_row_00814230(inputs, outputs));
        case 1:
            return static_cast<int>(
                core3_refine_prepared_float_row_008145b4(inputs, outputs));
        default:
            return static_cast<int>(Core3RefineStatus::invalid_argument);
    }
}
