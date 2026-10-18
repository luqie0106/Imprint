#include "../src/acr_source_crop.hpp"

#include <cstdint>

extern "C" int acr_test_compute_source_level0_reference_bounds(
    const double offsets[6], std::int32_t has_level0_image,
    const std::int32_t level0_image_bounds[4], std::int32_t output[4]) {
    using namespace imprint::acr;
    if (offsets == nullptr || output == nullptr ||
        (has_level0_image != 0 && has_level0_image != 1) ||
        (has_level0_image != 0 && level0_image_bounds == nullptr)) {
        return static_cast<int>(SourceCropStatus::invalid_input);
    }

    PreparedSourceCrop input{};
    input.offset_30 = offsets[0];
    input.offset_38 = offsets[1];
    input.offset_40 = offsets[2];
    input.offset_48 = offsets[3];
    input.offset_e0 = offsets[4];
    input.offset_e8 = offsets[5];
    input.has_level0_image = has_level0_image != 0;
    if (input.has_level0_image) {
        input.level0_image_bounds = {
            level0_image_bounds[0], level0_image_bounds[1],
            level0_image_bounds[2], level0_image_bounds[3],
        };
    }

    AnchorRectI32 result{};
    const SourceCropStatus status =
        compute_source_level0_reference_bounds(input, result);
    if (status != SourceCropStatus::ok) return static_cast<int>(status);

    const std::int32_t candidate[4] = {
        result.top, result.left, result.bottom, result.right,
    };
    for (int index = 0; index < 4; ++index) output[index] = candidate[index];
    return static_cast<int>(SourceCropStatus::ok);
}
