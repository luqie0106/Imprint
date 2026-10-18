#pragma once

#include "acr_anchor_geometry.hpp"

namespace imprint::acr {

// The four offset_30/38/40/48 values are returned by the already-recovered
// 0126b1dc preparation step. offset_e0/e8 are the source object's directly
// read double scales. Names retain object offsets and do not assign product
// metadata semantics to the values.
struct PreparedSourceCrop {
    double offset_30 = 0.0;
    double offset_38 = 0.0;
    double offset_40 = 0.0;
    double offset_48 = 0.0;
    double offset_e0 = 0.0;
    double offset_e8 = 0.0;
    bool has_level0_image = false;
    AnchorRectI32 level0_image_bounds{};
};

enum class SourceCropStatus : int {
    ok = 0,
    invalid_input = 1,
    round_int32_overflow = 2,
    coordinate_overflow = 3,
};

// Reconstructs FUN_0130c808's level-0 reference rectangle from the six
// prepared values. Bitmap bounds contribute only their nonnegative extents;
// their origin is deliberately not added to the reference rectangle.
// Arithmetic that would wrap an Adobe int32 coordinate is reported as
// coordinate_overflow. On every failure, `result` is left unchanged.
SourceCropStatus compute_source_level0_reference_bounds(
    const PreparedSourceCrop &input, AnchorRectI32 &result) noexcept;

}  // namespace imprint::acr
