#pragma once

#include "acr_anchor_geometry.hpp"

#include <cstdint>

namespace imprint::acr {

struct UnsignedRational {
    std::uint32_t numerator = 0;
    std::uint32_t denominator = 0;
};

struct IFDCropState {
    std::uint32_t new_subfile_type = 0;
    std::uint32_t image_width = 0;
    std::uint32_t image_height = 0;
    AnchorRectI32 active_area{};
    UnsignedRational crop_origin_x{0, 1};
    UnsignedRational crop_origin_y{0, 1};
    UnsignedRational crop_size_x{0, 0};
    UnsignedRational crop_size_y{0, 0};
};

enum class IFDSourceCropStatus : int {
    ok = 0,
    invalid_input = 1,
    invalid_image_dimensions = 2,
    coordinate_overflow = 3,
};

// Reproduces the supported FUN_0131c06c crop postparse branches. The input is
// the state after tag parsing, so exact NewSubfileType 16 resets are expected
// to have happened before this function is called. On failure, output is left
// unchanged; input and output may alias.
IFDSourceCropStatus finalize_source_ifd_crop(
    const IFDCropState &input,
    IFDCropState &output) noexcept;

}  // namespace imprint::acr
