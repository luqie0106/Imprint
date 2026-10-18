#include "acr_source_crop_metadata.hpp"

#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

bool is_all_zero(const AnchorRectI32 &rect) noexcept {
    return rect.top == 0 && rect.left == 0 && rect.bottom == 0 &&
           rect.right == 0;
}

IFDSourceCropStatus fill_missing_extent(
    std::int32_t end,
    std::int32_t start,
    UnsignedRational &size) noexcept {
    if (size.denominator != 0) return IFDSourceCropStatus::ok;

    if (end < start) {
        size = {0, 1};
        return IFDSourceCropStatus::ok;
    }

    const std::int64_t extent = static_cast<std::int64_t>(end) -
                                static_cast<std::int64_t>(start);
    if (extent > std::numeric_limits<std::int32_t>::max()) {
        return IFDSourceCropStatus::coordinate_overflow;
    }

    size = {static_cast<std::uint32_t>(extent), 1};
    return IFDSourceCropStatus::ok;
}

}  // namespace

IFDSourceCropStatus finalize_source_ifd_crop(
    const IFDCropState &input,
    IFDCropState &output) noexcept {
    constexpr std::uint32_t kMaxSupportedDimension =
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
    if (input.image_width > kMaxSupportedDimension ||
        input.image_height > kMaxSupportedDimension) {
        return IFDSourceCropStatus::invalid_image_dimensions;
    }

    IFDCropState candidate = input;
    if (is_all_zero(candidate.active_area)) {
        candidate.active_area = {
            0,
            0,
            static_cast<std::int32_t>(candidate.image_height),
            static_cast<std::int32_t>(candidate.image_width),
        };
    }

    if (candidate.new_subfile_type != 16) {
        IFDSourceCropStatus status = fill_missing_extent(
            candidate.active_area.right,
            candidate.active_area.left,
            candidate.crop_size_x);
        if (status != IFDSourceCropStatus::ok) return status;

        status = fill_missing_extent(
            candidate.active_area.bottom,
            candidate.active_area.top,
            candidate.crop_size_y);
        if (status != IFDSourceCropStatus::ok) return status;
    }

    output = candidate;
    return IFDSourceCropStatus::ok;
}

}  // namespace imprint::acr
