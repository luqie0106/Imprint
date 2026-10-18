#include "acr_source_crop.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

using Status = SourceCropStatus;

bool finite_prepared_values(const PreparedSourceCrop &input) noexcept {
    return std::isfinite(input.offset_30) && std::isfinite(input.offset_38) &&
           std::isfinite(input.offset_40) && std::isfinite(input.offset_48) &&
           std::isfinite(input.offset_e0) && std::isfinite(input.offset_e8);
}

Status round_product(double value, double scale, std::int32_t &out) noexcept {
    // Keep the two binary64 operations distinct. This mirrors the source's
    // fmul followed by fadd and remains explicit even if a caller builds
    // without the research bridge's -ffp-contract=off flag.
    volatile double product = value * scale;
    const double product_value = product;
    const double bias = product_value > 0.0 ? 0.5 : -0.5;
    volatile double biased_value = product_value + bias;
    const double biased = biased_value;
    if (!std::isfinite(biased) || !(biased > -2147483649.0) ||
        !(biased < 2147483648.0)) {
        return Status::round_int32_overflow;
    }
    out = static_cast<std::int32_t>(biased);  // truncate toward zero
    return Status::ok;
}

bool checked_i32(std::int64_t value, std::int32_t &out) noexcept {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
        return false;
    }
    out = static_cast<std::int32_t>(value);
    return true;
}

Status nonnegative_extent(std::int32_t start, std::int32_t end,
                          std::int32_t &out) noexcept {
    if (end < start) {
        out = 0;
        return Status::ok;
    }
    const std::int64_t extent = static_cast<std::int64_t>(end) - start;
    if (!checked_i32(extent, out)) return Status::coordinate_overflow;
    return Status::ok;
}

}  // namespace

SourceCropStatus compute_source_level0_reference_bounds(
    const PreparedSourceCrop &input, AnchorRectI32 &result) noexcept {
    if (!finite_prepared_values(input)) return Status::invalid_input;

    AnchorRectI32 candidate{};
    std::int32_t width = 0;
    std::int32_t height = 0;
    Status status = round_product(input.offset_30, input.offset_e0, width);
    if (status != Status::ok) return status;
    status = round_product(input.offset_38, input.offset_e8, height);
    if (status != Status::ok) return status;
    status = round_product(input.offset_40, input.offset_e0, candidate.left);
    if (status != Status::ok) return status;
    status = round_product(input.offset_48, input.offset_e8, candidate.top);
    if (status != Status::ok) return status;

    if (!checked_i32(static_cast<std::int64_t>(candidate.left) + width,
                     candidate.right) ||
        !checked_i32(static_cast<std::int64_t>(candidate.top) + height,
                     candidate.bottom)) {
        return Status::coordinate_overflow;
    }

    if (input.has_level0_image) {
        std::int32_t image_width = 0;
        std::int32_t image_height = 0;
        status = nonnegative_extent(input.level0_image_bounds.left,
                                    input.level0_image_bounds.right,
                                    image_width);
        if (status != Status::ok) return status;
        status = nonnegative_extent(input.level0_image_bounds.top,
                                    input.level0_image_bounds.bottom,
                                    image_height);
        if (status != Status::ok) return status;

        if (candidate.right > image_width) {
            candidate.right = image_width;
            if (!checked_i32(static_cast<std::int64_t>(image_width) - width,
                             candidate.left)) {
                return Status::coordinate_overflow;
            }
        }
        if (candidate.bottom > image_height) {
            candidate.bottom = image_height;
            if (!checked_i32(static_cast<std::int64_t>(image_height) - height,
                             candidate.top)) {
                return Status::coordinate_overflow;
            }
        }
    }

    result = candidate;
    return Status::ok;
}

}  // namespace imprint::acr
