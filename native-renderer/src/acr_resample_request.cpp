#include "acr_resample_request.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace imprint::acr {
namespace {

static_assert(sizeof(std::int64_t) == sizeof(std::uint64_t),
              "source request arithmetic requires 64-bit integers");
static_assert(sizeof(std::int32_t) == sizeof(std::uint32_t),
              "source request arithmetic requires 32-bit integers");
static_assert(std::numeric_limits<std::int64_t>::min() ==
                  -std::numeric_limits<std::int64_t>::max() - 1,
              "source request arithmetic requires two's-complement int64_t");
static_assert(std::numeric_limits<std::int32_t>::min() ==
                  -std::numeric_limits<std::int32_t>::max() - 1,
              "source request arithmetic requires two's-complement int32_t");

using Status = OrdinarySourceRequestStatus;

std::int64_t int64_from_bits(std::uint64_t bits) noexcept {
    std::int64_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::int32_t int32_from_bits(std::uint32_t bits) noexcept {
    std::int32_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::int32_t signed_high32(std::int64_t value) noexcept {
    std::uint64_t representation = 0;
    std::memcpy(&representation, &value, sizeof(representation));
    return int32_from_bits(static_cast<std::uint32_t>(representation >> 32));
}

bool checked_rectangle_end(std::int32_t start, std::size_t extent,
                           std::int64_t &end) noexcept {
    if (extent == 0 ||
        extent > static_cast<std::size_t>(
                     std::numeric_limits<std::int32_t>::max()))
        return false;
    end = static_cast<std::int64_t>(start) +
          static_cast<std::int64_t>(extent);
    return end <= std::numeric_limits<std::int32_t>::max();
}

bool valid_rectangle(const DestinationRect &rectangle,
                     std::int64_t &bottom, std::int64_t &right) noexcept {
    return checked_rectangle_end(rectangle.top, rectangle.height, bottom) &&
           checked_rectangle_end(rectangle.left, rectangle.width, right);
}

Status make_filtered_rectangle(const DestinationRect &input, bool vertical,
                               std::int32_t start,
                               std::int32_t exclusive_end,
                               DestinationRect &output) noexcept {
    const std::int64_t span = static_cast<std::int64_t>(exclusive_end) -
                              static_cast<std::int64_t>(start);
    if (span <= 0 || span > std::numeric_limits<std::int32_t>::max())
        return Status::invalid_rectangle;

    DestinationRect staged = input;
    if (vertical) {
        staged.top = start;
        staged.height = static_cast<std::size_t>(span);
    } else {
        staged.left = start;
        staged.width = static_cast<std::size_t>(span);
    }

    std::int64_t bottom = 0;
    std::int64_t right = 0;
    if (!valid_rectangle(staged, bottom, right))
        return Status::invalid_rectangle;
    output = staged;
    return Status::ok;
}

std::int64_t map_q32_wrapping(std::int64_t origin, std::int64_t step,
                              std::int32_t index) noexcept {
    // The saved rectangle methods use 64-bit integer MADD/ADD. Model those
    // operations as explicit modulo-2^64 arithmetic to avoid signed UB.
    const std::uint64_t mapped = static_cast<std::uint64_t>(origin) +
        static_cast<std::uint64_t>(step) *
        static_cast<std::uint64_t>(static_cast<std::int64_t>(index));
    return int64_from_bits(mapped);
}

std::int32_t map_clamped_integer_coordinate(
    const ResampleAxisPlan &axis, std::int32_t index) noexcept {
    std::int64_t mapped = map_q32_wrapping(axis.origin, axis.step, index);
    // Preserve the recovered order: signed 64-bit min with upper, then max
    // with lower, followed by extraction of the signed high word.
    mapped = std::min(mapped, axis.upper);
    mapped = std::max(mapped, axis.lower);
    return signed_high32(mapped);
}

Status reverse_axis_rectangle(
    const DestinationRect &input, const ResampleAxisPlan &axis,
    const ResampleCoefficientTable &coefficients, bool vertical,
    DestinationRect &output) noexcept {
    std::int64_t bottom = 0;
    std::int64_t right = 0;
    if (!valid_rectangle(input, bottom, right))
        return Status::invalid_rectangle;

    switch (axis.kind) {
        case ResampleAxisKind::identity:
            output = input;
            return Status::ok;

        case ResampleAxisKind::integer_translation: {
            const std::int64_t delta =
                -static_cast<std::int64_t>(axis.integer_offset);
            const std::int64_t source_start =
                static_cast<std::int64_t>(vertical ? input.top : input.left) +
                delta;
            const std::int64_t source_end =
                (vertical ? bottom : right) + delta;
            if (source_start < std::numeric_limits<std::int32_t>::min() ||
                source_start > std::numeric_limits<std::int32_t>::max() ||
                source_end < std::numeric_limits<std::int32_t>::min() ||
                source_end > std::numeric_limits<std::int32_t>::max())
                return Status::coordinate_overflow;
            return make_filtered_rectangle(
                input, vertical, static_cast<std::int32_t>(source_start),
                static_cast<std::int32_t>(source_end), output);
        }

        case ResampleAxisKind::resample: {
            if (axis.step <= 0 || axis.lower > axis.upper ||
                coefficients.tap_count == 0 ||
                (coefficients.tap_count & std::size_t{1}) != 0 ||
                coefficients.tap_count > static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max()))
                return Status::invalid_axis;

            const std::int32_t first = vertical ? input.top : input.left;
            const std::int32_t last = static_cast<std::int32_t>(
                (vertical ? bottom : right) - 1);
            const std::int32_t first_mapped =
                map_clamped_integer_coordinate(axis, first);
            const std::int32_t last_mapped =
                map_clamped_integer_coordinate(axis, last);
            const std::int32_t mapped_start =
                std::min(first_mapped, last_mapped);
            const std::int32_t mapped_last =
                std::max(first_mapped, last_mapped);

            // The input axis origin already includes its support/phase bias.
            // The reverse method adds exactly the coefficient tap count to
            // the mapped inclusive high coordinate using 32-bit wrap.
            const std::uint32_t exclusive_bits =
                static_cast<std::uint32_t>(mapped_last) +
                static_cast<std::uint32_t>(coefficients.tap_count);
            const std::int32_t mapped_end = int32_from_bits(exclusive_bits);
            return make_filtered_rectangle(input, vertical, mapped_start,
                                           mapped_end, output);
        }

        default:
            return Status::invalid_axis;
    }
}

}  // namespace

OrdinarySourceRequestStatus prepare_ordinary_source_request(
    const OrdinaryResamplePlan &plan, const DestinationRect &destination,
    OrdinarySourceRequest &output) noexcept {
    std::int64_t bottom = 0;
    std::int64_t right = 0;
    if (!valid_rectangle(destination, bottom, right))
        return Status::invalid_rectangle;

    OrdinarySourceRequest staged{};
    const Status horizontal_status = reverse_axis_rectangle(
        destination, plan.horizontal, plan.horizontal_coefficients, false,
        staged.intermediate);
    if (horizontal_status != Status::ok) return horizontal_status;

    const Status vertical_status = reverse_axis_rectangle(
        staged.intermediate, plan.vertical, plan.vertical_coefficients, true,
        staged.source);
    if (vertical_status != Status::ok) return vertical_status;

    output = staged;
    return Status::ok;
}

}  // namespace imprint::acr
