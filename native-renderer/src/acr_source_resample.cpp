#include "acr_source_resample.hpp"

#include "acr_resample_request.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

bool checked_multiply(std::size_t lhs, std::size_t rhs,
                      std::size_t &product) noexcept {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
        return false;
    product = lhs * rhs;
    return true;
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

bool valid_source(const SourceView &source, std::int64_t &bottom,
                  std::int64_t &right,
                  std::size_t &row_stride) noexcept {
    if (!source.data || source.height == 0 || source.width == 0 ||
        (source.channels != 1 && source.channels != 3) ||
        !checked_rectangle_end(source.top, source.height, bottom) ||
        !checked_rectangle_end(source.left, source.width, right) ||
        !checked_multiply(source.width, source.channels, row_stride))
        return false;

    std::size_t expected_count = 0;
    if (!checked_multiply(source.height, row_stride, expected_count) ||
        source.count < expected_count || source.count == 0 ||
        source.count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;

    const std::size_t byte_span = source.count * sizeof(float);
    if (byte_span > static_cast<std::size_t>(
                        std::numeric_limits<std::ptrdiff_t>::max()))
        return false;

    const auto address = reinterpret_cast<std::uintptr_t>(source.data);
    return address % alignof(float) == 0 &&
           byte_span <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool intersects(std::int64_t first_top, std::int64_t first_left,
                std::int64_t first_bottom, std::int64_t first_right,
                std::int64_t second_top, std::int64_t second_left,
                std::int64_t second_bottom,
                std::int64_t second_right) noexcept {
    return std::max(first_top, second_top) <
               std::min(first_bottom, second_bottom) &&
           std::max(first_left, second_left) <
               std::min(first_right, second_right);
}

}  // namespace

OrdinarySourceMode2Status process_ordinary_source_mode2_f32(
    const SourceView &source, const DestinationRect &destination,
    const OrdinaryResamplePlan &plan,
    std::vector<float> &result) noexcept {
    std::int64_t source_bottom = 0;
    std::int64_t source_right = 0;
    std::size_t source_row_stride = 0;
    if (!valid_source(source, source_bottom, source_right, source_row_stride))
        return OrdinarySourceMode2Status::invalid_source;

    OrdinarySourceRequest request{};
    if (prepare_ordinary_source_request(plan, destination, request) !=
        OrdinarySourceRequestStatus::ok)
        return OrdinarySourceMode2Status::request_failure;

    const auto &requested = request.source;
    std::int64_t requested_bottom = 0;
    std::int64_t requested_right = 0;
    if (!checked_rectangle_end(requested.top, requested.height,
                               requested_bottom) ||
        !checked_rectangle_end(requested.left, requested.width,
                               requested_right))
        return OrdinarySourceMode2Status::request_failure;

    if (!intersects(requested.top, requested.left, requested_bottom,
                    requested_right, source.top, source.left, source_bottom,
                    source_right))
        return OrdinarySourceMode2Status::unsupported_disjoint_request;

    std::size_t requested_row_stride = 0;
    std::size_t requested_count = 0;
    if (!checked_multiply(requested.width, source.channels,
                          requested_row_stride) ||
        !checked_multiply(requested.height, requested_row_stride,
                          requested_count) ||
        requested_row_stride > kResampleSeparableMaximumElements ||
        requested_count > kResampleSeparableMaximumElements)
        return OrdinarySourceMode2Status::resource_limit;

    try {
        std::vector<float> materialized(requested_count);
        const std::size_t pixel_bytes = source.channels * sizeof(float);
        for (std::size_t y = 0; y < requested.height; ++y) {
            const std::int64_t absolute_y =
                static_cast<std::int64_t>(requested.top) +
                static_cast<std::int64_t>(y);
            const std::int64_t source_y = std::clamp(
                absolute_y, static_cast<std::int64_t>(source.top),
                source_bottom - 1);
            const std::size_t source_row =
                static_cast<std::size_t>(source_y - source.top) *
                source_row_stride;
            const std::size_t destination_row = y * requested_row_stride;

            for (std::size_t x = 0; x < requested.width; ++x) {
                const std::int64_t absolute_x =
                    static_cast<std::int64_t>(requested.left) +
                    static_cast<std::int64_t>(x);
                const std::int64_t source_x = std::clamp(
                    absolute_x, static_cast<std::int64_t>(source.left),
                    source_right - 1);
                const std::size_t source_offset =
                    source_row +
                    static_cast<std::size_t>(source_x - source.left) *
                        source.channels;
                const std::size_t destination_offset =
                    destination_row + x * source.channels;
                std::memcpy(materialized.data() + destination_offset,
                            source.data + source_offset, pixel_bytes);
            }
        }

        const SourceView staged_source{
            materialized.data(), materialized.size(), requested.top,
            requested.left, requested.height, requested.width,
            source.channels};
        std::vector<float> staged_result;
        const auto execution_status = process_ordinary_resample(
            staged_source, destination, plan, staged_result);
        if (execution_status == ResampleSeparableStatus::allocation_failure)
            return OrdinarySourceMode2Status::allocation_failure;
        if (execution_status != ResampleSeparableStatus::ok)
            return OrdinarySourceMode2Status::execution_failure;

        result.swap(staged_result);
        return OrdinarySourceMode2Status::ok;
    } catch (const std::bad_alloc &) {
        return OrdinarySourceMode2Status::allocation_failure;
    } catch (const std::length_error &) {
        // A container length rejection means the requested local allocation is
        // outside this bounded materialization domain, rather than OOM.
        return OrdinarySourceMode2Status::resource_limit;
    } catch (...) {
        return OrdinarySourceMode2Status::execution_failure;
    }
}

}  // namespace imprint::acr
