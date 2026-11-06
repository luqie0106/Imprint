#include "../src/acr_anchor_geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

extern "C" {

struct AcrAnchorLevel0FilterInput {
    std::int32_t non_null;
    std::int32_t rects[16];  // +58, +78, +68, +88; each [t,l,b,r]
};

// Test-only prepared-data bridge. Spans are expressed in bytes so malformed
// counts can be rejected without claiming or emulating Adobe's private ABI.
// The output is in/out so helper failure atomicity is directly observable.
// Core statuses are 0..3; 4 is this bridge's allocation-failure result only.
int acr_test_compute_anchor_level0_bounds(
    std::int32_t transform_actual,
    std::int32_t transform_reference,
    const std::int32_t *actual_level0_bounds,
    std::size_t actual_bounds_span_bytes,
    const std::int32_t *reference_level0_bounds,
    std::size_t reference_bounds_span_bytes,
    const AcrAnchorLevel0FilterInput *filters,
    std::size_t filter_count,
    std::size_t filter_span_bytes,
    std::int32_t *output,
    std::size_t output_span_bytes) {
    using namespace imprint::acr;
    constexpr std::size_t kRectBytes = 4 * sizeof(std::int32_t);
    constexpr std::size_t kOutputBytes = 8 * sizeof(std::int32_t);

    if ((transform_actual != 0 && transform_actual != 1) ||
        (transform_reference != 0 && transform_reference != 1) ||
        actual_level0_bounds == nullptr ||
        actual_bounds_span_bytes < kRectBytes ||
        reference_level0_bounds == nullptr ||
        reference_bounds_span_bytes < kRectBytes || output == nullptr ||
        output_span_bytes < kOutputBytes) {
        return static_cast<int>(AnchorGeometryStatus::invalid_input);
    }

    PreparedAnchorLevel0Bounds input{};
    input.actual_level0_bounds = {
        actual_level0_bounds[0], actual_level0_bounds[1],
        actual_level0_bounds[2], actual_level0_bounds[3],
    };
    input.reference_level0_bounds = {
        reference_level0_bounds[0], reference_level0_bounds[1],
        reference_level0_bounds[2], reference_level0_bounds[3],
    };
    input.transform_actual = transform_actual != 0;
    input.transform_reference = transform_reference != 0;

    // A disabled chain does not consume or validate filters. In particular,
    // a null pointer and arbitrary count/span are irrelevant when both gates
    // are off.
    if (input.transform_actual || input.transform_reference) {
        if (filter_count > std::numeric_limits<std::size_t>::max() /
                               sizeof(AcrAnchorLevel0FilterInput) ||
            (filter_count != 0 && filters == nullptr) ||
            filter_span_bytes <
                filter_count * sizeof(AcrAnchorLevel0FilterInput)) {
            return static_cast<int>(AnchorGeometryStatus::invalid_input);
        }

        if (filter_count > input.filters.max_size()) {
            return static_cast<int>(AnchorGeometryStatus::invalid_input);
        }
        try {
            input.filters.reserve(filter_count);
            for (std::size_t index = 0; index < filter_count; ++index) {
                const AcrAnchorLevel0FilterInput &source = filters[index];
                if (source.non_null != 0 && source.non_null != 1) {
                    return static_cast<int>(AnchorGeometryStatus::invalid_input);
                }
                const std::int32_t *rects = source.rects;
                input.filters.push_back({
                    source.non_null != 0,
                    {rects[0], rects[1], rects[2], rects[3]},
                    {rects[4], rects[5], rects[6], rects[7]},
                    {rects[8], rects[9], rects[10], rects[11]},
                    {rects[12], rects[13], rects[14], rects[15]},
                });
            }
        } catch (const std::bad_alloc &) {
            return 4;
        } catch (...) {
            return 4;
        }
    }

    AnchorLevel0BoundsResult result{{output[0], output[1], output[2],
                                     output[3]},
                                    {output[4], output[5], output[6],
                                     output[7]}};
    const AnchorGeometryStatus status =
        compute_anchor_level0_bounds(input, result);

    const std::array<std::int32_t, 8> candidate{
        result.actual_level0_bounds.top,
        result.actual_level0_bounds.left,
        result.actual_level0_bounds.bottom,
        result.actual_level0_bounds.right,
        result.reference_level0_bounds.top,
        result.reference_level0_bounds.left,
        result.reference_level0_bounds.bottom,
        result.reference_level0_bounds.right,
    };
    for (std::size_t index = 0; index < candidate.size(); ++index) {
        output[index] = candidate[index];
    }
    return static_cast<int>(status);
}

}  // extern "C"
