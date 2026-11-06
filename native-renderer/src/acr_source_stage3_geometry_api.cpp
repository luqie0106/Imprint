#include "imprint_renderer.h"
#include "acr_source_stage3_geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

static_assert(sizeof(im_acr_source_stage3_geometry) == 64);
static_assert(offsetof(im_acr_source_stage3_geometry, scales) == 48);

im_status im_native_acr_source_stage3_geometry_normalize(
    const im_acr_source_stage3_geometry *input,
    im_acr_source_stage3_geometry *output) {
    constexpr std::uintptr_t bytes = sizeof(im_acr_source_stage3_geometry);
    const auto src = reinterpret_cast<std::uintptr_t>(input);
    const auto dst = reinterpret_cast<std::uintptr_t>(output);
    if (input == nullptr || output == nullptr ||
        src % alignof(im_acr_source_stage3_geometry) != 0 ||
        dst % alignof(im_acr_source_stage3_geometry) != 0 ||
        src > std::numeric_limits<std::uintptr_t>::max() - bytes ||
        dst > std::numeric_limits<std::uintptr_t>::max() - bytes ||
        (src != dst && src < dst + bytes && dst < src + bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    im_acr_source_stage3_geometry snapshot;
    std::memcpy(&snapshot, input, sizeof(snapshot));
    using namespace imprint::acr;
    const PreparedSourceStage3Geometry prepared{
        {snapshot.rationals[0], snapshot.rationals[1]},
        {snapshot.rationals[2], snapshot.rationals[3]},
        {snapshot.rationals[4], snapshot.rationals[5]},
        {snapshot.rationals[6], snapshot.rationals[7]},
        {snapshot.rationals[8], snapshot.rationals[9]},
        {snapshot.rationals[10], snapshot.rationals[11]},
        snapshot.scales[0], snapshot.scales[1]};
    PreparedSourceStage3Geometry result{};
    if (normalize_source_stage3_geometry(prepared, result) !=
        SourceStage3GeometryStatus::ok) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    const SourceRational pairs[6]{
        result.rational_30, result.rational_38, result.rational_40,
        result.rational_48, result.rational_90, result.rational_98};
    im_acr_source_stage3_geometry candidate{};
    for (std::size_t i = 0; i < 6; ++i) {
        candidate.rationals[2 * i] = pairs[i].numerator;
        candidate.rationals[2 * i + 1] = pairs[i].denominator;
    }
    candidate.scales[0] = result.scale_x;
    candidate.scales[1] = result.scale_y;
    *output = candidate;
    return IM_STATUS_OK;
}
