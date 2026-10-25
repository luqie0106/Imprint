#pragma once

#include "acr_anchor_geometry.hpp"
#include "acr_resample_matrix.hpp"

#include <array>
#include <cstdint>

namespace imprint::acr {

// Bounds are prepared inputs. A used reference or final actual rectangle must
// have each int64-computed span in [1, INT32_MAX]; bounds in unused levels are
// not validated. image is an opaque identity token and is never dereferenced.
struct RenderGeometryLevel {
    const void *image = nullptr;
    AnchorRectI32 reference_bounds{};
    AnchorRectI32 actual_bounds{};
};

struct RenderGeometrySource {
    std::array<RenderGeometryLevel, 6> levels{};
    std::uint8_t field_68a = 0;
    std::uint32_t field_178 = 0;
};

struct RenderGeometryContext {
    std::uint32_t field_00 = 0;
    std::uint32_t field_7c = 0;
    std::uint32_t field_8c = 0;
    std::uint32_t field_98 = 0;
    std::uint32_t field_911 = 0;
    std::uint8_t field_846 = 0;
    std::uint8_t field_847 = 0;
    std::uint8_t field_820 = 0;
    std::int32_t field_84c = 0;
    std::array<std::uint64_t, 7> field_8a8{};
    double field_810 = 0.0;
    std::int32_t field_818 = 0;
    std::uint32_t field_824 = 0;
    std::uint32_t field_828 = 0;
};

struct RenderGeometryInput {
    RenderGeometrySource source{};
    RenderGeometryContext context{};
    std::int32_t initial_level = 0;
    std::array<std::int32_t, 2> extent3{};
    std::array<std::int32_t, 2> extent4{};
    bool optional_present = false;
    ResampleMatrix3 optional_matrix{};
    std::int32_t param6 = 1;
    std::uint8_t feature_5a = 0;
};

struct RenderGeometryResult {
    std::uint32_t level = 0;
    const void *image = nullptr;
    AnchorRectI32 reference_bounds{};
    AnchorRectI32 actual_bounds{};
    AnchorRectI32 output_bounds{};
    ResampleMatrix3 matrix_e0{};
    ResampleMatrix3 matrix_170{};
    ResampleMatrix3 matrix_200{};
    RenderGeometryContext context_after{};
    bool flag_0c = false;
    bool mask_290 = false;
    std::uint32_t iterations = 0;
    double mask_first_value = 0.0;
};

enum class RenderGeometryStatus : int {
    ok = 0,
    invalid_input = 1,
    unavailable_level = 2,
    invalid_bounds = 3,
    non_finite_matrix = 4,
    unsupported_mask_conversion = 5,
};

// CPU preparation only. This consumes prepared fields and bounds; it does not
// parse a complete source object, render bitmap pixels, or establish Adobe
// runtime/pixel equivalence. Only levels reached through the retry path are
// validated; absent optional matrices and unused levels are ignored. The only
// context fields changed in context_after are +0x846 and +0x84c. The ordinary path
// preserves the static +0x50 == identity invariant, so the conditional
// projection/expand rectangle branch is statically unselected here; the final
// output rectangle is copied directly from the selected level's actual bounds.
// The one-time mask starts as (uint8_t(+0x846) XOR 1); only an eligible path
// with +0x846 exactly zero reads/computes the first qword. The API does not
// fabricate the complete seven-qword converter record or pixel mask. The
// mask_first_value diagnostic is populated only on that eligible zero-flag
// path, otherwise it remains +0.0.
// The converted path requires finite caller DPI at +0x810. Its unconditional
// multiplication by 2.54 may overflow to an endpoint infinity. A nonzero
// feature_5a selects modes +0x824 in [0,2]; feature zero selects the default
// table and ignores that mode. Unsupported used converter inputs return
// unsupported_mask_conversion. Unused converter fields are not validated;
// +0x828 affects other record members and is preserved without interpretation.
// context_after copies all prepared fields and changes only +0x846/+0x84c.
// On any failure, result is left byte-for-byte unchanged.
RenderGeometryStatus prepare_render_geometry(
    const RenderGeometryInput &input,
    RenderGeometryResult &result) noexcept;

}  // namespace imprint::acr
