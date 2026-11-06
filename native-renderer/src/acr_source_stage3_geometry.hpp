#pragma once

#include "acr_source_rational.hpp"

namespace imprint::acr {

// Prepared numerical subdomain for the recovered Stage3 source geometry
// normalization. It models six uint32 rational pairs and two finite scales;
// rational pairs such as denominator-zero values remain valid inputs. It does
// not represent the wrapper ABI or runtime branch selection.
struct PreparedSourceStage3Geometry {
    SourceRational rational_30{0U, 1U};
    SourceRational rational_38{0U, 1U};
    SourceRational rational_40{0U, 1U};
    SourceRational rational_48{0U, 1U};
    SourceRational rational_90{0U, 1U};
    SourceRational rational_98{0U, 1U};
    double scale_x = 1.0;
    double scale_y = 1.0;
};

enum class SourceStage3GeometryStatus : int {
    ok = 0,
    non_finite_scale = 1,
    rational_overflow = 2,
};

// For each scale greater than one, scales its three paired rationals in source
// order and then resets that scale to one. A scale at or below one and its
// paired rationals remain unchanged. The output is committed only on success,
// so any failure leaves it unchanged; input/output may alias.
SourceStage3GeometryStatus normalize_source_stage3_geometry(
    const PreparedSourceStage3Geometry &input,
    PreparedSourceStage3Geometry &output) noexcept;

}  // namespace imprint::acr
