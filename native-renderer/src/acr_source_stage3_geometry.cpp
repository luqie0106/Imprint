#include "acr_source_stage3_geometry.hpp"

#include <cmath>

namespace imprint::acr {
namespace {

bool scale_rational(SourceRational &value, double factor) noexcept {
    SourceRational result{};
    if (scale_source_rational(value, factor, result) !=
        SourceRationalStatus::ok) {
        return false;
    }
    value = result;
    return true;
}

}  // namespace

SourceStage3GeometryStatus normalize_source_stage3_geometry(
    const PreparedSourceStage3Geometry &input,
    PreparedSourceStage3Geometry &output) noexcept {
    if (!std::isfinite(input.scale_x) || !std::isfinite(input.scale_y)) {
        return SourceStage3GeometryStatus::non_finite_scale;
    }

    PreparedSourceStage3Geometry candidate = input;

    if (candidate.scale_x > 1.0) {
        if (!scale_rational(candidate.rational_30, candidate.scale_x) ||
            !scale_rational(candidate.rational_40, candidate.scale_x) ||
            !scale_rational(candidate.rational_90,
                            1.0 / candidate.scale_x)) {
            return SourceStage3GeometryStatus::rational_overflow;
        }
        candidate.scale_x = 1.0;
    }

    if (candidate.scale_y > 1.0) {
        if (!scale_rational(candidate.rational_38, candidate.scale_y) ||
            !scale_rational(candidate.rational_48, candidate.scale_y) ||
            !scale_rational(candidate.rational_98,
                            1.0 / candidate.scale_y)) {
            return SourceStage3GeometryStatus::rational_overflow;
        }
        candidate.scale_y = 1.0;
    }

    output = candidate;
    return SourceStage3GeometryStatus::ok;
}

}  // namespace imprint::acr
