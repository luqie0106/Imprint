#pragma once

namespace imprint::acr {

enum class ProfileGainScalarStatus : int {
    ok = 0,
    signaling_nan = 1,
};

// Quantizes the local b83674 scalar field for finite binary64 inputs
// (including subnormals) and applies its local IEEE special-value handling:
// qNaN/+Inf map to 2, -Inf maps to +0, and signaling NaN returns
// signaling_nan while leaving `output` bit-for-bit unchanged. This is not a
// complete Prepare, Adobe error-ABI, FPCR/FPSR, or final-Q contract.
ProfileGainScalarStatus quantize_profile_gain_scalar(
    double input, double &output) noexcept;

}  // namespace imprint::acr
