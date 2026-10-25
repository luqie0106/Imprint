#pragma once

namespace imprint::acr {

struct NoiseTransformPrepared {
    double slope;
    double offset;
    double baseline;
    double scale;
    double input_black;
    double output_black;
    double input_reciprocal;
    double input_range;
    double output_reciprocal;
    double output_range;
};

enum class NoiseTransformStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_parameters = 2,
    overlap = 3,
};

// FUN_00b31b40: prepare the scalar transform parameters. Inputs must be finite,
// slope > 0, offset >= 0, and both black values in [0, 1). Every intermediate
// is checked for finiteness and the derived scale must be positive. `out` must
// be naturally aligned and its object span must not wrap the address space.
// On every failure `out` is left untouched.
//
// The apply methods accept only a prepared value whose ten fields exactly
// match the constructor-derived values (including the sign of zero); callers
// should obtain it from this constructor. Each method checks every arithmetic
// intermediate for finiteness before committing its result. A finite sample
// outside [0, 1] is accepted and follows the recovered clamp behavior.
// `prepared` and `out` must be naturally aligned, their spans must not wrap,
// and the output double must not overlap the prepared object. On failure `out`
// is left untouched. These contracts protect address arithmetic and aliasing;
// they cannot establish that an arbitrary non-null address is readable or
// writable.
//
// This helper models scalar virtual methods only; it does not claim equivalence
// to pixels produced by the separate LUT-backed image path.
NoiseTransformStatus prepare_noise_transform_00b31b40(
    double slope,
    double offset,
    double input_black,
    double output_black,
    NoiseTransformPrepared* out) noexcept;

// FUN_00b4bb88 scalar forward virtual method (vtable 08079808 +0x18).
NoiseTransformStatus apply_noise_transform_forward_00b4bb88(
    double sample,
    const NoiseTransformPrepared& prepared,
    double* out) noexcept;

// FUN_00b4bc38 scalar inverse virtual method (vtable 08079808 +0x20).
NoiseTransformStatus apply_noise_transform_inverse_00b4bc38(
    double sample,
    const NoiseTransformPrepared& prepared,
    double* out) noexcept;

}  // namespace imprint::acr
