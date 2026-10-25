"""Independent binary64 reference for Camera Raw pixel epsilon E.

This transcribes the prepared double arithmetic at 0x009e02b0 from the
version-pinned ARM64 slice. It does not call the native implementation or
read research artifacts while tests run.
"""

from __future__ import annotations

import math
import operator

import numpy as np


REFERENCE_ID = "acr-009e02b0-exposure-pixel-epsilon-f64-to-f32-v1"
PRODUCER_ENTRY = 0x009E02B0
SOURCE_BINARY_SHA256 = "09978d95dfe76b18495fea9550b051a89f4598ab0fd4734d69d22d330496f40a"
SOURCE_FLARE_CAP = 2.0 ** -12
SOURCE_FLARE_SCALE = 2.0 ** -7
DEFAULT_EPSILON = 2.0 ** -19
MINIMUM_EPSILON = 2.0 ** -120
MAXIMUM_EPSILON = 2.0 ** -16


def _as_finite_double(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite binary64 value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite binary64 value")
    return result


def _exp2_cpp(value: float) -> float:
    """Match libm exp2's zero/infinity result at finite extreme exponents."""
    try:
        return math.exp2(value)
    except OverflowError:
        # C/C++ exp2 reports +infinity for a finite exponent above range.
        # The caller's recovered clamp then maps this to MAXIMUM_EPSILON.
        return math.inf


def _clamp_epsilon(value: float) -> float:
    value = min(value, MAXIMUM_EPSILON)
    if value < MINIMUM_EPSILON:
        value = MINIMUM_EPSILON
    return value


def prepare(colorimetric_reference: object, source_flare: object,
            original_ev90: object) -> np.float32:
    """Prepare E from uint32 flag, source flare, and original child EV90.

    All finite double inputs are supported. An overflowing or underflowing
    intermediate exp2 is retained as infinity or zero and handled by the
    second clamp, matching the recovered C++ path.
    """
    try:
        flag = operator.index(colorimetric_reference)
    except TypeError as exc:
        raise ValueError("colorimetric_reference must be an unsigned 32-bit integer") from exc
    if not 0 <= flag <= 0xFFFFFFFF:
        raise ValueError("colorimetric_reference must be an unsigned 32-bit integer")
    flare = _as_finite_double(source_flare, "source_flare")
    ev90 = _as_finite_double(original_ev90, "original_ev90")

    if flag == 0 and flare > 0.0:
        value = min(flare, SOURCE_FLARE_CAP) * SOURCE_FLARE_SCALE
    else:
        value = DEFAULT_EPSILON
    value = _clamp_epsilon(value)

    if ev90 != 0.0:
        value = _exp2_cpp(-ev90) * value
    value = _clamp_epsilon(value)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.float32(value)


def float32_bits(value: object) -> int:
    """Return the output's uint32 payload for literal regression witnesses."""
    return int(np.asarray([value], dtype=np.float32).view(np.uint32)[0])
