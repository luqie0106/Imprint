"""Independent binary64 transcription of the prepared Profile -> HSL24 loop.

Sources: saved dd974c/dd99f0/dd9a54 raw instructions in
2026-10-28-profile-gain-profile130-blend, b65564 caller in
2026-10-28-profile-gain-profile130, and cached b67ef0/78abb0 in
2026-11-08-profile-hsl-blend. Host libm supplies only explicit FMA operations.
This is not execution of Adobe Prepare, a metadata parser, or pixel HSL.
"""

from __future__ import annotations

import ctypes as C
import ctypes.util
import math
import struct
from typing import NamedTuple


LIBM = C.CDLL(None)
if not hasattr(LIBM, "fma"):
    name = C.util.find_library("m") or C.util.find_library("ucrtbase")
    if name is None:
        raise RuntimeError("profile HSL oracle requires a host libm fma")
    LIBM = C.CDLL(name)
FMA = LIBM.fma
FMA.argtypes = [C.c_double, C.c_double, C.c_double]
FMA.restype = C.c_double


def double_bits(value: float) -> int:
    return struct.unpack("=Q", struct.pack("=d", value))[0]


class Result(NamedTuple):
    values: tuple[int, ...]
    blended: tuple[float, ...]
    blended_bits: tuple[int, ...]
    wrote_mask: int


def normalize(value: int) -> float:
    if value <= -100:
        return -1.0
    if value >= 100:
        return 1.0
    t = (float(value) - (-100.0)) / 200.0
    return FMA(t, 2.0, -1.0)


def shape(value: float, scalar: float) -> float:
    if scalar <= 0.0:
        return 0.0
    if scalar <= 1.0:
        return value * scalar
    if value < 0.0:
        return -shape(-value, scalar)
    numerator = value * scalar
    shifted_scalar = scalar + (-1.0)
    denominator = FMA(shifted_scalar, value, 1.0)
    return numerator / denominator


def combine(current: float, default: float) -> float:
    if default < 0.0:
        return -combine(-current, -default)
    if current <= 0.0:
        return current + default
    if current >= 1.0:
        return current
    remaining = 1.0 - current
    denominator = FMA(current, default, remaining)
    ratio = remaining / denominator
    weighted = ratio * default
    return weighted + current


def lane_blend(current: int, profile_default: int, scalar: float) -> float:
    if profile_default == 0:
        return float(current)
    current_unit = normalize(current)
    default_unit = normalize(profile_default)
    shaped_default = shape(default_unit, scalar)
    q = combine(current_unit, shaped_default)
    if q <= -1.0:
        return -100.0
    if q >= 1.0:
        return 100.0
    t = (q + 1.0) * 0.5
    numerator = t * 1.0
    denominator = FMA(0.0, t, 1.0)
    z = numerator / denominator
    y = FMA(200.0, z, -100.0)
    return min(100.0, max(-100.0, y))


def evaluate(current, profile_defaults, scalar=1.0, monochrome=False,
             convert_to_gray=0) -> Result:
    if not math.isfinite(scalar):
        raise ValueError("non_finite_scalar")
    if len(current) != 24 or len(profile_defaults) != 24:
        raise ValueError("exactly 24 lanes are required")
    values, blended = [], []
    wrote_mask = 0
    for lane, (c, d) in enumerate(zip(current, profile_defaults)):
        c, d = int(c), int(d)
        if monochrome or convert_to_gray == 1 or d < -100 or d > 100:
            values.append(c)
            blended.append(float(c))
            continue
        wrote_mask |= 1 << lane
        y = lane_blend(c, d, scalar)
        offset = 0.5 if y > 0.0 else -0.5
        rounded = math.trunc(y + offset)
        values.append(min(100, max(-100, rounded)))
        blended.append(y)
    return Result(tuple(values), tuple(blended),
                  tuple(double_bits(v) for v in blended), wrote_mask)
