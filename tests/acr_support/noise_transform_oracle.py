"""Independent binary64 model for the scalar ACR noise transform.

This module deliberately does not import or inspect the native implementation.
Its arithmetic follows the scalar equations captured for this test, with FMA,
sqrt, and fmin supplied by the platform C math library.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math


def _load_libm():
    name = ctypes.util.find_library("m")
    candidates = (name, None) if name else (None,)
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            fma = library.fma
            sqrt = library.sqrt
            fmin = library.fmin
        except (AttributeError, OSError):
            continue
        fma.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
        fma.restype = ctypes.c_double
        sqrt.argtypes = (ctypes.c_double,)
        sqrt.restype = ctypes.c_double
        fmin.argtypes = (ctypes.c_double, ctypes.c_double)
        fmin.restype = ctypes.c_double
        return fma, sqrt, fmin
    raise RuntimeError("platform C math library does not expose fma, sqrt, and fmin")


_fma, _sqrt, _fmin = _load_libm()


def clamp(value: float) -> float:
    """Apply the scalar clamp sequence: fmin(value, 1), then a strict < 0 test."""
    value = _fmin(float(value), 1.0)
    return 0.0 if value < 0.0 else value


def prepare(slope: float, offset: float, input_black: float,
            output_black: float) -> list[float]:
    """Return the ten prepared values in the native bridge's documented order."""
    slope, offset, input_black, output_black = map(
        float, (slope, offset, input_black, output_black)
    )
    if not all(map(math.isfinite, (slope, offset, input_black, output_black))):
        raise ValueError("parameters must be finite")
    if slope <= 0.0 or offset < 0.0:
        raise ValueError("slope must be positive and offset nonnegative")
    if not (0.0 <= input_black < 1.0 and 0.0 <= output_black < 1.0):
        raise ValueError("black levels must be in [0, 1)")

    baseline_radicand = _fma(slope, 0.0, offset)
    baseline_root = _sqrt(baseline_radicand)
    baseline = (baseline_root + baseline_root) / slope

    reference_root = _sqrt(slope + offset)
    scale_denominator = ((reference_root + reference_root) / slope) - baseline
    scale = 1.0 / scale_denominator

    input_denominator = input_black + -1.0
    input_reciprocal = -1.0 / input_denominator
    input_range = -(input_black + -1.0)
    output_denominator = output_black + -1.0
    output_reciprocal = -1.0 / output_denominator
    output_range = -(output_black + -1.0)

    result = [
        slope,
        offset,
        baseline,
        scale,
        input_black,
        output_black,
        input_reciprocal,
        input_range,
        output_reciprocal,
        output_range,
    ]
    if not all(map(math.isfinite, result)):
        raise ValueError("prepared parameters are not finite")
    return result


def apply(prepared: list[float] | tuple[float, ...], sample: float,
          inverse: bool = False) -> float:
    """Apply one scalar forward or inverse transform using explicit operations."""
    (slope, offset, baseline, scale, input_black, output_black,
     input_reciprocal, input_range, output_reciprocal, output_range) = map(
        float, prepared
    )
    sample = float(sample)

    if inverse:
        z = _fma(output_reciprocal, sample - output_black, 0.0)
        distance = z / scale
        lower_root = (slope * (baseline - distance)) * 0.5
        upper_root = (slope * (baseline + distance)) * 0.5
        lower = -clamp(_fma(lower_root, lower_root, -offset) / slope)
        upper = clamp(_fma(upper_root, upper_root, -offset) / slope)
        return clamp(_fma(input_range, lower if z < 0.0 else upper, input_black))

    z = _fma(input_reciprocal, sample - input_black, 0.0)
    radicand = _fma(slope, z, offset) if z >= 0.0 else _fma(-slope, z, offset)
    root = _sqrt(radicand)
    value = clamp(scale * (((root + root) / slope) - baseline))
    if z < 0.0:
        value = -value
    return clamp(_fma(output_range, value, output_black))
