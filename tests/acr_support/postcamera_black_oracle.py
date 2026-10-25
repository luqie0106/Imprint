"""Independent scalar oracle for ACR's explicit mode2 post-camera black LUT.

This module follows the recovered instruction order directly.  It deliberately
does not import or inspect the native implementation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np


SEGMENTS = 4096
TABLE_SIZE = SEGMENTS + 2


def _load_fma():
    candidates = [ctypes.util.find_library("m"), None]
    for library_name in candidates:
        try:
            library = ctypes.CDLL(library_name) if library_name else ctypes.CDLL(None)
            fma64 = library.fma
            fma64.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
            fma64.restype = ctypes.c_double
            fma32 = library.fmaf
            fma32.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
            fma32.restype = ctypes.c_float
            return fma64, fma32
        except (AttributeError, OSError):
            continue
    raise RuntimeError("the platform C math library must provide fma and fmaf")


_FMA64, _FMA32 = _load_fma()


def fma64(a: float, b: float, c: float = 0.0) -> float:
    """One correctly rounded binary64 fused multiply-add."""
    return float(_FMA64(float(a), float(b), float(c)))


def fma32(a: float, b: float, c: float = 0.0) -> np.float32:
    """One correctly rounded binary32 fused multiply-add."""
    return np.float32(_FMA32(float(np.float32(a)), float(np.float32(b)),
                             float(np.float32(c))))


def _black_value(x: float, black: float, scale: float) -> float:
    # Keep each double operation explicit to match the recovered evaluator.
    u = float(x) - float(black)
    value = fma64(scale, u, +0.0)
    value = min(value, 1.0)  # min-number semantics; all supported inputs are finite.
    if value < 0.0:
        value = +0.0
    return value


def build_lut(black: float) -> np.ndarray:
    """Build 4097 sampled entries and the copied 4098th endpoint guard."""
    b = float(black)
    if not math.isfinite(b) or not 0.0 <= b < 1.0:
        raise ValueError("black must be finite and satisfy 0 <= black < 1")

    d = b - 1.0
    scale = -1.0 / d
    table = np.empty(TABLE_SIZE, dtype=np.float32)
    reciprocal_segments = 1.0 / float(SEGMENTS)
    for index in range(SEGMENTS + 1):
        x = float(index) * reciprocal_segments
        table[index] = np.float32(_black_value(x, b, scale))
    table[SEGMENTS + 1] = table[SEGMENTS]
    return table


def interpolate(table: np.ndarray, pixel: float) -> np.float32:
    """Apply one unsigned mode2 float32 LUT sample with recovered rounding."""
    value = np.float32(pixel)
    scaled = np.float32(value * np.float32(SEGMENTS))
    index = int(float(scaled))  # truncation toward zero (FCVTZS)
    if index < 0 or index > SEGMENTS:
        raise ValueError("scaled pixel is outside the unsigned table domain")
    fraction = np.float32(scaled - np.float32(index))
    coefficient = np.float32(np.float32(1.0) - fraction)
    product = np.float32(fraction * np.float32(table[index + 1]))
    return fma32(np.float32(table[index]), coefficient, product)


def apply(source: np.ndarray, black: float) -> np.ndarray:
    """Apply the independent mode2 contract to a 2D or 3D float32 image."""
    if not isinstance(source, np.ndarray) or source.dtype != np.float32:
        raise ValueError("source must be a float32 NumPy array")
    if source.ndim not in (2, 3) or source.size == 0:
        raise ValueError("source must be a nonempty 2D or 3D array")
    b = float(black)
    if not math.isfinite(b) or not 0.0 <= b < 1.0:
        raise ValueError("black must be finite and satisfy 0 <= black < 1")
    if not np.isfinite(source).all() or np.any(source < 0.0) or np.any(source > 1.0):
        raise ValueError("active source samples must be finite and lie in [0, 1]")

    # The b==0 stage is absent.  Do not emulate it with an identity LUT, which
    # could change signed zero or subnormal bits.
    if b == 0.0:
        return np.array(source, dtype=np.float32, order="C", copy=True)

    table = build_lut(b)
    output = np.empty(source.shape, dtype=np.float32)
    source_flat = source.reshape(-1)
    output_flat = output.reshape(-1)
    for index, pixel in enumerate(source_flat):
        output_flat[index] = interpolate(table, pixel)
    return output


def direct_formula(source: np.ndarray, black: float) -> np.ndarray:
    """A deliberately separate algebraic formula used only for counterexamples."""
    b = float(black)
    result = np.empty(source.shape, dtype=np.float32)
    for index, pixel in enumerate(np.asarray(source, dtype=np.float32).reshape(-1)):
        transformed = (float(pixel) - b) / (1.0 - b)
        result.reshape(-1)[index] = np.float32(min(1.0, max(0.0, transformed)))
    return result
