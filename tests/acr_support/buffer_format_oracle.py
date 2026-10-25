"""Independent scalar oracle for the ACR float32/u16 buffer converters.

The forward path follows the observed ARM64 scalar sequence: choose a 0/1
fallback at 0.5, conditionally retain values in (0, 1], then perform a single
binary32 fused multiply-add by 65535 and 0.5 before truncating to uint32/u16.
The inverse path rounds the reciprocal and each multiplication separately to
binary32, matching the observed FDIV followed by FMUL.
"""

from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np


def _load_fmaf():
    name = ctypes.util.find_library("m")
    for candidate in ((name, None) if name else (None,)):
        try:
            libm = ctypes.CDLL(candidate)
            function = libm.fmaf
        except (AttributeError, OSError):
            continue
        function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
        function.restype = ctypes.c_float
        return function
    raise RuntimeError("platform C math library does not expose fmaf")


_fmaf = _load_fmaf()
U16_SCALE = np.float32(65535.0)
HALF = np.float32(0.5)
U16_RECIPROCAL = np.float32(np.float64(1.0) / np.float64(65535.0))


def f32(value: float) -> np.float32:
    """Round one value to IEEE-754 binary32."""
    return np.float32(value)


def fma32(a: float, b: float, c: float) -> np.float32:
    """Perform one correctly rounded binary32 fused multiply-add."""
    return np.float32(_fmaf(float(f32(a)), float(f32(b)), float(f32(c))))


def float_to_u16(value: float) -> np.uint16:
    """Convert one binary32 value using the ARM64 scalar-store semantics."""
    value32 = f32(value)

    # FCMP + FCSEL select 1 only when value > 0.5; unordered NaN compares
    # false, so it selects zero. The following ordered tests retain only
    # values satisfying value <= 1 and value > 0.
    fallback = np.float32(1.0 if float(value32) > 0.5 else 0.0)
    if float(value32) <= 1.0 and float(value32) > 0.0:
        clamped = value32
    else:
        clamped = fallback

    rounded = fma32(clamped, U16_SCALE, HALF)
    # The value is nonnegative and no greater than 65535, so int() has the
    # same toward-zero behavior as FCVTZS before the low 16-bit store.
    return np.uint16(int(float(rounded)) & 0xFFFF)


def u16_to_float(value: int) -> np.float32:
    """Convert one uint16 using separately rounded binary32 FDIV then FMUL."""
    integer = np.float32(int(value) & 0xFFFF)
    return np.float32(U16_RECIPROCAL * integer)


def floats_to_u16(values) -> np.ndarray:
    """Apply :func:`float_to_u16` elementwise, preserving input shape."""
    array = np.asarray(values, dtype=np.float32)
    return np.fromiter(
        (float_to_u16(value) for value in array.flat),
        dtype=np.uint16,
        count=array.size,
    ).reshape(array.shape)


def u16s_to_float(values) -> np.ndarray:
    """Apply :func:`u16_to_float` elementwise, preserving input shape."""
    array = np.asarray(values, dtype=np.uint16)
    return np.fromiter(
        (u16_to_float(value) for value in array.flat),
        dtype=np.float32,
        count=array.size,
    ).reshape(array.shape)
