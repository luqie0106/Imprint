"""Independent binary32 oracle for the fixed 4096-segment ACR float LUT.

The model intentionally uses only the versioned scalar noise oracle and the C
math library.  It does not import or inspect the native LUT implementation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
from pathlib import Path
import sys

import numpy as np


_SUPPORT = Path(__file__).resolve().parent
if str(_SUPPORT) not in sys.path:
    sys.path.insert(0, str(_SUPPORT))
import noise_transform_oracle  # noqa: E402


SEGMENTS = 4096
TABLE_SIZE = SEGMENTS + 2


def _load_libm():
    name = ctypes.util.find_library("m")
    candidates = (name, None) if name else (None,)
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            fma = library.fma
            fmaf = library.fmaf
            fminf = library.fminf
        except (AttributeError, OSError):
            continue
        fma.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
        fma.restype = ctypes.c_double
        fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
        fmaf.restype = ctypes.c_float
        fminf.argtypes = (ctypes.c_float, ctypes.c_float)
        fminf.restype = ctypes.c_float
        return fma, fmaf, fminf
    raise RuntimeError("platform C math library does not expose fma, fmaf, and fminf")


_fma, _fmaf, _fminf = _load_libm()


def f32(value: float) -> np.float32:
    """Round one value to IEEE-754 binary32."""
    return np.float32(value)


def fma32(a: float, b: float, c: float) -> np.float32:
    """Perform one correctly rounded binary32 fused multiply-add."""
    return np.float32(_fmaf(float(f32(a)), float(f32(b)), float(f32(c))))


def evaluate(kind: int, x: float, params: list[float] | tuple[float, ...] | None = None) -> float:
    """Evaluate a scalar table node in binary64 before its final float cast."""
    x = float(x)
    if kind == 0:
        value = x
    elif kind == 1:
        value = _fma(0.37, x * x, 0.2)
    elif kind in (2, 3):
        if params is None or len(params) != 10:
            raise ValueError("noise LUT kinds require ten prepared parameters")
        value = noise_transform_oracle.apply(params, x, inverse=(kind == 3))
    else:
        raise ValueError("unknown LUT kind")
    if not math.isfinite(value):
        raise ValueError("LUT evaluation must be finite")
    return value


def build(kind: int, params: list[float] | tuple[float, ...] | None = None) -> np.ndarray:
    """Build 4098 binary32 values, including the duplicated final guard node."""
    if kind in (0, 1):
        if params not in (None, (), []):
            raise ValueError("identity and polynomial LUT kinds take no parameters")
        prepared = None
    elif kind in (2, 3):
        if params is None or len(params) != 10:
            raise ValueError("noise LUT kinds require ten prepared parameters")
        prepared = [float(value) for value in params]
        if not all(math.isfinite(value) for value in prepared):
            raise ValueError("prepared parameters must be finite")
    else:
        raise ValueError("unknown LUT kind")

    table = np.empty(TABLE_SIZE, dtype=np.float32)
    scale = 1.0 / SEGMENTS
    for index in range(SEGMENTS + 1):
        x = float(index) * scale
        table[index] = f32(evaluate(kind, x, prepared))
    table[SEGMENTS + 1] = table[SEGMENTS]
    if not np.isfinite(table).all():
        raise ValueError("built LUT must be finite")
    if table.view(np.uint32)[-1] != table.view(np.uint32)[-2]:
        raise ValueError("final LUT guard must duplicate the final node bit-for-bit")
    return table


def interpolate(table: np.ndarray | list[float], sample: float) -> np.float32:
    """Interpolate one unsigned-path sample using product-then-FMA arithmetic.

    The native FCVTZS conversion accepts scaled coordinates in (-1, 4097):
    truncation toward zero maps the small negative interval to node zero and
    the upper extension interval to node 4096 plus its guard entry.
    """
    values = np.asarray(table, dtype=np.float32)
    if values.shape != (TABLE_SIZE,) or not np.isfinite(values).all():
        raise ValueError("table must contain 4098 finite float values")
    if values.view(np.uint32)[-1] != values.view(np.uint32)[-2]:
        raise ValueError("final LUT guard must duplicate the final node")
    sample = f32(sample)
    if not np.isfinite(sample):
        raise ValueError("unsigned samples must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        t = f32(float(sample) * float(SEGMENTS))
    if not np.isfinite(t) or not (f32(-1.0) < t < f32(SEGMENTS + 1)):
        raise ValueError("scaled unsigned sample must be in (-1, 4097)")
    index = int(float(t))
    fraction = f32(float(t) - float(f32(index)))
    product = f32(fraction * values[index + 1])
    complement = f32(f32(1.0) - fraction)
    result = fma32(values[index], complement, product)
    if not np.isfinite(result):
        raise ValueError("interpolated LUT result must be finite")
    return result


def apply_signed(table: np.ndarray | list[float], sample: float) -> np.float32:
    """Apply the ACR symmetric extension, preserving the specified signed-zero path."""
    values = np.asarray(table, dtype=np.float32)
    if values.shape != (TABLE_SIZE,) or not np.isfinite(values).all():
        raise ValueError("table must contain 4098 finite float values")
    if values.view(np.uint32)[-1] != values.view(np.uint32)[-2]:
        raise ValueError("final LUT guard must duplicate the final node")
    sample = f32(sample)
    if not np.isfinite(sample):
        raise ValueError("signed samples must be finite")

    absolute = f32(-sample) if sample < f32(0.0) else sample
    clipped = np.float32(_fminf(float(absolute), float(f32(1.0))))
    if absolute > f32(1.0):
        extension = f32(float(absolute) + float(f32(-1.0)))
    else:
        extension = f32(-0.0)
    value = f32(float(extension) + float(interpolate(values, clipped)))
    zero = interpolate(values, f32(0.0))
    twice_zero = f32(float(zero) + float(zero))
    result = f32(float(twice_zero) - float(value)) if sample < f32(0.0) else value
    if not np.isfinite(result):
        raise ValueError("signed LUT result must be finite")
    return result

