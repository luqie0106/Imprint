"""Independent binary32 transcription of Camera Raw kernel 013466b4.

The operation order below follows the captured 013466b4 instruction listings
and ``combined-camera-spec.md``.  It intentionally has no dependency on the
native implementation or on the older standalone camera kernel.  Each basic
operation rounds to binary32; only the captured FMADD/FMSUB operations use the
platform C ``fmaf`` implementation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import struct
from collections.abc import Iterable

import numpy as np


def _load_fmaf():
    name = ctypes.util.find_library("m")
    candidates = (name, None) if name else (None,)
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            function = library.fmaf
        except (AttributeError, OSError):
            continue
        function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
        function.restype = ctypes.c_float
        return function
    raise RuntimeError("platform C math library does not expose fmaf")


_fmaf = _load_fmaf()
_F32_MAX = float(np.finfo(np.float32).max)
_ACTIVE_PIXEL_LIMIT = 128 * 1024 * 1024


def f32(value: float) -> float:
    """Round a Python number to binary32, retaining signed zero."""
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.float32(value))


def _finite(value: float, operation: str) -> float:
    value = f32(value)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite intermediate after {operation}")
    return value


def _add(a: float, b: float) -> float:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.float32(a) + np.float32(b)
    return _finite(result, "FADD")


def _sub(a: float, b: float) -> float:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.float32(a) - np.float32(b)
    return _finite(result, "FSUB")


def _mul(a: float, b: float) -> float:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.float32(a) * np.float32(b)
    return _finite(result, "FMUL")


def _div(a: float, b: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.float32(a) / np.float32(b)
    return _finite(result, "FDIV")


def _fma(a: float, b: float, c: float) -> float:
    return _finite(_fmaf(a, b, c), "FMADD")


def _max_gt(a: float, b: float) -> float:
    """FCMP/FCSEL GT: ties choose the second operand."""
    return a if a > b else b


def _maxnm_zero(value: float) -> float:
    """FMAXNM(value, +0), including its +0 result for either zero input."""
    return value if value > 0.0 else 0.0


def clamp(value: float) -> float:
    """FMINNM(value, 1), followed by a strict negative test.

    The strict comparison preserves a negative zero already in ``value``.
    """
    value = value if value < 1.0 else 1.0
    return 0.0 if value < 0.0 else value


def _smooth(z: list[float], q: float, q1: float, q2: float,
            i0: float, i1: float, i2: float) -> list[float]:
    """The smooth highlight merge shared by zero- and nonzero-b paths."""
    h = _max_gt(z[0], z[1])

    a = _maxnm_zero(_mul(_sub(z[0], q), i0))
    a = _mul(a, a)
    v0 = _fma(a, _sub(h, z[0]), z[0])

    a = _maxnm_zero(_mul(_sub(z[1], q1), i1))
    a = _mul(a, a)
    v1 = _fma(a, _sub(h, z[1]), z[1])

    mean = _mul(_add(v0, v1), 0.5)
    h2 = _max_gt(mean, z[2])

    a = _maxnm_zero(_mul(_sub(z[2], q2), i2))
    a = _mul(a, a)
    v2 = _fma(a, _sub(h2, z[2]), z[2])
    return [v0, v1, v2]


def _row(matrix: list[float], offset: int, z: list[float]) -> float:
    # The instruction sequence is column 1 FMUL, column 0 FMA, column 2 FMA.
    result = _mul(matrix[offset + 1], z[1])
    result = _fma(z[0], matrix[offset], result)
    return _fma(z[2], matrix[offset + 2], result)


def apply(
    source: np.ndarray,
    controls: Iterable[float],
    order3: Iterable[int],
    flags: int,
    *,
    height: int,
    width: int,
    src_row_stride: int,
    dst_row_stride: int | None = None,
    destination: np.ndarray | None = None,
    pixels: Iterable[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Apply the recovered pixel kernel to an HWC RGB buffer.

    ``controls`` is ``[A, B, U, b, M00..M22]``.  ``order3`` selects the three
    source HWC channels in the already gain-sorted plane order; the matrix
    columns are already in that order and output remains RGB.  All strides and
    counts are in float samples.  ``pixels`` can restrict which (row, column)
    pixels the reference computes while keeping the original geometry,
    strides, channel order, and controls intact.
    """
    src = np.asarray(source, dtype=np.float32).reshape(-1)
    src_snapshot = src.copy()
    out_stride = src_row_stride if dst_row_stride is None else dst_row_stride
    if destination is None:
        dst = src.copy()
    else:
        dst = np.asarray(destination, dtype=np.float32).reshape(-1)
        if np.shares_memory(dst, src):
            # Read source values from the pre-call state for exact in-place use.
            src_snapshot = src.copy()

    raw_controls = tuple(float(value) for value in controls)
    if len(raw_controls) != 13:
        raise ValueError("controls must contain exactly 13 values")
    c = [f32(value) for value in raw_controls]
    if not all(math.isfinite(value) for value in c):
        raise ValueError("controls must be finite after binary32 conversion")
    a, b_gain, upper, floor = c[:4]
    matrix = c[4:]
    if not (a >= b_gain >= 1.0):
        raise ValueError("prepared gains must satisfy A >= B >= 1")
    if not (f32(32768.0 / 65535.0) <= upper <= 1.0):
        raise ValueError("upper must be in [f32(32768/65535), 1]")
    if not (0.0 <= floor < 1.0):
        raise ValueError("prepared floor must be in [0, 1)")
    order = tuple(int(value) for value in order3)
    if len(order) != 3 or sorted(order) != [0, 1, 2]:
        raise ValueError("order3 must be a permutation of HWC channels")
    if height <= 0 or width <= 0 or height * width > _ACTIVE_PIXEL_LIMIT:
        raise ValueError("invalid active dimensions")
    if src_row_stride < width * 3 or out_stride < width * 3:
        raise ValueError("row strides must hold the active HWC pixels")

    # Common prologue, including values not consumed by the A == 1 direct path.
    ratio = _div(b_gain, a)
    t = _div(1.0, a)
    q = _mul(upper, 0.25)
    q1 = _mul(ratio, q)
    q2 = _mul(t, q)
    i0 = _div(1.0, _sub(1.0, q))
    i1 = _div(1.0, _sub(ratio, q1))
    i2 = _div(1.0, _sub(t, q2))
    d = _sub(floor, 1.0)
    inv = 0.0 if floor == 1.0 else _div(-1.0, d)

    zero_floor = floor == 0.0  # both +0 and -0 enter this branch
    if not zero_floor:
        bias = _fma(-inv, floor, 0.0)
        bout = _fma(d, 0.0, floor)
        scaled_matrix = [-_mul(d, value) for value in matrix]
        ratio_inv = _mul(ratio, inv)
        t_inv = _mul(t, inv)
        ratio_bias = _mul(ratio, bias)
        t_bias = _mul(t, bias)

    selected = None if pixels is None else set((int(r), int(col)) for r, col in pixels)
    coords = ((row, col) for row in range(height) for col in range(width))
    for row, col in coords:
        if selected is not None and (row, col) not in selected:
            continue
        base = row * src_row_stride + col * 3
        values = [float(src_snapshot[base + channel]) for channel in order]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("source pixel contains a nonfinite value")

        if zero_floor:
            z = [values[0], _mul(ratio, values[1]), _mul(t, values[2])]
            if t < 1.0:
                z = _smooth(z, q, q1, q2, i0, i1, i2)
            result = [_row(matrix, 3 * channel, z) for channel in range(3)]
        else:
            z = [
                _fma(inv, values[0], bias),
                _fma(ratio_inv, values[1], ratio_bias),
                _fma(t_inv, values[2], t_bias),
            ]
            if t < 1.0:
                z = _smooth(z, q, q1, q2, i0, i1, i2)
            result = [
                _add(bout, _row(scaled_matrix, 3 * channel, z))
                for channel in range(3)
            ]

        if (flags & 1) == 0:
            result = [clamp(value) for value in result]
        out_base = row * out_stride + col * 3
        dst[out_base : out_base + 3] = np.asarray(result, dtype=np.float32)
    return dst


__all__ = ["apply", "clamp", "f32"]
