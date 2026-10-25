"""Independent float32 references for the two recovered balance3 finish paths.

The operation order, lane predicates, and clamps are transcribed from the
saved balance3 finish instruction contract. This module does not load or call
the native implementation.
"""
from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np


_F = np.float32


def _load_fmaf():
    candidates = []
    libm_name = ctypes.util.find_library("m")
    if libm_name:
        candidates.append(libm_name)
    candidates.append(None)
    for name in candidates:
        try:
            library = ctypes.CDLL(name) if name is not None else ctypes.CDLL(None)
            function = library.fmaf
        except (OSError, AttributeError):
            continue
        function.restype = ctypes.c_float
        function.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        return function
    raise RuntimeError("platform C library does not expose fmaf")


_fmaf = _load_fmaf()


def f32(value) -> np.float32:
    """Round one scalar operation boundary to IEEE binary32."""
    return _F(value)


def add32(a, b) -> np.float32:
    with np.errstate(all="ignore"):
        return _F(_F(a) + _F(b))


def sub32(a, b) -> np.float32:
    with np.errstate(all="ignore"):
        return _F(_F(a) - _F(b))


def mul32(a, b) -> np.float32:
    with np.errstate(all="ignore"):
        return _F(_F(a) * _F(b))


def fma32(a, b, c) -> np.float32:
    """Perform the single-rounding binary32 fused operation with libm fmaf."""
    return _F(_fmaf(_F(a), _F(b), _F(c)))


def ieee_min32(a, b) -> np.float32:
    """IEEE FMIN behavior for the finite/infinite, non-NaN row domain."""
    left, right = _F(a), _F(b)
    if np.isnan(left):
        return right
    if np.isnan(right):
        return left
    if left == _F(0.0) and right == _F(0.0):
        return _F(-0.0 if np.signbit(left) or np.signbit(right) else 0.0)
    return left if left < right else right


def ieee_max32(a, b) -> np.float32:
    """IEEE FMAX behavior for the finite/infinite, non-NaN row domain."""
    left, right = _F(a), _F(b)
    if np.isnan(left):
        return right
    if np.isnan(right):
        return left
    if left == _F(0.0) and right == _F(0.0):
        return _F(0.0 if not np.signbit(left) or not np.signbit(right) else -0.0)
    return left if left > right else right


def optimized_delta_clamp(delta, lower, upper) -> np.float32:
    """0081403c clamp: FMAX(lower, FMIN(delta, upper))."""
    return ieee_max32(_F(lower), ieee_min32(_F(delta), _F(upper)))


def default_delta_clamp(delta, lower, upper) -> np.float32:
    """013312fc scalar-tail tests and conditionals, preserving equal cases."""
    result = _F(delta)
    if not result < _F(upper):
        result = _F(upper)
    if _F(lower) > result:
        result = _F(lower)
    return result


def _fminnm32(a, b) -> np.float32:
    """Numeric min used by the default path's FMINNM(value, 1) instruction."""
    return ieee_min32(a, b)


def raw_delta_at(source: np.ndarray, center: int, stride: int,
                 diff: np.ndarray, column: int, row_even: int) -> np.float32:
    """Evaluate the recovered pre-clamp row expression at one output lane."""
    src = _flat_f32(source, "source")
    differences = _flat_f32(diff, "diff")
    if row_even not in (0, 1):
        raise ValueError("row_even must be zero or one")
    if int(column) < 0 or int(column) >= differences.size:
        raise ValueError("column is outside the diff row")
    sample = _F(src[int(center) + int(column)])
    acc = add32(src[int(center) - int(stride) + int(column) - 1],
                src[int(center) - int(stride) + int(column) + 1])
    acc = add32(acc, src[int(center) + int(stride) + int(column) - 1])
    acc = add32(acc, src[int(center) + int(stride) + int(column) + 1])
    acc = mul32(acc, _F(0.125))
    acc = fma32(sample, _F(0.5), acc)
    sign = _F(-0.5 if row_even else 0.5)
    acc = fma32(differences[int(column)], sign, acc)
    return sub32(acc, sample)


def _flat_f32(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must have dtype float32")
    if array.ndim != 1 or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be a contiguous flat float32 array")
    return array


def reference_row(source: np.ndarray, center: int, stride: int,
                  diff: np.ndarray, width: int, phase: int, row_even: int,
                  limits, *, branch: str,
                  destination: np.ndarray | None = None) -> np.ndarray:
    """Return expected output bits for one prepared row.

    ``branch`` is ``"default"`` for 013312fc or ``"optimized"`` for
    0081403c. The destination copy models the scalar default's untouched
    lanes and also preserves any caller-provided output padding.
    """
    src = _flat_f32(source, "source")
    differences = _flat_f32(diff, "diff")
    width, center, stride = int(width), int(center), int(stride)
    if width <= 0 or phase not in (0, 1) or row_even not in (0, 1):
        raise ValueError("width must be positive and flags must be binary")
    if branch not in ("default", "optimized"):
        raise ValueError("branch must be 'default' or 'optimized'")
    if differences.size < width:
        raise ValueError("diff does not contain width samples")
    if center - stride - 1 < 0 or center + stride + width >= src.size:
        raise ValueError("source does not contain the guarded neighborhood")
    control = np.asarray(limits, dtype=np.float32)
    if control.shape != (4,) or not np.isfinite(control).all():
        raise ValueError("limits must be four finite float32 values")
    lower, upper = ((control[0], control[1]) if row_even
                    else (control[2], control[3]))
    if lower > upper:
        raise ValueError("selected lower limit must not exceed upper limit")
    if not np.isfinite(src[center - stride - 1:center - stride + width + 1]).all():
        raise ValueError("source neighborhood must be finite")
    if not np.isfinite(src[center:center + width]).all():
        raise ValueError("source center must be finite")
    if not np.isfinite(src[center + stride - 1:center + stride + width + 1]).all():
        raise ValueError("source neighborhood must be finite")
    if not np.isfinite(differences[:width]).all():
        raise ValueError("diff row must be finite")

    if destination is None:
        out = np.zeros(width, dtype=np.float32)
    else:
        dst = _flat_f32(destination, "destination")
        if dst.size < width:
            raise ValueError("destination does not contain width samples")
        out = dst.copy()

    sign = _F(-0.5 if row_even else 0.5)
    one = _F(1.0)
    zero = _F(0.0)
    for j in range(width):
        selected = ((j & 1) == phase if branch == "optimized"
                    else (j & 1) != phase)
        if not selected:
            if branch == "optimized":
                out[j] = src[center + j]
            continue

        delta = raw_delta_at(src, center, stride, differences, j, row_even)
        if branch == "optimized":
            delta = optimized_delta_clamp(delta, lower, upper)
            value = add32(src[center + j], delta)
            value = ieee_max32(zero, ieee_min32(value, one))
        else:
            delta = default_delta_clamp(delta, lower, upper)
            value = add32(src[center + j], delta)
            value = _fminnm32(value, one)
            if value < zero:
                value = zero
        out[j] = value
    return out


def uint32_bits(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float32):
        raise ValueError("values must have dtype float32")
    return array.view(np.uint32)


def assert_float32_bits(actual: np.ndarray, expected: np.ndarray,
                        *, message: str = "float32 bit patterns differ") -> None:
    np.testing.assert_array_equal(uint32_bits(actual), uint32_bits(expected),
                                  err_msg=message)
