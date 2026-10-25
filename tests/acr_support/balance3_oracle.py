"""Independent float32 row references for the recovered balance3 kernels.

The equations and operation order in this module come from the versioned
balance3 contract. Expected values are calculated here, without loading or
calling the C++ implementation.
"""
from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np


_F = np.float32


def _load_fmaf():
    """Find the platform C ``fmaf`` without adding a package dependency."""
    candidates = []
    libm_name = ctypes.util.find_library("m")
    if libm_name:
        candidates.append(libm_name)
    # macOS commonly exports libm symbols from the process image itself.
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
    return _F(_F(a) + _F(b))


def mul32(a, b) -> np.float32:
    return _F(_F(a) * _F(b))


def fma32(a, b, c) -> np.float32:
    """Perform a single-rounding float32 fused multiply-add via C ``fmaf``."""
    return _F(_fmaf(_F(a), _F(b), _F(c)))


def _flat_source(source: np.ndarray) -> np.ndarray:
    values = np.asarray(source)
    if values.dtype != np.dtype(np.float32):
        raise ValueError("source must have dtype float32")
    if values.ndim != 1 or not values.flags.c_contiguous:
        raise ValueError("source must be a contiguous flat float32 array")
    return values


def difference_row(source: np.ndarray, center_offset: int, stride: int,
                   width: int, reverse: int) -> np.ndarray:
    """Reference 00813870 for ``width`` outputs at a guarded center row."""
    src = _flat_source(source)
    if reverse not in (0, 1):
        raise ValueError("reverse must be zero or one")
    out = np.empty(width, dtype=np.float32)
    for j in range(width):
        acc = add32(src[center_offset - stride + j - 1],
                    src[center_offset - stride + j + 1])
        acc = add32(acc, src[center_offset + stride + j - 1])
        acc = add32(acc, src[center_offset + stride + j + 1])
        avg = mul32(acc, _F(0.25))
        if reverse == 0:
            out[j] = _F(src[center_offset + j] - avg)
        else:
            out[j] = _F(avg - src[center_offset + j])
    return out


def diagonal_row(source: np.ndarray, center_offset: int, stride: int,
                 width: int, coeff: np.ndarray, radius: int, *,
                 direction: str, destination: np.ndarray | None = None,
                 subtract: bool = False) -> np.ndarray:
    """Reference filter/subtract row for direction ``plus`` or ``minus``.

    The pair sum rounds before the explicitly fused multiply-add. For subtract,
    the old destination is read only after the full filter accumulator has been
    formed, and the final subtraction rounds separately.
    """
    src = _flat_source(source)
    coefficients = np.asarray(coeff, dtype=np.float32)
    if coefficients.ndim != 1 or coefficients.size < radius + 1:
        raise ValueError("coeff must contain c[0] through c[radius]")
    if direction not in ("plus", "minus"):
        raise ValueError("direction must be 'plus' or 'minus'")
    if destination is None:
        destination = np.zeros(width, dtype=np.float32)
    dst = np.asarray(destination)
    if dst.dtype != np.dtype(np.float32) or dst.ndim != 1 or dst.size < width:
        raise ValueError("destination must be a flat float32 array with width samples")

    diagonal = stride + 1 if direction == "plus" else stride - 1
    out = np.empty(width, dtype=np.float32)
    for j in range(width):
        acc = mul32(coefficients[0], src[center_offset + j])
        for i in range(1, radius + 1):
            offset = i * diagonal
            pair = add32(src[center_offset + j - offset],
                         src[center_offset + j + offset])
            acc = fma32(coefficients[i], pair, acc)
        if subtract:
            out[j] = _F(dst[j] - acc)
        else:
            out[j] = acc
    return out


def uint32_bits(values: np.ndarray) -> np.ndarray:
    """Return a bit-preserving uint32 view of float32 values."""
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float32):
        raise ValueError("values must have dtype float32")
    return array.view(np.uint32)


def assert_float32_bits(actual: np.ndarray, expected: np.ndarray,
                        *, message: str = "float32 bit patterns differ") -> None:
    np.testing.assert_array_equal(uint32_bits(actual), uint32_bits(expected),
                                  err_msg=message)
