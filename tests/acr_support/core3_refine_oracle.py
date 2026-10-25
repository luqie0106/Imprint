"""Instruction-derived float32 reference for the two Stage3 core3 row kernels.

This module intentionally depends only on the versioned contract and the
recorded instruction semantics.  It does not import or inspect the C++ port.
"""
from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np


OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
UNSUPPORTED_ALIGNMENT = 3
INVALID_PHASE = 4
INVALID_PARAMETERS = 5
ARITHMETIC_OVERFLOW = 6

EPS_BITS = 0x37800080
EPS = np.asarray([EPS_BITS], dtype=np.uint32).view(np.float32)[0]
F = np.float32
GUARD = 8
CENTER_ROW = 2

_libm_name = ctypes.util.find_library("m")
_libm = ctypes.CDLL(_libm_name or None)
_fmaf = _libm.fmaf
_fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_fmaf.restype = ctypes.c_float


def add(a: np.float32, b: np.float32) -> np.float32:
    with np.errstate(all="ignore"):
        return F(F(a) + F(b))


def sub(a: np.float32, b: np.float32) -> np.float32:
    with np.errstate(all="ignore"):
        return F(F(a) - F(b))


def mul(a: np.float32, b: np.float32) -> np.float32:
    with np.errstate(all="ignore"):
        return F(F(a) * F(b))


def div(a: np.float32, b: np.float32) -> np.float32:
    with np.errstate(all="ignore"):
        return F(np.divide(F(a), F(b), dtype=np.float32))


def fma(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    """One correctly rounded float32 FMA, supplied by the platform libm."""
    return F(_fmaf(float(F(a)), float(F(b)), float(F(c))))


def inv(value: np.float32) -> np.float32:
    value = F(value)
    return F(0.0) if value == F(0.0) else div(F(1.0), value)


def abs32(value: np.float32) -> np.float32:
    return F(np.abs(F(value)))


def min_nan(a: np.float32, b: np.float32) -> np.float32:
    """IEEE min with NaN propagation and -0 preference for equal zeros."""
    a, b = F(a), F(b)
    if np.isnan(a) or np.isnan(b):
        return F(np.nan)
    if a == b:
        if a == F(0.0):
            return F(-0.0) if np.signbit(a) or np.signbit(b) else F(0.0)
        return a
    return a if a < b else b


def max_nan(a: np.float32, b: np.float32) -> np.float32:
    """IEEE max with NaN propagation and +0 preference for equal zeros."""
    a, b = F(a), F(b)
    if np.isnan(a) or np.isnan(b):
        return F(np.nan)
    if a == b:
        if a == F(0.0):
            return F(0.0) if not np.signbit(a) or not np.signbit(b) else F(-0.0)
        return a
    return a if a > b else b


def _sum4(a, b, c, d, association: str) -> np.float32:
    if association == "right":
        return add(a, add(b, add(c, d)))
    if association == "left":
        return add(add(add(a, b), c), d)
    raise ValueError(f"unknown sum association: {association}")


def _refine_pixel(guide, channel, mask_value, y, x, lower, upper,
                  *, association="right", fused=True):
    g = lambda dy, dx: F(guide[y + dy, x + dx])
    v = lambda dy, dx: F(channel[y + dy, x + dx])

    vbase = add(abs32(sub(v(1, 0), v(-1, 0))), EPS)
    hbase = add(abs32(sub(v(0, -1), v(0, 1))), EPS)
    gc = g(0, 0)
    wu = inv(add(abs32(sub(gc, g(-2, 0))), vbase))
    wd = inv(add(abs32(sub(gc, g(2, 0))), vbase))
    wl = inv(add(abs32(sub(gc, g(0, -2))), hbase))
    wr = inv(add(abs32(sub(gc, g(0, 2))), hbase))

    tu = mul(sub(v(-1, 0), g(-1, 0)), wu)
    td = mul(sub(v(1, 0), g(1, 0)), wd)
    tl = mul(sub(v(0, -1), g(0, -1)), wl)
    tr = mul(sub(v(0, 1), g(0, 1)), wr)
    den = _sum4(wu, wd, wl, wr, association)
    num = _sum4(tu, td, tl, tr, association)
    candidate = fma(inv(den), num, gc) if fused else add(mul(inv(den), num), gc)

    center = v(0, 0)
    limited = max_nan(sub(center, lower),
                      min_nan(add(center, upper), candidate))
    delta = sub(limited, center)
    mixed = fma(mask_value, delta, center) if fused else add(mul(mask_value, delta), center)
    return max_nan(F(0.0), min_nan(mixed, F(1.0)))


def _mask_for_kernel(mask, y, x, kernel):
    if kernel == 0:
        return F(mask[y, x])
    if kernel == 1:
        # Recovered right-associated neighborhood mean, then one multiply.
        nested = add(F(mask[y, x - 1]),
                     add(F(mask[y, x + 1]), F(mask[y + 1, x])))
        nested = add(F(mask[y - 1, x]), nested)
        return mul(F(0.25), nested)
    raise ValueError("kernel must be 0 (00814230) or 1 (008145b4)")


def reference(kernel: int, guide: np.ndarray, channel_a: np.ndarray,
              channel_b: np.ndarray, mask: np.ndarray, width: int,
              phase: int, lower: float, upper: float, *,
              association: str = "right", fused: bool = True):
    """Compute one prepared row; pointers address the guarded center row."""
    arrays = (guide, channel_a, channel_b, mask)
    if any(a.dtype != np.float32 or a.ndim != 2 for a in arrays):
        raise ValueError("all planes must be two-dimensional float32 arrays")
    x0, y0 = GUARD, CENTER_ROW
    out_guide = np.empty(width, dtype=np.float32)
    out_a = np.empty(width, dtype=np.float32)
    out_b = np.empty(width, dtype=np.float32)
    # The guide copy and raw-copy lanes are bit copies, including NaN payloads.
    out_guide.view(np.uint32)[:] = guide[y0, x0:x0 + width].view(np.uint32)
    for j in range(width):
        for channel, output in ((channel_a, out_a), (channel_b, out_b)):
            refine = (((j ^ phase) & 1) == 0) if kernel == 0 else (((j ^ phase) & 1) == 1)
            if not refine:
                output.view(np.uint32)[j] = channel[y0, x0 + j].view(np.uint32)
                continue
            mask_value = _mask_for_kernel(mask, y0, x0 + j, kernel)
            output[j] = _refine_pixel(
                guide, channel, mask_value, y0, x0 + j,
                F(lower), F(upper), association=association, fused=fused)
    return out_guide, out_a, out_b


def aligned_rows(seed: int, width: int, stride: int | None = None,
                 *, height: int = 5, low=-0.5, high=1.5) -> np.ndarray:
    if stride is None:
        stride = width + 16
    if stride % 4 or stride < width + GUARD + 4:
        raise ValueError("stride must be a multiple of four with right guards")
    rng = np.random.default_rng(seed)
    array = rng.uniform(low, high, (height, stride)).astype(np.float32)
    if array.ctypes.data % 16:
        raise AssertionError("NumPy did not provide a 16-byte aligned allocation")
    return array


def pointer(array: np.ndarray, row=CENTER_ROW, col=GUARD, *, aligned=True):
    address = array.ctypes.data + (row * array.shape[1] + col) * array.itemsize
    if aligned and address % 16:
        raise AssertionError("plane center pointer is not 16-byte aligned")
    return ctypes.cast(ctypes.c_void_p(address), ctypes.POINTER(ctypes.c_float))


def assert_ieee_equal(actual: np.ndarray, expected: np.ndarray, message=""):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    assert actual.shape == expected.shape, message or "shape mismatch"
    a, e = actual.reshape(-1), expected.reshape(-1)
    finite = np.isfinite(e)
    assert np.array_equal(np.isfinite(a), finite), message or "finite classification mismatch"
    if np.any(finite):
        assert np.array_equal(a[finite].view(np.uint32), e[finite].view(np.uint32)), (
            message or "finite float32 bits differ")
    nan = np.isnan(e)
    assert np.array_equal(np.isnan(a), nan), message or "NaN classification mismatch"
    inf = np.isinf(e)
    assert np.array_equal(np.isinf(a), inf), message or "infinity classification mismatch"
    if np.any(inf):
        assert np.array_equal(np.signbit(a[inf]), np.signbit(e[inf])), (
            message or "infinity sign mismatch")


def same_classification_and_sign(actual: np.ndarray, expected: np.ndarray) -> bool:
    try:
        assert_ieee_equal(actual, expected)
    except AssertionError:
        return False
    return True

