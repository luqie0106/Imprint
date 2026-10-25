"""Independent float32 oracle and ctypes loader for FUN_008134b8."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path

import numpy as np

OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
UNSUPPORTED_ALIGNMENT = 3
INVALID_PHASE = 4

GUARD = 4

_float_pointer = ctypes.POINTER(ctypes.c_float)
_libm_name = ctypes.util.find_library("m")
_libm = ctypes.CDLL(_libm_name) if _libm_name else ctypes.CDLL(None)
try:
    _fmaf = _libm.fmaf
except AttributeError:
    _libm = ctypes.CDLL(None)
    _fmaf = _libm.fmaf
_fmaf.restype = ctypes.c_float
_fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]


def _f32(value) -> np.float32:
    return np.float32(value)


def _add(a, b) -> np.float32:
    with np.errstate(all="ignore"):
        return _f32(_f32(a) + _f32(b))


def _sub(a, b) -> np.float32:
    with np.errstate(all="ignore"):
        return _f32(_f32(a) - _f32(b))


def _fma(a, b, c) -> np.float32:
    # ctypes converts each input to C float and returns the single-rounded
    # result from the platform libm fmaf implementation.
    return _f32(_fmaf(_f32(a), _f32(b), _f32(c)))


def _clamp01(value) -> np.float32:
    value = _f32(value)
    if np.isnan(value):
        return value
    upper = value if value <= _f32(1.0) else _f32(1.0)
    if upper > _f32(0.0):
        return upper
    # This intentionally maps negative zero to positive zero.
    return _f32(0.0)


def reference(source_rows: np.ndarray, green_rows: np.ndarray, width: int,
              phase: int, center_row: int = 1, offset: int = GUARD):
    """Return two rows using the recovered per-column float32 operation order."""
    out_a = np.empty(int(width), dtype=np.float32)
    out_b = np.empty(int(width), dtype=np.float32)
    with np.errstate(all="ignore"):
        for column in range(int(width)):
            x = int(offset) + column
            source = _f32(source_rows[center_row, x])
            green = _f32(green_rows[center_row, x])

            dx_left = _sub(source_rows[center_row, x - 1],
                           green_rows[center_row, x - 1])
            dx_right = _sub(source_rows[center_row, x + 1],
                            green_rows[center_row, x + 1])
            horizontal_delta = _add(dx_left, dx_right)
            horizontal = _clamp01(_fma(0.5, horizontal_delta, green))

            dy_up = _sub(source_rows[center_row - 1, x],
                         green_rows[center_row - 1, x])
            dy_down = _sub(source_rows[center_row + 1, x],
                           green_rows[center_row + 1, x])
            vertical_delta = _add(dy_up, dy_down)
            vertical = _clamp01(_fma(0.5, vertical_delta, green))

            a = _sub(source_rows[center_row - 1, x - 1],
                     green_rows[center_row - 1, x - 1])
            b = _sub(source_rows[center_row - 1, x + 1],
                     green_rows[center_row - 1, x + 1])
            c = _sub(source_rows[center_row + 1, x - 1],
                     green_rows[center_row + 1, x - 1])
            d = _sub(source_rows[center_row + 1, x + 1],
                     green_rows[center_row + 1, x + 1])
            c_plus_d = _add(c, d)
            b_plus_cd = _add(b, c_plus_d)
            diagonal_delta = _add(a, b_plus_cd)
            diagonal = _clamp01(_fma(0.25, diagonal_delta, green))

            use_source = ((column & 1) == 0) == (int(phase) == 0)
            if use_source:
                out_a[column] = source
                out_b[column] = diagonal
            else:
                out_a[column] = horizontal
                out_b[column] = vertical
    return out_a, out_b


def aligned_float_array(size: int) -> np.ndarray:
    """Allocate a 16-byte-aligned contiguous float32 vector."""
    raw = np.empty(int(size) + 3, dtype=np.float32)
    skip = ((-int(raw.ctypes.data)) % 16) // np.dtype(np.float32).itemsize
    result = raw[skip:skip + int(size)]
    assert result.ctypes.data % 16 == 0
    return result


def aligned_rows(height: int, stride: int) -> np.ndarray:
    result = aligned_float_array(int(height) * int(stride)).reshape(
        int(height), int(stride))
    assert result.strides[0] % 16 == 0
    return result


def _pointer(array: np.ndarray, require_alignment: bool = True):
    pointer = array.ctypes.data_as(_float_pointer)
    if require_alignment and int(array.ctypes.data) % 16 != 0:
        raise AssertionError(f"expected 16-byte alignment, got {array.ctypes.data:#x}")
    return pointer


def same_nonfinite_semantics(got: np.ndarray, expected: np.ndarray) -> bool:
    got = np.asarray(got, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if got.shape != expected.shape:
        return False
    got_nan = np.isnan(got)
    expected_nan = np.isnan(expected)
    if not np.array_equal(got_nan, expected_nan):
        return False
    got_inf = np.isinf(got)
    expected_inf = np.isinf(expected)
    if not np.array_equal(got_inf, expected_inf):
        return False
    if np.any(got_inf) and not np.array_equal(
            np.signbit(got[got_inf]), np.signbit(expected[expected_inf])):
        return False
    finite = ~(got_nan | got_inf)
    return bool(np.array_equal(got[finite].view(np.uint32),
                               expected[finite].view(np.uint32)))


class Bridge:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self.run = self.lib.acr_test_core3_colour
        self.run.restype = ctypes.c_int
        self.run.argtypes = [
            _float_pointer, _float_pointer,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_uint32, _float_pointer, _float_pointer,
        ]

    def execute(self, source_rows: np.ndarray, green_rows: np.ndarray,
                width: int, phase: int, output_padding: int = 4):
        out_a = aligned_float_array(int(width) + int(output_padding))
        out_b = aligned_float_array(int(width) + int(output_padding))
        out_a.fill(np.float32(-101.0))
        out_b.fill(np.float32(-202.0))
        source_center = source_rows[1, GUARD:]
        green_center = green_rows[1, GUARD:]
        status = self.run(
            _pointer(source_center), _pointer(green_center),
            source_rows.strides[0] // 4, green_rows.strides[0] // 4,
            int(width), int(phase), _pointer(out_a), _pointer(out_b),
        )
        return status, out_a, out_b


def make_rows(seed: int, width: int, stride: int, height: int = 3):
    if int(stride) % 4 != 0 or int(stride) < int(width) + GUARD + 1:
        raise ValueError("stride must be a multiple of four with both guards")
    rng = np.random.default_rng(seed)
    rows = aligned_rows(height, stride)
    rows[:] = rng.uniform(-2.0, 3.0, size=rows.shape).astype(np.float32)
    return rows
