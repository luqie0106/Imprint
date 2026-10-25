"""Independent binary32 oracle for the recovered PGTM coordinate planes.

This is a scalar transcription of the instructions in
``raw-012cc8b8-012cccb8.txt`` (function ``0x012cc8b8``). It intentionally
does not import renderer code. The recovered contract only covers the two
temporary coordinate planes; it does not name the downstream transform or
claim Adobe runtime or image-pipeline equivalence.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Sequence

import numpy as np
import resample_matrix_oracle as matrix_oracle


RAW_EVIDENCE = (
    "research/acr/reports/evidence/2026-10-24-normalized-early-cpu-inventory/"
    "raw-012cc8b8-012cccb8.txt"
)
ENTRY = "0x012cc8b8"
COEFFICIENT_FIELDS = ("A@+0x48", "B@+0x4c", "C0@+0x50", "D@+0x54", "E@+0x58", "F@+0x5c")
INSTRUCTION_EVIDENCE = {
    "row_scvtf_and_half_add": ("0x12cc930", "scvtf s3, w9", "0x12cc934", "fadd s3, s3, s0"),
    "row_fmas": ("0x12cc940", "fmadd s4, s4, s3, s5", "0x12cc950", "fmadd s6, s6, s3, s7"),
    "row_base_add_then_subtract": ("0x12cc984", "fadd s4, s5, s4", "0x12cc988", "fadd s4, s4, s1"),
    "column_scvtf_and_fmas": ("0x12cc97c", "scvtf s3, w11", "0x12cc9b4", "fmadd s6, s6, s3, s4", "0x12cc9c0", "fmadd s6, s6, s3, s5"),
    "recursive_column_increment": ("0x12cc9c8", "fadd s3, s3, s2"),
}


def _load_fmaf():
    library_names = (
        ("ucrtbase.dll", "msvcrt.dll", ctypes.util.find_library("m"), None)
        if os.name == "nt"
        else (ctypes.util.find_library("m"), None)
    )
    for library_name in library_names:
        try:
            library = ctypes.CDLL(library_name) if library_name else ctypes.CDLL(None)
            function = library.fmaf
            function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
            function.restype = ctypes.c_float
            return function
        except (AttributeError, OSError):
            continue
    raise RuntimeError("platform C math library must provide correctly-rounded fmaf")


_FMAF = _load_fmaf()


def f32(value: float | np.floating) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(value)


def add32(left: float, right: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(f32(left) + f32(right))


def sub32(left: float, right: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(f32(left) - f32(right))


def mul32(left: float, right: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(f32(left) * f32(right))


def fma32(left: float, right: float, addend: float) -> np.float32:
    return np.float32(_FMAF(float(f32(left)), float(f32(right)), float(f32(addend))))


def bits32(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32)


def assert_float32_or_nan(actual: np.ndarray, expected: np.ndarray) -> None:
    """Compare NaNs by classification and every other result by raw bits."""
    actual_values = np.ascontiguousarray(np.asarray(actual, dtype=np.float32))
    expected_values = np.ascontiguousarray(np.asarray(expected, dtype=np.float32))
    if actual_values.shape != expected_values.shape:
        raise AssertionError(f"shape mismatch: {actual_values.shape} != {expected_values.shape}")
    actual_nan = np.isnan(actual_values)
    expected_nan = np.isnan(expected_values)
    if not np.array_equal(actual_nan, expected_nan):
        bad = np.flatnonzero(actual_nan.reshape(-1) != expected_nan.reshape(-1))[:8]
        raise AssertionError(f"NaN classification mismatch at flat indices {bad.tolist()}")
    actual_bits, expected_bits = actual_values.view(np.uint32), expected_values.view(np.uint32)
    non_nan = ~expected_nan
    if not np.array_equal(actual_bits[non_nan], expected_bits[non_nan]):
        bad = np.flatnonzero(
            non_nan.reshape(-1) & (actual_bits.reshape(-1) != expected_bits.reshape(-1))
        )[:8]
        raise AssertionError(f"binary32 mismatch at flat indices {bad.tolist()}")


def _wrapped_neg32(value: int) -> int:
    """Mirror ARM64 W-register NEG, including INT32_MIN wrapping to itself."""
    unsigned = (-int(value)) & 0xFFFFFFFF
    return unsigned if unsigned < 0x80000000 else unsigned - 0x100000000


def reference_prepare(
    zero_rect: Sequence[int],
    state_rect: Sequence[int],
    matrix50: Sequence[float],
) -> tuple[int, tuple[np.float32, ...] | None]:
    """Prepare the six first-two-row coefficients from the recovered matrix chain.

    The matrix path is ``((T*S)*U)*inverse(G)``. Matrix products reuse the
    independent ordinary-resample oracle's ascending-k binary64 FMA routine;
    matrix inversion uses its separately recovered ``_invert3`` contract.
    """
    if len(zero_rect) < 4 or len(state_rect) < 4 or len(matrix50) < 9:
        raise ValueError("expected two four-element rects and a 3x3 matrix")
    reference = tuple(int(value) for value in zero_rect[:4])
    state = tuple(int(value) for value in state_rect[:4])
    i32_min, i32_max = -(1 << 31), (1 << 31) - 1
    if any(value < i32_min or value > i32_max for value in (*reference, *state)):
        raise ValueError("rect values must be signed int32")

    zero_height = reference[2] - reference[0]
    zero_width = reference[3] - reference[1]
    state_height = state[2] - state[0]
    state_width = state[3] - state[1]
    if any(value <= 0 or value > i32_max for value in
           (zero_height, zero_width, state_height, state_width)):
        return 2, None  # invalid_dimensions

    try:
        geometry = tuple(float(value) for value in matrix50[:9])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("matrix50 must contain nine numeric values") from error
    if not all(np.isfinite(value) for value in geometry):
        return 9, None  # invalid_prepared

    # Coordinate vectors are [row, column, 1], matching the constructor's
    # top/left translation order and the two coordinate-plane coefficients.
    translation = (
        1.0, 0.0, float(reference[0]),
        0.0, 1.0, float(reference[1]),
        0.0, 0.0, 1.0,
    )
    scaling = (
        float(zero_height) / float(state_height), 0.0, 0.0,
        0.0, float(zero_width) / float(state_width), 0.0,
        0.0, 0.0, 1.0,
    )
    untranslate = (
        1.0, 0.0, float(_wrapped_neg32(state[0])),
        0.0, 1.0, float(_wrapped_neg32(state[1])),
        0.0, 0.0, 1.0,
    )

    status, inverse = matrix_oracle._invert3(geometry)
    if status == matrix_oracle.ResampleMatrixStatus.SINGULAR_MATRIX:
        return 9, None
    if status != matrix_oracle.ResampleMatrixStatus.OK or inverse is None:
        return 11, None

    status, ts = matrix_oracle._multiply3(translation, scaling)
    if status != matrix_oracle.ResampleMatrixStatus.OK or ts is None:
        return 11, None
    status, tsu = matrix_oracle._multiply3(ts, untranslate)
    if status != matrix_oracle.ResampleMatrixStatus.OK or tsu is None:
        return 11, None
    status, composed = matrix_oracle._multiply3(tsu, inverse)
    if status != matrix_oracle.ResampleMatrixStatus.OK or composed is None:
        return 11, None

    with np.errstate(all="ignore"):
        narrowed = tuple(np.float32(value) for value in composed[:6])
    if not all(np.isfinite(value) for value in narrowed):
        return 11, None
    return 0, narrowed


def reference_planes(
    coefficients: Sequence[float],
    rect: Sequence[int],
    row_stride: int,
    *,
    v_count: int | None = None,
    u_count: int | None = None,
    sentinel: float = -731.625,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate row-strided ``v`` and ``u`` planes in the recovered order.

    ``rect`` is ``(top, left, bottom, right)`` with exclusive bottom/right.
    Each signed row or column origin is converted to binary32 once per row or
    once at row start, then the float column recurrence advances by binary32
    ``+1`` after every active column.
    """
    if len(coefficients) < 6 or len(rect) < 4:
        raise ValueError("expected six coefficients and a four-coordinate rect")
    top, left, bottom, right = (int(value) for value in rect[:4])
    i32_min, i32_max = -(1 << 31), (1 << 31) - 1
    if any(value < i32_min or value > i32_max for value in (top, left, bottom, right)):
        raise ValueError("rect values must be signed int32")
    height, width = bottom - top, right - left
    if height <= 0 or width <= 0 or height > i32_max or width > i32_max:
        raise ValueError("rect dimensions must be positive signed int32 values")
    if row_stride < width:
        raise ValueError("row_stride must be at least rect width")

    a, b, c0, d, e, f = (f32(value) for value in coefficients[:6])
    v_size = (height - 1) * row_stride + width if v_count is None else int(v_count)
    u_size = (height - 1) * row_stride + width if u_count is None else int(u_count)
    v_plane = np.full(v_size, f32(sentinel), dtype=np.float32)
    u_plane = np.full(u_size, f32(sentinel), dtype=np.float32)
    half, minus_half, one = f32(0.5), f32(-0.5), f32(1.0)

    for row_index in range(height):
        row_i32 = top + row_index
        row = f32(row_i32)
        row_half = add32(row, half)

        # Keep the explicit coefficient * 0.5, FMA, add, subtract sequence.
        base0_term = fma32(a, row_half, mul32(b, half))
        base0 = sub32(add32(c0, base0_term), half)
        base1_term = fma32(d, row_half, mul32(e, half))
        base1 = sub32(add32(f, base1_term), half)

        col = f32(left)
        for column_index in range(width):
            offset = row_index * row_stride + column_index
            v_plane[offset] = fma32(b, col, base0)
            u_plane[offset] = fma32(e, col, base1)
            col = add32(col, one)

    return v_plane, u_plane
