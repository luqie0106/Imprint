"""Versioned independent model for the ordinary ACR V-then-H executor.

The model transcribes the prepared-coordinate, bound-clamp, V FMA, and H
baseline accumulation rules used by the historical independent row and
separable probes. It imports neither those ignored research files nor native
code. Coefficient rows are supplied by callers so expected output never comes
from the C++ coefficient builder.
"""

from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np


ORACLE_VERSION = "acr-ordinary-resample-vh-fma32-2026-10-23-v1"
ORACLE_SCOPE = (
    "prepared logical HWC ordinary axes: resample, identity, and integer-copy "
    "inside the bounded non-wrapping rectangle contract"
)
FIXED_ONE = 1 << 32
PHASE_SHIFT = 25
PHASE_COUNT = 128
KIND_RESAMPLE = 0
KIND_IDENTITY = 1
KIND_INTEGER_TRANSLATION = 2


def _load_fmaf():
    for name in (ctypes.util.find_library("m"), None):
        try:
            library = ctypes.CDLL(name) if name else ctypes.CDLL(None)
            function = library.fmaf
            function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
            function.restype = ctypes.c_float
            return function
        except (AttributeError, OSError):
            continue
    raise RuntimeError("the platform C math library must provide fmaf")


_FMAF = _load_fmaf()


def fma32(a: float, b: float, c: float) -> np.float32:
    """One correctly rounded binary32 fused multiply-add."""
    return np.float32(_FMAF(float(np.float32(a)), float(np.float32(b)),
                            float(np.float32(c))))


def coordinate_parts(coordinate: int) -> tuple[int, int]:
    """Return signed floor index and the 0..127 phase from a Q32 coordinate."""
    index, remainder = divmod(int(coordinate), FIXED_ONE)
    return index, remainder >> PHASE_SHIFT


def source_bounds(start: int, extent: int, taps: int,
                  geometry: dict[str, int]) -> tuple[int, int]:
    if extent < taps:
        raise ValueError("source window is shorter than the coefficient row")
    lower = max(int(geometry["lower"]), int(start) * FIXED_ONE)
    upper = min(int(geometry["upper"]),
                (int(start) + int(extent) - int(taps)) * FIXED_ONE
                + FIXED_ONE - (FIXED_ONE >> PHASE_SHIFT))
    if lower > upper:
        raise ValueError("geometry has no overlap with the source window")
    return lower, upper


def _generic_dot(samples: np.ndarray, weights: np.ndarray,
                 taps: int) -> np.float32:
    total = np.float32(weights[0] * samples[0])
    block_end = ((taps - 1) // 16) * 16
    for tap in range(1, block_end + 1):
        product = np.float32(weights[tap] * samples[tap])
        total = np.float32(total + product)
    for tap in range(block_end + 1, taps):
        total = fma32(weights[tap], samples[tap], total)
    return total


def _horizontal_baseline(samples: np.ndarray, weights: np.ndarray,
                         taps: int) -> np.float32:
    if taps in (4, 6, 8):
        second_product = np.float32(weights[1] * samples[1])
        total = fma32(weights[0], samples[0], second_product)
        for tap in range(2, taps):
            total = fma32(weights[tap], samples[tap], total)
        return total
    return _generic_dot(samples, weights, taps)


def _sample_axis(source: np.ndarray, source_start: int, absolute_index: int,
                 axis: dict, *, vertical: bool, horizontal_bounds=None):
    origin = int(axis["origin"])
    step = int(axis["step"])
    coordinate = origin + step * int(absolute_index)
    lower, upper = horizontal_bounds or axis["effective_bounds"]
    coordinate = min(max(coordinate, lower), upper)
    index, phase = coordinate_parts(coordinate)
    taps = int(axis["taps"])
    start = index - int(source_start)
    weights = axis["coefficients"][phase]
    if vertical:
        accumulator = np.float32(0.0)
        for tap in range(taps):
            weight = np.float32(weights[tap])
            if weight == np.float32(0.0):
                continue
            accumulator = fma32(source[start + tap], weight, accumulator)
        return accumulator
    return _horizontal_baseline(source[start:start + taps], weights, taps)


def reference(source: np.ndarray, source_top: int, source_left: int,
              destination: tuple[int, int, int, int],
              vertical: dict, horizontal: dict) -> np.ndarray:
    """Execute ordinary logical HWC V then H under bounded rectangle rules.

    Axis dictionaries contain ``kind``, ``origin``, ``step``, ``lower`` and
    ``upper``. General axes additionally contain ``taps`` and a caller-owned
    ``coefficients`` array shaped (128, stride). Identity axes copy same
    absolute coordinates and never inspect any coefficient or other geometry
    member. Integer-copy axes materialize source coordinates as destination
    minus stored offset. Their complete source rectangle shifted by the stored
    offset must stay in int32, then the requested source rectangle must fit the
    original source view; no wrapping, clamping, or edge fill is modeled.
    """
    image = np.asarray(source)
    if image.dtype != np.float32 or image.ndim != 3:
        raise ValueError("source must be HWC float32")
    dst_top, dst_left, dst_height, dst_width = map(int, destination)
    source_height, source_width, channels = image.shape
    source_bottom = int(source_top) + source_height
    source_right = int(source_left) + source_width
    destination_bottom = dst_top + dst_height
    destination_right = dst_left + dst_width
    if (dst_height <= 0 or dst_width <= 0
            or not (-(1 << 31) <= destination_bottom <= (1 << 31) - 1)
            or not (-(1 << 31) <= destination_right <= (1 << 31) - 1)):
        raise OverflowError("destination rectangle is outside the bounded int32 domain")

    def prepare(axis: dict, start: int, extent: int) -> dict:
        kind = int(axis["kind"])
        prepared = dict(axis)
        if kind in (KIND_IDENTITY, KIND_INTEGER_TRANSLATION):
            return prepared
        if kind != KIND_RESAMPLE:
            raise ValueError("unsupported axis kind")
        coefficients = np.asarray(axis["coefficients"])
        taps = int(axis["taps"])
        stride = ((taps + 7) // 8) * 8
        if (coefficients.dtype != np.float32 or coefficients.ndim != 2
                or coefficients.shape != (PHASE_COUNT, stride)):
            raise ValueError("coefficient rows have the wrong shape")
        prepared["effective_bounds"] = source_bounds(
            start, extent, taps, axis)
        return prepared

    v_axis = prepare(vertical, source_top, source_height)
    h_axis = prepare(horizontal, source_left, source_width)
    if v_axis["kind"] == KIND_IDENTITY:
        if dst_top < source_top or dst_top + dst_height > source_bottom:
            raise ValueError("vertical identity extent is outside source")
    elif v_axis["kind"] == KIND_INTEGER_TRANSLATION:
        offset = int(v_axis.get("integer_offset", 0))
        relocated_top = int(source_top) + offset
        relocated_bottom = source_bottom + offset
        if not (-(1 << 31) <= relocated_top <= (1 << 31) - 1
                and -(1 << 31) <= relocated_bottom <= (1 << 31) - 1):
            raise OverflowError("vertical shifted source view overflows int32")
        requested_top = dst_top - offset
        requested_bottom = destination_bottom - offset
        if requested_top < source_top or requested_bottom > source_bottom:
            raise ValueError("vertical translated extent is outside source")
    if h_axis["kind"] == KIND_IDENTITY:
        if dst_left < source_left or dst_left + dst_width > source_right:
            raise ValueError("horizontal identity extent is outside source")
    elif h_axis["kind"] == KIND_INTEGER_TRANSLATION:
        offset = int(h_axis.get("integer_offset", 0))
        relocated_left = int(source_left) + offset
        relocated_right = source_right + offset
        if not (-(1 << 31) <= relocated_left <= (1 << 31) - 1
                and -(1 << 31) <= relocated_right <= (1 << 31) - 1):
            raise OverflowError("horizontal shifted source view overflows int32")
        requested_left = dst_left - offset
        requested_right = destination_right - offset
        if requested_left < source_left or requested_right > source_right:
            raise ValueError("horizontal translated extent is outside source")

    middle = np.empty((dst_height, source_width, channels), dtype=np.float32)
    for y in range(dst_height):
        absolute_y = dst_top + y
        if v_axis["kind"] in (KIND_IDENTITY, KIND_INTEGER_TRANSLATION):
            source_y = absolute_y
            if v_axis["kind"] == KIND_INTEGER_TRANSLATION:
                source_y -= int(v_axis.get("integer_offset", 0))
            middle[y] = image[source_y - source_top]
        else:
            for x in range(source_width):
                for channel in range(channels):
                    middle[y, x, channel] = _sample_axis(
                        image[:, x, channel], source_top, absolute_y,
                        v_axis, vertical=True)

    output = np.empty((dst_height, dst_width, channels), dtype=np.float32)
    for y in range(dst_height):
        if h_axis["kind"] in (KIND_IDENTITY, KIND_INTEGER_TRANSLATION):
            first_absolute = dst_left
            if h_axis["kind"] == KIND_INTEGER_TRANSLATION:
                first_absolute -= int(h_axis.get("integer_offset", 0))
            first = first_absolute - source_left
            output[y] = middle[y, first:first + dst_width]
            continue
        for channel in range(channels):
            for x in range(dst_width):
                output[y, x, channel] = _sample_axis(
                    middle[y, :, channel], source_left, dst_left + x,
                    h_axis, vertical=False)
    return output

