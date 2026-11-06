"""Independent binary64 oracle for prepared level-zero anchor bounds.

The model follows the retained 702d64/21358 and 702e9c/213b8 composition,
then the 000196f0 biased FCVTZS packing checks. It does not import native code
or the ignored reverse-engineering workspace.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Sequence


ORACLE_VERSION = "acr-anchor-level0-mixed-gate-pack-2026-10-26-v1"
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


class Status(IntEnum):
    OK = 0
    INVALID_INPUT = 1
    INVALID_GEOMETRY = 2
    INT32_OVERFLOW = 3
    RESOURCE_EXHAUSTED = 4


class _Abort(Exception):
    def __init__(self, status: Status):
        self.status = status


def _load_fma():
    candidates = []
    math_library = ctypes.util.find_library("m")
    if math_library:
        candidates.append(math_library)
    candidates.append(None)
    for name in candidates:
        try:
            library = ctypes.CDLL(name)
            function = library.fma
            function.argtypes = [ctypes.c_double, ctypes.c_double,
                                 ctypes.c_double]
            function.restype = ctypes.c_double
            return function
        except (AttributeError, OSError):
            continue
    raise RuntimeError("the platform C math library does not expose fma")


_fma = _load_fma()


@dataclass(frozen=True)
class Filter:
    non_null: bool
    offset_58: tuple[int, int, int, int]
    offset_78: tuple[int, int, int, int]
    offset_68: tuple[int, int, int, int]
    offset_88: tuple[int, int, int, int]


@dataclass(frozen=True)
class Input:
    actual_level0_bounds: tuple[int, int, int, int]
    reference_level0_bounds: tuple[int, int, int, int]
    filters: tuple[Filter, ...] = ()
    transform_actual: bool = False
    transform_reference: bool = False


@dataclass(frozen=True)
class Output:
    actual_level0_bounds: tuple[int, int, int, int]
    reference_level0_bounds: tuple[int, int, int, int]


def _same(a: Sequence[int], b: Sequence[int]) -> bool:
    return tuple(a) == tuple(b)


def _relative(a: Sequence[float], b: Sequence[float]
              ) -> tuple[float, float, float, float]:
    height = b[2] - b[0]
    width = b[3] - b[1]
    if not (math.isfinite(height) and math.isfinite(width) and
            height > 0.0 and width > 0.0):
        raise _Abort(Status.INVALID_GEOMETRY)
    inv_height = 1.0 / height
    inv_width = 1.0 / width
    candidate = ((a[0] - b[0]) * inv_height,
                 (a[1] - b[1]) * inv_width,
                 (a[2] - b[0]) * inv_height,
                 (a[3] - b[1]) * inv_width)
    if not all(math.isfinite(value) for value in candidate):
        raise _Abort(Status.INVALID_GEOMETRY)
    return candidate


def _denormalize(fraction: Sequence[float], base: Sequence[float]
                 ) -> tuple[float, float, float, float]:
    height = base[2] - base[0]
    width = base[3] - base[1]
    if not (all(math.isfinite(value) for value in fraction) and
            math.isfinite(height) and math.isfinite(width) and
            height > 0.0 and width > 0.0):
        raise _Abort(Status.INVALID_GEOMETRY)
    candidate = (_fma(fraction[0], height, base[0]),
                 _fma(fraction[1], width, base[1]),
                 _fma(fraction[2], height, base[0]),
                 _fma(fraction[3], width, base[1]))
    if not all(math.isfinite(value) for value in candidate):
        raise _Abort(Status.INVALID_GEOMETRY)
    return candidate


def _transform_actual(base: Sequence[int], filters: Sequence[Filter]
                      ) -> tuple[float, float, float, float]:
    current = tuple(float(value) for value in base)
    for item in filters:
        if not item.non_null or _same(item.offset_58, item.offset_78):
            continue
        fraction = _relative(
            tuple(float(value) for value in item.offset_58),
            tuple(float(value) for value in item.offset_78))
        current = _denormalize(fraction, current)
    return current


def _transform_reference(base: Sequence[int], filters: Sequence[Filter]
                         ) -> tuple[float, float, float, float]:
    current = tuple(float(value) for value in base)
    for item in filters:
        if not item.non_null or _same(item.offset_68, item.offset_88):
            continue
        fraction = _relative(
            tuple(float(value) for value in item.offset_68),
            tuple(float(value) for value in item.offset_88))
        current = _denormalize(fraction, current)
    return current


def _pack(rect: Sequence[float]) -> tuple[int, int, int, int]:
    rounded = []
    for value in rect:
        if not math.isfinite(value):
            raise _Abort(Status.INT32_OVERFLOW)
        biased = value + (0.5 if value > 0.0 else -0.5)
        if not (math.isfinite(biased) and
                -2147483649.0 < biased < 2147483648.0):
            raise _Abort(Status.INT32_OVERFLOW)
        rounded.append(math.trunc(biased))

    height = rounded[2] - rounded[0]
    width = rounded[3] - rounded[1]
    if not (INT32_MIN <= height <= INT32_MAX and
            INT32_MIN <= width <= INT32_MAX):
        raise _Abort(Status.INT32_OVERFLOW)
    return tuple(rounded)  # type: ignore[return-value]


def compute(input: Input) -> tuple[Status, Output | None]:
    try:
        actual = input.actual_level0_bounds
        reference = input.reference_level0_bounds
        if input.transform_actual:
            actual = _pack(_transform_actual(actual, input.filters))
        if input.transform_reference:
            reference = _pack(_transform_reference(reference, input.filters))
        return Status.OK, Output(tuple(actual), tuple(reference))
    except _Abort as error:
        return error.status, None
