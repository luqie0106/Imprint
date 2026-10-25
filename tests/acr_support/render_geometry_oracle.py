"""Independent binary64 model of the bounded d0684c render-geometry loop.

This module is derived from the retained ARM64 evidence and matrix contract;
it does not import, parse, or mirror the native implementation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from enum import IntEnum
import math
import struct
from typing import Sequence


ORACLE_VERSION = "acr-render-geometry-mask-first-fma64-2026-10-24-v3"
SQRT2 = 1.4142135623730951
FIRST_RETRY_LIMIT = 0.25
LATE_RETRY_LIMIT = 1.0 / 64.0


class RenderGeometryStatus(IntEnum):
    OK = 0
    INVALID_INPUT = 1
    UNAVAILABLE_LEVEL = 2
    INVALID_BOUNDS = 3
    NON_FINITE_MATRIX = 4
    UNSUPPORTED_MASK_CONVERSION = 5


class _Abort(Exception):
    def __init__(self, status: RenderGeometryStatus):
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
class Rect:
    top: int
    left: int
    bottom: int
    right: int


@dataclass(frozen=True)
class Level:
    # Opaque identity token. It is checked for nullness and returned unchanged.
    image: int
    reference_bounds: Rect
    actual_bounds: Rect


@dataclass(frozen=True)
class Context:
    field_00: int = 0
    field_7c: int = 0
    field_8c: int = 0
    field_98: int = 0
    field_911: int = 0
    field_846: int = 0
    field_847: int = 0
    field_820: int = 0
    field_84c: int = 0
    field_8a8: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0)
    field_810: float = 0.0
    field_818: int = 0
    field_824: int = 0
    field_828: int = 0


@dataclass(frozen=True)
class RenderGeometryInput:
    levels: tuple[Level, ...]
    source_68a: int
    source_178: int
    context: Context
    initial_level: int
    extent3: tuple[int, int]
    extent4: tuple[int, int]
    optional_present: bool = False
    optional_matrix: tuple[float, ...] = (0.0,) * 9
    param6: int = 1
    feature_5a: int = 0


@dataclass(frozen=True)
class RenderGeometryOutput:
    level: int
    image: int
    reference_bounds: Rect
    actual_bounds: Rect
    output_bounds: Rect
    matrix_e0: tuple[float, ...]
    matrix_170: tuple[float, ...]
    matrix_200: tuple[float, ...]
    context_after: Context
    flag_0c: bool
    mask_290: bool
    iterations: int
    mask_first_value_bits: int


IDENTITY: tuple[float, ...] = (
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
)

_MASK_TABLE_BITS: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    # Key and +0x08 payload bits, independently mapped from constructor SP
    # stores. In mode 1/2, pool +0x70 goes to key 300's +0x08 and pool +0x80
    # to key 360's +0x08; pool +0x90/+0xc0 are later +0x20 pairs.
    (0, 0): ((0, 0x3FF8000000000000),),
    (1, 0): ((72, 0x4008000000000000),),
    (1, 1): (
        (180, 0x3FF3333333333333), (240, 0x3FFCCCCCCCCCCCCD),
        (300, 0x400999999999999A), (360, 0x400C000000000000),
        (480, 0x400F333333333333),
    ),
    (1, 2): (
        (180, 0x3FF3333333333333), (240, 0x3FFCCCCCCCCCCCCD),
        (300, 0x400999999999999A), (360, 0x400C000000000000),
        (480, 0x400F333333333333),
    ),
}


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _multiply3(left: tuple[float, ...], right: tuple[float, ...]
               ) -> tuple[float, ...]:
    """Conventional row-major multiply with ascending-k sequential FMA."""
    result = []
    for row in range(3):
        for column in range(3):
            acc = 0.0
            for inner in range(3):
                acc = _fma(left[row * 3 + inner],
                           right[inner * 3 + column], acc)
                if not math.isfinite(acc):
                    raise _Abort(RenderGeometryStatus.NON_FINITE_MATRIX)
            result.append(acc)
    return tuple(result)


def _span(start: int, end: int) -> int:
    # The source arithmetic is signed int32 values widened before subtraction.
    value = int(end) - int(start)
    if not 1 <= value <= 0x7FFFFFFF:
        raise _Abort(RenderGeometryStatus.INVALID_BOUNDS)
    return value


def _wrapped_neg_i32(value: int) -> int:
    bits = (-int(value)) & 0xFFFFFFFF
    return bits - 0x100000000 if bits & 0x80000000 else bits


def _make_level_matrix(level: int, current: Rect, zero: Rect,
                       extent3: tuple[int, int],
                       optional_present: bool,
                       optional: tuple[float, ...]) -> tuple[float, ...]:
    height = _span(zero.top, zero.bottom)
    width = _span(zero.left, zero.right)
    sy = 1.0 / float(height)
    sx = 1.0 / float(width)
    if level != 0:
        scale = math.exp2(float(level))
        sy *= scale
        sx *= scale
    top_neg = float(_wrapped_neg_i32(current.top))
    left_neg = float(_wrapped_neg_i32(current.left))
    ty = sy * top_neg
    tx = sx * left_neg
    level_matrix = (sy, 0.0, ty,
                    0.0, sx, tx,
                    0.0, 0.0, 1.0)
    if not _finite(level_matrix):
        raise _Abort(RenderGeometryStatus.NON_FINITE_MATRIX)

    if optional_present:
        level_matrix = _multiply3(optional, level_matrix)
    extent_matrix = (float(extent3[0]), 0.0, 0.0,
                     0.0, float(extent3[1]), 0.0,
                     0.0, 0.0, 1.0)
    return _multiply3(extent_matrix, level_matrix)


def _norm_std(matrix: tuple[float, ...]) -> float:
    # d0684c uses two FADDs, hypot, then division by sqrt(2).
    first = matrix[0] + matrix[1]
    second = matrix[3] + matrix[4]
    if not (math.isfinite(first) and math.isfinite(second)):
        raise _Abort(RenderGeometryStatus.NON_FINITE_MATRIX)
    hyp = math.hypot(first, second)
    if not math.isfinite(hyp):
        raise _Abort(RenderGeometryStatus.NON_FINITE_MATRIX)
    normalized = hyp / SQRT2
    if not math.isfinite(normalized):
        raise _Abort(RenderGeometryStatus.NON_FINITE_MATRIX)
    return normalized


def _mask_from_first_qword(raw: int) -> bool:
    # FUN_00b55034 performs FCMP(first_double, 0.0) and returns equality;
    # d0684c stores its logical negation. NaN is therefore true here.
    value = struct.unpack("=d", struct.pack("=Q", raw & 0xFFFFFFFFFFFFFFFF))[0]
    return value != 0.0


def _double_from_bits(raw: int) -> float:
    return struct.unpack("=d", struct.pack("=Q", raw & 0xFFFFFFFFFFFFFFFF))[0]


def _double_bits(value: float) -> int:
    return struct.unpack("=Q", struct.pack("=d", value))[0]


def _mask_converter_first(context: Context, feature_5a: int
                          ) -> tuple[RenderGeometryStatus, int | None]:
    """Evaluate only the converter's evidence-closed first result qword."""
    dpi = float(context.field_810)
    if not math.isfinite(dpi):
        return RenderGeometryStatus.UNSUPPORTED_MASK_CONVERSION, None

    feature_enabled = (int(feature_5a) & 0xFF) != 0
    mode = int(context.field_824) & 0xFFFFFFFF
    if feature_enabled:
        if mode > 2:
            return RenderGeometryStatus.UNSUPPORTED_MASK_CONVERSION, None
        table = _MASK_TABLE_BITS[(1, mode)]
    else:
        # The default constructor ignores mode and data entirely.
        table = _MASK_TABLE_BITS[(0, 0)]

    # ARM64 executes FMUL before FCSEL for all units values. An overflow to
    # signed infinity is allowed and only the selected DPI reaches lookup.
    scaled_dpi = dpi * 2.54
    adjusted_dpi = scaled_dpi if int(context.field_818) == 3 else dpi

    keys = tuple(item[0] for item in table)
    values = tuple(_double_from_bits(item[1]) for item in table)
    if adjusted_dpi <= float(keys[0]):
        return RenderGeometryStatus.OK, _double_bits(values[0])
    if adjusted_dpi >= float(keys[-1]):
        return RenderGeometryStatus.OK, _double_bits(values[-1])

    upper = next(index for index, key in enumerate(keys)
                 if float(key) > adjusted_dpi)
    lower = upper - 1
    low_key = float(keys[lower])
    high_key = float(keys[upper])
    reciprocal_key_span = 1.0 / (high_key - low_key)
    delta_dpi = adjusted_dpi - low_key
    fraction = delta_dpi * reciprocal_key_span
    # FMINNM(fraction, 1.0), then FCMP and the negative-to-zero select.
    if math.isnan(fraction):
        fraction = 1.0
    else:
        fraction = min(fraction, 1.0)
    if fraction < 0.0:
        fraction = 0.0
    delta = values[upper] - values[lower]
    first = _fma(fraction, delta, values[lower])
    return RenderGeometryStatus.OK, _double_bits(first)


def _context_limit(context: Context) -> bool:
    return (((int(context.field_00) & 0xFFFFFFFF) + 1) & 0xFFFFFFFF) < 0x05070001


def _with_context(context: Context, *, field_846: int | None = None,
                  field_84c: int | None = None) -> Context:
    return Context(
        field_00=context.field_00,
        field_7c=context.field_7c,
        field_8c=context.field_8c,
        field_98=context.field_98,
        field_911=context.field_911,
        field_846=context.field_846 if field_846 is None else field_846,
        field_847=context.field_847,
        field_820=context.field_820,
        field_84c=context.field_84c if field_84c is None else field_84c,
        field_8a8=context.field_8a8,
        field_810=context.field_810,
        field_818=context.field_818,
        field_824=context.field_824,
        field_828=context.field_828,
    )


def _validate_shape_and_bounds(inputs: RenderGeometryInput) -> None:
    if len(inputs.levels) != 6 or not 0 <= int(inputs.initial_level) <= 5:
        raise _Abort(RenderGeometryStatus.INVALID_INPUT)
    if inputs.levels[int(inputs.initial_level)].image == 0:
        raise _Abort(RenderGeometryStatus.UNAVAILABLE_LEVEL)
    if len(inputs.extent3) != 2 or len(inputs.extent4) != 2:
        raise _Abort(RenderGeometryStatus.INVALID_INPUT)
    if any(int(value) <= 0 for value in (*inputs.extent3, *inputs.extent4)):
        raise _Abort(RenderGeometryStatus.INVALID_INPUT)
    if inputs.optional_present and len(inputs.optional_matrix) != 9:
        raise _Abort(RenderGeometryStatus.INVALID_INPUT)
    if len(inputs.context.field_8a8) != 7:
        raise _Abort(RenderGeometryStatus.INVALID_INPUT)

    zero = inputs.levels[0].reference_bounds
    _span(zero.top, zero.bottom)
    _span(zero.left, zero.right)


def prepare_render_geometry(
        inputs: RenderGeometryInput
        ) -> tuple[RenderGeometryStatus, RenderGeometryOutput | None]:
    """Return the independently evaluated status and complete successful state."""
    try:
        _validate_shape_and_bounds(inputs)

        context = inputs.context
        level = int(inputs.initial_level)
        source_zero = int(inputs.source_68a) == 0
        flag_enabled = int(context.field_847) != 0
        param6_enabled = int(inputs.param6) != 0
        mask_eligible = (param6_enabled and flag_enabled and source_zero
                         and level == 0)
        mask_first_value_bits = 0
        if mask_eligible:
            field_846 = int(context.field_846) & 0xFF
            # The initial byte is EOR'd with one before branching. Any
            # nonzero original byte skips the record/converter path and keeps
            # this byte-level result, including non-boolean values such as 2.
            mask_290 = (field_846 ^ 1) != 0
            if field_846 == 0:
                if int(context.field_820) != 0:
                    converter_status, mask_first_value_bits = (
                        _mask_converter_first(context, inputs.feature_5a))
                    if converter_status != RenderGeometryStatus.OK:
                        raise _Abort(converter_status)
                    mask_290 = (_double_from_bits(mask_first_value_bits)
                                != 0.0)
                else:
                    mask_first_value_bits = int(context.field_8a8[0]) & (
                        0xFFFFFFFFFFFFFFFF)
                    mask_290 = _mask_from_first_qword(
                        mask_first_value_bits)
        else:
            # An ineligible invocation deliberately ignores both +0x820 and
            # all record words.
            mask_290 = False

        extent3 = tuple(map(int, inputs.extent3))
        extent4 = tuple(map(int, inputs.extent4))
        ratio0 = float(extent4[0]) / float(extent3[0])
        ratio1 = float(extent4[1]) / float(extent3[1])
        ratio = (ratio0, 0.0, 0.0,
                 0.0, ratio1, 0.0,
                 0.0, 0.0, 1.0)
        if not _finite(ratio):
            raise _Abort(RenderGeometryStatus.NON_FINITE_MATRIX)
        if ratio0 < 1.0 or ratio1 < 1.0:
            base_170, base_200 = ratio, IDENTITY
        elif ratio0 > 1.0 or ratio1 > 1.0:
            base_170, base_200 = IDENTITY, ratio
        else:
            base_170, base_200 = IDENTITY, IDENTITY

        if not inputs.optional_present:
            optional = IDENTITY
        else:
            optional = tuple(map(float, inputs.optional_matrix))
            if not _finite(optional):
                raise _Abort(RenderGeometryStatus.NON_FINITE_MATRIX)

        context_846_started = int(context.field_846) != 0
        iterations = 0
        final_e0 = IDENTITY
        final_170 = IDENTITY
        final_200 = base_200
        final_flag_0c = False

        while True:
            iterations += 1
            current = inputs.levels[level].reference_bounds
            _span(current.top, current.bottom)
            _span(current.left, current.right)
            final_e0 = _make_level_matrix(
                level, current, inputs.levels[0].reference_bounds, extent3,
                inputs.optional_present, optional)

            # The slots are reset at each pass. Rebuild the branch-selected R.
            if ratio0 < 1.0 or ratio1 < 1.0:
                loop_170, loop_200 = ratio, IDENTITY
            elif ratio0 > 1.0 or ratio1 > 1.0:
                loop_170, loop_200 = IDENTITY, ratio
            else:
                loop_170, loop_200 = IDENTITY, IDENTITY

            final_flag_0c = (flag_enabled and source_zero and level == 0)
            if not mask_290:
                final_e0 = _multiply3(loop_170, final_e0)
                loop_170 = IDENTITY

            b = (_context_limit(context) and source_zero and final_flag_0c)
            d = (_context_limit(context) and final_flag_0c
                 and ((int(context.field_911) & 0xFFFFFFFF) != 0
                      or (int(context.field_7c) & 0xFFFFFFFF) != 0))
            g = ((int(context.field_8c) == 0
                  and (int(context.field_98) == 0
                       or (int(inputs.source_178) & 0xFFFFFFFF) == 1))
                 or int(context.field_846) != 0)

            candidate = level + 1
            has_next = candidate <= 5 and inputs.levels[candidate].image != 0

            # B && !G is the only route that skips the 0.25 gate.
            if not (b and not g):
                if not d and has_next and _norm_std(final_e0) < FIRST_RETRY_LIMIT:
                    if final_flag_0c and int(context.field_84c) == 0:
                        context = _with_context(context, field_84c=5)
                    level += 1
                    continue

            late_matrix = _multiply3(loop_200, loop_170)
            late_matrix = _multiply3(late_matrix, final_e0)
            if _norm_std(late_matrix) < LATE_RETRY_LIMIT:
                if has_next:
                    level += 1
                    continue
                if not context_846_started:
                    context = _with_context(context, field_846=1)
                    context_846_started = True
                    continue

            final_170 = loop_170
            final_200 = loop_200
            break

        final_level = inputs.levels[level]
        _span(final_level.actual_bounds.top, final_level.actual_bounds.bottom)
        _span(final_level.actual_bounds.left, final_level.actual_bounds.right)
        output = RenderGeometryOutput(
            level=level,
            image=int(final_level.image),
            reference_bounds=final_level.reference_bounds,
            actual_bounds=final_level.actual_bounds,
            output_bounds=final_level.actual_bounds,
            matrix_e0=final_e0,
            matrix_170=final_170,
            matrix_200=final_200,
            context_after=context,
            flag_0c=final_flag_0c,
            mask_290=mask_290,
            iterations=iterations,
            mask_first_value_bits=mask_first_value_bits,
        )
        return RenderGeometryStatus.OK, output
    except _Abort as error:
        return error.status, None

