"""Independent scalar binary32 reference for the prepared signed Exposure CPU path.

The arithmetic order in this module follows the saved ARM64 instruction listing
for FUN_01372fb8.  It intentionally does not import, inspect, or call the
native implementation.  Every ordinary arithmetic operation is rounded to
binary32; recovered fused multiply-add instructions use ``fmaf``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np


def _load_libm():
    name = ctypes.util.find_library("m")
    for candidate in ((name, None) if name else (None,)):
        try:
            library = ctypes.CDLL(candidate)
            fmaf = library.fmaf
            sqrtf = library.sqrtf
        except (AttributeError, OSError):
            continue
        fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
        fmaf.restype = ctypes.c_float
        sqrtf.argtypes = (ctypes.c_float,)
        sqrtf.restype = ctypes.c_float
        return fmaf, sqrtf
    raise RuntimeError("platform C math library does not expose fmaf and sqrtf")


_fmaf, _sqrtf = _load_libm()
_ZERO = np.float32(0.0)
_ONE = np.float32(1.0)
_QUARTER = np.float32(0.25)
_FOUR = np.float32(4.0)
_HALF = np.float32(0.5)
_MASK_COEFFICIENT = np.float32(-0.33977)
_INPUT_LIMIT = np.float32(1000.0)
_MASK_LIMIT = np.float32(16.0)


class Lut(NamedTuple):
    values: np.ndarray
    max_index: int
    scale: np.float32


def f32(value: float) -> np.float32:
    """Round a scalar to IEEE-754 binary32."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.float32(value)


def _add(a: float, b: float) -> np.float32:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.float32(np.float32(a) + np.float32(b))


def _sub(a: float, b: float) -> np.float32:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.float32(np.float32(a) - np.float32(b))


def _mul(a: float, b: float) -> np.float32:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.float32(np.float32(a) * np.float32(b))


def _div(a: float, b: float) -> np.float32:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return np.float32(np.float32(a) / np.float32(b))


def fma32(a: float, b: float, c: float) -> np.float32:
    """One correctly rounded binary32 fused multiply-add."""
    return np.float32(_fmaf(float(f32(a)), float(f32(b)), float(f32(c))))


def _sqrt(value: float) -> np.float32:
    return np.float32(_sqrtf(float(f32(value))))


def _finite(value: float, where: str) -> np.float32:
    # This helper performs the required binary32 rounding.  Public inputs and
    # accessed LUT values are validated as finite, while the original kernel
    # deliberately lets intermediate IEEE infinities and NaNs reach its final
    # FMINNM/lower-bound clamp.
    del where
    return f32(value)


def _validate_lut(spec: object | None, name: str) -> Lut | None:
    if spec is None:
        return None
    if not isinstance(spec, (tuple, list)) or len(spec) != 3:
        raise ValueError(f"{name} must be None or (values, max_index, scale)")
    values, max_index, scale = spec
    if isinstance(max_index, (bool, np.bool_)) or not isinstance(max_index, (int, np.integer)):
        raise ValueError(f"{name} max_index must be an integer")
    max_index = int(max_index)
    table = np.asarray(values)
    if table.dtype != np.float32 or table.ndim != 1:
        raise ValueError(f"{name} values must be a one-dimensional float32 array")
    if max_index < 0 or table.size < max_index + 2:
        raise ValueError(f"{name} must include the max_index interpolation guard")
    scale32 = f32(scale)
    if not np.isfinite(scale32) or scale32 < _ZERO:
        raise ValueError(f"{name} scale must be finite and nonnegative")
    return Lut(table, max_index, scale32)


def _ordered_gray(red: float, green: float, blue: float) -> np.float32:
    # ARM64 order: B + (G + (G + R)); do not reassociate this expression.
    green_red = _add(green, red)
    green_pair = _add(green, green_red)
    return _finite(_add(blue, green_pair), "ordered gray sum")


def _mask_gain(gain: float, sample: float) -> np.float32:
    mask = f32(sample)
    if not np.isfinite(mask):
        raise ValueError("mask samples must be finite")
    if mask > _MASK_LIMIT:
        mask = _MASK_LIMIT
    elif mask < -_MASK_LIMIT:
        mask = f32(-_MASK_LIMIT)

    scaled = _finite(_mul(mask, _FOUR), "mask times four")
    exponent = math.floor(float(scaled))
    fraction = _finite(_sub(scaled, f32(exponent)), "mask fractional exponent")

    # The ARM code builds 2**k by adding k*2**23 to the 1.0 exponent bits.
    exponent_bits = (0x3F800000 + (exponent << 23)) & 0xFFFFFFFF
    power = np.asarray([exponent_bits], dtype=np.uint32).view(np.float32)[0]

    complement = _finite(_sub(_ONE, fraction), "mask polynomial complement")
    curve = _finite(fma32(complement, _MASK_COEFFICIENT, _ONE), "mask polynomial FMA 1")
    curve = _finite(fma32(fraction, curve, _ONE), "mask polynomial FMA 2")
    scale = _finite(_mul(curve, power), "mask power approximation")
    return _finite(_mul(scale, gain), "masked gain")


def _interpolate(lut: Lut, value: float, name: str) -> np.float32:
    scaled = _finite(_mul(value, lut.scale), f"{name} scaled coordinate")
    # ARM FCVTZS truncates toward zero, maps NaN to zero, and saturates values
    # outside the signed-int range before the subsequent signed-index guard.
    coordinate = float(scaled)
    if math.isnan(coordinate):
        index = 0
    elif coordinate >= 2147483648.0:
        index = 2147483647
    elif coordinate <= -2147483648.0:
        index = -2147483648
    else:
        index = math.trunc(coordinate)
    # In particular, coordinates in (-1, 0) select node zero and retain a
    # negative fraction; values <= -1 produce a negative index and fail.
    if index < 0 or index > lut.max_index or index + 1 >= lut.values.size:
        raise ArithmeticError(f"{name} LUT coordinate exceeds its guarded range")
    if not np.isfinite(lut.values[index]) or not np.isfinite(lut.values[index + 1]):
        raise ValueError(f"{name} accessed LUT values must be finite")
    fraction = _finite(_sub(scaled, f32(index)), f"{name} interpolation fraction")
    next_product = _finite(_mul(fraction, lut.values[index + 1]), f"{name} next-node product")
    complement = _finite(_sub(_ONE, fraction), f"{name} interpolation complement")
    return _finite(fma32(lut.values[index], complement, next_product), f"{name} interpolation")


def _pixel(
    pixel: np.ndarray,
    controls: np.ndarray,
    mask: float | None,
    channel_luts: tuple[Lut | None, Lut | None, Lut | None],
    gray_lut: Lut | None,
) -> np.ndarray:
    original = [f32(value) for value in pixel]
    if not all(np.isfinite(value) for value in original):
        raise ValueError("RGB samples must be finite")
    source = [
        _INPUT_LIMIT if value > _INPUT_LIMIT else
        f32(-_INPUT_LIMIT) if value < f32(-_INPUT_LIMIT) else value
        for value in original
    ]
    magnitude = [f32(-value) if value < _ZERO else value for value in source]

    gain, compression, expansion, epsilon = controls
    effective_gain = gain if mask is None else _mask_gain(gain, mask)

    # Compression and expansion are both disabled at and above one. When either
    # is active the tone gray is sum*(gain*0.25), with that multiplication order.
    if compression >= _ONE and expansion >= _ONE:
        factor = effective_gain
    else:
        gray_sum = _ordered_gray(*magnitude)
        gain_quarter = _finite(_mul(effective_gain, _QUARTER), "tone gain quarter")
        tone_input = _finite(_mul(gray_sum, gain_quarter), "tone gray")

        if compression < _ONE:
            denominator = _finite(_add(tone_input, _ONE), "compression denominator")
            first = _finite(_div(tone_input, denominator), "compression first ratio")
            second = _finite(_add(_div(tone_input, compression), _ONE), "compression second ratio")
            tone_output = _finite(_mul(first, second), "compression tone")
        else:
            delta = _finite(_sub(tone_input, _ONE), "expansion delta")
            gray_times_four = _finite(_mul(tone_input, _FOUR), "expansion gray times four")
            inverse_expansion = _finite(_div(_ONE, expansion), "inverse expansion")
            scaled_gray = _finite(_mul(inverse_expansion, gray_times_four), "expansion radicand term")
            radicand = _finite(fma32(delta, delta, scaled_gray), "expansion radicand FMA")
            root = _finite(_sqrt(radicand), "expansion square root")
            expansion_half = _finite(_mul(expansion, _HALF), "expansion half")
            tone_output = _finite(_mul(expansion_half, _add(delta, root)), "expansion tone")

        numerator = _finite(_add(tone_output, epsilon), "tone ratio numerator")
        denominator = _finite(_add(tone_input, epsilon), "tone ratio denominator")
        tone_ratio = _finite(_div(numerator, denominator), "tone ratio")
        factor = _finite(_mul(effective_gain, tone_ratio), "effective tone gain")

    channels = [_finite(_mul(value, factor), "exposed channel") for value in magnitude]

    # The three channel tables form an all-or-none group; gray remains independent.
    if all(table is not None for table in channel_luts):
        channels = [
            _interpolate(table, value, f"channel {index}") if value < _ONE else value
            for index, (table, value) in enumerate(zip(channel_luts, channels))
        ]

    if gray_lut is not None:
        post_channel_gray = _finite(_mul(_ordered_gray(*channels), _QUARTER), "post-channel gray")
        if post_channel_gray < _ONE:
            mapped = _interpolate(gray_lut, post_channel_gray, "gray")
            denominator = epsilon if post_channel_gray <= epsilon else post_channel_gray
            gray_ratio = _finite(_div(mapped, denominator), "gray LUT ratio")
            channels = [_finite(_mul(value, gray_ratio), "gray-mapped channel") for value in channels]

    restored = [
        f32(-value) if source[index] < _ZERO else value
        for index, value in enumerate(channels)
    ]
    result = []
    for value in restored:
        # ARM FMINNM(value, +1000) returns its numeric operand when value is
        # NaN, followed by the original ordered lower-bound select.
        if np.isnan(value) or value > _INPUT_LIMIT:
            upper_clamped = _INPUT_LIMIT
        else:
            upper_clamped = value
        result.append(upper_clamped if upper_clamped >= f32(-_INPUT_LIMIT)
                      else f32(-_INPUT_LIMIT))
    return np.asarray(result, dtype=np.float32)


def apply(
    rgb: np.ndarray,
    controls: object,
    *,
    mask: object | None = None,
    channel_luts: object = (None, None, None),
    gray_lut: object | None = None,
) -> np.ndarray:
    """Evaluate packed HWC float32 RGB under the prepared finite-value contract."""
    source = np.asarray(rgb)
    if source.dtype != np.float32 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("RGB must be an HxWx3 float32 array")
    if source.shape[0] * source.shape[1] == 0 or source.shape[0] * source.shape[1] > (1 << 26):
        raise ValueError("RGB must contain between one pixel and 2**26 pixels")

    raw_controls = np.asarray(controls)
    if raw_controls.shape != (4,) or not np.isfinite(raw_controls).all():
        raise ValueError("controls must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        prepared_controls = np.asarray(raw_controls, dtype=np.float32)
    if not np.isfinite(prepared_controls).all():
        raise ValueError("controls must remain finite after binary32 conversion")
    if prepared_controls[0] < _ZERO or np.any(prepared_controls[1:] <= _ZERO):
        raise ValueError("gain must be nonnegative and compression/expansion/epsilon positive")

    if not isinstance(channel_luts, (tuple, list)) or len(channel_luts) != 3:
        raise ValueError("channel_luts must contain three descriptors")
    if all(spec is not None for spec in channel_luts):
        prepared_channels = tuple(
            _validate_lut(spec, f"channel_luts[{index}]")
            for index, spec in enumerate(channel_luts)
        )
    else:
        # The recovered routine tests all three pointers before loading any
        # channel descriptor. A partial group is wholly bypassed, including
        # its unused scale, capacity, and table storage.
        prepared_channels = (None, None, None)
    prepared_gray = _validate_lut(gray_lut, "gray_lut")

    if mask is None:
        masks = None
    else:
        masks = np.asarray(mask)
        if masks.dtype != np.float32 or masks.shape != source.shape[:2]:
            raise ValueError("mask must be an HxW float32 array matching RGB")
        if not np.isfinite(masks).all():
            raise ValueError("mask samples must be finite")

    # Build a separate result so a bad intermediate never leaks partial output.
    result = np.empty(source.shape, dtype=np.float32)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            sample = None if masks is None else masks[y, x]
            result[y, x] = _pixel(
                source[y, x], prepared_controls, sample, prepared_channels, prepared_gray
            )
    return result
