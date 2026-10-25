"""Independent binary32/binary64 oracle for the recovered flare LUT path.

The scalar sampler below follows the bracket-constructor and forward/inverse
sampler instruction listings.  It deliberately uses libm FMA calls at the
instructions that fuse and explicit binary32 rounding between separate
instructions.  It does not import the native implementation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
from pathlib import Path
import sys

import numpy as np


SEGMENTS = 4096
FLOAT_TABLE_SIZE = SEGMENTS + 2
UINT16_TABLE_SIZE = 65536
RECIPROCAL_65535 = 1.0 / 65535.0
_SQRT_EPSILON = 1.0 / 256.0
_SQRT_BIAS = 1.0 / 16.0


def _float_from_bits(bits: int) -> float:
    return float(np.asarray([bits], dtype=np.uint32).view(np.float32)[0])


_K_FORWARD = _float_from_bits(0x3F883FF0)
_K_INVERSE = _float_from_bits(0x3F707FE0)


def _load_libm():
    name = ctypes.util.find_library("m")
    for candidate in ((name, None) if name else (None,)):
        try:
            library = ctypes.CDLL(candidate)
            fma64 = library.fma
            fma32 = library.fmaf
            sqrt32 = library.sqrtf
        except (AttributeError, OSError):
            continue
        fma64.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
        fma64.restype = ctypes.c_double
        fma32.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
        fma32.restype = ctypes.c_float
        sqrt32.argtypes = (ctypes.c_float,)
        sqrt32.restype = ctypes.c_float
        return fma64, fma32, sqrt32
    raise RuntimeError("platform C math library does not expose fma, fmaf, and sqrtf")


_fma64, _fma32_c, _sqrt32 = _load_libm()


def f32(value: float) -> float:
    """Round a finite value once to IEEE-754 binary32, returned as Python float."""
    return float(np.float32(value))


def _add32(left: float, right: float) -> float:
    return f32(f32(left) + f32(right))


def _sub32(left: float, right: float) -> float:
    return f32(f32(left) - f32(right))


def _mul32(left: float, right: float) -> float:
    return f32(f32(left) * f32(right))


def _fma32(a: float, b: float, c: float) -> float:
    return float(_fma32_c(float(np.float32(a)), float(np.float32(b)), float(np.float32(c))))


def _sqrtf(value: float) -> float:
    return float(_sqrt32(float(np.float32(value))))


def _black_and_q(black: int) -> tuple[float, float]:
    if isinstance(black, (bool, np.bool_)) or not isinstance(black, (int, np.integer)):
        raise ValueError("black must be an integer B16 value")
    black = int(black)
    if not 0 <= black <= 65534:
        raise ValueError("black must be in [0, 65534]")
    b = float(black) * RECIPROCAL_65535
    q = 0.25 if b > 0.0 else 0.0
    return b, q


def _kind(kind: int) -> int:
    if isinstance(kind, (bool, np.bool_)) or not isinstance(kind, (int, np.integer)):
        raise ValueError("kind must be an integer table kind")
    kind = int(kind)
    if kind not in (0, 1, 2):
        raise ValueError("kind must be 0 (black), 1 (forward), or 2 (inverse)")
    return kind


def _clip01(value: float) -> float:
    # The assembly uses FMINNM(value, 1) followed by a signed less-than-zero
    # select.  All supported table inputs are finite, so this keeps the exact
    # finite-value behavior while also preserving negative zero.
    value = min(value, 1.0)
    return 0.0 if value < 0.0 else value


def _sample(black: int, kind: int, x: float) -> float:
    """Return one sampler result as binary32, before the final table store."""
    b, q = _black_and_q(black)
    kind = _kind(kind)
    x = float(x)

    if kind == 0:
        # Constructor: scale = -1/(b-1); sampler: fma(scale, x-b, +0),
        # upper clamp, signed-negative clamp, then FCVT S,D.
        scale = -1.0 / (b - 1.0)
        normalized = _fma64(scale, x - b, 0.0)
        return f32(_clip01(normalized))

    if kind == 1:
        if b == 0.0:
            value = _add32(f32(x), _SQRT_EPSILON)
            value = _sqrtf(value)
            value = _add32(value, -_SQRT_BIAS)
            return f32(_mul32(value, _K_FORWARD))

        # The constructor stores -1/(b-1), and the sampler performs the
        # subtraction before its double-precision FMA.
        scale = -1.0 / (b - 1.0)
        normalized = _fma64(scale, x - b, 0.0)
        normalized32 = f32(normalized)
        if normalized < 0.0:
            radicand = _sub32(_SQRT_EPSILON, normalized32)
            value = _sqrtf(radicand)
            value = _add32(value, -_SQRT_BIAS)
            nonlinear = _mul32(value, -_K_FORWARD)
        else:
            radicand = _add32(normalized32, _SQRT_EPSILON)
            value = _sqrtf(radicand)
            value = _add32(value, -_SQRT_BIAS)
            nonlinear = _mul32(value, _K_FORWARD)
        expanded = _fma64(1.0 - q, float(nonlinear), q)
        return f32(_clip01(expanded))

    if q == 0.0:
        value = _fma32(f32(x), _K_INVERSE, _SQRT_BIAS)
        value = _fma32(value, value, -_SQRT_EPSILON)
        return f32(value)

    # The inverse sampler normalizes around q with the constructor's
    # -1/(q-1) coefficient, chooses the signed branch from binary64 norm,
    # then performs the square and final expansion as separate FMAs.
    scale = -1.0 / (q - 1.0)
    normalized = _fma64(scale, x - q, 0.0)
    normalized32 = f32(normalized)
    if normalized < 0.0:
        value = _fma32(normalized32, -_K_INVERSE, _SQRT_BIAS)
        value = _fma32(value, value, -_SQRT_EPSILON)
        nonlinear = -float(value)
    else:
        value = _fma32(normalized32, _K_INVERSE, _SQRT_BIAS)
        value = _fma32(value, value, -_SQRT_EPSILON)
        nonlinear = float(value)
    expanded = _fma64(1.0 - b, nonlinear, b)
    return f32(_clip01(expanded))


def build_float_table(black: int, kind: int) -> np.ndarray:
    """Build the 4096-segment float32 table and duplicated endpoint guard."""
    _black_and_q(black)
    kind = _kind(kind)
    table = np.empty(FLOAT_TABLE_SIZE, dtype=np.float32)
    for index in range(SEGMENTS + 1):
        x = float(index) * (1.0 / SEGMENTS)
        table[index] = _sample(black, kind, x)
    table[SEGMENTS + 1] = table[SEGMENTS]
    if not np.isfinite(table).all():
        raise ValueError("flare float table must be finite")
    if table.view(np.uint32)[-1] != table.view(np.uint32)[-2]:
        raise AssertionError("flare table guard must duplicate the endpoint")
    return table


_SUPPORT = Path(__file__).resolve().parent
if str(_SUPPORT) not in sys.path:
    sys.path.insert(0, str(_SUPPORT))
import u16_lut_oracle  # noqa: E402


def build_u16_table(black: int, kind: int) -> np.ndarray:
    """Build the 65536-entry integer table through the recovered 01300600 path."""
    table = build_float_table(black, kind)
    result = u16_lut_oracle.build(table)
    if result.shape != (UINT16_TABLE_SIZE,) or result.dtype != np.uint16:
        raise AssertionError("integer LUT oracle returned the wrong shape or dtype")
    return result


def reference_image(
    source: np.ndarray,
    crop: tuple[int, int, int, int],
    black: int,
    nonlinear: bool,
    scene: bool,
) -> dict[str, object]:
    """Reference the bounded unsigned16 packed-RGB edge2 flare source path.

    This models five lower pyramid levels for a crop with either dimension
    above 64.  Small crops bypass the lower pyramid and use full-image bounds,
    matching the explicitly recovered ordinary unsigned16 subset.
    """
    if (not isinstance(source, np.ndarray) or source.dtype != np.uint16
            or source.ndim != 3 or source.shape[2] != 3
            or source.shape[0] <= 0 or source.shape[1] <= 0):
        raise ValueError("source must be a nonempty uint16 HxWx3 array")
    if not isinstance(crop, (tuple, list)) or len(crop) != 4:
        raise ValueError("crop must be (x, y, width, height)")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in crop):
        raise ValueError("crop values must be integers")
    crop_x, crop_y, crop_width, crop_height = (int(value) for value in crop)
    image_height, image_width = source.shape[:2]
    if (crop_x < 0 or crop_y < 0 or crop_width <= 0 or crop_height <= 0
            or crop_x + crop_width > image_width or crop_y + crop_height > image_height):
        raise ValueError("crop must be positive and contained in the actual image")
    _black_and_q(black)
    if not isinstance(nonlinear, (bool, np.bool_)) or not isinstance(scene, (bool, np.bool_)):
        raise ValueError("nonlinear and scene flags must be booleans")

    current = np.asarray(source)
    if crop_width > 64 or crop_height > 64:
        forward = build_u16_table(black, 1) if nonlinear else None
        inverse = build_u16_table(black, 2) if nonlinear else None
        for level in range(5):
            target_width = max(1, (crop_width + (1 << level)) >> (level + 1))
            target_height = max(1, (crop_height + (1 << level)) >> (level + 1))
            origin_x, origin_y = (crop_x, crop_y) if level == 0 else (0, 0)
            source_height, source_width = current.shape[:2]
            x0 = np.minimum(origin_x + 2 * np.arange(target_width, dtype=np.int64), source_width - 1)
            x1 = np.minimum(origin_x + 2 * np.arange(target_width, dtype=np.int64) + 1, source_width - 1)
            next_level = np.empty((target_height, target_width, 3), dtype=np.uint16)
            for row_start in range(0, target_height, 64):
                row_end = min(row_start + 64, target_height)
                row_offsets = np.arange(row_start, row_end, dtype=np.int64) * 2 + origin_y
                y0 = np.minimum(row_offsets, source_height - 1)
                y1 = np.minimum(row_offsets + 1, source_height - 1)
                a = current[y0[:, None], x0[None, :], :]
                b = current[y0[:, None], x1[None, :], :]
                c = current[y1[:, None], x0[None, :], :]
                d = current[y1[:, None], x1[None, :], :]
                if forward is not None:
                    a, b, c, d = forward[a], forward[b], forward[c], forward[d]
                total = a.astype(np.uint32)
                total += b.astype(np.uint32)
                total += c.astype(np.uint32)
                total += d.astype(np.uint32)
                total += np.uint32(2)
                reduced = (total >> np.uint32(2)).astype(np.uint16)
                next_level[row_start:row_end] = inverse[reduced] if inverse is not None else reduced
            current = next_level

    if int(black) == 0:
        last_pixels = np.array(current, dtype=np.uint16, order="C", copy=True)
    else:
        black_lut = build_u16_table(black, 0)
        last_pixels = np.array(black_lut[current], dtype=np.uint16, order="C", copy=True)

    pixel_count = int(last_pixels.shape[0]) * int(last_pixels.shape[1])
    sums = [float(np.sum(last_pixels[:, :, channel], dtype=np.float64)) for channel in range(3)]
    mean_r = sums[0] / pixel_count
    mean_g = sums[1] / pixel_count
    mean_b = sums[2] / pixel_count
    mean = ((mean_r + mean_g) + mean_b) / 3.0 * RECIPROCAL_65535
    flare = min(max(mean / 1048576.0, 0.0), 1.0 / 4096.0)
    epsilon = flare if scene and flare > 0.0 else 1.0 / 4096.0
    return {
        "mean": mean,
        "flare": flare,
        "epsilon": epsilon,
        "last_width": int(last_pixels.shape[1]),
        "last_height": int(last_pixels.shape[0]),
        "level_count": 5 if crop_width > 64 or crop_height > 64 else 0,
        "last_pixels": last_pixels,
    }
