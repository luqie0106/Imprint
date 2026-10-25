"""Independent binary32 oracle for the gray and U16 ACR handlers.

The oracle is derived from the retained ARM64 instruction evidence.  It does
not import or inspect the native implementation.  In particular, fused
operations use the platform C library's ``fmaf`` and the dither tile uses the
direct Park--Miller modulo recurrence rather than the native Schrage form.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np


TILE_SIDE = 128
TILE_SIZE = TILE_SIDE * TILE_SIDE
TILE_MASK = TILE_SIDE - 1
PARK_MILLER_MULTIPLIER = 16807
PARK_MILLER_MODULUS = 2147483647
MINIMUM_ACCEPTED_LOW_WORD = 255
U16_SCALE = np.float32(65535.0)
U16_HALF = np.float32(0.5)
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _load_fmaf():
    candidates = [ctypes.util.find_library("m"), None]
    for library_name in candidates:
        try:
            library = ctypes.CDLL(library_name) if library_name else ctypes.CDLL(None)
            function = library.fmaf
            function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
            function.restype = ctypes.c_float
            return function
        except (AttributeError, OSError):
            continue
    raise RuntimeError("the platform C math library must provide fmaf")


_FMAF = _load_fmaf()


def fma32(a: float, b: float, c: float) -> np.float32:
    """Return one correctly rounded binary32 fused multiply-add."""
    return np.float32(_FMAF(float(np.float32(a)), float(np.float32(b)),
                            float(np.float32(c))))


def multiply32(a: float, b: float) -> np.float32:
    """Return one separately rounded binary32 multiplication."""
    return np.float32(np.float32(a) * np.float32(b))


def weights_from_lanes(lanes_rgb: tuple[int, int, int]) -> tuple[np.float32, ...]:
    """Reconstruct the effective RGB weights from their uint16 lanes."""
    if len(lanes_rgb) != 3 or any(not 0 <= int(lane) <= 0xFFFF for lane in lanes_rgb):
        raise ValueError("three uint16 coefficient lanes are required")
    scale = np.float32(16384.0)
    return tuple(np.float32(np.float32(int(lane)) / scale) for lane in lanes_rgb)


def gray_sample(rgb: tuple[float, float, float],
                lanes_rgb: tuple[int, int, int]) -> np.float32:
    """Apply FMUL(wG,G), FMA(wR,R,tmp), FMA(wB,B,tmp) in that order."""
    red, green, blue = (np.float32(value) for value in rgb)
    red_weight, green_weight, blue_weight = weights_from_lanes(lanes_rgb)
    green_term = multiply32(green_weight, green)
    red_accumulation = fma32(red_weight, red, green_term)
    return fma32(blue_weight, blue, red_accumulation)


def park_miller_tile() -> np.ndarray:
    """Build the row-major 128x128 tile by direct modular recurrence."""
    tile = np.empty(TILE_SIZE, dtype=np.uint16)
    seed = 1
    for index in range(TILE_SIZE):
        while True:
            seed = (PARK_MILLER_MULTIPLIER * seed) % PARK_MILLER_MODULUS
            low_word = seed & 0xFFFF
            if low_word >= MINIMUM_ACCEPTED_LOW_WORD:
                tile[index] = low_word
                break
    return tile.reshape(TILE_SIDE, TILE_SIDE)


def fcvtzs_s32(value: float) -> int:
    """Model ARM FCVTZS to signed 32-bit, including saturation and NaN."""
    sample = np.float32(value)
    if np.isnan(sample):
        return 0
    if sample >= np.float32(2.0 ** 31):
        return INT32_MAX
    if sample <= np.float32(-(2.0 ** 31)):
        return INT32_MIN
    return math.trunc(float(sample))


def quantize_gray_u16(gray: float, *, dither: bool,
                      noise: int = 0) -> int:
    """Quantize one finite grayscale sample with the recovered handler."""
    value = np.float32(gray)
    if not np.isfinite(value):
        raise ValueError("gray input must be finite")
    if dither:
        if not 0 <= int(noise) <= 0xFFFF:
            raise ValueError("dither noise must be a uint16")
        addend = multiply32(np.float32(noise), np.float32(2.0 ** -16))
    else:
        addend = U16_HALF

    scaled = fma32(value, U16_SCALE, addend)
    code = fcvtzs_s32(scaled)
    if dither:
        unsigned_code = code & 0xFFFFFFFF
        if unsigned_code >= 65536:
            code = 0 if code < 0 else 65535
    return code & 0xFFFF


def apply_gray_u16(gray: np.ndarray, width: int, height: int,
                   origin_x: int, origin_y: int,
                   *, dither: bool) -> np.ndarray:
    """Apply the U16 conversion and optional deterministic tiled dither."""
    values = np.asarray(gray, dtype=np.float32)
    if width <= 0 or height <= 0 or values.size != width * height:
        raise ValueError("gray sample count must match a nonempty image")
    if not np.isfinite(values).all():
        raise ValueError("gray input must be finite")

    tile = park_miller_tile() if dither else None
    output = np.empty((height, width), dtype=np.uint16)
    flat = values.reshape(-1)
    for y in range(height):
        for x in range(width):
            noise = 0
            if tile is not None:
                noise = int(tile[(y + int(origin_y)) & TILE_MASK,
                                 (x + int(origin_x)) & TILE_MASK])
            output[y, x] = quantize_gray_u16(
                flat[y * width + x], dither=dither, noise=noise)
    return output
