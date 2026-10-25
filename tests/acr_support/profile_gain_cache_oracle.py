"""Independent scalar oracle and payload fixtures for PGTM cache generation.

The fixtures are encoded locally and this module imports no repository
research output or product code.  It covers the task's bounded source-reader
domain and the recovered cache-table transform only.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import struct
from dataclasses import dataclass

import numpy as np


TAG_EXTENDED = 52544
OK = 0
INVALID_ARGUMENT = 1
INVALID_VALUE = 5
BUFFER_TOO_SMALL = 9
OVERLAP = 10
UNSUPPORTED_ALIGNMENT = 11
POINTER_OVERFLOW = 12

MIN_GAIN = np.float32(2.0**-12)
MAX_GAIN = np.float32(4096.0)
SENTINEL_META = np.float64(-991.125)
SENTINEL_FIELDS = np.float32(-731.625)
SENTINEL_GEOMETRY = np.uint32(0xA5A5A5A5)
SENTINEL_VALUES = np.float32(-317.25)
SENTINEL_WRITTEN = 0x12345678


def _load_math_function(name: str, *, double_args: bool = False):
    if os.name == "nt":
        libraries = ("ucrtbase.dll", "msvcrt.dll", ctypes.util.find_library("m"), None)
    else:
        libraries = (ctypes.util.find_library("m"), None)
    for library_name in libraries:
        try:
            library = ctypes.CDLL(library_name) if library_name else ctypes.CDLL(None)
            function = getattr(library, name)
            if name == "fma":
                function.argtypes = (ctypes.c_double,) * 3
                function.restype = ctypes.c_double
            elif name == "fmaf":
                function.argtypes = (ctypes.c_float,) * 3
                function.restype = ctypes.c_float
            else:
                function.argtypes = (ctypes.c_float,)
                function.restype = ctypes.c_float
            return function
        except (AttributeError, OSError):
            continue
    raise RuntimeError(f"platform C math library must provide {name}")


_FMA = _load_math_function("fma", double_args=True)
_FMAF = _load_math_function("fmaf")
_LOG2F = _load_math_function("log2f")
_EXP2F = _load_math_function("exp2f")


def f32(value: float | np.floating) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(value)


def fma32(a: float, b: float, c: float = 0.0) -> np.float32:
    return np.float32(_FMAF(float(f32(a)), float(f32(b)), float(f32(c))))


def fma64(a: float, b: float, c: float = 0.0) -> float:
    return float(_FMA(float(a), float(b), float(c)))


def _f32_from_bits(bits: int) -> np.float32:
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def _f64_from_bits(bits: int) -> float:
    return struct.unpack(">d", int(bits).to_bytes(8, "big"))[0]


def amount_exponent(amount: int) -> np.float32:
    """Mirror the recovered binary64 expansion, then one binary32 narrowing."""
    if not 0 < int(amount) <= 0x7FFFFFFF:
        raise ValueError("amount must be a positive signed int32")
    clamped = min(int(amount), 200)
    d = (float(clamped) - 100.0) / 33.0
    expanded = fma64(d, 15.0, 100.0)
    base = expanded if clamped > 100 else float(clamped)
    hundredth = _f64_from_bits(0x3F847AE147AE147B)
    return f32(base * hundredth)


@dataclass(frozen=True)
class PayloadCase:
    payload: bytes
    rows: int
    columns: int
    points_n: int
    spacing_v: float
    spacing_h: float
    origin_v: float
    origin_h: float
    weights: tuple[np.float32, ...]
    gamma: np.float32
    gain_min: np.float32
    gain_max: np.float32
    source_values: np.ndarray


def _case_samples(data_type: int) -> tuple[list[float | int], float, float]:
    if data_type == 0:
        # Include quantized values below and above unity and at both endpoints.
        return [0, 1, 64, 127, 128, 129, 192, 254, 255, 51, 205, 77], 0.5, 1.5
    if data_type == 1:
        return [0, 1, 16384, 32767, 32768, 32769, 49152, 65534, 65535, 12345, 54321, 27673], 0.5, 1.5
    if data_type == 2:
        return [2.0**-12, 0.25, 0.5, 0.9990234375, 1.0, 1.0009765625, 1.5, 4096.0, 0.3, 3.7, 0.875, 1.125], float(MIN_GAIN), float(MAX_GAIN)
    if data_type == 3:
        below_one = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
        above_one = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
        return [2.0**-12, 0.25, 0.5, below_one, 1.0, above_one, 1.5, 4096.0, 0.3, 3.7, 0.875, 1.125], float(MIN_GAIN), float(MAX_GAIN)
    raise ValueError("datatype must be in [0, 3]")


def make_payload(
    data_type: int,
    byteorder: int,
    *,
    weights=(0.125, -0.25, 0.5, 0.75, -1.0),
    gamma: float = 1.375,
    spacing_v: float = 1.25,
    spacing_h: float = 0.8,
    origin_v: float = -2.5,
    origin_h: float = 3.25,
) -> PayloadCase:
    """Create a local extended-tag payload and independently decode its samples."""
    if byteorder not in (0, 1):
        raise ValueError("byteorder must be little (0) or big (1)")
    prefix = "<" if byteorder == 0 else ">"
    samples, low, high = _case_samples(data_type)
    rows, columns, points_n = 2, 3, 2
    weights = tuple(f32(x) for x in weights)
    gamma = f32(gamma)
    gain_min, gain_max = f32(low), f32(high)

    header = struct.pack(
        prefix + "II4dI5fI3f",
        rows,
        columns,
        spacing_v,
        spacing_h,
        origin_v,
        origin_h,
        points_n,
        *weights,
        data_type,
        float(gamma),
        float(gain_min),
        float(gain_max),
    )
    assert len(header) == 80

    if data_type == 0:
        encoded = bytes(int(value) for value in samples)
        reciprocal = _f32_from_bits(0x3B808081)
        delta = f32(float(gain_max) - float(gain_min))
        decoded = [fma32(f32(f32(int(code)) * reciprocal), delta, gain_min) for code in samples]
    elif data_type == 1:
        encoded = b"".join(struct.pack(prefix + "H", int(value)) for value in samples)
        reciprocal = _f32_from_bits(0x37800080)
        delta = f32(float(gain_max) - float(gain_min))
        decoded = [fma32(f32(f32(int(code)) * reciprocal), delta, gain_min) for code in samples]
    elif data_type == 2:
        encoded = b"".join(struct.pack(prefix + "e", float(value)) for value in samples)
        decoded = [f32(struct.unpack(prefix + "e", encoded[i : i + 2])[0]) for i in range(0, len(encoded), 2)]
    else:
        encoded = b"".join(struct.pack(prefix + "f", float(value)) for value in samples)
        decoded = [f32(struct.unpack(prefix + "f", encoded[i : i + 4])[0]) for i in range(0, len(encoded), 4)]

    source_values = np.asarray(decoded, dtype=np.float32)
    assert np.isfinite(source_values).all()
    assert np.all(source_values >= MIN_GAIN) and np.all(source_values <= MAX_GAIN)
    return PayloadCase(
        header + encoded,
        rows,
        columns,
        points_n,
        spacing_v,
        spacing_h,
        origin_v,
        origin_h,
        weights,
        gamma,
        gain_min,
        gain_max,
        source_values,
    )


def expected_cache_values(source_values: np.ndarray, amount: int, lookup_flag: int, predicate: int) -> tuple[np.ndarray, np.float32, bool]:
    exponent = amount_exponent(amount)
    negative_remap = lookup_flag == 0 and (predicate & 1) != 0
    result = np.empty(source_values.size, dtype=np.float32)
    for index, source in enumerate(source_values):
        logarithm = np.float32(_LOG2F(float(f32(source))))
        u = fma32(exponent, logarithm, 0.0)
        transformed = u
        if negative_remap and u < np.float32(0.0):
            one_plus_u = f32(np.float32(1.0) + u)
            transformed = np.float32(min(float(one_plus_u), 0.0))
        result[index] = np.float32(_EXP2F(float(transformed)))
    return result, exponent, negative_remap


def expected_prepared(case: PayloadCase, values: np.ndarray, exponent: np.float32, negative_remap: bool, rect=(0, 0, 64, 128), scalar=0.75):
    top, left, bottom, right = (int(x) for x in rect)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("rectangle spans must be positive")

    # Reproduce the two normalized coordinate transforms in binary64, then
    # narrow the selected fields once to binary32.
    u_scale = f32(fma64(1.0 / case.spacing_h, 1.0 / float(width)))
    u_bias = f32(fma64(1.0 / case.spacing_h, 0.0) - case.origin_h / case.spacing_h)
    v_scale = f32(fma64(1.0 / case.spacing_v, 1.0 / float(height)))
    v_bias = f32(fma64(1.0 / case.spacing_v, 0.0) - case.origin_v / case.spacing_v)
    fields = np.asarray(
        [
            *case.weights,
            u_scale,
            u_bias,
            v_scale,
            v_bias,
            0.0,
            f32(case.columns) - f32(1.0),
            0.0,
            f32(case.rows) - f32(1.0),
            f32(scalar),
            f32(case.points_n),
            case.gamma,
        ],
        dtype=np.float32,
    )
    stride_v, stride_u = case.columns * case.points_n, case.points_n
    geometry = np.asarray(
        [stride_v, stride_u, case.columns - 1, case.rows - 1, case.points_n - 1],
        dtype=np.uint32,
    )
    meta = np.asarray(
        [
            case.rows,
            case.columns,
            case.spacing_v,
            case.spacing_h,
            case.origin_v,
            case.origin_h,
            case.points_n,
            *case.weights,
            2,
            case.gamma,
            case.gain_min,
            case.gain_max,
            stride_v,
            stride_u,
            int(values.size),
            exponent,
            1.0 if negative_remap else 0.0,
        ],
        dtype=np.float64,
    )
    return meta, fields, geometry, values


def bits32(values) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32)


def apply_pivot_zero(value: float, gain: float) -> np.float32:
    """Independent binary32 apply reference for pivot=0 and clip disabled."""
    sample = f32(value)
    gain32 = f32(gain)
    # Pivot preparation yields reciprocal=1, factor=1, bias=offset=0.
    curve = fma32(sample, 1.0, 0.0)
    scaled = f32(curve * gain32)
    return fma32(scaled, 1.0, 0.0)
