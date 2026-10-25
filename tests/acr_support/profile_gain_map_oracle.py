"""Independent reference for the recovered PGTM payload-to-CPU preparation.

This module transcribes the bounded serialized-map reader and preparation
contract from the checked-in ARM64 raw evidence. It does not import product
code. Its CPU pixel evaluation remains in ``profile_gain_oracle.py``; this
module supplies independently decoded table data and prepared bridge arrays.

The reference covers the recovered DNG-tag payload subset and the observed
call-snapshot rectangle/map matrices. It does not claim complete Adobe ABI,
wrapper selection, GPU behavior, or Adobe-render equivalence.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np


RAW_EVIDENCE = (
    "research/acr/reports/evidence/2026-10-24-profile-gain-map-binding/"
    "raw-012e724c-012e764c.txt",
    "research/acr/reports/evidence/2026-10-24-profile-gain-map-binding/"
    "raw-012e764c-012e7a4c.txt",
    "research/acr/reports/evidence/2026-10-24-profile-gain-map-binding/"
    "raw-012e675c-012e6b5c.txt",
    "research/acr/reports/evidence/2026-10-24-profile-gain-map-binding/"
    "raw-01386738-01386890.txt",
    "research/acr/reports/evidence/2026-10-24-profile-gain-map-binding/"
    "raw-0126979c-01269b50.txt",
    "research/acr/reports/evidence/2026-10-24-normalized-early-cpu-inventory/"
    "v11/raw-0138661c-01386a1c.txt",
)

TAG_LEGACY = 52525
TAG_EXTENDED = 52544
LEGACY_HEADER_BYTES = 64
EXTENDED_HEADER_BYTES = 80
MAX_TABLE_COUNT = 1 << 24
MIN_GAIN = 2.0**-12
MAX_GAIN = 4096.0

OK = 0
INVALID_ARGUMENT = 1
INVALID_HEADER = 2
UNSUPPORTED_TYPE = 3
INVALID_LENGTH = 4
INVALID_GAIN_VALUE = 5
INTEGER_OVERFLOW = 6
RESOURCE_LIMIT = 7
ALLOCATION_FAILURE = 8
SHORT_OUTPUT = 9
ALIAS = 10
ALIGNMENT = 11
POINTER_SPAN_OVERFLOW = 12
INVALID_GEOMETRY = 13
NUMERIC_FAILURE = 14


class ContractError(ValueError):
    """An expected bridge status raised by the independent reference."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)


def _load_math_function(name: str):
    if os.name == "nt":
        libraries = ("ucrtbase.dll", "msvcrt.dll", ctypes.util.find_library("m"), None)
    else:
        libraries = (ctypes.util.find_library("m"), None)
    for library_name in libraries:
        try:
            library = ctypes.CDLL(library_name) if library_name else ctypes.CDLL(None)
            function = getattr(library, name)
            function.argtypes = (
                (ctypes.c_float, ctypes.c_float, ctypes.c_float)
                if name == "fmaf"
                else (ctypes.c_double, ctypes.c_double, ctypes.c_double)
            )
            function.restype = ctypes.c_float if name == "fmaf" else ctypes.c_double
            return function
        except (AttributeError, OSError):
            continue
    raise RuntimeError(f"platform C math library must provide {name}")


_FMAF = _load_math_function("fmaf")
_FMA = _load_math_function("fma")


def f32(value: float | np.floating) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(value)


def fma32(a: float, b: float, c: float) -> np.float32:
    return np.float32(_FMAF(float(f32(a)), float(f32(b)), float(f32(c))))


def fma64(a: float, b: float, c: float) -> float:
    return float(_FMA(float(a), float(b), float(c)))


def _f32_bits(bits: int) -> np.float32:
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def _float32_mul(a: float, b: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(np.float32(a) * np.float32(b))


def _signed32(value: int) -> bool:
    return -(1 << 31) <= value <= (1 << 31) - 1


def _wrap_neg32(value: int) -> int:
    unsigned = (-int(value)) & 0xFFFFFFFF
    return unsigned if unsigned < 0x80000000 else unsigned - 0x100000000


@dataclass(frozen=True)
class ParsedPayload:
    rows: int
    cols: int
    spacing_v: float
    spacing_h: float
    origin_v: float
    origin_h: float
    slices: int
    weights: tuple[np.float32, ...]
    data_type: int
    gamma: np.float32
    gain_min: np.float32
    gain_max: np.float32
    table: np.ndarray

    @property
    def count(self) -> int:
        return self.rows * self.cols * self.slices


@dataclass(frozen=True)
class PreparedMap:
    meta: np.ndarray
    fields: np.ndarray
    geometry: np.ndarray
    table: np.ndarray
    source_width: int
    source_height: int


def parse_payload(payload: bytes | bytearray | memoryview, tag: int, byteorder: int) -> ParsedPayload:
    """Decode a tag 52525/52544 PGTM payload without mutating its input."""
    if tag not in (TAG_LEGACY, TAG_EXTENDED) or byteorder not in (0, 1):
        raise ContractError(INVALID_ARGUMENT, "tag or byteorder is outside the public contract")
    try:
        blob = memoryview(payload).cast("B")
    except (TypeError, ValueError):
        blob = memoryview(bytes(payload))
    header_size = LEGACY_HEADER_BYTES if tag == TAG_LEGACY else EXTENDED_HEADER_BYTES
    if blob.nbytes < header_size:
        raise ContractError(INVALID_LENGTH, "payload is shorter than its tag header")

    prefix = ">" if byteorder else "<"
    try:
        rows, cols = struct.unpack_from(prefix + "II", blob, 0)
        spacing_v, spacing_h, origin_v, origin_h = struct.unpack_from(prefix + "dddd", blob, 8)
        slices = struct.unpack_from(prefix + "I", blob, 40)[0]
        weights = tuple(np.float32(x) for x in struct.unpack_from(prefix + "5f", blob, 44))
        if tag == TAG_EXTENDED:
            data_type = struct.unpack_from(prefix + "I", blob, 64)[0]
            gamma, gain_min, gain_max = (
                np.float32(x) for x in struct.unpack_from(prefix + "3f", blob, 68)
            )
        else:
            data_type = 3
            gamma = gain_min = gain_max = np.float32(1.0)
    except struct.error as exc:
        raise ContractError(INVALID_HEADER, "payload header cannot be decoded") from exc

    if data_type > 3:
        raise ContractError(UNSUPPORTED_TYPE, "payload sample type is unsupported")
    if rows == 0 or cols == 0 or rows > 0x7FFFFFFF or cols > 0x7FFFFFFF or slices == 0:
        raise ContractError(INVALID_HEADER, "V, H and N must be positive signed dimensions")

    # The recovered singleton canonicalization precedes axis validation.
    if rows == 1:
        spacing_v, origin_v = 1.0, 0.0
    if cols == 1:
        spacing_h, origin_h = 1.0, 0.0
    if (
        not math.isfinite(spacing_v)
        or not math.isfinite(spacing_h)
        or spacing_v <= 0.0
        or spacing_h <= 0.0
        or not math.isfinite(origin_v)
        or not math.isfinite(origin_h)
        or any(not np.isfinite(x) for x in weights)
    ):
        raise ContractError(INVALID_HEADER, "map spacing, origins and coefficients must be finite")
    if tag == TAG_EXTENDED and (
        not np.isfinite(gamma)
        or float(gamma) < 0.125
        or float(gamma) > 8.0
        or not np.isfinite(gain_min)
        or float(gain_min) < MIN_GAIN
        or not np.isfinite(gain_max)
        or float(gain_max) > MAX_GAIN
    ):
        raise ContractError(INVALID_GAIN_VALUE, "gamma or gain range is outside the recovered limits")

    count = rows * cols * slices
    if count > 0xFFFFFFFF:
        raise ContractError(INTEGER_OVERFLOW, "V*H*N exceeds uint32")
    if count >= MAX_TABLE_COUNT:
        raise ContractError(RESOURCE_LIMIT, "table has at least 2^24 values")
    sample_bytes = (1, 2, 2, 4)[data_type]
    expected_size = header_size + count * sample_bytes
    if blob.nbytes != expected_size:
        raise ContractError(INVALID_LENGTH, "payload length is not exactly header plus table")

    data = blob[header_size:]
    if data_type == 0:
        encoded = np.frombuffer(data, dtype=np.uint8)
        reciprocal = _f32_bits(0x3B808081)
        delta = f32(float(gain_max) - float(gain_min))
        table = np.empty(count, dtype=np.float32)
        for index, code in enumerate(encoded):
            normalized = _float32_mul(int(code), reciprocal)
            table[index] = fma32(normalized, delta, gain_min)
    elif data_type == 1:
        encoded = np.frombuffer(data, dtype=np.dtype(prefix + "u2"))
        reciprocal = _f32_bits(0x37800080)
        delta = f32(float(gain_max) - float(gain_min))
        table = np.empty(count, dtype=np.float32)
        for index, code in enumerate(encoded):
            normalized = _float32_mul(int(code), reciprocal)
            table[index] = fma32(normalized, delta, gain_min)
    elif data_type == 2:
        encoded = np.frombuffer(data, dtype=np.dtype(prefix + "f2"))
        table = encoded.astype(np.float32, copy=True)
    else:
        encoded = np.frombuffer(data, dtype=np.dtype(prefix + "f4"))
        table = encoded.astype(np.float32, copy=True)

    if (
        table.size != count
        or not np.isfinite(table).all()
        or np.any(table < np.float32(MIN_GAIN))
        or np.any(table > np.float32(MAX_GAIN))
    ):
        raise ContractError(INVALID_GAIN_VALUE, "decoded table value is outside the recovered range")
    table.setflags(write=False)
    return ParsedPayload(
        rows=rows,
        cols=cols,
        spacing_v=spacing_v,
        spacing_h=spacing_h,
        origin_v=origin_v,
        origin_h=origin_h,
        slices=slices,
        weights=weights,
        data_type=data_type,
        gamma=gamma,
        gain_min=gain_min,
        gain_max=gain_max,
        table=table,
    )


def _matrix_product3(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for row in range(3):
        result_row: list[float] = []
        for col in range(3):
            acc = 0.0
            for k in range(3):
                acc = fma64(left[row][k], right[k][col], acc)
            result_row.append(acc)
        result.append(result_row)
    return result


def prepare_map(
    parsed: ParsedPayload,
    source_rect: Sequence[int],
    scalar: float,
) -> PreparedMap:
    """Build observed FP32 bridge fields from a prepared payload and rect.

    ``source_rect`` is signed-int32 ``[top,left,bottom,right]``. Matrix lanes
    are calculated from the actual A/B call snapshots in inventory v11 and
    narrowed only after the complete ordered FP64 A*B product.
    """
    if len(source_rect) != 4:
        raise ContractError(INVALID_GEOMETRY, "source rectangle must contain four signed int32 lanes")
    top, left, bottom, right = (int(x) for x in source_rect)
    if not all(_signed32(x) for x in (top, left, bottom, right)):
        raise ContractError(INVALID_GEOMETRY, "source rectangle lanes must fit signed int32")
    width, height = right - left, bottom - top
    if not (1 <= width <= 0x7FFFFFFF and 1 <= height <= 0x7FFFFFFF):
        raise ContractError(INVALID_GEOMETRY, "source rectangle spans must be positive signed int32")

    # A normalizes PGTM map coordinates; B normalizes source bounds. Preserve
    # the two independent divisions and the signed-int32 origin negation.
    a = [
        [1.0 / parsed.spacing_h, 0.0, -parsed.origin_h / parsed.spacing_h],
        [0.0, 1.0 / parsed.spacing_v, -parsed.origin_v / parsed.spacing_v],
        [0.0, 0.0, 1.0],
    ]
    b = [
        [1.0 / float(width), 0.0, float(_wrap_neg32(left)) / float(width)],
        [0.0, 1.0 / float(height), float(_wrap_neg32(top)) / float(height)],
        [0.0, 0.0, 1.0],
    ]
    matrix = _matrix_product3(a, b)
    scalar32 = f32(scalar)
    selected = (matrix[0][0], matrix[0][2], matrix[1][1], matrix[1][2], float(scalar32))
    narrowed = tuple(f32(x) for x in selected)
    if (
        not math.isfinite(float(scalar))
        or not np.isfinite(scalar32)
        or any(not math.isfinite(x) for row in matrix for x in row)
        or any(not np.isfinite(x) for x in narrowed)
    ):
        raise ContractError(NUMERIC_FAILURE, "preparation produced a nonfinite coordinate field")

    upper_u = f32(f32(parsed.cols) + f32(-1.0))
    upper_v = f32(f32(parsed.rows) + f32(-1.0))
    index_scale = f32(parsed.slices)
    if not np.isfinite(upper_u) or not np.isfinite(upper_v) or not np.isfinite(index_scale):
        raise ContractError(NUMERIC_FAILURE, "dimensions cannot be represented as finite FP32 fields")

    u_scale, u_bias, v_scale, v_bias, prelookup_scale = narrowed
    fields = np.asarray(
        [
            *parsed.weights,
            u_scale,
            u_bias,
            v_scale,
            v_bias,
            0.0,
            upper_u,
            0.0,
            upper_v,
            prelookup_scale,
            index_scale,
            parsed.gamma,
        ],
        dtype=np.float32,
    )
    geometry = np.asarray(
        [parsed.cols * parsed.slices, parsed.slices, parsed.cols - 1, parsed.rows - 1, parsed.slices - 1],
        dtype=np.uint32,
    )
    count = parsed.count
    meta = np.asarray(
        [
            parsed.rows,
            parsed.cols,
            parsed.spacing_v,
            parsed.spacing_h,
            parsed.origin_v,
            parsed.origin_h,
            parsed.slices,
            *parsed.weights,
            parsed.data_type,
            parsed.gamma,
            parsed.gain_min,
            parsed.gain_max,
            parsed.cols * parsed.slices,
            parsed.slices,
            count,
        ],
        dtype=np.float64,
    )
    return PreparedMap(meta, fields, geometry, parsed.table, width, height)


def payload_to_prepared(
    payload: bytes | bytearray | memoryview,
    tag: int,
    byteorder: int,
    source_rect: Sequence[int],
    scalar: float,
) -> PreparedMap:
    return prepare_map(parse_payload(payload, tag, byteorder), source_rect, scalar)


def expected_bits32(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32)


def expected_bits64(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64)).view(np.uint64)
