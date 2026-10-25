"""Independent binary32 oracle for the recovered ACR profile-gain loop.

This is a scalar transcription of the ARM64 instructions at
``raw-012cd2d4-012cd6d4.txt`` (function entry 0x012cd2d4), using the fixed
prepared-field/geometry contract in the bridge tests.  It intentionally does
not import product code or generated analysis scripts.  It is an oracle for
the bounded CPU subset only; it does not claim full-ABI or Adobe runtime
equivalence.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
from typing import Sequence

import numpy as np


ORACLE_VERSION = "acr-profile-gain-arm64-entryfi-2026-10-24-v1"
ORACLE_SCOPE = "binary32 CPU profile-gain loop for finite, validated prepared buffers"
RAW_EVIDENCE = "research/acr/reports/evidence/2026-10-24-normalized-early-cpu-inventory/v6/raw-012cd2d4-012cd6d4.txt"
ENTRY = "0x012cd2d4"

# ARM64 evidence anchors, copied from the raw listing rather than inferred
# from implementation source.  Offsets are file/listing addresses.
INSTRUCTION_EVIDENCE = {
    "u_v_transform": ("12cd384", "1f031ca3 fmadd s3, s5, s3, s7", "1f0440c4 fmadd s4, s6, s4, s16"),
    "u_v_strict_bounds_and_truncation": ("12cd3ac", "1e254c63 fcsel s3, s3, s5, mi", "12cd3d0 fcvtzs w11, s3", "12cd3f4 fcvtzs w21, s4"),
    "ordered_minmax_and_polynomial": ("12cd414", "1e214c43 fcsel s3, s2, s1, mi", "1e21cc42 fcsel s2, s2, s1, gt", "12cd438 fmadd s0, s4, s0, s5", "12cd448 fmadd s0, s7, s2, s0"),
    "prelookup_powf_and_index": ("12cd450", "1e200820 fmul s0, s1, s0", "1e2b7800 fminnm s0, s0, s11", "12cd468 bl powf", "12cd47c fcvtzs w8, s0"),
    "q_cap_and_table_strides": ("12cd480", "1a89b108 csel w8, w8, w9, lt", "12cd498 mul w12, w11, w21", "12cd4b8 mul w13, w10, w13"),
    "z_then_u_then_v_fma": ("12cd50c", "1f010801 fmadd s1, s0, s1, s2", "12cd52c fmadd s1, s13, s2, s1", "12cd53c fmadd s0, s14, s0, s1"),
}


def _load_libm_float_function(name: str):
    if os.name == "nt":
        library_names = ("ucrtbase.dll", "msvcrt.dll", ctypes.util.find_library("m"), None)
    else:
        library_names = (ctypes.util.find_library("m"), None)
    for library_name in library_names:
        try:
            library = ctypes.CDLL(library_name) if library_name else ctypes.CDLL(None)
            function = getattr(library, name)
            function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float) if name == "fmaf" else (ctypes.c_float, ctypes.c_float)
            function.restype = ctypes.c_float
            return function
        except (AttributeError, OSError):
            continue
    raise RuntimeError(f"platform C math library must provide {name}")


_FMAF = _load_libm_float_function("fmaf")
_POWF = _load_libm_float_function("powf")


def f32(value: float | np.floating) -> np.float32:
    return np.float32(value)


def _binary32_binary(a: float, b: float, operation) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(operation(f32(a), f32(b)))


def add32(a: float, b: float) -> np.float32:
    return _binary32_binary(a, b, np.add)


def sub32(a: float, b: float) -> np.float32:
    return _binary32_binary(a, b, np.subtract)


def mul32(a: float, b: float) -> np.float32:
    return _binary32_binary(a, b, np.multiply)


def fma32(a: float, b: float, c: float) -> np.float32:
    return np.float32(_FMAF(float(f32(a)), float(f32(b)), float(f32(c))))


def pow32(base: float, exponent: float) -> np.float32:
    return np.float32(_POWF(float(f32(base)), float(f32(exponent))))


def bits32(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32)


def _min_second_on_equal(a: np.float32, b: np.float32) -> np.float32:
    """Ordered AArch64 comparison selection: strictly less selects `a`; tie `b`."""
    return a if a < b else b


def _max_second_on_equal(a: np.float32, b: np.float32) -> np.float32:
    """Ordered AArch64 comparison selection: strictly greater selects `a`; tie `b`."""
    return a if a > b else b


def _clamp_ordered(value: np.float32, lower: np.float32, upper: np.float32) -> np.float32:
    # Raw ARM64 selects upper when the value is equal to upper, then replaces
    # it with lower only when lower is strictly greater than the value.
    if not value < upper:
        value = upper
    if lower > value:
        value = lower
    return value


def _trunc_to_int32(value: np.float32) -> int:
    if not np.isfinite(value) or float(value) < -2147483648.0 or float(value) >= 2147483648.0:
        raise ValueError("oracle inputs must keep FCVTZS in the signed int32 domain")
    return math.trunc(float(value))


def reference_sample(
    fields: Sequence[float],
    geometry: Sequence[int],
    table: np.ndarray | Sequence[float],
    p0: float,
    p1: float,
    p2: float,
    coord_v: float,
    coord_u: float,
) -> np.float32:
    """Evaluate one active sample in ARM64 instruction order.

    ``fields`` are c0..c4, uScale/uBias, vScale/vBias, u lower/upper,
    v lower/upper, prelookup scale, index scale and pow exponent.  Geometry is
    ``stride_v, stride_u, max_u, max_v, max_z`` in float-table elements.
    """
    if len(fields) != 16 or len(geometry) != 5:
        raise ValueError("expected 16 prepared fields and 5 geometry words")
    f = tuple(f32(x) for x in fields)
    stride_v, stride_u, max_u, max_v, max_z = (int(x) for x in geometry)
    values = np.asarray(table, dtype=np.float32).reshape(-1)
    p0, p1, p2 = f32(p0), f32(p1), f32(p2)
    coord_v, coord_u = f32(coord_v), f32(coord_u)

    # ARM FMADD coordinates are (scale * (coord + .5)) + bias.  The add of
    # .5 is a separate FADD before the fused multiply-add.
    u = fma32(f[5], add32(coord_u, f32(0.5)), f[6])
    v = fma32(f[7], add32(coord_v, f32(0.5)), f[8])
    u = _clamp_ordered(u, f[9], f[10])
    v = _clamp_ordered(v, f[11], f[12])

    u0 = _trunc_to_int32(u)
    v0 = _trunc_to_int32(v)
    u1 = min(u0 + 1, max_u)
    v1 = min(v0 + 1, max_v)
    fu = sub32(u, f32(u0))
    fv = sub32(v, f32(v0))

    # Preserve the compare/conditional-select operand order from the raw
    # listing.  On equal values the later operand is selected, which matters
    # for +0/-0 ties.
    minimum12 = _min_second_on_equal(p1, p2)
    maximum12 = _max_second_on_equal(p1, p2)
    minimum = _min_second_on_equal(p0, minimum12)
    maximum = _max_second_on_equal(p0, maximum12)

    # c1*p1 is a distinct FMUL; the following four terms are FMADDs.
    z = mul32(f[1], p1)
    z = fma32(f[0], p0, z)
    z = fma32(f[2], p2, z)
    z = fma32(f[3], minimum, z)
    z = fma32(f[4], maximum, z)
    z = mul32(z, f[13])
    if z > f32(1.0):
        z = f32(1.0)
    # FMINNM(z, 1) preserves a finite negative zero; the subsequent ordered
    # less-than selects +0 only for values strictly below zero.
    if z < f32(0.0):
        z = f32(0.0)

    powered = pow32(z, f[15])
    selected = z if f[15] == f32(1.0) else powered
    q = mul32(selected, f[14])
    q_base = _trunc_to_int32(q)
    z0 = min(q_base, max_z)
    z1 = min(z0 + 1, max_z)
    fz = sub32(q, f32(z0))

    def at(zi: int, ui: int, vi: int) -> np.float32:
        index = zi + ui * stride_u + vi * stride_v
        if index < 0 or index >= values.size:
            raise ValueError("prepared geometry addresses beyond the supplied table")
        return values[index]

    # Eight corners.  Interpolation order is z, then u, then v.  Every diff
    # is a separate binary32 subtraction before its fmaf call.
    z00v0 = fma32(fz, sub32(at(z1, u0, v0), at(z0, u0, v0)), at(z0, u0, v0))
    z01v0 = fma32(fz, sub32(at(z1, u1, v0), at(z0, u1, v0)), at(z0, u1, v0))
    z00v1 = fma32(fz, sub32(at(z1, u0, v1), at(z0, u0, v1)), at(z0, u0, v1))
    z01v1 = fma32(fz, sub32(at(z1, u1, v1), at(z0, u1, v1)), at(z0, u1, v1))
    u_v0 = fma32(fu, sub32(z01v0, z00v0), z00v0)
    u_v1 = fma32(fu, sub32(z01v1, z00v1), z00v1)
    return fma32(fv, sub32(u_v1, u_v0), u_v0)


def reference_rows(
    fields: Sequence[float],
    geometry: Sequence[int],
    table: np.ndarray | Sequence[float],
    inputs: Sequence[np.ndarray],
    outer: int,
    inner: int,
    value_stride: int,
    coord_stride: int,
) -> np.ndarray:
    """Return compact active results while reading padded row-strided inputs."""
    if len(inputs) != 5:
        raise ValueError("expected p0, p1, p2, coord_v, coord_u")
    p0, p1, p2, cv, cu = (np.asarray(x, dtype=np.float32).reshape(-1) for x in inputs)
    output = np.empty((outer, inner), dtype=np.float32)
    for row in range(outer):
        for col in range(inner):
            output[row, col] = reference_sample(
                fields,
                geometry,
                table,
                p0[row * value_stride + col],
                p1[row * value_stride + col],
                p2[row * value_stride + col],
                cv[row * coord_stride + col],
                cu[row * coord_stride + col],
            )
    return output
