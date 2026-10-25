"""Independent float32 oracle for the inline balance3 basic prepared row.

The arithmetic follows the recovered 007ee058..007ee150 instruction order.
The only fused operations are delegated to the versioned balance3 finish
oracle's libm ``fmaf`` wrapper; this module never calls the native kernel.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_F = np.float32
_FINISH_ORACLE_PATH = Path(__file__).with_name("balance3_finish_oracle.py")
_FINISH_SPEC = importlib.util.spec_from_file_location(
    "balance3_basic_finish_fmaf_reference", _FINISH_ORACLE_PATH)
assert _FINISH_SPEC is not None and _FINISH_SPEC.loader is not None
_FINISH = importlib.util.module_from_spec(_FINISH_SPEC)
_FINISH_SPEC.loader.exec_module(_FINISH)
fma32 = _FINISH.fma32
_fminnm32 = _FINISH._fminnm32


def add32(a, b) -> np.float32:
    """One separately rounded binary32 addition."""
    with np.errstate(all="ignore"):
        return _F(_F(a) + _F(b))


def mul32(a, b) -> np.float32:
    """One separately rounded binary32 multiplication."""
    with np.errstate(all="ignore"):
        return _F(_F(a) * _F(b))


def sub32(a, b) -> np.float32:
    """One separately rounded binary32 subtraction, for wrong-path probes."""
    with np.errstate(all="ignore"):
        return _F(_F(a) - _F(b))


def _flat_f32(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must have dtype float32")
    if array.ndim != 1 or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be a contiguous flat float32 array")
    return array


def accumulator(source: np.ndarray, center: int, stride: int,
                diff: np.ndarray, lane: int, row_even: int) -> np.float32:
    """Evaluate the raw direct accumulator, without delta or re-addition."""
    src = _flat_f32(source, "source")
    differences = _flat_f32(diff, "diff")
    if row_even not in (0, 1):
        raise ValueError("row_even must be zero or one")
    lane, center, stride = int(lane), int(center), int(stride)
    if lane < 0 or lane >= differences.size:
        raise ValueError("lane is outside the diff row")
    if center - stride + lane - 1 < 0 or center + stride + lane + 1 >= src.size:
        raise ValueError("source does not contain the guarded neighborhood")

    acc = add32(src[center - stride + lane - 1],
                src[center - stride + lane + 1])
    acc = add32(acc, src[center + stride + lane - 1])
    acc = add32(acc, src[center + stride + lane + 1])
    acc = mul32(acc, _F(0.125))
    acc = fma32(src[center + lane], _F(0.5), acc)
    sign = _F(-0.5 if row_even else 0.5)
    return fma32(differences[lane], sign, acc)


def reference_row(source: np.ndarray, center: int, stride: int,
                  diff: np.ndarray, width: int, phase: int, row_even: int,
                  destination: np.ndarray) -> np.ndarray:
    """Return output bits while preserving unselected lanes and padding."""
    src = _flat_f32(source, "source")
    differences = _flat_f32(diff, "diff")
    original = _flat_f32(destination, "destination")
    width, center, stride = int(width), int(center), int(stride)
    if width <= 0 or phase not in (0, 1) or row_even not in (0, 1):
        raise ValueError("width must be positive and flags must be binary")
    if differences.size < width or original.size < width:
        raise ValueError("diff and destination must contain width samples")
    if center - stride - 1 < 0 or center + stride + width >= src.size:
        raise ValueError("source does not contain the guarded neighborhood")

    out = original.copy()
    for lane in range(width):
        if (lane & 1) == phase:
            continue
        value = _fminnm32(accumulator(src, center, stride, differences,
                                     lane, row_even), _F(1.0))
        if value < _F(0.0):
            value = _F(0.0)
        out[lane] = value
    return out


def uint32_bits(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float32):
        raise ValueError("values must have dtype float32")
    return array.view(np.uint32)


def assert_float32_bits(actual: np.ndarray, expected: np.ndarray,
                        *, message: str = "float32 bit patterns differ") -> None:
    np.testing.assert_array_equal(uint32_bits(actual), uint32_bits(expected),
                                  err_msg=message)
