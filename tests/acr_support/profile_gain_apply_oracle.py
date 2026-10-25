"""Independent scalar oracle for the PGTM profile-gain tail consumer.

The pivot setup and per-value arithmetic follow the original ARM64 listing in
the v13 normalized CPU inventory.  In particular, the FMSUB instruction is
interpreted as ``addend - op1 * op2`` (ARM ARM D4.33), and all intermediate
binary32 operations remain in instruction order.  This is a numeric consumer
oracle only; it does not claim complete Adobe RGB-pipeline equivalence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import profile_gain_oracle as arithmetic


RAW_EVIDENCE = (
    "research/acr/reports/evidence/2026-10-24-normalized-early-cpu-inventory/"
    "v13/raw-0083a798-0083a998.txt",
    "research/acr/reports/evidence/2026-10-24-normalized-early-cpu-inventory/"
    "v13/raw-0134fab0-0134feb0.txt",
    "research/acr/reports/evidence/2026-10-24-normalized-early-cpu-inventory/"
    "v13/raw-0134feb0-013501a0.txt",
)
ARM_FMSUB_REFERENCE = "https://documentation-service.arm.com/static/6245c734b059dc5ff9a8bdab"

SCALAR_MODE = 0
CPU_VECTOR_MODE = 1


def f32(value: float | np.floating) -> np.float32:
    """Round explicitly to IEEE binary32."""
    return np.float32(value)


def add32(left: float, right: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(np.float32(left) + np.float32(right))


def sub32(left: float, right: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(np.float32(left) - np.float32(right))


def mul32(left: float, right: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(np.float32(left) * np.float32(right))


def fma32(left: float, right: float, addend: float) -> np.float32:
    """One correctly rounded binary32 fused multiply-add."""
    return arithmetic.fma32(f32(left), f32(right), f32(addend))


@dataclass(frozen=True)
class PivotTerms:
    pivot: np.float32
    delta: np.float32
    reciprocal: np.float32
    factor: np.float32
    bias: np.float32
    offset: np.float32


def prepare_pivot(pivot: float) -> PivotTerms:
    """Transcribe the scalar pivot setup without algebraic simplification.

    The division is evaluated even for pivot 1.0, before the equality select
    chooses positive zero for the reciprocal.
    """
    pivot64 = float(pivot)
    if not math.isfinite(pivot64) or abs(pivot64) > float(np.finfo(np.float32).max):
        raise ValueError("pivot must be finite and representable as binary32")

    a = f32(pivot64)
    if not np.isfinite(a):
        raise ValueError("pivot must be finite after binary32 conversion")
    delta = sub32(a, f32(1.0))
    with np.errstate(all="ignore"):
        reciprocal = np.float32(np.float32(-1.0) / delta)
    if a == f32(1.0):
        reciprocal = f32(+0.0)

    factor = f32(-delta)
    # ARM FMSUB computes addend - op1*op2, hence 0 - reciprocal*pivot.
    bias = fma32(f32(-reciprocal), a, f32(+0.0))
    # The following FMADD has a separately prepared +0 multiply operand.
    offset = fma32(delta, f32(+0.0), a)
    return PivotTerms(a, delta, reciprocal, factor, bias, offset)


def _clip_scalar_fminnm(value: np.float32) -> np.float32:
    # FMINNM returns the numeric operand when exactly one operand is NaN.
    if np.isnan(value):
        return f32(1.0)
    return value if value < f32(1.0) else f32(1.0)


def _clip_cpu_vector_fmin(value: np.float32) -> np.float32:
    # FMIN propagates NaNs.  NaN payload/sign are deliberately not normative.
    if np.isnan(value):
        return f32(np.nan)
    return value if value < f32(1.0) else f32(1.0)


def reference_value(
    value: float,
    gain: float,
    pivot: float,
    mode: int,
    clip: int | bool,
) -> np.float32:
    """Compute one consumer output in original binary32 instruction order."""
    if mode not in (SCALAR_MODE, CPU_VECTOR_MODE):
        raise ValueError("mode must be scalar (0) or cpu_vector (1)")
    if clip not in (0, 1, False, True):
        raise ValueError("clip must be 0 or 1")

    terms = prepare_pivot(pivot)
    p = f32(value)
    g = f32(gain)
    curve = fma32(p, terms.reciprocal, terms.bias)
    scaled = mul32(curve, g)
    result = fma32(scaled, terms.factor, terms.offset)
    if clip:
        if mode == SCALAR_MODE:
            return _clip_scalar_fminnm(result)
        return _clip_cpu_vector_fmin(result)
    return result


def reference_rows(
    inputs: Sequence[np.ndarray],
    outer: int,
    inner: int,
    value_stride: int,
    gain_stride: int,
    pivot: float,
    mode: int,
    clip: int | bool,
    *,
    output_count: int | None = None,
    sentinel: float = -731.625,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three padded output views; inactive row padding stays sentinel."""
    if len(inputs) != 4:
        raise ValueError("expected p0, p1, p2 and gain inputs")
    channels = tuple(np.asarray(item, dtype=np.float32).reshape(-1) for item in inputs)
    count = outer * value_stride if output_count is None else int(output_count)
    outputs = tuple(np.full(count, f32(sentinel), dtype=np.float32) for _ in range(3))
    for row in range(outer):
        for col in range(inner):
            value_offset = row * value_stride + col
            gain_offset = row * gain_stride + col
            gain = channels[3][gain_offset]
            for channel in range(3):
                outputs[channel][value_offset] = reference_value(
                    channels[channel][value_offset], gain, pivot, mode, clip
                )
    return outputs


def bits32(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32)


def assert_float32_results(actual: np.ndarray, expected: np.ndarray) -> None:
    """Compare every non-NaN bit exactly and compare NaNs by classification."""
    actual_values = np.ascontiguousarray(np.asarray(actual, dtype=np.float32))
    expected_values = np.ascontiguousarray(np.asarray(expected, dtype=np.float32))
    if actual_values.shape != expected_values.shape:
        raise AssertionError(f"shape mismatch: {actual_values.shape} != {expected_values.shape}")
    actual_nan = np.isnan(actual_values)
    expected_nan = np.isnan(expected_values)
    if not np.array_equal(actual_nan, expected_nan):
        bad = np.flatnonzero(actual_nan.reshape(-1) != expected_nan.reshape(-1))[:8]
        raise AssertionError(f"NaN classification mismatch at flat indices {bad.tolist()}")
    finite_mask = ~expected_nan
    actual_bits = actual_values.view(np.uint32)
    expected_bits = expected_values.view(np.uint32)
    if not np.array_equal(actual_bits[finite_mask], expected_bits[finite_mask]):
        mismatches = np.flatnonzero(
            finite_mask.reshape(-1)
            & (actual_bits.reshape(-1) != expected_bits.reshape(-1))
        )[:8]
        raise AssertionError(f"binary32 mismatch at flat indices {mismatches.tolist()}")
