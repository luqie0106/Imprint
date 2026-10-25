"""Independent, versioned reference for ordinary float resample rows.

This is a test oracle copied into the maintained test support tree from the
saved ARM64 coefficient specification. It imports neither ignored research
scripts nor the C++ candidate. The model preserves the recovered binary64 FMA
schedule, float stores, ascending double row sum, float normalizer, and float
normalization multiply. Zero padding is the local oracle contract; the saved
Adobe evidence does not establish padding contents.
"""

from __future__ import annotations

from fractions import Fraction
import math
import struct


ORACLE_VERSION = "acr-resample-coefficients-float32-128phase-q-v1-2026-10-23"
REFERENCE_ID = "acr-012e3a20-float-coefficient-rows-arm64-fma32-v1"
PHASES = 128
STRIDE_QUANTUM = 8
MAX_TAPS = 4096
ONE_SIXTH = float.fromhex("0x1.5555555555555p-3")
TAP_COUNT_BIAS = struct.unpack(">d", bytes.fromhex("3fefff2e48e8a71e"))[0]
PHASE_STEP = float.fromhex("0x1p-7")


def f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def f32_bits(value: float) -> int:
    return int.from_bytes(struct.pack(">f", value), "big")


def f64_bits(value: float) -> int:
    return int.from_bytes(struct.pack(">d", value), "big")


def exact_fma(a: float, b: float, c: float) -> float:
    """Round the exact rational ``a*b+c`` once to binary64."""
    exact = Fraction.from_float(a) * Fraction.from_float(b) + Fraction.from_float(c)
    return float(exact)


def filter_parameters(q: float) -> tuple[float, float]:
    if q <= 1.0:
        return 0.0, 0.5
    offset = q + -2.0
    scaled_offset = offset * ONE_SIXTH
    c = scaled_offset if scaled_offset < 1.0 else 1.0
    a = 0.0 if c < 0.0 else c * 0.5
    b = (1.0 - a) * 0.5
    return a, b


def tap_count_for_q(q: float) -> tuple[float, int, int]:
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("q must be finite and positive")
    scale = q if q < 1.0 else 1.0
    radius_value = (2.0 / scale) + TAP_COUNT_BIAS
    if not math.isfinite(radius_value) or not 2.0 <= radius_value < 2049.0:
        raise ValueError("q is outside the recovered tap-count range")
    radius = int(radius_value)
    taps = radius * 2
    if taps > MAX_TAPS:
        raise ValueError("q exceeds the local oracle resource limit")
    return scale, radius, taps


def reference_weight(a: float, b: float, x: float) -> float:
    u = -x if x < 0.0 else x
    if u >= 2.0:
        return 0.0
    if u <= 1.0:
        t = b * 6.0
        c0 = exact_fma(a, -2.0, 6.0)
        c1 = exact_fma(a, 12.0, t)
        c1 += -18.0
        c2 = exact_fma(a, -9.0, -t)
        c2 += 12.0
        u2 = u * u
        value = exact_fma(u, c2, c1)
        value = exact_fma(u2, value, c0)
        return value * ONE_SIXTH

    c0 = exact_fma(a, 8.0, b * 24.0)
    c1 = exact_fma(a, -12.0, b * -48.0)
    c2 = exact_fma(a, 6.0, b * 30.0)
    c3 = b * -6.0 - a
    value = exact_fma(u, c3, c2)
    value = exact_fma(u, value, c1)
    value = exact_fma(u, value, c0)
    return value * ONE_SIXTH


def reference_table(q: float) -> dict[str, object]:
    """Build all 128 float32 rows, including local zero-filled padding."""
    scale, radius, taps = tap_count_for_q(q)
    stride = ((taps + STRIDE_QUANTUM - 1) // STRIDE_QUANTUM) * STRIDE_QUANTUM
    a, b = filter_parameters(q)
    values = [0.0] * (PHASES * stride)
    for phase_index in range(PHASES):
        phase = float(phase_index) * PHASE_STEP
        row = phase_index * stride
        row_sum = 0.0
        for tap in range(taps):
            integer_offset = tap - radius + 1
            x = scale * (float(integer_offset) - phase)
            stored = f32(reference_weight(a, b, x))
            values[row + tap] = stored
            row_sum += float(stored)
        if not math.isfinite(row_sum) or row_sum <= 0.0:
            raise ArithmeticError("coefficient row has a nonpositive sum")
        normalizer = f32(1.0 / row_sum)
        for tap in range(taps):
            values[row + tap] = f32(values[row + tap] * normalizer)
    return {
        "q": q,
        "scale": scale,
        "parameter_a": a,
        "parameter_b": b,
        "radius": radius,
        "tap_count": taps,
        "stride": stride,
        "values": values,
    }
