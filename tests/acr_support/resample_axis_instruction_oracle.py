"""Independent instruction-level reference for ordinary ACR axis preparation.

The reference was derived from the pinned ARM64 instruction captures for
FUN_00d34190, FUN_00d357b0, and FUN_00d36a50. It does not inspect or import the
native implementation, and tests do not read the evidence files at runtime.
"""

from __future__ import annotations

import math
import operator


REFERENCE_ID = "acr-00d36a50-ordinary-resample-axis-arm64-q32-v1"
SOURCE_SLICE_SHA256 = "09978d95dfe76b18495fea9550b051a89f4598ab0fd4734d69d22d330496f40a"
RAW_CAPTURE_SHA256 = (
    "df698c3d92fd69f6b0f59c8004b66f3c1f0d6415ba6c1b2659a4529a3621dd4c",
    "a62c29c2be74bf94bc41ea8b758282ce0c13073b22caf887b8aabc6dbd3fae21",
    "e164ba12b22e1e7ce3f7490e4f488c4c9959fb93e67659029ba8e565f694087e",
)

STATUS_OK = 0
STATUS_INVALID = 1
STATUS_UNSUPPORTED_SKEW = 2

KIND_RESAMPLE = 0
KIND_IDENTITY = 1
KIND_INTEGER_TRANSLATION = 2

_Q32 = float(1 << 32)
_I32_MASK = (1 << 32) - 1
_I64_MASK = (1 << 64) - 1
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


def _signed_word(value: object, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a signed 32-bit integer") from exc
    if not -(1 << 31) <= result < (1 << 31):
        raise ValueError(f"{name} must be a signed 32-bit integer")
    return result


def wrap32(value: int) -> int:
    """Return the low 32 bits of an ARM64 W-register arithmetic result."""
    return int(value) & _I32_MASK


def wrap64(value: int) -> int:
    """Return the low 64 bits of an ARM64 X-register arithmetic result."""
    return int(value) & _I64_MASK


def signed32(value: int) -> int:
    """Interpret the low 32 bits as a two's-complement signed word."""
    word = int(value) & _I32_MASK
    return word - (1 << 32) if word & (1 << 31) else word


def _fcvtzs_i64(value: float) -> int:
    """Model AArch64 FCVTZS to X: truncate, NaN -> 0, saturate on overflow."""
    if math.isnan(value):
        return 0
    if value >= float(1 << 63):
        return _I64_MAX
    if value < float(-(1 << 63)):
        return _I64_MIN
    return math.trunc(value)


def quantize_q32(value: object) -> int:
    """Apply the factory's binary64 multiply, sign-selected bias, and FCVTZS.

    The multiplication and bias addition are separate binary64 operations.
    NaN compares false for the nonnegative branch, then FCVTZS returns zero.
    """
    try:
        source = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("value must be convertible to binary64") from exc
    scaled = source * _Q32
    bias = 0.5 if scaled >= 0.0 else -0.5
    return _fcvtzs_i64(scaled + bias)


def prepare(translation: object, along_scale: object, cross_scale: object,
            bound_start: object, bound_end: object,
            half_taps: object) -> tuple[int, tuple[int, ...] | None]:
    """Return ``(status, six output words)`` for the recovered axis factory.

    ``half_taps`` is the constructor's signed ``N``; the ordinary generated
    resampler uses ``2*N`` taps. Its value is immaterial for identity and exact
    integer translations, and must be positive for a general resample axis.
    Output words represent ``kind, origin, step, lower, upper, offset``.
    """
    start = _signed_word(bound_start, "bound_start")
    end = _signed_word(bound_end, "bound_end")
    n = _signed_word(half_taps, "half_taps")

    t = quantize_q32(translation)
    s = quantize_q32(along_scale)
    x = quantize_q32(cross_scale)
    if x != 0:
        return STATUS_UNSUPPORTED_SKEW, None

    if s == (1 << 32) and t == 0:
        return STATUS_OK, (KIND_IDENTITY, 0, 0, 0, 0, 0)

    if s == (1 << 32) and (t & _I32_MASK) == 0:
        offset = wrap32(-signed32(t >> 32))
        return STATUS_OK, (KIND_INTEGER_TRANSLATION, 0, 0, 0, 0, offset)

    if n <= 0:
        return STATUS_INVALID, None

    one_minus_n = wrap32(1 - n)
    origin = wrap64(
        t + (s >> 1) + (x >> 1) + (one_minus_n << 32) - 0x7F000000
    )
    lower = wrap32(start + one_minus_n) << 32
    upper = wrap32(end - n) << 32
    return STATUS_OK, (KIND_RESAMPLE, origin, s & _I64_MASK,
                       lower, upper, 0)
