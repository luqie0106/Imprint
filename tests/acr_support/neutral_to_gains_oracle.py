"""Independent binary64 reference for neutral-to-gain conversion.

The normalization intentionally rounds the reciprocal, each product, and each
final division as separate IEEE-754 binary64 operations.  Do not replace the
sequence with the algebraically equivalent original-minimum/input quotient.
"""

from fractions import Fraction

import numpy as np


F64 = np.float64
DEFAULT_NEUTRAL = np.asarray(
    [F64(0.4), F64(1.0), F64(0.6)],
    dtype=np.float64,
)
MIN_NORMALIZED = F64(0.125)


def _normalize(neutral):
    maximum = F64(np.max(neutral))
    reciprocal = F64(np.divide(F64(1.0), maximum))
    if not np.isfinite(reciprocal):
        raise ValueError("non-finite reciprocal")

    normalized = np.empty(3, dtype=np.float64)
    for i in range(3):
        normalized[i] = F64(np.multiply(F64(neutral[i]), reciprocal))
    if not np.all(np.isfinite(normalized)):
        raise ValueError("non-finite normalized neutral")

    minimum = F64(np.min(normalized))
    if minimum < MIN_NORMALIZED:
        return None

    result = np.empty(3, dtype=np.float64)
    for i in range(3):
        result[i] = F64(np.divide(minimum, F64(normalized[i])))
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite gain")
    return result


def reference(neutral=None, count=None):
    """Return gains for a three-value input, or for the default neutral.

    A non-three count and a finite three-value neutral with a nonpositive
    minimum both select the default.  Nonfinite three-value inputs and
    nonfinite arithmetic are rejected with ``ValueError``.
    """
    if count is None:
        count = 0 if neutral is None else np.asarray(neutral).size

    if count != 3:
        values = DEFAULT_NEUTRAL.copy()
    else:
        if neutral is None:
            raise ValueError("three-value input is null")
        values = np.asarray(neutral, dtype=np.float64)
        if values.size != 3:
            raise ValueError("count does not match input size")
        values = values.reshape(3).copy()
        if not np.all(np.isfinite(values)):
            raise ValueError("non-finite neutral")
        if F64(np.min(values)) <= F64(0.0):
            values = DEFAULT_NEUTRAL.copy()

    result = _normalize(values)
    if result is None:
        result = _normalize(DEFAULT_NEUTRAL)
    if result is None:  # The fixed default is required to stay above threshold.
        raise AssertionError("default neutral unexpectedly fell below threshold")
    return result


def _bits32(value):
    return int(np.asarray([value], dtype=np.float32).view(np.uint32)[0])


def _from_bits32(bits):
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def _fraction32(value):
    """Return the exact rational value represented by one finite float32."""
    bits = _bits32(value)
    sign = -1 if bits >> 31 else 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent == 0xFF:
        raise ValueError("non-finite float32")
    if exponent == 0:
        significand, power = mantissa, -149
    else:
        significand, power = (1 << 23) | mantissa, exponent - 127 - 23
    result = Fraction(sign * significand, 1)
    return result * (1 << power) if power >= 0 else result / (1 << -power)


def _round_ratio_even(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    twice_remainder = remainder * 2
    if twice_remainder > denominator or (
        twice_remainder == denominator and quotient & 1
    ):
        quotient += 1
    return quotient


def _scale_power_of_two(value, power):
    if power >= 0:
        return Fraction(value.numerator << power, value.denominator)
    return Fraction(value.numerator, value.denominator << -power)


def _round_fraction32(value, negative_zero=False):
    """Correctly round an exact rational to IEEE-754 binary32, ties to even."""
    negative = value < 0 or (value == 0 and negative_zero)
    magnitude = -value if value < 0 else value
    sign_bit = 0x80000000 if negative else 0
    if magnitude == 0:
        return _from_bits32(sign_bit)

    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < _scale_power_of_two(Fraction(1), exponent):
        exponent -= 1

    if exponent >= -126:
        scaled = _scale_power_of_two(magnitude, 23 - exponent)
        significand = _round_ratio_even(scaled.numerator, scaled.denominator)
        if significand >= 1 << 24:
            significand >>= 1
            exponent += 1
        if exponent > 127:
            return _from_bits32(sign_bit | 0x7F800000)
        exponent_bits = exponent + 127
        mantissa_bits = significand - (1 << 23)
    else:
        scaled = _scale_power_of_two(magnitude, 149)
        significand = _round_ratio_even(scaled.numerator, scaled.denominator)
        if significand >= 1 << 23:
            exponent_bits, mantissa_bits = 1, 0
        else:
            exponent_bits, mantissa_bits = 0, significand

    return _from_bits32(sign_bit | (exponent_bits << 23) | mantissa_bits)


def _f32_from_binary64(value):
    return _round_fraction32(Fraction.from_float(float(value)))


def _f32_neg(value):
    return _from_bits32(_bits32(value) ^ 0x80000000)


def _f32_add(left, right):
    exact = _fraction32(left) + _fraction32(right)
    negative_zero = False
    if exact == 0 and _fraction32(left) == 0 and _fraction32(right) == 0:
        left_negative = bool(_bits32(left) >> 31)
        right_negative = bool(_bits32(right) >> 31)
        negative_zero = left_negative and right_negative
    return _round_fraction32(exact, negative_zero)


def _f32_div(numerator, denominator):
    exact = _fraction32(numerator) / _fraction32(denominator)
    negative_zero = bool((_bits32(numerator) ^ _bits32(denominator)) >> 31)
    return _round_fraction32(exact, negative_zero)


def _f32_fma(left, right, addend):
    product = _fraction32(left) * _fraction32(right)
    addend_exact = _fraction32(addend)
    exact = product + addend_exact
    negative_zero = False
    if exact == 0 and product == 0 and addend_exact == 0:
        product_negative = bool((_bits32(left) ^ _bits32(right)) >> 31)
        addend_negative = bool(_bits32(addend) >> 31)
        negative_zero = product_negative and addend_negative
    return _round_fraction32(exact, negative_zero)


SOURCE_FLOOR_RECIPROCAL = float.fromhex("0x1.0001000100010p-16")


def source_floor_reference(gains, source_floor, inverse=False):
    """Prepare gain/bias values with the independent source-floor sequence.

    The source floor is converted to binary64, multiplied by the recovered
    binary64 reciprocal, then rounded to float32 before the existing gain
    preparation arithmetic.  Each float32 operation, including the fused
    multiply-adds, is rounded independently to match the native instruction
    sequence without calling the native implementation.
    """
    gains = np.asarray(gains, dtype=np.float64).reshape(-1)
    if gains.size != 3:
        raise ValueError("exactly three gains are required")
    if not 0 <= int(source_floor) <= 65535:
        raise ValueError("source floor is outside uint16 range")

    prepared_gains = np.empty(3, dtype=np.float32)
    clip = False
    for channel, input_gain in enumerate(gains):
        input_gain = float(input_gain)
        if not np.isfinite(input_gain) or input_gain <= 0.0 or input_gain >= 128.0:
            raise ValueError("invalid input gain")
        adjusted = 1.0 / input_gain if inverse else input_gain
        if not np.isfinite(adjusted) or adjusted <= 0.0 or adjusted > np.finfo(np.float32).max:
            raise ValueError("invalid adjusted gain")
        prepared = _f32_from_binary64(adjusted)
        if not np.isfinite(prepared) or prepared <= 0.0:
            raise ValueError("adjusted gain is not representable as float32")
        prepared_gains[channel] = prepared
        clip = clip or adjusted > 1.0

    pivot64 = float(int(source_floor)) * SOURCE_FLOOR_RECIPROCAL
    pivot = _f32_from_binary64(pivot64)
    if not np.isfinite(pivot) or pivot < 0.0 or pivot >= 1.0:
        raise ValueError("source floor produces an invalid pivot")

    den = _f32_add(pivot, np.float32(-1.0))
    t = _f32_div(np.float32(-1.0), den)
    neg = _f32_div(_f32_fma(_f32_neg(t), pivot, np.float32(0.0)), t)
    pos = _f32_div(_f32_fma(t, pivot, np.float32(-0.0)), t)
    biases = np.asarray(
        [_f32_fma(gain, neg, pos) for gain in prepared_gains], dtype=np.float32
    )
    return np.concatenate((prepared_gains, biases)), int(clip)
