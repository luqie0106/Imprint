"""Independent binary64 transcription of saved ARM64 optional Exposure curves.

Authority: 2026-10-22-optional-curve-shapes-offline raw 00926ba4,
00926c24, 00dd33bc, 0093a348, 00926758; initialization 00923194.
No native implementation is imported. Host libm results are not Adobe parity.
"""
from __future__ import annotations
import ctypes
import ctypes.util
import math
import struct
import numpy as np

_libm = ctypes.CDLL(ctypes.util.find_library("m") or None)
fma = _libm.fma
fma.argtypes = [ctypes.c_double] * 3
fma.restype = ctypes.c_double
# The saved imported math calls are evaluated independently through host libm.
for _name, _arity in (("pow", 2), ("exp2", 1), ("sqrt", 1)):
    _fn = getattr(_libm, _name)
    _fn.argtypes = [ctypes.c_double] * _arity
    _fn.restype = ctypes.c_double
pow64, exp2, sqrt = _libm.pow, _libm.exp2, _libm.sqrt


def constant(bits):
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _validate(value, upper, inclusive=True):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value > upper if inclusive else value >= upper):
        raise ValueError("prepared curve input is outside its finite domain")
    return value


def clip(value):
    value = min(value, 1.0)
    return 0.0 if value < 0.0 else value


def channel_state(a, b, h):
    a, b, h = _validate(a, 1, False), _validate(b, 1, False), _validate(h, .5)
    ar = -1.0 / (a + -1.0)
    r = -1.0 / (b + -1.0)
    x0 = b * h
    x1 = fma(1.0 - b, 0.0, b)
    y1 = fma(r, x1 - b, 0.0)
    x2 = fma(1.0 - b, .5, b)
    y2 = fma(r, x2 - b, 0.0)
    return a, ar, b, r, x0, x1, y1, x2, y2


def quadratic(A, B, C):
    if A == 0:
        return [] if B == 0 else [(-C) / B]
    z = A * -4.0
    z = z * C
    D = fma(B, B, z)
    if D < 0:
        return []
    s = 1.0 if B >= 0 else -1.0
    q = fma(s, sqrt(D), B)
    q = q * -.5
    # IEEE division is required for degenerate roots, rather than Python's trap.
    with np.errstate(divide="ignore", invalid="ignore"):
        return [float(np.float64(q) / A), float(np.float64(C) / q)]


def channel(x, a, b, h):
    x = _validate(x, 1)
    a, ar, b, r, x0, x1, y1, x2, y2 = channel_state(a, b, h)
    u = clip(fma(ar, x - a, 0.0))
    if x0 >= u:
        return 0.0
    if x2 <= u:
        return fma(r, u - b, 0.0)
    A = x2 + fma(x1, -2.0, x0)
    B = fma(x0, -2.0, x1 + x1)
    C = x0 - u
    t = next((v for v in quadratic(A, B, C) if 0.0 <= v <= 1.0), 0.0)
    v = 1.0 - t
    w = 0.0 * v
    z = t * (y1 + y1)
    z = v * z
    y = fma(w, v, z)
    w = t * y2
    return fma(w, t, y)


def encode(x):
    if x <= constant(0x3f69a5c37387b719):
        return x * constant(0x4029d70a3d70a3d7)
    return fma(pow64(x, constant(0x3fdaaaaaaaaaaaab)),
               constant(0x3ff0e147ae147ae1), constant(0xbfac28f5c28f5c29))


def decode(s):
    if s <= constant(0x3fa4b5daa07d0fa8):
        return s * constant(0x3fb3d0722149b580)
    return pow64((s + constant(0x3fac28f5c28f5c29)) * constant(0x3fee54edcd0aeb60),
                 constant(0x4003333333333333))


def post(x, p):
    x, p = _validate(x, 1), _validate(p, 1)
    A = clip(p) * 4.0
    K = fma(clip(p), -2.0, 4.0)
    K = K if K < 3.0 else 3.0
    s = encode(x)
    z = decode(s)
    u = exp2(-(K + A))
    v = (z + u) / (u + 1.0)
    u = exp2(-K)
    w = (z + u) / (u + 1.0)
    w = 1.0 * w
    r = w / v
    v = r * x
    w = x - v
    return fma(x, w, v)


def _table(evaluator, controls):
    result = np.empty(4098, dtype=np.float32)
    for i in range(4097):
        result[i] = evaluator((1.0 / 4096.0) * float(i), *controls)
    result[4097] = result[4096]
    return result


def channel_lut(a, b, h):
    channel_state(a, b, h)
    return _table(channel, (a, b, h))


def post_lut(p):
    _validate(p, 1)
    return _table(post, (p,))
