"""Independent binary64 model of ARM64 009226b4 / 009227d0 q path.

Authority: 2026-10-22-exposure-qcap-offline raw instructions. This module
never imports native_renderer or reads its implementation. ctypes calls only
platform libm scalar primitives; there is no Adobe runtime claim.
"""
from __future__ import annotations

import ctypes
import math
import struct

_libm = ctypes.CDLL(None)
_libm.fma.argtypes = [ctypes.c_double] * 3
_libm.fma.restype = ctypes.c_double
fma = _libm.fma


def bits64(bits):
    return struct.unpack('>d', int(bits, 16).to_bytes(8, 'big'))[0]


ENC = bits64('3fdd1745d1745d17')
DEC = bits64('400199999999999a')
BASE = bits64('3ff5ecf52776a115')
WHITEBASE = bits64('3ffbb67ae8584caa')
SQRT2 = bits64('3ff6a09e667f3bcd')
LN2 = bits64('3fe62e42fefa39ef')
STOP = bits64('3ddb7cdfd9d7bdbb')
QMAX = bits64('3feccccccccccccd')


def clip01(x):
    # Prepared supported domain is finite and canonicalizes zero to +0.
    return max(0.0, min(x, 1.0)) + 0.0


def _segment(s):
    g = math.pow(BASE, s)
    h = max(.1, 1.0 / fma(g + (-1.0), 12.0, 1.0))
    k = .5 / g
    return g, h, k, 1.0-k


def _forward(u, seg):
    g, h, k, w = seg
    if u < k:
        return g*u
    t = (u-k)/w
    v = 1.0-t
    a = t+(2.0-v)
    b = w*g
    b = t*b
    a = fma(a, .5, b)
    v2 = v*v
    c = (2.0-t)+v
    b = w*h
    b = v*b
    c = c-b
    t2 = t*t
    c = t2*c
    return fma(a, v2, c)


def _inverse(target, seg):
    xold, yold = 0.0, _forward(0.0, seg)
    xcur, ycur = 1.0, _forward(1.0, seg)
    for _ in range(30):
        dy = ycur-yold
        if abs(dy) < STOP:
            return xcur
        delta = target-ycur
        dx = xcur-xold
        numerator = delta*dx
        quotient = numerator/dy
        proposed = xcur+quotient
        xnew = clip01(proposed)
        ynew = _forward(xnew, seg)
        xold, yold = xcur, ycur
        xcur, ycur = xnew, ynew
    return xcur


def wrapper(x, ev):
    u = math.pow(x, ENC)
    a = abs(ev)
    strengths = [min(a, 2.0), max(0.0, min(a+(-2.0), 2.0)),
                 max(0.0, min(a+(-4.0), 2.0)), max(0.0, min(a+(-6.0), 2.0))]
    segments = [(s, _segment(s)) for s in strengths]
    if ev < 0:
        for s, seg in reversed(segments):
            if s != 0:
                u = _inverse(u, seg)
    else:
        for s, seg in segments:
            if s != 0:
                u = _forward(u, seg)
    return math.pow(u, DEC) + 0.0


def _validate(ctx):
    if len(ctx) != 8 or not all(math.isfinite(float(x)) for x in ctx):
        raise ValueError('eight finite prepared doubles required')
    e98, a0, b0, w80, *_ = ctx
    if not (-120 <= e98 <= 120 and -120 <= a0 <= 120 and
            -2 <= b0 <= 2 and -2 <= w80 <= 2):
        raise ValueError('outside adapter supported domain')


def negative_clip(ctx):
    e98, a0, _, white, lo, hi, *_ = ctx
    r = hi-lo
    fraction = 0.0 if r <= 0 else 1.0 if r >= 5 else fma(r/5.0, 1.0, 0.0)
    upper = hi-fraction
    lower = fma(r, .75, lo)
    anchor = upper if upper > lower else lower
    input_gain = math.exp2(anchor)
    linear = math.exp2(e98)*input_gain
    response = clip01(wrapper(clip01(linear), a0))
    base = min(response, .5)
    half = white*.5
    exponent = math.pow(half if white > 0 else 0.0, SQRT2)
    if base == 0:
        if exponent == 0:
            raise ValueError('0 times negative infinity')
        return 0.0
    logbase = math.log(base)
    logbase = logbase/LN2
    loggain = fma(exponent, logbase, 0.0)
    return math.exp2(loggain)


def response(ctx, input_value):
    _validate(ctx)
    if not math.isfinite(input_value) or input_value < 0:
        raise ValueError('invalid response input')
    e98, a0, b0, *_ = ctx
    linear = math.exp2(e98)*float(input_value)
    y = wrapper(clip01(linear), a0)
    if b0 > 0:
        power = math.pow(WHITEBASE, b0)
        powered = math.pow(y, power)
        delta = powered-y
        y = fma(y, delta, y)
    elif b0 < 0:
        c = negative_clip(ctx)
        c2 = c*c
        if c <= 0 or not math.isfinite(c2):
            raise ValueError('negative Whites clip outside supported domain')
        a = y+1.0
        a = y/a
        b = (y/c2 if c2 != 0 else (math.nan if y == 0 else math.inf))
        b = b+1.0
        product = a*b
        y = 1.0 if math.isnan(product) else min(product, 1.0)
    if not math.isfinite(y):
        raise ValueError('nonfinite response')
    return y+0.0


def prepare(ctx):
    _validate(ctx)
    *_, a, high = ctx
    left = a+2.0
    right = high-a
    right = fma(right, .5, a)
    v = left if left < right else right
    if not math.isfinite(v) or not -120 <= v <= 120:
        raise ValueError('computed v outside supported domain')
    r = response(ctx, math.exp2(v))
    q = min(max(clip01(r), .0625), QMAX)
    return v, r, q
