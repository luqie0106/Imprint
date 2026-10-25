"""Independent binary64 transcription of ARM64 009227d0/00922c4c.

Derived from the offline raw disassembly (00926a10, dd3204, dd32b8,
dd33bc included), without consulting the implementation under test. Host libm
is used only for fused arithmetic and imported atan/tan/sqrt; this is not Adobe
runtime or cross-platform libm bit parity evidence.
"""
from __future__ import annotations
import ctypes as C
import math
import numpy as np

_LIBM = C.CDLL(None)
for _name, _arity in (("fma", 3), ("atan", 1), ("tan", 1), ("sqrt", 1), ("fmin", 2)):
    _fn = getattr(_LIBM, _name)
    _fn.argtypes = [C.c_double] * _arity
    _fn.restype = C.c_double
fma, atan, tan, sqrt = (_LIBM.fma, _LIBM.atan, _LIBM.tan, _LIBM.sqrt)
EPS = float.fromhex('0x1.5798ee2308c3ap-27')
UPPER = float.fromhex('0x1.ffffffaa19c47p-1')

class InvalidCurve(ValueError):
    pass

class NonfiniteResult(ValueError):
    pass

def _finite(*values):
    if not all(math.isfinite(v) for v in values):
        raise NonfiniteResult('nonfinite construction or result')

def _div(a, b):
    with np.errstate(all='ignore'):
        return float(np.float64(a) / np.float64(b))

def _roots(a, b, c):
    # Preserve root order, zero denominators, and discarded NaN/Inf roots.
    if a == 0.0:
        return [] if b == 0.0 else [_div(-c, b)]
    p = a * -4.0
    p = p * c
    delta = fma(b, b, p)
    if delta < 0.0:
        return []
    sign = 1.0 if b >= 0.0 else -1.0
    root = sqrt(delta)
    qr = fma(sign, root, b)
    qr = qr * -0.5
    r2 = _div(c, qr)
    r1 = _div(qr, a)
    return [r1, r2]

def _q(points, z):
    x0, y0, x1, y1, x2, y2 = points
    _finite(*points)
    if not (x0 < x1 < x2 or x0 > x1 > x2):
        raise InvalidCurve('original Q points are not strictly monotonic')
    lo, hi, yl, yh = (x0, x2, y0, y2) if x0 < x2 else (x2, x0, y2, y0)
    if lo >= z:
        return yl
    if hi <= z:
        return yh
    a = fma(x1, -2.0, x0)
    a = a + x2
    b = x1 + x1
    b = fma(x0, -2.0, b)
    c = x0 - z
    s = next((r for r in _roots(a, b, c) if 0.0 <= r <= 1.0), 0.0)
    v = 1.0 - s
    w = y0 * v
    j = y1 + y1
    j = s * j
    j = v * j
    r = fma(w, v, j)
    w = s * y2
    return fma(w, s, r)

def _factory(t, k, m0, q, m1):
    delta = m0 - m1
    ad = -delta if delta < 0.0 else delta
    if ad < EPS:
        # Literal register order, intentionally unusual.
        return (t, 1.0, (t + 1.0) * 0.5, (k + q) * 0.5, k, q)
    n = fma(m0, t, -m1)
    n = n - k
    n = n + q
    u = _div(n, delta)
    y = fma(m0, u, k)  # No subtraction of t in the target.
    return (t, k, u, y, 1.0, q)

def response(z, t0, t1, k, q) -> float:
    """Frozen R(z,t0,t1,k,q), finite inputs and t0<t1<1 only."""
    z, t0, t1, k, q = map(float, (z, t0, t1, k, q))
    if not all(map(math.isfinite, (z, t0, t1, k, q))) or not t0 < t1 < 1.0:
        raise ValueError('invalid response input')
    t = t1
    if z <= t:
        m = _div(0.0 - k, t0 - t)
        result = fma(m, z - t0, 0.0)
        _finite(result)
        return result
    m0 = _div(k, t - t0)
    dq = q - k
    m1 = _div(dq, 1.0 - t)
    if m1 == m0:
        points = _factory(t, k, m0, q, m1)
    elif m1 < m0:
        points = _factory(t, k, m0, q, 0.0)
    else:
        a0 = atan(m0)
        a1 = atan(m1)
        h = a1 - a0
        h = a1 + h
        w = tan(h)
        w = w * 1.5
        n = fma(t, m0, dq)
        n = n - w
        n = n + EPS
        d = m0 - w
        d = d + EPS
        u = _div(n, d)
        lo = t + EPS
        u = _LIBM.fmin(u, UPPER)
        u = lo if u < lo else u
        y = fma(m0, u - t, k)
        points = (t, k, u, y, 1.0, q)
    result = _q(points, z)
    _finite(result)
    return result

def prepare(blacks, pg, active, mode, k, q) -> np.ndarray:
    """Frozen resolved-control producer; active/mode are supplied predicates."""
    for value in (blacks, pg):
        if not isinstance(value, (int, np.integer)) or not -(2**31) <= value < 2**31:
            raise ValueError('control must be int32')
    if active not in (0, 1) or mode not in (0, 1):
        raise ValueError('resolved flags must be 0 or 1')
    k, q = float(k), float(q)
    if not math.isfinite(k) or not math.isfinite(q) or not .0625 <= q <= .9:
        raise ValueError('invalid prepared k/q')
    z = min(float(blacks) * -.01, 1.0)
    z = -1.0 if z < -1.0 else z
    if active:
        amount = .25 if z <= 0.0 else (1.0 - z) * .25
        gain = float(pg) * -.01
        amount = amount * gain
        corrected = z + amount
        corrected = min(corrected, 1.0)
        z = -1.0 if corrected < -1.0 else corrected
    if mode:
        b = -z if z < 0.0 else 0.0
        d = 0.0 if z <= 0.0 else response(z, 0.0, .25, k, q)
    else:
        b = fma(z + 1.0, -4.0 / 3.0, 1.0) if z < -.25 else 0.0
        d = 0.0 if z <= -.25 else response(z, -.25, 0.0, k, q)
    _finite(z, b, d)
    return np.array((z, b, d), dtype=np.float64)
