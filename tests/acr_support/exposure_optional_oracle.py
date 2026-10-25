"""Independent prepared assembly of 00922f20 using original saved ARM64 raw.

Never reads the optional implementation. Reuses independent scalar oracles,
not production functions. Prepared predicate flags are separate observations.
"""
from __future__ import annotations
import math
import numpy as np
import exposure_qcap_oracle as qcap
import exposure_black_prepare_oracle as black
import exposure_context_lut_oracle as lut

UPPER = qcap.bits64('3fefae147ae147ae')  # .99, exact MOV/MOVK literal


def _int32(value):
    value = float(value)
    if not math.isfinite(value) or not value.is_integer() or not -(2**31) <= value < 2**31:
        raise ValueError('int32 prepared control required')
    return int(value)


def corrected_z(blacks, pg, active):
    z = min(float(blacks)*-.01, 1.)
    z = -1. if z < -1. else z
    if active:
        coefficient = .25 if z <= 0. else (1.-z)*.25
        pgscaled = float(pg)*-.01
        product = coefficient*pgscaled
        z = z+product
        z = min(z, 1.)
        z = -1. if z < -1. else z
    return z


def prepare(controls):
    if len(controls) != 18 or not all(math.isfinite(float(v)) for v in controls):
        raise ValueError('18 finite prepared controls required')
    controls = tuple(map(float, controls))
    blacks, pg = map(_int32, controls[:2])
    producer, mode, k = controls[2:5]
    ctx = controls[5:13]
    a8, *bases = controls[13:17]
    caller = controls[17]
    if producer not in (0., 1.) or mode not in (0., 1.) or caller not in (0., 1.):
        raise ValueError('prepared predicate flags must be exactly0/1')
    if not -120. <= a8 <= 120. or any(v < 0. for v in bases):
        raise ValueError('unsupported prepared channel gains')
    qv, qraw, q = qcap.prepare(ctx)
    z, b, d = map(float, black.prepare(blacks, pg, int(producer), int(mode), k, q))
    gain = math.exp2(a8)
    channel_values = []
    for base in bases:
        g = gain*base
        if not math.isfinite(g):
            raise ValueError('scaled channel gain outside adapter domain')
        addend = d+g
        # AArch64 FMSUB is addend minus product, with only final rounding.
        t = qcap.fma(-d, g, addend)
        # FMINNM recovers a NaN operand to the numeric upper bound. No
        # intermediate finite guard exists between FMSUB and that instruction.
        channel_values.append(UPPER if math.isnan(t) else min(t, UPPER))
    channel_active = any(t > 0. for t in channel_values)
    post_active = b > 0.
    cap, ar, br, ag, bg, ab, bb, h, zcaller = (0.,)*9
    if channel_active:
        # ctx[-2] is A658; do not use qv and do not combine exp2/divide.
        cap_input = math.exp2(ctx[6])
        cap_input = cap_input/qcap.SQRT2
        cap = qcap.response(ctx, cap_input)
        pairs = []
        for t in channel_values:
            a = t if t < cap else cap
            residual = t-a
            bv = max(residual, 0.)
            pairs.extend((a, bv))
        zcaller = corrected_z(blacks, pg, bool(caller))
        h = qcap.fma(zcaller, 2., 0.)
        h = min(h, .5)
        h = 0. if h < 0. else h
        ar, br, ag, bg, ab, bb = pairs
        for a, bv in ((ar, br), (ag, bg), (ab, bb)):
            lut.channel_state(a, bv, h)  # Validate the existing scalar factory domain.
    p = b*.25 if post_active else 0.
    if post_active and not 0. <= p <= 1.:
        raise ValueError('post factory p outside original constructor success')
    return np.array((qv, qraw, q, z, b, d, cap, float(channel_active), float(post_active),
                     ar, br, ag, bg, ab, bb, h, p, zcaller), dtype=np.float64)
