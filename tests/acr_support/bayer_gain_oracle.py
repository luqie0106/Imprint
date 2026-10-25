"""Independent binary32 instruction model of the recovered gain consumer.

Controls follow 007ef334..364 / 007ef5b0..604; RGB selection follows
007ef304..324. This module does not load native operator code.
"""
import ctypes
import numpy as np

F = np.float32
_fmaf = ctypes.CDLL(None).fmaf
_fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_fmaf.restype = ctypes.c_float


def fused(a, b, c):
    return F(_fmaf(float(a), float(b), float(c)))


def prepare(gains, pivot, inverse=False):
    doubles = [1.0 / float(g) if inverse else float(g) for g in gains]
    gs = np.asarray(doubles, dtype=np.float32)
    b = F(pivot)
    t = F(F(-1) / F(b + F(-1)))
    # Preserve signs as distributed by FMSUB and FNMADD, not negated-FMA.
    minus = F(fused(-t, b, F(0.0)) / t)
    plus = F(fused(t, b, F(-0.0)) / t)
    biases = np.asarray([fused(g, minus, plus) for g in gs], dtype=np.float32)
    return np.concatenate((gs, biases)), max(doubles) > 1.0


def pixel(sample, gain, bias, clip):
    result = fused(sample, gain, bias)
    if clip:
        result = np.fmin(result, F(1))
        if result < F(0):
            result = F(0)
    return result


def rgb(rows, width, values, clip):
    result = np.array(rows, dtype=np.float32, copy=True)
    gains, bias = values[:3], values[3:]
    channels = (0 if gains[0] != F(1) else 1,
                1 if gains[0] != F(1) and gains[1] != F(1) else 2)
    for channel in channels:
        for x in range(width):
            result[channel, x] = pixel(result[channel, x], gains[channel], bias[channel], clip)
    return result


def cfa(row, width, values, clip, row_parity, column_parity):
    result = np.array(row, dtype=np.float32, copy=True)
    for x in range(width):
        column = (column_parity + x) & 1
        channel = 0 if row_parity == column == 0 else 2 if row_parity == column == 1 else 1
        result[x] = pixel(result[x], values[channel], values[3 + channel], clip)
    return result
