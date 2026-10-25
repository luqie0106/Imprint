"""Independent instruction sequence for 01300600, including accumulator drift."""
import ctypes
import ctypes.util
import math
import numpy as np

libm = ctypes.CDLL(ctypes.util.find_library('m') or None)
fma = libm.fma
fma.argtypes = [ctypes.c_double] * 3
fma.restype = ctypes.c_double


def build(table):
    values = np.asarray(table, dtype=np.float32)
    if values.shape != (4098,) or not np.isfinite(values).all():
        raise ValueError('bad finite table')
    if values.view(np.uint32)[4096] != values.view(np.uint32)[4097]:
        raise ValueError('bad guard')
    left, right = float(values[0]), float(values[1])
    step = float(4096) / 65535.
    base, slope = fma(left, 65535., .5), (right-left)*65535.
    index, fraction = 1, 0.
    out = np.empty(65536, dtype=np.uint16)
    for i in range(65536):
        result = fma(slope, fraction, base)
        if not math.isfinite(result) or not -2147483648. <= result < 2147483648.:
            raise ValueError('unrepresentable signed conversion')
        out[i] = int(result) & 65535
        fraction = step + fraction
        if fraction > 1.:
            index += 1
            fraction = fraction + -1.
            value = float(values[index])
            base = fma(right, 65535., .5)
            slope = (value-right)*65535.
            right = value
    return out
