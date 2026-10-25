"""Instruction-derived double reference for the no-matrix bounded branch."""
import ctypes
import math
import struct
import numpy as np

_fma = ctypes.CDLL(None).fma
_fma.argtypes = (ctypes.c_double,) * 3
_fma.restype = ctypes.c_double
_bits = (0x3FEFAE147AE147AE, 0x3F847AE147AE147B, 0x40126BB1BBB55514, 0x3FCBCB7B1526E510)
C99, C01, CLOG, CSCALE = (struct.unpack('<d', struct.pack('<Q', b))[0] for b in _bits)


def converted(p):
    z = _fma(p, C99, C01)
    if not math.isfinite(z) or z <= 0:
        raise ValueError('log domain')
    return (math.log(z) + CLOG) * CSCALE - 1.0


def reference(lower, upper, direct):
    if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
        return 2, None
    if lower <= 0 and upper <= 0:
        return 3, None
    if not direct:
        try:
            lower, upper = converted(lower), converted(upper)
        except ValueError:
            return 4, None
    # Explicit IEEE min with +0 includes a negative-zero input unchanged.
    negative = lambda x: x if x < 0 or (x == 0 and math.copysign(1, x) < 0) else 0.0
    positive = lambda x: x if x > 0 else 0.0
    with np.errstate(over='ignore'):
        result = np.asarray([negative(-(upper * .5)), positive(-(lower * .5)),
                             negative(lower * .5), positive(upper * .5)], dtype=np.float32)
    if not np.isfinite(result).all():
        return 4, None
    return 0, result
