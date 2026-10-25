"""Instruction-order reference for the Stage3 P2 composition, independent of C++."""
import ctypes
import ctypes.util
import math
import struct

libm = ctypes.CDLL(ctypes.util.find_library('m') or None)
fma = libm.fma
fma.argtypes = [ctypes.c_double] * 3
fma.restype = ctypes.c_double
fmin = libm.fmin
fmin.argtypes = [ctypes.c_double] * 2
fmin.restype = ctypes.c_double


def number(bits):
    return struct.unpack('d', struct.pack('Q', bits))[0]


C99 = number(0x3fefae147ae147ae)
C01 = number(0x3f847ae147ae147b)
CLOG = number(0x40126bb1bbb55514)
CSCALE = number(0x3fcbcb7b1526e510)
CINV99 = number(0x3ff0295fad40a57f)
EPSILON = number(0x3ddb7cdfd9d7bdbb)


def clamp(v):
    v = fmin(v, 1.)
    return 0. if v < 0. else v


def prepare(pivot, cubic):
    pivot = float(pivot)
    if not math.isfinite(pivot) or not 0 <= pivot < 1 or cubic not in (0, 1):
        raise ValueError('invalid prepared domain')
    return [pivot, .25 if pivot > 0 else 0., float(cubic)]


def cubic_forward(sample, pivot):
    z = fma(-1. / (pivot + -1.), sample - pivot, 0.)
    product = z * .9921875
    product = z * product
    product = z * product
    value = fma(z, .0078125, product)
    return fma(-(pivot + -1.), value, pivot)


def cubic_inverse(target, pivot):
    previous_x, current_x = 0., 1.
    previous_y, current_y = cubic_forward(0., pivot), cubic_forward(1., pivot)
    for _ in range(30):
        difference = current_y - previous_y
        absolute = -difference if difference < 0 else difference
        if absolute < EPSILON:
            return current_x
        numerator = (target - current_y) * (current_x - previous_x)
        candidate = clamp(current_x + numerator / difference)
        value = cubic_forward(candidate, pivot)
        previous_x, current_x = current_x, candidate
        previous_y, current_y = current_y, value
    return current_x


def log_forward(sample, pivot, log_black):
    z = fma(-1. / (pivot + -1.), sample - pivot, 0.)
    negative = z < 0.
    absolute = -z if negative else z
    value = (math.log(fma(absolute, C99, C01)) + CLOG) * CSCALE
    if negative:
        value = -value
    return clamp(fma(-(log_black + -1.), value, log_black))


def log_inverse(sample, pivot, log_black):
    z = fma(-1. / (log_black + -1.), sample - log_black, 0.)
    negative = z < 0.
    absolute = -z if negative else z
    value = (math.exp(fma(absolute, CLOG, -CLOG)) + -C01) * CINV99
    if negative:
        value = -value
    return clamp(fma(-(pivot + -1.), value, pivot))


def evaluate(prepared, sample, inverse=False):
    pivot, log_black, cubic = prepared
    if inverse:
        value = log_inverse(float(sample), pivot, log_black)
        return cubic_inverse(value, pivot) if cubic else value
    value = cubic_forward(float(sample), pivot) if cubic else float(sample)
    return log_forward(clamp(value), pivot, log_black)
