"""Saved ARM64 BuildStats selected fields; host libm, no Adobe runtime claim."""
import ctypes as C
from collections import Counter
import math
import struct

LIBM = C.CDLL(None)
for name, argc in (('fma', 3), ('pow', 2), ('exp2', 1), ('log', 1), ('log2', 1)):
    function = getattr(LIBM, name)
    function.argtypes = [C.c_double] * argc
    function.restype = C.c_double


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def linear(x, epsilon):
    return max(0., LIBM.exp2(max(-120., min(120., x))) - epsilon)


def encode(x):
    return 12.92*x if x <= .0031308 else LIBM.fma(LIBM.pow(x, 1./2.4), 1.055, -.055)


def decode(x):
    return x*.07739938080495357 if x <= .040449936 else LIBM.pow((x+.055)*.9478672985781991, 2.4)


def center_from_groups(groups, lo, hi, epsilon):
    """009d85dc..8798, including the actual post-decrement upper-rank scan.

    The upper scan subtracts counts[current] BEFORE moving to the previous
    group (009d8670..80). Do not replace it with a mathematical prefix-rank
    lookup: nonuniform repeated groups distinguish those two algorithms.
    """
    counts = [count for _, count in groups]
    values = [linear(value, epsilon) for value, _ in groups]
    cumulative = 0
    left_index = 0
    for i, count in enumerate(counts):
        cumulative += count
        if cumulative > lo:
            left_index = i
            break
    right_index = len(counts)-1
    right_count = 1
    start = sum(counts)-counts[-1]
    for i in range(len(counts)-1, -1, -1):
        if start <= hi:
            right_index = i
            right_count = 0 if left_index == i else hi-start+1
            break
        start = (start-counts[i]) & 0xffffffff
    left, right = values[left_index], values[right_index]
    inverse = 0. if left == right else -1./(left-right)
    total = 0. if left_index == right_index else float(right_count)
    for i in range(left_index+1, right_index):
        z = min(1., max(0., LIBM.fma(inverse, values[i]-left, 0.)))
        total = LIBM.fma(float(counts[i]), encode(z), total)
    decoded = decode(total/(hi-lo+1))
    representative = LIBM.fma(-(left-right), decoded, left)
    center = LIBM.log2(epsilon+max(0., min(2.**120, representative)))
    return center, (left_index, right_index, right_count)


def selected_stats(samples, epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('positive finite epsilon required')
    values = sorted(f32(x) for x in samples)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('nonempty finite input required')
    n = len(values)
    raw_low = values[math.floor((n-1)*.01)]
    raw_high = values[math.ceil((n-1)*.99)]
    threshold = max(raw_high-.5, LIBM.fma(raw_high-raw_low, .5, raw_low))
    fallback_log = LIBM.log(epsilon)/.6931471805599453
    effective = []
    previous = -1000.
    for value in values:
        if value >= fallback_log+1 and (value != previous or value < threshold):
            effective.append(value)
        previous = value
    fallback = not effective
    if fallback:
        effective = [fallback_log]
    groups = list(Counter(effective).items())
    m = len(effective)
    lo, hi = min(m-1, int(m*.0001)), min(m-1, int(m*.9999))
    center, ranks = center_from_groups(groups, lo, hi, epsilon)
    fields = [center, effective[lo], effective[hi],
              (effective[(m-1)//2]+effective[m//2])*.5, float(m), float(fallback)]
    return fields, dict(groups=groups, lo=lo, hi=hi, selected_groups=ranks,
                        raw_low=raw_low, raw_high=raw_high, threshold=threshold)
