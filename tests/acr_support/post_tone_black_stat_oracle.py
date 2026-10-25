"""Independent prepared-log statistics recovered from ARM64 009d6fac.

Domain: one finite, prepared float32 log plane. This does not construct the
normalized post-tone render request or claim product/Adobe runtime parity.
Sources: 2026-10-15-source-color-stats/{stats-builder,stats_gate_asm}.json;
2026-10-22-optional-state-producers-offline/raw-009d81f0-009d83d0.txt and
quantile-literal.json (03bed910: 2d431cebe2361a3ffca9f1d24d62503f).
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
import math

import numpy as np

REFERENCE_ID = "acr-009d6fac-prepared-log-black-stat-v1"
SOURCE_BINARY_SHA256 = "09978d95dfe76b18495fea9550b051a89f4598ab0fd4734d69d22d330496f40a"
DOMAIN = "prepared_log_plane"
MAX_PIXELS = 1 << 26
# Resolve the host's C log once, independently of the tested renderer.
_LIBM = C.CDLL(ctypes.util.find_library("m") or None)
_LOG = _LIBM.log
_LOG.argtypes = [C.c_double]
_LOG.restype = C.c_double


def compute(log_pixels, epsilon) -> np.ndarray:
    """Return [filtered p.0001, filtered p.01, filtered p.99, high, M]."""
    pixels = np.asarray(log_pixels)
    if (pixels.dtype != np.dtype(np.float32) or pixels.ndim not in (1, 2)
            or not pixels.flags.c_contiguous or not pixels.size
            or pixels.size > MAX_PIXELS or not np.isfinite(pixels).all()):
        raise ValueError("expected nonempty finite contiguous float32 log plane")
    eps = float(epsilon)
    if not math.isfinite(eps) or not 0 < eps <= 2.0 ** -12:
        raise ValueError("epsilon must be finite in (0, 2^-12]")
    # Python's independent stable numeric sort, rather than the native sorter.
    population = sorted(float(value) for value in pixels.flat)
    p1 = population[math.floor((len(population) - 1) * .01)]
    p99 = population[math.floor((len(population) - 1) * .99)]
    threshold = max(p99 - .5, p1 + (p99 - p1) * .5)
    log_epsilon = _LOG(eps) / 0.6931471805599453
    low = log_epsilon + 1.0
    filtered = []
    previous = -1000.0
    for value in population:
        if value >= low and not (value >= threshold and value == previous):
            filtered.append(value)
        previous = value
    if not filtered:
        filtered = [log_epsilon]
    count = len(filtered)
    return np.asarray([filtered[min(math.floor(count * .0001), count - 1)],
                       filtered[min(math.floor(count * .01), count - 1)],
                       filtered[min(math.floor(count * .99), count - 1)],
                       filtered[min(math.floor(count * .9999), count - 1)],
                       float(count)], dtype=np.float64)
