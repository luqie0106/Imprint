"""Independent coordinate oracle for 2x2 Bayer-preserving border extension.

Coordinates are mapped independently on each axis.  Outside the available
rectangle, the coordinate is moved to one of the two edge samples with the
same parity; this is deliberately neither image reflection nor edge/wrap
padding.
"""
from __future__ import annotations

import numpy as np


SENTINEL = np.uint16(0xA55A)


def map_coordinate(coordinate: int, lo: int, hi: int) -> int:
    """Map one integer coordinate into ``[lo, hi)`` while keeping parity."""
    if lo <= coordinate < hi:
        return coordinate
    if coordinate < lo:
        return lo + (coordinate - lo) % 2
    return hi - 2 + (coordinate - hi) % 2


def reference(source: np.ndarray, source_rect, request_rect, *,
              destination_stride: int | None = None,
              sentinel: int | np.uint16 = SENTINEL) -> np.ndarray:
    """Return the requested rectangle and untouched row padding.

    ``source`` is a 2-D uint16 buffer whose first row/column correspond to
    ``source_rect[:2]``. Its second dimension may include row padding. The
    returned array includes ``destination_stride`` samples per row, with
    padding initialized to ``sentinel``.
    """
    samples = np.asarray(source)
    if samples.dtype != np.uint16 or samples.ndim != 2:
        raise ValueError("source must be a two-dimensional uint16 array")
    st, sl, sb, sr = map(int, source_rect)
    qt, ql, qb, qr = map(int, request_rect)
    source_height, source_width = sb - st, sr - sl
    request_height, request_width = qb - qt, qr - ql
    if source_height < 2 or source_width < 2:
        raise ValueError("source dimensions must both be at least two")
    if request_height <= 0 or request_width <= 0:
        raise ValueError("request dimensions must be positive")
    if samples.shape[0] < source_height or samples.shape[1] < source_width:
        raise ValueError("source buffer does not cover source_rect")
    if destination_stride is None:
        destination_stride = request_width
    if destination_stride < request_width:
        raise ValueError("destination stride is shorter than request width")

    result = np.full((request_height, destination_stride), sentinel,
                     dtype=np.uint16)
    for out_y, request_y in enumerate(range(qt, qb)):
        source_y = map_coordinate(request_y, st, sb) - st
        for out_x, request_x in enumerate(range(ql, qr)):
            source_x = map_coordinate(request_x, sl, sr) - sl
            result[out_y, out_x] = samples[source_y, source_x]
    return result
