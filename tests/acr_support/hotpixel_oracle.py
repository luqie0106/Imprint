"""Independent scalar reference for the ACR Bayer hot/cold pixel pass.

The reference consumes five padded rows and writes one output per center pixel.
It deliberately uses Python integers for the signed threshold arithmetic so
negative ``// 4`` follows the recovered floor-division behavior.
"""
from __future__ import annotations

import numpy as np


_SAME_COLOR = {
    0: ((-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2),
        (2, -2), (2, 0), (2, 2)),
    1: ((-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 2),
        (1, -1), (1, 1), (2, 0)),
}
_VERTICAL = ((-1, -2), (-1, 0), (-1, 2), (1, -2), (1, 0), (1, 2))
_HORIZONTAL = ((-2, -1), (0, -1), (2, -1), (-2, 1), (0, 1), (2, 1))
_REPLACEMENT = {
    0: ((-2, 0), (2, 0), (0, -2), (0, 2)),
    1: ((-1, -1), (-1, 1), (1, -1), (1, 1)),
}


def _median_average(values: list[int]) -> int:
    ordered = sorted(values)
    return (ordered[2] + ordered[3] + 1) // 2


def _pixel(rows: np.ndarray, x: int, phase: int, hot: int, cold: int) -> int:
    column_phase = phase ^ (x & 1)
    center = int(rows[2, x + 2])

    def sample(offset: tuple[int, int]) -> int:
        dy, dx = offset
        return int(rows[dy + 2, x + dx + 2])

    same = [sample(offset) for offset in _SAME_COLOR[column_phase]]
    median = sorted([center, *same])[4]
    vertical = _median_average([sample(offset) for offset in _VERTICAL])
    horizontal = _median_average([sample(offset) for offset in _HORIZONTAL])

    residual = (
        sample((-1, 0)) + sample((1, 0)) + sample((0, -1)) + sample((0, 1))
        - 2 * (vertical + horizontal)
    )
    threshold = ((center - median) * 3 + 2) // 4

    if center > hot and all(value < center - hot for value in same) and residual < threshold:
        return max(sample(offset) for offset in _REPLACEMENT[column_phase])
    if center + cold < min(same) and residual > threshold:
        return min(sample(offset) for offset in _REPLACEMENT[column_phase])
    return center


def reference(source: np.ndarray, width: int, phase: int, hot: int, cold: int) -> np.ndarray:
    """Return filtered center pixels from a ``(5, stride)`` uint16 buffer.

    ``source`` stores five rows with center column 2. Each output x uses the
    source center at column ``x + 2`` and phase ``phase ^ (x & 1)``. The native
    bridge owns pointer, capacity, overlap, and overflow validation; this
    mathematical oracle only accepts a well-formed array.
    """
    array = np.asarray(source)
    if array.dtype != np.uint16 or array.ndim != 2 or array.shape[0] != 5:
        raise ValueError("source must be a 5-row uint16 array")
    if width <= 0 or array.shape[1] < width + 4:
        raise ValueError("source stride must provide width + 4 samples")
    if phase not in (0, 1):
        raise ValueError("phase must be 0 or 1")

    output = np.empty(width, dtype=np.uint16)
    for x in range(width):
        output[x] = _pixel(array, x, phase, int(hot), int(cold))
    return output
