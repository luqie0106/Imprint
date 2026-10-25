"""Independent two-dimensional reference for the prepared core3 CPU prefix.

Workspace axes are (spatial row, plane, column); plane 6 has no producer in
these two passes. No dependency on the per-row C++ or its Python oracle.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


def reference(padded: np.ndarray, width: int) -> np.ndarray:
    height = padded.shape[0]
    result = np.full((height, 9, width + 4), np.nan, dtype=np.float32)
    # Gradient samples include horizontal guards from the prepared input.
    c = padded[2:height-2, 4:width+4]
    forward_x = np.abs(c - padded[2:height-2, 6:width+6])
    forward_y = np.abs(c - padded[4:height, 4:width+4])
    result[2:height-2, 0, :width] = forward_x
    result[2:height-2, 1, :width] = np.abs(c - padded[2:height-2, 2:width+2]) + forward_x
    result[2:height-2, 2, :width] = forward_y
    result[2:height-2, 5, :width] = np.abs(c - padded[:height-4, 4:width+4]) + forward_y
    twice = c + c
    result[2:height-2, 3, :width] = np.abs(
        (padded[2:height-2, 3:width+3] + padded[2:height-2, 5:width+5]) - twice)
    result[2:height-2, 4, :width] = np.abs(
        (padded[1:height-3, 4:width+4] + padded[3:height-1, 4:width+4]) - twice)

    def energy(values):
        # mean is accumulated -2,-1,0,+1,+2; squares +2,+1,0,-2,-1.
        m = (((values[0] + values[1]) + values[2]) + values[3]) + values[4]
        m = m * np.float32(0.2)
        d = [v - m for v in values]
        total = d[4]*d[4] + d[3]*d[3]
        total = total + d[2]*d[2]
        total = total + d[0]*d[0]
        return np.sqrt(total + d[1]*d[1])

    # Vertical offsets here change the FIRST axis, never the channel index.
    horizontal = [result[4:height-4, 3, 4+d:width-4+d] for d in (-2, -1, 0, 1, 2)]
    vertical = [result[4+d:height-4+d, 4, 4:width-4] for d in (-2, -1, 0, 1, 2)]
    result[4:height-4, 7, 4:width-4] = energy(horizontal)
    result[4:height-4, 8, 4:width-4] = energy(vertical)
    return result


class Bridge:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self.run = self.lib.acr_test_core3_prefix
        self.run.restype = ctypes.c_int
        self.run.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]

    def execute(self, padded: np.ndarray, width: int):
        height = padded.shape[0]
        output = np.full((height, 9, width+4), -1234, dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        status = self.run(padded.ctypes.data_as(pointer), padded.size, width,
                          height, padded.shape[1], output.ctypes.data_as(pointer), output.size)
        return status, output


def assert_workspace_equal(actual: np.ndarray, expected: np.ndarray):
    mask = ~np.isnan(expected)
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), "validity/halo mask differs"
    assert np.array_equal(actual[mask].view(np.uint32), expected[mask].view(np.uint32)), "defined bits differ"
