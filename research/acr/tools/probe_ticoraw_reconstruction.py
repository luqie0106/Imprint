#!/usr/bin/env python3
"""Compare a scalar model extracted from ACR with an external step2 reference.

This does not execute Adobe code and does not decode an entropy stream.
Only complete-image, replicated-boundary, LUT-enabled uint16 output is covered.
"""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np


def scalar_model(p1, p2, p3, p4, lut, shift=2, bits=14):
    """FUN_013c8bc8 scalar tail: four-plane neighbourhoods, bias, LUT, clip.

    p1..p4 follow the decompiler's first four plane parameters. Formula and
    signed shifts are from the scalar tail around 013c9214..013c93c0; fixed
    shift=2 and bits=14 are external reference settings, not recovered ABI.
    """
    rows, cols = p1.shape
    out = np.empty((rows * 2, cols * 2), dtype=np.uint16)
    bias = 1 << (shift + bits - 1)
    rounding = 1 << (shift - 1) if shift > 0 else 0

    def convert(value):
        index = min(max(int(value), 0), len(lut) - 1)
        return min(max((int(lut[index]) + rounding) >> shift, 0), (1 << bits) - 1)

    for row in range(rows):
        prev = max(row - 1, 0)
        following = min(row + 1, rows - 1)
        for col in range(cols):
            left = max(col - 1, 0)
            right = min(col + 1, cols - 1)
            a, b, c, d = (int(p[row, col]) for p in (p1, p2, p3, p4))
            top0 = a + bias + ((int(p2[row, left]) + int(p3[prev, col]) + b + c) >> 2)
            bottom1 = d + bias + ((int(p3[row, right]) + b + c + int(p2[following, col])) >> 2)
            out[2 * row, 2 * col] = convert(top0)
            out[2 * row, 2 * col + 1] = convert(b + bias)
            out[2 * row + 1, 2 * col] = convert(c + bias)
            out[2 * row + 1, 2 * col + 1] = convert(bottom1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_library", type=Path,
                        help="dylib exposing run_step2; build instructions in phase3 report")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.reference_library.resolve():
        parser.error("refusing to overwrite output or reference library")
    library = ctypes.CDLL(str(args.reference_library.resolve()))
    function = library.run_step2
    ptr = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [ptr] * 4 + [ctypes.c_int, ctypes.c_int, ptr,
                                    ctypes.POINTER(ctypes.c_uint16)]
    function.restype = None
    rng = np.random.default_rng(3520)
    cases = []
    for rows, cols in [(1, 1), (1, 19), (7, 1), (7, 19), (32, 64)]:
        for amplitude in [5000, 80000]:
            planes = [rng.integers(-amplitude, amplitude + 1, (rows, cols), dtype=np.int32)
                      for _ in range(4)]
            # Deliberately exercises negative LUT entries, clipping, rounding and
            # nonmultiples of four. Intermediate sums stay within int32.
            lut = np.arange(81792, dtype=np.int32) * 3 - 1500
            actual = np.zeros((rows * 2, cols * 2), dtype=np.uint16)
            function(*(p.ctypes.data_as(ptr) for p in planes), rows, cols,
                     lut.ctypes.data_as(ptr), actual.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
            expected = scalar_model(*planes, lut)
            different = int(np.count_nonzero(actual != expected))
            cases.append({"rows": rows, "columns": cols, "amplitude": amplitude,
                          "pixels": int(actual.size), "different": different})
    report = {"model_origin": "ACR 17.2 ARM64 FUN_013c8bc8 scalar tail",
              "reference": "redlamp-libraw 4abfcd2 step2_bayer_rows, not Adobe execution",
              "scope": "whole-image replicated edges, LUT enabled, shift 2, bits 14",
              "seed": 3520, "cases": cases,
              "all_equal": all(c["different"] == 0 for c in cases)}
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["all_equal"] else 1)


if __name__ == "__main__":
    main()
