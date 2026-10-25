#!/usr/bin/env python3
"""Independent float32 oracle for the recovered core3 row kernels.

Covers only the two ported pixel consumers:

  * FUN_0081376c: six gradient rows (center, +/-1 and +/-2 rows).
  * FUN_00813368: five-tap local standard deviation along a row and across
    five rows.

The full cr_stage_bayer_core3 pipeline is not implemented; this module does
not claim full-resolution RGB output.

The reference below takes the minus2/plus2 rows explicitly. It is deliberately
not derived from the C++ source; the expected direction is fixed by the saved
0081376c disassembly:

  q0 = C[j..j+3]      q2 = C[j-2..j+1]     q3 = C[j+2..j+5]
  q1 = Vminus2[...]   q4 = Vplus2[...]
  out0 = |C - C[j+2]|
  out1 = |C - C[j-2]| + |C - C[j+2]|
  out2 = |C - Vplus2|
  out3 = |C - Vminus2| + |C - Vplus2|
  out4 = |(C[j-1] + C[j+1]) - 2*C|
  out5 = |(Vminus1 + Vplus1) - 2*C|
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BRIDGE = ROOT / "native-renderer/build/libimprint_stage3_core3_test_bridge.dylib"

OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
UNSUPPORTED_REMAINING = 3
UNSUPPORTED_ALIGNMENT = 4

STATUS_NAMES = {
    OK: "ok",
    INVALID_ARGUMENT: "invalid_argument",
    INVALID_DIMENSIONS: "invalid_dimensions",
    UNSUPPORTED_REMAINING: "unsupported_remaining",
    UNSUPPORTED_ALIGNMENT: "unsupported_alignment",
}

# Aligned test buffers put column 0 at this many floats from the row base, so a
# 16-byte-aligned row base yields a 16-byte-aligned column 0. Two guard samples
# are required by the kernel; the extra two make the aligned domain explicit.
GUARD = 4
OUTPUT_STRIDE_SLACK = 4


def gradient_reference(center, minus2, plus2, minus1, plus1, width, offset=GUARD):
    """Per-column form of FUN_0081376c with explicit vertical rows.

    `center`, `minus2`, `plus2`, `minus1`, `plus1` are padded 1-D float32 rows.
    `offset` is the index of column 0 in each padded row.
    """
    width = int(width)
    offset = int(offset)
    c = center[offset:offset + width]
    c_minus2 = center[offset - 2:offset - 2 + width]
    c_minus1 = center[offset - 1:offset - 1 + width]
    c_plus1 = center[offset + 1:offset + 1 + width]
    c_plus2 = center[offset + 2:offset + 2 + width]
    twice_c = c + c
    out0 = np.abs(c - c_plus2)
    out1 = np.abs(c - c_minus2) + np.abs(c - c_plus2)
    out2 = np.abs(c - plus2[offset:offset + width])
    out3 = (np.abs(c - minus2[offset:offset + width])
            + np.abs(c - plus2[offset:offset + width]))
    out4 = np.abs((c_minus1 + c_plus1) - twice_c)
    out5 = np.abs((minus1[offset:offset + width]
                   + plus1[offset:offset + width]) - twice_c)
    return [out0, out1, out2, out3, out4, out5]


def _energy5(a2, a1, a0, am1, am2):
    """a2..am2 are float32 arrays for samples +2,+1,0,-1,-2."""
    mean = (am2 + am1 + a0 + a1 + a2) * np.float32(0.2)
    d2 = a2 - mean
    d1 = a1 - mean
    d0 = a0 - mean
    dm2 = am2 - mean
    dm1 = am1 - mean
    return np.sqrt(d2 * d2 + d1 * d1 + d0 * d0 + dm2 * dm2 + dm1 * dm1)


def energy_reference(horizontal, vertical_rows, width, offset=GUARD):
    """FUN_00813368: horizontal row std and vertical 5-row std.

    `vertical_rows` is [minus2, minus1, center, plus1, plus2], each padded.
    """
    width = int(width)
    offset = int(offset)
    horizontal_out = _energy5(
        horizontal[offset + 2:offset + 2 + width],
        horizontal[offset + 1:offset + 1 + width],
        horizontal[offset:offset + width],
        horizontal[offset - 1:offset - 1 + width],
        horizontal[offset - 2:offset - 2 + width],
    )
    vertical_out = _energy5(
        vertical_rows[4][offset:offset + width],
        vertical_rows[3][offset:offset + width],
        vertical_rows[2][offset:offset + width],
        vertical_rows[1][offset:offset + width],
        vertical_rows[0][offset:offset + width],
    )
    return horizontal_out, vertical_out


def _raw_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ptr(array):
    pointer = _raw_ptr(array)
    address = ctypes.cast(pointer, ctypes.c_void_p).value
    if address is None or address % 16 != 0:
        raise AssertionError(f"test buffer is not 16-byte aligned: {address}")
    return pointer


def _u32(array):
    return np.ascontiguousarray(array).view(np.uint32)


def same_bits(got, expected):
    """Exact bit comparison for finite values (NaN payloads not compared)."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    return bool(np.array_equal(_u32(got), _u32(expected)))


def same_nonfinite_semantics(got, expected):
    """Compare NaN/Inf classification plus exact bits for finite values."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    if got.shape != expected.shape:
        return False
    got_nan = np.isnan(got)
    expected_nan = np.isnan(expected)
    if not np.array_equal(got_nan, expected_nan):
        return False
    got_inf = np.isinf(got)
    expected_inf = np.isinf(expected)
    if not np.array_equal(got_inf, expected_inf):
        return False
    if np.any(got_inf):
        if not np.array_equal(np.signbit(got[got_inf]),
                              np.signbit(expected[expected_inf])):
            return False
    finite = ~(got_nan | got_inf)
    return bool(np.array_equal(_u32(got[finite]), _u32(expected[finite])))


def _max_finite_diff(got, expected):
    got = np.asarray(got)
    expected = np.asarray(expected)
    finite = np.isfinite(got) & np.isfinite(expected)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(got[finite].astype(np.float64)
                               - expected[finite].astype(np.float64))))


class Bridge:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self.gradient = self.lib.acr_test_core3_gradient
        self.gradient.restype = ctypes.c_int
        self.gradient.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
        ]
        self.energy = self.lib.acr_test_core3_energy
        self.energy.restype = ctypes.c_int
        self.energy.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]

    def call_gradient(self, center, source_stride, width, output_stride,
                      require_alignment=True):
        storage = np.full((6, output_stride), np.float32(np.nan), dtype=np.float32)
        pointers = [_ptr(storage[row]) for row in range(6)]
        center_ptr = _ptr(center) if require_alignment else _raw_ptr(center)
        status = self.gradient(
            center_ptr, source_stride, width,
            pointers[0], pointers[1], pointers[2], pointers[3], pointers[4],
            pointers[5], output_stride,
        )
        return status, storage

    def call_energy(self, horizontal, vertical_center, row_stride, width):
        horizontal_out = np.full(width, np.float32(np.nan), dtype=np.float32)
        vertical_out = np.full(width, np.float32(np.nan), dtype=np.float32)
        status = self.energy(
            _ptr(horizontal), _ptr(vertical_center),
            row_stride, width, _ptr(horizontal_out), _ptr(vertical_out),
        )
        return status, horizontal_out, vertical_out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_rows(seed: int, width: int, stride: int, height: int = 5) -> np.ndarray:
    """Random padded stack; row bases and column 0 are 16-byte aligned."""
    if stride % 4 != 0 or stride < width + GUARD + 2:
        raise ValueError("stride must be a multiple of 4 with enough guards")
    rng = np.random.default_rng(seed)
    rows = rng.uniform(-10.0, 10.0, size=(height, stride)).astype(np.float32)
    # Keep the guard samples consistent with the row edges; the reference uses
    # the same padded rows, so this only makes the probes readable.
    rows[:, :GUARD] = rows[:, GUARD:GUARD + 1]
    rows[:, GUARD + width:] = rows[:, GUARD + width - 1:GUARD + width]
    return rows


def constant_rows(values, width: int, stride: int) -> np.ndarray:
    rows = np.empty((len(values), stride), dtype=np.float32)
    for index, value in enumerate(values):
        rows[index, :] = np.float32(value)
    return rows


def workspace_rows(got):
    """0081376c output index -> 007eaf20 workspace row (verified call site)."""
    mapping = {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4}
    workspace = [None] * 9
    for output_index, row in mapping.items():
        workspace[row] = got[output_index]
    return workspace


def case_records(bridge: Bridge):
    records = []

    def add_gradient(name, rows, center_row, width, output_stride,
                     expected=None, nonfinite=False):
        center = rows[center_row][GUARD:]
        status, storage = bridge.call_gradient(
            center, rows.shape[1], width, output_stride)
        reference = gradient_reference(
            rows[center_row], rows[center_row - 2], rows[center_row + 2],
            rows[center_row - 1], rows[center_row + 1], width)
        got = [storage[row][:width] for row in range(6)]
        if expected is None:
            expected = reference
            expected_source = "reference"
        else:
            expected = [np.asarray(row, dtype=np.float32) for row in expected]
            expected_source = "hardcoded"
        compare = same_nonfinite_semantics if nonfinite else same_bits
        expected_match = all(
            compare(got[row], expected[row]) for row in range(6))
        reference_match = all(
            compare(got[row], reference[row]) for row in range(6))
        padding_ok = all(
            np.all(np.isnan(storage[row][width:])) for row in range(6))
        max_diff = max(
            (_max_finite_diff(got[row], expected[row]) for row in range(6)),
            default=0.0,
        )
        records.append({
            "name": name,
            "kind": "gradient",
            "status": STATUS_NAMES.get(status, str(status)),
            "match": bool(status == OK and expected_match and reference_match),
            "expected_source": expected_source,
            "expected_match": bool(expected_match),
            "reference_match": bool(reference_match),
            "padding_untouched": bool(padding_ok),
            "nonfinite_contract": bool(nonfinite),
            "max_abs_diff": max_diff,
            "width": width,
            "output_stride": output_stride,
        })

    def add_energy(name, horizontal, vertical_rows, width, expected=None,
                   nonfinite=False):
        status, got_h, got_v = bridge.call_energy(
            horizontal[GUARD:], vertical_rows[2][GUARD:],
            horizontal.shape[0], width)
        expected_h, expected_v = energy_reference(
            horizontal, vertical_rows, width)
        if expected is None:
            expected_hard = (expected_h, expected_v)
            expected_source = "reference"
        else:
            expected_hard = (np.asarray(expected[0], dtype=np.float32),
                             np.asarray(expected[1], dtype=np.float32))
            expected_source = "hardcoded"
        compare = same_nonfinite_semantics if nonfinite else same_bits
        expected_match = (compare(got_h, expected_hard[0])
                          and compare(got_v, expected_hard[1]))
        reference_match = (compare(got_h, expected_h)
                           and compare(got_v, expected_v))
        max_diff = max(_max_finite_diff(got_h, expected_hard[0]),
                       _max_finite_diff(got_v, expected_hard[1]))
        records.append({
            "name": name,
            "kind": "energy",
            "status": STATUS_NAMES.get(status, str(status)),
            "match": bool(status == OK and expected_match and reference_match),
            "expected_source": expected_source,
            "expected_match": bool(expected_match),
            "reference_match": bool(reference_match),
            "nonfinite_contract": bool(nonfinite),
            "max_abs_diff": max_diff,
            "width": width,
        })

    # Random coverage across aligned widths and padded output strides.
    for width in (4, 8, 16, 32):
        stride = width + 8
        rows = make_rows(1000 + width, width, stride)
        add_gradient(f"random_gradient_{width}", rows, 2, width,
                     width + OUTPUT_STRIDE_SLACK)
        add_energy(f"random_energy_{width}", rows[2],
                   [rows[row] for row in range(5)], width)

    # ---- Hardcoded direction probes (not derived from the C++ source). ------

    # 1. Vertical step: five constant rows 0/10/20/30/40, center row 2.
    #    out2 must be |20-40| = 20 and out3 = |20-0| + |20-40| = 40.
    #    The previous implementation returned six zero rows for this input.
    width, stride = 4, 12
    vertical = constant_rows([0, 10, 20, 30, 40], width, stride)
    zero = np.zeros(width, dtype=np.float32)
    twenty = np.full(width, 20.0, dtype=np.float32)
    forty = np.full(width, 40.0, dtype=np.float32)
    add_gradient("probe_vertical_step_5rows", vertical, 2, width, 4,
                 expected=[zero, zero, twenty, forty, zero, zero])

    # 2. Pure horizontal ramp, every row identical: out0=2, out1=4, and the
    #    vertical rows out2/out3 must be zero. Horizontal/vertical axis swap
    #    would make out2/out3 nonzero here.
    width, stride = 16, 24
    horizontal = np.empty((5, stride), dtype=np.float32)
    for row in range(5):
        horizontal[row, :] = np.arange(-GUARD, stride - GUARD, dtype=np.float32)
    add_gradient(
        "probe_horizontal_ramp_identical_rows", horizontal, 2, width, 16,
        expected=[np.full(width, 2.0, dtype=np.float32),
                  np.full(width, 4.0, dtype=np.float32),
                  np.zeros(width, dtype=np.float32),
                  np.zeros(width, dtype=np.float32),
                  np.zeros(width, dtype=np.float32),
                  np.zeros(width, dtype=np.float32)])

    # 3. Different horizontal and vertical slopes (j + 10*r): out0/1 stay 2/4,
    #    out2 = 20, out3 = 40. This catches reading only one of the two rows
    #    or swapping the axes.
    width, stride = 16, 24
    sloped = np.empty((5, stride), dtype=np.float32)
    for row in range(5):
        sloped[row, :] = (np.arange(-GUARD, stride - GUARD, dtype=np.float32)
                          + np.float32(10 * row))
    add_gradient(
        "probe_distinct_horizontal_vertical_slopes", sloped, 2, width, 16,
        expected=[np.full(width, 2.0, dtype=np.float32),
                  np.full(width, 4.0, dtype=np.float32),
                  np.full(width, 20.0, dtype=np.float32),
                  np.full(width, 40.0, dtype=np.float32),
                  np.zeros(width, dtype=np.float32),
                  np.zeros(width, dtype=np.float32)])

    # 4. Asymmetric rows: minus2=100, minus1=-5, center=0, plus1=7, plus2=40.
    #    out2=40, out3=140, out4=|(-5+7)-0|=2, out5=2.
    width, stride = 8, 16
    asymmetric = constant_rows([100, -5, 0, 7, 40], width, stride)
    add_gradient(
        "probe_asymmetric_rows", asymmetric, 2, width, 8,
        expected=[np.zeros(width, dtype=np.float32),
                  np.zeros(width, dtype=np.float32),
                  np.full(width, 40.0, dtype=np.float32),
                  np.full(width, 140.0, dtype=np.float32),
                  np.zeros(width, dtype=np.float32),
                  np.full(width, 2.0, dtype=np.float32)])

    # 5. Impulse in the plus2 row only at column 5: out2/out3 see it at column
    #    5, and no other row does. The old horizontal-difference form also
    #    lit columns 3 and 7 and doubled out3 at column 5.
    width, stride = 8, 16
    plus2_impulse = np.zeros((5, stride), dtype=np.float32)
    plus2_impulse[4, GUARD + 5] = 1.0
    impulse_expected = np.zeros(width, dtype=np.float32)
    impulse_expected[5] = 1.0
    add_gradient(
        "probe_plus2_row_impulse", plus2_impulse, 2, width, 8,
        expected=[np.zeros(width, dtype=np.float32),
                  np.zeros(width, dtype=np.float32),
                  impulse_expected.copy(),
                  impulse_expected.copy(),
                  np.zeros(width, dtype=np.float32),
                  np.zeros(width, dtype=np.float32)])

    # 6. Impulse in the minus2 row only: out2 must stay zero, out3 must see it.
    #    A kernel missing the minus2 read fails here.
    minus2_impulse = np.zeros((5, stride), dtype=np.float32)
    minus2_impulse[0, GUARD + 5] = -2.0
    minus2_expected = np.zeros(width, dtype=np.float32)
    minus2_expected[5] = 2.0
    add_gradient(
        "probe_minus2_row_impulse", minus2_impulse, 2, width, 8,
        expected=[np.zeros(width, dtype=np.float32),
                  np.zeros(width, dtype=np.float32),
                  np.zeros(width, dtype=np.float32),
                  minus2_expected,
                  np.zeros(width, dtype=np.float32),
                  np.zeros(width, dtype=np.float32)])

    # 7. Center-row impulse: horizontal/edge rows see it, vertical rows do not.
    center_impulse = np.zeros((5, stride), dtype=np.float32)
    center_impulse[2, GUARD + 5] = 1.0
    out0 = np.zeros(width, dtype=np.float32)
    out1 = np.zeros(width, dtype=np.float32)
    out4 = np.zeros(width, dtype=np.float32)
    out5 = np.zeros(width, dtype=np.float32)
    out0[[3, 5]] = 1.0
    out1[[3, 7]] = 1.0
    out1[5] = 2.0
    out2 = np.zeros(width, dtype=np.float32)
    out2[5] = 1.0
    out3 = np.zeros(width, dtype=np.float32)
    out3[5] = 2.0
    out4[[4, 6]] = 1.0
    out4[5] = 2.0
    out5[5] = 2.0
    add_gradient(
        "probe_center_row_impulse", center_impulse, 2, width, 8,
        expected=[out0, out1, out2, out3, out4, out5])

    # ---- 00813368 direction probes. ---------------------------------------

    width, stride = 16, 24
    constant = np.full((5, stride), 0.25, dtype=np.float32)
    add_energy("probe_energy_constant", constant[2],
               [constant[row] for row in range(5)], width,
               expected=(np.zeros(width, dtype=np.float32),
                         np.zeros(width, dtype=np.float32)))

    # All rows identical horizontal ramp: horizontal std = sqrt(10), vertical 0.
    horizontal = np.empty((5, stride), dtype=np.float32)
    for row in range(5):
        horizontal[row, :] = np.arange(-GUARD, stride - GUARD, dtype=np.float32)
    sqrt10 = np.float32(np.sqrt(np.float32(10.0)))
    add_energy("probe_energy_horizontal_ramp", horizontal[2],
               [horizontal[row] for row in range(5)], width,
               expected=(np.full(width, sqrt10, dtype=np.float32),
                         np.zeros(width, dtype=np.float32)))

    # Vertical ramp only: horizontal 0, vertical sqrt(1000).
    vertical = constant_rows([0, 10, 20, 30, 40], width, stride)
    sqrt1000 = np.float32(np.sqrt(np.float32(1000.0)))
    add_energy("probe_energy_vertical_ramp", vertical[2],
               [vertical[row] for row in range(5)], width,
               expected=(np.zeros(width, dtype=np.float32),
                         np.full(width, sqrt1000, dtype=np.float32)))

    # Distinct slopes: horizontal sqrt(10), vertical sqrt(1000).
    sloped = np.empty((5, stride), dtype=np.float32)
    for row in range(5):
        sloped[row, :] = (np.arange(-GUARD, stride - GUARD, dtype=np.float32)
                          + np.float32(10 * row))
    add_energy("probe_energy_distinct_slopes", sloped[2],
               [sloped[row] for row in range(5)], width,
               expected=(np.full(width, sqrt10, dtype=np.float32),
                         np.full(width, sqrt1000, dtype=np.float32)))

    # Non-finite contract: propagate, do not clamp. The reference and the C++
    # must agree on NaN/Inf classification (bitwise for finite lanes).
    width, stride = 8, 16
    nonfinite = np.zeros((5, stride), dtype=np.float32)
    nonfinite[2, GUARD + 2] = np.inf
    nonfinite[2, GUARD + 4] = np.nan
    add_gradient("probe_nonfinite_propagation", nonfinite, 2, width, 8,
                 nonfinite=True)

    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.bridge.exists():
        raise SystemExit(f"bridge not found: {args.bridge}")
    bridge = Bridge(args.bridge)
    records = case_records(bridge)
    failures = [record for record in records if not record["match"]
                or not record.get("padding_untouched", True)]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "bridge": str(args.bridge),
        "bridge_sha256": sha256(args.bridge),
        "scope": "FUN_0081376c and FUN_00813368 only; no full-res RGB",
        "case_count": len(records),
        "passed": len(records) - len(failures),
        "failed": len(failures),
        "cases": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{payload['passed']}/{payload['case_count']} core3 row-kernel cases passed")
    for record in failures:
        print(f"FAIL: {record['name']}: max_abs_diff={record['max_abs_diff']}",
              file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
