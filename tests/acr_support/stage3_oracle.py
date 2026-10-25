#!/usr/bin/env python3
"""Version-controlled independent oracle for the Stage3 closed domain.

This module is the maintainable copy used by ordinary pytest and by the
official-object bridge check. It does not import anything under the ignored
``research/acr`` tree.

Closed domain (recovered from Camera Raw 17.2 ARM64):

  * ``cr_mosaic_info`` mode 1;
  * canonical RGGB CFA byte pattern ``{0,1,1,2}``;
  * plane count 3, plane map ``[0,1,2]``;
  * ``mosaic+0x7c == 0`` and ``mosaic+0x80/+0x84 == 0``;
  * pair (2,2), no task transform, even rect origin and size.

Pair (1,1) is reported as ``unsupported_full_res_core3`` because FUN_007f2124
constructs the multi-stage ``cr_stage_bayer_core3`` pipeline. Other pairs,
plane maps, task transforms, offsets and CFA phases are explicit unsupported
results.

Why only RGGB: FUN_00ab21f0 leaves ``mosaic+0x80/+0x84`` at zero only for the
identity RGGB arrangement. The other standard Bayer phases store a packed
nonzero offset pair (0x100000000, 0x1, 0x100000001) there and then enter the
unrecovered FUN_00ab4094 transform. The fixed-position FUN_007f851c body also
assumes RGGB: top-left -> plane 0, top-right+bottom-left -> plane 1,
bottom-right -> plane 2.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BRIDGE = ROOT / "native-renderer/build/libimprint_stage3_test_bridge.dylib"

OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
INVALID_CFA = 3
UNSUPPORTED_MODE = 4
UNSUPPORTED_PAIR = 5
UNSUPPORTED_OFFSET = 6
UNSUPPORTED_PLANE_MAP = 7
UNSUPPORTED_TASK_TRANSFORM = 8
UNSUPPORTED_FULL_RES_CORE3 = 9
UNSUPPORTED_KERNEL = 10
INVALID_STRIDE = 11
BUFFER_TOO_SMALL = 12
UNSUPPORTED_ALIASING = 13
ALLOCATION_FAILURE = 14
INVALID_RECT = 15

DIRECT = 0
TASK = 1
GENERIC = 2

KERNEL_NONE = 0
KERNEL_2X2 = 1
KERNEL_2X4 = 2
KERNEL_1X1 = 3
KERNEL_GENERIC = 4

STATUS_NAMES = {
    OK: "ok",
    INVALID_ARGUMENT: "invalid_argument",
    INVALID_DIMENSIONS: "invalid_dimensions",
    INVALID_CFA: "invalid_cfa",
    UNSUPPORTED_MODE: "unsupported_mode",
    UNSUPPORTED_PAIR: "unsupported_pair",
    UNSUPPORTED_OFFSET: "unsupported_offset",
    UNSUPPORTED_PLANE_MAP: "unsupported_plane_map",
    UNSUPPORTED_TASK_TRANSFORM: "unsupported_task_transform",
    UNSUPPORTED_FULL_RES_CORE3: "unsupported_full_res_core3",
    UNSUPPORTED_KERNEL: "unsupported_kernel",
    INVALID_STRIDE: "invalid_stride",
    BUFFER_TOO_SMALL: "buffer_too_small",
    UNSUPPORTED_ALIASING: "unsupported_aliasing",
    ALLOCATION_FAILURE: "allocation_failure",
    INVALID_RECT: "invalid_rect",
}

# FUN_00ab21f0 writes these packed pairs at mosaic+0x80 for the four standard
# three-colour 2x2 phases. Only RGGB leaves the pair at zero.
MODE_WRITER_OFFSETS = {
    "rggb": ((0, 1, 1, 2), (0, 0)),
    "bggr": ((2, 1, 1, 0), (1, 1)),
    "grbg": ((1, 0, 2, 1), (0, 1)),
    "gbrg": ((1, 2, 0, 1), (1, 0)),
}


@dataclass
class Prepared:
    width: int = 8
    height: int = 4
    input_stride: int = 8
    pattern: tuple[int, int, int, int] = (0, 1, 1, 2)
    plane_count: int = 3
    plane_map: tuple[int, int, int] = (0, 1, 2)
    mode: int = 1
    pair_x: int = 1
    pair_y: int = 1
    rect_top: int = 0
    rect_left: int = 0
    rect_bottom: int = 4
    rect_right: int = 8
    mosaic_7c: int = 0
    offset_80: int = 0
    offset_84: int = 0
    task_transform: bool = False


def pattern_is_canonical_rggb(pattern: Iterable[int]) -> bool:
    return tuple(pattern) == (0, 1, 1, 2)


def rect_valid(prepared: Prepared) -> bool:
    if prepared.rect_top < 0 or prepared.rect_left < 0:
        return False
    if prepared.rect_bottom <= prepared.rect_top or prepared.rect_right <= prepared.rect_left:
        return False
    return prepared.rect_right <= prepared.width and prepared.rect_bottom <= prepared.height


def expected_dispatch(prepared: Prepared) -> tuple[int, int, int, int, int, int]:
    """Return (status, branch, kernel, out_w, out_h, out_samples)."""
    if prepared.width == 0 or prepared.height == 0:
        return INVALID_DIMENSIONS, GENERIC, KERNEL_NONE, 0, 0, 0
    if prepared.input_stride < prepared.width:
        return INVALID_STRIDE, GENERIC, KERNEL_NONE, 0, 0, 0
    if prepared.plane_count != 3 or not pattern_is_canonical_rggb(prepared.pattern):
        return INVALID_CFA, GENERIC, KERNEL_NONE, 0, 0, 0
    if not rect_valid(prepared):
        return INVALID_RECT, GENERIC, KERNEL_NONE, 0, 0, 0
    if prepared.mode != 1:
        return UNSUPPORTED_MODE, GENERIC, KERNEL_NONE, 0, 0, 0
    if prepared.mosaic_7c != 0:
        return UNSUPPORTED_OFFSET, GENERIC, KERNEL_NONE, 0, 0, 0
    if prepared.offset_80 != 0 or prepared.offset_84 != 0:
        return UNSUPPORTED_OFFSET, GENERIC, KERNEL_NONE, 0, 0, 0
    if tuple(prepared.plane_map) != (0, 1, 2):
        return UNSUPPORTED_PLANE_MAP, TASK, KERNEL_NONE, 0, 0, 0

    pair_equal = prepared.pair_x == prepared.pair_y
    pair_2_4 = prepared.pair_x == 2 and prepared.pair_y == 4
    if not pair_equal and not pair_2_4:
        return UNSUPPORTED_PAIR, GENERIC, KERNEL_NONE, 0, 0, 0

    rect_w = prepared.rect_right - prepared.rect_left
    rect_h = prepared.rect_bottom - prepared.rect_top
    if rect_w <= 0 or rect_h <= 0:
        return INVALID_RECT, DIRECT, KERNEL_NONE, 0, 0, 0

    if prepared.pair_x == 1 and prepared.pair_y == 1:
        return (
            UNSUPPORTED_FULL_RES_CORE3,
            DIRECT,
            KERNEL_1X1,
            rect_w,
            rect_h,
            rect_w * rect_h * 3,
        )

    if rect_w % 2 or rect_h % 2:
        return UNSUPPORTED_KERNEL, DIRECT, KERNEL_NONE, 0, 0, 0

    if prepared.pair_x == 2 and prepared.pair_y == 2:
        out_w, out_h = rect_w // 2, rect_h // 2
        samples = out_w * out_h * 3
        if prepared.task_transform:
            return UNSUPPORTED_TASK_TRANSFORM, DIRECT, KERNEL_2X2, out_w, out_h, samples
        if prepared.rect_left % 2 or prepared.rect_top % 2:
            return UNSUPPORTED_KERNEL, DIRECT, KERNEL_2X2, out_w, out_h, samples
        return OK, DIRECT, KERNEL_2X2, out_w, out_h, samples

    if prepared.pair_x == 2 and prepared.pair_y == 4:
        return UNSUPPORTED_KERNEL, DIRECT, KERNEL_2X4, 0, 0, 0

    return UNSUPPORTED_KERNEL, DIRECT, KERNEL_GENERIC, 0, 0, 0


def oracle_2x2(input_image: np.ndarray, prepared: Prepared) -> np.ndarray:
    """Reference for the closed pair-(2,2) RGGB branch, using explicit indices."""
    top, left = prepared.rect_top, prepared.rect_left
    bottom, right = prepared.rect_bottom, prepared.rect_right
    region = input_image[top:bottom, left:right]
    height, width = region.shape
    if height % 2 or width % 2:
        raise ValueError("oracle_2x2 requires an even rect")
    out = np.empty((height // 2, width // 2, 3), dtype=np.uint16)
    top_row = region[0::2, :].astype(np.uint32)
    bottom_row = region[1::2, :].astype(np.uint32)
    out[:, :, 0] = top_row[:, 0::2].astype(np.uint16)
    out[:, :, 1] = ((top_row[:, 1::2] + bottom_row[:, 0::2] + 1) >> 1).astype(np.uint16)
    out[:, :, 2] = bottom_row[:, 1::2].astype(np.uint16)
    return out


class Bridge:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self.dispatch = self.lib.acr_test_stage3_dispatch
        self.dispatch.restype = ctypes.c_int
        self.dispatch.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32),
        ]
        self.run = self.lib.acr_test_stage3_run_2x2
        self.run.restype = ctypes.c_int
        self.run.argtypes = [
            ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint64,
        ]

    @staticmethod
    def _pattern(prepared: Prepared):
        return (ctypes.c_uint8 * 4)(*prepared.pattern)

    @staticmethod
    def _map(prepared: Prepared):
        return (ctypes.c_uint8 * 3)(*prepared.plane_map)

    def call_dispatch(self, prepared: Prepared):
        pattern = self._pattern(prepared)
        plane_map = self._map(prepared)
        branch = ctypes.c_int32()
        kernel = ctypes.c_int32()
        out_w = ctypes.c_uint64()
        out_h = ctypes.c_uint64()
        out_samples = ctypes.c_uint64()
        half = ctypes.c_int32()
        status = self.dispatch(
            prepared.width, prepared.height, prepared.input_stride,
            prepared.rect_top, prepared.rect_left, prepared.rect_bottom,
            prepared.rect_right, pattern, prepared.plane_count, plane_map,
            prepared.pair_x, prepared.pair_y, prepared.mode, prepared.mosaic_7c,
            prepared.offset_80, prepared.offset_84, int(prepared.task_transform),
            ctypes.byref(branch), ctypes.byref(kernel), ctypes.byref(out_w),
            ctypes.byref(out_h), ctypes.byref(out_samples), ctypes.byref(half),
        )
        return (
            status, branch.value, kernel.value, out_w.value, out_h.value,
            out_samples.value, half.value,
        )

    def call_run(self, image: np.ndarray, prepared: Prepared, sentinel: int | None = None):
        pattern = self._pattern(prepared)
        plane_map = self._map(prepared)
        status, _branch, _kernel, out_w, out_h, out_samples, _half = self.call_dispatch(prepared)
        capacity = max(out_samples, 1) if sentinel is None else 64
        output = np.full(capacity, 0xABCD if sentinel is None else sentinel, dtype=np.uint16)
        data = np.ascontiguousarray(image, dtype=np.uint16)
        status = self.run(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_uint64(data.size),
            prepared.width, prepared.height, prepared.input_stride,
            prepared.rect_top, prepared.rect_left, prepared.rect_bottom,
            prepared.rect_right, pattern, prepared.plane_count, plane_map,
            prepared.pair_x, prepared.pair_y, prepared.mode, prepared.mosaic_7c,
            prepared.offset_80, prepared.offset_84, int(prepared.task_transform),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_uint64(output.size),
        )
        return status, output, out_w, out_h, out_samples


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_image(prepared: Prepared, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 65536, size=(prepared.height, prepared.input_stride), dtype=np.uint16)
    if prepared.input_stride > prepared.width:
        image[:, prepared.width:] = 0xFFFF
    return image


def case_record(name: str, prepared: Prepared, image: np.ndarray | None, bridge: Bridge):
    record = {"name": name, "prepared": asdict(prepared)}
    expected = expected_dispatch(prepared)
    actual = bridge.call_dispatch(prepared)
    expected_status, expected_branch, expected_kernel, expected_w, expected_h, expected_samples = expected
    actual_status, actual_branch, actual_kernel, actual_w, actual_h, actual_samples, actual_half = actual
    ok = (
        actual_status == expected_status
        and actual_branch == expected_branch
        and actual_kernel == expected_kernel
        and actual_w == expected_w
        and actual_h == expected_h
        and actual_samples == expected_samples
    )
    record["expected"] = {
        "status": STATUS_NAMES[expected_status],
        "branch": expected_branch,
        "kernel": expected_kernel,
        "output_width": expected_w,
        "output_height": expected_h,
        "output_samples": expected_samples,
    }
    record["actual"] = {
        "status": STATUS_NAMES.get(actual_status, str(actual_status)),
        "branch": actual_branch,
        "kernel": actual_kernel,
        "output_width": actual_w,
        "output_height": actual_h,
        "output_samples": actual_samples,
        "half_pixel_shift": bool(actual_half),
    }
    if not ok:
        record["ok"] = False
        record["error"] = "dispatch mismatch"
        return record

    if expected_status == OK and name == "short_buffer":
        assert image is not None
        run_status, output, _w, _h, _s = bridge.call_run(image, prepared, sentinel=0xABCD)
        record["expected_run_status"] = STATUS_NAMES[BUFFER_TOO_SMALL]
        record["run"] = {
            "status": STATUS_NAMES.get(run_status, str(run_status)),
            "output_unchanged": bool(np.all(output == 0xABCD)),
        }
        if run_status != BUFFER_TOO_SMALL or not np.all(output == 0xABCD):
            record["ok"] = False
            record["error"] = "short buffer did not fail atomically"
            return record
        record["ok"] = True
        return record

    if expected_status == OK:
        assert image is not None
        run_status, output, out_w, out_h, out_samples = bridge.call_run(image, prepared)
        expected_pixels = oracle_2x2(image, prepared)
        got = output[:expected_pixels.size].reshape(expected_pixels.shape)
        pixels_ok = run_status == OK and np.array_equal(got, expected_pixels)
        record["run"] = {
            "status": STATUS_NAMES.get(run_status, str(run_status)),
            "pixels_match": bool(pixels_ok),
            "pixel_count": int(expected_pixels.size),
            "min": int(got.min()),
            "max": int(got.max()),
            "sha256": hashlib.sha256(got.tobytes()).hexdigest(),
        }
        if not pixels_ok:
            record["ok"] = False
            record["error"] = "pixel mismatch"
            return record
    else:
        run_status, output, _w, _h, _s = bridge.call_run(image, prepared, sentinel=0xABCD)
        unchanged = bool(np.all(output == 0xABCD))
        record["run"] = {
            "status": STATUS_NAMES.get(run_status, str(run_status)),
            "output_unchanged": unchanged,
        }
        if run_status != expected_status or not unchanged:
            record["ok"] = False
            record["error"] = "unsupported branch did not stop atomically"
            return record

    record["ok"] = True
    return record


def build_cases():
    cases: list[tuple[str, Prepared, np.ndarray | None]] = []

    full = Prepared()
    cases.append(("full_res_pair_1_1_rggb", full, make_image(full, 1)))

    preview = Prepared(pair_x=2, pair_y=2, rect_bottom=4, rect_right=8)
    cases.append(("preview_pair_2_2_rggb", preview, make_image(preview, 2)))

    # Actual FUN_00ab21f0 output for the four standard phases: only RGGB keeps
    # the +0x80/+0x84 pair at zero.
    for name, (pattern, offsets) in MODE_WRITER_OFFSETS.items():
        prepared = Prepared(width=2, height=2, input_stride=2, pattern=pattern,
                            offset_80=offsets[0], offset_84=offsets[1],
                            rect_bottom=2, rect_right=2, pair_x=2, pair_y=2)
        cases.append((f"mode_writer_{name}", prepared, make_image(prepared, 3)))

    for name, pattern in {
        "non_bayer_0_0_1_2": (0, 0, 1, 2),
        "non_bayer_0_1_2_2": (0, 1, 2, 2),
    }.items():
        prepared = Prepared(width=2, height=2, input_stride=2, pattern=pattern,
                            rect_bottom=2, rect_right=2, pair_x=2, pair_y=2)
        cases.append((name, prepared, make_image(prepared, 4)))

    for name, offset_80, offset_84 in (
        ("rggb_with_offset_80", 1, 0),
        ("rggb_with_offset_84", 0, 1),
    ):
        prepared = Prepared(pair_x=2, pair_y=2, offset_80=offset_80,
                            offset_84=offset_84)
        cases.append((name, prepared, make_image(prepared, 5)))

    padded = Prepared(width=8, height=6, input_stride=10, rect_top=2, rect_left=2,
                      rect_bottom=6, rect_right=8, pair_x=2, pair_y=2)
    cases.append(("nonzero_origin_padded_stride", padded, make_image(padded, 6)))

    odd = Prepared(width=9, height=4, input_stride=9, rect_right=9,
                   pair_x=2, pair_y=2)
    cases.append(("odd_preview_rect", odd, make_image(odd, 7)))

    odd_origin = Prepared(width=8, height=6, input_stride=8, rect_top=1, rect_left=1,
                          rect_bottom=5, rect_right=7, pair_x=2, pair_y=2)
    cases.append(("odd_preview_origin", odd_origin, make_image(odd_origin, 8)))

    pair_2_4 = Prepared(pair_x=2, pair_y=4)
    cases.append(("pair_2_4_explicit", pair_2_4, make_image(pair_2_4, 9)))

    pair_generic = Prepared(pair_x=3, pair_y=3)
    cases.append(("pair_3_3_explicit", pair_generic, make_image(pair_generic, 10)))

    pair_nonequal = Prepared(pair_x=1, pair_y=2)
    cases.append(("pair_1_2_generic", pair_nonequal, make_image(pair_nonequal, 11)))

    mode2 = Prepared(mode=2)
    cases.append(("mode_2_rejected", mode2, make_image(mode2, 12)))

    mosaic_7c = Prepared(mosaic_7c=1)
    cases.append(("mosaic_7c_rejected", mosaic_7c, make_image(mosaic_7c, 13)))

    plane_map = Prepared(plane_map=(0, 2, 1))
    cases.append(("plane_map_task_rejected", plane_map, make_image(plane_map, 14)))

    task = Prepared(pair_x=2, pair_y=2, task_transform=True)
    cases.append(("preview_task_transform_rejected", task, make_image(task, 15)))

    bad_cfa = Prepared(pattern=(0, 1, 1, 1))
    cases.append(("cfa_missing_plane", bad_cfa, make_image(bad_cfa, 16)))

    short_input = Prepared(pair_x=2, pair_y=2)
    image = make_image(short_input, 17)
    cases.append(("short_buffer", short_input, image[:3]))

    rng = np.random.default_rng(20261018)
    for index in range(100):
        width = int(rng.integers(1, 17)) * 2
        height = int(rng.integers(1, 9)) * 2
        stride = width + int(rng.integers(0, 4))
        rect_w = int(rng.integers(1, width // 2 + 1)) * 2
        rect_h = int(rng.integers(1, height // 2 + 1)) * 2
        left = int(rng.integers(0, (width - rect_w) // 2 + 1)) * 2
        top = int(rng.integers(0, (height - rect_h) // 2 + 1)) * 2
        prepared = Prepared(
            width=width, height=height, input_stride=stride,
            pair_x=2, pair_y=2,
            rect_top=top, rect_left=left,
            rect_bottom=top + rect_h, rect_right=left + rect_w,
        )
        cases.append((f"random_{index:03d}", prepared, make_image(prepared, 1000 + index)))

    return cases


def run_bridge(bridge_path: Path, output_path: Path | None = None) -> dict:
    bridge = Bridge(bridge_path)
    records = []
    failures = []
    for name, prepared, image in build_cases():
        try:
            record = case_record(name, prepared, image, bridge)
        except Exception as exc:  # noqa: BLE001 - evidence must record the failure
            record = {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        if not record.get("ok"):
            failures.append(record)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "bridge": str(bridge_path),
        "bridge_sha256": sha256(bridge_path),
        "closed_domain": {
            "mode": 1,
            "pair": [2, 2],
            "cfa_pattern": [0, 1, 1, 2],
            "plane_map": [0, 1, 2],
            "mosaic_7c": 0,
            "offset_80_84": [0, 0],
            "task_transform": False,
            "rect": "even origin and even size, inside the image",
        },
        "unsupported": {
            "pair_1_1": "007f2124 cr_stage_bayer_core3 pipeline not recovered",
            "pair_2_4": "007f9228 kernel not recovered",
            "generic_pair": "007fb724 kernel not recovered",
            "task_transform": "00ab3964 half-pixel task transform not recovered",
            "plane_map": "0127cfe0 task constructor not recovered",
            "offset_80_84": "00ab4094 scaling path not recovered",
            "cfa_other_than_rggb": "FUN_00ab21f0 stores nonzero offsets / transform",
            "mode_other": "only mode 1 is closed",
        },
        "case_count": len(records),
        "passed": len(records) - len(failures),
        "failed": len(failures),
        "cases": records,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.bridge.exists():
        raise SystemExit(f"bridge not found: {args.bridge}")
    payload = run_bridge(args.bridge, args.output)
    print(f"{payload['passed']}/{payload['case_count']} Stage3 oracle cases passed")
    failures = [record for record in payload["cases"] if not record.get("ok")]
    if failures:
        for record in failures:
            print(f"FAIL: {record['name']}: {record.get('error')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
