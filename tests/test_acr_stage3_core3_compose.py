from __future__ import annotations

import ctypes
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BUILD = NATIVE / "build"
SPEC = importlib.util.spec_from_file_location(
    "core3_compose_oracle", ROOT / "tests/acr_support/core3_compose_oracle.py")
oracle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(oracle)

U16 = ctypes.POINTER(ctypes.c_uint16)
F32 = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t
SIZE_PTR = ctypes.POINTER(SIZE)
SHARED_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"

SOURCE_NAMES = (
    "acr_stage3_core3.cpp",
    "acr_stage3_core3_prefix.cpp",
    "acr_stage3_core3_green.cpp",
    "acr_stage3_core3_colour.cpp",
    "acr_stage3_core3_refine.cpp",
    "acr_stage3_core3_format.cpp",
    "acr_stage3_core3_compose.cpp",
)
BRIDGE_CPP = NATIVE / "tools/acr_stage3_core3_compose_bridge.cpp"
SOURCES = tuple(NATIVE / "src" / name for name in SOURCE_NAMES)
OFFICIAL_OBJECTS = tuple(
    BUILD / "CMakeFiles/imprint_renderer.dir/src" / f"{name}.o"
    for name in SOURCE_NAMES)
COMPOSE_CPP = NATIVE / "src/acr_stage3_core3_compose.cpp"
OK = 0
BUFFER_TOO_SMALL = 4
SENTINEL_FLOAT = np.float32(-1234.5)
SENTINEL_U16 = np.uint16(0xD00D)
SENTINEL_META = np.iinfo(np.uintp).max


def _compile_bridge(target: Path, linked_files: tuple[Path, ...]) -> Path:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "a C++ compiler is required to build the core3 compose bridge"
    shared_flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    command = [
        compiler, "-std=c++17", "-O2", "-fPIC", "-fno-fast-math",
        "-ffp-contract=off", shared_flag, "-I", str(NATIVE / "src"),
        str(BRIDGE_CPP), *(str(path) for path in linked_files), "-o", str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, (
        "core3 compose bridge link failed:\n" + completed.stdout + completed.stderr)
    return target


class Bridge:
    def __init__(self, path: Path):
        self.library = ctypes.CDLL(str(path))
        self.function = self.library.acr_test_core3_compose
        self.function.argtypes = [
            U16, SIZE, SIZE, SIZE, SIZE, F32, F32, ctypes.c_uint32,
            ctypes.c_uint32, F32, SIZE, U16, SIZE, SIZE_PTR, SIZE,
        ]
        self.function.restype = ctypes.c_int

    def call(self, storage: np.ndarray, width: int, *, controls=(25, 0.03, 0, 0.2),
             lower=0.4, upper=0.4, row_origin=0, column_origin=0,
             passed_width: int | None = None,
             source_sample_count: int | None = None,
             float_capacity: int | None = None,
             u16_capacity: int | None = None,
             metadata_capacity: int = 12):
        height, row_stride = storage.shape
        passed_width = width if passed_width is None else passed_width
        expected_count = max(0, height - 2 * (8 if controls[2] == 0 else 12)) * max(
            0, passed_width - 2 * (8 if controls[2] == 0 else 12)) * 3
        float_capacity = expected_count + 8 if float_capacity is None else float_capacity
        u16_capacity = expected_count + 8 if u16_capacity is None else u16_capacity
        output_float = np.full(float_capacity, SENTINEL_FLOAT, dtype=np.float32)
        output_u16 = np.full(u16_capacity, SENTINEL_U16, dtype=np.uint16)
        metadata = np.full(metadata_capacity, SENTINEL_META, dtype=np.uintp)
        green = np.ascontiguousarray(controls, dtype=np.float32)
        refine = np.asarray([lower, upper], dtype=np.float32)
        count = storage.size if source_sample_count is None else source_sample_count
        status = self.function(
            storage.ctypes.data_as(U16), count, passed_width, height, row_stride,
            green.ctypes.data_as(F32), refine.ctypes.data_as(F32),
            row_origin, column_origin,
            output_float.ctypes.data_as(F32), float_capacity,
            output_u16.ctypes.data_as(U16), u16_capacity,
            metadata.ctypes.data_as(SIZE_PTR), metadata_capacity,
        )
        return status, output_float, output_u16, metadata


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    if request.param == "source":
        missing = [str(path) for path in (BRIDGE_CPP, *SOURCES) if not path.exists()]
        assert not missing, "compose source bridge inputs are missing: " + ", ".join(missing)
        linked_files = SOURCES
    else:
        missing = [str(path) for path in OFFICIAL_OBJECTS if not path.exists()]
        assert not missing, (
            "formal imprint_renderer core3 compose objects are missing; rebuild the "
            "renderer before running official_object cases: " + ", ".join(missing))
        stale = [str(obj) for obj, source in zip(OFFICIAL_OBJECTS, SOURCES)
                 if obj.stat().st_mtime_ns < source.stat().st_mtime_ns]
        assert not stale, (
            "formal imprint_renderer core3 compose objects are stale; rebuild the "
            "renderer before running official_object cases: " + ", ".join(stale))
        linked_files = OFFICIAL_OBJECTS

    target = tmp_path_factory.mktemp("core3_compose_" + request.param) / (
        "libcore3_compose" + SHARED_SUFFIX)
    return Bridge(_compile_bridge(target, linked_files))


def _tile(height: int, width: int, padding: int, seed: int) -> np.ndarray:
    stride = width + padding
    assert stride % 8 == 0
    rng = np.random.default_rng(seed)
    storage = np.full((height, stride), np.uint16(0xA55A), dtype=np.uint16)
    storage[:, :width] = rng.integers(0, 65536, size=(height, width), dtype=np.uint16)
    return storage


def _run_and_compare(bridge, storage, width, *, controls, lower, upper,
                     row_origin, column_origin):
    before = storage.copy()
    expected = oracle.reference(
        storage, width=width, green_parameters=controls, lower=lower, upper=upper,
        row_origin_parity=row_origin, column_origin_parity=column_origin)
    status, actual_float, actual_u16, metadata = bridge.call(
        storage, width, controls=controls, lower=lower, upper=upper,
        row_origin=row_origin, column_origin=column_origin)
    assert status == OK
    count = expected["sample_count"]
    float_rgb = actual_float[:count].reshape(expected["rgb_float"].shape)
    u16_rgb = actual_u16[:count].reshape(expected["rgb_u16"].shape)
    assert np.isfinite(float_rgb).all()
    np.testing.assert_array_equal(
        float_rgb.view(np.uint32), expected["rgb_float"].view(np.uint32),
        err_msg="compact float RGB bits differ from independent composition oracle")
    np.testing.assert_array_equal(
        u16_rgb, expected["rgb_u16"],
        err_msg="compact uint16 RGB differs from independent composition oracle")
    assert metadata[:9].tolist() == [
        expected["input_width"], expected["input_height"],
        *expected["valid_region"], expected["output_width"],
        expected["output_height"], expected["sample_count"],
    ]
    assert np.all(actual_float[count:] == SENTINEL_FLOAT)
    assert np.all(actual_u16[count:] == SENTINEL_U16)
    assert np.all(metadata[9:] == SENTINEL_META)
    np.testing.assert_array_equal(storage, before, err_msg="composition modified source/padding")
    return float_rgb, u16_rgb, expected


def _constant_rgb_mosaic(height, width, row_origin, column_origin,
                         values=(10000, 20000, 40000), padding=8):
    storage = np.full((height, width + padding), np.uint16(0xE17E), dtype=np.uint16)
    for y in range(height):
        for x in range(width):
            global_row = (row_origin + y) & 1
            global_column = (column_origin + x) & 1
            if global_row == 0 and global_column == 0:
                value = values[0]
            elif global_row == 1 and global_column == 1:
                value = values[2]
            else:
                value = values[1]
            storage[y, x] = value
    return storage


def _canonical_random_rggb_tile(height, width, padding, seed):
    """Make one deterministic random mosaic whose global origin is RGGB (0, 0)."""
    storage = np.full((height, width + padding), np.uint16(0xE17E), dtype=np.uint16)
    rng = np.random.default_rng(seed)
    ranges = ((7000, 60001), (3000, 50001), (1000, 42001))
    for y in range(height):
        for x in range(width):
            if (y & 1) == 0 and (x & 1) == 0:
                channel = 0  # R in canonical RGGB.
            elif (y & 1) == 1 and (x & 1) == 1:
                channel = 2  # B in canonical RGGB.
            else:
                channel = 1  # G sites.
            low, high = ranges[channel]
            storage[y, x] = np.uint16(rng.integers(low, high))
    return storage


@pytest.mark.parametrize("controls", [
    (25.0, 0.03, 0.0, 0.2),
    (25.0, 0.03, 0.7, 0.2),
], ids=["m-zero", "m-nonzero"])
def test_random_strided_tiles_match_independent_six_kernel_oracle(bridge, controls):
    cases = [(29, 32, 8, 410), (34, 40, 16, 920)]
    for height, width, padding, seed in cases:
        storage = _tile(height, width, padding, seed + int(controls[2] * 100))
        for row_origin in (0, 1):
            for column_origin in (0, 1):
                _run_and_compare(
                    bridge, storage, width, controls=controls, lower=0.12, upper=0.09,
                    row_origin=row_origin, column_origin=column_origin)


@pytest.mark.parametrize("m,height,width", [(0.0, 29, 32), (0.7, 33, 40)])
def test_constant_rgb_bayer_mosaics_remain_strictly_constant(
        bridge, m, height, width):
    values = np.asarray([10000, 20000, 40000], dtype=np.uint16)
    controls = (25.0, 0.03, m, 0.2)
    for row_origin in (0, 1):
        for column_origin in (0, 1):
            storage = _constant_rgb_mosaic(
                height, width, row_origin, column_origin, values=values)
            _, actual, expected = _run_and_compare(
                bridge, storage, width, controls=controls, lower=0.4, upper=0.4,
                row_origin=row_origin, column_origin=column_origin)
            repeated = np.broadcast_to(values, actual.shape)
            np.testing.assert_array_equal(actual, repeated)
            scale = np.asarray([0x37800080], dtype=np.uint32).view(np.float32)[0]
            expected_float = np.multiply(values.astype(np.float32), scale,
                                         dtype=np.float32)
            np.testing.assert_array_equal(
                expected["rgb_float"], np.broadcast_to(expected_float, actual.shape))


@pytest.mark.parametrize("m,height,width", [(0.0, 17, 20), (0.7, 25, 28)])
def test_minimum_dimensions_return_exact_compact_roi(bridge, m, height, width):
    controls = (25.0, 0.03, m, 0.2)
    storage = _tile(height, width, 4, int(1500 + width))
    _, _, expected = _run_and_compare(
        bridge, storage, width, controls=controls, lower=0.2, upper=0.2,
        row_origin=1, column_origin=0)
    margin = 8 if m == 0.0 else 12
    assert expected["valid_region"] == (margin, margin, height - margin, width - margin)
    assert expected["output_height"] == height - 2 * margin
    assert expected["output_width"] == width - 2 * margin


@pytest.mark.parametrize("kind,m", [("grayscale", 0.7), ("ramp", 0.0), ("impulse", 0.7)])
def test_grayscale_ramp_and_impulse_patterns_match_oracle(bridge, kind, m):
    height, width, padding = (33, 40, 8) if m else (29, 32, 8)
    storage = np.full((height, width + padding), np.uint16(0xCC33), dtype=np.uint16)
    if kind == "grayscale":
        storage[:, :width] = np.uint16(23456)
    elif kind == "ramp":
        y, x = np.indices((height, width), dtype=np.uint32)
        storage[:, :width] = ((1200 + 37 * x + 53 * y) % 65536).astype(np.uint16)
    else:
        storage[:, :width] = np.uint16(28000)
        storage[height // 2, width // 2] = np.uint16(65535)
    controls = (25.0, 0.03, m, 0.2)
    _, actual_u16, _ = _run_and_compare(
        bridge, storage, width, controls=controls, lower=0.3, upper=0.3,
        row_origin=1, column_origin=1)
    if kind == "grayscale":
        assert np.all(actual_u16 == np.uint16(23456))


@pytest.mark.parametrize("m", [0.0, 0.7], ids=["m-zero", "m-nonzero"])
def test_compact_output_is_independent_of_discarded_tile_halo(bridge, m):
    large_height, large_width = 49, 64
    large = _canonical_random_rggb_tile(large_height, large_width, 8, 88031)
    crop_origins = ((8, 8), (9, 9))
    controls = (25.0, 0.03, m, 0.2)
    lower, upper = 0.12, 0.09

    # Evaluate the large tile and two interior crops as independent calls.
    large_float, large_u16, large_expected = _run_and_compare(
        bridge, large, large_width, controls=controls, lower=lower, upper=upper,
        row_origin=0, column_origin=0)
    records = [(large_float, large_u16, large_expected, (0, 0))]
    for origin_y, origin_x in crop_origins:
        crop_height, crop_width = 33, 40
        crop = np.full((crop_height, 48), np.uint16(0xBEEF), dtype=np.uint16)
        crop[:, :crop_width] = large[
            origin_y:origin_y + crop_height,
            origin_x:origin_x + crop_width]
        crop_float, crop_u16, crop_expected = _run_and_compare(
            bridge, crop, crop_width, controls=controls, lower=lower, upper=upper,
            row_origin=origin_y & 1, column_origin=origin_x & 1)
        records.append((crop_float, crop_u16, crop_expected,
                        (origin_y, origin_x)))

    global_rects = []
    for _, _, expected, (origin_y, origin_x) in records:
        top, left, bottom, right = expected["valid_region"]
        global_rects.append((origin_y + top, origin_x + left,
                             origin_y + bottom, origin_x + right))
    common_top = max(rect[0] for rect in global_rects)
    common_left = max(rect[1] for rect in global_rects)
    common_bottom = min(rect[2] for rect in global_rects)
    common_right = min(rect[3] for rect in global_rects)
    assert common_top < common_bottom and common_left < common_right
    margin = 8 if m == 0.0 else 12
    assert common_bottom - common_top == 33 - 2 * margin - 1
    assert common_right - common_left == 40 - 2 * margin - 1

    common_float_bits = []
    common_u16 = []
    for rgb_float, rgb_u16, expected, (origin_y, origin_x) in records:
        roi_top, roi_left, _, _ = expected["valid_region"]
        local_top = common_top - (origin_y + roi_top)
        local_left = common_left - (origin_x + roi_left)
        local_bottom = local_top + (common_bottom - common_top)
        local_right = local_left + (common_right - common_left)
        common_float_bits.append(
            rgb_float[local_top:local_bottom, local_left:local_right].view(np.uint32))
        common_u16.append(rgb_u16[local_top:local_bottom, local_left:local_right])

    for candidate in common_float_bits[1:]:
        np.testing.assert_array_equal(
            candidate, common_float_bits[0],
            err_msg="valid float RGB depends on discarded input halo")
    for candidate in common_u16[1:]:
        np.testing.assert_array_equal(
            candidate, common_u16[0],
            err_msg="valid uint16 RGB depends on discarded input halo")


def test_failures_and_short_output_capacities_leave_all_destinations_untouched(bridge):
    width, height = 32, 29
    storage = _tile(height, width, 8, 3501)
    controls = (25.0, 0.03, 0.0, 0.2)
    before = storage.copy()
    failures = [
        {"passed_width": 19},
        {"row_origin": 2},
        {"column_origin": 2},
        {"source_sample_count": (height - 1) * storage.shape[1] + width - 1},
        {"lower": -0.1},
        {"upper": np.nan},
        {"float_capacity": 1},
        {"u16_capacity": 1},
        {"metadata_capacity": 8},
    ]
    for overrides in failures:
        status, output_float, output_u16, metadata = bridge.call(
            storage, width, controls=controls, **overrides)
        assert status != OK, f"unexpected success for {overrides}"
        if overrides in ({"float_capacity": 1}, {"u16_capacity": 1},
                         {"metadata_capacity": 8}):
            assert status == BUFFER_TOO_SMALL
        assert np.all(output_float == SENTINEL_FLOAT)
        assert np.all(output_u16 == SENTINEL_U16)
        assert np.all(metadata == SENTINEL_META)
    np.testing.assert_array_equal(storage, before)
