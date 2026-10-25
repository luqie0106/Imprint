from __future__ import annotations

import ctypes
import importlib.util
import shutil
import sys
import subprocess
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
OK = 0
BUFFER_TOO_SMALL = 4
SENTINEL_FLOAT = np.float32(-1234.5)
SENTINEL_U16 = np.uint16(0xD00D)
SENTINEL_META = np.iinfo(np.uintp).max
U16_SCALE = np.asarray([0x37800080], dtype=np.uint32).view(np.float32)[0]


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
        "core3 float compose bridge link failed:\n" +
        completed.stdout + completed.stderr)
    return target


class Bridge:
    def __init__(self, path: Path):
        self.library = ctypes.CDLL(str(path))
        self.float_function = self.library.acr_test_core3_compose_f32
        self.u16_function = self.library.acr_test_core3_compose
        signature = [
            ctypes.c_void_p, SIZE, SIZE, SIZE, SIZE, F32, F32,
            ctypes.c_uint32, ctypes.c_uint32, F32, SIZE, U16, SIZE,
            SIZE_PTR, SIZE,
        ]
        self.float_function.argtypes = signature
        self.float_function.restype = ctypes.c_int
        self.u16_function.argtypes = signature
        self.u16_function.restype = ctypes.c_int

    def call(self, storage: np.ndarray, width: int, *, input_kind="f32",
             controls=(25.0, 0.03, 0.0, 0.2), lower=0.4, upper=0.4,
             row_origin=0, column_origin=0,
             source_sample_count: int | None = None,
             float_capacity: int | None = None,
             u16_capacity: int | None = None,
             metadata_capacity: int = 12):
        height, row_stride = storage.shape
        margin = 8 if controls[2] == 0 else 12
        expected_count = max(0, height - 2 * margin) * max(
            0, width - 2 * margin) * 3
        float_capacity = expected_count + 8 if float_capacity is None else float_capacity
        u16_capacity = expected_count + 8 if u16_capacity is None else u16_capacity
        output_float = np.full(float_capacity, SENTINEL_FLOAT, dtype=np.float32)
        output_u16 = np.full(u16_capacity, SENTINEL_U16, dtype=np.uint16)
        metadata = np.full(metadata_capacity, SENTINEL_META, dtype=np.uintp)
        green = np.ascontiguousarray(controls, dtype=np.float32)
        refine = np.asarray([lower, upper], dtype=np.float32)
        required_count = (height - 1) * row_stride + width
        count = required_count if source_sample_count is None else source_sample_count
        if input_kind == "f32":
            assert storage.dtype == np.float32
            function = self.float_function
            source_pointer = storage.ctypes.data_as(F32)
        elif input_kind == "u16":
            assert storage.dtype == np.uint16
            function = self.u16_function
            source_pointer = storage.ctypes.data_as(U16)
        else:
            raise ValueError(f"unknown input kind: {input_kind}")
        status = function(
            source_pointer, count, width, height, row_stride,
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

    target = tmp_path_factory.mktemp("core3_float_compose_" + request.param) / (
        "libcore3_float_compose" + SHARED_SUFFIX)
    return Bridge(_compile_bridge(target, linked_files))


def _tile(height: int, width: int, *, pattern: str, seed: int,
          padding: int = 4) -> np.ndarray:
    stride = width + padding
    storage = np.full((height, stride), np.float32(np.nan), dtype=np.float32)
    y, x = np.indices((height, width), dtype=np.float32)
    if pattern == "constant":
        values = np.full((height, width), np.float32(0.25), dtype=np.float32)
    elif pattern == "ramp":
        values = (np.float32(0.12) + np.float32(0.20) * x / (width - 1) +
                  np.float32(0.14) * y / (height - 1)).astype(np.float32)
    elif pattern == "impulse":
        values = np.full((height, width), np.float32(0.22), dtype=np.float32)
        values[height // 2, width // 2] = np.float32(0.7)
    elif pattern == "random":
        values = np.random.default_rng(seed).uniform(
            0.1, 0.6, size=(height, width)).astype(np.float32)
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    storage[:, :width] = values
    return storage


def _compare(bridge, storage, width, *, controls, lower=0.12, upper=0.09,
             row_origin=0, column_origin=0):
    assert storage.ctypes.data % 16 == 0, "float input must be 16-byte aligned"
    before = storage.copy()
    expected = oracle.reference_f32(
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
        err_msg="prepared-float RGB bits differ from independent composition oracle")
    np.testing.assert_array_equal(
        u16_rgb, expected["rgb_u16"],
        err_msg="prepared-float uint16 output differs from independent oracle")
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


@pytest.mark.parametrize("m,height,width", [(0.0, 29, 32), (0.8, 33, 32)],
                         ids=["m-zero", "m-nonzero"])
@pytest.mark.parametrize("pattern", ["constant", "ramp", "impulse", "random"])
@pytest.mark.parametrize("row_origin,column_origin", [
    (0, 0), (0, 1), (1, 0), (1, 1),
])
def test_prepared_float_patterns_match_six_kernel_oracle(
        bridge, m, height, width, pattern, row_origin, column_origin):
    storage = _tile(height, width, pattern=pattern,
                    seed=1700 + int(m * 100) + row_origin * 2 + column_origin)
    assert storage.shape[1] % 4 == 0
    assert storage.shape[1] % 8 != 0
    assert np.isnan(storage[:, width:]).all()
    _compare(
        bridge, storage, width, controls=(25.0, 0.03, m, 0.2),
        row_origin=row_origin, column_origin=column_origin)


@pytest.mark.parametrize("m,height,width,margin", [
    (0.0, 17, 20, 8),
    (0.8, 25, 28, 12),
    (0.0, 29, 32, 8),
    (0.8, 33, 32, 12),
])
def test_minimum_and_regular_float_dimensions_use_compact_roi(
        bridge, m, height, width, margin):
    storage = _tile(height, width, pattern="random", seed=width + height)
    _, _, expected = _compare(
        bridge, storage, width, controls=(25.0, 0.03, m, 0.2),
        row_origin=1, column_origin=0)
    assert expected["valid_region"] == (
        margin, margin, height - margin, width - margin)
    assert expected["output_width"] == width - 2 * margin
    assert expected["output_height"] == height - 2 * margin


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf],
                         ids=["nan", "positive-infinity", "negative-infinity"])
def test_nonfinite_valid_input_rejects_without_changing_destinations(bridge, bad_value):
    width, height = 32, 29
    storage = _tile(height, width, pattern="constant", seed=0)
    storage[height // 2, width // 2] = np.float32(bad_value)
    status, output_float, output_u16, metadata = bridge.call(storage, width)
    assert status != OK
    assert np.all(output_float == SENTINEL_FLOAT)
    assert np.all(output_u16 == SENTINEL_U16)
    assert np.all(metadata == SENTINEL_META)


def test_short_source_and_output_buffers_leave_destinations_untouched(bridge):
    width, height = 32, 29
    storage = _tile(height, width, pattern="random", seed=662)
    required_count = (height - 1) * storage.shape[1] + width
    for overrides in (
        {"source_sample_count": required_count - 1},
        {"float_capacity": 1},
        {"u16_capacity": 1},
        {"metadata_capacity": 8},
    ):
        status, output_float, output_u16, metadata = bridge.call(
            storage, width, **overrides)
        assert status != OK, f"unexpected success for {overrides}"
        if overrides != {"source_sample_count": required_count - 1}:
            assert status == BUFFER_TOO_SMALL
        assert np.all(output_float == SENTINEL_FLOAT)
        assert np.all(output_u16 == SENTINEL_U16)
        assert np.all(metadata == SENTINEL_META)


def test_finite_out_of_range_input_is_not_clipped_before_output_validation(bridge):
    width, height = 32, 29
    storage = np.full((height, width + 4), np.float32(1.1), dtype=np.float32)
    status, output_float, output_u16, metadata = bridge.call(storage, width)
    assert status != OK
    assert np.all(output_float == SENTINEL_FLOAT)
    assert np.all(output_u16 == SENTINEL_U16)
    assert np.all(metadata == SENTINEL_META)


@pytest.mark.parametrize("m", [0.0, 0.8], ids=["m-zero", "m-nonzero"])
@pytest.mark.parametrize("row_origin,column_origin", [
    (0, 0), (0, 1), (1, 0), (1, 1),
])
def test_exact_u16_normalization_matches_legacy_interface_bit_for_bit(
        bridge, m, row_origin, column_origin):
    height, width = (29, 32) if m == 0.0 else (33, 32)
    rng = np.random.default_rng(9500 + row_origin * 2 + column_origin)
    u16 = np.full((height, width + 8), np.uint16(0xA55A), dtype=np.uint16)
    u16[:, :width] = rng.integers(0, 65536, size=(height, width), dtype=np.uint16)
    prepared = np.full((height, width + 4), np.float32(np.nan), dtype=np.float32)
    prepared[:, :width] = np.multiply(
        u16[:, :width].astype(np.float32), U16_SCALE, dtype=np.float32)
    controls = (25.0, 0.03, m, 0.2)
    float_expected = oracle.reference_f32(
        prepared, width=width, green_parameters=controls, lower=0.12, upper=0.09,
        row_origin_parity=row_origin, column_origin_parity=column_origin)
    u16_expected = oracle.reference(
        u16, width=width, green_parameters=controls, lower=0.12, upper=0.09,
        row_origin_parity=row_origin, column_origin_parity=column_origin)
    np.testing.assert_array_equal(
        float_expected["rgb_float"].view(np.uint32),
        u16_expected["rgb_float"].view(np.uint32))
    np.testing.assert_array_equal(float_expected["rgb_u16"], u16_expected["rgb_u16"])

    status_f, output_f, output_f_u16, _ = bridge.call(
        prepared, width, input_kind="f32", controls=controls, lower=0.12,
        upper=0.09, row_origin=row_origin, column_origin=column_origin)
    status_u, output_u, output_u_u16, _ = bridge.call(
        u16, width, input_kind="u16", controls=controls, lower=0.12,
        upper=0.09, row_origin=row_origin, column_origin=column_origin)
    assert status_f == status_u == OK
    count = float_expected["sample_count"]
    np.testing.assert_array_equal(output_f[:count].view(np.uint32),
                                  output_u[:count].view(np.uint32))
    np.testing.assert_array_equal(output_f_u16[:count], output_u_u16[:count])


def test_sub_u16_float_bits_reach_native_output_without_quantization(bridge):
    width, height = 32, 29
    storage = np.full((height, width + 4), np.float32(np.nan), dtype=np.float32)
    storage[:, :width] = np.float32(0.25)
    storage[::2, ::2] += np.float32(2**-20)
    codes = np.rint(storage[:, :width] * np.float32(65535.0)).astype(np.uint16)
    assert np.all(codes == np.uint16(16384)), "canary must collapse to one u16 value"
    quantized = np.full_like(storage, np.float32(np.nan))
    quantized[:, :width] = np.multiply(codes.astype(np.float32), U16_SCALE,
                                      dtype=np.float32)
    controls = (25.0, 0.03, 0.0, 0.2)
    direct = oracle.reference_f32(storage, width=width, green_parameters=controls)
    round_trip = oracle.reference_f32(quantized, width=width,
                                      green_parameters=controls)
    assert np.any(direct["rgb_float"].view(np.uint32) !=
                  round_trip["rgb_float"].view(np.uint32))

    _, actual, _, _ = bridge.call(storage, width, controls=controls)
    count = direct["sample_count"]
    np.testing.assert_array_equal(
        actual[:count].view(np.uint32), direct["rgb_float"].reshape(-1).view(np.uint32))
    assert np.any(actual[:count].view(np.uint32) !=
                  round_trip["rgb_float"].reshape(-1).view(np.uint32)), (
        "native output matches the direct float reference and differs from a u16 round trip")
