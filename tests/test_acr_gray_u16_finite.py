"""Bitwise tests for finite-domain expansion in ACR gray and U16 handlers."""

from __future__ import annotations

import ctypes as C
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
GRAY_SOURCE = NATIVE / "src/acr_gray.cpp"
GRAY_HEADER = NATIVE / "src/acr_gray.hpp"
U16_SOURCE = NATIVE / "src/acr_u16_store.cpp"
U16_HEADER = NATIVE / "src/acr_u16_store.hpp"
BRIDGE_SOURCE = NATIVE / "tools/acr_gray_u16_finite_bridge.cpp"
GRAY_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_gray.cpp.o"
U16_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_u16_store.cpp.o"

sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import gray_u16_finite_oracle as oracle  # noqa: E402


FLOAT = C.c_float
FLOAT_PTR = C.POINTER(FLOAT)
U16 = C.c_uint16
U16_PTR = C.POINTER(U16)
SIZE = C.c_size_t

OK = 0
INVALID_ARGUMENT = 1
INVALID_RGB = 3
GRAY_OVERFLOW = 4
INVALID_GRAY = 2
SENTINEL_F32 = np.uint32(0x7FC12345).view(np.float32)
SENTINEL_U16 = np.uint16(0xD39A)
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _float_ptr(array: np.ndarray):
    return array.ctypes.data_as(FLOAT_PTR)


def _u16_ptr(array: np.ndarray):
    return array.ctypes.data_as(U16_PTR)


class Bridge:
    def __init__(self, library_path: Path):
        self.library = C.CDLL(str(library_path))
        self.apply_gray_fn = self.library.acr_test_gray_u16_finite_apply_gray
        self.apply_gray_fn.argtypes = [
            FLOAT_PTR, FLOAT_PTR, SIZE, U16, U16, U16,
        ]
        self.apply_gray_fn.restype = C.c_int

        self.store_fn = self.library.acr_test_gray_u16_finite_store
        self.store_fn.argtypes = [
            FLOAT_PTR, U16_PTR, SIZE, C.c_uint32, C.c_uint32,
            C.c_int32, C.c_int32, C.c_int,
        ]
        self.store_fn.restype = C.c_int

        self.copy_tile_fn = self.library.acr_test_gray_u16_finite_copy_tile
        self.copy_tile_fn.argtypes = [U16_PTR, SIZE]
        self.copy_tile_fn.restype = C.c_int

    def apply_gray(self, rgb: np.ndarray, lanes=(16384, 0, 0), output=None):
        source = np.ascontiguousarray(rgb, dtype=np.float32).reshape(-1)
        if output is None:
            output = np.full(source.size // 3, SENTINEL_F32, dtype=np.float32)
        status = self.apply_gray_fn(
            _float_ptr(source), _float_ptr(output), source.size // 3,
            *(int(lane) for lane in lanes),
        )
        return status, output, source

    def store(self, gray: np.ndarray, width: int, height: int,
              origin_x=0, origin_y=0, dither=False, output=None):
        source = np.ascontiguousarray(gray, dtype=np.float32).reshape(-1)
        if output is None:
            output = np.full(source.size, SENTINEL_U16, dtype=np.uint16)
        status = self.store_fn(
            _float_ptr(source), _u16_ptr(output), source.size, width, height,
            origin_x, origin_y, int(dither),
        )
        return status, output, source

    def copy_tile(self):
        output = np.full(oracle.TILE_SIZE, SENTINEL_U16, dtype=np.uint16)
        status = self.copy_tile_fn(_u16_ptr(output), output.size)
        return status, output.reshape(oracle.TILE_SIDE, oracle.TILE_SIDE)


def _bridge_inputs(kind: str) -> list[Path]:
    if kind == "source":
        return [BRIDGE_SOURCE, GRAY_SOURCE, U16_SOURCE]
    if kind == "formal_object":
        for obj, source, header in (
            (GRAY_OBJECT, GRAY_SOURCE, GRAY_HEADER),
            (U16_OBJECT, U16_SOURCE, U16_HEADER),
        ):
            assert obj.is_file(), f"build imprint_renderer first: missing {obj}"
            assert obj.stat().st_mtime_ns >= max(
                source.stat().st_mtime_ns, header.stat().st_mtime_ns
            ), f"rebuild imprint_renderer: stale official object {obj.name}"
        return [BRIDGE_SOURCE, GRAY_OBJECT, U16_OBJECT]
    raise AssertionError(
        "ACR_GRAY_U16_FINITE_BRIDGE must be source or formal_object"
    )


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    kind = os.environ.get("ACR_GRAY_U16_FINITE_BRIDGE", "source")
    inputs = _bridge_inputs(kind)
    assert BRIDGE_SOURCE.is_file(), f"missing gray/U16 bridge: {BRIDGE_SOURCE}"
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the gray/U16 bridge"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_gray_u16_finite_" + kind) / (
        "libgrayu16finite" + suffix
    )
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I",
        str(NATIVE / "src"),
        *(str(path) for path in inputs),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return Bridge(output)


def _gray_expected(rgb: np.ndarray, lanes) -> np.ndarray:
    flat = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    return np.asarray(
        [oracle.gray_sample(tuple(pixel), tuple(lanes)) for pixel in flat],
        dtype=np.float32,
    )


def _assert_f32_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32).view(np.uint32),
        np.asarray(expected, dtype=np.float32).view(np.uint32),
    )


def test_gray_legacy_unit_domain_matches_independent_fmaf_oracle(bridge):
    rng = np.random.default_rng(20261024)
    rgb = rng.random((257, 3), dtype=np.float32)
    lanes = (4899, 9617, 1868)
    status, actual, _ = bridge.apply_gray(rgb, lanes)
    assert status == OK
    _assert_f32_bits_equal(actual, _gray_expected(rgb, lanes))


def test_gray_accepts_finite_negative_and_overrange_values_without_clamping(bridge):
    lanes = (16384, 8192, 4096)
    rgb = np.asarray([
        [-2.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-0.25, 3.0, 0.125],
        [1.5, -2.0, 4.0],
    ], dtype=np.float32)
    status, actual, _ = bridge.apply_gray(rgb, lanes)
    expected = _gray_expected(rgb, lanes)
    assert status == OK
    _assert_f32_bits_equal(actual, expected)
    assert actual[0] == np.float32(-2.0)
    assert actual[1] == np.float32(2.5)
    assert np.any(actual < 0.0)
    assert np.any(actual > 1.0)


def test_gray_preserves_signed_zero_bits_from_ordered_fma_chain(bridge):
    rgb = np.full((4, 3), np.float32(-0.0), dtype=np.float32)
    lanes = (16384, 16384, 16384)
    expected = _gray_expected(rgb, lanes)
    assert np.all(np.signbit(expected))
    status, actual, _ = bridge.apply_gray(rgb, lanes)
    assert status == OK
    _assert_f32_bits_equal(actual, expected)


def test_gray_nonfinite_and_overflow_failures_leave_entire_output_untouched(bridge):
    rgb = np.asarray([
        [0.25, 0.5, 0.75],
        [0.5, np.nan, 0.25],
    ], dtype=np.float32)
    output = np.full(2, SENTINEL_F32, dtype=np.float32)
    before = output.view(np.uint32).copy()
    status, actual, _ = bridge.apply_gray(rgb, (16384, 8192, 4096), output)
    assert status == INVALID_RGB
    np.testing.assert_array_equal(actual.view(np.uint32), before)

    maximum = np.finfo(np.float32).max
    overflow_rgb = np.asarray([
        [0.25, 0.5, 0.75],
        [maximum, 0.0, 0.0],
    ], dtype=np.float32)
    output.fill(SENTINEL_F32)
    before = output.view(np.uint32).copy()
    status, actual, _ = bridge.apply_gray(overflow_rgb, (65535, 0, 0), output)
    assert status == GRAY_OVERFLOW
    np.testing.assert_array_equal(actual.view(np.uint32), before)


def test_gray_alias_rejection_is_atomic(bridge):
    backing = np.arange(6, dtype=np.float32)
    before = backing.view(np.uint8).copy()
    output_alias = backing[:2]
    status = bridge.apply_gray_fn(
        _float_ptr(backing), _float_ptr(output_alias), 2, 16384, 0, 0
    )
    assert status == INVALID_ARGUMENT
    np.testing.assert_array_equal(backing.view(np.uint8), before)


def test_tile_matches_independent_direct_modulo_park_miller_recurrence(bridge):
    expected = oracle.park_miller_tile()
    assert int(expected.min()) >= oracle.MINIMUM_ACCEPTED_LOW_WORD
    status, actual = bridge.copy_tile()
    assert status == OK
    np.testing.assert_array_equal(actual, expected)


def test_u16_legacy_unit_domain_and_finite_out_of_range_counterexamples(bridge):
    values = np.asarray([
        0.0, 0.5 / 65535.0, 0.5, 1.0,
        -1.0, -0.25, -0.0001, 1.0001, 1.1, 2.0,
    ], dtype=np.float32)
    width = values.size
    status, no_dither, _ = bridge.store(values, width, 1, dither=False)
    assert status == OK
    expected_no_dither = oracle.apply_gray_u16(
        values, width, 1, 0, 0, dither=False
    ).reshape(-1)
    np.testing.assert_array_equal(no_dither, expected_no_dither)

    status, dithered, _ = bridge.store(values, width, 1, dither=True)
    assert status == OK
    expected_dithered = oracle.apply_gray_u16(
        values, width, 1, 0, 0, dither=True
    ).reshape(-1)
    np.testing.assert_array_equal(dithered, expected_dithered)

    # The no-dither handler wraps the low 16 bits; the dither handler clamps
    # its signed conversion result when the unsigned compare selects the cap.
    assert no_dither[4] != 0 and dithered[4] == 0
    assert no_dither[8] != 65535 and dithered[8] == 65535
    assert no_dither[3] == 65535 and dithered[3] == 65535


def test_u16_legacy_unit_domain_matches_all_normalized_u16_codes(bridge):
    values = (np.arange(65536, dtype=np.float64) / 65535.0).astype(np.float32)
    width = 256
    height = values.size // width
    for dither in (False, True):
        status, actual, _ = bridge.store(values, width, height, dither=dither)
        assert status == OK
        expected = oracle.apply_gray_u16(
            values, width, height, 0, 0, dither=dither
        ).reshape(-1)
        np.testing.assert_array_equal(actual, expected)


def test_u16_signed_zero_matches_fused_conversion_for_both_handlers(bridge):
    values = np.asarray([-0.0, 0.0, -0.0, 0.0], dtype=np.float32)
    for dither in (False, True):
        status, actual, _ = bridge.store(
            values, values.size, 1, origin_x=-7, origin_y=13, dither=dither
        )
        assert status == OK
        expected = oracle.apply_gray_u16(
            values, values.size, 1, -7, 13, dither=dither
        ).reshape(-1)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "width,origin_x,origin_y",
    [
        (31, 0, 0),
        (32, -19, 257),
        (33, INT32_MIN, INT32_MAX),
        (64, INT32_MAX, INT32_MIN),
    ],
)
@pytest.mark.parametrize("dither", [False, True], ids=("no-dither", "dither"))
def test_u16_scalar_vector_boundary_widths_and_signed_origins(
    bridge, width, origin_x, origin_y, dither
):
    height = 3
    values = np.linspace(-0.25, 1.25, width * height, dtype=np.float64).astype(
        np.float32
    )
    status, actual, _ = bridge.store(
        values, width, height, origin_x, origin_y, dither
    )
    assert status == OK
    expected = oracle.apply_gray_u16(
        values, width, height, origin_x, origin_y, dither=dither
    )
    np.testing.assert_array_equal(actual.reshape(height, width), expected)


def _boundary_neighbors(value: np.float32, direction: float, count: int):
    result = []
    current = value
    for _ in range(count):
        current = np.nextafter(current, np.float32(direction), dtype=np.float32)
        result.append(current)
    return result


def test_u16_scaled_signed_int32_boundaries_and_floatmax_infinities(bridge):
    positive_boundary = np.float32((2.0 ** 31) / 65535.0)
    negative_boundary = np.float32(-(2.0 ** 31) / 65535.0)
    values = np.asarray([
        *_boundary_neighbors(positive_boundary, -np.inf, 3),
        positive_boundary,
        *_boundary_neighbors(positive_boundary, np.inf, 3),
        *_boundary_neighbors(negative_boundary, np.inf, 3),
        negative_boundary,
        *_boundary_neighbors(negative_boundary, -np.inf, 3),
        np.finfo(np.float32).max,
        -np.finfo(np.float32).max,
    ], dtype=np.float32)

    positive_scaled = [oracle.fma32(value, oracle.U16_SCALE, oracle.U16_HALF)
                       for value in values[:7]]
    negative_start = 7
    negative_scaled = [oracle.fma32(value, oracle.U16_SCALE, oracle.U16_HALF)
                       for value in values[negative_start:negative_start + 7]]
    assert any(value < np.float32(2.0 ** 31) for value in positive_scaled)
    assert any(value >= np.float32(2.0 ** 31) for value in positive_scaled)
    assert any(value > np.float32(-(2.0 ** 31)) for value in negative_scaled)
    assert any(value <= np.float32(-(2.0 ** 31)) for value in negative_scaled)
    assert np.isposinf(oracle.fma32(values[-2], oracle.U16_SCALE, oracle.U16_HALF))
    assert np.isneginf(oracle.fma32(values[-1], oracle.U16_SCALE, oracle.U16_HALF))

    for dither in (False, True):
        status, actual, _ = bridge.store(
            values, values.size, 1, origin_x=-128, origin_y=INT32_MAX,
            dither=dither,
        )
        assert status == OK
        expected = oracle.apply_gray_u16(
            values, values.size, 1, -128, INT32_MAX, dither=dither
        ).reshape(-1)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_u16_nonfinite_input_rejection_is_atomic(bridge, bad_value):
    values = np.asarray([0.25, 0.75, bad_value, 0.5], dtype=np.float32)
    for dither in (False, True):
        output = np.full(values.size, SENTINEL_U16, dtype=np.uint16)
        before = output.copy()
        status, actual, _ = bridge.store(
            values, values.size, 1, dither=dither, output=output
        )
        assert status == INVALID_GRAY
        np.testing.assert_array_equal(actual, before)


def test_u16_alias_rejection_is_atomic(bridge):
    backing = np.asarray([0.125, 0.25, 0.5, 0.75], dtype=np.float32)
    before = backing.view(np.uint8).copy()
    output_alias = backing.view(np.uint16)
    status = bridge.store_fn(
        _float_ptr(backing), _u16_ptr(output_alias), backing.size,
        backing.size, 1, 0, 0, 0,
    )
    assert status == INVALID_ARGUMENT
    np.testing.assert_array_equal(backing.view(np.uint8), before)
