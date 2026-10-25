"""Bitwise mode2 post-camera-black tests for source and official objects."""

from __future__ import annotations

import ctypes as C
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_postcamera_black.cpp"
HEADER = SOURCE.with_suffix(".hpp")
LUT_SOURCE = NATIVE / "src/acr_float_lut.cpp"
LUT_HEADER = LUT_SOURCE.with_suffix(".hpp")
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_postcamera_black.cpp.o"
LUT_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_float_lut.cpp.o"
BRIDGE_SOURCE = NATIVE / "tools/acr_postcamera_black_bridge.cpp"
sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import postcamera_black_oracle as oracle  # noqa: E402


FLOAT = C.c_float
FLOAT_PTR = C.POINTER(FLOAT)
SIZE = C.c_size_t
SIZE_MAX = C.c_size_t(-1).value
SENTINEL = np.float32(-1234.5)

OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
INVALID_STRIDE = 3
BUFFER_TOO_SMALL = 4
UNSUPPORTED_ALIGNMENT = 5
OVERLAP = 6
ARITHMETIC_OVERFLOW = 7
RESOURCE_LIMIT = 8
NON_FINITE_INPUT = 9
INVALID_BLACK = 10
OUT_OF_DOMAIN = 11

BLACKS = (
    4100 / 65535,
    0x404 / 65535,
    float(np.nextafter(np.float64(0.0), np.float64(1.0))),
    float(np.nextafter(np.float64(1.0), np.float64(0.0))),
)


def _float_ptr(array):
    return array.ctypes.data_as(FLOAT_PTR)


def _null_float_ptr():
    return C.cast(C.c_void_p(0), FLOAT_PTR)


def _bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


class Bridge:
    def __init__(self, path):
        self.library = C.CDLL(str(path))
        self.apply_fn = self.library.acr_test_postcamera_black
        self.apply_fn.argtypes = [
            FLOAT_PTR, SIZE, SIZE, SIZE, SIZE, C.c_double,
            FLOAT_PTR, SIZE, SIZE,
        ]
        self.apply_fn.restype = C.c_int
        self.build_fn = self.library.acr_test_postcamera_black_lut
        self.build_fn.argtypes = [C.c_double, FLOAT_PTR, SIZE]
        self.build_fn.restype = C.c_int

    def apply_raw(self, src, src_count, src_stride, height, row_samples,
                  black, dst, dst_count, dst_stride, *, null_source=False):
        source_pointer = _null_float_ptr() if null_source else _float_ptr(src)
        return self.apply_fn(
            source_pointer, src_count, src_stride, height, row_samples, black,
            _float_ptr(dst), dst_count, dst_stride,
        )

    def build_lut(self, black, count=oracle.TABLE_SIZE):
        output = np.full(oracle.TABLE_SIZE, SENTINEL, dtype=np.float32)
        status = self.build_fn(black, _float_ptr(output), count)
        return status, output


def _bridge_inputs(kind):
    u16_source = NATIVE / "src/acr_u16_lut.cpp"
    u16_header = u16_source.with_suffix(".hpp")
    u16_object = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_u16_lut.cpp.o"
    if kind == "source":
        return [BRIDGE_SOURCE, SOURCE, LUT_SOURCE, u16_source]
    for obj, source, header in (
        (OBJECT, SOURCE, HEADER),
        (LUT_OBJECT, LUT_SOURCE, LUT_HEADER),
        (u16_object, u16_source, u16_header),
    ):
        assert obj.is_file(), f"rebuild renderer: missing official object {obj}"
        assert obj.stat().st_mtime_ns >= max(source.stat().st_mtime_ns, header.stat().st_mtime_ns), (
            f"rebuild renderer: stale official object {obj.name}"
        )
    return [BRIDGE_SOURCE, OBJECT, LUT_OBJECT, u16_object]


@pytest.fixture(scope="module", params=("source", "official_object"),
                ids=("source", "official_object"))
def bridge(request, tmp_path_factory):
    inputs = _bridge_inputs(request.param)
    assert BRIDGE_SOURCE.is_file(), f"missing post-camera-black bridge: {BRIDGE_SOURCE}"
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the post-camera-black bridge"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_postcamera_black_" + request.param) / ("libpostblack" + suffix)
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


@pytest.mark.parametrize("black", BLACKS, ids=("b16-4100", "b16-1028", "near-zero", "near-one"))
def test_all_4098_table_entries_match_independent_fma_oracle(bridge, black):
    status, actual = bridge.build_lut(black)
    assert status == OK
    expected = oracle.build_lut(black)
    _assert_bits(actual, expected)
    _assert_bits(actual[-2:], expected[-2:])


@pytest.mark.parametrize("black", BLACKS[:2], ids=("b16-4100", "b16-1028"))
def test_grid_neighbors_kink_endpoints_and_random_samples_match(bridge, black):
    table_status, table = bridge.build_lut(black)
    assert table_status == OK

    grid = np.arange(oracle.SEGMENTS + 1, dtype=np.float32) / np.float32(oracle.SEGMENTS)
    probes = [
        np.nextafter(grid, np.float32(-np.inf)),
        grid,
        np.nextafter(grid, np.float32(np.inf)),
        np.asarray([0.0, -0.0, 1.0, np.float32(black)], dtype=np.float32),
        np.asarray([
            np.nextafter(np.float32(black), np.float32(-np.inf)),
            np.nextafter(np.float32(black), np.float32(np.inf)),
        ], dtype=np.float32),
    ]
    pivot_segment = min(oracle.SEGMENTS - 1, int(black * oracle.SEGMENTS))
    pivot_left = np.float32(pivot_segment / oracle.SEGMENTS)
    pivot_right = np.float32((pivot_segment + 1) / oracle.SEGMENTS)
    probes.append(np.asarray([
        pivot_left,
        np.nextafter(pivot_left, np.float32(np.inf)),
        np.nextafter(pivot_right, np.float32(-np.inf)),
        pivot_right,
        np.float32((float(pivot_left) + float(pivot_right)) * 0.5),
    ], dtype=np.float32))
    rng = np.random.default_rng(20261021 + int(black * 65535))
    probes.append(rng.random(4096, dtype=np.float32))
    samples = np.concatenate(probes)
    samples = samples[(samples >= 0.0) & (samples <= 1.0)]

    destination = np.full(samples.size, SENTINEL, dtype=np.float32)
    status = bridge.apply_raw(
        samples, samples.size, samples.size, 1, samples.size, black,
        destination, destination.size, destination.size,
    )
    assert status == OK
    expected = oracle.apply(samples.reshape(1, -1), black).reshape(-1)
    _assert_bits(destination, expected)


def test_lut_execution_has_a_real_counterexample_to_direct_formula(bridge):
    black = BLACKS[0]
    table_status, table = bridge.build_lut(black)
    assert table_status == OK
    _assert_bits(table, oracle.build_lut(black))
    rng = np.random.default_rng(20261022)
    samples = np.concatenate((
        rng.random(16384, dtype=np.float32),
        np.arange(4097, dtype=np.float32) / np.float32(4096),
    ))
    actual = np.empty_like(samples)
    assert bridge.apply_raw(
        samples, samples.size, samples.size, 1, samples.size, black,
        actual, actual.size, actual.size,
    ) == OK
    _assert_bits(actual, oracle.apply(samples.reshape(1, -1), black).reshape(-1))
    algebraic = oracle.direct_formula(samples, black)
    different = np.flatnonzero(_bits(actual) != _bits(algebraic))
    assert different.size, "LUT interpolation unexpectedly collapsed to direct algebra"
    witness = int(different[0])
    assert _bits(actual[witness]) != _bits(algebraic[witness]), (
        float(samples[witness]), float(actual[witness]), float(algebraic[witness])
    )


def test_zero_black_is_a_bit_preserving_bypass(bridge):
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    source = np.asarray([-0.0, 0.0, tiny, np.float32(0.5), 1.0], dtype=np.float32)
    output = np.full_like(source, SENTINEL)
    assert bridge.apply_raw(
        source, source.size, source.size, 1, source.size, 0.0,
        output, output.size, output.size,
    ) == OK
    _assert_bits(output, source)


def test_strided_rows_ignore_source_padding_and_preserve_destination_padding(bridge):
    height, row_samples = 3, 5
    src_stride, dst_stride = 8, 7
    source = np.full(height * src_stride, np.nan, dtype=np.float32)
    rng = np.random.default_rng(20261023)
    active = rng.random((height, row_samples), dtype=np.float32)
    for row in range(height):
        source[row * src_stride:row * src_stride + row_samples] = active[row]
    destination = np.full(height * dst_stride, SENTINEL, dtype=np.float32)
    destination_before = destination.copy()
    black = BLACKS[0]

    status = bridge.apply_raw(
        source, source.size, src_stride, height, row_samples, black,
        destination, destination.size, dst_stride,
    )
    assert status == OK
    expected = oracle.apply(active, black)
    for row in range(height):
        start = row * dst_stride
        _assert_bits(destination[start:start + row_samples], expected[row])
        _assert_bits(destination[start + row_samples:start + dst_stride],
                     destination_before[start + row_samples:start + dst_stride])


def test_exact_in_place_operation_and_zero_black_preserve_padding(bridge):
    stride, height, row_samples = 6, 2, 4
    backing = np.asarray([
        -0.0, 0.0, 0.25, 1.0, 77.0, 88.0,
        np.nextafter(np.float32(0.0), np.float32(1.0)), 0.5, 0.75, 0.875, 99.0, 111.0,
    ], dtype=np.float32)
    before = backing.copy()
    assert bridge.apply_raw(
        backing, backing.size, stride, height, row_samples, 0.0,
        backing, backing.size, stride,
    ) == OK
    for row in range(height):
        start = row * stride
        _assert_bits(backing[start:start + row_samples], before[start:start + row_samples])
        _assert_bits(backing[start + row_samples:start + stride], before[start + row_samples:start + stride])


def _expect_failure_unchanged(bridge, status, source, destination, *, black=BLACKS[0],
                              src_count=None, src_stride=4, height=1, row_samples=4,
                              dst_count=None, dst_stride=4, null_source=False):
    before = destination.view(np.uint8).copy()
    actual = bridge.apply_raw(
        source, source.size if src_count is None else src_count, src_stride,
        height, row_samples, black, destination,
        destination.size if dst_count is None else dst_count, dst_stride,
        null_source=null_source,
    )
    assert actual == status
    np.testing.assert_array_equal(destination.view(np.uint8), before)


def test_statuses_and_failed_calls_leave_destination_byte_identical(bridge):
    valid = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    output = np.full(8, SENTINEL, dtype=np.float32)
    _expect_failure_unchanged(bridge, INVALID_ARGUMENT, valid, output, null_source=True)
    _expect_failure_unchanged(bridge, INVALID_DIMENSIONS, valid, output, height=0)
    _expect_failure_unchanged(bridge, INVALID_STRIDE, valid, output, src_stride=3)
    _expect_failure_unchanged(bridge, BUFFER_TOO_SMALL, valid, output, dst_count=3)
    _expect_failure_unchanged(bridge, INVALID_BLACK, valid, output, black=float("nan"))

    nonfinite = valid.copy()
    nonfinite[2] = np.inf
    _expect_failure_unchanged(bridge, NON_FINITE_INPUT, nonfinite, output)
    outside = valid.copy()
    outside[1] = np.nextafter(np.float32(1.0), np.float32(np.inf))
    _expect_failure_unchanged(bridge, OUT_OF_DOMAIN, outside, output)

    before = output.view(np.uint8).copy()
    status = bridge.apply_fn(
        _float_ptr(valid), SIZE_MAX, 4, 1, 4, BLACKS[0],
        _float_ptr(output), output.size, 4,
    )
    assert status == ARITHMETIC_OVERFLOW
    np.testing.assert_array_equal(output.view(np.uint8), before)

    status = bridge.apply_raw(
        valid, valid.size, 129 * 1024 * 1024, 1, 129 * 1024 * 1024, BLACKS[0],
        output, 129 * 1024 * 1024, 129 * 1024 * 1024,
    )
    assert status == RESOURCE_LIMIT
    np.testing.assert_array_equal(output.view(np.uint8), before)


def test_partial_overlap_is_rejected_without_touching_any_byte(bridge):
    backing = np.asarray([0.1, 0.2, 0.3, 0.4, SENTINEL], dtype=np.float32)
    before = backing.view(np.uint8).copy()
    status = bridge.apply_raw(
        backing[:4], 4, 4, 1, 4, BLACKS[0],
        backing[1:], 4, 4,
    )
    assert status == OVERLAP
    np.testing.assert_array_equal(backing.view(np.uint8), before)


def test_unsupported_alignment_is_rejected_before_dereference(bridge):
    source = np.asarray([0.25], dtype=np.float32)
    destination_storage = np.full(2, SENTINEL, dtype=np.float32)
    before = destination_storage.view(np.uint8).copy()
    misaligned_destination = C.cast(
        C.c_void_p(destination_storage.ctypes.data + 1), FLOAT_PTR
    )
    status = bridge.apply_fn(
        _float_ptr(source), 1, 1, 1, 1, BLACKS[0],
        misaligned_destination, 1, 1,
    )
    assert status == UNSUPPORTED_ALIGNMENT
    np.testing.assert_array_equal(destination_storage.view(np.uint8), before)


def test_lut_builder_requires_exact_4098_count_and_is_atomic(bridge):
    output = np.full(oracle.TABLE_SIZE, SENTINEL, dtype=np.float32)
    before = output.view(np.uint8).copy()
    status = bridge.build_fn(BLACKS[0], _float_ptr(output), oracle.TABLE_SIZE - 1)
    assert status == INVALID_ARGUMENT
    np.testing.assert_array_equal(output.view(np.uint8), before)

    status = bridge.build_fn(1.0, _float_ptr(output), oracle.TABLE_SIZE)
    assert status == INVALID_BLACK
    np.testing.assert_array_equal(output.view(np.uint8), before)


def test_invalid_black_and_sample_validation_are_separate_from_allocation_failures():
    """Oracle domain check; allocation/length failure hooks are intentionally absent."""
    for black in (-1.0, 1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            oracle.build_lut(black)
    with pytest.raises(ValueError):
        oracle.apply(np.asarray([[0.0, 1.01]], dtype=np.float32), BLACKS[0])
