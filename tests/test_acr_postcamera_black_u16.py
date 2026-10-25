"""Mode1 unsigned post-camera-black oracle parity and memory safety tests."""

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
BLACK_SOURCE = NATIVE / "src/acr_postcamera_black.cpp"
BLACK_HEADER = BLACK_SOURCE.with_suffix(".hpp")
FLOAT_SOURCE = NATIVE / "src/acr_float_lut.cpp"
FLOAT_HEADER = FLOAT_SOURCE.with_suffix(".hpp")
U16_SOURCE = NATIVE / "src/acr_u16_lut.cpp"
U16_HEADER = U16_SOURCE.with_suffix(".hpp")
API_SOURCE = NATIVE / "src/acr_postcamera_black_api.cpp"
PUBLIC_HEADER = NATIVE / "include/imprint_renderer.h"
BRIDGE_SOURCE = NATIVE / "tools/acr_postcamera_black_bridge.cpp"
BLACK_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_postcamera_black.cpp.o"
FLOAT_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_float_lut.cpp.o"
U16_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_u16_lut.cpp.o"
sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
sys.path.insert(0, str(ROOT / "src"))
import postcamera_black_oracle as float_oracle  # noqa: E402
import u16_lut_oracle  # noqa: E402
import native_renderer as native  # noqa: E402


U16 = C.c_uint16
U16_PTR = C.POINTER(U16)
SIZE = C.c_size_t
SIZE_MAX = SIZE(-1).value
TABLE_SIZE = 65536
SENTINEL = np.uint16(0xD39A)

OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
INVALID_STRIDE = 3
BUFFER_TOO_SMALL = 4
UNSUPPORTED_ALIGNMENT = 5
OVERLAP = 6
ARITHMETIC_OVERFLOW = 7
RESOURCE_LIMIT = 8
INVALID_BLACK = 10

NEAR_ZERO = float(np.nextafter(np.float64(0.0), np.float64(1.0)))
NEAR_ONE = float(np.nextafter(np.float64(1.0), np.float64(0.0)))
BLACKS = (0.0, 4100 / 65535, 0x404 / 65535, NEAR_ZERO, NEAR_ONE)


def _u16_ptr(array):
    return array.ctypes.data_as(U16_PTR)


def _null_u16_ptr():
    return C.cast(C.c_void_p(0), U16_PTR)


def expected_u16_table(black):
    float_table = float_oracle.build_lut(float(black))
    return u16_lut_oracle.build(float_table)


def expected_pixels(source, black, table=None):
    if black == 0.0:
        return np.array(source, dtype=np.uint16, order="C", copy=True)
    return np.asarray(table, dtype=np.uint16)[np.asarray(source, dtype=np.uint16)]


class Bridge:
    def __init__(self, path):
        self.library = C.CDLL(str(path))
        self.apply_fn = self.library.acr_test_postcamera_black_u16
        self.apply_fn.argtypes = [
            U16_PTR, SIZE, SIZE, SIZE, SIZE, C.c_double,
            U16_PTR, SIZE, SIZE,
        ]
        self.apply_fn.restype = C.c_int
        self.build_fn = self.library.acr_test_postcamera_black_u16_lut
        self.build_fn.argtypes = [C.c_double, U16_PTR, SIZE]
        self.build_fn.restype = C.c_int

    def apply_raw(self, src, src_count, src_stride, height, row_samples,
                  black, dst, dst_count, dst_stride, *, null_source=False):
        source_pointer = _null_u16_ptr() if null_source else _u16_ptr(src)
        return self.apply_fn(
            source_pointer, src_count, src_stride, height, row_samples, black,
            _u16_ptr(dst), dst_count, dst_stride,
        )

    def build_lut(self, black, count=TABLE_SIZE):
        output = np.full(TABLE_SIZE + 4, SENTINEL, dtype=np.uint16)
        status = self.build_fn(black, _u16_ptr(output), count)
        return status, output


def _bridge_inputs(kind):
    if kind == "source":
        return [BRIDGE_SOURCE, BLACK_SOURCE, FLOAT_SOURCE, U16_SOURCE]
    for obj, source, header in (
        (BLACK_OBJECT, BLACK_SOURCE, BLACK_HEADER),
        (FLOAT_OBJECT, FLOAT_SOURCE, FLOAT_HEADER),
        (U16_OBJECT, U16_SOURCE, U16_HEADER),
    ):
        assert obj.is_file(), f"rebuild renderer: missing official object {obj}"
        assert obj.stat().st_mtime_ns >= max(source.stat().st_mtime_ns, header.stat().st_mtime_ns), (
            f"rebuild renderer: stale official object {obj.name}"
        )
    return [BRIDGE_SOURCE, BLACK_OBJECT, FLOAT_OBJECT, U16_OBJECT]


@pytest.fixture(scope="module", params=("source", "official_object"),
                ids=("source", "official_object"))
def bridge(request, tmp_path_factory):
    inputs = _bridge_inputs(request.param)
    assert BRIDGE_SOURCE.is_file(), f"missing post-camera-black bridge: {BRIDGE_SOURCE}"
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the post-camera-black bridge"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_postcamera_black_u16_" + request.param) / ("libpostblacku16" + suffix)
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


@pytest.mark.parametrize("black", BLACKS, ids=("zero", "b16-4100", "b16-1028", "near-zero", "near-one"))
def test_all_65536_lut_entries_match_independent_float_then_u16_oracles(bridge, black):
    status, actual = bridge.build_lut(black)
    assert status == OK
    expected = expected_u16_table(black)
    np.testing.assert_array_equal(actual[:TABLE_SIZE], expected)
    assert np.all(actual[TABLE_SIZE:] == SENTINEL)


@pytest.mark.parametrize("black", BLACKS[1:], ids=("b16-4100", "b16-1028", "near-zero", "near-one"))
def test_all_uint16_inputs_follow_composed_lut_lookup(bridge, black):
    table_status, actual_table = bridge.build_lut(black)
    assert table_status == OK
    table = expected_u16_table(black)
    source = np.arange(TABLE_SIZE, dtype=np.uint16)
    output = np.full(TABLE_SIZE + 4, SENTINEL, dtype=np.uint16)
    status = bridge.apply_raw(
        source, source.size, source.size, 1, source.size, black,
        output, output.size, output.size,
    )
    assert status == OK
    np.testing.assert_array_equal(output[:TABLE_SIZE], table)
    np.testing.assert_array_equal(actual_table[:TABLE_SIZE], table)
    assert np.all(output[TABLE_SIZE:] == SENTINEL)


def test_zero_black_bypasses_table_bit_exactly_in_place(bridge):
    source = np.arange(TABLE_SIZE, dtype=np.uint16)
    source = np.concatenate((source, np.asarray([0xCAFE, 0xBABE], dtype=np.uint16)))
    before = source.copy()
    status = bridge.apply_raw(
        source, source.size, source.size, 1, TABLE_SIZE, 0.0,
        source, source.size, source.size,
    )
    assert status == OK
    np.testing.assert_array_equal(source, before)


def test_strided_rows_ignore_padding_and_preserve_output_padding(bridge):
    height, row_samples = 4, 17
    src_stride, dst_stride = 23, 21
    rng = np.random.default_rng(20261021)
    source = np.full(height * src_stride, 0xEEEE, dtype=np.uint16)
    active = rng.integers(0, TABLE_SIZE, size=(height, row_samples), dtype=np.uint16)
    for row in range(height):
        start = row * src_stride
        source[start:start + row_samples] = active[row]
    output = np.full(height * dst_stride, SENTINEL, dtype=np.uint16)
    output_before = output.copy()
    black = BLACKS[1]
    table = expected_u16_table(black)

    status = bridge.apply_raw(
        source, source.size, src_stride, height, row_samples, black,
        output, output.size, dst_stride,
    )
    assert status == OK
    expected = table[active]
    for row in range(height):
        start = row * dst_stride
        np.testing.assert_array_equal(output[start:start + row_samples], expected[row])
        np.testing.assert_array_equal(
            output[start + row_samples:start + dst_stride],
            output_before[start + row_samples:start + dst_stride],
        )


def test_exact_in_place_nonzero_black_and_padding(bridge):
    height, row_samples, stride = 3, 4, 7
    source = np.asarray([
        0, 1, 4100, 65535, 0xAAAA, 0xBBBB, 0xCCCC,
        2, 1028, 30000, 62000, 0xDDDD, 0xEEEE, 0xABCD,
        11, 22, 33, 44, 0xCAFE, 0xBABE, 0x7777,
    ], dtype=np.uint16)
    before = source.copy()
    black = BLACKS[2]
    table = expected_u16_table(black)
    status = bridge.apply_raw(
        source, source.size, stride, height, row_samples, black,
        source, source.size, stride,
    )
    assert status == OK
    for row in range(height):
        start = row * stride
        np.testing.assert_array_equal(source[start:start + row_samples], table[before[start:start + row_samples]])
        np.testing.assert_array_equal(source[start + row_samples:start + stride], before[start + row_samples:start + stride])


def test_partial_aliases_with_different_strides_or_counts_are_rejected(bridge):
    cases = (
        # Shifted bases and different row strides.
        (0, 10, 6, 1, 4, 1, 10, 5),
        # Same base, but a different declared destination span/count.
        (0, 10, 6, 1, 4, 0, 9, 6),
    )
    for src_start, src_count, src_stride, height, row_samples, dst_start, dst_count, dst_stride in cases:
        backing = np.arange(16, dtype=np.uint16)
        before = backing.copy()
        status = bridge.apply_raw(
            backing[src_start:], src_count, src_stride, height, row_samples,
            BLACKS[1], backing[dst_start:], dst_count, dst_stride,
        )
        assert status == OVERLAP
        np.testing.assert_array_equal(backing, before)


def test_failures_are_atomic_for_null_dimensions_stride_capacity_black_and_overflow(bridge):
    source = np.arange(8, dtype=np.uint16)
    output = np.full(8, SENTINEL, dtype=np.uint16)
    before = output.view(np.uint8).copy()

    assert bridge.apply_fn(_null_u16_ptr(), 8, 4, 2, 4, BLACKS[1],
                          _u16_ptr(output), 8, 4) == INVALID_ARGUMENT
    assert bridge.apply_fn(_u16_ptr(source), 8, 4, 0, 4, BLACKS[1],
                          _u16_ptr(output), 8, 4) == INVALID_DIMENSIONS
    assert bridge.apply_raw(source, 8, 3, 2, 4, BLACKS[1], output, 8, 4) == INVALID_STRIDE
    assert bridge.apply_raw(source, 8, 4, 2, 4, BLACKS[1], output, 7, 4) == BUFFER_TOO_SMALL
    assert bridge.apply_raw(source, 8, 4, 2, 4, 1.0, output, 8, 4) == INVALID_BLACK
    np.testing.assert_array_equal(output.view(np.uint8), before)

    status = bridge.apply_fn(
        _u16_ptr(source), SIZE_MAX, 4, 1, 4, BLACKS[1],
        _u16_ptr(output), output.size, 4,
    )
    assert status == ARITHMETIC_OVERFLOW
    np.testing.assert_array_equal(output.view(np.uint8), before)

    oversized = 129 * 1024 * 1024
    status = bridge.apply_fn(
        _u16_ptr(source), oversized, oversized, 1, oversized, BLACKS[1],
        _u16_ptr(output), oversized, oversized,
    )
    assert status == RESOURCE_LIMIT
    np.testing.assert_array_equal(output.view(np.uint8), before)


def test_unsupported_alignment_is_rejected_without_writing(bridge):
    source = np.asarray([10, 20], dtype=np.uint16)
    destination_storage = np.full(4, SENTINEL, dtype=np.uint16)
    before = destination_storage.view(np.uint8).copy()
    misaligned = C.cast(C.c_void_p(destination_storage.ctypes.data + 1), U16_PTR)
    status = bridge.apply_fn(
        _u16_ptr(source), 2, 2, 1, 2, BLACKS[1], misaligned, 2, 2,
    )
    assert status == UNSUPPORTED_ALIGNMENT
    np.testing.assert_array_equal(destination_storage.view(np.uint8), before)


def test_lut_builder_count_and_invalid_black_fail_atomically(bridge):
    output = np.full(TABLE_SIZE + 4, SENTINEL, dtype=np.uint16)
    before = output.view(np.uint8).copy()
    assert bridge.build_fn(BLACKS[1], _u16_ptr(output), TABLE_SIZE - 1) == INVALID_ARGUMENT
    np.testing.assert_array_equal(output.view(np.uint8), before)
    assert bridge.build_fn(1.0, _u16_ptr(output), TABLE_SIZE) == INVALID_BLACK
    np.testing.assert_array_equal(output.view(np.uint8), before)


@pytest.fixture(scope="module")
def formal_library():
    path = NATIVE / "build" / (
        "libimprint_renderer.0.1.0.dylib" if sys.platform == "darwin"
        else "libimprint_renderer.0.1.0.so"
    )
    assert path.is_file(), f"build the formal native renderer library: missing {path}"
    sources = (API_SOURCE, BLACK_SOURCE, BLACK_HEADER, FLOAT_SOURCE, FLOAT_HEADER,
               U16_SOURCE, U16_HEADER, PUBLIC_HEADER)
    assert path.stat().st_mtime_ns >= max(item.stat().st_mtime_ns for item in sources), (
        f"rebuild the formal native renderer library: stale artifact {path}"
    )
    library = C.CDLL(str(path))
    native._configure_library(library)
    assert hasattr(library, "im_native_acr_postcamera_black_prepared_u16"), (
        "formal native library is missing im_native_acr_postcamera_black_prepared_u16"
    )
    return library, str(path)


def test_formal_symbol_matches_full_domain_oracle(formal_library):
    library, _ = formal_library
    function = library.im_native_acr_postcamera_black_prepared_u16
    source = np.arange(TABLE_SIZE, dtype=np.uint16)
    for black in (0.0, BLACKS[1], BLACKS[2]):
        output = np.full(TABLE_SIZE, SENTINEL, dtype=np.uint16)
        status = function(
            _u16_ptr(source), source.size, source.size, 1, source.size, black,
            _u16_ptr(output), output.size, output.size,
        )
        assert status == 0
        table = None if black == 0.0 else expected_u16_table(black)
        np.testing.assert_array_equal(output, expected_pixels(source, black, table))


def test_formal_status_mapping_and_atomicity(formal_library):
    library, _ = formal_library
    function = library.im_native_acr_postcamera_black_prepared_u16
    source = np.asarray([10, 20, 30, 40], dtype=np.uint16)
    output = np.full(4, SENTINEL, dtype=np.uint16)
    before = output.view(np.uint8).copy()
    assert function(_u16_ptr(source), 4, 4, 1, 4, float("nan"),
                    _u16_ptr(output), 4, 4) == 1
    np.testing.assert_array_equal(output.view(np.uint8), before)
    assert function(_u16_ptr(source), 4, 4, 1, 4, BLACKS[1],
                    _u16_ptr(output), 3, 4) == 4
    np.testing.assert_array_equal(output.view(np.uint8), before)

    oversized = 129 * 1024 * 1024
    assert function(_u16_ptr(source), oversized, oversized, 1, oversized, BLACKS[1],
                    _u16_ptr(output), oversized, oversized) == 5
    np.testing.assert_array_equal(output.view(np.uint8), before)


def test_python_adapter_handles_noncontiguous_uint16_and_zero_bypass(formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    backing = np.random.default_rng(20261022).integers(0, TABLE_SIZE, (6, 10, 17), dtype=np.uint16)
    source = backing[:, ::2, :]
    before = backing.copy()
    black = BLACKS[1]
    actual = native.native_acr_postcamera_black_prepared_u16(source, black)
    table = expected_u16_table(black)
    np.testing.assert_array_equal(actual, table[source])
    np.testing.assert_array_equal(backing, before)
    assert actual.shape == source.shape and actual.dtype == np.uint16 and actual.flags.c_contiguous

    zero = native.native_acr_postcamera_black_prepared_u16(source, 0.0)
    np.testing.assert_array_equal(zero, source)


class _StubFunction:
    def __init__(self, status):
        self.status = status
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.status


class _StubLibrary:
    def __init__(self, status):
        self.im_native_acr_postcamera_black_prepared_u16 = _StubFunction(status)


@pytest.mark.parametrize("status", (1, 4))
def test_python_adapter_maps_invalid_and_capacity_to_value_error(monkeypatch, status):
    monkeypatch.setattr(native, "_get_library", lambda: (_StubLibrary(status), "stub"))
    with pytest.raises(ValueError):
        native.native_acr_postcamera_black_prepared_u16(
            np.full((2, 3), 10, dtype=np.uint16), BLACKS[1]
        )


def test_python_adapter_maps_other_native_status_to_renderer_error(monkeypatch):
    monkeypatch.setattr(native, "_get_library", lambda: (_StubLibrary(5), "stub"))
    with pytest.raises(native.NativeRendererError, match="status 5"):
        native.native_acr_postcamera_black_prepared_u16(
            np.full((2, 3), 10, dtype=np.uint16), BLACKS[1]
        )


@pytest.mark.parametrize("black", (-1.0, 1.0, float("nan"), float("inf")))
def test_python_adapter_rejects_invalid_black_before_loading_library(monkeypatch, black):
    monkeypatch.setattr(native, "_get_library", lambda: pytest.fail("invalid black loaded native library"))
    with pytest.raises(ValueError):
        native.native_acr_postcamera_black_prepared_u16(
            np.full((2, 3), 10, dtype=np.uint16), black
        )


def test_python_adapter_rejects_non_uint16_before_loading_library(monkeypatch):
    monkeypatch.setattr(native, "_get_library", lambda: pytest.fail("invalid dtype loaded native library"))
    for image in (np.zeros((2, 3), dtype=np.float32), np.zeros((6,), dtype=np.uint16)):
        with pytest.raises(ValueError):
            native.native_acr_postcamera_black_prepared_u16(image, BLACKS[1])
