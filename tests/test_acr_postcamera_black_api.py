"""Formal library and Python adaptation tests for prepared mode2 postblack."""

from __future__ import annotations

import ctypes as C
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
API_SOURCE = NATIVE / "src/acr_postcamera_black_api.cpp"
BLACK_SOURCE = NATIVE / "src/acr_postcamera_black.cpp"
BLACK_HEADER = NATIVE / "src/acr_postcamera_black.hpp"
LUT_SOURCE = NATIVE / "src/acr_float_lut.cpp"
LUT_HEADER = NATIVE / "src/acr_float_lut.hpp"
PUBLIC_HEADER = NATIVE / "include/imprint_renderer.h"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import native_renderer as native  # noqa: E402
import postcamera_black_oracle as oracle  # noqa: E402


FLOAT = C.c_float
FLOAT_PTR = C.POINTER(FLOAT)
SIZE = C.c_size_t
SENTINEL = np.float32(-1234.5)
BLACKS = (4100 / 65535, 0x404 / 65535)


def _float_ptr(array):
    return array.ctypes.data_as(FLOAT_PTR)


def _bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


@pytest.fixture(scope="module")
def formal_library():
    path = NATIVE / "build" / (
        "libimprint_renderer.0.1.0.dylib" if sys.platform == "darwin"
        else "libimprint_renderer.0.1.0.so"
    )
    assert path.is_file(), f"build the formal native renderer library: missing {path}"
    sources = (API_SOURCE, BLACK_SOURCE, BLACK_HEADER, LUT_SOURCE, LUT_HEADER, PUBLIC_HEADER,
               NATIVE / "src/acr_u16_lut.cpp", NATIVE / "src/acr_u16_lut.hpp")
    newest_source = max(item.stat().st_mtime_ns for item in sources)
    assert path.stat().st_mtime_ns >= newest_source, (
        f"rebuild the formal native renderer library: stale artifact {path}"
    )
    library = C.CDLL(str(path))
    native._configure_library(library)
    assert hasattr(library, "im_native_acr_postcamera_black_prepared_f32"), (
        "formal native library is missing im_native_acr_postcamera_black_prepared_f32"
    )
    return library, str(path)


def _native_call(function, source, source_count, source_stride, height,
                 row_samples, black, destination, destination_count,
                 destination_stride):
    return function(
        _float_ptr(source), source_count, source_stride, height, row_samples,
        black, _float_ptr(destination), destination_count, destination_stride,
    )


@pytest.mark.parametrize("black", BLACKS, ids=("b16-4100", "b16-1028"))
def test_formal_symbol_matches_oracle_on_hwc_samples(formal_library, black):
    library, _ = formal_library
    monkeypatch_source = np.random.default_rng(20261024).random((6, 5, 17), dtype=np.float32)
    source = np.ascontiguousarray(monkeypatch_source)
    expected = oracle.apply(source, black)
    output = np.full_like(source, SENTINEL)
    function = library.im_native_acr_postcamera_black_prepared_f32
    status = _native_call(
        function, source, source.size, source.shape[1] * source.shape[2],
        source.shape[0], source.shape[1] * source.shape[2], black,
        output, output.size, source.shape[1] * source.shape[2],
    )
    assert status == 0
    _assert_bits(output, expected)


def test_python_adapter_copies_noncontiguous_source_and_preserves_zero_bits(
        formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    backing = np.random.default_rng(20261025).random((6, 10, 17), dtype=np.float32)
    source = backing[:, ::2, :]
    before = backing.copy()
    actual = native.native_acr_postcamera_black_prepared(source, BLACKS[0])
    _assert_bits(actual, oracle.apply(source, BLACKS[0]))
    np.testing.assert_array_equal(backing, before)
    assert actual.shape == source.shape and actual.dtype == np.float32
    assert actual.flags.c_contiguous

    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    zero_source = np.asarray([[-0.0, 0.0, tiny, 0.5, 1.0]], dtype=np.float32)
    zero_actual = native.native_acr_postcamera_black_prepared(zero_source, 0.0)
    _assert_bits(zero_actual, zero_source)


def test_formal_status_mapping_and_atomicity(formal_library):
    library, _ = formal_library
    function = library.im_native_acr_postcamera_black_prepared_f32
    source = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    output = np.full(4, SENTINEL, dtype=np.float32)
    before = output.view(np.uint8).copy()

    assert _native_call(function, source, source.size, 4, 1, 4, float("nan"),
                        output, output.size, 4) == 1
    np.testing.assert_array_equal(output.view(np.uint8), before)

    nonfinite = source.copy()
    nonfinite[2] = np.nan
    assert _native_call(function, nonfinite, nonfinite.size, 4, 1, 4, BLACKS[0],
                        output, output.size, 4) == 1
    np.testing.assert_array_equal(output.view(np.uint8), before)

    outside = source.copy()
    outside[1] = np.float32(1.25)
    assert _native_call(function, outside, outside.size, 4, 1, 4, BLACKS[0],
                        output, output.size, 4) == 1
    np.testing.assert_array_equal(output.view(np.uint8), before)

    assert _native_call(function, source, source.size, 4, 1, 4, BLACKS[0],
                        output, 3, 4) == 4
    np.testing.assert_array_equal(output.view(np.uint8), before)

    # The resource gate is reached from metadata before either tiny buffer is read.
    oversized_count = 129 * 1024 * 1024
    status = _native_call(
        function, source, oversized_count, oversized_count, 1, oversized_count,
        BLACKS[0], output, oversized_count, oversized_count,
    )
    assert status == 5
    np.testing.assert_array_equal(output.view(np.uint8), before)


class _StubFunction:
    def __init__(self, status):
        self.status = status
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.status


class _StubLibrary:
    def __init__(self, status):
        self.im_native_acr_postcamera_black_prepared_f32 = _StubFunction(status)


@pytest.mark.parametrize("status", (1, 4))
def test_python_adapter_maps_invalid_argument_and_capacity_to_value_error(
        monkeypatch, status):
    monkeypatch.setattr(native, "_get_library", lambda: (_StubLibrary(status), "stub"))
    with pytest.raises(ValueError):
        native.native_acr_postcamera_black_prepared(
            np.full((2, 3), 0.25, dtype=np.float32), BLACKS[0]
        )


def test_python_adapter_maps_remaining_native_status_to_renderer_error(monkeypatch):
    monkeypatch.setattr(native, "_get_library", lambda: (_StubLibrary(5), "stub"))
    with pytest.raises(native.NativeRendererError, match="status 5"):
        native.native_acr_postcamera_black_prepared(
            np.full((2, 3), 0.25, dtype=np.float32), BLACKS[0]
        )


@pytest.mark.parametrize("black", (-1.0, 1.0, float("nan"), float("inf")))
def test_python_adapter_rejects_invalid_black_before_loading_library(monkeypatch, black):
    monkeypatch.setattr(native, "_get_library", lambda: pytest.fail("invalid black loaded native library"))
    with pytest.raises(ValueError):
        native.native_acr_postcamera_black_prepared(
            np.full((2, 3), 0.25, dtype=np.float32), black
        )
