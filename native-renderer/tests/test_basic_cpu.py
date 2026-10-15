"""Direct parity and ABI checks for the eight native basic preview controls."""
import ctypes as ct
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import native_renderer
import ricoh_filter


class _BasicParams(ct.Structure):
    _fields_ = [
        ("exposure", ct.c_float),
        ("contrast", ct.c_float),
        ("highlights", ct.c_float),
        ("shadows", ct.c_float),
        ("whites", ct.c_float),
        ("blacks", ct.c_float),
        ("vibrance", ct.c_float),
        ("saturation", ct.c_float),
    ]


_FIELDS = tuple(name for name, _ in _BasicParams._fields_)


def _function(library, dtype):
    sample_type = ct.POINTER(ct.c_uint8 if dtype == np.dtype(np.uint8) else ct.c_uint16)
    function = getattr(library, "im_native_basic_rgb8" if dtype == np.dtype(np.uint8)
                       else "im_native_basic_rgb16")
    function.argtypes = [ct.c_uint32, ct.c_uint32, sample_type, ct.c_size_t,
                         ct.POINTER(_BasicParams), sample_type, ct.c_size_t]
    function.restype = ct.c_int
    return function


@pytest.fixture(scope="module")
def native_functions():
    try:
        library, _ = native_renderer._get_library()
    except native_renderer.NativeRendererError as exc:
        if os.environ.get("IMPRINT_NATIVE_RENDERER_LIB"):
            pytest.fail(f"Configured native renderer library unavailable: {exc}")
        pytest.skip(f"Build the native renderer for basic CPU parity tests: {exc}")
    symbols = ("im_native_basic_rgb8", "im_native_basic_rgb16")
    missing = [symbol for symbol in symbols if not hasattr(library, symbol)]
    if missing:
        if os.environ.get("IMPRINT_NATIVE_RENDERER_LIB"):
            pytest.fail(f"Configured native renderer library is missing symbols: {missing}")
        pytest.skip(f"Native renderer lacks basic CPU ABI symbols: {missing}")
    return {np.dtype(np.uint8): _function(library, np.dtype(np.uint8)),
            np.dtype(np.uint16): _function(library, np.dtype(np.uint16))}


def _params(values=None):
    values = values or {}
    return _BasicParams(*(float(values.get(name, 0.0)) for name in _FIELDS))


def _run(function, source, controls):
    source = np.ascontiguousarray(source)
    destination = np.full_like(source, 0xA5)
    params = _params(controls)
    status = function(source.shape[1], source.shape[0],
                      source.ctypes.data_as(function.argtypes[2]), source.size,
                      ct.byref(params),
                      destination.ctypes.data_as(function.argtypes[5]), destination.size)
    assert status == 0
    return destination


def _reference(source, controls):
    complete = {name: float(controls.get(name, 0.0)) for name in _FIELDS}
    return ricoh_filter._apply_basic_preview_effect_python(source, complete)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_random_and_extreme_samples_match_python(native_functions, dtype):
    rng = np.random.default_rng(317)
    maximum = np.iinfo(dtype).max
    source = rng.integers(0, maximum + 1, size=(43, 61, 3), dtype=dtype)
    source[0, :6] = np.array([
        [0, 0, 0], [maximum, maximum, maximum], [maximum, 0, 0],
        [0, maximum, 0], [0, 0, maximum], [maximum // 2] * 3,
    ], dtype=dtype)
    original = source.copy()
    controls = {"exposure": 1.35, "contrast": -47, "highlights": -72,
                "shadows": 68, "whites": -34, "blacks": 53,
                "vibrance": 71, "saturation": -39}
    actual = _run(native_functions[np.dtype(dtype)], source, controls)
    expected = _reference(source, controls)
    assert int(np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32)))) <= 1
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("field", _FIELDS)
@pytest.mark.parametrize("amount", [-100.0, 100.0])
def test_each_control_matches_python(native_functions, dtype, field, amount):
    rng = np.random.default_rng(600 + _FIELDS.index(field))
    maximum = np.iinfo(dtype).max
    source = rng.integers(0, maximum + 1, size=(13, 29, 3), dtype=dtype)
    value = amount if field != "exposure" else amount / 20.0
    controls = {field: value}
    actual = _run(native_functions[np.dtype(dtype)], source, controls)
    expected = _reference(source, controls)
    difference = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
    assert int(difference.max()) <= 1, f"{dtype.__name__} {field}={value}: max diff {difference.max()}"


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("shape", [(1, 67, 3), (71, 1, 3)])
def test_single_row_column_and_zero_controls(native_functions, dtype, shape):
    rng = np.random.default_rng(202)
    maximum = np.iinfo(dtype).max
    source = rng.integers(0, maximum + 1, size=shape, dtype=dtype)
    original = source.copy()
    function = native_functions[np.dtype(dtype)]
    actual = _run(function, source, {})
    np.testing.assert_array_equal(actual, source)
    np.testing.assert_array_equal(source, original)
    controls = {"exposure": -2.3, "contrast": 85, "highlights": 100,
                "shadows": -100, "whites": 100, "blacks": -100,
                "vibrance": -100, "saturation": 100}
    expected = _reference(source, controls)
    actual = _run(function, source, controls)
    assert int(np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32)))) <= 1


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_abi_rejects_invalid_counts_params_and_overlap_without_writing(native_functions, dtype):
    function = native_functions[np.dtype(dtype)]
    ctype = ct.c_uint8 if dtype == np.uint8 else ct.c_uint16
    source = np.arange(18, dtype=dtype).reshape(2, 3, 3)
    destination = np.full_like(source, 123)
    params = _params()

    def call(src=source, dst=destination, width=3, height=2,
             source_count=18, destination_count=18, p=params):
        return function(width, height, src.ctypes.data_as(ct.POINTER(ctype)), source_count,
                        ct.byref(p), dst.ctypes.data_as(ct.POINTER(ctype)), destination_count)

    before = destination.copy()
    assert call(source_count=17) != 0
    np.testing.assert_array_equal(destination, before)
    assert call(width=0) != 0
    np.testing.assert_array_equal(destination, before)
    assert call(destination_count=17) != 0
    np.testing.assert_array_equal(destination, before)

    params.exposure = float("nan")
    assert call() != 0
    np.testing.assert_array_equal(destination, before)
    params.exposure = 0.0
    params.saturation = 100.01
    assert call() != 0
    np.testing.assert_array_equal(destination, before)
    params.saturation = 0.0

    aliased = source.copy()
    assert call(src=aliased, dst=aliased) != 0


def test_tie_to_even_quantization_matches_reference(native_functions):
    # Halving the last even/odd source codes creates exact .5 output codes.
    for dtype in (np.uint8, np.uint16):
        maximum = np.iinfo(dtype).max
        source = np.array([[[maximum, maximum - 2, maximum - 4]]], dtype=dtype)
        controls = {"exposure": -1.0}
        actual = _run(native_functions[np.dtype(dtype)], source, controls)
        expected = _reference(source, controls)
        np.testing.assert_array_equal(
            actual,
            np.array([[[maximum // 2 + 1, (maximum - 2) // 2,
                        (maximum - 4) // 2 + 1]]], dtype=dtype),
        )
        assert int(np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32)))) <= 1
