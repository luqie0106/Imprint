"""Formal C ABI and Python wrapper tests for prepared Stage3 geometry."""

import ctypes
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import native_renderer as native

OK = 0
INVALID_ARGUMENT = 1
PAIR_INPUT = [
    (1, 1), (14, 21), (10, 3), (1, 1), (1, 1), (12, 4),
]
# Fixed values transcribed after locally probing the x pairs (+30/+40/+90)
# then y pairs (+38/+48/+98), each with its factor/factor/reciprocal sequence,
# using the temporary instruction-reference dylib. Runtime tests load only
# the formal library and have no dependency on that temporary file.
PAIR_GOLDEN = [
    (19661, 16384), (14, 7), (4, 1), (3, 1),
    (894784853, 1073741824), (1, 1),
]


def _words(pairs):
    return [word for pair in pairs for word in pair]


def _snapshot(pairs=PAIR_INPUT, scales=(1.2, 3.0)):
    value = native._AcrSourceStage3Geometry()
    value.rationals[:] = _words(pairs)
    value.scales[:] = scales
    return value


def _sentinel():
    value = native._AcrSourceStage3Geometry()
    value.rationals[:] = [0xA5A5A5A5] * 12
    value.scales[:] = (123.25, -77.5)
    return value


def _arrays(pairs=PAIR_INPUT, scales=(1.2, 3.0)):
    return (np.asarray(pairs, dtype=np.uint32).reshape(6, 2),
            np.asarray(scales, dtype=np.float64))


@pytest.fixture(scope="module")
def library():
    build = ROOT / "native-renderer" / "build"
    names = (["libimprint_renderer.dylib"] if sys.platform == "darwin" else
             ["Release/imprint_renderer.dll", "imprint_renderer.dll"]
             if sys.platform == "win32" else ["libimprint_renderer.so"])
    path = next((build / name for name in names if (build / name).is_file()), None)
    assert path is not None, "build imprint_renderer before checking the public ABI"
    api_source = ROOT / "native-renderer/src/acr_source_stage3_geometry_api.cpp"
    assert path.stat().st_mtime_ns >= api_source.stat().st_mtime_ns, (
        "formal library predates the Stage3 geometry API source")
    lib = ctypes.CDLL(str(path))
    function = lib.im_native_acr_source_stage3_geometry_normalize
    geometry_type = native._AcrSourceStage3Geometry
    function.argtypes = [ctypes.POINTER(geometry_type),
                         ctypes.POINTER(geometry_type)]
    function.restype = ctypes.c_int
    return lib, str(path)


def test_public_layout_and_normal_python_wrapper(library, monkeypatch):
    geometry_type = native._AcrSourceStage3Geometry
    assert ctypes.sizeof(geometry_type) == 64
    assert ctypes.alignment(geometry_type) == ctypes.alignment(ctypes.c_double)
    assert geometry_type.rationals.offset == 0
    assert geometry_type.scales.offset == 48

    rationals, scales = _arrays()
    rationals_before = rationals.copy()
    scales_before = scales.copy()
    monkeypatch.setattr(native, "_get_library", lambda: library)
    output_pairs, output_scales = native.native_acr_normalize_source_stage3_geometry(
        rationals, scales)

    assert output_pairs.dtype == np.uint32
    assert output_pairs.shape == (6, 2)
    assert output_pairs.tolist() == [list(pair) for pair in PAIR_GOLDEN]
    assert output_scales.dtype == np.float64
    assert output_scales.shape == (2,)
    assert output_scales.tolist() == [1.0, 1.0]
    np.testing.assert_array_equal(rationals, rationals_before)
    np.testing.assert_array_equal(scales, scales_before)


def test_c_abi_exact_in_place_and_independent_reference_golden(library):
    function = library[0].im_native_acr_source_stage3_geometry_normalize
    in_place = _snapshot()
    assert function(ctypes.byref(in_place), ctypes.byref(in_place)) == OK
    assert list(in_place.rationals) == _words(PAIR_GOLDEN)
    assert list(in_place.scales) == [1.0, 1.0]


def test_finite_scales_at_or_below_one_preserve_all_bits(library):
    pairs = [(0xFFFFFFFF, 0), (3, 7), (13, 17), (19, 23), (29, 31), (37, 41)]
    input_value = _snapshot(pairs, (1.0, -0.0))
    output_value = _sentinel()
    function = library[0].im_native_acr_source_stage3_geometry_normalize

    assert function(ctypes.byref(input_value), ctypes.byref(output_value)) == OK
    assert bytes(output_value) == bytes(input_value)
    assert struct.pack("=d", output_value.scales[1]) == struct.pack("=d", -0.0)


def test_null_misaligned_and_partial_overlap_reject_without_writes(library):
    function = library[0].im_native_acr_source_stage3_geometry_normalize
    valid_input = _snapshot()
    valid_output = _sentinel()
    output_before = bytes(valid_output)

    assert function(None, ctypes.byref(valid_output)) == INVALID_ARGUMENT
    assert bytes(valid_output) == output_before
    assert function(ctypes.byref(valid_input), None) == INVALID_ARGUMENT
    assert bytes(valid_output) == output_before

    misaligned_storage = ctypes.create_string_buffer(80)
    base = ctypes.addressof(misaligned_storage)
    assert base % ctypes.alignment(native._AcrSourceStage3Geometry) == 0
    misaligned_input_pointer = ctypes.cast(
        base + 1, ctypes.POINTER(native._AcrSourceStage3Geometry))
    assert function(misaligned_input_pointer, ctypes.byref(valid_output)) == INVALID_ARGUMENT
    assert bytes(valid_output) == output_before

    misaligned_output_pointer = ctypes.cast(
        base + 1, ctypes.POINTER(native._AcrSourceStage3Geometry))
    misaligned_before = bytes(misaligned_storage)
    assert function(ctypes.byref(valid_input), misaligned_output_pointer) == INVALID_ARGUMENT
    assert bytes(misaligned_storage) == misaligned_before

    overlap_storage = ctypes.create_string_buffer(80)
    overlap_base = ctypes.addressof(overlap_storage)
    input_pointer = ctypes.cast(
        overlap_base, ctypes.POINTER(native._AcrSourceStage3Geometry))
    partial_output_pointer = ctypes.cast(
        overlap_base + 8, ctypes.POINTER(native._AcrSourceStage3Geometry))
    ctypes.memmove(overlap_base, ctypes.byref(valid_input), ctypes.sizeof(valid_input))
    overlap_before = bytes(overlap_storage)
    assert function(input_pointer, partial_output_pointer) == INVALID_ARGUMENT
    assert bytes(overlap_storage) == overlap_before


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scale_rejects_without_output_writes(library, axis, bad_scale):
    function = library[0].im_native_acr_source_stage3_geometry_normalize
    scales = [1.2, 3.0]
    scales[axis] = bad_scale
    input_value = _snapshot(scales=scales)
    output_value = _sentinel()
    before = bytes(output_value)

    assert function(ctypes.byref(input_value), ctypes.byref(output_value)) == INVALID_ARGUMENT
    assert bytes(output_value) == before


def test_late_rational_failure_rolls_back_the_whole_public_output(library):
    function = library[0].im_native_acr_source_stage3_geometry_normalize
    pairs = [(1, 1), (2, 3), (0xFFFFFFFF, 1), (6, 7), (8, 9), (10, 11)]
    input_value = _snapshot(pairs, (1.2, 1.0))
    output_value = _sentinel()
    before = bytes(output_value)

    assert function(ctypes.byref(input_value), ctypes.byref(output_value)) == INVALID_ARGUMENT
    assert bytes(output_value) == before


@pytest.mark.parametrize("bad_rationals", [
    np.zeros((12,), dtype=np.uint32),
    np.zeros((6, 2), dtype=np.uint64),
    np.zeros((6, 2), dtype=np.int32),
    [[0, 1]] * 6,
])
def test_python_rejects_wrong_rational_shape_or_dtype_before_loading(
        monkeypatch, bad_rationals):
    _, scales = _arrays()
    monkeypatch.setattr(native, "_get_library",
                        lambda: pytest.fail("invalid inputs must not load the library"))
    with pytest.raises(ValueError):
        native.native_acr_normalize_source_stage3_geometry(bad_rationals, scales)


@pytest.mark.parametrize("bad_scales", [
    np.zeros((1, 2), dtype=np.float64),
    np.zeros((2,), dtype=np.float32),
    [1.0, 1.0],
    np.asarray([np.nan, 1.0], dtype=np.float64),
    np.asarray([np.inf, 1.0], dtype=np.float64),
])
def test_python_rejects_wrong_scale_shape_dtype_or_nonfinite_before_loading(
        monkeypatch, bad_scales):
    rationals, _ = _arrays()
    monkeypatch.setattr(native, "_get_library",
                        lambda: pytest.fail("invalid inputs must not load the library"))
    with pytest.raises(ValueError):
        native.native_acr_normalize_source_stage3_geometry(rationals, bad_scales)


def test_python_surfaces_native_numeric_failure(library, monkeypatch):
    rationals, scales = _arrays(
        [(1, 1), (2, 3), (0xFFFFFFFF, 1), (6, 7), (8, 9), (10, 11)],
        (1.2, 1.0),
    )
    monkeypatch.setattr(native, "_get_library", lambda: library)
    with pytest.raises(native.NativeRendererError, match="status 1"):
        native.native_acr_normalize_source_stage3_geometry(rationals, scales)
