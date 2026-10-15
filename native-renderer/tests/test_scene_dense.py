"""Direct ctypes parity and ABI checks for the remaining dense CPU kernels."""

import ctypes as ct
import os
from pathlib import Path
import sys

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[2]
_FLOAT_POINTER = ct.POINTER(ct.c_float)
_OK = 0
_INVALID_ARGUMENT = 1


def _library_candidates():
    configured = os.environ.get("IMPRINT_NATIVE_RENDERER_LIB")
    if configured:
        return [Path(configured).expanduser()]
    names = (
        ("libimprint_renderer.dylib", "imprint_renderer.dylib")
        if sys.platform == "darwin"
        else ("imprint_renderer.dll", "libimprint_renderer.dll")
        if sys.platform == "win32"
        else ("libimprint_renderer.so", "imprint_renderer.so")
    )
    roots = (
        _ROOT / "native-renderer" / "build",
        _ROOT / "native-renderer" / "cmake-build-release",
        _ROOT / "native-renderer" / "cmake-build-debug",
    )
    return [root / name for root in roots for name in names]


@pytest.fixture(scope="module")
def dense_library():
    path = next((candidate for candidate in _library_candidates() if candidate.is_file()), None)
    if path is None:
        pytest.skip("native renderer library is not built; set IMPRINT_NATIVE_RENDERER_LIB")
    library = ct.CDLL(str(path))
    library.im_native_exposure_float.argtypes = [
        ct.c_uint32, ct.c_uint32, _FLOAT_POINTER, ct.c_size_t, ct.c_float,
        _FLOAT_POINTER, ct.c_size_t,
    ]
    library.im_native_exposure_float.restype = ct.c_int
    library.im_native_rgb_peak.argtypes = [
        ct.c_uint32, ct.c_uint32, _FLOAT_POINTER, ct.c_size_t, _FLOAT_POINTER,
    ]
    library.im_native_rgb_peak.restype = ct.c_int
    library.im_native_refine_transmission.argtypes = [
        ct.c_uint32, ct.c_uint32, _FLOAT_POINTER, ct.c_size_t,
        _FLOAT_POINTER, ct.c_size_t, _FLOAT_POINTER, ct.c_size_t,
        ct.c_float, ct.c_float, _FLOAT_POINTER, ct.c_size_t,
    ]
    library.im_native_refine_transmission.restype = ct.c_int
    library.im_native_relief_transmission.argtypes = [
        ct.c_uint32, ct.c_uint32, _FLOAT_POINTER, ct.c_size_t,
        _FLOAT_POINTER, ct.c_size_t, _FLOAT_POINTER,
        ct.c_float, ct.c_float, ct.c_float, _FLOAT_POINTER, ct.c_size_t,
    ]
    library.im_native_relief_transmission.restype = ct.c_int
    return library


def _ptr(values):
    return values.ctypes.data_as(_FLOAT_POINTER)


def _luma(source):
    return ((source[..., 0] * np.float32(0.2126) +
             source[..., 1] * np.float32(0.7152)) +
            source[..., 2] * np.float32(0.0722))


def _smoothstep(low, high, value):
    position = np.clip(
        (value - np.float32(low)) / np.float32(high - low), 0.0, 1.0,
    )
    return position * position * (np.float32(3.0) - np.float32(2.0) * position)


def _call_exposure(library, source, gain, destination=None, *, width=None, height=None,
                   source_count=None, destination_count=None):
    source = np.ascontiguousarray(source, dtype=np.float32)
    if destination is None:
        destination = np.empty_like(source)
    h, w = source.shape[:2] if source.ndim == 3 else (1, source.size // 3)
    width = w if width is None else width
    height = h if height is None else height
    source_count = source.size if source_count is None else source_count
    destination_count = destination.size if destination_count is None else destination_count
    status = library.im_native_exposure_float(
        width, height, _ptr(source), source_count, gain,
        _ptr(destination), destination_count,
    )
    return status, source, destination


def _call_peak(library, source, peak=None, *, width=None, height=None, source_count=None):
    source = np.ascontiguousarray(source, dtype=np.float32)
    if peak is None:
        peak = np.full(1, -9.0, dtype=np.float32)
    h, w = source.shape[:2] if source.ndim == 3 else (1, source.size // 3)
    width = w if width is None else width
    height = h if height is None else height
    source_count = source.size if source_count is None else source_count
    status = library.im_native_rgb_peak(
        width, height, _ptr(source), source_count, _ptr(peak),
    )
    return status, source, peak


def _call_refine(library, source, slope, intercept, depth_min, depth_max,
                 destination=None, *, width=None, height=None, source_count=None,
                 slope_count=None, intercept_count=None, destination_count=None):
    source = np.ascontiguousarray(source, dtype=np.float32)
    slope = np.ascontiguousarray(slope, dtype=np.float32)
    intercept = np.ascontiguousarray(intercept, dtype=np.float32)
    h, w = source.shape[:2]
    if destination is None:
        destination = np.empty((h, w), dtype=np.float32)
    width = w if width is None else width
    height = h if height is None else height
    source_count = source.size if source_count is None else source_count
    slope_count = slope.size if slope_count is None else slope_count
    intercept_count = intercept.size if intercept_count is None else intercept_count
    destination_count = destination.size if destination_count is None else destination_count
    status = library.im_native_refine_transmission(
        width, height, _ptr(source), source_count,
        _ptr(slope), slope_count, _ptr(intercept), intercept_count,
        depth_min, depth_max, _ptr(destination), destination_count,
    )
    return status, source, slope, intercept, destination


def _call_relief(library, source, relief, airlight, base, knee, initial_t,
                 destination=None, *, width=None, height=None, source_count=None,
                 relief_count=None, destination_count=None):
    source = np.ascontiguousarray(source, dtype=np.float32)
    relief = np.ascontiguousarray(relief, dtype=np.float32)
    airlight = np.ascontiguousarray(airlight, dtype=np.float32)
    h, w = source.shape[:2]
    if destination is None:
        destination = np.empty((h, w), dtype=np.float32)
    width = w if width is None else width
    height = h if height is None else height
    source_count = source.size if source_count is None else source_count
    relief_count = relief.size if relief_count is None else relief_count
    destination_count = destination.size if destination_count is None else destination_count
    status = library.im_native_relief_transmission(
        width, height, _ptr(source), source_count, _ptr(relief), relief_count,
        _ptr(airlight), base, knee, initial_t, _ptr(destination), destination_count,
    )
    return status, source, relief, airlight, destination


def test_exposure_is_bit_exact_float32_multiply_and_does_not_clip(dense_library):
    rng = np.random.default_rng(814)
    source = rng.uniform(0.0, 1.0, (31, 53, 3)).astype(np.float32)
    source[0, 0] = (0.0, 0.125, 1.0)
    before = source.copy()

    for gain in (1.0, 1.375, 4.0):
        status, actual_source, output = _call_exposure(dense_library, source, gain)
        assert status == _OK
        expected = np.multiply(actual_source, np.float32(gain), dtype=np.float32)
        np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
        assert output[0, 0, 2] == np.float32(gain)
    np.testing.assert_array_equal(source, before)


def test_rgb_peak_is_bit_exact_and_does_not_mutate_source(dense_library):
    rng = np.random.default_rng(3107)
    source = rng.uniform(0.0, 1.0, (29, 37, 3)).astype(np.float32)
    source[0, 0] = (0.0, 1.0, 0.25)
    before = source.copy()
    status, source_arg, peak = _call_peak(dense_library, source)
    assert status == _OK
    assert peak.view(np.uint32)[0] == np.max(source_arg).view(np.uint32)
    np.testing.assert_array_equal(source, before)

    all_negative_zero = np.full((2, 3, 3), -0.0, dtype=np.float32)
    status, source_arg, peak = _call_peak(dense_library, all_negative_zero)
    assert status == _OK
    assert peak.view(np.uint32)[0] == np.max(source_arg).view(np.uint32)


def test_rgb_peak_parallel_scan_and_signed_zero_match_numpy(dense_library):
    rng = np.random.default_rng(9811)
    source = rng.uniform(0.0, 1.0, (257, 263, 3)).astype(np.float32)
    source[0, 0] = (0.0, 0.0, 0.0)
    status, source_arg, peak = _call_peak(dense_library, source)
    assert status == _OK
    assert peak.view(np.uint32)[0] == np.max(source_arg).view(np.uint32)

    signed_zeros = np.full((2, 2, 3), -0.0, dtype=np.float32)
    signed_zeros[1, 1, 2] = 0.0
    status, source_arg, peak = _call_peak(dense_library, signed_zeros)
    assert status == _OK
    assert peak.view(np.uint32)[0] == np.max(source_arg).view(np.uint32)


def test_refine_transmission_matches_float32_numpy_reference(dense_library):
    rng = np.random.default_rng(531)
    source = rng.uniform(0.0, 1.0, (23, 41, 3)).astype(np.float32)
    slope = rng.uniform(-8.0, 8.0, source.shape[:2]).astype(np.float32)
    intercept = rng.uniform(-2.0, 4.0, source.shape[:2]).astype(np.float32)
    source_before, slope_before, intercept_before = source.copy(), slope.copy(), intercept.copy()

    status, source_arg, slope_arg, intercept_arg, actual = _call_refine(
        dense_library, source, slope, intercept, 0.15, 3.2,
    )
    expected_depth = np.clip(
        slope_arg * _luma(source_arg) + intercept_arg,
        np.float32(0.15), np.float32(3.2),
    )
    expected = np.exp(-expected_depth)
    assert status == _OK
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=0)
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(slope, slope_before)
    np.testing.assert_array_equal(intercept, intercept_before)


@pytest.mark.parametrize(
    "depth_min,depth_max,slope_value,intercept_value,expected_depth",
    ((0.0, 10.0, 0.0, 2.25, 2.25), (1.75, 1.75, -8.0, 0.0, 1.75),
     (0.0, 0.5, 0.0, 4.0, 0.5)),
)
def test_refine_constant_fields_and_projection_bounds(
    dense_library, depth_min, depth_max, slope_value, intercept_value, expected_depth,
):
    source = np.full((3, 5, 3), 0.4, dtype=np.float32)
    slope = np.full((3, 5), slope_value, dtype=np.float32)
    intercept = np.full((3, 5), intercept_value, dtype=np.float32)
    status, *_args, actual = _call_refine(
        dense_library, source, slope, intercept, depth_min, depth_max,
    )
    assert status == _OK
    np.testing.assert_allclose(
        actual, np.full((3, 5), np.exp(-np.float32(expected_depth)), dtype=np.float32),
        atol=3e-6, rtol=0,
    )


def test_relief_transmission_matches_numpy_reference_and_support_boundaries(dense_library):
    rng = np.random.default_rng(309)
    source = rng.uniform(0.0, 1.0, (29, 37, 3)).astype(np.float32)
    airlight = np.array((0.67, 0.74, 0.81), dtype=np.float32)
    source[0, 0] = airlight
    source[0, 1] = np.minimum(airlight, np.float32(0.0))
    source[0, 2] = np.minimum(airlight, np.float32(0.2))
    relief = rng.uniform(0.0, 1.0, source.shape[:2]).astype(np.float32)
    relief[0, :3] = 1.0
    base, knee, initial_t = 0.72, 0.018, 0.41
    source_before, relief_before, air_before = source.copy(), relief.copy(), airlight.copy()

    status, source_arg, relief_arg, air_arg, actual = _call_relief(
        dense_library, source, relief, airlight, base, knee, initial_t,
    )
    deficit = air_arg - source_arg
    radius = np.linalg.norm(deficit, axis=2) / max(
        float(np.linalg.norm(air_arg)), 1e-6,
    )
    below_air = np.min(deficit / np.maximum(air_arg, np.float32(1e-6)), axis=2)
    support = _smoothstep(0.10, 0.25, radius) * _smoothstep(0.0, 0.05, below_air)
    target = (np.float32(base) - np.float32(knee) +
              (np.float32(1.0) - np.float32(base)) * np.float32(0.5) *
              relief_arg * support)
    expected = np.maximum(np.float32(initial_t), target)
    assert status == _OK
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=0)
    assert actual[0, 0] == pytest.approx(base - knee)
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(relief, relief_before)
    np.testing.assert_array_equal(airlight, air_before)


def test_large_image_parallel_paths_match_reference(dense_library):
    rng = np.random.default_rng(922)
    source = rng.uniform(0.0, 1.0, (257, 263, 3)).astype(np.float32)
    slope = rng.uniform(-5.0, 5.0, source.shape[:2]).astype(np.float32)
    intercept = rng.uniform(-1.0, 3.0, source.shape[:2]).astype(np.float32)
    relief = rng.uniform(0.0, 1.0, source.shape[:2]).astype(np.float32)
    airlight = np.array((0.61, 0.72, 0.83), dtype=np.float32)

    exposure_status, exposure_source, exposure = _call_exposure(dense_library, source, 2.25)
    assert exposure_status == _OK
    np.testing.assert_array_equal(
        exposure.view(np.uint32),
        np.multiply(exposure_source, np.float32(2.25), dtype=np.float32).view(np.uint32),
    )

    refine_status, refine_source, refine_slope, refine_intercept, refined = _call_refine(
        dense_library, source, slope, intercept, 0.02, 5.1,
    )
    expected_depth = np.clip(
        refine_slope * _luma(refine_source) + refine_intercept,
        np.float32(0.02), np.float32(5.1),
    )
    assert refine_status == _OK
    np.testing.assert_allclose(refined, np.exp(-expected_depth), atol=3e-6, rtol=0)

    relief_status, relief_source, relief_values, air_values, relieved = _call_relief(
        dense_library, source, relief, airlight, 0.78, 0.015, 0.5,
    )
    deficit = air_values - relief_source
    radius = np.linalg.norm(deficit, axis=2) / max(float(np.linalg.norm(air_values)), 1e-6)
    below_air = np.min(deficit / np.maximum(air_values, np.float32(1e-6)), axis=2)
    support = _smoothstep(0.10, 0.25, radius) * _smoothstep(0.0, 0.05, below_air)
    expected = np.maximum(
        np.float32(0.5), np.float32(0.78 - 0.015) +
        np.float32(1.0 - 0.78) * np.float32(0.5) * relief_values * support,
    )
    assert relief_status == _OK
    np.testing.assert_allclose(relieved, expected, atol=3e-6, rtol=0)


@pytest.mark.parametrize("gain", (0.999, 4.001, np.nan, np.inf, -np.inf))
def test_exposure_rejects_invalid_gain_without_writing(dense_library, gain):
    source = np.full((2, 3, 3), 0.5, dtype=np.float32)
    output = np.full_like(source, -9.0)
    status, *_ = _call_exposure(dense_library, source, gain, output)
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(output, np.full_like(output, -9.0))


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf, -0.01, 1.01))
def test_all_kernels_reject_invalid_rgb_without_writing(dense_library, bad):
    source = np.full((2, 3, 3), 0.5, dtype=np.float32)
    source.flat[2] = bad
    peak = np.full(1, -9.0, dtype=np.float32)
    status, *_ = _call_peak(dense_library, source, peak)
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(peak, -9.0)

    output_rgb = np.full_like(source, -9.0)
    status, *_ = _call_exposure(dense_library, source, 1.5, output_rgb)
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(output_rgb, -9.0)

    slope = np.ones((2, 3), dtype=np.float32)
    intercept = np.ones((2, 3), dtype=np.float32)
    output_scalar = np.full((2, 3), -9.0, dtype=np.float32)
    status, *_ = _call_refine(
        dense_library, source, slope, intercept, 0.0, 10.0, output_scalar,
    )
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(output_scalar, -9.0)

    relief = np.ones((2, 3), dtype=np.float32)
    airlight = np.array((0.7, 0.7, 0.7), dtype=np.float32)
    status, *_ = _call_relief(
        dense_library, source, relief, airlight, 0.8, 0.02, 0.6, output_scalar,
    )
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(output_scalar, -9.0)


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_refine_rejects_nonfinite_coefficients_without_writing(dense_library, bad):
    source = np.full((2, 3, 3), 0.5, dtype=np.float32)
    slope = np.ones((2, 3), dtype=np.float32)
    intercept = np.ones((2, 3), dtype=np.float32)
    slope[0, 0] = bad
    output = np.full((2, 3), -9.0, dtype=np.float32)
    status, *_ = _call_refine(dense_library, source, slope, intercept, 0.0, 10.0, output)
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(output, -9.0)

    slope[0, 0] = 1.0
    intercept[0, 0] = bad
    status, *_ = _call_refine(dense_library, source, slope, intercept, 0.0, 10.0, output)
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(output, -9.0)


@pytest.mark.parametrize("depth_min,depth_max", ((-0.01, 1.0), (0.0, 10.01), (2.0, 1.0),
                                                     (np.nan, 1.0), (0.0, np.inf)))
def test_refine_rejects_invalid_depth_bounds(dense_library, depth_min, depth_max):
    source = np.full((2, 3, 3), 0.5, dtype=np.float32)
    slope = np.ones((2, 3), dtype=np.float32)
    intercept = np.ones((2, 3), dtype=np.float32)
    output = np.full((2, 3), -9.0, dtype=np.float32)
    status, *_ = _call_refine(
        dense_library, source, slope, intercept, depth_min, depth_max, output,
    )
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(output, -9.0)


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf, -0.01, 1.01))
def test_relief_rejects_invalid_relief_and_airlight_without_writing(dense_library, bad):
    source = np.full((2, 3, 3), 0.5, dtype=np.float32)
    relief = np.ones((2, 3), dtype=np.float32)
    airlight = np.array((0.7, 0.7, 0.7), dtype=np.float32)
    output = np.full((2, 3), -9.0, dtype=np.float32)

    relief[0, 0] = bad
    status, *_ = _call_relief(dense_library, source, relief, airlight, 0.8, 0.02, 0.6, output)
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(output, -9.0)

    relief[0, 0] = 1.0
    airlight[0] = bad
    status, *_ = _call_relief(dense_library, source, relief, airlight, 0.8, 0.02, 0.6, output)
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(output, -9.0)


@pytest.mark.parametrize("base,knee,initial_t", (
    (-0.01, 0.01, 0.5), (1.01, 0.01, 0.5), (0.8, -0.01, 0.5),
    (0.8, 0.101, 0.5), (0.8, 0.01, -0.01), (0.8, 0.01, 1.01),
    (np.nan, 0.01, 0.5), (0.8, np.inf, 0.5), (0.8, 0.01, -np.inf),
))
def test_relief_rejects_invalid_parameters(dense_library, base, knee, initial_t):
    source = np.full((2, 3, 3), 0.5, dtype=np.float32)
    relief = np.ones((2, 3), dtype=np.float32)
    airlight = np.array((0.7, 0.7, 0.7), dtype=np.float32)
    output = np.full((2, 3), -9.0, dtype=np.float32)
    status, *_ = _call_relief(
        dense_library, source, relief, airlight, base, knee, initial_t, output,
    )
    assert status == _INVALID_ARGUMENT
    np.testing.assert_array_equal(output, -9.0)


def test_all_kernels_reject_bad_counts_dimensions_and_output_overlap(dense_library):
    source = np.full((2, 3, 3), 0.5, dtype=np.float32)
    scalar = np.full((2, 3), 0.4, dtype=np.float32)
    slope = np.full((2, 3), 0.2, dtype=np.float32)
    intercept = np.full((2, 3), 0.3, dtype=np.float32)
    relief = np.full((2, 3), 0.5, dtype=np.float32)
    airlight = np.array((0.7, 0.8, 0.9), dtype=np.float32)

    assert _call_exposure(dense_library, source, 1.5, source_count=source.size - 1)[0] == _INVALID_ARGUMENT
    assert _call_exposure(dense_library, source, 1.5, destination_count=source.size - 1)[0] == _INVALID_ARGUMENT
    assert _call_exposure(dense_library, source, 1.5, width=0)[0] == _INVALID_ARGUMENT
    assert _call_exposure(dense_library, source, 1.5, destination=source)[0] == _INVALID_ARGUMENT

    peak = np.full(1, -9.0, dtype=np.float32)
    assert _call_peak(dense_library, source, peak, source_count=source.size - 1)[0] == _INVALID_ARGUMENT
    assert _call_peak(dense_library, source, peak, width=0)[0] == _INVALID_ARGUMENT
    assert _call_peak(dense_library, source, source.reshape(-1)[:1])[0] == _INVALID_ARGUMENT
    np.testing.assert_array_equal(peak, -9.0)

    assert _call_refine(dense_library, source, slope, intercept, 0.0, 10.0,
                        source_count=source.size - 1)[0] == _INVALID_ARGUMENT
    assert _call_refine(dense_library, source, slope, intercept, 0.0, 10.0,
                        slope_count=slope.size - 1)[0] == _INVALID_ARGUMENT
    assert _call_refine(dense_library, source, slope, intercept, 0.0, 10.0,
                        intercept_count=intercept.size - 1)[0] == _INVALID_ARGUMENT
    assert _call_refine(dense_library, source, slope, intercept, 0.0, 10.0,
                        destination_count=scalar.size - 1)[0] == _INVALID_ARGUMENT
    assert _call_refine(dense_library, source, slope, intercept, 0.0, 10.0,
                        width=0)[0] == _INVALID_ARGUMENT
    assert _call_refine(dense_library, source, slope, intercept, 0.0, 10.0,
                        destination=slope)[0] == _INVALID_ARGUMENT

    assert _call_relief(dense_library, source, relief, airlight, 0.8, 0.02, 0.5,
                        source_count=source.size - 1)[0] == _INVALID_ARGUMENT
    assert _call_relief(dense_library, source, relief, airlight, 0.8, 0.02, 0.5,
                        relief_count=relief.size - 1)[0] == _INVALID_ARGUMENT
    assert _call_relief(dense_library, source, relief, airlight, 0.8, 0.02, 0.5,
                        destination_count=scalar.size - 1)[0] == _INVALID_ARGUMENT
    assert _call_relief(dense_library, source, relief, airlight, 0.8, 0.02, 0.5,
                        height=0)[0] == _INVALID_ARGUMENT
    assert _call_relief(dense_library, source, relief, airlight, 0.8, 0.02, 0.5,
                        destination=relief)[0] == _INVALID_ARGUMENT
