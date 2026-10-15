"""Native RGB16 lens remap parity for both OpenCV float-map paths."""
import ctypes as ct
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import native_renderer


@pytest.fixture(scope="module")
def remap_function():
    try:
        library, _ = native_renderer._get_library()
    except native_renderer.NativeRendererError as exc:
        if os.environ.get("IMPRINT_NATIVE_RENDERER_LIB"):
            pytest.fail(f"Requested native renderer library unavailable: {exc}")
        pytest.skip(f"Build the native renderer before lens remap parity tests: {exc}")
    if not hasattr(library, "im_native_lens_remap_rgb16"):
        if os.environ.get("IMPRINT_NATIVE_RENDERER_LIB"):
            pytest.fail("Requested library lacks lens remap ABI")
        pytest.skip("Native renderer does not export im_native_lens_remap_rgb16")

    function = library.im_native_lens_remap_rgb16
    function.argtypes = [
        ct.c_uint32,
        ct.c_uint32,
        ct.POINTER(ct.c_uint16),
        ct.c_size_t,
        ct.POINTER(ct.c_float),
        ct.c_size_t,
        ct.c_uint32,
        ct.c_uint32,
        ct.POINTER(ct.c_uint16),
        ct.c_size_t,
        ct.c_uint32,
    ]
    function.restype = ct.c_int
    return function


def _pointer(array, ctype):
    return array.ctypes.data_as(ct.POINTER(ctype))


def _invoke(function, source, maps, map_channels, interpolation_mode, *,
            source_values=None, map_values=None, destination=None):
    source = np.ascontiguousarray(source, dtype=np.uint16)
    maps = np.ascontiguousarray(maps, dtype=np.float32)
    strip_height, width = maps.shape[:2]
    if destination is None:
        destination = np.full((strip_height, width, 3), 0xA55A, dtype=np.uint16)
    status = function(
        width,
        source.shape[0],
        _pointer(source, ct.c_uint16),
        source.size if source_values is None else source_values,
        _pointer(maps, ct.c_float),
        maps.size if map_values is None else map_values,
        strip_height,
        map_channels,
        _pointer(destination, ct.c_uint16),
        destination.size,
        interpolation_mode,
    )
    return status, destination, source, maps


def _maps_for_channel(maps, map_channels, channel):
    if map_channels == 1:
        return maps[..., 0], maps[..., 1]
    return maps[:, :, channel, 0], maps[:, :, channel, 1]


def _fixed_map_reference(source, maps, map_channels):
    strip_height, width = maps.shape[:2]
    expected = np.empty((strip_height, width, 3), dtype=np.uint16)
    for channel in range(3):
        map_x, map_y = _maps_for_channel(maps, map_channels, channel)
        fixed_map, fractions = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        expected[..., channel] = cv2.remap(
            source[..., channel], fixed_map, fractions, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
    return expected


def _reflect101_coordinates(coordinates, extent):
    if extent <= 1:
        return np.zeros_like(coordinates, dtype=np.float64)
    edge = float(extent - 1)
    period = 2.0 * edge
    folded = np.mod(coordinates.astype(np.float64), period)
    return np.where(folded > edge, period - folded, folded)


def _continuous_reference(source, maps, map_channels):
    strip_height, width = maps.shape[:2]
    expected = np.empty((strip_height, width, 3), dtype=np.uint16)
    for channel in range(3):
        map_x, map_y = _maps_for_channel(maps, map_channels, channel)
        x = _reflect101_coordinates(map_x, source.shape[1])
        y = _reflect101_coordinates(map_y, source.shape[0])
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx, fy = x - x0, y - y0
        x1 = np.where(x0 + 1 >= source.shape[1], source.shape[1] - 2, x0 + 1)
        y1 = np.where(y0 + 1 >= source.shape[0], source.shape[0] - 2, y0 + 1)
        p00 = source[y0, x0, channel].astype(np.float64)
        p01 = source[y0, x1, channel].astype(np.float64)
        p10 = source[y1, x0, channel].astype(np.float64)
        p11 = source[y1, x1, channel].astype(np.float64)
        top = p00 * (1.0 - fx) + p01 * fx
        bottom = p10 * (1.0 - fx) + p11 * fx
        expected[..., channel] = np.clip(np.rint(top * (1.0 - fy) + bottom * fy),
                                         0, 65535).astype(np.uint16)
    return expected


@pytest.mark.parametrize("map_channels", [1, 3])
def test_fixed_mode_matches_opencv_convert_maps(remap_function, map_channels):
    rng = np.random.default_rng(14)
    height, width, strip_height = 111, 167, 45
    source = rng.integers(0, 65536, (height, width, 3), dtype=np.uint16)
    map_shape = ((strip_height, width, 2) if map_channels == 1
                 else (strip_height, width, 3, 2))
    maps = rng.uniform(-400, 400, map_shape).astype(np.float32)
    source_before, maps_before = source.copy(), maps.copy()

    status, actual, _, _ = _invoke(remap_function, source, maps, map_channels, 0)
    expected = _fixed_map_reference(source, maps, map_channels)
    assert status == 0
    assert np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32))) <= 1
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(maps, maps_before)


@pytest.mark.parametrize("map_channels", [1, 3])
def test_continuous_mode_matches_numpy_reference(remap_function, map_channels):
    rng = np.random.default_rng(1401)
    height, width, strip_height = 19, 23, 11
    source = rng.integers(0, 65536, (height, width, 3), dtype=np.uint16)
    map_shape = ((strip_height, width, 2) if map_channels == 1
                 else (strip_height, width, 3, 2))
    maps = rng.uniform(-80, 80, map_shape).astype(np.float32)
    status, actual, _, _ = _invoke(remap_function, source, maps, map_channels, 1)
    expected = _continuous_reference(source, maps, map_channels)
    assert status == 0
    assert np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32))) <= 1


@pytest.mark.parametrize("interpolation_mode", [0, 1])
def test_boundary_and_tie_coordinates(remap_function, interpolation_mode):
    source = np.arange(5 * 7 * 3, dtype=np.uint16).reshape(5, 7, 3) * np.uint16(997)
    values = np.array([-9.0, -1.5, -0.5, -0.015625, 0.0, 0.015625,
                       0.5 / 32.0, 1.5 / 32.0, 1.5, 5.5, 6.0, 7.0], np.float32)
    maps = np.empty((5, 7, 2), dtype=np.float32)
    maps[..., 0] = np.resize(values, maps.shape[:2])
    maps[..., 1] = np.resize(values[::-1], maps.shape[:2])
    status, actual, source_after, maps_after = _invoke(
        remap_function, source, maps, 1, interpolation_mode
    )
    if interpolation_mode == 0:
        expected = _fixed_map_reference(source, maps, 1)
    else:
        expected = _continuous_reference(source, maps, 1)
    assert status == 0
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source_after, source)
    np.testing.assert_array_equal(maps_after, maps)


def test_invalid_maps_counts_mode_and_overlap_leave_destination_unchanged(remap_function):
    source = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    maps = np.zeros((2, 5, 2), dtype=np.float32)
    destination = np.full((2, 5, 3), 12345, dtype=np.uint16)
    invalid_maps = maps.copy()
    invalid_maps[0, 0, 0] = np.nan

    status, actual, _, _ = _invoke(
        remap_function, source, invalid_maps, 1, 1, destination=destination.copy()
    )
    assert status != 0
    np.testing.assert_array_equal(actual, 12345)

    for kwargs in ({"map_values": maps.size - 1}, {"source_values": source.size - 1}):
        destination = np.full((2, 5, 3), 12345, dtype=np.uint16)
        status, actual, _, _ = _invoke(
            remap_function, source, maps, 1, 1, destination=destination, **kwargs
        )
        assert status != 0
        np.testing.assert_array_equal(actual, 12345)

    destination = np.full((2, 5, 3), 12345, dtype=np.uint16)
    status, actual, _, _ = _invoke(
        remap_function, source, maps, 1, 2, destination=destination
    )
    assert status != 0
    np.testing.assert_array_equal(actual, 12345)

    overlapping = source.copy()
    status = remap_function(
        5, 4, _pointer(overlapping, ct.c_uint16), overlapping.size,
        _pointer(maps, ct.c_float), maps.size, 2, 1,
        _pointer(overlapping, ct.c_uint16), overlapping.size, 1,
    )
    assert status != 0


@pytest.mark.parametrize("map_channels", [1, 3])
def test_python_bridge_matches_installed_opencv_float_maps(remap_function, map_channels):
    import native_dense
    rng = np.random.default_rng(14)
    source = rng.integers(0, 65536, (111, 167, 3), dtype=np.uint16)
    shape = (45, 167, 2) if map_channels == 1 else (45, 167, 3, 2)
    maps = rng.uniform(-400, 400, shape).astype(np.float32)
    actual = native_dense.lens_remap(source, maps)
    expected = np.empty_like(actual)
    for channel in range(3):
        x, y = _maps_for_channel(maps, map_channels, channel)
        expected[..., channel] = cv2.remap(source[..., channel], x, y, cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT101)
    assert np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32))) <= 1


@pytest.mark.parametrize("shape", [(1, 7, 3), (5, 1, 3)])
def test_singleton_axes_are_safe(remap_function, shape):
    import native_dense
    source = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) * np.uint16(997)
    maps = np.full((*shape[:2], 2), -.25, dtype=np.float32)
    actual = native_dense.lens_remap(source, maps)
    expected = np.empty_like(actual)
    for channel in range(3):
        expected[..., channel] = cv2.remap(source[..., channel], maps[..., 0], maps[..., 1],
                                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    assert np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32))) <= 1
