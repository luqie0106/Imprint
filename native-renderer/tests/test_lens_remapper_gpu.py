"""Real Metal contract tests for the streaming RGB16 lens remapper."""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import native_renderer


@pytest.fixture(scope="module")
def gpu_library():
    try:
        library, _ = native_renderer._get_library()
    except native_renderer.NativeRendererError as exc:
        if os.environ.get("IMPRINT_NATIVE_RENDERER_LIB"):
            pytest.fail(f"Requested native renderer library unavailable: {exc}")
        pytest.skip(f"Build the native renderer before Metal lens remap tests: {exc}")
    names = (
        "im_renderer_supports_lens_remap",
        "im_renderer_lens_remap_begin",
        "im_renderer_lens_remap_strip",
        "im_renderer_lens_remap_finish",
        "im_native_lens_remap_rgb16",
    )
    if any(getattr(library, name, None) is None for name in names):
        if os.environ.get("IMPRINT_NATIVE_RENDERER_LIB"):
            pytest.fail("Requested native renderer library lacks streaming lens remap ABI")
        pytest.skip("Native renderer does not export streaming lens remap ABI")
    native_renderer._configure_library(library)
    renderer = None
    try:
        renderer, backend = native_renderer._create_renderer(library, backend_kind=1)
        if backend.lower() != "metal" or not library.im_renderer_supports_lens_remap(renderer):
            if os.environ.get("IMPRINT_NATIVE_RENDERER_LIB"):
                pytest.fail("Requested native renderer has no Metal lens remap capability")
            pytest.skip("Metal lens remap is unavailable on this machine")
    except native_renderer.NativeRendererError as exc:
        if os.environ.get("IMPRINT_NATIVE_RENDERER_LIB"):
            pytest.fail(f"Requested Metal renderer could not be created: {exc}")
        pytest.skip(f"Metal renderer is unavailable: {exc}")
    finally:
        if renderer is not None:
            library.im_renderer_destroy(renderer)
    return library


def _cpu_reference(library, source, maps, map_channels, interpolation_mode, output_y):
    function = library.im_native_lens_remap_rgb16
    import ctypes as ct

    function.argtypes = [
        ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_uint16), ct.c_size_t,
        ct.POINTER(ct.c_float), ct.c_size_t, ct.c_uint32, ct.c_uint32,
        ct.POINTER(ct.c_uint16), ct.c_size_t, ct.c_uint32,
    ]
    function.restype = ct.c_int
    strip_height, width = maps.shape[:2]
    result = np.empty((strip_height, width, 3), dtype=np.uint16)
    status = function(
        width, source.shape[0],
        source.ctypes.data_as(ct.POINTER(ct.c_uint16)), source.size,
        maps.ctypes.data_as(ct.POINTER(ct.c_float)), maps.size,
        strip_height, map_channels,
        result.ctypes.data_as(ct.POINTER(ct.c_uint16)), result.size,
        interpolation_mode,
    )
    assert status == 0
    return output_y, output_y + strip_height, result


@pytest.mark.parametrize("interpolation_mode", [0, 1])
@pytest.mark.parametrize("map_channels", [1, 3])
@pytest.mark.parametrize("shape", [(7, 9), (1, 9), (7, 1), (1, 1)])
def test_streamed_metal_matches_cpu_and_reads_full_destination(
    gpu_library, interpolation_mode, map_channels, shape
):
    height, width = shape
    rng = np.random.default_rng(921 + height * 31 + width)
    source = rng.integers(0, 65536, (height, width, 3), dtype=np.uint16)
    source_before = source.copy()
    output = np.full_like(source, 0xA55A)
    chunks = []
    y = 0
    while y < height:
        strip_height = min(2, height - y)
        map_shape = ((strip_height, width, 2) if map_channels == 1
                     else (strip_height, width, 3, 2))
        maps = rng.uniform(-30.0, 30.0, map_shape).astype(np.float32)
        if strip_height and width:
            # Exercise negative coordinates, reflection at both edges, and
            # fractions near OpenCV's 1/32 ties in each shape configuration.
            flat = maps.reshape(-1)
            flat[0] = np.float32(-1.5)
            flat[-1] = np.float32((width - 1) + 0.5)
        maps_before = maps.copy()
        chunks.append((y, maps, maps_before))
        y += strip_height

    with native_renderer.NativeLensRemapper(source, interpolation_mode) as remapper:
        assert remapper.shape == source.shape
        for strip_y, maps, _ in chunks:
            remapper.remap(maps, strip_y)
        remapper.finish(output)

    expected = np.empty_like(source)
    for strip_y, maps, maps_before in chunks:
        start, end, strip = _cpu_reference(
            gpu_library, source, maps, map_channels, interpolation_mode, strip_y
        )
        expected[start:end] = strip
        np.testing.assert_array_equal(maps, maps_before)
    error = np.abs(output.astype(np.int32) - expected.astype(np.int32))
    assert int(error.max(initial=0)) <= 1
    np.testing.assert_array_equal(source, source_before)


@pytest.mark.parametrize("bad_coordinate", [np.inf, 1.0e10])
def test_native_lens_remapper_releases_renderer_after_failure(gpu_library, bad_coordinate):
    source = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    maps = np.zeros((2, 5, 2), dtype=np.float32)
    maps[0, 0, 0] = bad_coordinate
    remapper = None
    with pytest.raises(native_renderer.NativeRendererError, match="strip failed"):
        with native_renderer.NativeLensRemapper(source, 1) as remapper:
            remapper.remap(maps, 0)
    assert remapper is not None
    assert remapper._renderer is None
    remapper.close()


def test_native_lens_remapper_rejects_incomplete_and_aliased_finish(gpu_library):
    source = np.arange(3 * 4 * 3, dtype=np.uint16).reshape(3, 4, 3)
    maps = np.zeros((1, 4, 2), dtype=np.float32)
    with native_renderer.NativeLensRemapper(source, 0) as remapper:
        with pytest.raises(native_renderer.NativeRendererError, match="cannot finish"):
            remapper.finish(np.empty_like(source))
        remapper.remap(maps, 0)
        with pytest.raises(native_renderer.NativeRendererError, match="overlaps source"):
            remapper.finish(source)
