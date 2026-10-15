"""C++ CPU kernels compared against the unchanged NumPy mathematics."""
import ctypes as ct
import os
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import native_dense as dense
import native_renderer
import dehaze_nonlocal as nonlocal_impl
import dehaze_physical as physical
from dehaze import DehazeParams


@pytest.fixture(autouse=True)
def require_library():
    try:
        library, _ = native_renderer._get_library()
        if not hasattr(library, "im_native_physical_float_run"):
            raise native_renderer.NativeRendererError("dense CPU ABI missing")
    except native_renderer.NativeRendererError as exc:
        if os.environ.get("IMPRINT_NATIVE_RENDERER_LIB"):
            pytest.fail(f"Requested dense CPU library unavailable: {exc}")
        pytest.skip(f"Build the new native renderer for dense parity tests: {exc}")


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.uint8])
def test_pipeline_and_zero_strength(dtype, monkeypatch):
    source = np.random.default_rng(42).uniform(0, 1, (40, 51, 3)).astype(np.float32)
    if dtype != np.float32:
        source = np.rint(source * np.iinfo(dtype).max).astype(dtype)
    saved = source.copy()
    p = DehazeParams(strength=.8, local_contrast=.6, color_recovery=.7)
    diag = {}
    actual = physical.apply_physical_dehaze(source, p, backend="cpu", diagnostics=diag)
    assert diag["pixel_backend"].startswith("C++ CPU")
    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    expected = physical.apply_physical_dehaze(source, p, backend="cpu")
    if dtype == np.float32:
        # Match the existing GPU float contract; quantized differences remain
        # bounded to one RGB16 code value for this CPU fixture.
        assert np.max(np.abs(actual - expected)) <= 3e-6
        assert np.max(np.abs(np.rint(actual * 65535) - np.rint(expected * 65535))) <= 1
    else:
        assert np.max(np.abs(actual.astype(float) - expected.astype(float))) <= 1
    np.testing.assert_array_equal(source, saved)
    zero = physical.apply_physical_dehaze(source, DehazeParams(strength=0))
    np.testing.assert_array_equal(zero, source)
    assert not np.shares_memory(zero, source)


@pytest.mark.parametrize("floor", [0., .0001, .015, .08])
def test_dark_guard(floor):
    rng = np.random.default_rng(41)
    source = rng.uniform(0, .12, (37, 259, 3)).astype(np.float32)
    source[0, 0] = 0
    result = source * rng.uniform(0, 1, source.shape).astype(np.float32)
    expected = physical._protect_dark_background(source, result, floor)
    actual = dense.dark_guard(source[:, ::-1], result[:, ::-1], floor)[:, ::-1]
    np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=3e-6)
    assert np.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1


@pytest.mark.parametrize("edge", [33, 65])
def test_scalar_lut_edges_and_nonfinite(edge, monkeypatch):
    rng = np.random.default_rng(12)
    source = rng.uniform(-.2, 1.2, (259, 27, 3)).astype(np.float32)
    source[0, 0] = (1, 1, 1)
    source[0, 1] = (0, 0, 0)
    source[1, 0, 0] = np.nan
    source[1, 1, 2] = np.inf
    lut = rng.uniform(0, 2, (edge,) * 3).astype(np.float32)
    actual = dense.scalar_lut(source, lut)
    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    expected = nonlocal_impl._trilinear_lookup_block(source, lut)
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=2e-6)
    assert actual[1, 0] == actual[1, 1] == 0


def test_splat_matches_python_and_ignores_invalid(monkeypatch):
    rng = np.random.default_rng(45)
    image = rng.uniform(-.1, 1.1, (33, 55, 3)).astype(np.float32)
    depth = rng.uniform(0, 1, image.shape[:2]).astype(np.float32)
    confidence = rng.uniform(-.1, 1, depth.shape).astype(np.float32)
    image[0, 0, 0] = np.nan
    depth[0, 1] = np.inf
    confidence[0, 2] = np.nan
    actual, stats = nonlocal_impl._project_optical_depth_lut(image, depth, confidence, .5, .2, .8)
    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    expected, ref_stats = nonlocal_impl._project_optical_depth_lut(image, depth, confidence, .5, .2, .8)
    np.testing.assert_allclose(actual, expected, atol=6e-8, rtol=2e-6)
    assert stats["field_lut_supported_cells"] == ref_stats["field_lut_supported_cells"]


def test_invalid_native_inputs_rejected():
    source = np.ones((2, 3, 3), np.float32)
    with pytest.raises(native_renderer.NativeRendererError):
        dense.physical_pixels(source * np.nan, np.ones((2, 3)), np.ones(3), DehazeParams())
    with pytest.raises(native_renderer.NativeRendererError):
        dense.dark_guard(source, source, float("nan"))
    with pytest.raises(native_renderer.NativeRendererError):
        dense.scalar_lut(source, np.full((33,) * 3, np.inf))
    # Exercise the C ABI counts directly, before it can access source memory.
    fn = dense._function("im_native_scalar_lut_run", dense._IMAGE + [dense._F, ct.c_size_t, ct.c_uint32, dense._F, ct.c_size_t])
    lut = np.zeros((33,) * 3, np.float32)
    output = np.full((2, 3), -1., np.float32)
    status = fn(3, 2, dense._pointer(source), source.size - 1, dense._pointer(lut), lut.size, 33,
                dense._pointer(output), output.size)
    assert status != 0
    np.testing.assert_array_equal(output, -1.)


def test_missing_symbol_uses_python(monkeypatch):
    monkeypatch.setattr(native_renderer, "_get_library", lambda: (object(), "old-library"))
    monkeypatch.setattr(dense, "_get_library", native_renderer._get_library)
    source = np.full((23, 31, 3), .03, np.float32)
    diag = {}
    physical.apply_physical_dehaze(source, DehazeParams(strength=.8), backend="auto", diagnostics=diag)
    assert diag["pixel_backend"].startswith("Python CPU")
    assert diag["dark_guard_backend"] == "python_cpu"


def test_cpu_status_survives_gpu_creation_failure(monkeypatch):
    library = SimpleNamespace(im_native_physical_float_run=object(), im_renderer_render_physical_float=object())
    monkeypatch.setattr(native_renderer, "_get_library", lambda: (library, "cpu-library"))
    def unavailable(_):
        raise native_renderer.NativeRendererError("No GPU")
    monkeypatch.setattr(native_renderer, "_create_renderer", unavailable)
    assert native_renderer.get_native_physical_status() == {
        "available": True, "backend": "cpp_cpu", "gpu_available": False, "cpu_available": True}


def test_parallel_blocks_match_reference_without_seams(monkeypatch):
    rng = np.random.default_rng(86)
    # Exceed the C++ parallel threshold, including a final partial partition.
    source = rng.uniform(0, .8, (257, 259, 3)).astype(np.float32)
    original = source.copy()
    t = rng.uniform(.2, 1, source.shape[:2]).astype(np.float32)
    air = np.array([.5, .6, .7], np.float32)
    p = DehazeParams(strength=.87, local_contrast=.3, color_recovery=.4)
    actual = dense.physical_pixels(source, t, air, p)
    expected = physical._physical_pixels(source, t, air, p)
    assert np.max(np.abs(actual - expected)) <= 3e-6
    np.testing.assert_array_equal(actual, dense.physical_pixels(source, t, air, p))
    rendered = source * .4
    np.testing.assert_allclose(dense.dark_guard(source, rendered, .1),
                               physical._protect_dark_background(source, rendered, .1),
                               atol=3e-8, rtol=3e-6)
    lut = rng.random((33,) * 3, dtype=np.float32)
    actual_lut = dense.scalar_lut(source, lut)
    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    np.testing.assert_array_equal(actual_lut, nonlocal_impl._trilinear_lookup_block(source, lut))
    np.testing.assert_array_equal(source, original)
