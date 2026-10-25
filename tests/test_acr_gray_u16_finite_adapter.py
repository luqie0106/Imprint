"""Exercise the existing Python adapters against the formal native library."""

from __future__ import annotations

import ctypes as C
import ctypes.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/acr_support"))
import native_renderer as native  # noqa: E402
import gray_u16_finite_oracle as oracle  # noqa: E402


@pytest.fixture(scope="module")
def formal_library():
    candidates = [ROOT / "native-renderer/build/libimprint_renderer.0.1.0.dylib",
                  ROOT / "native-renderer/build/libimprint_renderer.so"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        pytest.skip("formal renderer library is not built")
    library = C.CDLL(str(path))
    native._configure_library(library)
    return library, str(path)


@pytest.fixture
def adapter(monkeypatch, formal_library):
    monkeypatch.setattr(native, "_get_library", lambda: formal_library)
    return native


def _log32(value):
    library = C.CDLL(ctypes.util.find_library("m") or None)
    function = library.log2f
    function.argtypes = [C.c_float]
    function.restype = C.c_float
    return np.float32(function(float(np.float32(value))))


def _source_log(values, epsilon):
    epsilon = np.float32(epsilon)
    twice_log_epsilon = np.float32(_log32(epsilon) + _log32(epsilon))
    output = np.empty(np.shape(values), dtype=np.float32)
    for index in np.ndindex(output.shape):
        value = np.float32(values[index])
        mapped = _log32(np.float32(epsilon + abs(value)))
        output[index] = mapped if value >= 0 else np.float32(twice_log_epsilon - mapped)
    return output


@pytest.mark.parametrize("dither", [False, True])
def test_existing_u16_log_adapter_accepts_signed_and_overrange_rgb(adapter, dither):
    green = np.array([[-0.1, 0.0, 0.25, 1.0, 1.1]], dtype=np.float32)
    rgb = np.repeat(green[:, :, None], 3, axis=2)
    before = rgb.view(np.uint32).copy()
    epsilon = 2.0 ** -16
    output, diagnostics = adapter.native_acr_gray_u16_log(
        rgb, [0.0, 1.0, 0.0], epsilon,
        origin_x=-129, origin_y=257, dither=dither)
    stored = oracle.apply_gray_u16(green, 5, 1, -129, 257, dither=dither)
    reloaded = stored.astype(np.float32) * np.float32(1.0 / 65535.0)
    expected = _source_log(reloaded, epsilon)
    np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(diagnostics, [0, 16384, 0, 0, 1, 0])
    np.testing.assert_array_equal(rgb.view(np.uint32), before)
    if dither:
        assert stored[0, 0] == 0 and stored[0, -1] == 65535
    else:
        assert stored[0, 0] > 0 and stored[0, -1] < 65535


def test_existing_float_log_adapter_preserves_signed_gray(adapter):
    green = np.array([[-1.1, -0.1, 0.0, 0.25, 1.1]], dtype=np.float32)
    rgb = np.repeat(green[:, :, None], 3, axis=2)
    epsilon = 2.0 ** -16
    output, _diagnostics = adapter.native_acr_gray_log(rgb, [0.0, 1.0, 0.0], epsilon)
    expected = _source_log(green, epsilon)
    np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))


def test_existing_u16_adapter_handles_scaled_float_overflow(adapter):
    maximum = np.finfo(np.float32).max
    green = np.array([[-maximum, maximum]], dtype=np.float32)
    rgb = np.repeat(green[:, :, None], 3, axis=2)
    epsilon = 2.0 ** -16
    expected = _source_log(np.array([[0.0, 1.0]], dtype=np.float32), epsilon)
    for dither in (False, True):
        output, _diagnostics = adapter.native_acr_gray_u16_log(
            rgb, [0.0, 1.0, 0.0], epsilon, dither=dither)
        np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_adapters_still_reject_nonfinite_rgb(adapter, value):
    rgb = np.zeros((1, 1, 3), dtype=np.float32)
    rgb[0, 0, 1] = value
    for function in (adapter.native_acr_gray_log, adapter.native_acr_gray_u16_log):
        with pytest.raises(ValueError, match="finite"):
            function(rgb, [0.0, 1.0, 0.0], 2.0 ** -16)
