"""Metal physical float dehaze parity and Python ABI validation."""

from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import native_renderer  # noqa: E402
from dehaze import DehazeParams  # noqa: E402
from dehaze_physical import _physical_pixels  # noqa: E402
from native_renderer import NativeRendererError, native_physical_dehaze  # noqa: E402


def _inputs(case: str):
    rng = np.random.default_rng(4821)
    if case == "random":
        source = rng.uniform(0.0, 1.0, (31, 53, 3)).astype(np.float32)
        transmission = rng.uniform(0.22, 1.0, (31, 53)).astype(np.float32)
        airlight = np.array((0.66, 0.71, 0.75), dtype=np.float32)
    elif case == "gray-ramp":
        ramp = np.linspace(0.0, 1.0, 257, dtype=np.float32)
        source = np.broadcast_to(ramp[None, :, None], (7, 257, 3)).copy()
        source[2:5, 116:142] = np.array((0.24, 0.30, 0.37), dtype=np.float32)
        source[1:6, 206:245] = np.array((0.985, 0.965, 0.91), dtype=np.float32)
        transmission = np.broadcast_to(
            np.linspace(0.24, 0.98, 257, dtype=np.float32)[None, :], (7, 257)
        ).copy()
        airlight = np.array((0.72, 0.70, 0.69), dtype=np.float32)
    else:
        source = rng.uniform(0.08, 0.93, (19, 23, 3)).astype(np.float32)
        source[:, :8] *= np.array((1.0, 0.82, 0.64), dtype=np.float32)
        source[3:13, 11:20] = np.array((0.996, 0.963, 0.91), dtype=np.float32)
        y, x = np.mgrid[:19, :23]
        transmission = (0.24 + 0.72 * (x + y) / (22 + 18)).astype(np.float32)
        airlight = np.array((0.59, 0.68, 0.77), dtype=np.float32)
    return source, transmission, airlight


def _params():
    return DehazeParams(
        strength=0.84, naturalness=0.43, fog_retention=0.52,
        local_contrast=0.61, color_recovery=0.73, color_protection=0.66,
        highlight_protection=0.81, shadow_protection=0.72,
        brightness_protection=0.76,
    )


@pytest.mark.parametrize("case", ("random", "gray-ramp", "cast-nearwhite"))
def test_metal_physical_float_matches_numpy_reference(case):
    requested = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND")
    if requested and requested.lower() != "metal":
        pytest.skip(f"Physical float parity is implemented for Metal, not {requested}")
    source, transmission, airlight = _inputs(case)
    original = source.copy()
    params = _params()
    expected = _physical_pixels(source, transmission, airlight, params)
    try:
        actual = native_physical_dehaze(source, params, transmission, airlight)
    except NativeRendererError as exc:
        if requested and requested.lower() == "metal":
            pytest.fail(f"Requested Metal physical float operator failed: {exc}")
        pytest.skip(f"Metal physical float renderer is unavailable: {exc}")

    assert actual.shape == source.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual - expected)) <= 3e-6
    actual_u16 = np.rint(actual * 65535.0).astype(np.int64)
    expected_u16 = np.rint(expected * 65535.0).astype(np.int64)
    assert np.max(np.abs(actual_u16 - expected_u16)) <= 2
    np.testing.assert_array_equal(source, original)
    assert native_renderer.get_last_native_physical_backend() == "Metal"


@pytest.mark.parametrize(
    "params", (DehazeParams(strength=0.7), _params()),
    ids=("default-parameters-strength-0.7", "nontrivial-parameters"),
)
def test_metal_physical_float_nearwhite_clipped_channels_match_reference(params):
    requested = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND")
    if requested and requested.lower() != "metal":
        pytest.skip(f"Physical float parity is implemented for Metal, not {requested}")
    nearwhite = np.array(
        ((1.0, 1.0, 0.90624857), (1.0, 1.0, 0.98956281), (1.0, 1.0, 0.999)),
        dtype=np.float32,
    )
    yy, xx = np.mgrid[:18, :33]
    source = nearwhite[(xx + 2 * yy) % len(nearwhite)].copy()
    transmission = (0.85 + 0.15 * (xx + yy) / (32 + 17)).astype(np.float32)
    airlight = np.array((0.2342, 0.21455, 0.1864), dtype=np.float32)
    original = source.copy()
    expected = _physical_pixels(source, transmission, airlight, params)
    try:
        actual = native_physical_dehaze(source, params, transmission, airlight)
    except NativeRendererError as exc:
        if requested and requested.lower() == "metal":
            pytest.fail(f"Requested Metal physical float operator failed: {exc}")
        pytest.skip(f"Metal physical float renderer is unavailable: {exc}")

    assert actual.shape == source.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual - expected)) <= 3e-6
    actual_u16 = np.rint(actual * 65535.0).astype(np.int64)
    expected_u16 = np.rint(expected * 65535.0).astype(np.int64)
    assert np.max(np.abs(actual_u16 - expected_u16)) <= 2
    np.testing.assert_array_equal(source, original)
    assert native_renderer.get_last_native_physical_backend() == "Metal"


def test_old_renderer_without_optional_float_abi_raises(monkeypatch):
    monkeypatch.setattr(native_renderer, "_get_library",
                        lambda: (SimpleNamespace(), "old-renderer"))
    source, transmission, airlight = _inputs("random")
    with pytest.raises(NativeRendererError, match="ABI is unavailable"):
        native_physical_dehaze(source, _params(), transmission, airlight)


def test_physical_status_marks_old_renderer_unavailable(monkeypatch):
    monkeypatch.setattr(native_renderer, "_get_library",
                        lambda: (SimpleNamespace(), "/private/local/old-renderer.dylib"))
    monkeypatch.setattr(native_renderer, "_create_renderer",
                        lambda _library: pytest.fail("Old ABI must not create a renderer"))
    status = native_renderer.get_native_physical_status()
    assert status == {"available": False, "backend": None,
                      "gpu_available": False, "cpu_available": False}


@pytest.mark.parametrize("backend", ("D3D12", "CUDA", "Unknown"))
def test_physical_status_rejects_non_metal_backend_and_destroys_renderer(backend, monkeypatch):
    destroyed = []
    library = SimpleNamespace(
        im_renderer_render_physical_float=object(),
        im_renderer_destroy=lambda renderer: destroyed.append(renderer),
    )
    monkeypatch.setattr(native_renderer, "_get_library", lambda: (library, "/private/local/lib.dylib"))
    monkeypatch.setattr(native_renderer, "_create_renderer",
                        lambda _library: ("fake-renderer", backend))
    status = native_renderer.get_native_physical_status()
    assert status == {"available": False, "backend": None,
                      "gpu_available": False, "cpu_available": False}
    assert destroyed == ["fake-renderer"]


def test_physical_status_reports_actual_metal_backend_and_never_reports_cpu(monkeypatch):
    destroyed = []
    library = SimpleNamespace(
        im_renderer_render_physical_float=object(),
        im_renderer_destroy=lambda renderer: destroyed.append(renderer),
    )
    monkeypatch.setattr(native_renderer, "_get_library", lambda: (library, "/private/local/lib.dylib"))
    monkeypatch.setattr(native_renderer, "_create_renderer",
                        lambda _library: ("metal-renderer", "Metal"))
    status = native_renderer.get_native_physical_status()
    assert status == {"available": True, "backend": "Metal",
                      "gpu_available": True, "cpu_available": False}
    assert destroyed == ["metal-renderer"]


def test_physical_status_fails_safe_without_leaking_library_path(monkeypatch):
    monkeypatch.setattr(
        native_renderer, "_get_library",
        lambda: (_ for _ in ()).throw(NativeRendererError("missing /private/user/native.dylib")),
    )
    status = native_renderer.get_native_physical_status()
    assert status == {"available": False, "backend": None,
                      "gpu_available": False, "cpu_available": False}
    assert "/private" not in repr(status)


@pytest.mark.parametrize("which", ("source", "transmission", "airlight", "params"))
def test_physical_float_bridge_rejects_invalid_inputs_before_loading_native(which, monkeypatch):
    def unexpected_load():
        pytest.fail("Invalid input must be rejected before attempting to load a native library")

    monkeypatch.setattr(native_renderer, "_get_library", unexpected_load)
    source, transmission, airlight = _inputs("random")
    params = _params()
    if which == "source":
        source[0, 0, 0] = np.nan
        error = ValueError
    elif which == "transmission":
        transmission[0, 0] = 1.01
        error = ValueError
    elif which == "airlight":
        airlight[0] = -0.01
        error = ValueError
    else:
        params = {"strength": 0.5}
        error = ValueError
    with pytest.raises(error):
        native_physical_dehaze(source, params, transmission, airlight)


def test_zero_strength_metal_physical_float_is_an_exact_copy():
    requested = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND")
    if requested and requested.lower() != "metal":
        pytest.skip(f"Physical float parity is implemented for Metal, not {requested}")
    source, transmission, airlight = _inputs("cast-nearwhite")
    params = DehazeParams(strength=0.0)
    try:
        actual = native_physical_dehaze(source, params, transmission, airlight)
    except NativeRendererError as exc:
        if requested and requested.lower() == "metal":
            pytest.fail(f"Requested Metal physical float operator failed: {exc}")
        pytest.skip(f"Metal physical float renderer is unavailable: {exc}")
    np.testing.assert_array_equal(actual, source)
