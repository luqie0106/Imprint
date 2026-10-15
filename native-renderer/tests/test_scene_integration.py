from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import auto_exposure  # noqa: E402
import dehaze_physical  # noqa: E402
import native_dense  # noqa: E402
import native_renderer  # noqa: E402
from dehaze import DehazeParams  # noqa: E402
from native_renderer import NativeRendererError  # noqa: E402


def _exposure_scene() -> np.ndarray:
    rng = np.random.default_rng(20261008)
    image = rng.uniform(.025, .075, (13, 17, 3)).astype(np.float32)
    image[4, 9] = (.40, .31, .24)
    return image


@pytest.mark.parametrize(
    ("enabled", "ev_override", "reason_override"),
    [
        (True, None, None),
        (False, None, None),
        (True, .6, None),
        (True, .6, "applied"),
    ],
)
def test_auto_exposure_native_peak_matches_dense_disabled(
    monkeypatch, enabled, ev_override, reason_override
):
    image = _exposure_scene()
    before = image.copy()
    diagnostics_native = {}

    monkeypatch.delenv("IMPRINT_NATIVE_DENSE", raising=False)
    try:
        native_dense.rgb_peak(image)
    except NativeRendererError as error:
        pytest.skip(f"native RGB peak kernel unavailable: {error}")
    output_native = auto_exposure.apply_auto_exposure(
        image,
        enabled=enabled,
        ev_override=ev_override,
        reason_override=reason_override,
        diagnostics=diagnostics_native,
    )

    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    diagnostics_python = {}
    output_python = auto_exposure.apply_auto_exposure(
        image,
        enabled=enabled,
        ev_override=ev_override,
        reason_override=reason_override,
        diagnostics=diagnostics_python,
    )

    np.testing.assert_array_equal(output_native, output_python)
    assert diagnostics_native == diagnostics_python
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -0.01, 1.01])
@pytest.mark.parametrize("dense_enabled", [True, False])
def test_auto_exposure_rejects_nonfinite_and_out_of_range_inputs(
    monkeypatch, bad_value, dense_enabled
):
    image = np.full((3, 4, 3), .1, dtype=np.float32)
    image[1, 2, 0] = bad_value
    before = image.copy()
    if dense_enabled:
        monkeypatch.delenv("IMPRINT_NATIVE_DENSE", raising=False)
    else:
        monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")

    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        auto_exposure.estimate_auto_exposure(image)
    np.testing.assert_array_equal(image, before)


def _transmission_fixture() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:48, :57].astype(np.float32)
    luma = .04 + .32 * x / 56 + .09 * y / 47
    source = np.stack((luma, luma * .91, luma * .78), axis=2).astype(np.float32)
    transmission = (.24 + .59 * (x / 56) * (y / 47)).astype(np.float32)
    return source, transmission


def test_regularize_transmission_native_refine_matches_python_reference(monkeypatch):
    source, transmission = _transmission_fixture()
    source_before = source.copy()
    transmission_before = transmission.copy()
    native_refine = native_dense.refine_transmission
    successful_calls = []

    def tracked_refine(*args):
        result = native_refine(*args)
        successful_calls.append(True)
        return result

    monkeypatch.delenv("IMPRINT_NATIVE_DENSE", raising=False)
    monkeypatch.setattr(native_dense, "refine_transmission", tracked_refine)
    output_native = dehaze_physical._regularize_transmission(source, transmission)
    if not successful_calls:
        pytest.skip("native transmission refinement kernel unavailable")

    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    output_python = dehaze_physical._regularize_transmission(source, transmission)

    assert output_native.shape == transmission.shape
    assert output_native.dtype == np.float32
    assert np.isfinite(output_native).all()
    np.testing.assert_allclose(output_native, output_python, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(transmission, transmission_before)


def test_refine_transmission_native_error_uses_python_fallback(monkeypatch):
    source, transmission = _transmission_fixture()
    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    expected = dehaze_physical._regularize_transmission(source, transmission)
    calls = []

    def unavailable(*args):
        calls.append(True)
        raise NativeRendererError("simulated old native library")

    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "1")
    monkeypatch.setattr(native_dense, "refine_transmission", unavailable)
    actual = dehaze_physical._regularize_transmission(source, transmission)

    assert calls == [True]
    np.testing.assert_array_equal(actual, expected)


def _conservative_relief_scene(monkeypatch, source: np.ndarray) -> tuple[np.ndarray, DehazeParams]:
    height, width = source.shape[:2]
    y, x = np.mgrid[:height, :width].astype(np.float32)
    relief = ((x + y) / max(width + height - 2, 1)).astype(np.float32)
    airlight = np.array((.80, .78, .76), dtype=np.float32)
    params = DehazeParams(strength=.8).normalized()

    monkeypatch.setattr(
        dehaze_physical, "_estimate_airlight",
        lambda _source, _params: (airlight.copy(), .99),
    )
    monkeypatch.setattr(
        dehaze_physical, "_global_transmission", lambda _source, _params: (.78, .3)
    )
    monkeypatch.setattr(dehaze_physical, "_backlit_optical_scale", lambda *_args: 1.0)
    monkeypatch.setattr(
        dehaze_physical,
        "build_reliability_lut",
        lambda *_args: (np.ones((2, 2, 2), dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        dehaze_physical, "lookup_transmission_relief", lambda *_args: relief.copy()
    )
    return relief, params


def _relief_source() -> np.ndarray:
    y, x = np.mgrid[:12, :15].astype(np.float32)
    value = .10 + .17 * x / 14 + .07 * y / 11
    return np.stack((value, value * .91, value * .82), axis=2).astype(np.float32)


def test_conservative_relief_native_matches_python_reference(monkeypatch):
    source = _relief_source()
    _relief, params = _conservative_relief_scene(monkeypatch, source)
    native_relief = native_dense.relief_transmission
    successful_calls = []

    def tracked_relief(*args):
        result = native_relief(*args)
        successful_calls.append(True)
        return result

    monkeypatch.delenv("IMPRINT_NATIVE_DENSE", raising=False)
    monkeypatch.setattr(native_dense, "relief_transmission", tracked_relief)
    output_native, _air_native, stats_native = dehaze_physical._estimate_scene(
        source, params, True, "conservative"
    )
    if not successful_calls:
        pytest.skip("native conservative transmission relief kernel unavailable")

    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    output_python, _air_python, stats_python = dehaze_physical._estimate_scene(
        source, params, True, "conservative"
    )

    assert stats_native["nonlocal_active"] is True
    assert stats_python["nonlocal_active"] is True
    np.testing.assert_allclose(output_native, output_python, rtol=2e-5, atol=2e-6)


def test_conservative_relief_native_error_uses_python_fallback(monkeypatch):
    source = _relief_source()
    _relief, params = _conservative_relief_scene(monkeypatch, source)
    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    expected, _air, _stats = dehaze_physical._estimate_scene(
        source, params, True, "conservative"
    )
    calls = []

    def unavailable(*args):
        calls.append(True)
        raise NativeRendererError("simulated old native library")

    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "1")
    monkeypatch.setattr(native_dense, "relief_transmission", unavailable)
    actual, _air, stats = dehaze_physical._estimate_scene(
        source, params, True, "conservative"
    )

    assert calls == [True]
    assert stats["nonlocal_active"] is True
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("backend", "override", "expected_call"),
    [
        ("Metal", None, "guarded"),
        ("D3D12", None, "plain"),
        ("Metal", "0", "plain"),
        ("Metal", "1", "guarded"),
        ("D3D12", "0", "plain"),
        ("D3D12", "1", "guarded"),
    ],
)
def test_guarded_gpu_backend_default_and_environment_override(
    monkeypatch, backend, override, expected_call
):
    calls = []

    class FakeLibrary:
        def im_renderer_render_physical_float(self, *_args):
            calls.append("plain")
            return 0

        def im_renderer_render_physical_guarded_float(self, *_args):
            calls.append("guarded")
            return 0

        def im_renderer_supports_physical_guarded_float(self, _renderer):
            return 1

        def im_renderer_destroy(self, _renderer):
            pass

    library = FakeLibrary()
    monkeypatch.setattr(native_renderer, "_get_library", lambda: (library, "fake"))
    monkeypatch.setattr(native_renderer, "_create_renderer", lambda _library: ("renderer", backend))
    monkeypatch.setattr(native_renderer, "_supports_physical_float", lambda *_args: True)
    if override is None:
        monkeypatch.delenv("IMPRINT_NATIVE_GUARDED_FLOAT", raising=False)
    else:
        monkeypatch.setenv("IMPRINT_NATIVE_GUARDED_FLOAT", override)

    source = np.full((3, 4, 3), .08, dtype=np.float32)
    transmission = np.full((3, 4), .7, dtype=np.float32)
    atmosphere = np.array((.3, .32, .34), dtype=np.float32)
    output = native_renderer.native_physical_dehaze(
        source, DehazeParams(strength=.7), transmission, atmosphere, dark_floor=.04
    )

    assert output.shape == source.shape
    assert calls == [expected_call]
    assert native_renderer.get_last_native_physical_guarded() is (expected_call == "guarded")


def _patch_dark_guard_scene(monkeypatch, source: np.ndarray) -> None:
    transmission = np.full(source.shape[:2], .72, dtype=np.float32)
    atmosphere = np.full(3, .03, dtype=np.float32)
    monkeypatch.setattr(
        dehaze_physical,
        "_estimate_scene",
        lambda *_args: (transmission.copy(), atmosphere.copy(), {"nonlocal_active": False}),
    )


def test_fused_gpu_dark_guard_does_not_repeat_cpu_guard(monkeypatch):
    source = np.full((8, 10, 3), (.04, .035, .03), dtype=np.float32)
    _patch_dark_guard_scene(monkeypatch, source)
    monkeypatch.setattr(
        native_renderer,
        "native_physical_dehaze",
        lambda image, *_args, **_kwargs: np.full_like(image, .01),
    )
    monkeypatch.setattr(native_renderer, "get_last_native_physical_backend", lambda: "Metal")
    monkeypatch.setattr(native_renderer, "get_last_native_physical_guarded", lambda: True)

    def unexpected_guard(*_args):
        pytest.fail("fused GPU dark guard must not run a second CPU guard")

    monkeypatch.setattr(native_dense, "physical_pixels", unexpected_guard)
    monkeypatch.setattr(native_dense, "dark_guard", unexpected_guard)
    diagnostics = {}
    result = dehaze_physical.apply_physical_dehaze(
        source, DehazeParams(strength=.8, brightness_protection=.9),
        backend="native", diagnostics=diagnostics,
    )

    np.testing.assert_array_equal(result, np.full_like(source, .01))
    assert diagnostics["dark_guard_backend"] == "gpu_fused"


def test_gpu_dark_guard_unavailable_runs_cpu_guard(monkeypatch):
    source = np.full((8, 10, 3), (.04, .035, .03), dtype=np.float32)
    _patch_dark_guard_scene(monkeypatch, source)
    gpu_calls = []

    def gpu_unavailable(*_args, **_kwargs):
        gpu_calls.append(True)
        raise NativeRendererError("simulated unavailable GPU")

    monkeypatch.setattr(native_renderer, "native_physical_dehaze", gpu_unavailable)
    cpu_pixels = np.full_like(source, .01)
    monkeypatch.setattr(native_dense, "physical_pixels", lambda *_args: cpu_pixels.copy())
    guard_calls = []
    guarded_pixels = np.full_like(source, .025)

    def cpu_guard(image, result, floor):
        guard_calls.append((image.copy(), result.copy(), floor))
        return guarded_pixels.copy()

    monkeypatch.setattr(native_dense, "dark_guard", cpu_guard)
    diagnostics = {}
    result = dehaze_physical.apply_physical_dehaze(
        source, DehazeParams(strength=.8, brightness_protection=.9),
        backend="native", diagnostics=diagnostics,
    )

    assert gpu_calls == [True]
    assert len(guard_calls) == 1
    assert guard_calls[0][2] > 1e-8
    assert diagnostics["dark_guard_backend"] == "cpp_cpu"
    np.testing.assert_array_equal(result, guarded_pixels)
