"""Conservative non-local integration: colour, skyline and fallback contracts."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dehaze import DehazeParams
import dehaze_physical as physical


@pytest.fixture(autouse=True)
def enable_experimental_relief(monkeypatch):
    monkeypatch.setenv("IMPRINT_NONLOCAL_RELIEF", "1")


def _ray_scene():
    air = np.array([.55, .60, .65], np.float32)
    # A continuous depth range for one material, with repeated exact RGB at
    # asymmetric roof-like edges. The upper frame supplies unchanged airlight.
    source = np.broadcast_to(air, (192, 288, 3)).copy()
    depth = np.linspace(.1, .95, 144, dtype=np.float32)
    material = np.array([.06, .08, .10], np.float32)
    strip = air + depth[:, None] * (material - air)
    source[48:] = strip[:, None, :]
    source[20:48, 60:68] = strip[-1]
    return source, air


def _fixed_air(monkeypatch, air):
    monkeypatch.setattr(physical, "_estimate_airlight", lambda rgb, p: (air.copy(), 1.0))


def test_reliable_relief_only_weakens_inverse_and_keeps_equal_sky(monkeypatch):
    source, air = _ray_scene()
    _fixed_air(monkeypatch, air)
    params = DehazeParams(strength=1, color_protection=1, color_recovery=0,
                          local_contrast=0)
    t, atmosphere, stats = physical._estimate_scene(source, params, True)
    assert stats["nonlocal_active"]
    assert float(np.ptp(t)) > .005
    original = source.copy()
    result = physical.apply_physical_dehaze(source, params, spatial=True)
    baseline = physical._physical_pixels(source, np.full(source.shape[:2],
                                             float(np.min(t)), np.float32), air, params)
    assert np.max(baseline - result) < 2e-7
    assert np.max(result - source) < 2e-7
    assert np.mean(result[-24:] - baseline[-24:]) > 1e-4
    np.testing.assert_array_equal(result[30, 59], result[30, 250])
    np.testing.assert_array_equal(result[30, 64], result[-1, 250])
    # A scalar transmission must not turn neutral/coloured material purple.
    y = physical._luminance(result)
    src_y = physical._luminance(source)
    np.testing.assert_allclose(result / y[..., None], source / src_y[..., None], atol=2e-6)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(atmosphere, air)


@pytest.mark.parametrize("mode", ["manual", "night", "uncertain", "sun"])
def test_estimator_keeps_established_path_when_evidence_is_ineligible(monkeypatch, mode):
    source, air = _ray_scene()
    confidence = 1.0
    if mode == "night":
        source *= .03
        air *= .03
    elif mode == "uncertain":
        confidence = .3
    elif mode == "sun":
        source[:12, :24] = 1
    monkeypatch.setattr(physical, "_estimate_airlight", lambda rgb, p: (air, confidence))
    def forbidden(*args):
        raise AssertionError("Ineligible scenes must not run the non-local fit")
    monkeypatch.setattr(physical, "build_reliability_lut", forbidden)
    t, _, stats = physical._estimate_scene(source, DehazeParams(strength=1), mode != "manual")
    assert not stats["nonlocal_active"]
    assert np.isfinite(t).all()
    np.testing.assert_array_equal(t, np.full_like(t, t.flat[0]))


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_active_public_pipeline_keeps_dtype_input_and_zero_strength(monkeypatch, dtype):
    source, air = _ray_scene()
    _fixed_air(monkeypatch, air)
    if dtype == np.uint16:
        source = np.rint(source * 65535).astype(dtype)
    original = source.copy()
    result = physical.apply_physical_dehaze(source, DehazeParams(strength=1), spatial=True)
    assert result.dtype == dtype and result.shape == source.shape
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= (65535 if dtype == np.uint16 else 1)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(physical.apply_physical_dehaze(
        source, DehazeParams(strength=0), spatial=True), original)


def test_relief_keeps_smooth_depth_ramp_monotone(monkeypatch):
    source, air = _ray_scene()
    _fixed_air(monkeypatch, air)
    result = physical.apply_physical_dehaze(source, DehazeParams(
        strength=1, color_protection=1, color_recovery=0, local_contrast=0), spatial=True)
    profile = physical._luminance(result[48:, 250])
    assert np.max(np.diff(profile)) <= 2e-7


def test_active_nonlocal_pipeline_matches_actual_native_renderer(monkeypatch):
    from native_renderer import NativeRendererError, native_physical_dehaze
    source, air = _ray_scene()
    _fixed_air(monkeypatch, air)
    params = DehazeParams(strength=1)
    t, _, stats = physical._estimate_scene(source, params, True)
    assert stats["nonlocal_active"] and float(np.ptp(t)) > .005
    cpu = physical._physical_pixels(source, t, air, params)
    try:
        native = native_physical_dehaze(source, params, t, air)
    except NativeRendererError as exc:
        pytest.skip(str(exc))
    assert np.max(np.abs(cpu - native)) <= 3e-6


def test_trial_is_disabled_by_default_and_has_distinct_cache_token(monkeypatch):
    source, air = _ray_scene()
    _fixed_air(monkeypatch, air)
    params = DehazeParams(strength=1)
    enabled_token = params.cache_token()
    monkeypatch.delenv("IMPRINT_NONLOCAL_RELIEF")
    assert params.cache_token() != enabled_token
    t, _, stats = physical._estimate_scene(source, params, True)
    assert not stats["nonlocal_active"]
    np.testing.assert_array_equal(t, np.full_like(t, t.flat[0]))
