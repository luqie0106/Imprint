"""Strong experimental integration: active recovery and exact safe fallback."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dehaze import DehazeParams
import dehaze_physical as physical


@pytest.fixture(autouse=True)
def enable_strong(monkeypatch):
    monkeypatch.setenv("IMPRINT_NONLOCAL_RELIEF", "strong")


def ray_scene(monkeypatch):
    air = np.array([.55, .60, .65], np.float32)
    source = np.broadcast_to(air, (192, 288, 3)).copy()
    depth = np.linspace(.12, .95, 144, dtype=np.float32)
    material = np.array([.06, .08, .10], np.float32)
    source[48:] = (air + depth[:, None] * (material - air))[:, None, :]
    source[20:48, 60:68] = source[-1, 0]
    monkeypatch.setattr(physical, "_estimate_airlight", lambda rgb, p: (air.copy(), 1.0))
    return source, air


def test_active_strong_preserves_hue_equal_rgb_and_monotone_depth(monkeypatch):
    source, air = ray_scene(monkeypatch)
    original = source.copy()
    p = DehazeParams(strength=1)
    t, _, stats = physical._estimate_scene(source, p, True)
    assert stats["nonlocal_field_active"] and np.ptp(t) > .05
    result = physical.apply_physical_dehaze(source, p, spatial=True)
    np.testing.assert_array_equal(result[30, 64], result[-1, 250])
    y = physical._luminance(result)
    source_y = physical._luminance(source)
    chroma = result - y[..., None]
    source_chroma = source - source_y[..., None]
    # Chroma direction is retained even when tone/luminance changes.
    np.testing.assert_allclose(np.cross(chroma, source_chroma), 0, atol=2e-7)
    assert np.max(np.diff(y[48:, 250])) <= 2e-6
    assert np.max(result - source) <= 2e-7
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("mode", ["sun", "night", "dim_daylight", "uncertain", "manual"])
def test_strong_ineligible_is_identical_to_default(monkeypatch, mode):
    source, air = ray_scene(monkeypatch)
    confidence = 1.0
    if mode == "sun":
        source[:16, :32] = 1.0
    elif mode == "night":
        source *= .03
        air *= .03
    elif mode == "dim_daylight":
        source *= .27
        air *= .27
    elif mode == "uncertain":
        confidence = .4
    monkeypatch.setattr(physical, "_estimate_airlight", lambda rgb, p: (air, confidence))
    p = DehazeParams(strength=1)
    result = physical.apply_physical_dehaze(source, p, spatial=mode != "manual")
    stats = physical.physical_diagnostics(source, p, spatial=mode != "manual")
    assert not stats["nonlocal_active"]
    monkeypatch.delenv("IMPRINT_NONLOCAL_RELIEF")
    baseline = physical.apply_physical_dehaze(source, p, spatial=mode != "manual")
    np.testing.assert_array_equal(result, baseline)


def test_mode_cache_isolation_and_zero(monkeypatch):
    source, _ = ray_scene(monkeypatch)
    p = DehazeParams(strength=0)
    np.testing.assert_array_equal(physical.apply_physical_dehaze(source, p, spatial=True), source)
    tokens = []
    for mode in ["strong", "1", "off", "unknown"]:
        monkeypatch.setenv("IMPRINT_NONLOCAL_RELIEF", mode)
        tokens.append(p.cache_token())
    assert len(set(tokens[:3])) == 3
    assert tokens[2] == tokens[3]


def test_active_strong_native_matches_effective_operator(monkeypatch):
    from native_renderer import native_physical_dehaze, NativeRendererError
    source, air = ray_scene(monkeypatch)
    p = DehazeParams(strength=1)
    t, _, stats = physical._estimate_scene(source, p, True)
    assert stats["nonlocal_field_active"]
    p = physical._operator_params(p, stats)
    cpu = physical._physical_pixels(source, t, air, p)
    try:
        native = native_physical_dehaze(source, p, t, air)
    except NativeRendererError as exc:
        pytest.skip(str(exc))
    assert np.max(np.abs(cpu - native)) <= 3e-6
