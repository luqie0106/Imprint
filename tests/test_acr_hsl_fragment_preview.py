"""Explicit HSL-only fragments: product renderer, request validation and caches."""
from pathlib import Path
import sys

import numpy as np
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import acr_hsl_fragments as fragments
import app_api as api
import camera_profile as cp
import native_renderer as native
from test_acr_color_preview_api import acr_scene, _color, _request
from test_acr_native_camera_profile import _prepared_camera, _profile


@pytest.mark.parametrize("name,lanes", [
    ("modern03_hsl", {9: -14, 17: -7}),
    ("modern04_hsl", {9: -10}),
    ("vintage07_hsl", {13: -29}),
])
def test_explicit_defaults_are_sparse_and_immutable(name, lanes):
    values = fragments.hsl_fragment_defaults(name)
    assert isinstance(values, tuple) and len(values) == 24
    assert {i: value for i, value in enumerate(values) if value != -999999} == lanes
    assert fragments.hsl_fragment_defaults("none") is None
    with pytest.raises(ValueError):
        fragments.hsl_fragment_defaults("Adobe Landscape")


def test_fragment_request_validation_and_legacy_fractional_controls():
    default = api.AcrColorPreviewRequest()
    assert default.hsl_fragment == "none" and default.hsl_fragment_amount == 1
    fractional = [0.25] + [0.] * 24
    assert api.AcrColorPreviewRequest(hsl=fractional).hsl == fractional
    for patch in ({"hsl_fragment": "guess_photo_look"},
                  {"hsl_fragment_amount": float("nan")},
                  {"hsl_fragment_amount": float("inf")},
                  {"hsl_fragment_amount": -0.01}, {"hsl_fragment_amount": 2.01},
                  {"hsl_fragment": "modern03_hsl", "hsl": fractional}):
        with pytest.raises(ValidationError):
            api.AcrColorPreviewRequest(**patch)


@pytest.mark.parametrize("full", [False, True])
def test_fragment_and_amount_are_forwarded_and_isolate_cache(acr_scene, full):
    client, _path, calls = acr_scene
    old_render = api.render_profile
    forwarded = []

    def render(*args, acr_color=None, **kwargs):
        forwarded.append((acr_color["hsl_fragment"], acr_color["hsl_fragment_amount"]))
        # The HTTP fixture supplies pixels; prove the routing/cache key separately
        # from native arithmetic, which is tested below with the real library.
        return old_render(*args, acr_color=acr_color, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(api, "render_profile", render)
        for name, amount in (("none", 1.), ("modern03_hsl", 1.),
                             ("modern03_hsl", .5), ("vintage07_hsl", .5)):
            config = dict(_color(0), hsl_fragment=name, hsl_fragment_amount=amount)
            first = _request(client, color=config, full_resolution=full)
            repeat = _request(client, color=config, full_resolution=full)
            assert first.status_code == repeat.status_code == 200
            assert first.headers["x-acr-color-status"] == "applied"
            assert first.content == repeat.content
        assert forwarded == [("none", 1.), ("modern03_hsl", 1.),
                             ("modern03_hsl", .5), ("vintage07_hsl", .5)]
        assert len(calls["render"]) == 4


@pytest.mark.parametrize("full", [False, True])
def test_original_mode_ignores_fragment_selection(acr_scene, full):
    client, _path, calls = acr_scene
    config = dict(_color(0), hsl_fragment="modern03_hsl", hsl_fragment_amount=2.)
    response = _request(client, color=config, mode="original", full_resolution=full)
    assert response.status_code == 200 and response.headers["x-acr-color-status"] == "off"
    assert calls["render"] == [None]


def test_profile_renderer_uses_composed_cpu_before_tone_and_grading(monkeypatch):
    events = []
    profile = _profile()
    monkeypatch.setattr(cp, "native_acr_illuminant_temperatures", lambda _: np.array([2850., 6500.]))
    monkeypatch.setattr(cp, "native_acr_prepare_profile_camera", lambda _: _prepared_camera())
    monkeypatch.setattr(cp, "native_acr_buffer_to_float", lambda x: x.astype(np.float32) / 65535)
    monkeypatch.setattr(cp, "native_acr_camera_float", lambda rgb, _: rgb.copy())
    monkeypatch.setattr(cp, "native_acr_scale_sdr", lambda rgb, _: rgb.copy())
    monkeypatch.setattr(cp, "native_acr_profile_stage_sdr", lambda rgb, *_: events.append("look") or rgb.copy())
    monkeypatch.setattr(cp, "native_acr_hsl_sdr", lambda *_: pytest.fail("manual HSL bypassed fragment"))

    def composed(rgb, current, defaults, scalar):
        assert tuple(defaults) == fragments.hsl_fragment_defaults("modern03_hsl")
        assert scalar == 1.75 and current[24] == 12
        events.append("composed-hsl")
        return rgb.copy()

    monkeypatch.setattr(cp, "native_acr_profile_hsl_sdr", composed)
    monkeypatch.setattr(cp, "native_acr_profile_tone", lambda rgb, _: events.append("tone") or rgb.copy())
    monkeypatch.setattr(cp, "native_acr_grading_sdr", lambda rgb, _: events.append("grading") or rgb.copy())
    monkeypatch.setattr(cp, "native_acr_project_srgb", lambda rgb, _: np.zeros(rgb.shape, np.uint8))
    camera = np.full((130, 3, 3), 12000, np.uint16)
    before = camera.copy()
    config = dict(_color(0), hsl=[0.] * 24 + [12.],
                  hsl_fragment="modern03_hsl", hsl_fragment_amount=1.75)
    output = cp.render_profile(camera, (1., 1., 1.), profile, 0., acr_color=config)
    assert output.shape == camera.shape and output.dtype == np.uint8
    np.testing.assert_array_equal(camera, before)
    assert events == ["look", "composed-hsl", "tone", "grading"] * 2
    assert cp.get_last_profile_backend() == "cpp_cpu_acr"


@pytest.mark.parametrize("name", ["modern03_hsl", "modern04_hsl", "vintage07_hsl"])
@pytest.mark.parametrize("scalar", [0., .25, 1., 1.75, 2.])
def test_composed_native_pixels_equal_separate_recovered_operators(name, scalar):
    rng = np.random.default_rng(422)
    rgb = rng.random((5, 17, 3), dtype=np.float32)
    current = np.r_[rng.integers(-100, 101, 24), 17.].astype(np.float64)
    defaults = fragments.hsl_fragment_defaults(name)
    before = rgb.copy(), current.copy()
    merged, mask, _ = native.native_acr_profile_hsl_controls(current[:24].astype(np.int32), defaults, scalar)
    controls = np.r_[merged, current[24]].astype(np.float64)
    expected = native.native_acr_hsl_sdr(rgb, controls)
    actual = native.native_acr_profile_hsl_sdr(rgb, current, defaults, scalar)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(rgb, before[0])
    np.testing.assert_array_equal(current, before[1])
    assert mask == sum(1 << i for i, value in enumerate(defaults) if value != -999999)


@pytest.mark.parametrize("bad", [
    {"rgb": np.zeros((1, 2, 3), np.uint16)},
    {"rgb": np.full((1, 2, 3), np.nan, np.float32)},
    {"rgb": np.full((1, 2, 3), 1.1, np.float32)},
    {"rgb": np.zeros((0, 2, 3), np.float32)},
    {"current": [0.] * 24}, {"current": [.5] + [0.] * 24},
    {"current": [float("inf")] + [0.] * 24},
    {"profile_defaults": [0] * 23}, {"profile_defaults": [False] * 24},
    {"profile_scalar": True}, {"profile_scalar": float("nan")},
    {"profile_scalar": 10**1000}, {"profile_scalar": -1}, {"profile_scalar": 2.1},
])
def test_composed_wrapper_rejects_bad_input_before_native_load(monkeypatch, bad):
    monkeypatch.setattr(native, "_get_library", lambda: pytest.fail("invalid input loaded native library"))
    values = dict(rgb=np.zeros((1, 2, 3), np.float32), current=[0.] * 25,
                  profile_defaults=[-999999] * 24, profile_scalar=1.)
    values.update(bad)
    with pytest.raises(ValueError):
        native.native_acr_profile_hsl_sdr(**values)
