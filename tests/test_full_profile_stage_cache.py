"""Full-resolution camera profile stages reuse exposure-independent RGB16 data."""
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_api as api


@pytest.fixture
def profile_scene(tmp_path, monkeypatch):
    path = tmp_path / "profile.nef"
    path.write_bytes(b"raw fixture")
    image = np.full((12, 17, 3), 24000, dtype=np.uint16)
    metadata = SimpleNamespace(source_kind="raw", color_space="Linear sRGB", exif={})
    profile = SimpleNamespace(fingerprint=("profile", 1), name="Test Profile")
    calls = {name: 0 for name in ("decode", "dehaze", "correct", "camera", "transfer", "render")}

    def read_image(_path, *, preview):
        assert preview is False
        calls["decode"] += 1
        return image.copy(), metadata

    def correct(value, *_args, **_kwargs):
        calls["correct"] += 1
        return value.copy(), None, False

    def load_camera(_path, shape, preview):
        assert preview is False
        calls["camera"] += 1
        return np.full(shape, 22000, dtype=np.uint16), (1.0, 1.0, 1.0)

    def render_dehaze(value, params, _backend, *, diagnostics, **kwargs):
        calls["dehaze"] += 1
        diagnostics.update(
            auto_exposure_ev=0.5 if kwargs.get("auto_exposure") else 0.0,
            auto_exposure_reason="applied" if kwargs.get("auto_exposure") else "off",
        )
        return value * np.float32(1.0 - params.strength * 0.1)

    def transfer(_camera, _reference, processed, *, backend="cpu"):
        calls["transfer"] += 1
        return processed.copy()

    def render_profile(camera, _neutral, _profile, ev, *, backend="cpu"):
        calls["render"] += 1
        # Keep a visible and deterministic EV response for JPEG equality checks.
        level = int(np.clip(40 + round(ev * 20) + camera[0, 0, 0] // 4000, 0, 255))
        return np.full(camera.shape, level, dtype=np.uint8)

    def build_source_cache(source_path, _raw_camera, reference, camera, corrected_reference,
                           camera_lens, reference_lens, *, correction_version, **_kwargs):
        matrix = np.eye(3, dtype=np.float32)
        matrix.setflags(write=False)
        return SimpleNamespace(
            source_identity=api.enhanced_dng_source_identity(source_path),
            source_shape=tuple(reference.shape), camera_rgb16=camera,
            reference_rgb16=corrected_reference, srgb_to_camera=matrix,
            camera_lens_result=camera_lens, reference_lens_result=reference_lens,
            correction_version=correction_version,
        )

    monkeypatch.setattr(api, "read_image", read_image)
    monkeypatch.setattr(api, "_correct_enhanced_raw", correct)
    monkeypatch.setattr(api, "load_camera_rgb", load_camera)
    monkeypatch.setattr(api, "_render_dehaze", render_dehaze)
    monkeypatch.setattr(api, "transfer_enhancement", transfer)
    monkeypatch.setattr(api, "render_profile", render_profile)
    monkeypatch.setattr(api, "build_enhanced_dng_source_cache", build_source_cache)
    monkeypatch.setattr(api, "resolve_profile", lambda *_args: profile)
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS["profile-stage-test"] = {"files": {"photo": path}}
        api._ENHANCE_PREVIEW_CACHE.clear()
    api._shutdown_full_resolution_preview_cache()
    yield TestClient(api.app), path, profile, calls
    api._shutdown_full_resolution_preview_cache()
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS.pop("profile-stage-test", None)
        api._ENHANCE_PREVIEW_CACHE.clear()


def request(client, **extra):
    data = {
        "session_id": "profile-stage-test", "photo_id": "photo",
        "params": {"strength": 0.25}, "basic_params": {"exposure": 0},
        "max_edge": 320, "render_backend": "cpu", "basic_backend": "python",
        "full_resolution": True, "preview_level": 0,
    }
    data.update(extra)
    response = client.post("/api/enhance/preview", json=data)
    assert response.status_code == 200, response.text
    return response


def test_ev_reuses_processed_stage_and_matches_forced_rebuild(profile_scene):
    client, _path, _profile, calls = profile_scene
    first = request(client, basic_params={"exposure": 0})
    assert first.headers["x-preview-camera-cache"] == "miss"
    camera_source_key = next(iter(api._FULL_RESOLUTION_CAMERA_SOURCE_CACHE))
    assert camera_source_key[-1] == api.enhanced_dng_source_identity(_path)
    assert calls == {"decode": 1, "dehaze": 1, "correct": 3,
                     "camera": 1, "transfer": 1, "render": 1}

    exposed = request(client, basic_params={"exposure": 1})
    assert exposed.headers["x-preview-camera-cache"] == "stage-hit"
    assert calls == {"decode": 1, "dehaze": 1, "correct": 3,
                     "camera": 1, "transfer": 1, "render": 2}

    contrasted = request(client, basic_params={"exposure": 1, "contrast": 20})
    assert contrasted.headers["x-preview-base-cache"] == "hit"
    assert calls["dehaze"] == 1
    assert calls["correct"] == 3
    assert calls["camera"] == calls["transfer"] == 1

    # Force the same EV through a fresh dehaze and transfer stage; output must
    # remain byte-identical to the result made from the cached RGB16 stage.
    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        api._FULL_RESOLUTION_BASE_CACHE.clear()
        api._FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.clear()
    rebuilt = request(client, basic_params={"exposure": 1})
    assert rebuilt.content == exposed.content
    assert calls["dehaze"] == 2
    assert calls["camera"] == 1  # Corrected camera source remains reusable.
    assert calls["transfer"] == 2


def test_camera_raw_decode_overlaps_dehaze_and_hot_source_is_reused(
    profile_scene, monkeypatch,
):
    client, _path, _profile, calls = profile_scene
    camera_started = threading.Event()
    dehaze_started = threading.Event()
    release_camera = threading.Event()
    camera_finished = threading.Event()
    render_count = 0

    def load_camera(_path, shape, preview):
        assert preview is False
        calls["camera"] += 1
        camera_started.set()
        assert dehaze_started.wait(2)
        assert release_camera.wait(2)
        camera_finished.set()
        return np.full(shape, 22000, dtype=np.uint16), (1.0, 1.0, 1.0)

    def render_dehaze(value, params, _backend, *, diagnostics, **kwargs):
        nonlocal render_count
        render_count += 1
        if render_count == 1:
            assert camera_started.wait(2)
            dehaze_started.set()
            assert not camera_finished.is_set()
            release_camera.set()
        diagnostics.update(auto_exposure_ev=0.0, auto_exposure_reason="off")
        return value * np.float32(1.0 - params.strength * 0.1)

    monkeypatch.setattr(api, "load_camera_rgb", load_camera)
    monkeypatch.setattr(api, "_render_dehaze", render_dehaze)
    first = request(client, params={"strength": 0.25})
    assert first.status_code == 200
    assert camera_started.is_set() and dehaze_started.is_set()
    assert camera_finished.is_set()
    assert calls["camera"] == 1

    # Force a fresh dehaze and transfer stage while retaining the camera source.
    request(client, params={"strength": 0.5})
    assert calls["camera"] == 1


def test_camera_source_decode_failure_joins_and_uses_profile_fallback(
    profile_scene, monkeypatch,
):
    client, _path, _profile, _calls = profile_scene
    started = threading.Event()
    finished = threading.Event()

    def fail_camera(*_args):
        started.set()
        try:
            raise ValueError("camera decode failed")
        finally:
            finished.set()

    monkeypatch.setattr(api, "load_camera_rgb", fail_camera)
    response = request(client, params={"strength": 0.4})
    assert response.headers["x-camera-profile-status"] == "fallback"
    assert started.is_set() and finished.is_set()


def test_dehaze_failure_still_joins_inflight_camera_decode(profile_scene, monkeypatch):
    client, _path, _profile, _calls = profile_scene
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def load_camera(_path, shape, preview):
        assert preview is False
        started.set()
        assert release.wait(2)
        finished.set()
        return np.full(shape, 22000, dtype=np.uint16), (1.0, 1.0, 1.0)

    def fail_dehaze(*_args, **_kwargs):
        assert started.wait(2)
        assert not finished.is_set()
        release.set()
        raise RuntimeError("dehaze failed")

    monkeypatch.setattr(api, "load_camera_rgb", load_camera)
    monkeypatch.setattr(api, "_render_dehaze", fail_dehaze)
    response = client.post("/api/enhance/preview", json={
        "session_id": "profile-stage-test", "photo_id": "photo",
        "params": {"strength": 0.4}, "basic_params": {"exposure": 0},
        "max_edge": 320, "render_backend": "cpu", "basic_backend": "python",
        "full_resolution": True, "preview_level": 0,
    })
    assert response.status_code == 500
    assert started.is_set() and finished.is_set()


def test_dehaze_change_invalidates_only_processed_stage(profile_scene):
    client, _path, _profile, calls = profile_scene
    request(client, basic_params={"exposure": 0})
    changed = request(client, params={"strength": 0.8}, basic_params={"exposure": 0})
    assert changed.headers["x-preview-camera-cache"] == "miss"
    assert calls["decode"] == 1
    assert calls["dehaze"] == 2
    assert calls["correct"] == 4
    assert calls["camera"] == 1
    assert calls["transfer"] == 2
    assert len(api._FULL_RESOLUTION_CAMERA_PROCESSED_CACHE) == 1


def test_auto_exposure_diagnostics_survive_manual_ev_stage_reuse(profile_scene):
    client, _path, _profile, calls = profile_scene
    first = request(client, auto_exposure=True, basic_params={"exposure": 0})
    second = request(client, auto_exposure=True, basic_params={"exposure": 1})
    assert second.headers["x-preview-camera-cache"] == "stage-hit"
    assert second.headers["x-auto-exposure-ev"] == first.headers["x-auto-exposure-ev"] == "0.5"
    assert second.headers["x-auto-exposure-reason"] == first.headers["x-auto-exposure-reason"] == "applied"
    assert calls["dehaze"] == 1


@pytest.mark.parametrize(("selector", "expected"), [
    ("cpu", "cpu"), ("auto", "auto"), ("native", "native"),
    ("legacy_gpu", "auto"), ("pytorch", "auto"),
])
def test_camera_profile_backend_maps_legacy_gpu_modes(selector, expected):
    assert api._camera_profile_render_backend(selector) == expected


def test_source_stat_change_and_profile_fallback_do_not_reuse_stale_stage(profile_scene, monkeypatch):
    client, path, _profile, calls = profile_scene
    request(client, basic_params={"exposure": 0})
    path.write_bytes(b"raw fixture changed")
    changed = request(client, basic_params={"exposure": 1})
    assert changed.headers["x-preview-camera-cache"] == "miss"
    assert calls["decode"] == 2
    assert calls["camera"] == 2
    assert calls["dehaze"] == 2

    def old_renderer_signature(_camera, _neutral, _profile, _ev):
        raise AssertionError("legacy renderer path should be handled as fallback")

    monkeypatch.setattr(api, "render_profile", old_renderer_signature)
    fallback = request(client, basic_params={"exposure": 1.5})
    assert fallback.headers["x-camera-profile-status"] == "fallback"
    # Transfer can still be complete; a later exposure request will retry the
    # profile renderer and preserve the one-time legacy exposure fallback.
    assert api._FULL_RESOLUTION_CAMERA_PROCESSED_CACHE


def test_photo_switch_and_profile_removal_clear_camera_layers(profile_scene, monkeypatch):
    client, first_path, _profile, calls = profile_scene
    request(client, basic_params={"exposure": 0})
    second_path = first_path.with_name("second.nef")
    second_path.write_bytes(b"second raw")
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS["profile-stage-test"]["files"]["second"] = second_path
    switched = request(client, photo_id="second", basic_params={"exposure": 0})
    assert switched.headers["x-preview-camera-cache"] == "miss"
    assert calls["decode"] == calls["camera"] == calls["dehaze"] == 2
    assert len(api._FULL_RESOLUTION_CAMERA_SOURCE_CACHE) == 1
    assert len(api._FULL_RESOLUTION_CAMERA_PROCESSED_CACHE) == 1
    assert len(api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE) == 1

    monkeypatch.setattr(api, "resolve_profile", lambda *_args: None)
    legacy = request(client, photo_id="second", basic_params={"exposure": 1.5})
    assert legacy.headers["x-camera-profile-status"] == "legacy"
    assert legacy.headers["x-preview-camera-cache"] == "not-needed"
    assert not api._FULL_RESOLUTION_CAMERA_SOURCE_CACHE
    assert not api._FULL_RESOLUTION_CAMERA_PROCESSED_CACHE
    assert not api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE


def test_transfer_failure_never_retains_incomplete_processed_layer(profile_scene, monkeypatch):
    client, _path, _profile, _calls = profile_scene
    monkeypatch.setattr(
        api, "transfer_enhancement",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad transfer")),
    )
    response = request(client, basic_params={"exposure": 1.5})
    assert response.headers["x-camera-profile-status"] == "fallback"
    assert not api._FULL_RESOLUTION_CAMERA_PROCESSED_CACHE


def test_full_profile_layers_obey_budget_and_idle_shutdown_cleanup(profile_scene, monkeypatch):
    client, _path, _profile, _calls = profile_scene
    request(client, basic_params={"exposure": 0})
    assert api._FULL_RESOLUTION_CAMERA_SOURCE_CACHE
    assert api._FULL_RESOLUTION_CAMERA_PROCESSED_CACHE
    assert api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE

    generation = api._FULL_RESOLUTION_CACHE_GENERATION
    timer = api._FULL_RESOLUTION_CACHE_TIMER
    assert timer is not None
    timer.cancel()
    api._expire_full_resolution_preview_caches(generation)
    assert not api._FULL_RESOLUTION_DECODED_CACHE
    assert not api._FULL_RESOLUTION_BASE_CACHE
    assert not api._FULL_RESOLUTION_CAMERA_SOURCE_CACHE
    assert not api._FULL_RESOLUTION_CAMERA_PROCESSED_CACHE
    assert not api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE

    # A tiny combined cap must reject all full-size retention, including the
    # decoded image and camera profile layers.
    monkeypatch.setattr(api, "_MAX_FULL_RESOLUTION_PREVIEW_CACHE_BYTES", 1)
    request(client, basic_params={"exposure": 0})
    assert api._full_resolution_preview_cache_bytes() <= 1
    api._shutdown_full_resolution_preview_cache()
    assert api._full_resolution_preview_cache_bytes() == 0
