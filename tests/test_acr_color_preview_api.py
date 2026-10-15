"""Manual ACR CPU color experiment routing and preview-cache behavior."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_api as api


@pytest.fixture
def acr_scene(tmp_path, monkeypatch):
    path = tmp_path / "acr.nef"
    path.write_bytes(b"raw fixture")
    source = np.full((8, 12, 3), 12000, dtype=np.uint16)
    metadata = SimpleNamespace(
        source_kind="raw", color_space="Linear sRGB", width=12, height=8, exif={},
    )
    profile = SimpleNamespace(fingerprint=("acr-profile", 1), name="ACR Test Profile")
    calls = {"camera": 0, "transfer": 0, "render": []}
    backend_result = ["cpp_cpu_acr"]

    def read_image(_path, *, preview, **_kwargs):
        return source.copy(), metadata

    def render_profile(camera, _neutral, _profile, _exposure, *, backend="cpu", acr_color=None):
        calls["render"].append(None if acr_color is None else {
            "profile_tone": acr_color["profile_tone"],
            "hsl": tuple(acr_color["hsl"]),
            "grading": tuple(acr_color["grading"]),
        })
        backend_result[0] = "cpp_cpu_acr" if acr_color is not None else "cpp_cpu"
        level = 30 if acr_color is None else 30 + int(acr_color["hsl"][0])
        return np.full(camera.shape, level, dtype=np.uint8)

    def load_camera(_path, shape, _preview):
        calls["camera"] += 1
        return np.full(shape, 24000, dtype=np.uint16), (1.0, 1.0, 1.0)

    def transfer(_camera, _reference, processed, *, backend="cpu"):
        calls["transfer"] += 1
        return processed.copy()

    monkeypatch.setattr(api, "read_image", read_image)
    monkeypatch.setattr(api, "resolve_profile", lambda *_args: profile)
    monkeypatch.setattr(api, "_prepare_dehaze_input", lambda image, *_args, **_kwargs: np.full(
        image.shape, 0.2, dtype=np.float32,
    ))
    monkeypatch.setattr(api, "_render_dehaze", lambda image, _params, _backend, *, diagnostics, **_kwargs: (
        diagnostics.update(auto_exposure_ev=0.0, auto_exposure_reason="off") or image.copy()
    ))
    monkeypatch.setattr(api, "_linear_float_to_uint16", lambda image, **_kwargs: np.full(
        image.shape, 20000, dtype=np.uint16,
    ))
    monkeypatch.setattr(api, "_correct_enhanced_raw", lambda image, *_args, **_kwargs: (image.copy(), None, False))
    monkeypatch.setattr(api, "load_camera_rgb", load_camera)
    monkeypatch.setattr(api, "transfer_enhancement", transfer)
    monkeypatch.setattr(api, "render_profile", render_profile)
    monkeypatch.setattr(api, "get_last_profile_backend", lambda: backend_result[0])
    monkeypatch.setattr(api, "_render_basic", lambda image, *_args: image)

    api._shutdown_full_resolution_preview_cache()
    api._clear_full_resolution_preview_caches()
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS["acr-preview-test"] = {"files": {"photo": path}}
        api._ENHANCE_PREVIEW_CACHE.clear()
    with api._DISPLAY_PREVIEW_LOCK:
        api._PROCESSING_PREVIEW_CACHE.clear()
        api._PREPARED_DEHAZE_PREVIEW_CACHE.clear()
        api._DEHAZED_PREVIEW_CACHE.clear()
        api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
        api._CAMERA_PROFILE_SOURCE_CACHE.clear()
    yield TestClient(api.app), path, calls
    api._shutdown_full_resolution_preview_cache()
    api._clear_full_resolution_preview_caches()
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS.pop("acr-preview-test", None)
        api._ENHANCE_PREVIEW_CACHE.clear()
    with api._DISPLAY_PREVIEW_LOCK:
        api._PROCESSING_PREVIEW_CACHE.clear()
        api._PREPARED_DEHAZE_PREVIEW_CACHE.clear()
        api._DEHAZED_PREVIEW_CACHE.clear()
        api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
        api._CAMERA_PROFILE_SOURCE_CACHE.clear()


def _color(hue: float) -> dict:
    hsl = [0.0] * 25
    hsl[0] = hue
    return {"enabled": True, "profile_tone": True, "hsl": hsl,
            "grading": [0.0] * 12 + [50.0, 0.0]}


def _request(client, *, color=None, headers=None, **extra):
    payload = {
        "session_id": "acr-preview-test", "photo_id": "photo",
        "params": {"strength": 0}, "basic_params": {"exposure": 0},
        "max_edge": 320, "render_backend": "cpu", "basic_backend": "python",
    }
    if color is not None:
        payload["acr_color"] = color
    payload.update(extra)
    return client.post("/api/enhance/preview", json=payload, headers=headers)


def test_acr_color_defaults_and_parameter_validation():
    config = api.AcrColorPreviewRequest()
    assert config.enabled is False
    assert config.profile_tone is True
    assert config.hsl == [0.0] * 25
    assert config.grading == [0.0] * 12 + [50.0, 0.0]

    invalid = []
    invalid.append({"hsl": [0.0] * 24})
    invalid.append({"hsl": [float("nan")] + [0.0] * 24})
    invalid.append({"hsl": [0.0] * 24 + [100.1]})
    invalid.append({"grading": [0.0] * 13})
    invalid.append({"grading": [float("inf")] + [0.0] * 11 + [50.0, 0.0]})
    for index, value in ((0, -0.1), (3, 360.1), (6, 361), (9, -1)):
        grading = [0.0] * 12 + [50.0, 0.0]
        grading[index] = value
        invalid.append({"grading": grading})
    for index, value in ((1, -1), (4, 100.1), (7, 101), (10, -0.1)):
        grading = [0.0] * 12 + [50.0, 0.0]
        grading[index] = value
        invalid.append({"grading": grading})
    for index, value in ((2, -100.1), (5, 100.1), (8, -101), (11, 101)):
        grading = [0.0] * 12 + [50.0, 0.0]
        grading[index] = value
        invalid.append({"grading": grading})
    for index, value in ((12, -0.1), (12, 100.1), (13, -100.1), (13, 100.1)):
        grading = [0.0] * 12 + [50.0, 0.0]
        grading[index] = value
        invalid.append({"grading": grading})

    for fields in invalid:
        with pytest.raises(ValidationError):
            api.AcrColorPreviewRequest(**fields)


def test_low_resolution_acr_color_is_forwarded_and_cache_isolated(acr_scene):
    client, _path, calls = acr_scene
    first = _request(client, color=_color(10))
    second = _request(client, color=_color(20))
    repeated = _request(client, color=_color(20))

    assert first.status_code == second.status_code == repeated.status_code == 200
    assert first.headers["x-acr-color-status"] == "applied"
    assert second.headers["x-acr-color-status"] == "applied"
    assert repeated.headers["x-acr-color-status"] == "applied"
    assert first.content != second.content
    assert repeated.content == second.content
    assert len(calls["render"]) == 2
    assert [call["hsl"][0] for call in calls["render"]] == [10.0, 20.0]


def test_full_resolution_acr_color_reuses_only_pre_exposure_stage(acr_scene):
    client, _path, calls = acr_scene
    first = _request(client, color=_color(10), full_resolution=True)
    second = _request(client, color=_color(20), full_resolution=True)
    repeated = _request(client, color=_color(20), full_resolution=True)

    assert first.status_code == second.status_code == repeated.status_code == 200
    assert first.headers["x-preview-base-cache"] == "miss"
    assert first.headers["x-preview-camera-cache"] == "miss"
    assert second.headers["x-preview-base-cache"] == "miss"
    assert second.headers["x-preview-camera-cache"] == "stage-hit"
    assert repeated.headers["x-preview-base-cache"] == "hit"
    assert first.headers["x-acr-color-status"] == "applied"
    assert second.headers["x-acr-color-status"] == "applied"
    assert repeated.headers["x-acr-color-status"] == "applied"
    assert first.content != second.content
    assert repeated.content == second.content
    assert calls["transfer"] == 1
    assert len(calls["render"]) == 2
    assert [call["hsl"][0] for call in calls["render"]] == [10.0, 20.0]


@pytest.mark.parametrize("full", [False, True])
def test_original_mode_ignores_acr_color(acr_scene, full):
    client, _path, calls = acr_scene
    response = _request(client, color=_color(35), mode="original", full_resolution=full)

    assert response.status_code == 200
    assert response.headers["x-acr-color-status"] == "off"
    assert calls["render"] == [None]


def test_acr_color_rejects_ricoh_and_missing_profile(acr_scene, monkeypatch):
    client, _path, _calls = acr_scene
    conflict = _request(client, color=_color(10), ricoh_preset_id="gr3-vivid")
    assert conflict.status_code == 422
    assert "不支持同时使用理光预设" in conflict.json()["error"]

    monkeypatch.setattr(api, "resolve_profile", lambda *_args: None)
    no_profile = _request(client, color=_color(10))
    assert no_profile.status_code == 422
    assert "需要可用的相机配置文件" in no_profile.json()["error"]


def test_acr_color_fallback_status_and_cors_exposure(acr_scene, monkeypatch):
    client, _path, _calls = acr_scene
    monkeypatch.setattr(api, "get_last_profile_backend", lambda: "cpp_cpu")
    fallback_backend = _request(
        client, color=_color(10), headers={"Origin": "http://localhost"},
    )
    assert fallback_backend.status_code == 200
    assert fallback_backend.headers["x-camera-profile-status"] == "applied"
    assert fallback_backend.headers["x-acr-color-status"] == "fallback"

    def fail_profile(*_args, **_kwargs):
        raise ValueError("native ACR renderer unavailable")

    monkeypatch.setattr(api, "render_profile", fail_profile)
    failed = _request(client, color=_color(20))
    assert failed.status_code == 200
    assert failed.headers["x-camera-profile-status"] == "fallback"
    assert failed.headers["x-acr-color-status"] == "fallback"

    exposed = fallback_backend.headers.get("access-control-expose-headers", "").lower()
    assert "x-acr-color-status" in exposed
