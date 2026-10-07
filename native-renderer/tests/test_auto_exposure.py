from __future__ import annotations

from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from types import MappingProxyType

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import app_api
import ricoh_filter
from auto_exposure import apply_auto_exposure, estimate_auto_exposure
from dehaze import DehazeParams


def test_auto_exposure_applies_bounded_scalar_without_mutating_input():
    image = np.full((12, 20, 3), 0.09, dtype=np.float32)
    original = image.copy()
    diagnostics = {}

    adjusted = apply_auto_exposure(image, diagnostics=diagnostics)

    assert np.allclose(adjusted, 0.18, atol=1e-6)
    assert adjusted is not image
    assert np.array_equal(image, original)
    assert diagnostics["auto_exposure_ev"] == pytest.approx(1.0)
    assert diagnostics["auto_exposure_reason"] == "applied"


def test_auto_exposure_uses_one_scalar_for_color_and_offsets_minus_one_ev():
    luminance = np.linspace(0.10, 0.16, 30, dtype=np.float32).reshape(5, 6, 1)
    image = luminance * np.array([1.0, 0.75, 0.45], dtype=np.float32)
    original_result = apply_auto_exposure(image)
    darkened = image / np.float32(2.0)
    darkened_result = apply_auto_exposure(darkened)

    assert np.allclose(darkened_result, original_result, atol=1e-6)
    channel_ratios = original_result / image
    assert np.allclose(channel_ratios, channel_ratios[..., :1], atol=1e-6)
    assert estimate_auto_exposure(darkened)[0] == pytest.approx(
        estimate_auto_exposure(image)[0] + 1.0,
    )


def test_auto_exposure_disabled_is_identity_and_positive_adjustment_caps_at_two_ev():
    image = np.full((3, 4, 3), 0.01, dtype=np.float32)
    original = image.copy()
    diagnostics = {}

    disabled = apply_auto_exposure(image, enabled=False, diagnostics=diagnostics)
    capped = apply_auto_exposure(image)

    assert np.array_equal(disabled, original)
    assert np.array_equal(image, original)
    assert diagnostics == {"auto_exposure_ev": 0.0, "auto_exposure_reason": "off"}
    assert estimate_auto_exposure(image) == (2.0, "applied")
    assert np.allclose(capped, 0.04, atol=1e-7)


def test_auto_exposure_reports_target_black_and_highlight_limits():
    target = np.full((4, 4, 3), 0.18, dtype=np.float32)
    black = np.zeros((4, 4, 3), dtype=np.float32)
    highlight = np.full((4, 4, 3), 0.09, dtype=np.float32)
    highlight[0, 0] = 0.8

    assert estimate_auto_exposure(target) == (0.0, "target_reached")
    assert estimate_auto_exposure(black) == (0.0, "black")
    ev, reason = estimate_auto_exposure(highlight)
    assert reason == "highlight_limited"
    assert ev == pytest.approx(np.log2(0.95 / 0.8))
    adjusted = apply_auto_exposure(highlight)
    assert np.max(adjusted) == pytest.approx(0.95, abs=1e-6)


def test_auto_exposure_uses_full_input_peak_when_luma_analysis_is_reduced():
    image = np.full((1024, 1024, 3), 0.09, dtype=np.float32)
    image[3, 3] = 0.99

    ev, reason = estimate_auto_exposure(image)

    assert ev == 0.0
    assert reason == "highlight_limited"


@pytest.mark.parametrize(
    "image, error",
    [
        (np.zeros((2, 2), dtype=np.float32), ValueError),
        (np.zeros((2, 2, 4), dtype=np.float32), ValueError),
        (np.zeros((2, 2, 3), dtype=np.uint8), TypeError),
        (np.full((2, 2, 3), np.nan, dtype=np.float32), ValueError),
        (np.full((2, 2, 3), np.inf, dtype=np.float32), ValueError),
        (np.full((2, 2, 3), -0.01, dtype=np.float32), ValueError),
        (np.full((2, 2, 3), 1.01, dtype=np.float32), ValueError),
    ],
)
def test_auto_exposure_rejects_invalid_rgb(image, error):
    with pytest.raises(error):
        apply_auto_exposure(image)


def test_render_dehaze_applies_auto_exposure_at_zero_dehaze_strength():
    image = np.full((5, 7, 3), 0.09, dtype=np.float32)
    diagnostics = {}

    rendered = app_api._render_dehaze(
        image, DehazeParams(strength=0.0), "cpu", auto_exposure=True,
        diagnostics=diagnostics,
    )

    assert np.allclose(rendered, 0.18, atol=1e-6)
    assert diagnostics["auto_exposure_ev"] == pytest.approx(1.0)
    assert diagnostics["auto_exposure_reason"] == "applied"
    assert diagnostics["fallback_reason"] == "zero_strength"
    assert np.all(image == 0.09)


def test_cached_preview_level_keeps_the_pre_resize_exposure_estimate(tmp_path, monkeypatch):
    path = tmp_path / "level.raw"
    path.write_bytes(b"fixture")
    y = np.linspace(0.04, 0.13, 900, dtype=np.float32)[:, None]
    image = np.broadcast_to(y, (900, 900))
    linear = np.repeat(image[..., None], 3, axis=2).copy()
    metadata = SimpleNamespace(color_space="Linear sRGB", source_kind="raw")
    monkeypatch.setattr(app_api, "read_image", lambda *_a, **_k: (linear.copy(), metadata))
    with app_api._DISPLAY_PREVIEW_LOCK:
        app_api._DISPLAY_PREVIEW_CACHE.clear()
        app_api._PROCESSING_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()

    estimates = []
    try:
        for level in (0, 1, 2):
            status = {}
            app_api._cached_dehazed_display_preview(
                f"level-session-{level}", f"photo-{level}", path, 900,
                DehazeParams(strength=0.0), "cpu", False,
                preview_level=level, auto_exposure=True, status_out=status,
            )
            estimates.append((status["auto_exposure_ev"], status["auto_exposure_reason"]))
    finally:
        with app_api._DISPLAY_PREVIEW_LOCK:
            app_api._DISPLAY_PREVIEW_CACHE.clear()
            app_api._PROCESSING_PREVIEW_CACHE.clear()
            app_api._DEHAZED_PREVIEW_CACHE.clear()
            app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()

    assert estimates[0] == estimates[1] == estimates[2]
    assert estimates[0][0] > 0


def test_preview_headers_keep_auto_exposure_diagnostics_on_cache_hit(tmp_path, monkeypatch):
    path = tmp_path / "preview.jpg"
    path.write_bytes(b"fixture")
    session_id, photo_id = "auto-preview-session", "photo-1"
    metadata = SimpleNamespace(
        color_space="Linear sRGB", source_kind="rgb", width=48, height=48,
    )
    linear = np.full((48, 48, 3), round(0.09 * 65535), dtype=np.uint16)
    decoded = []

    def read_preview(*_args, **_kwargs):
        decoded.append(True)
        return linear.copy(), metadata

    monkeypatch.setattr(app_api, "read_image", read_preview)
    monkeypatch.setattr(
        app_api, "_correct_enhanced_raw",
        lambda image, *_a, **_k: (image, SimpleNamespace(), False),
    )
    monkeypatch.setattr(
        app_api, "_encode_preview",
        lambda image, **_k: np.ascontiguousarray(image).tobytes(),
    )
    with app_api._ENHANCE_LOCK:
        app_api._ENHANCE_SESSIONS[session_id] = {
            "files": {photo_id: path}, "created": 0.0,
        }
    with app_api._ENHANCE_LOCK:
        app_api._ENHANCE_PREVIEW_CACHE.clear()
    with app_api._DISPLAY_PREVIEW_LOCK:
        app_api._DISPLAY_PREVIEW_CACHE.clear()
        app_api._PROCESSING_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()

    client = TestClient(app_api.app)
    payload = {
        "session_id": session_id, "photo_id": photo_id, "mode": "dehazed",
        "auto_exposure": True,
    }
    try:
        origin = {"Origin": "http://localhost"}
        first = client.post("/api/enhance/preview", json=payload, headers=origin)
        cached = client.post("/api/enhance/preview", json=payload, headers=origin)
        assert first.status_code == cached.status_code == 200
        assert float(first.headers["X-Auto-Exposure-EV"]) == pytest.approx(1.0, abs=1e-4)
        assert first.headers["X-Auto-Exposure-Reason"] == "applied"
        assert cached.headers["X-Auto-Exposure-EV"] == first.headers["X-Auto-Exposure-EV"]
        assert cached.headers["X-Auto-Exposure-Reason"] == first.headers["X-Auto-Exposure-Reason"]
        assert "X-Auto-Exposure-EV" in first.headers.get("access-control-expose-headers", "")

        # Rebuild only the JPEG response: the dehazed display cache must carry
        # its EV diagnostics, and its key must distinguish the disabled mode.
        with app_api._ENHANCE_LOCK:
            app_api._ENHANCE_PREVIEW_CACHE.clear()
        decoded_before_display_cache_hit = len(decoded)
        replay = client.post("/api/enhance/preview", json=payload, headers=origin)
        assert replay.status_code == 200
        assert replay.headers["X-Auto-Exposure-EV"] == first.headers["X-Auto-Exposure-EV"]
        assert replay.headers["X-Auto-Exposure-Reason"] == first.headers["X-Auto-Exposure-Reason"]
        assert len(decoded) == decoded_before_display_cache_hit

        disabled = client.post(
            "/api/enhance/preview",
            json={**payload, "auto_exposure": False}, headers=origin,
        )
        assert disabled.status_code == 200
        assert disabled.headers["X-Auto-Exposure-EV"] == "0"
        assert disabled.headers["X-Auto-Exposure-Reason"] == "off"
        assert disabled.content != first.content

        # Selecting the original preview keeps its pixels independent of the toggle.
        original_off = client.post(
            "/api/enhance/preview",
            json={**payload, "mode": "original", "auto_exposure": False},
        )
        original_on = client.post(
            "/api/enhance/preview",
            json={**payload, "mode": "original", "auto_exposure": True},
        )
        assert original_on.status_code == 200, original_on.text
        assert original_off.content == original_on.content
        assert original_on.headers["X-Auto-Exposure-Reason"] == "off"
    finally:
        with app_api._ENHANCE_LOCK:
            app_api._ENHANCE_SESSIONS.pop(session_id, None)
            app_api._ENHANCE_PREVIEW_CACHE.clear()
        with app_api._DISPLAY_PREVIEW_LOCK:
            app_api._DISPLAY_PREVIEW_CACHE.clear()
            app_api._PROCESSING_PREVIEW_CACHE.clear()
            app_api._DEHAZED_PREVIEW_CACHE.clear()
            app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()


def test_xmp_auto_exposure_round_trips_through_complete_settings(tmp_path: Path):
    photo = tmp_path / "settings.jpg"
    photo.write_bytes(b"fixture")
    params = DehazeParams(strength=0.0).__dict__
    basic = {key: 0.0 for key in ricoh_filter._BASIC_FIELDS}

    ricoh_filter.write_photo_settings(
        photo, params, basic, None, False, auto_exposure=True,
    )
    assert ricoh_filter.read_photo_settings(photo)["dehaze_auto_exposure"] is True

    root = ricoh_filter._parse_xmp(photo.with_suffix(".xmp").read_bytes())
    description = ricoh_filter._description(root)
    assert description is not None
    assert description.attrib[
        "{" + ricoh_filter._IMPRINT_NS + "}DehazeAutoExposure"
    ] == "true"

    ricoh_filter.write_dehaze_settings(
        photo, params, auto_mode=False, auto_exposure=False,
    )
    assert ricoh_filter.read_photo_settings(photo)["dehaze_auto_exposure"] is False


def test_auto_exposure_session_xmp_uses_scoped_per_photo_flags(tmp_path, monkeypatch):
    first, second = tmp_path / "first.jpg", tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    session_id = "auto-xmp-session"
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"photo-1": first, "photo-2": second}, "created": 0.0,
    }
    monkeypatch.setattr(app_api, "_build_dehaze_xmp_curves", lambda *_a, **_k: None)
    try:
        result = app_api.save_enhance_session_xmp(app_api.EnhanceXmpRequest(
            session_id=session_id,
            params_by_photo={
                "photo-1": app_api.EnhanceParamsRequest(),
                "photo-2": app_api.EnhanceParamsRequest(),
            },
            auto_exposure=True,
            auto_exposures_by_photo={"photo-1": False, "photo-2": True, "stale": True},
        ))
        assert result["written"] == 2
        assert ricoh_filter.read_photo_settings(first)["dehaze_auto_exposure"] is False
        assert ricoh_filter.read_photo_settings(second)["dehaze_auto_exposure"] is True
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_auto_exposure_request_snapshots_are_per_photo_and_scoped():
    request = app_api.EnhanceRunRequest(
        session_id="session", auto_exposure=True,
        auto_exposures_by_photo={"photo-1": False, "outside": True},
    )

    default, scoped = app_api._snapshot_enhance_auto_exposures(request, {"photo-1", "photo-2"})

    assert default is True
    assert dict(scoped) == {"photo-1": False}
    assert app_api._select_enhance_auto_exposure("photo-1", scoped, default) is False
    assert app_api._select_enhance_auto_exposure("photo-2", scoped, default) is True


def test_batch_export_uses_frozen_per_photo_exposure_and_reports_xmp_flag(tmp_path, monkeypatch):
    path = tmp_path / "export.jpg"
    path.write_bytes(b"fixture")
    session_id, job_id = "auto-export-session", "auto-export-job"
    output_path = tmp_path / "export_dehaze.dng"
    metadata = SimpleNamespace(
        color_space="Linear sRGB", source_kind="rgb", exif={},
    )
    linear = np.full((6, 8, 3), 0.09, dtype=np.float32)
    captured = {}
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"photo-1": path}, "created": 0.0,
    }
    app_api._ENHANCE_JOBS[job_id] = {
        "status": "queued", "files": [{"photo_id": "photo-1", "status": "waiting"}],
        "_cancel": threading.Event(), "processed": 0, "total": 1,
        "progress": 0.0, "success": 0, "failed": 0,
    }
    monkeypatch.setattr(app_api, "read_image", lambda *_a, **_k: (linear.copy(), metadata))
    monkeypatch.setattr(
        app_api, "_correct_enhanced_raw",
        lambda image, *_a, **_k: (
            image,
            SimpleNamespace(
                distortion_applied=False, tca_applied=False,
                vignetting_applied=False, applied=False,
                camera_name=None, lens_name=None,
            ),
            False,
        ),
    )
    monkeypatch.setattr(
        app_api, "write_linear_dng",
        lambda image, *_a, **_k: (captured.update(export_pixels=image.copy()) or output_path),
    )
    monkeypatch.setattr(app_api, "_build_dehaze_xmp_curves", lambda *_a, **_k: None)
    monkeypatch.setattr(
        app_api, "write_dehaze_settings",
        lambda *_a, **kwargs: (captured.update(xmp_auto_exposure=kwargs["auto_exposure"]) or "export.xmp"),
    )
    frozen_flags = MappingProxyType({"photo-1": True})

    try:
        app_api._run_enhance_job(
            job_id, session_id, tmp_path, DehazeParams(strength=0.0), {},
            default_auto_exposure=False, auto_exposures_by_photo=frozen_flags,
        )
        item = app_api._ENHANCE_JOBS[job_id]["files"][0]
        assert np.allclose(captured["export_pixels"].astype(np.float32) / 65535.0, 0.18, atol=1 / 65535)
        assert captured["xmp_auto_exposure"] is True
        assert item["auto_exposure_ev"] == pytest.approx(1.0)
        assert item["auto_exposure_reason"] == "applied"
        assert item["status"] == "success"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)
