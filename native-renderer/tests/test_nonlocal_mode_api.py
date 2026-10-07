from __future__ import annotations

from pathlib import Path
import sys
import threading

import numpy as np
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import app_api
import ricoh_filter


def _photo(path: Path) -> None:
    path.write_bytes(b"photo fixture")


def _session(session_id: str, **photos: Path) -> None:
    app_api._ENHANCE_SESSIONS[session_id] = {"files": photos, "created": 0.0}


@pytest.mark.parametrize("model,payload", [
    (app_api.EnhancePreviewRequest, {"session_id": "s", "photo_id": "p", "nonlocal_mode": "invalid"}),
    (app_api.EnhanceRunRequest, {"session_id": "s", "nonlocal_mode": "invalid"}),
    (app_api.EnhanceXmpRequest, {"session_id": "s", "nonlocal_mode": "invalid"}),
    (app_api.PhotoSettingsRequest, {
        "session_id": "s", "photo_id": "p", "nonlocal_mode": "invalid",
        "dehaze_params": app_api.EnhanceParamsRequest().model_dump(),
        "basic_params": app_api.BasicParamsRequest().model_dump(),
    }),
    (app_api.EnhanceRunRequest, {
        "session_id": "s", "nonlocal_modes_by_photo": {"p": "invalid"},
    }),
])
def test_nonlocal_api_modes_reject_values_outside_the_experiment_set(model, payload):
    with pytest.raises(ValidationError):
        model(**payload)


def test_xmp_round_trip_and_ricoh_preset_preserves_nonlocal_mode(tmp_path: Path, monkeypatch):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _photo(first)
    _photo(second)
    session_id = "per-photo-nonlocal-xmp-session"
    _session(session_id, **{"photo-1": first, "photo-2": second})
    curve_modes: dict[str, str] = {}

    def build_curves(_session, photo_id, _path, _params, _auto, mode=None, auto_exposure=False):
        curve_modes[photo_id] = mode
        return None

    monkeypatch.setattr(app_api, "_build_dehaze_xmp_curves", build_curves)
    try:
        result = app_api.save_enhance_session_xmp(app_api.EnhanceXmpRequest(
            session_id=session_id,
            params_by_photo={
                "photo-1": app_api.EnhanceParamsRequest(strength=0.3),
                "photo-2": app_api.EnhanceParamsRequest(strength=0.4),
            },
            nonlocal_mode="off",
            nonlocal_modes_by_photo={
                "photo-1": "strong", "photo-2": "conservative", "stale-id": "strong",
            },
        ))
        assert result["written"] == 2
        assert curve_modes == {"photo-1": "strong", "photo-2": "conservative"}
        assert ricoh_filter.read_photo_settings(first)["dehaze_nonlocal_mode"] == "strong"
        assert ricoh_filter.read_photo_settings(second)["dehaze_nonlocal_mode"] == "conservative"
        root = ricoh_filter._parse_xmp(first.with_suffix(".xmp").read_bytes())
        description = ricoh_filter._description(root)
        assert description is not None
        assert "{" + ricoh_filter._CRS_NS + "}Dehaze" not in description.attrib

        ricoh_filter.write_ricoh_preset(first, "gr3_standard")
        assert ricoh_filter.read_photo_settings(first)["dehaze_nonlocal_mode"] == "strong"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_preview_cache_isolated_by_nonlocal_mode(tmp_path: Path, monkeypatch):
    photo = tmp_path / "preview.jpg"
    _photo(photo)
    session_id = "nonlocal-preview-session"
    _session(session_id, **{"photo-1": photo})
    app_api._ENHANCE_PREVIEW_CACHE.clear()
    app_api._DEHAZED_PREVIEW_CACHE.clear()
    app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
    app_api._PROCESSING_PREVIEW_CACHE.clear()
    rendered: list[str | None] = []

    class Metadata:
        color_space = "Linear sRGB"
        source_kind = "rgb"

    monkeypatch.setattr(app_api, "read_image", lambda *_args, **_kwargs: (
        np.full((16, 16, 3), 80, dtype=np.uint8), Metadata(),
    ))

    def fake_render(image, _params, _backend, *, auto_mode=False, nonlocal_mode=None, diagnostics=None):
        rendered.append(nonlocal_mode)
        if diagnostics is not None:
            diagnostics.update(nonlocal_mode=nonlocal_mode, nonlocal_active=False,
                               fallback_reason="no_reliable_rays")
        delta = {"off": 0.0, "conservative": 0.1, "strong": 0.25}[nonlocal_mode]
        return np.clip(image + delta, 0.0, 1.0).astype(np.float32)

    monkeypatch.setattr(app_api, "_render_dehaze", fake_render)
    try:
        def preview(mode: str):
            return app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
                session_id=session_id, photo_id="photo-1", nonlocal_mode=mode,
                color_manage_srgb=False,
            ))

        conservative = preview("conservative")
        strong = preview("strong")
        conservative_again = preview("conservative")
        assert rendered == ["conservative", "strong"]
        assert conservative.body != strong.body
        assert conservative_again.body == conservative.body
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
        app_api._PROCESSING_PREVIEW_CACHE.clear()


def test_legacy_cache_mode_is_resolved_before_keying(tmp_path: Path, monkeypatch):
    photo = tmp_path / "legacy-preview.jpg"
    _photo(photo)
    app_api._DEHAZED_PREVIEW_CACHE.clear()
    app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
    app_api._PROCESSING_PREVIEW_CACHE.clear()
    monkeypatch.setenv("IMPRINT_NONLOCAL_RELIEF", "strong")
    modes: list[str | None] = []

    class Metadata:
        color_space = "Linear sRGB"
        source_kind = "rgb"

    monkeypatch.setattr(app_api, "read_image", lambda *_args, **_kwargs: (
        np.full((16, 16, 3), 80, dtype=np.uint8), Metadata(),
    ))

    def fake_render(image, _params, _backend, *, auto_mode=False, nonlocal_mode=None, diagnostics=None):
        modes.append(nonlocal_mode)
        if diagnostics is not None:
            diagnostics.update(nonlocal_mode=nonlocal_mode, nonlocal_active=False,
                               fallback_reason="manual_mode")
        delta = 0.2 if nonlocal_mode == "strong" else 0.0
        return np.clip(image + delta, 0.0, 1.0).astype(np.float32)

    monkeypatch.setattr(app_api, "_render_dehaze", fake_render)
    try:
        legacy = app_api._cached_dehazed_display_preview(
            "legacy-mode-session", "photo-1", photo, 1800,
            app_api.DehazeParams(strength=0.5), "cpu", False,
        )
        explicit_off = app_api._cached_dehazed_display_preview(
            "legacy-mode-session", "photo-1", photo, 1800,
            app_api.DehazeParams(strength=0.5), "cpu", False,
            nonlocal_mode="off",
        )
        assert modes == ["strong", "off"]
        assert not np.array_equal(legacy, explicit_off)
    finally:
        app_api._DEHAZED_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
        app_api._PROCESSING_PREVIEW_CACHE.clear()


def test_run_snapshots_per_photo_mode_for_render_export_and_xmp(tmp_path: Path, monkeypatch):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _photo(first)
    _photo(second)
    session_id = "nonlocal-snapshot-session"
    job_id = "nonlocal-snapshot-job"
    _session(session_id, **{"photo-1": first, "photo-2": second})
    app_api._ENHANCE_JOBS[job_id] = {
        "status": "queued", "files": [
            {"photo_id": "photo-1", "status": "waiting"},
            {"photo_id": "photo-2", "status": "waiting"},
        ], "_cancel": threading.Event(), "success": 0, "failed": 0,
        "processed": 0, "total": 2, "progress": 0.0,
    }

    class Metadata:
        color_space = "Linear sRGB"
        source_kind = "rgb"
        exif = {}

    render_modes: list[str] = []
    curve_modes: list[str] = []
    xmp_modes: list[str] = []
    monkeypatch.setattr(app_api, "read_image", lambda *_args, **_kwargs: (
        np.full((2, 2, 3), 0.2, dtype=np.float32), Metadata(),
    ))

    def render(image, _params, _backend, *, auto_mode=False, nonlocal_mode=None, diagnostics=None):
        render_modes.append(nonlocal_mode)
        if diagnostics is not None:
            diagnostics.update(nonlocal_mode=nonlocal_mode, nonlocal_active=False,
                               fallback_reason="no_reliable_rays")
        return image

    monkeypatch.setattr(app_api, "_render_dehaze", render)
    monkeypatch.setattr(app_api, "_correct_enhanced_raw", lambda image, *_a, **_k: (
        image, app_api.LensCorrectionResult(False, None, None, False, False, False), False,
    ))

    def write_dng(_image, source, output_dir, _metadata, **_kwargs):
        output = output_dir / (Path(source).stem + "_dehaze.dng")
        output.touch()
        return output

    monkeypatch.setattr(app_api, "write_linear_dng", write_dng)

    def build_curves(_session, photo_id, _path, _params, _auto, mode=None, auto_exposure=False):
        curve_modes.append(mode)
        return None

    def write_xmp(_path, _params, _basic, **kwargs):
        xmp_modes.append(kwargs["nonlocal_mode"])

    monkeypatch.setattr(app_api, "_build_dehaze_xmp_curves", build_curves)
    monkeypatch.setattr(app_api, "write_dehaze_settings", write_xmp)
    request = app_api.EnhanceRunRequest(
        session_id=session_id,
        nonlocal_mode="off",
        nonlocal_modes_by_photo={
            "photo-1": "strong", "photo-2": "conservative", "stale-id": "strong",
        },
    )
    default_mode, frozen_modes = app_api._snapshot_enhance_nonlocal_modes(
        request, {"photo-1", "photo-2"},
    )
    request.nonlocal_modes_by_photo["photo-1"] = "off"
    request.nonlocal_modes_by_photo["photo-2"] = "off"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    try:
        app_api._run_enhance_job(
            job_id, session_id, output_dir, app_api.DehazeParams(), {},
            default_nonlocal_mode=default_mode,
            nonlocal_modes_by_photo=frozen_modes,
        )
        assert render_modes == ["strong", "conservative"]
        assert curve_modes == render_modes
        assert xmp_modes == render_modes
        assert app_api._ENHANCE_JOBS[job_id]["status"] == "completed"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)
