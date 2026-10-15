from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app_api


def test_ricoh_slider_previews_reuse_color_managed_source_and_notice_file_change(
    monkeypatch, tmp_path: Path,
):
    path = tmp_path / "photo.nef"
    path.write_bytes(b"first")
    session_id = "ricoh-input-cache-test"
    app_api._ENHANCE_SESSIONS[session_id] = {"files": {"photo-1": path}, "created": 0.0}
    app_api._RICOH_PREVIEW_CACHE.clear()
    app_api._ENHANCE_PREVIEW_CACHE.clear()
    with app_api._DISPLAY_PREVIEW_LOCK:
        app_api._PROCESSING_PREVIEW_CACHE.clear()
        app_api._DISPLAY_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
    calls = []

    def fake_read(_path, *, preview, max_edge):
        calls.append((preview, max_edge))
        image = np.full((8, 12, 3), 100 + len(calls), dtype=np.uint8)
        return image, SimpleNamespace(
            width=12, height=8, color_space="sRGB", source_kind="rgb",
        )

    monkeypatch.setattr(app_api, "read_image", fake_read)
    monkeypatch.setattr(app_api, "resolve_profile", lambda *_args: None)
    monkeypatch.setattr(app_api, "_standard_rgb_to_srgb", lambda image, _path: image)
    monkeypatch.setattr(app_api, "standard_preview_to_srgb", lambda image, _path: image)
    monkeypatch.setattr(app_api, "_render_dehaze", lambda image, *_args, **_kwargs: image.copy())
    monkeypatch.setattr(app_api, "apply_ricoh_preview_effect", lambda image, *_args, **_kwargs: image.copy())
    try:
        original = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id="photo-1", mode="original", color_manage_srgb=True,
        ))
        first = app_api.create_ricoh_preview(app_api.RicohPreviewRequest(
            session_id=session_id, photo_id="photo-1", preset_id="gr3_positive_film",
        ))
        second = app_api.create_ricoh_preview(app_api.RicohPreviewRequest(
            session_id=session_id, photo_id="photo-1", preset_id="gr3_positive_film",
            basic_params=app_api.BasicParamsRequest(exposure=0.5),
        ))
        assert original.status_code == first.status_code == second.status_code == 200
        assert first.headers["X-Preview-Approximation"] == "true"
        assert first.headers["X-Preview-Empirical-Color"] == "hsl-and-grading-response"
        assert calls == [(True, 1800)]

        path.write_bytes(b"changed content")
        third = app_api.create_ricoh_preview(app_api.RicohPreviewRequest(
            session_id=session_id, photo_id="photo-1", preset_id="gr3_positive_film",
            basic_params=app_api.BasicParamsRequest(exposure=1.0),
        ))
        assert third.status_code == 200
        assert calls == [(True, 1800), (True, 1800)]
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._RICOH_PREVIEW_CACHE.clear()
        app_api._ENHANCE_PREVIEW_CACHE.clear()
        with app_api._DISPLAY_PREVIEW_LOCK:
            app_api._PROCESSING_PREVIEW_CACHE.clear()
            app_api._DISPLAY_PREVIEW_CACHE.clear()
            app_api._DEHAZED_PREVIEW_CACHE.clear()
            app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
