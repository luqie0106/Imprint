from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import app_api


def test_preview_reports_actual_nonlocal_state_through_image_and_jpeg_caches(
    tmp_path: Path, monkeypatch,
):
    photo = tmp_path / "status-preview.jpg"
    photo.write_bytes(b"preview fixture")
    session_id = "nonlocal-status-session"
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"photo-1": photo}, "created": 0.0,
    }
    app_api._ENHANCE_PREVIEW_CACHE.clear()
    app_api._DEHAZED_PREVIEW_CACHE.clear()
    app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
    app_api._PROCESSING_PREVIEW_CACHE.clear()
    monkeypatch.setattr(app_api, "_MAX_DEHAZED_PREVIEW_STATUS_ENTRIES", 1)
    renders: list[str | None] = []

    class Metadata:
        color_space = "Linear sRGB"
        source_kind = "rgb"
        width = 14
        height = 12

    monkeypatch.setattr(app_api, "read_image", lambda *_args, **_kwargs: (
        np.full((12, 14, 3), 80, dtype=np.uint8), Metadata(),
    ))

    def render(image, _params, _backend, *, auto_mode=False, nonlocal_mode=None, diagnostics=None):
        renders.append(nonlocal_mode)
        if diagnostics is not None:
            if nonlocal_mode == "conservative":
                diagnostics.update(nonlocal_mode=nonlocal_mode, nonlocal_active=True,
                                   fallback_reason="")
            else:
                diagnostics.update(nonlocal_mode=nonlocal_mode, nonlocal_active=False,
                                   fallback_reason="uncertain_airlight")
        return np.asarray(image, dtype=np.float32).copy()

    monkeypatch.setattr(app_api, "_render_dehaze", render)

    def preview(mode: str, *, preview_mode: str = "dehazed"):
        return app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id=session_id,
            photo_id="photo-1",
            nonlocal_mode=mode,
            mode=preview_mode,
            color_manage_srgb=False,
        ))

    try:
        active = preview("conservative")
        assert active.headers["X-Dehaze-Nonlocal-Status"] == "active"
        assert active.headers["X-Dehaze-Nonlocal-Reason"] == ""

        # The JPEG cache must replay the state paired with the rendered image.
        active_jpeg_hit = preview("conservative")
        assert active_jpeg_hit.body == active.body
        assert active_jpeg_hit.headers["X-Dehaze-Nonlocal-Status"] == "active"
        assert renders == ["conservative"]

        # Clearing only the JPEG cache exercises the independent display-image
        # and status caches without invoking diagnostics a second time.
        app_api._ENHANCE_PREVIEW_CACHE.clear()
        active_image_hit = preview("conservative")
        assert active_image_hit.body == active.body
        assert active_image_hit.headers["X-Dehaze-Nonlocal-Status"] == "active"
        assert renders == ["conservative"]

        fallback = preview("strong")
        assert fallback.headers["X-Dehaze-Nonlocal-Status"] == "fallback"
        assert fallback.headers["X-Dehaze-Nonlocal-Reason"] == "uncertain_airlight"
        assert fallback.headers["X-Dehaze-Nonlocal-Reason"].isascii()

        # A status-cache eviction also evicts its image, so a later render
        # cannot pair an old JPEG base with another render's status.
        app_api._ENHANCE_PREVIEW_CACHE.clear()
        active_after_status_eviction = preview("conservative")
        assert active_after_status_eviction.headers["X-Dehaze-Nonlocal-Status"] == "active"
        assert renders == ["conservative", "strong", "conservative"]

        original = preview("strong", preview_mode="original")
        assert original.headers["X-Dehaze-Nonlocal-Status"] == "off"
        assert original.headers["X-Dehaze-Nonlocal-Reason"] == ""
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
        app_api._PROCESSING_PREVIEW_CACHE.clear()
