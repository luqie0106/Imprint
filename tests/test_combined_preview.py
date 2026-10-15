from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app_api


def _add_preview_photo(session_id: str, photo_id: str, path: Path) -> None:
    path.touch()
    app_api._ENHANCE_SESSIONS.setdefault(session_id, {"files": {}, "created": 0.0})
    app_api._ENHANCE_SESSIONS[session_id]["files"][photo_id] = path


def _clear_preview_state(session_id: str) -> None:
    app_api._ENHANCE_SESSIONS.pop(session_id, None)
    app_api._ENHANCE_PREVIEW_CACHE.clear()
    with app_api._DISPLAY_PREVIEW_LOCK:
        app_api._DISPLAY_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()


def test_combined_preview_reuses_identical_snapshot_and_separates_effect_inputs(
    monkeypatch, tmp_path: Path,
):
    session_id = "combined-preview-cache-session"
    photo_id = "photo-1"
    path = tmp_path / "sample.jpg"
    _add_preview_photo(session_id, photo_id, path)
    _clear_preview_state(session_id)
    _add_preview_photo(session_id, photo_id, path)

    yy, xx = np.mgrid[:24, :32]
    source = np.stack((40 + xx * 3, 50 + yy * 3, 65 + (xx + yy) * 2), axis=2).astype(np.uint8)
    metadata = SimpleNamespace(
        width=source.shape[1], height=source.shape[0], color_space="sRGB", source_kind="rgb",
    )
    read_edges: list[int] = []
    backends: list[str] = []
    strengths: list[float] = []
    basic_effect_calls: list[dict[str, float]] = []
    ricoh_effect_calls: list[tuple[str, bool]] = []

    def fake_read_image(_path, *, preview: bool, max_edge: int):
        assert preview is True
        read_edges.append(max_edge)
        # Make max_edge part of the rendered result as well as the cache identity.
        return np.clip(source.astype(np.int16) + (max_edge % 7), 0, 255).astype(np.uint8), metadata

    def fake_apply_physical(image, params, *, backend: str, spatial: bool, nonlocal_mode=None, diagnostics=None):
        assert nonlocal_mode == "off"
        assert image.dtype == np.float32
        backends.append(backend)
        strengths.append(params.strength)
        amount = params.strength * 0.16 + (0.015 if backend == "auto" else 0.0)
        return np.clip(image + amount, 0.0, 1.0).astype(np.float32)

    def fake_color_manage(image, _path):
        return np.clip(image.astype(np.int16) + 4, 0, 255).astype(np.uint8)

    def fake_basic_effect(image, basic):
        basic_effect_calls.append(dict(basic))
        amount = round(basic["exposure"] * 18)
        return np.clip(image.astype(np.int16) + amount, 0, 255).astype(np.uint8)

    def fake_ricoh_effect(image, preset_id, basic, *, use_measured_color=False):
        ricoh_effect_calls.append((preset_id, use_measured_color))
        assert image.dtype == np.uint16
        assert basic["exposure"] == 0
        amount = 35 if preset_id == "gr3-vivid" else 50
        return np.clip(image.astype(np.uint32) + amount * 257, 0, 65535).astype(np.uint16)

    monkeypatch.setattr(app_api, "read_image", fake_read_image)
    monkeypatch.setattr(app_api, "apply_physical_dehaze", fake_apply_physical)
    monkeypatch.setattr(app_api, "_standard_rgb_to_srgb", fake_color_manage)
    monkeypatch.setattr(app_api, "apply_basic_preview_effect", fake_basic_effect)
    monkeypatch.setattr(app_api, "apply_ricoh_preview_effect", fake_ricoh_effect)
    monkeypatch.setattr(app_api, "list_ricoh_presets", lambda: [{"id": "gr3-vivid"}])

    base_request = app_api.EnhancePreviewRequest(
        session_id=session_id, photo_id=photo_id, color_manage_srgb=True,
    )
    try:
        base = app_api.create_enhance_preview(base_request)
        repeat = app_api.create_enhance_preview(base_request)
        assert base.status_code == repeat.status_code == 200
        assert base_request.params.strength == 0.0
        assert repeat.body == base.body
        assert len(app_api._ENHANCE_PREVIEW_CACHE) == 1
        assert len(backends) == 1

        with_preset = app_api.create_enhance_preview(base_request.model_copy(
            update={"ricoh_preset_id": "gr3-vivid"},
        ))
        removed_preset = app_api.create_enhance_preview(base_request)
        changed_dehaze = app_api.create_enhance_preview(base_request.model_copy(
            update={"params": app_api.EnhanceParamsRequest(strength=0.92)},
        ))
        changed_basic = app_api.create_enhance_preview(base_request.model_copy(
            update={"basic_params": app_api.BasicParamsRequest(exposure=1.5)},
        ))
        unmanaged_rgb = app_api.create_enhance_preview(base_request.model_copy(
            update={"color_manage_srgb": False},
        ))
        changed_edge = app_api.create_enhance_preview(base_request.model_copy(
            update={"max_edge": 1200},
        ))
        gpu = app_api.create_enhance_preview(base_request.model_copy(
            update={"use_gpu": True},
        ))
        assert all(response.status_code == 200 for response in (
            with_preset, removed_preset, changed_dehaze, changed_basic,
            unmanaged_rgb, changed_edge, gpu,
        ))
        assert removed_preset.body == base.body
        assert len({base.body, with_preset.body, changed_dehaze.body,
                    changed_basic.body, unmanaged_rgb.body, changed_edge.body,
                    gpu.body}) == 7
        assert len(app_api._ENHANCE_PREVIEW_CACHE) == 7
        # Basic edits reuse the base; switching presets rebuilds its baked Ricoh layer.
        assert len(backends) == 6
        assert strengths[0] == 0.0
        assert set(read_edges) == {1200, 1800}
        assert "auto" in backends
        assert basic_effect_calls
        assert ricoh_effect_calls == [("gr3-vivid", True)]

        # A dehaze implementation change must also invalidate an otherwise identical key.
        monkeypatch.setattr(app_api, "DEHAZE_ALGORITHM_VERSION", "test-algorithm-version")
        algorithm_changed = app_api.create_enhance_preview(base_request)
        assert algorithm_changed.status_code == 200
        assert len(app_api._ENHANCE_PREVIEW_CACHE) == 8
        assert len(backends) == 7
    finally:
        _clear_preview_state(session_id)


def test_combined_preview_calls_effect_stages_in_order_and_uses_measured_color(
    monkeypatch, tmp_path: Path,
):
    session_id = "combined-preview-order-session"
    photo_id = "photo-1"
    path = tmp_path / "sample.jpg"
    _add_preview_photo(session_id, photo_id, path)
    _clear_preview_state(session_id)
    _add_preview_photo(session_id, photo_id, path)

    native = np.full((3, 4, 3), 20, dtype=np.uint8)
    dehazed_linear = np.full((3, 4, 3), 0.1, dtype=np.float32)
    displayed = np.full((3, 4, 3), 40, dtype=np.uint8)
    managed = np.full((3, 4, 3), 50, dtype=np.uint8)
    effected = np.full((3, 4, 3), 60, dtype=np.uint8)
    metadata = SimpleNamespace(width=4, height=3, color_space="sRGB", source_kind="rgb")
    calls: list[str] = []
    request = app_api.EnhancePreviewRequest(
        session_id=session_id, photo_id=photo_id, ricoh_preset_id="gr3-vivid",
        color_manage_srgb=True, use_gpu=True,
        params=app_api.EnhanceParamsRequest(strength=0.63),
        basic_params=app_api.BasicParamsRequest(exposure=0.4),
    )

    def fake_read_image(read_path, *, preview: bool, max_edge: int):
        calls.append("read")
        assert Path(read_path) == path
        assert preview is True and max_edge == request.max_edge
        return native, metadata

    def fake_physical(image, params, *, backend: str, spatial: bool, nonlocal_mode=None, diagnostics=None):
        assert nonlocal_mode == "off"
        calls.append("dehaze")
        assert image.dtype == np.float32
        np.testing.assert_allclose(image, ((50 / 255 + 0.055) / 1.055) ** 2.4, atol=1e-7)
        assert params.strength == 0.63
        assert backend == "auto"
        return dehazed_linear

    def fake_display(image, *, linear: bool):
        calls.append("display")
        assert image.dtype == np.uint16
        assert float(image.mean()) > round(0.1 * 65535)
        assert linear is True
        return displayed

    def fake_standard(image, source_path):
        calls.append("standard_to_srgb")
        assert image is native
        assert Path(source_path) == path
        return managed

    def fake_ricoh(image, preset_id, basic, *, use_measured_color=False):
        calls.append("ricoh")
        assert image.dtype == np.uint16
        assert preset_id == "gr3-vivid"
        assert basic == app_api.BasicParamsRequest().values()
        assert use_measured_color is True
        return np.full_like(image, 32000)

    def fake_basic(image, basic, mode):
        calls.append("basic")
        assert image is displayed
        assert basic["exposure"] == 0.4
        assert mode == "python"
        return effected

    def fake_encode(image):
        calls.append("encode")
        assert image is effected
        return b"jpeg-preview"

    monkeypatch.setattr(app_api, "read_image", fake_read_image)
    monkeypatch.setattr(app_api, "apply_physical_dehaze", fake_physical)
    monkeypatch.setattr(app_api, "_display_rgb8", fake_display)
    monkeypatch.setattr(app_api, "_standard_rgb_to_srgb", fake_standard)
    monkeypatch.setattr(app_api, "apply_ricoh_preview_effect", fake_ricoh)
    monkeypatch.setattr(app_api, "_render_basic", fake_basic)
    monkeypatch.setattr(app_api, "_encode_preview", fake_encode)
    monkeypatch.setattr(app_api, "list_ricoh_presets", lambda: [{"id": "gr3-vivid"}])

    try:
        response = app_api.create_enhance_preview(request)
        assert response.status_code == 200
        assert response.body == b"jpeg-preview"
        assert calls == ["read", "standard_to_srgb", "dehaze", "ricoh", "display", "basic", "encode"]
    finally:
        _clear_preview_state(session_id)


def test_combined_preview_rgb_and_linear_conversion_follow_color_mode(
    monkeypatch, tmp_path: Path,
):
    session_id = "combined-preview-color-session"
    rgb_path = tmp_path / "ordinary.jpg"
    linear_path = tmp_path / "linear.dng"
    _add_preview_photo(session_id, "rgb-photo", rgb_path)
    _add_preview_photo(session_id, "linear-photo", linear_path)
    _clear_preview_state(session_id)
    _add_preview_photo(session_id, "rgb-photo", rgb_path)
    _add_preview_photo(session_id, "linear-photo", linear_path)

    rgb = np.full((2, 2, 3), 80, dtype=np.uint8)
    linear = np.full((2, 2, 3), 32768, dtype=np.uint16)
    metadata_by_name = {
        rgb_path.name: SimpleNamespace(width=2, height=2, color_space="sRGB", source_kind="rgb"),
        linear_path.name: SimpleNamespace(
            width=2, height=2, color_space="Linear sRGB", source_kind="raw",
        ),
    }
    basic_inputs: dict[str, np.ndarray] = {}
    color_calls: list[str] = []

    def fake_read_image(path, *, preview: bool, max_edge: int):
        assert preview is True
        return (rgb if Path(path).name == rgb_path.name else linear), metadata_by_name[Path(path).name]

    def fake_standard(image, path):
        color_calls.append(Path(path).name)
        return image

    def fake_basic(image, _params):
        basic_inputs["current"] = image.copy()
        return image

    monkeypatch.setattr(app_api, "read_image", fake_read_image)
    monkeypatch.setattr(
        app_api, "apply_physical_dehaze",
        lambda image, _params, *, backend, spatial, nonlocal_mode=None, diagnostics=None: image,
    )
    # This fixture tests colour transfer, not optical decoding of its empty DNG.
    monkeypatch.setattr(app_api, "_correct_enhanced_raw", lambda image, *_a, **_k: (image, None, False))
    monkeypatch.setattr(app_api, "_standard_rgb_to_srgb", fake_standard)
    monkeypatch.setattr(app_api, "apply_basic_preview_effect", fake_basic)

    try:
        ordinary = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id="rgb-photo", color_manage_srgb=False,
        ))
        assert ordinary.status_code == 200
        assert np.all(basic_inputs["current"] == 80)
        assert color_calls == []

        converted_linear = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id="linear-photo", color_manage_srgb=True,
        ))
        assert converted_linear.status_code == 200
        assert np.all(basic_inputs["current"] == 188)
        assert color_calls == []
    finally:
        _clear_preview_state(session_id)


def test_combined_preview_rejects_expired_photo_and_unknown_preset_without_reading(
    monkeypatch, tmp_path: Path,
):
    session_id = "combined-preview-invalid-session"
    path = tmp_path / "sample.jpg"
    _add_preview_photo(session_id, "known-photo", path)
    _clear_preview_state(session_id)
    _add_preview_photo(session_id, "known-photo", path)
    read_calls = 0

    def fail_read(*_args, **_kwargs):
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("invalid requests must not read an image")

    monkeypatch.setattr(app_api, "read_image", fail_read)
    monkeypatch.setattr(app_api, "list_ricoh_presets", lambda: [{"id": "gr3-vivid"}])

    try:
        expired_session = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id="expired-session", photo_id="known-photo",
        ))
        expired_photo = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id="missing-photo",
        ))
        unknown_preset = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id="known-photo", ricoh_preset_id="unknown-preset",
        ))

        assert expired_session.status_code == 404
        assert expired_photo.status_code == 404
        assert unknown_preset.status_code == 400
        assert read_calls == 0
    finally:
        _clear_preview_state(session_id)
