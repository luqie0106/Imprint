from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import ImageCms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app_api
from dehaze import DehazeParams
from lens_correction import LensCorrectionResult


def _srgb_eotf(encoded: np.ndarray) -> np.ndarray:
    values = encoded.astype(np.float32) / 255.0
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def test_raw_and_linear_rgb_skip_srgb_eotf(tmp_path: Path):
    raw_linear = np.array([[[32768, 12000, 65535]]], dtype=np.uint16)
    raw_metadata = SimpleNamespace(source_kind="raw", color_space="Linear sRGB")

    actual = app_api._prepare_dehaze_input(
        raw_linear, raw_metadata, tmp_path / "sample.nef", color_manage_srgb=True,
    )

    expected = raw_linear.astype(np.float32) / 65535.0
    np.testing.assert_array_equal(actual, expected)

    explicitly_linear = app_api._prepare_dehaze_input(
        raw_linear,
        SimpleNamespace(source_kind="rgb", color_space="Linear sRGB"),
        tmp_path / "linear.tif",
        color_manage_srgb=True,
    )
    np.testing.assert_array_equal(explicitly_linear, expected)


def test_ordinary_rgb_is_icc_managed_before_srgb_eotf(tmp_path: Path, monkeypatch):
    encoded = np.array([[[64, 128, 192]]], dtype=np.uint8)
    icc_output = np.array([[[80, 128, 176]]], dtype=np.uint8)
    calls = []

    def convert(image: np.ndarray, path: Path) -> np.ndarray:
        calls.append((image.dtype, path))
        return icc_output

    monkeypatch.setattr(app_api, "_standard_rgb_to_srgb", convert)
    path = tmp_path / "tagged.jpg"
    actual = app_api._prepare_dehaze_input(
        encoded,
        SimpleNamespace(source_kind="rgb", color_space="sRGB"),
        path,
        color_manage_srgb=True,
    )

    assert calls == [(np.dtype("uint8"), path)]
    np.testing.assert_allclose(actual, _srgb_eotf(icc_output), rtol=0, atol=1e-7)


def test_rgb16_icc_transform_retains_more_than_8bit_precision():
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    levels = np.linspace(0, 65535, 2048, dtype=np.uint16)
    source = np.repeat(levels[None, :, None], 3, axis=2)

    converted = app_api._icc_rgb16_to_srgb(source, profile)

    assert converted.dtype == np.uint16
    assert np.unique(converted[..., 0]).size > 256
    np.testing.assert_allclose(converted, source, atol=2, rtol=0)


def test_cached_and_full_resolution_previews_share_linear_renderer(
    tmp_path: Path, monkeypatch,
):
    session_id = "linear-preview-pipeline"
    photo_id = "photo-1"
    path = tmp_path / "sample.nef"
    path.write_bytes(b"raw")
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {photo_id: path}, "created": 0.0,
    }
    app_api._ENHANCE_PREVIEW_CACHE.clear()
    app_api._DEHAZED_PREVIEW_CACHE.clear()
    app_api._PROCESSING_PREVIEW_CACHE.clear()
    source = np.full((8, 12, 3), 32768, dtype=np.uint16)
    metadata = SimpleNamespace(
        width=12, height=8, source_kind="raw", color_space="Linear sRGB",
    )
    seen: list[np.ndarray] = []

    def read_image(_path, *, preview, max_edge=2048):
        return source.copy(), metadata

    def physical(image, _params, *, backend, spatial, nonlocal_mode, diagnostics=None):
        assert nonlocal_mode == "off"
        assert image.dtype == np.float32
        seen.append(image.copy())
        return image

    monkeypatch.setattr(app_api, "read_image", read_image)
    monkeypatch.setattr(app_api, "apply_physical_dehaze", physical)
    monkeypatch.setattr(
        app_api, "_correct_enhanced_raw",
        lambda image, *_args, **_kwargs: (image, None, False),
    )
    try:
        preview = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id=photo_id, mode="dehazed",
            color_manage_srgb=True,
        ))
        full = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id=photo_id, mode="dehazed",
            color_manage_srgb=True, full_resolution=True,
        ))

        assert preview.status_code == full.status_code == 200
        assert len(seen) == 2
        np.testing.assert_array_equal(seen[0], seen[1])
        np.testing.assert_allclose(seen[0], 32768 / 65535, atol=1e-7)
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()
        app_api._PROCESSING_PREVIEW_CACHE.clear()


def test_rgb_export_decodes_to_linear16_and_uses_16bit_dng(
    tmp_path: Path, monkeypatch,
):
    session_id = "linear-export-pipeline"
    job_id = "linear-export-job"
    photo_id = "photo-1"
    path = tmp_path / "ordinary.jpg"
    path.write_bytes(b"photo")
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {photo_id: path}, "created": 0.0,
    }
    app_api._ENHANCE_JOBS[job_id] = {
        "job_id": job_id, "status": "queued", "total": 1, "processed": 0,
        "success": 0, "failed": 0, "progress": 0.0, "current_file": "",
        "files": [{"photo_id": photo_id, "name": path.name, "status": "waiting"}],
        "_cancel": app_api.threading.Event(),
    }
    encoded = np.full((2, 3, 3), 128, dtype=np.uint8)
    metadata = SimpleNamespace(
        source_kind="rgb", color_space="sRGB", bit_depth=8, exif={},
    )
    physical_inputs: list[np.ndarray] = []
    written: dict[str, object] = {}

    monkeypatch.setattr(
        app_api, "read_image", lambda *_args, **_kwargs: (encoded.copy(), metadata),
    )
    monkeypatch.setattr(
        app_api, "_standard_rgb_to_srgb",
        lambda image, _path: np.full_like(image, 128),
    )

    def physical(image, _params, *, backend, spatial, nonlocal_mode, diagnostics=None):
        assert nonlocal_mode == "off"
        physical_inputs.append(image.copy())
        return image

    monkeypatch.setattr(app_api, "apply_physical_dehaze", physical)
    monkeypatch.setattr(
        app_api, "_correct_enhanced_raw",
        lambda image, *_args, **_kwargs: (
            image, LensCorrectionResult(False, None, None, False, False, False), False,
        ),
    )

    def write_linear(image, _source, _output_dir, _exif, *, bits_per_sample):
        written["image"] = image.copy()
        written["bits_per_sample"] = bits_per_sample
        return tmp_path / "ordinary_dehaze.dng"

    monkeypatch.setattr(app_api, "write_linear_dng", write_linear)
    monkeypatch.setattr(app_api, "write_dehaze_settings", lambda *_args, **_kwargs: None)
    try:
        app_api._run_enhance_job(
            job_id, session_id, tmp_path, DehazeParams(), {},
        )

        assert app_api._ENHANCE_JOBS[job_id]["status"] == "completed"
        assert len(physical_inputs) == 1
        expected_linear = float(_srgb_eotf(np.array([[[128, 128, 128]]], dtype=np.uint8))[0, 0, 0])
        np.testing.assert_allclose(physical_inputs[0], expected_linear, atol=1e-7)
        assert isinstance(written["image"], np.ndarray)
        assert written["image"].dtype == np.uint16
        assert written["bits_per_sample"] == 16
        assert abs(int(written["image"][0, 0, 0]) - round(expected_linear * 65535)) <= 1
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


def test_preview_level_clamps_resize_roundoff_before_physical_core(
    tmp_path: Path, monkeypatch,
):
    session_id = "preview-roundoff-pipeline"
    photo_id = "photo-1"
    path = tmp_path / "sample.nef"
    path.write_bytes(b"raw")
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {photo_id: path}, "created": 0.0,
    }
    with app_api._DISPLAY_PREVIEW_LOCK:
        app_api._PROCESSING_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()
    metadata = SimpleNamespace(source_kind="raw", color_space="Linear sRGB")
    seen: list[np.ndarray] = []

    def read_image(_path, *, preview, max_edge):
        assert preview is True
        return np.ones((4, 4, 3), dtype=np.float32), metadata

    def resized_with_roundoff(_image, dimensions, *, interpolation):
        assert interpolation == app_api.cv2.INTER_AREA
        width, height = dimensions
        return np.full(
            (height, width, 3), 1.0 + np.finfo(np.float32).eps,
            dtype=np.float32,
        )

    def physical(image, _params, *, backend, spatial, nonlocal_mode, diagnostics=None):
        assert nonlocal_mode == "off"
        seen.append(image.copy())
        return image

    monkeypatch.setattr(app_api, "read_image", read_image)
    monkeypatch.setattr(app_api.cv2, "resize", resized_with_roundoff)
    monkeypatch.setattr(app_api, "apply_physical_dehaze", physical)
    monkeypatch.setattr(
        app_api, "_correct_enhanced_raw",
        lambda image, *_args, **_kwargs: (image, None, False),
    )
    try:
        display = app_api._cached_dehazed_display_preview(
            session_id, photo_id, path, 64, DehazeParams(), "cpu", False,
            preview_level=1,
        )

        assert len(seen) == 1
        assert seen[0].dtype == np.float32
        assert float(seen[0].min()) == float(seen[0].max()) == 1.0
        assert display.shape == (2, 2, 3)
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        with app_api._DISPLAY_PREVIEW_LOCK:
            app_api._PROCESSING_PREVIEW_CACHE.clear()
            app_api._DEHAZED_PREVIEW_CACHE.clear()
