from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app_api
from dehaze import DehazeParams
from lens_correction import LensCorrectionResult, LensMatchError


def test_enhance_params_request_accepts_and_converts_brightness_protection():
    request = app_api.EnhanceParamsRequest(brightness_protection=0.23)

    assert request.brightness_protection == 0.23
    assert request.to_params() == DehazeParams(brightness_protection=0.23)


def test_snapshot_scopes_photo_ids_and_copies_request_params():
    request = app_api.EnhanceRunRequest(
        session_id="session",
        params=app_api.EnhanceParamsRequest(strength=0.2, brightness_protection=0.31),
        params_by_photo={
            "photo-1": app_api.EnhanceParamsRequest(strength=0.9, brightness_protection=0.91),
            "outside-session": app_api.EnhanceParamsRequest(strength=0.1),
        },
    )

    default_params, params_by_photo = app_api._snapshot_enhance_params(
        request,
        {"photo-1", "photo-2"},
    )

    assert default_params == DehazeParams(strength=0.2, brightness_protection=0.31)
    assert params_by_photo["photo-1"] == DehazeParams(strength=0.9, brightness_protection=0.91)
    assert "outside-session" not in params_by_photo
    assert request.params_by_photo["photo-1"].strength == 0.9
    assert params_by_photo["photo-1"].strength == 0.9

    # The worker receives an immutable mapping and frozen parameter values.
    try:
        params_by_photo["photo-2"] = DehazeParams()  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("per-photo parameter snapshot must be immutable")


def test_enhance_job_uses_per_photo_params_and_default_fallback(monkeypatch, tmp_path: Path):
    session_id = "session-for-params"
    job_id = "job-for-params"
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    first_path.touch()
    second_path.touch()

    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"photo-1": first_path, "photo-2": second_path},
        "created": 0.0,
    }
    app_api._ENHANCE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "total": 2,
        "processed": 0,
        "success": 0,
        "failed": 0,
        "progress": 0.0,
        "current_file": "",
        "files": [
            {"photo_id": "photo-1", "name": first_path.name, "status": "waiting"},
            {"photo_id": "photo-2", "name": second_path.name, "status": "waiting"},
        ],
        "_cancel": app_api.threading.Event(),
    }

    seen: dict[int, DehazeParams] = {}

    class Metadata:
        exif = {}
        bit_depth = 14

    def fake_read_image(path: Path, *, preview: bool = False):
        value = 1 if Path(path).name == first_path.name else 2
        return np.full((1, 1, 3), value, dtype=np.uint16), Metadata()

    def fake_apply_dehaze(image: np.ndarray, params: DehazeParams):
        seen[int(image[0, 0, 0])] = params
        return image

    monkeypatch.setattr(app_api, "read_image", fake_read_image)
    monkeypatch.setattr(app_api, "to_uint16", lambda image: image)
    monkeypatch.setattr(app_api, "apply_dehaze", fake_apply_dehaze)
    monkeypatch.setattr(
        app_api,
        "write_linear_dng",
        lambda image, source_path, output_dir, exif, *, bits_per_sample: output_dir / f"{int(image[0, 0, 0])}.dng",
    )

    default_params = DehazeParams(strength=0.2)
    params_by_photo = {"photo-1": DehazeParams(strength=0.9)}
    try:
        app_api._run_enhance_job(
            job_id,
            session_id,
            tmp_path,
            default_params,
            params_by_photo,
        )
        assert seen[1] == DehazeParams(strength=0.9)
        assert seen[2] == default_params
        assert app_api._ENHANCE_JOBS[job_id]["status"] == "completed"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


def test_enhance_thumbnail_uses_preview_path_and_reuses_cache(monkeypatch, tmp_path: Path):
    session_id = "thumbnail-session"
    photo_id = "photo-1"
    path = tmp_path / "photo.jpg"
    path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {photo_id: path},
        "created": 0.0,
    }
    app_api._ENHANCE_THUMBNAIL_CACHE.clear()
    calls: list[tuple[Path, bool, int]] = []

    class Metadata:
        width = 12
        height = 8

    def fake_read_image(image_path: Path, *, preview: bool = False, max_edge: int = 2048):
        calls.append((Path(image_path), preview, max_edge))
        return np.full((8, 12, 3), 127, dtype=np.uint8), Metadata()

    monkeypatch.setattr(app_api, "read_image", fake_read_image)
    try:
        response = app_api.get_enhance_thumbnail(session_id, photo_id)
        assert response.status_code == 200
        assert response.media_type == "image/jpeg"
        with Image.open(BytesIO(response.body)) as image:
            assert image.format == "JPEG"
            assert image.size == (12, 8)
        assert calls == [(path, True, 360)]

        cached_response = app_api.get_enhance_thumbnail(session_id, photo_id)
        assert cached_response.status_code == 200
        assert cached_response.body == response.body
        assert calls == [(path, True, 360)]
        assert cached_response.headers["cache-control"] == "private, max-age=3600"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_THUMBNAIL_CACHE.clear()


def test_enhance_thumbnail_rejects_unknown_photo_id_without_reading(monkeypatch, tmp_path: Path):
    session_id = "thumbnail-invalid-session"
    path = tmp_path / "photo.jpg"
    path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"known-photo": path},
        "created": 0.0,
    }
    app_api._ENHANCE_THUMBNAIL_CACHE.clear()
    read_calls = 0

    def fail_read_image(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("unknown photo IDs must not read local files")

    monkeypatch.setattr(app_api, "read_image", fail_read_image)
    try:
        response = app_api.get_enhance_thumbnail(session_id, "unknown-photo")
        assert response.status_code == 404
        assert read_calls == 0
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_THUMBNAIL_CACHE.clear()


def test_enhance_preview_maps_raw_unsupported_without_leaking_exception_or_path(monkeypatch, tmp_path: Path):
    session_id = "unsupported-preview-session"
    photo_id = "photo-1"
    path = tmp_path / "nikon-he-star-nef"
    path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {photo_id: path},
        "created": 0.0,
    }

    def fail_read_image(*args, **kwargs):
        raise rawpy.LibRawFileUnsupportedError(
            b"Unsupported file format or not RAW file: /private/photos/secret.nef"
        )

    monkeypatch.setattr(app_api, "read_image", fail_read_image)
    try:
        response = app_api.create_enhance_preview(
            app_api.EnhancePreviewRequest(session_id=session_id, photo_id=photo_id)
        )
        assert response.status_code == 422
        body = response.body.decode("utf-8")
        assert "当前 RAW 或压缩方式暂不受支持" in body
        assert "Nikon NX Studio" in body
        assert "Unsupported file format" not in body
        assert "secret.nef" not in body
        assert str(path) not in body
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_enhance_job_stores_friendly_raw_error_for_failed_item(monkeypatch, tmp_path: Path):
    session_id = "unsupported-job-session"
    job_id = "unsupported-job"
    path = tmp_path / "private-source.nef"
    path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"photo-1": path},
        "created": 0.0,
    }
    app_api._ENHANCE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "total": 1,
        "processed": 0,
        "success": 0,
        "failed": 0,
        "progress": 0.0,
        "current_file": "",
        "files": [{"photo_id": "photo-1", "name": path.name, "status": "waiting"}],
        "_cancel": app_api.threading.Event(),
    }

    def fail_read_image(*args, **kwargs):
        raise rawpy.LibRawFileUnsupportedError(
            b"Unsupported file format or not RAW file: /private/photos/private-source.nef"
        )

    monkeypatch.setattr(app_api, "read_image", fail_read_image)
    try:
        app_api._run_enhance_job(
            job_id,
            session_id,
            tmp_path,
            DehazeParams(),
            {},
        )
        item = app_api._ENHANCE_JOBS[job_id]["files"][0]
        assert item["status"] == "failed"
        assert "当前 RAW 或压缩方式暂不受支持" in item["error"]
        assert "Unsupported file format" not in item["error"]
        assert "private-source.nef" not in item["error"]
        assert app_api._ENHANCE_JOBS[job_id]["failed"] == 1
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


def test_raw_export_requires_and_records_baked_lens_correction(monkeypatch, tmp_path: Path):
    session_id = "corrected-raw-session"
    job_id = "corrected-raw-job"
    path = tmp_path / "source.nef"
    path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"photo-1": path},
        "created": 0.0,
    }
    app_api._ENHANCE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "total": 1,
        "processed": 0,
        "success": 0,
        "failed": 0,
        "progress": 0.0,
        "current_file": "",
        "files": [{"photo_id": "photo-1", "name": path.name, "status": "waiting"}],
        "_cancel": app_api.threading.Event(),
    }

    class Metadata:
        source_kind = "raw"
        bit_depth = 14
        exif = {"Make": "Nikon", "Model": "Z6_3", "LensModel": "NIKKOR Z 14-24mm f/2.8 S"}

    require_values: list[bool] = []
    written_metadata: dict = {}

    def fake_correction(image, metadata, *, require_correction):
        require_values.append(require_correction)
        return image + 1, LensCorrectionResult(
            applied=True,
            camera_name="Nikon Z6_3",
            lens_name="Nikkor Z 14-24mm f/2.8 S",
            distortion_applied=True,
            tca_applied=True,
            vignetting_applied=False,
            scale=0.0,
        )

    def fake_write(image, source_path, output_dir, metadata, *, bits_per_sample):
        written_metadata.update(metadata)
        assert int(image[0, 0, 0]) == 2
        assert bits_per_sample == 14
        return tmp_path / "output.dng"

    monkeypatch.setattr(
        app_api,
        "read_image",
        lambda *args, **kwargs: (np.ones((2, 2, 3), dtype=np.uint16), Metadata()),
    )
    monkeypatch.setattr(app_api, "apply_dehaze", lambda image, params: image)
    monkeypatch.setattr(app_api, "apply_lens_correction", fake_correction)
    monkeypatch.setattr(app_api, "write_linear_dng", fake_write)
    try:
        app_api._run_enhance_job(job_id, session_id, tmp_path, DehazeParams(), {})
        item = app_api._ENHANCE_JOBS[job_id]["files"][0]
        assert require_values == [True]
        assert item["status"] == "success"
        assert item["lens_correction"]["distortion"] is True
        assert item["lens_correction"]["tca"] is True
        assert written_metadata["LensCorrectionApplied"] is True
        assert written_metadata["LensCorrectionOperations"] == "distortion,tca"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


def test_lens_match_failure_never_reports_an_uncorrected_raw_success(monkeypatch, tmp_path: Path):
    assert "未生成未校正的 DNG" in app_api._enhance_error_message(
        LensMatchError("/private/photos/secret.nef")
    )
    assert "secret.nef" not in app_api._enhance_error_message(
        LensMatchError("/private/photos/secret.nef")
    )
