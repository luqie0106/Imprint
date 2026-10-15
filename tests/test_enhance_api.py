from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rawpy
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app_api
import ricoh_filter
from dehaze import DehazeParams
from dng_gainmap import DNGGainMapError
from lens_correction import (
    LensCorrectionError,
    LensCorrectionNotAppliedError,
    LensCorrectionResult,
    LensMatchError,
    LensMetadataUnavailableError,
    LensfunUnavailableError,
)


def test_export_depth_uses_sensor_instead_of_dng_jpeg_preview():
    metadata = SimpleNamespace(source_kind="raw", bit_depth=16,
                               exif={"BitsPerSample": (8, 8, 8)})
    assert app_api._resolve_export_bit_depth(metadata, "source") == (
        16, 16, "sensor_white_level",
    )
    metadata.bit_depth = 14
    assert app_api._resolve_export_bit_depth(metadata, "source")[0:2] == (14, 14)
    assert app_api._resolve_export_bit_depth(metadata, "16")[0:2] == (16, 14)


def test_export_bakes_auto_exposure_before_dehaze_without_adding_it_to_xmp(monkeypatch, tmp_path):
    path = tmp_path / "source.png"
    path.touch()
    session_id, job_id = "auto-order-session", "auto-order-job"
    app_api._ENHANCE_SESSIONS[session_id] = {"files": {"photo": path}}
    job = {"files": [{"photo_id": "photo", "status": "waiting"}],
           "_cancel": app_api.threading.Event(), "success": 0, "failed": 0,
           "processed": 0, "total": 1}
    app_api._ENHANCE_JOBS[job_id] = job
    metadata = SimpleNamespace(source_kind="rgb", color_space="Linear sRGB", exif={})
    monkeypatch.setattr(app_api, "read_image", lambda *_a, **_k: (
        np.full((2, 2, 3), 1000, np.uint16), metadata))

    def render(image, params, backend, *, auto_exposure, diagnostics, **kwargs):
        assert auto_exposure is True
        diagnostics.update(auto_exposure_ev=2.0, auto_exposure_reason="test")
        return image * 4.0

    monkeypatch.setattr(app_api, "_render_dehaze", render)
    correction = SimpleNamespace(applied=False, distortion_applied=False,
                                 tca_applied=False, vignetting_applied=False,
                                 camera_name=None, lens_name=None)
    monkeypatch.setattr(app_api, "_correct_enhanced_raw", lambda image, *_a, **_k: (
        image, correction, False))
    captured = {}

    def write(image, source, directory, info, **kwargs):
        captured.update(image=image.copy(), info=info)
        return directory / "result.dng"

    monkeypatch.setattr(app_api, "write_linear_dng", write)
    try:
        basic = app_api.BasicParamsRequest(exposure=1.9).values()
        app_api._run_enhance_job(job_id, session_id, tmp_path, DehazeParams(), {},
                                basic_params_by_photo={"photo": basic},
                                default_auto_exposure=True,
                                preset_ids_by_photo={"photo": None})
        assert job["success"] == 1
        np.testing.assert_array_equal(captured["image"], np.full((2, 2, 3), 4000, np.uint16))
        assert captured["info"]["ACRBasicParams"]["exposure"] == 1.9
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


def test_linear_preview_encodes_display_srgb_brightness():
    samples = np.array([0, 32768, 65535], dtype=np.uint16).reshape(1, 1, 3)
    display = app_api._display_rgb8(samples, linear=True)
    assert display.tolist() == [[[0, 188, 255]]]


def test_enhance_preview_displays_linear_raw_with_srgb_transfer(monkeypatch, tmp_path: Path):
    path = tmp_path / "sample.nef"
    path.touch()
    session_id = "linear-preview-session"
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"photo-1": path}, "created": 0.0,
    }
    app_api._ENHANCE_PREVIEW_CACHE.clear()
    monkeypatch.setattr(
        app_api, "read_image",
        lambda *_args, **_kwargs: (
            np.full((32, 32, 3), 32768, dtype=np.uint16),
            SimpleNamespace(width=32, height=32, color_space="Linear sRGB"),
        ),
    )
    try:
        response = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id="photo-1", mode="original",
        ))
        assert response.status_code == 200
        with Image.open(BytesIO(response.body)) as preview:
            assert all(abs(channel - 188) <= 1 for channel in preview.getpixel((16, 16)))
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_PREVIEW_CACHE.clear()


def test_enhance_params_request_accepts_and_converts_brightness_protection():
    request = app_api.EnhanceParamsRequest(brightness_protection=0.23)

    assert request.brightness_protection == 0.23
    assert request.to_params() == DehazeParams(brightness_protection=0.23)


def test_old_nonphysical_xmp_algorithm_is_ignored_and_writes_are_physical(tmp_path: Path):
    photo = tmp_path / "sample.jpg"
    photo.write_bytes(b"photo")
    params = DehazeParams().__dict__
    basic = {key: 0.0 for key in ricoh_filter._BASIC_FIELDS}
    ricoh_filter.write_photo_settings(photo, params, basic, None, False)
    sidecar = photo.with_suffix(".xmp")

    for legacy_algorithm in ("ridcp", "future-model"):
        root = ricoh_filter._parse_xmp(sidecar.read_bytes())
        description = ricoh_filter._description(root)
        assert description is not None
        description.set(
            "{" + ricoh_filter._IMPRINT_NS + "}DehazeAlgorithm",
            legacy_algorithm,
        )
        sidecar.write_bytes(ricoh_filter._serialize_xmp(root))
        assert ricoh_filter.read_photo_settings(photo)["dehaze_algorithm"] == "physical"

    for write in (
        lambda: ricoh_filter.write_photo_settings(photo, params, basic, None, False),
        lambda: ricoh_filter.write_dehaze_settings(photo, params, basic, False),
    ):
        write()
        root = ricoh_filter._parse_xmp(sidecar.read_bytes())
        description = ricoh_filter._description(root)
        assert description is not None
        assert description.attrib["{" + ricoh_filter._IMPRINT_NS + "}DehazeAlgorithm"] == "physical"


def test_ridcp_status_route_is_removed():
    assert not any(
        getattr(route, "path", None) == "/api/enhance/ridcp-status"
        for route in app_api.app.routes
    )


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

    seen: list[DehazeParams] = []
    seen_backends: list[str] = []

    class Metadata:
        exif = {}
        bit_depth = 14
        source_kind = "rgb"

    def fake_read_image(path: Path, *, preview: bool = False):
        value = 1 if Path(path).name == first_path.name else 2
        return np.full((1, 1, 3), value, dtype=np.uint16), Metadata()

    def fake_apply_physical(image: np.ndarray, params: DehazeParams, *, backend: str, spatial: bool, nonlocal_mode: str, diagnostics=None):
        assert image.dtype == np.float32
        assert nonlocal_mode == "off"
        seen.append(params)
        seen_backends.append(backend)
        return image

    monkeypatch.setattr(app_api, "read_image", fake_read_image)
    monkeypatch.setattr(app_api, "_standard_rgb_to_srgb", lambda image, _path: image)
    monkeypatch.setattr(app_api, "apply_physical_dehaze", fake_apply_physical)
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
        assert seen == [DehazeParams(strength=0.9), default_params]
        assert seen_backends == ["cpu", "cpu"]
        assert app_api._ENHANCE_JOBS[job_id]["status"] == "completed"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


def test_enhance_preview_uses_physical_backend_and_separates_gpu_cache(monkeypatch, tmp_path: Path):
    session_id = "gpu-preview-session"
    photo_id = "photo-1"
    path = tmp_path / "photo.jpg"
    path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {photo_id: path},
        "created": 0.0,
    }
    app_api._ENHANCE_PREVIEW_CACHE.clear()
    calls: list[str] = []

    class Metadata:
        width = 2
        height = 2

    def fake_read_image(*_args, **_kwargs):
        return np.full((2, 2, 3), 100, dtype=np.uint8), Metadata()

    def fake_apply_physical(image, _params, *, backend, spatial, nonlocal_mode, diagnostics=None):
        assert nonlocal_mode == "off"
        calls.append(backend)
        return image

    monkeypatch.setattr(app_api, "read_image", fake_read_image)
    monkeypatch.setattr(app_api, "_standard_rgb_to_srgb", lambda image, _path: image)
    monkeypatch.setattr(app_api, "apply_physical_dehaze", fake_apply_physical)
    try:
        cpu_request = app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id=photo_id, use_gpu=False,
        )
        gpu_request = app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id=photo_id, use_gpu=True,
        )
        first_cpu = app_api.create_enhance_preview(cpu_request)
        cached_cpu = app_api.create_enhance_preview(cpu_request)
        first_gpu = app_api.create_enhance_preview(gpu_request)
        cached_gpu = app_api.create_enhance_preview(gpu_request)

        assert first_cpu.status_code == cached_cpu.status_code == first_gpu.status_code == cached_gpu.status_code == 200
        # The GPU request selects the optional new native float renderer,
        # while the default preview uses the deterministic CPU path.
        assert calls == ["cpu", "auto"]
        assert len(app_api._ENHANCE_PREVIEW_CACHE) == 2
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_PREVIEW_CACHE.clear()


def test_enhance_preview_passes_progressive_levels_to_native_cache(monkeypatch, tmp_path: Path):
    session_id = "progressive-preview-session"
    photo_id = "photo-1"
    path = tmp_path / "photo.jpg"
    path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {"files": {photo_id: path}, "created": 0.0}
    app_api._ENHANCE_PREVIEW_CACHE.clear()
    app_api._DEHAZED_PREVIEW_CACHE.clear()
    seen_shapes: list[tuple[int, int]] = []
    reads: list[int] = []

    class Metadata:
        color_space = "Linear sRGB"
        source_kind = "raw"

    def fake_read_image(_path, *, preview, max_edge):
        assert preview
        reads.append(max_edge)
        return np.full((40, 60, 3), 32768, dtype=np.uint16), Metadata()

    def fake_apply_physical(image, _params, *, backend, spatial, nonlocal_mode, diagnostics=None):
        assert nonlocal_mode == "off"
        seen_shapes.append((image.shape[1], image.shape[0]))
        assert image.dtype == np.float32
        assert backend == "native"
        assert spatial
        return image

    monkeypatch.setattr(app_api, "read_image", fake_read_image)
    monkeypatch.setattr(app_api, "apply_physical_dehaze", fake_apply_physical)
    try:
        for level, expected_size in ((2, (15, 10)), (1, (30, 20)), (0, (60, 40))):
            response = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
                session_id=session_id, photo_id=photo_id, render_backend="native",
                basic_backend="python", preview_level=level, auto_mode=True,
            ))
            assert response.status_code == 200
            with Image.open(BytesIO(response.body)) as preview:
                assert preview.size == expected_size
        assert seen_shapes == [(15, 10), (30, 20), (60, 40)]
        assert reads == [1800]
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()


def test_enhance_thumbnail_uses_embedded_preview_and_reuses_cache(monkeypatch, tmp_path: Path):
    session_id = "thumbnail-session"
    photo_id = "photo-1"
    path = tmp_path / "photo.nef"
    path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {photo_id: path},
        "created": 0.0,
    }
    app_api._ENHANCE_THUMBNAIL_CACHE.clear()
    calls: list[Path] = []

    class FakeEvaluator:
        def extract_preview(self, image_path: Path):
            calls.append(Path(image_path))
            return np.full((480, 720, 3), 127, dtype=np.uint8)

    def fail_read_image(*args, **kwargs):
        raise AssertionError("thumbnail must not decode full RAW pixels")

    monkeypatch.setattr(app_api, "RawEvaluator", FakeEvaluator)
    monkeypatch.setattr(app_api, "read_image", fail_read_image)
    try:
        response = app_api.get_enhance_thumbnail(session_id, photo_id)
        assert response.status_code == 200
        assert response.media_type == "image/jpeg"
        with Image.open(BytesIO(response.body)) as image:
            assert image.format == "JPEG"
            assert image.size == (360, 240)
        assert calls == [path]

        cached_response = app_api.get_enhance_thumbnail(session_id, photo_id)
        assert cached_response.status_code == 200
        assert cached_response.body == response.body
        assert calls == [path]
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

    class FakeEvaluator:
        def extract_preview(self, *args, **kwargs):
            nonlocal read_calls
            read_calls += 1
            raise AssertionError("unknown photo IDs must not read local files")

    monkeypatch.setattr(app_api, "RawEvaluator", FakeEvaluator)
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

    def fake_correction(image, metadata, *, require_correction, backend="auto"):
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

    def fake_source_data(image, reference, source_path, source_exif):
        assert int(image[0, 0, 0]) == 2
        assert int(reference[0, 0, 0]) == 1
        return image, np.ones((2, 2), dtype=np.uint16), np.array([[0, 1], [1, 2]], dtype=np.uint8), {
            "DNGColorMatrix1": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            "DNGAsShotNeutral": (0.5, 1.0, 0.8),
            "DNGUniqueCameraModel": "Nikon Z6_3",
            "DNGBlackLevel": (100, 100, 100, 100),
            "DNGWhiteLevel": 16383,
        }, 1

    def fake_write(image, mosaic, pattern, source_path, output_dir, metadata, *, orientation, preview_rgb16):
        written_metadata.update(metadata)
        assert int(image[0, 0, 0]) == 2
        assert mosaic.shape == (2, 2)
        assert orientation == 1
        assert preview_rgb16.dtype == np.uint8
        assert preview_rgb16.shape == (2, 2, 3)
        return tmp_path / "output.dng"

    monkeypatch.setattr(
        app_api,
        "read_image",
        lambda *args, **kwargs: (np.ones((2, 2, 3), dtype=np.uint16), Metadata()),
    )
    def fake_apply_physical(image, _params, *, backend, spatial, nonlocal_mode, diagnostics=None):
        assert nonlocal_mode == "off"
        return image

    monkeypatch.setattr(app_api, "apply_physical_dehaze", fake_apply_physical)
    monkeypatch.setattr(app_api, "apply_lens_correction", fake_correction)
    monkeypatch.setattr(
        app_api, "camera_profile_names",
        lambda _path: {"SourceCameraProfileName": "Camera Flexible Color"},
    )
    monkeypatch.setattr(app_api, "enhanced_dng_source_data", fake_source_data)
    monkeypatch.setattr(app_api, "write_enhanced_dng", fake_write)
    try:
        app_api._run_enhance_job(job_id, session_id, tmp_path, DehazeParams(), {})
        item = app_api._ENHANCE_JOBS[job_id]["files"][0]
        assert require_values == [True]
        assert item["status"] == "success"
        assert item["warnings"] == []
        assert item["lens_correction"]["distortion"] is True
        assert item["lens_correction"]["tca"] is True
        assert written_metadata["LensCorrectionApplied"] is True
        assert written_metadata["LensCorrectionOperations"] == "distortion,tca"
        assert written_metadata["SourceCameraProfileName"] == "Camera Flexible Color"
        assert written_metadata["DNGUniqueCameraModel"] == "Nikon Z6_3"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


def _run_lens_fallback_export(
    monkeypatch, tmp_path: Path, failure: BaseException, *, suffix: str = ".nef",
    gain_map_error: BaseException | None = None,
):
    session_id = f"lens-fallback-{suffix[1:]}"
    job_id = f"lens-fallback-job-{suffix[1:]}"
    path = tmp_path / f"source{suffix}"
    path.write_bytes(b"fixture")

    class Metadata:
        source_kind = "raw"
        color_space = "Linear sRGB"
        bit_depth = 14
        exif = {
            "Make": "Nikon", "Model": "Z 8", "LensModel": "NIKKOR Z",
            "LensCorrectionApplied": True, "LensCorrectionEngine": "old engine",
            "LensCorrectionCamera": "old camera", "LensCorrectionLens": "old lens",
            "LensCorrectionOperations": "distortion,tca",
        }

    metadata = Metadata()
    source_image = np.full((2, 2, 3), 1000, dtype=np.uint16)
    source_identity = app_api.enhanced_dng_source_identity(path)
    cached_snapshot = SimpleNamespace()
    app_api._ENHANCE_SESSIONS[session_id] = {"files": {"photo-1": path}}
    item = {
        "photo_id": "photo-1", "name": path.name, "status": "waiting",
        "warnings": ["old warning"],
    }
    app_api._ENHANCE_JOBS[job_id] = {
        "job_id": job_id, "status": "queued", "total": 1,
        "processed": 0, "success": 0, "failed": 0, "progress": 0.0,
        "current_file": "", "files": [item], "_cancel": app_api.threading.Event(),
    }
    captured: dict = {}

    monkeypatch.setattr(app_api, "_capture_full_resolution_export_sources", lambda *_args: (
        source_image, metadata, cached_snapshot, True, source_identity,
    ))
    monkeypatch.setattr(app_api, "_valid_full_resolution_export_source_cache", lambda *_args: True)
    monkeypatch.setattr(app_api, "_prepare_dehaze_input", lambda image, *_a, **_k: image)

    def render(image, *_args, diagnostics, **_kwargs):
        diagnostics.update(auto_exposure_ev=0.0, auto_exposure_reason="off")
        return image.astype(np.float32) / 65535.0

    monkeypatch.setattr(app_api, "_render_dehaze", render)

    def fail_lens(_image, _metadata, *, require_correction, backend="auto"):
        assert require_correction is True
        raise failure

    monkeypatch.setattr(app_api, "apply_lens_correction", fail_lens)
    gain_map_calls: list[dict] = []

    def apply_gain_map(image, _path, **kwargs):
        gain_map_calls.append(kwargs)
        if gain_map_error is not None:
            raise gain_map_error
        return image.copy(), True

    monkeypatch.setattr(app_api, "apply_dng_gain_map", apply_gain_map)

    def source_data(processed, reference, _path, source_metadata, **kwargs):
        captured["source_metadata"] = dict(source_metadata)
        captured["source_data_kwargs"] = dict(kwargs)
        assert source_metadata.get("LensCorrectionUnavailable") is True
        for key in (
            "LensCorrectionApplied", "LensCorrectionEngine", "LensCorrectionCamera",
            "LensCorrectionLens", "LensCorrectionOperations",
        ):
            assert key not in source_metadata
        return (
            processed.copy(), np.ones((2, 2), dtype=np.uint16),
            np.array([[0, 1], [1, 2]], dtype=np.uint8),
            {"LensCorrectionApplied": True, "LensCorrectionEngine": "stale"}, 1,
        )

    monkeypatch.setattr(app_api, "enhanced_dng_source_data", source_data)
    monkeypatch.setattr(app_api, "camera_profile_names", lambda _path: {})
    monkeypatch.setattr(app_api, "matching_embedded_profile_dng", lambda *_args: None)

    def write_dng(*args, **_kwargs):
        captured["output_metadata"] = dict(args[5])
        return tmp_path / "output.dng"

    monkeypatch.setattr(app_api, "write_enhanced_dng", write_dng)
    try:
        app_api._run_enhance_job(job_id, session_id, tmp_path, DehazeParams(), {})
        return item, captured, gain_map_calls, app_api._ENHANCE_JOBS[job_id]
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


@pytest.mark.parametrize(
    "failure",
    [
        LensMatchError("/private/photos/secret.nef"),
        LensfunUnavailableError("lensfun unavailable"),
        LensCorrectionNotAppliedError("no geometry data"),
        LensMetadataUnavailableError("FocalLength 无效，无法初始化 Lensfun"),
    ],
)
def test_raw_export_degrades_on_missing_reliable_lens_data(
    monkeypatch, tmp_path: Path, failure: LensCorrectionError,
):
    item, captured, _gain_map_calls, job = _run_lens_fallback_export(
        monkeypatch, tmp_path, failure,
    )

    assert item["status"] == "success"
    assert item["lens_correction"]["applied"] is False
    assert item["warnings"] == [app_api._LENS_CORRECTION_FALLBACK_WARNING]
    assert "secret.nef" not in item["warnings"][0]
    assert item["source_cache"] == {"decoded": "hit", "camera": "miss"}
    assert captured["source_data_kwargs"] == {}
    assert captured["output_metadata"]["LensCorrectionUnavailable"] is True
    for key in (
        "LensCorrectionApplied", "LensCorrectionEngine", "LensCorrectionCamera",
        "LensCorrectionLens", "LensCorrectionOperations",
    ):
        assert key not in captured["output_metadata"]
    assert job["success"] == 1 and job["failed"] == 0


def test_unmatched_dng_still_applies_valid_gain_map_on_export(
    monkeypatch, tmp_path: Path,
):
    item, captured, gain_map_calls, _job = _run_lens_fallback_export(
        monkeypatch, tmp_path, LensMatchError("no reliable match"), suffix=".dng",
    )
    assert item["status"] == "success"
    assert gain_map_calls == [{"preserve_highlights": True}]
    assert captured["source_metadata"]["DNGGainMapApplied"] is True
    assert captured["source_metadata"]["DNGGainMapHighlightProtection"] is True


def test_generic_lens_correction_error_still_fails_export(monkeypatch, tmp_path: Path):
    item, _captured, _gain_map_calls, job = _run_lens_fallback_export(
        monkeypatch, tmp_path, LensCorrectionError("pixel transform failed"),
    )
    assert item["status"] == "failed"
    assert item["warnings"] == []
    assert job["success"] == 0 and job["failed"] == 1


def test_corrupt_gain_map_still_fails_unmatched_dng_export(monkeypatch, tmp_path: Path):
    item, _captured, gain_map_calls, job = _run_lens_fallback_export(
        monkeypatch, tmp_path, LensMatchError("no reliable match"),
        suffix=".dng", gain_map_error=DNGGainMapError("bad opcode list"),
    )
    assert item["status"] == "failed"
    assert item["warnings"] == []
    assert gain_map_calls == [{"preserve_highlights": True}]
    assert job["success"] == 0 and job["failed"] == 1


def test_session_exposes_xmp_settings_and_writes_through_photo_ids(tmp_path: Path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 48), (150, 170, 190)).save(path)

    session = app_api.create_enhance_session(app_api.EnhanceSessionRequest(paths=[str(path)]))
    session_id = session["session_id"]
    photo = session["files"][0]
    try:
        assert photo["dehaze_params"] is None
        assert photo["ricoh_preset_id"] is None
        assert photo["basic_params"]["exposure"] == 0
        result = app_api.save_enhance_session_xmp(app_api.EnhanceXmpRequest(
            session_id=session_id,
            params_by_photo={photo["photo_id"]: app_api.EnhanceParamsRequest(strength=0.27)},
            basic_params_by_photo={photo["photo_id"]: app_api.BasicParamsRequest(exposure=0.6, highlights=-15)},
        ))
        assert result["written"] == 1
        assert result["failed"] == 0

        applied = app_api.apply_ricoh_preset_session(app_api.RicohSessionRequest(
            session_id=session_id, preset_id="gr3_standard",
        ))
        assert applied["written"] == 1
        assert applied["updated"] == 1
        settings = app_api.read_photo_settings(path)
        assert settings["dehaze_params"]["strength"] == 0.27
        assert settings["ricoh_preset_id"] == "gr3_standard"
        assert settings["basic_params"]["exposure"] == 0.6
        assert settings["basic_params"]["highlights"] == -15
        root = ricoh_filter._parse_xmp(path.with_suffix(".xmp").read_bytes())
        description = ricoh_filter._description(root)
        assert description is not None
        assert description.attrib["{" + ricoh_filter._CRS_NS + "}Exposure2012"] == "0.6"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_ricoh_preview_uses_session_path_and_returns_approximate_jpeg(monkeypatch, tmp_path: Path):
    path = tmp_path / "preview.jpg"
    path.touch()
    session_id = "ricoh-preview-session"
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"photo-1": path}, "created": 0.0,
    }
    app_api._ENHANCE_PREVIEW_CACHE.clear()

    class Metadata:
        width = 8
        height = 8
        color_space = "sRGB"
        source_kind = "rgb"

    monkeypatch.setattr(
        app_api, "read_image",
        lambda *_args, **_kwargs: (np.full((8, 8, 3), [40, 120, 220], dtype=np.uint8), Metadata()),
    )
    monkeypatch.setattr(app_api, "_standard_rgb_to_srgb", lambda image, _path: image.copy())
    try:
        response = app_api.create_ricoh_preview(app_api.RicohPreviewRequest(
            session_id=session_id, photo_id="photo-1", preset_id="gr3_high_contrast_bw",
        ))
        assert response.status_code == 200
        assert response.media_type == "image/jpeg"
        assert response.headers["X-Preview-Approximation"] == "true"
        assert response.headers["X-Preview-Empirical-Color"] == "hsl-and-grading-response"
        with Image.open(BytesIO(response.body)) as preview:
            red, green, blue = preview.getpixel((4, 4))
            assert abs(red - green) <= 2
            assert abs(green - blue) <= 2
        brighter = app_api.create_ricoh_preview(app_api.RicohPreviewRequest(
            session_id=session_id, photo_id="photo-1", preset_id="gr3_high_contrast_bw",
            basic_params=app_api.BasicParamsRequest(exposure=1),
        ))
        assert brighter.status_code == 200
        assert brighter.body != response.body
        assert len(app_api._ENHANCE_PREVIEW_CACHE) == 2
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_PREVIEW_CACHE.clear()


def test_ricoh_preview_enables_preset_measured_color(monkeypatch, tmp_path: Path):
    path = tmp_path / "preset.jpg"
    path.touch()
    session_id = "ricoh-measured-preset-session"
    app_api._ENHANCE_SESSIONS[session_id] = {"files": {"photo-1": path}, "created": 0.0}
    app_api._ENHANCE_PREVIEW_CACHE.clear()
    seen: list[tuple[str, bool]] = []

    class Metadata:
        width = 4
        height = 4
        color_space = "sRGB"
        source_kind = "rgb"

    monkeypatch.setattr(app_api, "read_image", lambda *_args, **_kwargs: (
        np.full((4, 4, 3), [170, 90, 60], dtype=np.uint8), Metadata(),
    ))

    def fake_effect(image, preset_id, _basic, *, use_measured_color=False):
        seen.append((preset_id, use_measured_color))
        return image

    monkeypatch.setattr(app_api, "apply_ricoh_preview_effect", fake_effect)
    monkeypatch.setattr(app_api, "_standard_rgb_to_srgb", lambda image, _path: image.copy())
    try:
        response = app_api.create_ricoh_preview(app_api.RicohPreviewRequest(
            session_id=session_id, photo_id="photo-1", preset_id="gr3_positive_film",
        ))
        assert response.status_code == 200
        assert seen == [("gr3_positive_film", True)]
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_PREVIEW_CACHE.clear()


def test_ricoh_preset_preview_renders_effect_without_manual_hsl_controls(tmp_path: Path):
    path = tmp_path / "color-chart.png"
    chart = np.empty((32, 48, 3), dtype=np.uint8)
    chart[:, :16] = [200, 75, 65]
    chart[:, 16:32] = [45, 140, 190]
    chart[:, 32:] = [120, 165, 60]
    Image.fromarray(chart, "RGB").save(path)
    session_id = "ricoh-preset-effect-session"
    app_api._ENHANCE_SESSIONS[session_id] = {"files": {"photo-1": path}, "created": 0.0}
    app_api._RICOH_PREVIEW_CACHE.clear()
    try:
        original = app_api.create_enhance_preview(app_api.EnhancePreviewRequest(
            session_id=session_id, photo_id="photo-1", max_edge=320,
            mode="original", color_manage_srgb=True,
        ))
        effect = app_api.create_ricoh_preview(app_api.RicohPreviewRequest(
            session_id=session_id, photo_id="photo-1", max_edge=320,
            preset_id="gr3_positive_film",
        ))
        assert original.status_code == effect.status_code == 200
        assert effect.headers["X-Preview-Empirical-Color"] == "hsl-and-grading-response"
        assert effect.body != original.body
        with Image.open(BytesIO(effect.body)) as preview:
            assert preview.size == (48, 32)
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._RICOH_PREVIEW_CACHE.clear()


def test_ricoh_dng_job_exports_linear_full_resolution_and_xmp(monkeypatch, tmp_path: Path):
    session_id = "ricoh-run-session"
    job_id = "ricoh-run-job"
    source = tmp_path / "source.jpg"
    output = tmp_path / "source_ricoh.dng"
    source.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"photo-1": source}, "created": 0.0,
    }
    app_api._RICOH_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "total": 1,
        "processed": 0,
        "success": 0,
        "failed": 0,
        "progress": 0.0,
        "current_file": "",
        "files": [{"photo_id": "photo-1", "name": source.name, "status": "waiting"}],
        "_cancel": app_api.threading.Event(),
    }
    written: dict[str, object] = {}

    class Metadata:
        color_space = "sRGB"
        exif = {"Make": "Ricoh", "Model": "GR III"}

    def fake_write(image, source_path, output_dir, metadata, *, bits_per_sample, name_suffix):
        written["image"] = image.copy()
        written["metadata"] = metadata
        written["bits_per_sample"] = bits_per_sample
        written["name_suffix"] = name_suffix
        output.touch()
        return output

    monkeypatch.setattr(
        app_api, "read_image",
        lambda *_args, **_kwargs: (np.full((3, 4, 3), [100, 150, 200], dtype=np.uint8), Metadata()),
    )
    monkeypatch.setattr(app_api, "write_linear_dng", fake_write)
    try:
        app_api._run_ricoh_job(job_id, session_id, "gr3_standard", tmp_path)
        job = app_api._RICOH_JOBS[job_id]
        assert job["status"] == "completed"
        assert job["success"] == 1
        assert job["files"][0]["status"] == "success"
        assert job["files"][0]["xmp_status"] == {"source": "written", "output": "embedded"}
        assert written["metadata"]["ACRBasicParams"] == app_api.BasicParamsRequest().values()
        assert written["name_suffix"] == "_ricoh"
        assert written["bits_per_sample"] == 16
        assert written["image"].dtype == np.uint16
        assert app_api.read_photo_settings(source)["ricoh_preset_id"] == "gr3_standard"
        assert not output.with_suffix(".xmp").exists()
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._RICOH_JOBS.pop(job_id, None)
