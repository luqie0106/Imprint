from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import app_api
from dehaze import DehazeParams


def _job(job_id: str, photo_ids: list[str]) -> dict:
    return {
        "job_id": job_id,
        "status": "queued",
        "total": len(photo_ids),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "progress": 0.0,
        "current_file": "",
        "files": [
            {"photo_id": photo_id, "name": f"{photo_id}.jpg", "status": "waiting"}
            for photo_id in photo_ids
        ],
        "_cancel": threading.Event(),
    }


def test_unified_export_request_defaults_and_depth_resolution():
    request = app_api.EnhanceRunRequest(session_id="session")

    assert request.photo_ids is None
    assert request.ricoh_backend == "python"
    assert request.compression == "lossless_jpeg"
    assert request.bit_depth == "source"
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={"BitsPerSample": (12, 12, 12)}, bit_depth=14),
        "source",
    ) == (14, 14, "sensor_white_level")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={}, bit_depth=14), "source",
    ) == (14, 14, "sensor_white_level")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={}, bit_depth=None), "source",
    ) == (16, 16, "fallback_16")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="rgb", exif={}, bit_depth=8), "source",
    ) == (16, None, "rgb_16")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={"BitsPerSample": 12}, bit_depth=14), "16",
    ) == (16, 14, "sensor_white_level")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={"BitsPerSample": 12, "DNGWhiteLevel": 16384}, bit_depth=14),
        "source",
    ) == (14, 14, "sensor_white_level")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={"DNGWhiteLevel": 4096}, bit_depth=14),
        "source",
    ) == (14, 14, "sensor_white_level")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={"DNGWhiteLevel": 16384}, bit_depth=12),
        "source",
    ) == (12, 12, "sensor_white_level")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={"BitsPerSample": 8}, bit_depth=14),
        "source",
    ) == (14, 14, "sensor_white_level")
    assert app_api._resolve_output_bits_per_sample(12, "lossless_jpeg") == 12
    assert app_api._resolve_output_bits_per_sample(12, "none") == 12
    assert app_api._resolve_output_bits_per_sample(12, "jpegxl") == 16


def test_run_enhance_rejects_empty_unknown_selection_and_preset(monkeypatch, tmp_path: Path):
    session_id = "unified-export-validation"
    paths = {"photo-1": tmp_path / "one.jpg"}
    paths["photo-1"].touch()
    app_api._ENHANCE_SESSIONS[session_id] = {"files": paths, "created": 0.0}
    monkeypatch.setattr(app_api, "list_ricoh_presets", lambda: [{"id": "valid"}])
    try:
        empty = app_api.run_enhance(app_api.EnhanceRunRequest(
            session_id=session_id, photo_ids=[],
        ))
        unknown_photo = app_api.run_enhance(app_api.EnhanceRunRequest(
            session_id=session_id, photo_ids=["missing"],
        ))
        unknown_preset = app_api.run_enhance(app_api.EnhanceRunRequest(
            session_id=session_id, preset_ids_by_photo={"photo-1": "missing"},
        ))

        assert empty.status_code == 400
        assert unknown_photo.status_code == 400
        assert unknown_preset.status_code == 400
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_run_enhance_freezes_selected_subset_and_export_options(monkeypatch, tmp_path: Path):
    session_id = "unified-export-subset"
    paths = {f"photo-{index}": tmp_path / f"{index}.jpg" for index in range(1, 4)}
    for path in paths.values():
        path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {"files": paths, "created": 0.0}
    monkeypatch.setattr(app_api, "list_ricoh_presets", lambda: [{"id": "valid"}])

    created_threads = []

    class CaptureThread:
        def __init__(self, *, target, args, name, daemon):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            created_threads.append(self)

        def start(self):
            pass

    monkeypatch.setattr(app_api.threading, "Thread", CaptureThread)
    try:
        result = app_api.run_enhance(app_api.EnhanceRunRequest(
            session_id=session_id,
            photo_ids=["photo-2"],
            output_dir=str(tmp_path / "out"),
            params_by_photo={"photo-2": app_api.EnhanceParamsRequest(strength=0.4)},
            basic_params_by_photo={"photo-2": app_api.BasicParamsRequest(exposure=0.5)},
            preset_ids_by_photo={
                "photo-1": "valid", "photo-2": None, "photo-3": "valid",
            },
            ricoh_backend="native",
            compression="lossless_jpeg",
            bit_depth="16",
        ))
        job_id = result["job_id"]
        thread = created_threads[0]
        preset_snapshot = thread.args[14]

        assert result["total"] == 1
        assert result["photo_ids"] == ["photo-2"]
        assert [item["photo_id"] for item in result["files"]] == ["photo-2"]
        assert preset_snapshot == {"photo-2": None}
        assert thread.args[15:] == ("native", "lossless_jpeg", "16")
        with pytest.raises(TypeError):
            preset_snapshot["photo-2"] = "valid"
        assert app_api._ENHANCE_JOBS[job_id]["status"] == "queued"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        for job_id in list(app_api._ENHANCE_JOBS):
            if app_api._ENHANCE_JOBS[job_id].get("photo_ids") == ["photo-2"]:
                app_api._ENHANCE_JOBS.pop(job_id, None)


def test_enhance_job_applies_per_photo_ricoh_or_basic_once_and_passes_writer_options(
    monkeypatch, tmp_path: Path,
):
    session_id = "unified-export-pipeline"
    job_id = "unified-export-pipeline-job"
    paths = {
        "with-preset": tmp_path / "preset.jpg",
        "without-preset": tmp_path / "basic.jpg",
        "not-selected": tmp_path / "ignored.jpg",
    }
    for path in paths.values():
        path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {"files": paths, "created": 0.0}
    app_api._ENHANCE_JOBS[job_id] = _job(job_id, ["with-preset", "without-preset"])
    metadata = SimpleNamespace(
        color_space="Linear sRGB", source_kind="rgb", bit_depth=8, exif={},
    )
    reads: list[str] = []
    ricoh_calls: list[tuple[str, dict[str, float] | None, str, bool]] = []
    basic_calls: list[dict[str, float]] = []
    writer_calls: list[dict] = []
    basic = app_api.BasicParamsRequest(exposure=0.5).values()
    zero_basic = app_api.BasicParamsRequest().values()
    preset_id = "preset-a"

    def read_image(path, *, preview=False):
        reads.append(Path(path).name)
        value = 10000 if Path(path).name == "preset.jpg" else 20000
        return np.full((2, 2, 3), value, dtype=np.uint16), metadata

    def render_ricoh(image, selected_preset, selected_basic, backend, *, use_measured_color=False):
        ricoh_calls.append((selected_preset, selected_basic, backend, use_measured_color))
        return image

    def render_basic(image, selected_basic, backend):
        basic_calls.append(selected_basic)
        return image

    def write_linear(image, source, output_dir, output_metadata, **kwargs):
        writer_calls.append((image.copy(), output_metadata, kwargs))
        return output_dir / f"{Path(source).stem}_dehaze.dng"

    monkeypatch.setattr(app_api, "read_image", read_image)
    monkeypatch.setattr(app_api, "_render_dehaze", lambda image, *_a, **_kw: image)
    monkeypatch.setattr(app_api, "_correct_enhanced_raw", lambda image, *_a, **_kw: (
        image,
        app_api.LensCorrectionResult(False, None, None, False, False, False),
        False,
    ))
    monkeypatch.setattr(app_api, "_render_ricoh", render_ricoh)
    monkeypatch.setattr(app_api, "_render_basic", render_basic)
    monkeypatch.setattr(app_api, "write_linear_dng", write_linear)

    try:
        app_api._run_enhance_job(
            job_id,
            session_id,
            tmp_path,
            DehazeParams(),
            {},
            basic_params_by_photo={"with-preset": basic, "without-preset": basic},
            preset_ids_by_photo=MappingProxyType({"with-preset": preset_id}),
            ricoh_backend="native",
            compression="lossless_jpeg",
            requested_bit_depth="source",
        )

        assert reads == ["preset.jpg", "basic.jpg"]
        assert ricoh_calls == [(preset_id, zero_basic, "native", True)]
        assert basic_calls == []
        assert [call[2] for call in writer_calls] == [
            {"bits_per_sample": 16, "compression": "lossless_jpeg"},
            {"bits_per_sample": 16, "compression": "lossless_jpeg"},
        ]
        assert [call[1]["ACRBasicParams"] for call in writer_calls] == [basic, basic]
        assert all(call[0].shape == (2, 2, 3) for call in writer_calls)
        job = app_api._ENHANCE_JOBS[job_id]
        assert job["status"] == "completed"
        assert job["success"] == 2
        assert [item["xmp_status"] for item in job["files"]] == ["unchanged", "unchanged"]
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


@pytest.mark.parametrize("compression", ["none", "lossless_jpeg", "jpegxl"])
def test_raw_export_separates_sensor_precision_from_enhanced_storage(
    monkeypatch, tmp_path: Path, compression,
):
    session_id = "unified-export-raw-depth"
    job_id = "unified-export-raw-depth-job"
    source = tmp_path / "camera.nef"
    source.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"raw-photo": source}, "created": 0.0,
    }
    app_api._ENHANCE_JOBS[job_id] = _job(job_id, ["raw-photo"])
    metadata = SimpleNamespace(
        color_space="Linear sRGB",
        source_kind="raw",
        bit_depth=14,
        exif={
            "BitsPerSample": (12, 12, 12),
            "XMPPacket": "source-side packet",
            "DehazeCompatibilityCurves": "source-side compatibility curve",
        },
    )
    captured: dict = {}
    render_basic_calls = []
    basic = app_api.BasicParamsRequest(exposure=1.5, highlights=-20).values()

    monkeypatch.setattr(app_api, "read_image", lambda *_a, **_k: (
        np.full((2, 2, 3), 12000, dtype=np.uint16), metadata,
    ))
    monkeypatch.setattr(app_api, "_render_dehaze", lambda image, *_a, **_kw: image)
    monkeypatch.setattr(app_api, "_correct_enhanced_raw", lambda image, *_a, **_kw: (
        image,
        app_api.LensCorrectionResult(False, None, None, False, False, False),
        False,
    ))
    monkeypatch.setattr(app_api, "camera_profile_names", lambda _path: {})
    monkeypatch.setattr(app_api, "matching_embedded_profile_dng", lambda *_a: None)
    monkeypatch.setattr(app_api, "enhanced_dng_source_data", lambda image, *_a: (
        captured.update(camera_rgb=image.copy()) or image,
        np.ones((2, 2), dtype=np.uint16),
        np.array([[0, 1], [1, 2]], dtype=np.uint8),
        {"DNGWhiteLevel": 16383},
        1,
    ))

    def write_enhanced(*_args, **kwargs):
        captured.update(kwargs)
        captured["output_metadata"] = _args[5]
        return tmp_path / "camera_dehaze.dng"

    def render_basic(image, params, backend):
        render_basic_calls.append((image.copy(), params, backend))
        return image + 1

    monkeypatch.setattr(app_api, "write_enhanced_dng", write_enhanced)
    monkeypatch.setattr(app_api, "_render_basic", render_basic)
    try:
        app_api._run_enhance_job(
            job_id,
            session_id,
            tmp_path,
            DehazeParams(),
            {},
            basic_params_by_photo={"raw-photo": basic},
            preset_ids_by_photo=MappingProxyType({}),
            compression=compression,
            requested_bit_depth="source",
        )

        assert captured["orientation"] == 1
        assert captured["preview_rgb16"].dtype == np.uint8
        assert captured["preview_rgb16"].shape == (2, 2, 3)
        assert len(render_basic_calls) == 1
        assert render_basic_calls[0][1] == basic
        assert np.array_equal(
            captured["preview_rgb16"], render_basic_calls[0][0] + 1,
        )
        assert np.array_equal(captured["camera_rgb"], np.full((2, 2, 3), 12000, dtype=np.uint16))
        assert captured["output_metadata"]["ACRBasicParams"] == basic
        assert "XMPPacket" not in captured["output_metadata"]
        assert "DehazeCompatibilityCurves" not in captured["output_metadata"]
        assert captured["bits_per_sample"] == 14
        assert captured["raw_bits_per_sample"] == 14
        assert captured["compression"] == compression
        item = app_api._ENHANCE_JOBS[job_id]["files"][0]
        assert item["status"] == "success"
        assert item["output_bits_per_sample"] == 16
        assert item["sample_precision_bits"] == 14
        assert item["bit_depth_source"] == "sensor_white_level"
        assert item["raw_bits_per_sample"] == 14
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


def test_basic_exposure_is_embedded_without_changing_unified_export_pixels(
    monkeypatch, tmp_path: Path,
):
    session_id = "unified-export-deferred-basic"
    job_id = "unified-export-deferred-basic-job"
    photo_ids = ["minus-two", "zero", "plus-two"]
    paths = {photo_id: tmp_path / f"{photo_id}.jpg" for photo_id in photo_ids}
    for path in paths.values():
        path.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": paths, "created": 0.0,
    }
    app_api._ENHANCE_JOBS[job_id] = _job(job_id, photo_ids)
    source = np.array(
        [[[65530, 64000, 62000], [9000, 12000, 15000]],
         [[16000, 18000, 20000], [30000, 28000, 26000]]],
        dtype=np.uint16,
    )
    metadata = SimpleNamespace(
        color_space="Linear sRGB", source_kind="rgb", bit_depth=16, exif={},
    )
    outputs: dict[str, tuple[np.ndarray, dict]] = {}

    monkeypatch.setattr(app_api, "read_image", lambda *_a, **_k: (source.copy(), metadata))
    monkeypatch.setattr(app_api, "_render_dehaze", lambda image, *_a, **_k: image)
    monkeypatch.setattr(app_api, "_correct_enhanced_raw", lambda image, *_a, **_k: (
        image,
        app_api.LensCorrectionResult(False, None, None, False, False, False),
        False,
    ))

    def write_linear(image, source_path, _output_dir, output_metadata, **_kwargs):
        outputs[Path(source_path).stem] = (image.copy(), output_metadata)
        return tmp_path / f"{Path(source_path).stem}_dehaze.dng"

    monkeypatch.setattr(app_api, "write_linear_dng", write_linear)
    basic_by_photo = {
        photo_id: app_api.BasicParamsRequest(exposure=exposure).values()
        for photo_id, exposure in zip(photo_ids, (-2.0, 0.0, 2.0))
    }

    try:
        app_api._run_enhance_job(
            job_id, session_id, tmp_path, DehazeParams(strength=0.0), {},
            basic_params_by_photo=basic_by_photo,
            preset_ids_by_photo=MappingProxyType({photo_id: None for photo_id in photo_ids}),
        )

        exported_pixels = [outputs[photo_id][0] for photo_id in photo_ids]
        assert all(np.array_equal(exported_pixels[0], pixels) for pixels in exported_pixels[1:])
        assert np.array_equal(exported_pixels[0], source)
        metadata_by_photo = {photo_id: outputs[photo_id][1] for photo_id in photo_ids}
        assert [metadata_by_photo[photo_id]["ACRBasicParams"]["exposure"] for photo_id in photo_ids] == [
            -2.0, 0.0, 2.0,
        ]
        assert all(
            set(metadata["ACRBasicParams"]) == {
                "exposure", "contrast", "highlights", "shadows",
                "whites", "blacks", "vibrance", "saturation",
            }
            for metadata in metadata_by_photo.values()
        )
        assert np.max(exported_pixels[0]) == 65530
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


def test_legacy_ricoh_export_embeds_basic_params_without_baking_them(
    monkeypatch, tmp_path: Path,
):
    session_id = "legacy-ricoh-deferred-basic"
    job_id = "legacy-ricoh-deferred-basic-job"
    path = tmp_path / "legacy.jpg"
    path.touch()
    source = np.array(
        [[[65530, 64000, 62000], [9000, 12000, 15000]]], dtype=np.uint16,
    )
    metadata = SimpleNamespace(
        color_space="Linear sRGB", source_kind="rgb", bit_depth=16, exif={},
    )
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"legacy-photo": path}, "created": 0.0,
    }
    app_api._RICOH_JOBS[job_id] = _job(job_id, ["legacy-photo"])
    captured: dict = {}
    basic = app_api.BasicParamsRequest(exposure=2.0, highlights=-40).values()

    monkeypatch.setattr(app_api, "read_image", lambda *_a, **_k: (source.copy(), metadata))
    monkeypatch.setattr(app_api, "_render_basic", lambda *_a, **_k: pytest.fail(
        "legacy basic settings must be embedded, not baked into image pixels",
    ))

    # Capture the positional writer metadata separately from its options.
    def write_linear(image, _source, _output, output_metadata, **kwargs):
        captured.update(pixels=image.copy(), metadata=output_metadata, options=kwargs)
        return tmp_path / "legacy_ricoh.dng"

    monkeypatch.setattr(app_api, "write_linear_dng", write_linear)
    try:
        app_api._run_ricoh_job(
            job_id, session_id, None, tmp_path,
            basic_params_by_photo={"legacy-photo": basic},
            preset_ids_by_photo=MappingProxyType({"legacy-photo": None}),
        )

        assert np.array_equal(captured["pixels"], source)
        assert captured["metadata"]["ACRBasicParams"] == basic
        assert app_api._RICOH_JOBS[job_id]["files"][0]["xmp_status"]["output"] == "embedded"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._RICOH_JOBS.pop(job_id, None)


def test_enhance_job_preserves_cancelled_and_failed_states(monkeypatch, tmp_path: Path):
    session_id = "unified-export-cancel"
    cancel_job_id = "unified-export-cancel-job"
    error_job_id = "unified-export-error-job"
    source = tmp_path / "photo.jpg"
    source.touch()
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {"photo": source}, "created": 0.0,
    }
    app_api._ENHANCE_JOBS[cancel_job_id] = _job(cancel_job_id, ["photo"])
    app_api._ENHANCE_JOBS[cancel_job_id]["_cancel"].set()
    app_api._ENHANCE_JOBS[error_job_id] = _job(error_job_id, ["photo"])
    monkeypatch.setattr(app_api, "read_image", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("decode error")))
    try:
        app_api._run_enhance_job(cancel_job_id, session_id, tmp_path, DehazeParams(), {})
        app_api._run_enhance_job(error_job_id, session_id, tmp_path, DehazeParams(), {})

        assert app_api._ENHANCE_JOBS[cancel_job_id]["status"] == "cancelled"
        assert app_api._ENHANCE_JOBS[cancel_job_id]["processed"] == 0
        assert app_api._ENHANCE_JOBS[error_job_id]["status"] == "completed"
        assert app_api._ENHANCE_JOBS[error_job_id]["failed"] == 1
        assert app_api._ENHANCE_JOBS[error_job_id]["files"][0]["status"] == "failed"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(cancel_job_id, None)
        app_api._ENHANCE_JOBS.pop(error_job_id, None)
