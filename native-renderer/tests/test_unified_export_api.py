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
    ) == (12, 12, "exif")
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
    ) == (16, 12, "exif")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={"BitsPerSample": 12, "DNGWhiteLevel": 16384}, bit_depth=14),
        "source",
    ) == (12, 12, "exif")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={"DNGWhiteLevel": 4096}, bit_depth=14),
        "source",
    ) == (12, 12, "dng_white_level")
    assert app_api._resolve_export_bit_depth(
        SimpleNamespace(source_kind="raw", exif={"DNGWhiteLevel": 16384}, bit_depth=12),
        "source",
    ) == (14, 14, "dng_white_level")
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

    def write_linear(image, source, output_dir, _metadata, **kwargs):
        writer_calls.append(kwargs)
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
        assert ricoh_calls == [(preset_id, basic, "native", True)]
        assert basic_calls == [basic]
        assert writer_calls == [
            {"bits_per_sample": 16, "compression": "lossless_jpeg"},
            {"bits_per_sample": 16, "compression": "lossless_jpeg"},
        ]
        job = app_api._ENHANCE_JOBS[job_id]
        assert job["status"] == "completed"
        assert job["success"] == 2
        assert [item["xmp_status"] for item in job["files"]] == ["unchanged", "unchanged"]
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


def test_raw_export_forwards_exif_depth_and_compression_to_enhanced_writer(
    monkeypatch, tmp_path: Path,
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
        exif={"BitsPerSample": (12, 12, 12)},
    )
    captured: dict = {}

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
        image,
        np.ones((2, 2), dtype=np.uint16),
        np.array([[0, 1], [1, 2]], dtype=np.uint8),
        {"DNGWhiteLevel": 16383},
        1,
    ))

    def write_enhanced(*_args, **kwargs):
        captured.update(kwargs)
        return tmp_path / "camera_dehaze.dng"

    monkeypatch.setattr(app_api, "write_enhanced_dng", write_enhanced)
    try:
        app_api._run_enhance_job(
            job_id,
            session_id,
            tmp_path,
            DehazeParams(),
            {},
            preset_ids_by_photo=MappingProxyType({}),
            compression="jpegxl",
            requested_bit_depth="source",
        )

        assert captured["orientation"] == 1
        assert captured["preview_rgb16"].dtype == np.uint8
        assert captured["preview_rgb16"].shape == (2, 2, 3)
        assert captured["bits_per_sample"] == 16
        assert captured["raw_bits_per_sample"] == 12
        assert captured["compression"] == "jpegxl"
        item = app_api._ENHANCE_JOBS[job_id]["files"][0]
        assert item["status"] == "success"
        assert item["output_bits_per_sample"] == 16
        assert item["bit_depth_source"] == "exif"
        assert item["raw_bits_per_sample"] == 12
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._ENHANCE_JOBS.pop(job_id, None)


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
