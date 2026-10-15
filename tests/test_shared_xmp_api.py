from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app_api
import ricoh_filter


def _snapshot_request(session_id: str, photo_id: str, *, preset_id: str | None = None,
                      strength: float = 0.23, exposure: float = 0.5):
    return app_api.PhotoSettingsRequest(
        session_id=session_id,
        photo_id=photo_id,
        dehaze_params=app_api.EnhanceParamsRequest(strength=strength),
        basic_params=app_api.BasicParamsRequest(exposure=exposure),
        ricoh_preset_id=preset_id,
    )


def _session(session_id: str, **photos: Path) -> None:
    app_api._ENHANCE_SESSIONS[session_id] = {"files": photos, "created": 0.0}


def _write_valid_photo(photo: Path) -> bytes:
    ramp = np.linspace(0, 255, 64).astype(np.uint8)
    rgb = np.broadcast_to(ramp[None, :, None], (48, 64, 3)).copy()
    Image.fromarray(rgb).save(photo)
    return photo.read_bytes()


def test_photo_settings_api_creates_and_updates_xmp_without_touching_photo(tmp_path: Path):
    photo = tmp_path / "sample.jpg"
    original = _write_valid_photo(photo)
    session_id = "shared-settings-session"
    _session(session_id, **{"photo-1": photo})
    try:
        created = app_api.save_photo_settings_snapshot(_snapshot_request(
            session_id, "photo-1", preset_id="gr3_standard",
        ))
        assert created == {
            "photo_id": "photo-1", "status": "saved",
            "sidecar_status": "written", "name": "sample.xmp",
            "dehaze_compatibility": "approximate",
        }
        assert photo.read_bytes() == original
        sidecar = photo.with_suffix(".xmp")
        first_payload = sidecar.read_bytes()

        updated = app_api.save_photo_settings_snapshot(_snapshot_request(
            session_id, "photo-1", preset_id=None, strength=0.71, exposure=-0.25,
        ))
        assert updated["sidecar_status"] == "updated"
        assert sidecar.read_bytes() != first_payload
        settings = ricoh_filter.read_photo_settings(photo)
        assert settings["dehaze_params"]["strength"] == 0.71
        assert settings["ricoh_preset_id"] == "gr3_standard"
        assert settings["basic_params"]["exposure"] == -0.25
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_photo_settings_snapshot_merges_both_pages_and_commits_once(tmp_path: Path, monkeypatch):
    photo = tmp_path / "shared.jpg"
    _write_valid_photo(photo)
    sidecar = photo.with_suffix(".xmp")
    sidecar.write_text(
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="{ricoh_filter._RDF_NS}" '
        'xmlns:vendor="https://example.test/ns/">'
        '<rdf:RDF><rdf:Description rdf:about="" vendor:KeepMe="untouched"/>'
        '</rdf:RDF></x:xmpmeta>',
        encoding="utf-8",
    )
    session_id = "two-pages-session"
    _session(session_id, **{"photo-1": photo})
    commits: list[Path] = []
    atomic_write = ricoh_filter._atomic_write_sidecar

    def count_commit(path: Path, payload: bytes, **kwargs):
        commits.append(path)
        return atomic_write(path, payload, **kwargs)

    monkeypatch.setattr(ricoh_filter, "_atomic_write_sidecar", count_commit)
    try:
        result = app_api.save_photo_settings_snapshot(_snapshot_request(
            session_id, "photo-1", preset_id="gr3_positive_film",
            strength=0.62, exposure=0.35,
        ))
        assert result["sidecar_status"] == "updated"
        assert commits == [photo]
        settings = ricoh_filter.read_photo_settings(photo)
        assert settings["dehaze_params"]["strength"] == 0.62
        assert settings["ricoh_preset_id"] == "gr3_positive_film"
        assert settings["basic_params"]["exposure"] == 0.35
        root = ricoh_filter._parse_xmp(sidecar.read_bytes())
        description = ricoh_filter._description(root)
        assert description is not None
        assert description.attrib["{https://example.test/ns/}KeepMe"] == "untouched"
        assert description.attrib["{" + ricoh_filter._CRS_NS + "}Exposure2012"] == "0.35"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_photo_settings_api_rejects_invalid_session_and_photo_ids(tmp_path: Path):
    photo = tmp_path / "sample.jpg"
    photo.touch()
    session_id = "scoped-session"
    _session(session_id, **{"photo-1": photo})
    try:
        stale_session = app_api.save_photo_settings_snapshot(_snapshot_request(
            "not-the-session", "photo-1",
        ))
        unknown_photo = app_api.save_photo_settings_snapshot(_snapshot_request(
            session_id, "../sample.jpg",
        ))
        assert stale_session.status_code == 404
        assert unknown_photo.status_code == 404
        assert not photo.with_suffix(".xmp").exists()
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_photo_settings_api_does_not_accept_client_paths(tmp_path: Path):
    payload = {
        "session_id": "session",
        "photo_id": "photo-1",
        "dehaze_params": app_api.EnhanceParamsRequest().model_dump(),
        "basic_params": app_api.BasicParamsRequest().model_dump(),
        "ricoh_preset_id": None,
        "path": str(tmp_path / "outside.jpg"),
    }
    with pytest.raises(ValidationError):
        app_api.PhotoSettingsRequest(**payload)
    assert "/api/photo/settings" in app_api.app.openapi()["paths"]


def test_failed_atomic_replace_keeps_old_xmp_and_cleans_temporary_file(tmp_path: Path, monkeypatch):
    photo = tmp_path / "safe.jpg"
    _write_valid_photo(photo)
    sidecar = photo.with_suffix(".xmp")
    original_xmp = b"<?xpacket begin=\"\xef\xbb\xbf\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\" " \
        b"xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:RDF><rdf:Description " \
        b"xmlns:vendor=\"https://example.test/ns/\" vendor:KeepMe=\"old\"/></rdf:RDF></x:xmpmeta>"
    sidecar.write_bytes(original_xmp)
    session_id = "atomic-failure-session"
    _session(session_id, **{"photo-1": photo})

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(ricoh_filter.os, "replace", fail_replace)
    try:
        response = app_api.save_photo_settings_snapshot(_snapshot_request(
            session_id, "photo-1", preset_id="gr3_standard",
        ))
        assert response.status_code == 500
        assert sidecar.read_bytes() == original_xmp
        assert list(tmp_path.glob(".imprint-xmp-*.tmp")) == []
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_ricoh_apply_session_uses_per_photo_presets_with_global_fallback(tmp_path: Path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.touch()
    second.touch()
    session_id = "per-photo-preset-session"
    _session(session_id, **{"photo-1": first, "photo-2": second})
    try:
        result = app_api.apply_ricoh_preset_session(app_api.RicohSessionRequest(
            session_id=session_id,
            preset_id="gr3_standard",
            preset_ids_by_photo={"photo-1": "gr3_positive_film"},
        ))
        assert result["written"] == 2
        assert ricoh_filter.read_photo_settings(first)["ricoh_preset_id"] == "gr3_positive_film"
        assert ricoh_filter.read_photo_settings(second)["ricoh_preset_id"] == "gr3_standard"
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_enhance_xmp_saves_dehaze_and_per_photo_presets(tmp_path: Path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_valid_photo(first)
    _write_valid_photo(second)
    session_id = "shared-enhance-xmp-session"
    _session(session_id, **{"photo-1": first, "photo-2": second})
    try:
        result = app_api.save_enhance_session_xmp(app_api.EnhanceXmpRequest(
            session_id=session_id,
            params_by_photo={
                "photo-1": app_api.EnhanceParamsRequest(strength=0.11),
                "photo-2": app_api.EnhanceParamsRequest(strength=0.22),
            },
            basic_params_by_photo={
                "photo-1": app_api.BasicParamsRequest(exposure=0.2),
                "photo-2": app_api.BasicParamsRequest(exposure=-0.3),
            },
            preset_ids_by_photo={
                "photo-1": "gr3_standard",
                "photo-2": "gr3_positive_film",
            },
        ))
        assert result["written"] == 2
        first_settings = ricoh_filter.read_photo_settings(first)
        second_settings = ricoh_filter.read_photo_settings(second)
        assert first_settings["dehaze_params"]["strength"] == 0.11
        assert first_settings["ricoh_preset_id"] == "gr3_standard"
        assert first_settings["basic_params"]["exposure"] == 0.2
        assert second_settings["dehaze_params"]["strength"] == 0.22
        assert second_settings["ricoh_preset_id"] == "gr3_positive_film"
        assert second_settings["basic_params"]["exposure"] == -0.3
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)


def test_ricoh_export_worker_uses_per_photo_presets(tmp_path: Path, monkeypatch):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.touch()
    second.touch()
    session_id = "per-photo-export-session"
    job_id = "per-photo-export-job"
    _session(session_id, **{"photo-1": first, "photo-2": second})
    app_api._RICOH_JOBS[job_id] = {
        "status": "queued", "files": [
            {"photo_id": "photo-1", "status": "waiting"},
            {"photo_id": "photo-2", "status": "waiting"},
        ], "_cancel": app_api.threading.Event(), "success": 0, "failed": 0,
        "processed": 0, "total": 2,
    }
    selected: list[str] = []

    class Metadata:
        color_space = "sRGB"
        exif = {}

    monkeypatch.setattr(app_api, "read_image", lambda *_a, **_k: (
        np.zeros((2, 2, 3), dtype=np.uint8), Metadata(),
    ))
    monkeypatch.setattr(app_api, "apply_ricoh_preview_effect", lambda image, preset, _basic: (
        selected.append(preset) or image
    ))

    def fake_write(_image, path, output_dir, _metadata, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / (Path(path).stem + "_ricoh.dng")
        output.touch()
        return output

    monkeypatch.setattr(app_api, "write_linear_dng", fake_write)
    monkeypatch.setattr(app_api, "write_ricoh_preset", lambda _path, preset, _basic: preset)
    try:
        app_api._run_ricoh_job(
            job_id, session_id, "gr3_standard", tmp_path / "out",
            preset_ids_by_photo={"photo-1": "gr3_positive_film"},
        )
        assert selected == ["gr3_positive_film", "gr3_standard"]
        assert app_api._RICOH_JOBS[job_id]["status"] == "completed"
        assert app_api._RICOH_JOBS[job_id]["success"] == 2
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
        app_api._RICOH_JOBS.pop(job_id, None)
