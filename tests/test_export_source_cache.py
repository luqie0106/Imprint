"""Full-resolution preview camera snapshots reused by DNG export."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_api as api
from dehaze import DehazeParams
from lens_correction import LensCorrectionResult


@pytest.fixture
def export_scene(tmp_path, monkeypatch):
    api._shutdown_full_resolution_preview_cache()
    path = tmp_path / "source.nef"
    path.write_bytes(b"raw fixture")
    image = np.full((16, 20, 3), 24000, dtype=np.uint16)
    metadata = SimpleNamespace(
        source_kind="raw", color_space="Linear sRGB", bit_depth=14, exif={},
    )
    profile = SimpleNamespace(fingerprint=("test-profile",), name="Test Profile")
    lens = LensCorrectionResult(
        applied=True, camera_name="Test Camera", lens_name="Test Lens",
        distortion_applied=True, tca_applied=False, vignetting_applied=False,
        scale=1.0,
    )
    calls = {
        "decode": 0, "dehaze": 0, "correct": 0, "camera": 0,
        "build_cache": 0, "transfer": 0, "source_data": 0, "write": 0,
    }

    def read_image(_path, *, preview):
        assert preview is False
        calls["decode"] += 1
        return image.copy(), metadata

    def correct(value, *_args, **_kwargs):
        calls["correct"] += 1
        return value.copy(), lens, False

    def load_camera(_path, shape, preview):
        calls["camera"] += 1
        return np.full(shape, 22000, dtype=np.uint16), (1.0, 1.0, 1.0)

    def build_cache(source_path, raw_camera, reference, camera, corrected_reference,
                    camera_lens, reference_lens, *, correction_version,
                    camera_gain_applied=False, reference_gain_applied=False):
        calls["build_cache"] += 1
        assert camera_lens is reference_lens is lens
        assert camera_gain_applied is reference_gain_applied is False
        matrix = np.eye(3, dtype=np.float32)
        matrix.setflags(write=False)
        return SimpleNamespace(
            source_identity=api.enhanced_dng_source_identity(source_path),
            source_shape=tuple(reference.shape),
            camera_rgb16=camera,
            reference_rgb16=corrected_reference,
            srgb_to_camera=matrix,
            camera_lens_result=camera_lens,
            reference_lens_result=reference_lens,
            correction_version=correction_version,
        )

    def render_dehaze(value, params, _backend, *, diagnostics, **_kwargs):
        calls["dehaze"] += 1
        diagnostics.update(auto_exposure_ev=0.0, auto_exposure_reason="off")
        return value * np.float32(1.0 - params.strength * 0.2)

    def transfer(_camera, _reference, processed, *, backend="cpu"):
        calls["transfer"] += 1
        return processed.copy()

    def render_profile(camera, _neutral, _profile, _ev, *, backend="cpu"):
        return np.full(camera.shape, 96, dtype=np.uint8)

    def source_data(processed, reference, source_path, _source_exif, **kwargs):
        calls["source_data"] += 1
        assert processed.dtype == reference.dtype == np.uint16
        assert processed.shape == reference.shape
        diagnostics = kwargs.get("cache_diagnostics")
        if "source_cache" in kwargs:
            assert kwargs["correction_version"] == api.LENS_PREVIEW_VERSION
            assert kwargs["source_cache"].source_identity == api.enhanced_dng_source_identity(source_path)
            assert diagnostics is not None
            diagnostics["camera"] = "hit"
        else:
            # The old four-argument call remains valid for cold or incompatible sources.
            assert not kwargs
        return (
            np.full(processed.shape, 18000, dtype=np.uint16),
            np.ones((2, 2), dtype=np.uint16),
            np.array([[0, 1], [1, 2]], dtype=np.uint8),
            {"DNGUniqueCameraModel": "Test Camera"},
            1,
        )

    def write_dng(*_args, **_kwargs):
        calls["write"] += 1
        return tmp_path / "output.dng"

    monkeypatch.setattr(api, "read_image", read_image)
    monkeypatch.setattr(api, "_correct_enhanced_raw", correct)
    monkeypatch.setattr(api, "_correct_profile_inputs",
                        lambda images, *args, **kwargs: [correct(image) for image in images])
    monkeypatch.setattr(api, "load_camera_rgb", load_camera)
    monkeypatch.setattr(api, "build_enhanced_dng_source_cache", build_cache)
    monkeypatch.setattr(api, "_render_dehaze", render_dehaze)
    monkeypatch.setattr(api, "transfer_enhancement", transfer)
    monkeypatch.setattr(api, "render_profile", render_profile)
    monkeypatch.setattr(api, "enhanced_dng_source_data", source_data)
    monkeypatch.setattr(api, "write_enhanced_dng", write_dng)
    monkeypatch.setattr(api, "camera_profile_names", lambda _path: {})
    monkeypatch.setattr(api, "matching_embedded_profile_dng", lambda *_args: None)
    monkeypatch.setattr(api, "resolve_profile", lambda *_args: profile)

    session_id = "export-source-cache-test"
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS[session_id] = {"files": {"photo": path}}
    yield session_id, path, image, metadata, profile, calls
    api._shutdown_full_resolution_preview_cache()
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS.pop(session_id, None)
        for job_id in ("export-source-cache-job", "export-source-cache-cold-job",
                       "export-source-cache-other-job", "export-source-cache-changed-job"):
            api._ENHANCE_JOBS.pop(job_id, None)


def _seed_preview(session_id, path, profile, strength=0.2):
    req = api.EnhancePreviewRequest(
        session_id=session_id, photo_id="photo",
        params=api.EnhanceParamsRequest(strength=strength),
        full_resolution=True, render_backend="cpu",
    )
    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        return api._full_resolution_display_base(
            req, path, DehazeParams(strength=strength), "cpu", profile, 0.0,
        )


def _run_export(session_id, path, job_id, *, photo_id="photo", strength=0.8):
    item = {"photo_id": photo_id, "name": path.name, "status": "waiting"}
    with api._ENHANCE_LOCK:
        api._ENHANCE_JOBS[job_id] = {
            "job_id": job_id, "status": "queued", "total": 1,
            "processed": 0, "success": 0, "failed": 0, "progress": 0.0,
            "current_file": "", "files": [item], "_cancel": api.threading.Event(),
        }
    api._run_enhance_job(
        job_id, session_id, path.parent, DehazeParams(),
        {photo_id: DehazeParams(strength=strength)},
        preset_ids_by_photo={},
    )
    return api._ENHANCE_JOBS[job_id]["files"][0]


def test_export_reuses_snapshot_across_parameter_changes_and_decode_eviction(export_scene):
    session_id, path, _image, _metadata, profile, calls = export_scene
    _seed_preview(session_id, path, profile, strength=0.2)
    assert len(api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE) == 1
    assert calls["decode"] == calls["dehaze"] == 1

    hot = _run_export(session_id, path, "export-source-cache-job", strength=0.85)
    assert hot["status"] == "success"
    assert hot["source_cache"] == {"decoded": "hit", "camera": "hit"}
    assert calls["decode"] == 1  # The cached full-resolution decode skipped read_image.
    assert calls["dehaze"] == 2  # Export still reran the requested effect.
    assert calls["correct"] == 4  # Preview correction plus a fresh export correction.
    assert calls["source_data"] == calls["write"] == 1

    # Evict only the decode; the independent camera snapshot remains usable.
    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        api._FULL_RESOLUTION_DECODED_CACHE.clear()
        api._FULL_RESOLUTION_DECODED_SOURCE_IDENTITIES.clear()
    cold = _run_export(
        session_id, path, "export-source-cache-cold-job", strength=0.1,
    )
    assert cold["status"] == "success"
    assert cold["source_cache"] == {"decoded": "miss", "camera": "hit"}
    assert calls["decode"] == 2
    assert calls["dehaze"] == 3
    assert calls["source_data"] == 2


def test_changed_source_misses_both_caches(export_scene):
    session_id, path, _image, _metadata, profile, calls = export_scene
    _seed_preview(session_id, path, profile)
    path.write_bytes(b"raw fixture replaced with a different identity")

    item = _run_export(
        session_id, path, "export-source-cache-changed-job", strength=0.3,
    )
    assert item["status"] == "success"
    assert item["source_cache"] == {"decoded": "miss", "camera": "miss"}
    assert calls["decode"] == 2
    assert calls["source_data"] == 1
    assert not api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE


def test_export_of_another_photo_keeps_foreground_preview_cache(export_scene):
    session_id, path, _image, _metadata, profile, calls = export_scene
    _seed_preview(session_id, path, profile)
    camera_keys = tuple(api._FULL_RESOLUTION_CAMERA_SOURCE_CACHE)
    export_keys = tuple(api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE)
    base_keys = tuple(api._FULL_RESOLUTION_BASE_CACHE)

    other_path = path.with_name("other.nef")
    other_path.write_bytes(b"other raw fixture")
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS[session_id]["files"]["other"] = other_path
    item = _run_export(
        session_id, other_path, "export-source-cache-other-job",
        photo_id="other", strength=0.4,
    )

    assert item["status"] == "success"
    assert item["source_cache"] == {"decoded": "miss", "camera": "miss"}
    assert tuple(api._FULL_RESOLUTION_CAMERA_SOURCE_CACHE) == camera_keys
    assert tuple(api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE) == export_keys
    assert tuple(api._FULL_RESOLUTION_BASE_CACHE) == base_keys
    assert calls["decode"] == 2


def test_export_snapshot_shares_budget_and_expires_with_preview_layers(export_scene, monkeypatch):
    session_id, path, _image, _metadata, profile, _calls = export_scene
    _seed_preview(session_id, path, profile)
    source = next(iter(api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE.values()))
    assert not source.camera_rgb16.flags.writeable
    assert not source.reference_rgb16.flags.writeable

    arrays = {}

    def include(array):
        arrays[id(array)] = array

    for value in api._FULL_RESOLUTION_DECODED_CACHE.values():
        include(value[0])
    for value in api._FULL_RESOLUTION_BASE_CACHE.values():
        include(value[0])
    for camera, reference, neutral in api._FULL_RESOLUTION_CAMERA_SOURCE_CACHE.values():
        for array in (camera, reference, neutral):
            include(array)
    for processed, neutral, _status in api._FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.values():
        include(processed)
        include(neutral)
    for retained in api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE.values():
        for array in (retained.camera_rgb16, retained.reference_rgb16, retained.srgb_to_camera):
            include(array)
    assert api._full_resolution_preview_cache_bytes() == sum(a.nbytes for a in arrays.values())

    monkeypatch.setattr(api, "_MAX_FULL_RESOLUTION_PREVIEW_CACHE_BYTES", 1)
    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        api._trim_full_resolution_preview_cache_budget()
    assert api._full_resolution_preview_cache_bytes() <= 1
    assert not api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE

    api._shutdown_full_resolution_preview_cache()
    assert api._full_resolution_preview_cache_bytes() == 0


def test_low_resolution_and_mutable_camera_arrays_are_not_export_snapshots(export_scene):
    session_id, path, image, metadata, profile, calls = export_scene
    identity = api.enhanced_dng_source_identity(path)
    full_cache_identity = (session_id, "photo", str(path), path.stat().st_mtime_ns, path.stat().st_size)
    display, status, _backend = api._profile_display_base(
        image.copy(), image, metadata, path, profile, 0.0,
        preview=True, full_cache_identity=full_cache_identity,
        full_export_identity=identity,
    )
    assert display.dtype == np.uint8
    assert status == "applied"
    assert calls["build_cache"] == 0
    assert not api._FULL_RESOLUTION_EXPORT_SOURCE_CACHE

    mutable_camera = np.zeros_like(image)
    mutable_reference = np.zeros_like(image)
    matrix = np.eye(3, dtype=np.float32)
    matrix.setflags(write=False)
    mutable_source = SimpleNamespace(
        source_identity=identity,
        source_shape=tuple(image.shape),
        camera_rgb16=mutable_camera,
        reference_rgb16=mutable_reference,
        srgb_to_camera=matrix,
        correction_version=api.LENS_PREVIEW_VERSION,
    )
    assert not api._valid_full_resolution_export_source_cache(
        mutable_source, identity, tuple(image.shape), api.LENS_PREVIEW_VERSION,
    )
