from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app_api


def _metadata():
    return SimpleNamespace(color_space="sRGB", source_kind="raw")


def _clear_preview_caches():
    with app_api._DISPLAY_PREVIEW_LOCK:
        app_api._DISPLAY_PREVIEW_CACHE.clear()
        app_api._PROCESSING_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()


def test_distinct_photos_decode_outside_shared_preview_lock(tmp_path, monkeypatch):
    path_a = tmp_path / "a.raw"
    path_b = tmp_path / "b.raw"
    path_a.write_bytes(b"a")
    path_b.write_bytes(b"b")
    barrier = Barrier(2)
    calls = []

    def read_image(path, *, preview, max_edge):
        assert preview is True
        assert max_edge == 64
        calls.append(Path(path).name)
        barrier.wait(timeout=5)
        value = 48 if Path(path) == path_a else 96
        return np.full((12, 12, 3), value, dtype=np.uint8), _metadata()

    monkeypatch.setattr(app_api, "read_image", read_image)
    _clear_preview_caches()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(app_api._cached_display_preview, "s", "a", path_a, 64),
            pool.submit(app_api._cached_display_preview, "s", "b", path_b, 64),
        ]
        results = [future.result(timeout=8) for future in futures]

    assert sorted(calls) == ["a.raw", "b.raw"]
    assert [int(result[0, 0, 0]) for result in results] == [48, 96]

    cached = app_api._cached_display_preview("s", "a", path_a, 64)
    assert cached is results[0]
    assert len(calls) == 2


def test_processing_preview_decodes_distinct_photos_concurrently(tmp_path, monkeypatch):
    path_a = tmp_path / "a.raw"
    path_b = tmp_path / "b.raw"
    path_a.write_bytes(b"a")
    path_b.write_bytes(b"b")
    barrier = Barrier(2)

    def read_image(path, *, preview, max_edge):
        assert preview is True
        assert max_edge == 64
        barrier.wait(timeout=5)
        value = 48 if Path(path) == path_a else 96
        return np.full((12, 12, 3), value, dtype=np.uint8), _metadata()

    monkeypatch.setattr(app_api, "read_image", read_image)
    _clear_preview_caches()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(app_api._cached_processing_preview, "s", "a", path_a, 64),
            pool.submit(app_api._cached_processing_preview, "s", "b", path_b, 64),
        ]
        results = [future.result(timeout=8) for future in futures]

    assert [int(result[0][0, 0, 0]) for result in results] == [48, 96]
    assert [result[1].color_space for result in results] == ["sRGB", "sRGB"]


def test_dehazed_rendering_runs_outside_shared_preview_lock(tmp_path, monkeypatch):
    path_a = tmp_path / "a.raw"
    path_b = tmp_path / "b.raw"
    path_a.write_bytes(b"a")
    path_b.write_bytes(b"b")
    barrier = Barrier(2)
    calls = []

    def read_image(path, *, preview, max_edge):
        assert preview is True
        assert max_edge == 64
        value = 0.25 if Path(path) == path_a else 0.5
        return np.full((12, 12, 3), value, dtype=np.float32), _metadata()

    def render(image, _params, *, backend, spatial, nonlocal_mode, diagnostics=None):
        assert backend == "native"
        assert spatial is False
        assert nonlocal_mode == "off"
        barrier.wait(timeout=5)
        calls.append(True)
        return image

    monkeypatch.setattr(app_api, "read_image", read_image)
    monkeypatch.setattr(app_api, "apply_physical_dehaze", render)
    _clear_preview_caches()
    params = app_api.DehazeParams()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                app_api._cached_dehazed_display_preview,
                "s", "a", path_a, 64, params, "native", False,
            ),
            pool.submit(
                app_api._cached_dehazed_display_preview,
                "s", "b", path_b, 64, params, "native", False,
            ),
        ]
        results = [future.result(timeout=8) for future in futures]

    assert len(calls) == 2
    assert [int(result[0, 0, 0]) for result in results] == [137, 188]


def test_thumbnail_size_cache_and_session_boundary(tmp_path, monkeypatch):
    path = tmp_path / "photo.nef"
    path.write_bytes(b"placeholder")
    session_id = "opaque-session-id"
    photo_id = "opaque-photo-id"
    extracted = []

    def extract_preview(_self, requested_path):
        extracted.append(requested_path)
        return np.zeros((1200, 2400, 3), dtype=np.uint8)

    monkeypatch.setattr(app_api.RawEvaluator, "extract_preview", extract_preview)
    monkeypatch.setattr(app_api, "_orient_enhance_thumbnail", lambda image, _path: image)
    with app_api._ENHANCE_LOCK:
        app_api._ENHANCE_SESSIONS[session_id] = {"files": {photo_id: path}}
        for key in [key for key in app_api._ENHANCE_THUMBNAIL_CACHE if key[0] == session_id]:
            app_api._ENHANCE_THUMBNAIL_CACHE.pop(key, None)

    try:
        client = TestClient(app_api.app)
        assert client.get("/api/health").json() == {"ok": True}
        large = client.get(
            f"/api/enhance/thumbnail/{session_id}/{photo_id}?max_edge=900"
        )
        default = client.get(f"/api/enhance/thumbnail/{session_id}/{photo_id}")
        large_cached = client.get(
            f"/api/enhance/thumbnail/{session_id}/{photo_id}?max_edge=900"
        )

        assert large.status_code == 200
        assert default.status_code == 200
        assert large_cached.content == large.content
        assert Image.open(io.BytesIO(large.content)).size == (900, 450)
        assert Image.open(io.BytesIO(default.content)).size == (360, 180)
        assert len(extracted) == 2
        assert client.get(
            f"/api/enhance/thumbnail/unknown-session/{photo_id}"
        ).status_code == 404
        assert client.get(
            f"/api/enhance/thumbnail/{session_id}/{photo_id}?max_edge=1801"
        ).status_code == 422
    finally:
        with app_api._ENHANCE_LOCK:
            app_api._ENHANCE_SESSIONS.pop(session_id, None)
            for key in [key for key in app_api._ENHANCE_THUMBNAIL_CACHE if key[0] == session_id]:
                app_api._ENHANCE_THUMBNAIL_CACHE.pop(key, None)


def test_standard_jpeg_thumbnail_uses_scaled_decode(tmp_path, monkeypatch):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (2400, 1200), (40, 80, 120)).save(path, format="JPEG")
    session_id = "jpeg-session-id"
    photo_id = "jpeg-photo-id"

    def fail_extract(_self, _path):
        raise AssertionError("JPEG thumbnail should use Pillow's scaled decode")

    monkeypatch.setattr(app_api.RawEvaluator, "extract_preview", fail_extract)
    with app_api._ENHANCE_LOCK:
        app_api._ENHANCE_SESSIONS[session_id] = {"files": {photo_id: path}}
    try:
        response = TestClient(app_api.app).get(
            f"/api/enhance/thumbnail/{session_id}/{photo_id}?max_edge=900"
        )
        assert response.status_code == 200
        assert Image.open(io.BytesIO(response.content)).size == (900, 450)
    finally:
        with app_api._ENHANCE_LOCK:
            app_api._ENHANCE_SESSIONS.pop(session_id, None)
            for key in [key for key in app_api._ENHANCE_THUMBNAIL_CACHE if key[0] == session_id]:
                app_api._ENHANCE_THUMBNAIL_CACHE.pop(key, None)
