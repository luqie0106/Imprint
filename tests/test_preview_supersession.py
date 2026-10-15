"""Generation-aware cancellation for full-resolution enhance previews."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_api as api


@pytest.fixture
def preview_api(tmp_path, monkeypatch):
    session_id = f"supersession-{uuid.uuid4().hex}"
    photo_id = "photo"
    path = tmp_path / "photo.nef"
    path.write_bytes(b"fixture")
    image = np.full((16, 24, 3), 12000, dtype=np.uint16)
    metadata = SimpleNamespace(
        source_kind="raw", color_space="Linear sRGB", width=24, height=16,
    )
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS[session_id] = {
            "files": {photo_id: path}, "created": time.time(),
        }

    read_calls = []
    encode_calls = []

    def read_image(_path, *, preview, max_edge=None):
        read_calls.append((preview, max_edge))
        return image.copy(), metadata

    monkeypatch.setattr(api, "read_image", read_image)
    monkeypatch.setattr(api, "resolve_profile", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api, "_encode_preview", lambda value, **_kwargs: encode_calls.append(value) or b"jpeg")
    monkeypatch.setattr(api, "_render_basic", lambda value, *_args, **_kwargs: value)

    client = TestClient(api.app)
    yield SimpleNamespace(
        client=client, session_id=session_id, photo_id=photo_id,
        path=path, image=image, metadata=metadata,
        read_calls=read_calls, encode_calls=encode_calls,
    )
    client.close()
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS.pop(session_id, None)
        for key in [key for key in api._ENHANCE_PREVIEW_GENERATIONS if key[0] == session_id]:
            api._ENHANCE_PREVIEW_GENERATIONS.pop(key, None)
        for key in [key for key in api._ENHANCE_PREVIEW_CACHE if key[0] == session_id]:
            api._ENHANCE_PREVIEW_CACHE.pop(key, None)
    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        if (api._FULL_RESOLUTION_ACTIVE_SOURCE_KEY is not None
                and api._FULL_RESOLUTION_ACTIVE_SOURCE_KEY[0] == session_id):
            api._clear_full_resolution_preview_caches(clear_previous=True)


def post_preview(env, **overrides):
    request = {
        "session_id": env.session_id,
        "photo_id": env.photo_id,
        "mode": "original",
        "render_backend": "cpu",
        "basic_backend": "python",
    }
    request.update(overrides)
    return env.client.post("/api/enhance/preview", json=request)


def test_new_l2_generation_expires_full_request_waiting_for_lock(preview_api, monkeypatch):
    env = preview_api
    lock_held = threading.Event()
    original_lock_context = api._full_resolution_preview_lock

    @contextmanager
    def observed_lock(req):
        if req.preview_generation == 1:
            lock_held.set()
        with original_lock_context(req):
            yield

    monkeypatch.setattr(api, "_full_resolution_preview_lock", observed_lock)
    assert api._FULL_RESOLUTION_PREVIEW_LOCK.acquire(timeout=1)
    responses = []
    errors = []

    def old_full_request():
        try:
            responses.append(post_preview(
                env, full_resolution=True, preview_level=0,
                preview_client_id="page-a", preview_generation=1,
            ))
        except Exception as exc:  # surface worker-thread failures in the test
            errors.append(exc)

    worker = threading.Thread(target=old_full_request, daemon=True)
    worker.start()
    try:
        assert lock_held.wait(2)
        newer_l2 = post_preview(
            env, full_resolution=False, preview_level=2,
            preview_client_id="page-a", preview_generation=2,
        )
        assert newer_l2.status_code == 200
    finally:
        api._FULL_RESOLUTION_PREVIEW_LOCK.release()
    worker.join(3)

    assert not worker.is_alive()
    assert not errors
    assert len(responses) == 1 and responses[0].status_code == 204
    # The L2 read and encode ran; the stale full-resolution path did neither.
    assert env.read_calls == [(True, 1800)]
    assert len(env.encode_calls) == 1


def test_expired_decode_stops_at_boundary_and_joins_camera_worker(preview_api, monkeypatch):
    env = preview_api
    profile = SimpleNamespace(fingerprint=("supersession", 1), name="Fixture")
    monkeypatch.setattr(api, "resolve_profile", lambda *_args, **_kwargs: profile)
    monkeypatch.setattr(api, "_prepare_dehaze_input", lambda value, *_args, **_kwargs: value.astype(np.float32))

    render_started = threading.Event()
    release_render = threading.Event()
    camera_started = threading.Event()
    release_camera = threading.Event()
    camera_finished = threading.Event()
    camera_threads = []
    stages = {"render": 0, "correct": 0, "transfer": 0, "profile": 0}

    def blocking_render(value, *_args, diagnostics, **_kwargs):
        stages["render"] += 1
        render_started.set()
        assert release_render.wait(5)
        diagnostics.update(auto_exposure_ev=0.0, auto_exposure_reason="off")
        return np.zeros_like(value)

    def blocking_camera(_path, shape, _preview):
        camera_threads.append(threading.current_thread())
        camera_started.set()
        assert release_camera.wait(5)
        camera_finished.set()
        return np.full(shape, 12000, dtype=np.uint16), (1.0, 1.0, 1.0)

    def count_correct(value, *_args, **_kwargs):
        stages["correct"] += 1
        return value, None, False

    def count_transfer(*_args, **_kwargs):
        stages["transfer"] += 1
        return env.image.copy()

    def count_profile(*_args, **_kwargs):
        stages["profile"] += 1
        return np.zeros((16, 24, 3), dtype=np.uint8)

    monkeypatch.setattr(api, "_render_dehaze", blocking_render)
    monkeypatch.setattr(api, "load_camera_rgb", blocking_camera)
    monkeypatch.setattr(api, "_correct_enhanced_raw", count_correct)
    monkeypatch.setattr(api, "transfer_enhancement", count_transfer)
    monkeypatch.setattr(api, "render_profile", count_profile)

    responses = []
    errors = []

    def old_full_request():
        try:
            responses.append(post_preview(
                env, mode="dehazed", full_resolution=True, preview_level=0,
                preview_client_id="page-a", preview_generation=1,
            ))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=old_full_request, daemon=True)
    worker.start()
    try:
        assert render_started.wait(3)
        assert camera_started.wait(3)
        canceled = post_preview(
            env, preview_client_id="page-a", preview_generation=2,
            preview_cancel_only=True,
        )
        assert canceled.status_code == 204 and canceled.content == b""
        release_render.set()
        # The request must join the outstanding camera decoder before returning.
        assert not responses
        release_camera.set()
    finally:
        release_render.set()
        release_camera.set()
    worker.join(5)

    assert not worker.is_alive()
    assert not errors
    assert camera_finished.is_set() and camera_threads
    assert all(not thread.is_alive() for thread in camera_threads)
    assert len(responses) == 1 and responses[0].status_code == 204
    assert stages == {"render": 1, "correct": 0, "transfer": 0, "profile": 0}
    assert env.encode_calls == []
    # A complete source decode remains reusable after its render was superseded.
    assert any(key[0] == env.session_id for key in api._FULL_RESOLUTION_DECODED_CACHE)


def test_generation_isolated_by_client_and_session_and_never_moves_back(preview_api):
    env = preview_api
    session_two = f"other-{uuid.uuid4().hex}"
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS[session_two] = {
            "files": {env.photo_id: env.path}, "created": time.time(),
        }

    def cancel(session_id, client_id, generation):
        return env.client.post("/api/enhance/preview", json={
            "session_id": session_id, "photo_id": env.photo_id,
            "preview_client_id": client_id, "preview_generation": generation,
            "preview_cancel_only": True,
        })

    try:
        assert cancel(env.session_id, "page-a", 10).status_code == 204
        assert cancel(env.session_id, "page-a", 8).status_code == 204
        assert cancel(env.session_id, "page-b", 1).status_code == 204
        assert cancel(session_two, "page-a", 0).status_code == 204
        assert api._ENHANCE_PREVIEW_GENERATIONS[(env.session_id, "page-a")] == 10
        assert api._ENHANCE_PREVIEW_GENERATIONS[(env.session_id, "page-b")] == 1
        assert api._ENHANCE_PREVIEW_GENERATIONS[(session_two, "page-a")] == 0

        missing_id = env.client.post("/api/enhance/preview", json={
            "session_id": env.session_id, "photo_id": env.photo_id,
            "preview_cancel_only": True,
        })
        assert missing_id.status_code == 400
    finally:
        with api._ENHANCE_LOCK:
            api._ENHANCE_SESSIONS.pop(session_two, None)
            api._ENHANCE_PREVIEW_GENERATIONS.pop((session_two, "page-a"), None)


def test_legacy_request_without_generation_keeps_preview_cache_behavior(preview_api):
    env = preview_api
    first = post_preview(env)
    second = post_preview(env)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content == b"jpeg"
    assert env.read_calls == [(True, 1800)]
    assert len(env.encode_calls) == 1
    assert not any(key[0] == env.session_id for key in api._ENHANCE_PREVIEW_GENERATIONS)


def test_generation_registry_is_bounded_and_rejects_unsafe_js_integer(preview_api):
    env = preview_api
    with api._ENHANCE_LOCK:
        for key in [key for key in api._ENHANCE_PREVIEW_GENERATIONS if key[0] == env.session_id]:
            api._ENHANCE_PREVIEW_GENERATIONS.pop(key, None)

    for index in range(api._MAX_ENHANCE_PREVIEW_GENERATIONS + 12):
        req = api.EnhancePreviewRequest(
            session_id=env.session_id, photo_id=env.photo_id,
            preview_client_id=f"client-{index}", preview_generation=index,
        )
        assert api._register_preview_generation(req)

    session_keys = [key for key in api._ENHANCE_PREVIEW_GENERATIONS if key[0] == env.session_id]
    assert len(session_keys) == api._MAX_ENHANCE_PREVIEW_GENERATIONS
    assert (env.session_id, "client-0") not in api._ENHANCE_PREVIEW_GENERATIONS
    assert (env.session_id, f"client-{api._MAX_ENHANCE_PREVIEW_GENERATIONS + 11}") in api._ENHANCE_PREVIEW_GENERATIONS

    with pytest.raises(ValidationError):
        api.EnhancePreviewRequest(
            session_id=env.session_id, photo_id=env.photo_id,
            preview_client_id="overflow", preview_generation=1 << 53,
        )
