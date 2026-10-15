from __future__ import annotations

import os
from pathlib import Path
import sys
import json
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app_api


@pytest.fixture(autouse=True)
def clear_preview_caches():
    with app_api._DISPLAY_PREVIEW_LOCK:
        app_api._PROCESSING_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
    app_api._clear_prepared_dehaze_preview_cache()
    with app_api._ENHANCE_LOCK:
        app_api._ENHANCE_PREVIEW_CACHE.clear()
    yield
    with app_api._DISPLAY_PREVIEW_LOCK:
        app_api._PROCESSING_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_CACHE.clear()
        app_api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
    app_api._clear_prepared_dehaze_preview_cache()
    with app_api._ENHANCE_LOCK:
        app_api._ENHANCE_PREVIEW_CACHE.clear()


def _metadata():
    return SimpleNamespace(
        color_space="Linear sRGB", source_kind="raw", width=18, height=12,
        exif={},
    )


def _write_source(path: Path, content: bytes = b"source") -> None:
    path.write_bytes(content)


def _add_preview_session(session_id: str, path: Path) -> None:
    with app_api._ENHANCE_LOCK:
        app_api._ENHANCE_SESSIONS[session_id] = {"files": {"photo": path}}


def _preview_request(session_id: str, *, mode="dehazed", full=False, strength=0.2):
    return app_api.EnhancePreviewRequest(
        session_id=session_id,
        photo_id="photo",
        params=app_api.EnhanceParamsRequest(strength=strength),
        max_edge=320,
        preview_level=0 if full else 2,
        full_resolution=full,
        mode=mode,
        color_manage_srgb=False,
        render_backend="cpu",
        basic_backend="python",
    )


def _install_route_render_stubs(monkeypatch):
    def render_dehaze(image, _params, _mode, *, diagnostics=None, **_kwargs):
        diagnostics.update(auto_exposure_ev=0.0, auto_exposure_reason="off")
        return image.copy()

    monkeypatch.setattr(app_api, "_render_dehaze", render_dehaze)
    monkeypatch.setattr(
        app_api, "_correct_enhanced_raw",
        lambda image, _metadata, _path, *, preview, backend="auto": (image, None, False),
    )
    monkeypatch.setattr(
        app_api, "_display_rgb8",
        lambda image, *, linear: np.zeros((*image.shape[:2], 3), dtype=np.uint8),
    )
    monkeypatch.setattr(app_api, "_render_basic", lambda image, _basic, _mode: image)
    monkeypatch.setattr(app_api, "_encode_preview", lambda *_args, **_kwargs: b"preview")


def _install_preparation_stubs(
    monkeypatch, image: np.ndarray, decoded_image: np.ndarray | None = None,
):
    calls = {"prepare": [], "estimate": []}
    decoded = image if decoded_image is None else decoded_image

    def cached_processing(_session, _photo, _path, _max_edge):
        return decoded, _metadata()

    def prepare(_image, _metadata, _path, *, color_manage_srgb):
        assert _image is decoded
        calls["prepare"].append(color_manage_srgb)
        return image

    def estimate(candidate):
        calls["estimate"].append(candidate.shape)
        return 0.75, "applied"

    monkeypatch.setattr(app_api, "_cached_processing_preview", cached_processing)
    monkeypatch.setattr(app_api, "_prepare_dehaze_input", prepare)
    monkeypatch.setattr(app_api, "estimate_auto_exposure", estimate)
    return calls


def _prepared(path: Path, *, session="session", photo="photo", max_edge=1800,
              level=0, color=True, auto=False):
    return app_api._cached_prepared_dehaze_preview(
        session, photo, path, max_edge, level, color, auto,
    )


def test_l2_is_three_to_one_and_auto_exposure_precedes_scaling(tmp_path, monkeypatch):
    path = tmp_path / "large.raw"
    _write_source(path)
    image = np.full((900, 1800, 3), 0.25, dtype=np.float32)
    image.setflags(write=False)
    calls = _install_preparation_stubs(monkeypatch, image)

    prepared, _, exposure_ev, exposure_reason = _prepared(path, level=2, auto=True)

    assert prepared.shape == (300, 600, 3)
    assert prepared.dtype == np.float32
    assert not prepared.flags.writeable
    assert calls["estimate"] == [(900, 1800, 3)]
    assert exposure_ev == 0.75
    assert exposure_reason == "applied"
    assert np.all(image == 0.25)
    assert app_api._prepared_dehaze_preview_scale(0) == 1
    assert app_api._prepared_dehaze_preview_scale(1) == 2
    assert app_api._prepared_dehaze_preview_scale(2) == 3


def test_prepared_cache_reuses_parameter_independent_source_and_renderer_copy(
    tmp_path, monkeypatch,
):
    path = tmp_path / "photo.raw"
    _write_source(path)
    source = np.full((12, 18, 3), 0.25, dtype=np.float32)
    source.setflags(write=False)
    decoded = np.full((12, 18, 3), 8192, dtype=np.uint16)
    calls = _install_preparation_stubs(monkeypatch, source, decoded)
    rendered = []

    def render(image, params, _mode, *, diagnostics=None, auto_exposure_ev=None,
               auto_exposure_reason=None, **_kwargs):
        assert image.flags.writeable
        assert not np.shares_memory(image, source)
        rendered.append((image.copy(), params.strength, auto_exposure_ev,
                         auto_exposure_reason))
        image.fill(0.5)
        diagnostics.update(
            auto_exposure_ev=auto_exposure_ev,
            auto_exposure_reason=auto_exposure_reason,
        )
        return image

    monkeypatch.setattr(app_api, "_render_dehaze", render)
    monkeypatch.setattr(
        app_api, "_correct_enhanced_raw",
        lambda image, _metadata, _path, *, preview, backend="auto": (image, None, False),
    )
    monkeypatch.setattr(
        app_api, "_display_rgb8",
        lambda image, *, linear: np.zeros((*image.shape[:2], 3), dtype=np.uint8),
    )
    profile = SimpleNamespace(fingerprint="profile")

    def profile_display(corrected, reference, _metadata, _path, _profile,
                        _exposure, **_kwargs):
        assert reference is decoded
        assert reference.dtype == np.uint16
        return np.zeros((*corrected.shape[:2], 3), dtype=np.uint8), "applied", "test"

    monkeypatch.setattr(app_api, "_profile_display_base", profile_display)

    for strength in (0.2, 0.8):
        app_api._cached_dehazed_display_preview(
            "session", "photo", path, 64,
            app_api.DehazeParams(strength=strength), "cpu", True,
            preview_level=2, auto_exposure=True,
            profile=profile,
        )

    assert calls["prepare"] == [True]
    assert calls["estimate"] == [(12, 18, 3)]
    assert [entry[1] for entry in rendered] == [0.2, 0.8]
    assert [entry[2:] for entry in rendered] == [(0.75, "applied")] * 2
    assert all(entry[0].shape == (4, 6, 3) for entry in rendered)
    assert all(np.all(entry[0] == 0.25) for entry in rendered)
    assert np.all(source == 0.25)
    cached_linear = next(iter(app_api._PREPARED_DEHAZE_PREVIEW_CACHE.values()))[0]
    assert not cached_linear.flags.writeable
    assert np.all(cached_linear == 0.25)


def test_cache_key_covers_preview_inputs_versions_and_file_identity(
    tmp_path, monkeypatch,
):
    path = tmp_path / "photo.raw"
    _write_source(path)
    image = np.full((8, 12, 3), 0.3, dtype=np.float32)
    calls = _install_preparation_stubs(monkeypatch, image)

    _prepared(path)
    _prepared(path)
    assert len(calls["prepare"]) == 1

    _prepared(path, max_edge=900)
    _prepared(path, level=1)
    _prepared(path, color=False)
    _prepared(path, auto=True)
    assert len(calls["prepare"]) == 5

    monkeypatch.setattr(app_api, "DEHAZE_ALGORITHM_VERSION", "test-dehaze-version")
    _prepared(path)
    monkeypatch.setattr(app_api, "AUTO_EXPOSURE_ALGORITHM_VERSION", "test-auto-version")
    _prepared(path)
    monkeypatch.setattr(
        app_api, "_PREPARED_DEHAZE_PREVIEW_VERSION", "test-prepared-version",
    )
    _prepared(path)
    assert len(calls["prepare"]) == 8

    _write_source(path, b"source changed")
    _prepared(path)
    assert len(calls["prepare"]) == 9
    assert len(app_api._PREPARED_DEHAZE_PREVIEW_CACHE) == 1

    _prepared(path, session="another-session")
    _prepared(path, photo="another-photo")
    assert len(calls["prepare"]) == 11


def test_prepared_cache_enforces_byte_and_entry_budgets_with_lru(tmp_path, monkeypatch):
    monkeypatch.setattr(app_api, "_MAX_PREPARED_DEHAZE_PREVIEW_BYTES", 500)
    monkeypatch.setattr(app_api, "_MAX_PREPARED_DEHAZE_PREVIEW_ENTRIES", 2)
    image = np.full((4, 4, 3), 0.2, dtype=np.float32)
    calls = _install_preparation_stubs(monkeypatch, image)
    paths = {}
    for photo in ("a", "b", "c"):
        paths[photo] = tmp_path / f"{photo}.raw"
        _write_source(paths[photo], photo.encode())

    _prepared(paths["a"], photo="a")
    _prepared(paths["b"], photo="b")
    _prepared(paths["a"], photo="a")  # Touch a so b is least recently used.
    _prepared(paths["c"], photo="c")

    assert {key[1] for key in app_api._PREPARED_DEHAZE_PREVIEW_CACHE} == {"a", "c"}
    assert sum(value[0].nbytes for value in app_api._PREPARED_DEHAZE_PREVIEW_CACHE.values()) <= 500
    assert len(calls["prepare"]) == 3

    _prepared(paths["b"], photo="b")
    assert len(calls["prepare"]) == 4
    assert len(app_api._PREPARED_DEHAZE_PREVIEW_CACHE) <= 2

    monkeypatch.setattr(app_api, "_MAX_PREPARED_DEHAZE_PREVIEW_BYTES", 100)
    _prepared(paths["a"], photo="a", level=1)
    assert all(value[0].nbytes <= 100 for value in app_api._PREPARED_DEHAZE_PREVIEW_CACHE.values())


def test_session_cleanup_removes_only_that_sessions_prepared_arrays(tmp_path, monkeypatch):
    image = np.full((4, 4, 3), 0.2, dtype=np.float32)
    _install_preparation_stubs(monkeypatch, image)
    first = tmp_path / "first.raw"
    second = tmp_path / "second.raw"
    _write_source(first)
    _write_source(second)

    _prepared(first, session="expired", photo="first")
    _prepared(second, session="active", photo="second")
    app_api._clear_prepared_dehaze_preview_cache("expired")

    assert all(key[0] != "expired" for key in app_api._PREPARED_DEHAZE_PREVIEW_CACHE)
    assert {key[0] for key in app_api._PREPARED_DEHAZE_PREVIEW_CACHE} == {"active"}


def test_file_rewrite_with_restored_mtime_is_invalidated_by_ctime(tmp_path, monkeypatch):
    path = tmp_path / "same-size.raw"
    _write_source(path, b"source")
    image = np.full((4, 4, 3), 0.2, dtype=np.float32)
    calls = _install_preparation_stubs(monkeypatch, image)
    original_stat = path.stat()

    _prepared(path)
    _write_source(path, b"target")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    rewritten_stat = path.stat()
    assert rewritten_stat.st_size == original_stat.st_size
    assert rewritten_stat.st_mtime_ns == original_stat.st_mtime_ns
    assert rewritten_stat.st_ctime_ns != original_stat.st_ctime_ns

    _prepared(path)

    assert len(calls["prepare"]) == 2
    assert len(app_api._PREPARED_DEHAZE_PREVIEW_CACHE) == 1


def test_same_size_rewrite_redecodes_underlying_processing_preview(tmp_path, monkeypatch):
    path = tmp_path / "source.raw"
    _write_source(path, b"a")
    original_stat = path.stat()
    decoded_values = []

    def read_image(requested_path, *, preview, max_edge):
        assert Path(requested_path) == path
        assert preview is True
        assert max_edge == 64
        value = 32 if path.read_bytes() == b"a" else 224
        decoded_values.append(value)
        return np.full((4, 4, 3), value, dtype=np.uint8), _metadata()

    monkeypatch.setattr(app_api, "read_image", read_image)
    first = app_api._cached_prepared_dehaze_preview(
        "session", "photo", path, 64, 0, False, False,
    )[0]
    _write_source(path, b"b")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert path.stat().st_ctime_ns != original_stat.st_ctime_ns
    second = app_api._cached_prepared_dehaze_preview(
        "session", "photo", path, 64, 0, False, False,
    )[0]

    assert decoded_values == [32, 224]
    assert np.allclose(first, 32 / 255.0)
    assert np.allclose(second, 224 / 255.0)


def test_shutdown_handler_clears_prepared_preview_cache(tmp_path, monkeypatch):
    path = tmp_path / "photo.raw"
    _write_source(path)
    _install_preparation_stubs(
        monkeypatch, np.full((4, 4, 3), 0.2, dtype=np.float32),
    )
    _prepared(path)
    assert app_api._PREPARED_DEHAZE_PREVIEW_CACHE
    assert app_api._clear_prepared_dehaze_preview_cache in app_api.app.router.on_shutdown

    app_api._clear_prepared_dehaze_preview_cache()

    assert not app_api._PREPARED_DEHAZE_PREVIEW_CACHE


def test_cold_dehazed_preview_overlaps_profile_probe_and_decode_then_reuses_decode(
    tmp_path, monkeypatch,
):
    path = tmp_path / "photo.raw"
    _write_source(path)
    _add_preview_session("overlap-session", path)
    _install_route_render_stubs(monkeypatch)
    profile_started = Event()
    decode_started = Event()
    overlapped = Event()
    profile_threads = []
    decode_calls = []

    def resolve_profile(_path, _source_info):
        profile_threads.append(get_ident())
        profile_started.set()
        if decode_started.wait(timeout=2):
            overlapped.set()
        return None

    def read_image(_path, *, preview, max_edge):
        assert preview is True
        assert max_edge == 320
        decode_calls.append(get_ident())
        decode_started.set()
        if profile_started.wait(timeout=2):
            overlapped.set()
        return np.full((12, 18, 3), 4096, dtype=np.uint16), _metadata()

    monkeypatch.setattr(app_api, "resolve_profile", resolve_profile)
    monkeypatch.setattr(app_api, "read_image", read_image)
    caller_thread = get_ident()

    first = app_api.create_enhance_preview(_preview_request("overlap-session"))

    assert first.status_code == 200
    assert overlapped.is_set()
    assert profile_threads[0] != caller_thread
    assert len(decode_calls) == 1

    second = app_api.create_enhance_preview(
        _preview_request("overlap-session", strength=0.8),
    )

    assert second.status_code == 200
    assert profile_threads[1] == caller_thread  # Processing-preview cache is hot.
    assert len(decode_calls) == 1

    with app_api._ENHANCE_LOCK:
        app_api._ENHANCE_SESSIONS.pop("overlap-session", None)


def test_full_resolution_and_original_previews_do_not_use_cold_preloader(
    tmp_path, monkeypatch,
):
    path = tmp_path / "photo.raw"
    _write_source(path)
    _add_preview_session("no-preload-session", path)
    _install_route_render_stubs(monkeypatch)
    profile_threads = []

    def resolve_profile(_path, _source_info):
        profile_threads.append(get_ident())
        return None

    def fail_cached_processing(*_args, **_kwargs):
        raise AssertionError("full/original preview must not run cold dehazed preloading")

    monkeypatch.setattr(app_api, "resolve_profile", resolve_profile)
    monkeypatch.setattr(app_api, "_cached_processing_preview", fail_cached_processing)
    monkeypatch.setattr(
        app_api, "_full_resolution_display_base",
        lambda *_args, **_kwargs: (
            np.zeros((12, 18, 3), dtype=np.uint8),
            ("off", "", 0.0, "off", "legacy", "legacy", "not-needed"),
            False, "miss",
        ),
    )
    caller_thread = get_ident()

    full = app_api.create_enhance_preview(
        _preview_request("no-preload-session", full=True),
    )
    assert full.status_code == 200
    assert profile_threads == [caller_thread]

    def read_original(_path, *, preview, max_edge):
        assert preview is True
        assert max_edge == 320
        return np.zeros((12, 18, 3), dtype=np.uint8), _metadata()

    monkeypatch.setattr(app_api, "read_image", read_original)
    original = app_api.create_enhance_preview(
        _preview_request("no-preload-session", mode="original"),
    )
    assert original.status_code == 200
    assert profile_threads == [caller_thread, caller_thread]

    with app_api._ENHANCE_LOCK:
        app_api._ENHANCE_SESSIONS.pop("no-preload-session", None)
    app_api._shutdown_full_resolution_preview_cache()


def test_decode_failure_joins_profile_probe_before_returning_error(tmp_path, monkeypatch):
    path = tmp_path / "broken.raw"
    _write_source(path)
    _add_preview_session("failure-session", path)
    profile_started = Event()
    decode_started = Event()
    profile_finished = Event()

    def resolve_profile(_path, _source_info):
        profile_started.set()
        try:
            assert decode_started.wait(timeout=2)
            return None
        finally:
            profile_finished.set()

    def fail_read(_path, *, preview, max_edge):
        assert preview is True
        decode_started.set()
        assert profile_started.wait(timeout=2)
        raise ValueError("simulated preview decode failure")

    monkeypatch.setattr(app_api, "resolve_profile", resolve_profile)
    monkeypatch.setattr(app_api, "read_image", fail_read)

    response = app_api.create_enhance_preview(_preview_request("failure-session"))

    assert profile_finished.is_set()
    assert response.status_code == (
        422 if app_api._is_enhance_unsupported_error(
            ValueError("simulated preview decode failure"),
        ) else 500
    )
    assert json.loads(response.body) == {
        "error": app_api._enhance_error_message(
            ValueError("simulated preview decode failure"),
        ),
    }
    with app_api._ENHANCE_LOCK:
        app_api._ENHANCE_SESSIONS.pop("failure-session", None)
