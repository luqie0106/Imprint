"""Memory and reuse boundaries for the full-resolution decoded preview cache."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_api as api


@pytest.fixture(autouse=True)
def clean_full_preview_caches():
    api._shutdown_full_resolution_preview_cache()
    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        api._clear_full_resolution_preview_caches()
    yield
    api._shutdown_full_resolution_preview_cache()


def request(session="decoded-cache", photo="photo"):
    return api.EnhancePreviewRequest(
        session_id=session, photo_id=photo, full_resolution=True,
        preview_level=0, render_backend="cpu", basic_backend="python",
    )


def decoded(req, path):
    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        return api._full_resolution_decoded_image(req, path)


def test_decoded_image_reuses_readonly_pixels_until_file_or_photo_changes(monkeypatch, tmp_path):
    first_path = tmp_path / "first.nef"
    second_path = tmp_path / "second.nef"
    first_path.touch()
    second_path.touch()
    source = np.full((8, 11, 3), .25, dtype=np.float32)
    calls = []

    def read(path, *, preview):
        assert preview is False
        calls.append(Path(path))
        return source.copy(), SimpleNamespace(color_space="Linear sRGB", source_kind="raw")

    monkeypatch.setattr(api, "read_image", read)
    req = request()
    image, metadata, hit = decoded(req, first_path)
    assert not hit and not image.flags.writeable
    assert decoded(req, first_path)[2]
    assert calls == [first_path]

    # A file identity change drops the old decoded array and rendered base.
    api._FULL_RESOLUTION_BASE_CACHE[("decoded-cache", "photo", str(first_path),
                                     first_path.stat().st_mtime_ns, first_path.stat().st_size,
                                     "extra")] = (np.zeros((1, 1, 3), np.uint8), ())
    first_path.write_bytes(b"changed")
    changed, _, hit = decoded(req, first_path)
    assert not hit and changed is not image
    assert len(api._FULL_RESOLUTION_DECODED_CACHE) == 1
    assert not api._FULL_RESOLUTION_BASE_CACHE

    switched, _, hit = decoded(request(photo="other"), second_path)
    assert not hit and switched is not changed
    assert list(api._FULL_RESOLUTION_DECODED_CACHE) == [
        ("decoded-cache", "other", str(second_path),
         second_path.stat().st_mtime_ns, second_path.stat().st_size)
    ]
    assert api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE is not None
    assert api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE[2] is changed
    assert len(calls) == 3
    assert metadata.color_space == "Linear sRGB"


def test_previous_source_reuses_a_b_a_decode_and_swaps_one_snapshot(monkeypatch, tmp_path):
    first_path = tmp_path / "first.nef"
    second_path = tmp_path / "second.nef"
    first_path.touch()
    second_path.touch()
    calls = []

    def read(path, *, preview):
        assert preview is False
        calls.append(Path(path))
        return np.full((12, 16, 3), len(calls), dtype=np.uint16), SimpleNamespace(
            source_kind="raw",
        )

    monkeypatch.setattr(api, "read_image", read)
    first_req = request(session="aba", photo="a")
    second_req = request(session="aba", photo="b")
    first, _, first_hit = decoded(first_req, first_path)
    decoded(second_req, second_path)
    reused, _, reused_hit = decoded(first_req, first_path)

    assert not first_hit and reused_hit
    assert reused is first
    assert calls == [first_path, second_path]
    assert list(api._FULL_RESOLUTION_DECODED_CACHE) == [
        ("aba", "a", str(first_path), first_path.stat().st_mtime_ns,
         first_path.stat().st_size)
    ]
    assert api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE[0][:2] == ("aba", "b")
    assert api._full_resolution_total_preview_cache_bytes() <= (
        api._MAX_FULL_RESOLUTION_PREVIEW_CACHE_BYTES
        + api._MAX_FULL_RESOLUTION_PREVIOUS_DECODED_BYTES
    )


def test_changed_strong_identity_discards_matching_previous_snapshot(monkeypatch, tmp_path):
    first_path = tmp_path / "first.nef"
    second_path = tmp_path / "second.nef"
    first_path.touch()
    second_path.touch()
    original_identity = api.enhanced_dng_source_identity
    identity_versions = {first_path: 0, second_path: 0}

    def identity(path):
        resolved = Path(path).resolve()
        return (*original_identity(resolved), identity_versions[resolved])

    monkeypatch.setattr(api, "enhanced_dng_source_identity", identity)
    calls = []

    def read(path, *, preview):
        assert preview is False
        calls.append(Path(path))
        return np.full((3, 4, 3), len(calls), dtype=np.uint16), SimpleNamespace(
            source_kind="raw",
        )

    monkeypatch.setattr(api, "read_image", read)
    first_req = request(session="identity", photo="a")
    original, _, _ = decoded(first_req, first_path)
    decoded(request(session="identity", photo="b"), second_path)
    assert api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE[2] is original

    # Keep the legacy stat key stable while changing the stronger filesystem
    # identity, as can happen on same-size, timestamp-preserving replacements.
    identity_versions[first_path.resolve()] = 1
    replaced, _, hit = decoded(first_req, first_path)
    assert not hit and replaced is not original
    assert calls == [first_path, second_path, first_path]
    assert api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE[2] is not original


def test_failed_new_decode_keeps_previous_valid_source(monkeypatch, tmp_path):
    first_path = tmp_path / "first.nef"
    second_path = tmp_path / "second.nef"
    first_path.touch()
    second_path.touch()
    calls = []

    def read(path, *, preview):
        assert preview is False
        calls.append(Path(path))
        if Path(path) == second_path:
            raise RuntimeError("decode failed")
        return np.full((3, 4, 3), .25, dtype=np.float32), SimpleNamespace(
            source_kind="raw",
        )

    monkeypatch.setattr(api, "read_image", read)
    first_req = request(session="failed-switch", photo="a")
    source, _, _ = decoded(first_req, first_path)
    with pytest.raises(RuntimeError, match="decode failed"):
        decoded(request(session="failed-switch", photo="b"), second_path)

    previous = api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE
    assert previous is not None and previous[2] is source
    reused, _, hit = decoded(first_req, first_path)
    assert hit and reused is source
    assert calls == [first_path, second_path]


def test_oversized_previous_source_bypasses_retention(monkeypatch, tmp_path):
    first_path = tmp_path / "first.nef"
    second_path = tmp_path / "second.nef"
    first_path.touch()
    second_path.touch()
    monkeypatch.setattr(api, "_MAX_FULL_RESOLUTION_PREVIOUS_DECODED_BYTES", 1)
    calls = []

    def read(path, *, preview):
        assert preview is False
        calls.append(Path(path))
        return np.zeros((2, 2, 3), np.uint16), SimpleNamespace(source_kind="raw")

    monkeypatch.setattr(api, "read_image", read)
    first_req = request(session="oversized-previous", photo="a")
    decoded(first_req, first_path)
    decoded(request(session="oversized-previous", photo="b"), second_path)
    decoded(first_req, first_path)
    assert calls == [first_path, second_path, first_path]
    assert api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE is None


def test_previous_source_cleanup_is_session_scoped(monkeypatch, tmp_path):
    first_path = tmp_path / "first.nef"
    second_path = tmp_path / "second.nef"
    first_path.touch()
    second_path.touch()
    monkeypatch.setattr(api, "read_image", lambda *_a, **_k: (
        np.zeros((2, 2, 3), np.uint8), SimpleNamespace(source_kind="raw"),
    ))
    decoded(request(session="old-session", photo="a"), first_path)
    decoded(request(session="active-session", photo="b"), second_path)
    previous = api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE
    assert previous is not None and previous[0][0] == "old-session"

    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        api._clear_previous_full_resolution_decoded_source(session_id="other-session")
    assert api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE is previous
    active = api._FULL_RESOLUTION_ACTIVE_SOURCE_KEY
    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        api._clear_previous_full_resolution_decoded_source(session_id="old-session")
    assert api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE is None
    assert api._FULL_RESOLUTION_ACTIVE_SOURCE_KEY == active


def test_oversized_decode_is_immutable_but_not_retained(monkeypatch, tmp_path):
    path = tmp_path / "large.nef"
    path.touch()
    monkeypatch.setattr(api, "_MAX_FULL_RESOLUTION_DECODED_BYTES", 1)
    calls = []

    def read(_path, *, preview):
        calls.append(preview)
        return np.zeros((2, 2, 3), np.uint16), SimpleNamespace(source_kind="raw")

    monkeypatch.setattr(api, "read_image", read)
    first, _, hit = decoded(request(), path)
    assert not hit and not first.flags.writeable
    second, _, hit = decoded(request(), path)
    assert not hit and second is not first
    assert calls == [False, False]
    assert not api._FULL_RESOLUTION_DECODED_CACHE


def test_float32_renderer_gets_writable_working_copy(monkeypatch, tmp_path):
    path = tmp_path / "linear.nef"
    path.touch()
    source = np.full((4, 5, 3), .25, dtype=np.float32)
    original = source.copy()

    monkeypatch.setattr(
        api, "read_image",
        lambda *_args, **_kwargs: (source, SimpleNamespace(
            color_space="Linear sRGB", source_kind="raw",
        )),
    )

    def render(image, *_args, **_kwargs):
        assert image.flags.writeable
        image[0, 0] = .5
        return image

    monkeypatch.setattr(api, "_render_dehaze", render)
    monkeypatch.setattr(api, "_correct_enhanced_raw", lambda image, *_a, **_k: (image, None, False))
    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        result, _, _, decode_cache = api._full_resolution_display_base(
            request(), path, api.DehazeParams(), "cpu",
        )
    assert decode_cache == "miss"
    assert result.dtype == np.uint8 and not result.flags.writeable
    assert not source.flags.writeable
    np.testing.assert_array_equal(source, original)


def test_idle_expiration_and_shutdown_release_both_full_caches(tmp_path, monkeypatch):
    path = tmp_path / "timer.nef"
    path.touch()
    release_calls = []

    def released():
        assert not api._FULL_RESOLUTION_DECODED_CACHE
        assert not api._FULL_RESOLUTION_BASE_CACHE
        assert api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE is None
        release_calls.append(True)

    monkeypatch.setattr(api, "_release_unused_preview_memory", released)
    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        api._FULL_RESOLUTION_DECODED_CACHE[("s", "p", str(path), 1, 1)] = (
            np.zeros((1, 1, 3), np.uint8), object(),
        )
        api._FULL_RESOLUTION_BASE_CACHE[("s", "p", str(path), 1, 1, "base")] = (
            np.zeros((1, 1, 3), np.uint8), (),
        )
        old_key = ("old", "photo", str(path), path.stat().st_mtime_ns, path.stat().st_size)
        previous_image = np.zeros((1, 1, 3), np.uint8)
        previous_image.setflags(write=False)
        api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE = (
            old_key, api.enhanced_dng_source_identity(path), previous_image, object(),
        )
        api._touch_full_resolution_preview_cache()
        stale_generation = api._FULL_RESOLUTION_CACHE_GENERATION
        stale_timer = api._FULL_RESOLUTION_CACHE_TIMER
        assert stale_timer is not None and stale_timer.daemon
        stale_timer.cancel()
        api._touch_full_resolution_preview_cache()
        generation = api._FULL_RESOLUTION_CACHE_GENERATION
        timer = api._FULL_RESOLUTION_CACHE_TIMER
        assert generation > stale_generation
        assert timer is not None and timer.daemon
        assert timer.interval == 300.0
        timer.cancel()

    api._expire_full_resolution_preview_caches(stale_generation)
    assert api._FULL_RESOLUTION_DECODED_CACHE
    assert api._FULL_RESOLUTION_BASE_CACHE
    assert api._FULL_RESOLUTION_CACHE_TIMER is timer
    assert release_calls == []
    api._expire_full_resolution_preview_caches(generation)
    assert not api._FULL_RESOLUTION_DECODED_CACHE
    assert not api._FULL_RESOLUTION_BASE_CACHE
    assert api._FULL_RESOLUTION_CACHE_TIMER is None
    assert len(release_calls) == 1

    with api._FULL_RESOLUTION_PREVIEW_LOCK:
        api._FULL_RESOLUTION_DECODED_CACHE[("s", "p", str(path), 1, 1)] = (
            np.zeros((1, 1, 3), np.uint8), object(),
        )
        api._FULL_RESOLUTION_BASE_CACHE[("s", "p", str(path), 1, 1, "base")] = (
            np.zeros((1, 1, 3), np.uint8), (),
        )
        previous_key = ("older", "photo", str(path), path.stat().st_mtime_ns,
                        path.stat().st_size)
        previous_image = np.zeros((1, 1, 3), np.uint8)
        previous_image.setflags(write=False)
        api._FULL_RESOLUTION_PREVIOUS_DECODED_SOURCE = (
            previous_key, api.enhanced_dng_source_identity(path), previous_image, object(),
        )
        api._touch_full_resolution_preview_cache()
        timer = api._FULL_RESOLUTION_CACHE_TIMER
    api._shutdown_full_resolution_preview_cache()
    assert not api._FULL_RESOLUTION_DECODED_CACHE
    assert not api._FULL_RESOLUTION_BASE_CACHE
    assert len(release_calls) == 2
    assert timer is not None and timer.finished.is_set()


def test_allocator_release_hook_platform_bindings_and_failure_safety(monkeypatch):
    gc_calls = []
    monkeypatch.setattr(api.gc, "collect", lambda: gc_calls.append(True))

    monkeypatch.setattr(api.sys, "platform", "freebsd")
    monkeypatch.setattr(api.ctypes, "CDLL", lambda *_args: pytest.fail("unsupported platform should skip allocator hook"))
    api._release_unused_preview_memory()
    assert len(gc_calls) == 1

    class FakeFunction:
        argtypes = None
        restype = None

        def __init__(self):
            self.calls = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    darwin_function = FakeFunction()
    monkeypatch.setattr(api.sys, "platform", "darwin")
    monkeypatch.setattr(
        api.ctypes, "CDLL",
        lambda name: SimpleNamespace(malloc_zone_pressure_relief=darwin_function)
        if name is None else pytest.fail("expected process allocator handle"),
    )
    api._release_unused_preview_memory()
    assert darwin_function.calls == [(None, 0)]
    assert darwin_function.argtypes == [api.ctypes.c_void_p, api.ctypes.c_size_t]
    assert darwin_function.restype is api.ctypes.c_size_t

    linux_function = FakeFunction()
    monkeypatch.setattr(api.sys, "platform", "linux")
    monkeypatch.setattr(api.ctypes, "CDLL", lambda _name: SimpleNamespace(malloc_trim=linux_function))
    api._release_unused_preview_memory()
    assert linux_function.calls == [(0,)]
    assert linux_function.argtypes == [api.ctypes.c_size_t]
    assert linux_function.restype is api.ctypes.c_int

    # Missing symbols and loader failures are both safe no-ops after collection.
    monkeypatch.setattr(api.sys, "platform", "darwin")
    monkeypatch.setattr(api.ctypes, "CDLL", lambda _name: object())
    api._release_unused_preview_memory()
    monkeypatch.setattr(api.ctypes, "CDLL", lambda _name: (_ for _ in ()).throw(OSError("unavailable")))
    api._release_unused_preview_memory()
    assert len(gc_calls) == 5


def test_jpeg_preview_cache_enforces_byte_and_entry_limits():
    cache = {}
    api._insert_bounded_jpeg_preview(
        cache, ("first",), (b"1234", 1), max_entries=2,
        max_bytes=6, payload_index=0,
    )
    api._insert_bounded_jpeg_preview(
        cache, ("second",), (b"5678", 2), max_entries=2,
        max_bytes=6, payload_index=0,
    )
    assert list(cache) == [("second",)]
    api._insert_bounded_jpeg_preview(
        cache, ("too-large",), (b"1234567", 3), max_entries=2,
        max_bytes=6, payload_index=0,
    )
    assert list(cache) == [("second",)]

    ricoh = {}
    api._insert_bounded_jpeg_preview(
        ricoh, ("first",), b"1234", max_entries=1, max_bytes=8,
    )
    api._insert_bounded_jpeg_preview(
        ricoh, ("second",), b"5678", max_entries=1, max_bytes=8,
    )
    assert list(ricoh) == [("second",)]
