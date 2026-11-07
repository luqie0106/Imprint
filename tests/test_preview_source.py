"""隔离 ML 依赖，验证实际 API 函数的源文件版本与会话边界。"""

import ast
from collections import OrderedDict
import concurrent.futures
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, content=None, status_code=200, headers=None, media_type=None):
        self.content = content
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.media_type = media_type


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    for node in functions:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def api(tmp_path):
    source = tmp_path / "photo.png"
    source.write_bytes(b"original")
    namespace = {
        "Path": Path, "hashlib": hashlib, "json": json, "np": np,
        "JSONResponse": Response, "Response": Response,
        "_ENHANCE_LOCK": threading.RLock(),
        "_ENHANCE_SESSIONS": {"session": {"files": {"photo": source}}},
    }
    load_functions(ROOT / "src/image_io.py", ["enhanced_dng_source_identity"], namespace)
    load_functions(ROOT / "src/app_api.py", [
        "_preview_source_version", "get_enhance_source_version", "create_enhance_preview",
        "_create_enhance_preview", "_cached_display_preview", "_insert_bounded_jpeg_preview",
    ], namespace)
    namespace["_create_enhance_preview"] = lambda req: Response(b"jpeg")
    namespace["source"] = source
    namespace["req"] = SimpleNamespace(session_id="session", photo_id="photo")
    return namespace


def replace_source(source):
    before = source.stat()
    replacement = source.with_suffix(".replacement")
    replacement.write_bytes(b"modified")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, source)
    assert source.stat().st_size == before.st_size
    assert source.stat().st_mtime_ns == before.st_mtime_ns


def test_version_changes_when_same_size_and_mtime_file_is_replaced(api):
    version = api["_preview_source_version"](api["source"])
    assert len(version) == 64
    assert str(api["source"]) not in version
    assert version == api["_preview_source_version"](api["source"])
    replace_source(api["source"])
    assert version != api["_preview_source_version"](api["source"])


@pytest.mark.parametrize("session,photo", [("unknown", "photo"), ("session", "unknown")])
def test_source_version_rejects_unknown_ids_without_reading_files(api, session, photo):
    api["_preview_source_version"] = lambda path: pytest.fail("不应读取未授权的路径")
    assert api["get_enhance_source_version"](session, photo).status_code == 404


def test_source_version_is_scoped_and_not_http_cached(api):
    result = api["get_enhance_source_version"]("session", "photo")
    assert result.status_code == 200
    assert result.content == {"version": api["_preview_source_version"](api["source"])}
    assert result.headers["Cache-Control"] == "no-store"


def test_missing_source_returns_safe_error(api):
    api["source"].unlink()
    for result in [api["get_enhance_source_version"]("session", "photo"), api["create_enhance_preview"](api["req"])]:
        assert result.status_code == 404
        assert str(api["source"]) not in str(result.content)


def test_source_version_rechecks_session_after_stat(api):
    original = api["_preview_source_version"]

    def evict(path):
        version = original(path)
        api["_ENHANCE_SESSIONS"].clear()
        return version

    api["_preview_source_version"] = evict
    assert api["get_enhance_source_version"]("session", "photo").status_code == 404


def test_preview_response_carries_source_version(api):
    result = api["create_enhance_preview"](api["req"])
    assert result.status_code == 200
    assert result.headers["X-Source-Version"] == api["_preview_source_version"](api["source"])


def test_preview_discards_pixels_if_source_changes_during_render(api):
    def render(req):
        replace_source(api["source"])
        return Response(b"old-jpeg")

    api["_create_enhance_preview"] = render
    result = api["create_enhance_preview"](api["req"])
    assert result.status_code == 409
    assert "X-Source-Version" not in result.headers


def test_preview_discards_pixels_if_session_is_evicted_during_render(api):
    def render(req):
        api["_ENHANCE_SESSIONS"].clear()
        return Response(b"old-jpeg")

    api["_create_enhance_preview"] = render
    assert api["create_enhance_preview"](api["req"]).status_code == 404


@pytest.mark.parametrize("status", [204, 400, 422, 500])
def test_preview_preserves_non_image_responses(api, status):
    expected = Response(status_code=status)
    api["_create_enhance_preview"] = lambda req: expected
    assert api["create_enhance_preview"](api["req"]) is expected


def test_display_cache_reloads_replaced_source(api):
    decodes = []

    def decode(session, photo, path, edge):
        decodes.append(path.read_bytes())
        pixels = np.full((2, 2, 3), len(decodes), dtype=np.uint8)
        return pixels, SimpleNamespace(source_kind="raw", color_space="sRGB")

    api.update({
        "_DISPLAY_PREVIEW_LOCK": threading.RLock(),
        "_DISPLAY_PREVIEW_CACHE": OrderedDict(),
        "_MAX_DISPLAY_PREVIEW_BYTES": 1024,
        "LENS_PREVIEW_VERSION": "test",
        "_cached_processing_preview": decode,
        "_display_rgb8": lambda image, **kwargs: image.copy(),
    })
    first = api["_cached_display_preview"]("session", "photo", api["source"], 1800)
    assert api["_cached_display_preview"]("session", "photo", api["source"], 1800) is first
    replace_source(api["source"])
    second = api["_cached_display_preview"]("session", "photo", api["source"], 1800)
    assert len(decodes) == 2
    assert not np.array_equal(first, second)


def test_jpeg_cache_reloads_replaced_source(api):
    load_functions(ROOT / "src/app_api.py", ["_create_enhance_preview"], api)
    requests = []

    def display(session, photo, path, edge, **kwargs):
        requests.append(path.read_bytes())
        return np.full((2, 2, 3), len(requests), dtype=np.uint8)

    api.update({
        "_register_preview_generation": lambda req: True,
        "_assert_preview_current": lambda req: None,
        "_render_mode": lambda *args: "cpu", "_basic_mode": lambda *args: "python",
        "resolve_profile": lambda *args: None,
        "_ENHANCE_PREVIEW_CACHE": {}, "_MAX_ENHANCE_PREVIEW_CACHE_BYTES": 1024,
        "_cached_display_preview": display, "_encode_preview": lambda image: image.tobytes(),
        **{name: "test" for name in ["DEHAZE_ALGORITHM_VERSION", "AUTO_EXPOSURE_ALGORITHM_VERSION", "LENS_PREVIEW_VERSION", "BASIC_PREVIEW_VERSION", "PROFILE_PREVIEW_VERSION"]},
    })
    req = SimpleNamespace(
        session_id="session", photo_id="photo", full_resolution=False,
        preview_level=0, preview_cancel_only=False, params=SimpleNamespace(to_params=lambda: SimpleNamespace(strength=0.0)),
        basic_params=SimpleNamespace(values=lambda: {"exposure": 0.0}),
        ricoh_preset_id=None, mode="original", render_backend="cpu", use_gpu=False,
        basic_backend="python", ricoh_backend="python", max_edge=1800,
        algorithm="physical", auto_mode=False, auto_exposure=False,
        nonlocal_mode="off", color_manage_srgb=True,
    )
    first = api["create_enhance_preview"](req)
    assert first.status_code == 200
    assert api["create_enhance_preview"](req).content == first.content
    replace_source(api["source"])
    second = api["create_enhance_preview"](req)
    assert second.status_code == 200
    assert second.content != first.content
    assert second.headers["X-Source-Version"] != first.headers["X-Source-Version"]
    assert len(requests) == 2


def test_profile_cache_cannot_relabel_a_future_from_the_previous_source(api):
    load_functions(ROOT / "src/app_api.py", [
        "_cached_dehazed_display_preview", "_profile_display_base",
        "_camera_profile_preview_source_key", "_prepared_dehaze_preview_scale",
    ], api)
    profile = SimpleNamespace(fingerprint=("test-profile",))
    metadata = SimpleNamespace(source_kind="raw", color_space="Linear sRGB")
    decoded_cameras = []
    transferred_cameras = []
    should_replace = True

    def pixels():
        value = 11 if api["source"].read_bytes() == b"original" else 22
        return np.full((2, 2, 3), value, dtype=np.uint16)

    def start_camera(*args, **kwargs):
        camera = pixels()
        decoded_cameras.append(int(camera[0, 0, 0]))
        future = concurrent.futures.Future()
        future.set_result((camera, (1.0, 1.0, 1.0)))
        return None, future

    def render(image, *args, **kwargs):
        nonlocal should_replace
        if should_replace:
            should_replace = False
            replace_source(api["source"])
        return image

    def transfer(camera, *args, **kwargs):
        transferred_cameras.append(int(camera[0, 0, 0]))
        return camera.copy()

    class Superseded(Exception):
        pass

    api.update({
        "concurrent": concurrent, "logging": logging,
        "_PreviewSuperseded": Superseded,
        "_DISPLAY_PREVIEW_LOCK": threading.RLock(),
        "_CAMERA_PROFILE_SOURCE_CACHE": OrderedDict(),
        "_DEHAZED_PREVIEW_CACHE": OrderedDict(),
        "_DEHAZED_PREVIEW_STATUS_CACHE": OrderedDict(),
        "_MAX_CAMERA_PROFILE_SOURCE_BYTES": 1024,
        "_MAX_DEHAZED_PREVIEW_BYTES": 1024,
        "_cached_processing_preview": lambda *args: (pixels(), metadata),
        "_cached_prepared_dehaze_preview": lambda *args: (pixels(), metadata, 0.0, "off"),
        "_start_camera_profile_source_decode": start_camera,
        "_finish_camera_profile_source_decode": lambda executor: None,
        "resolve_nonlocal_mode": lambda mode: mode or "off",
        "_render_dehaze": render,
        "_nonlocal_preview_status": lambda *args: ("off", ""),
        "_set_preview_status": lambda *args: None,
        "_remember_dehazed_preview_status": lambda key, status: api["_DEHAZED_PREVIEW_STATUS_CACHE"].__setitem__(key, status),
        "_linear_float_to_uint16": lambda image, **kwargs: image.astype(np.uint16),
        "_correct_profile_inputs": lambda images, *args, **kwargs: [(image, None, False) for image in images],
        "_correct_enhanced_raw": lambda image, *args, **kwargs: (image, None, False),
        "_camera_profile_render_backend": lambda backend: backend,
        "transfer_enhancement": transfer,
        "render_profile": lambda image, *args, **kwargs: image.copy(),
        "get_last_profile_backend": lambda: "test",
        "_display_rgb8": lambda image, **kwargs: image.astype(np.uint8),
        **{name: "test" for name in ["DEHAZE_ALGORITHM_VERSION", "AUTO_EXPOSURE_ALGORITHM_VERSION", "LENS_PREVIEW_VERSION", "BASIC_PREVIEW_VERSION", "PROFILE_PREVIEW_VERSION"]},
    })

    def preview(req):
        image = api["_cached_dehazed_display_preview"](
            "session", "photo", api["source"], 1800,
            SimpleNamespace(strength=0.0), "cpu", True, profile=profile,
        )
        return Response(image.tobytes())

    api["_create_enhance_preview"] = preview
    first = api["create_enhance_preview"](api["req"])
    assert first.status_code == 409
    assert api["_CAMERA_PROFILE_SOURCE_CACHE"] == {}

    second = api["create_enhance_preview"](api["req"])
    assert second.status_code == 200
    assert second.headers["X-Source-Version"] == api["_preview_source_version"](api["source"])
    assert np.frombuffer(second.content, dtype=np.uint16).tolist() == [22] * 12
    assert decoded_cameras == [11, 22]
    assert transferred_cameras == [22]
    assert len(api["_CAMERA_PROFILE_SOURCE_CACHE"]) == 1
    cached = next(iter(api["_CAMERA_PROFILE_SOURCE_CACHE"].values()))
    assert np.all(cached[0] == 22)
    assert np.all(cached[1] == 22)
