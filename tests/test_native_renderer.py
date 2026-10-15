from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import native_renderer  # noqa: E402


class _Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class _FakeLibrary:
    def __init__(self, *, create_status: int = 0):
        self.create_status = create_status
        self.output = None
        self.render_calls = []
        self.preview_render_calls = []
        self.upload_calls = []
        self.preview_images = {}
        self.outputs = {}
        self.create_count = 0
        self.destroyed = 0
        self.im_renderer_create = _Function(self._create)
        self.im_renderer_destroy = _Function(self._destroy)
        self.im_renderer_last_error = _Function(lambda _renderer: b"fake backend unavailable")
        self.im_renderer_backend_name = _Function(lambda _renderer: b"Fake GPU")
        self.im_renderer_render_full = _Function(self._render_full)
        self.im_renderer_upload_preview_image = _Function(self._upload_preview)
        self.im_renderer_render = _Function(self._render_preview)
        self.im_renderer_get_output_size = _Function(self._get_output_size)
        self.im_renderer_copy_output = _Function(self._copy_output)

    def _create(self, _backend, output_handle):
        self.create_count += 1
        if self.create_status == 0:
            ctypes.cast(output_handle, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
                122 + self.create_count
            )
        return self.create_status

    def _destroy(self, _renderer):
        self.destroyed += 1

    @staticmethod
    def _handle_id(renderer):
        return renderer.value if hasattr(renderer, "value") else renderer

    def _render_full(self, _renderer, width, height, input_pointer, value_count,
                     dehaze_pointer, basic_pointer):
        source = np.ctypeslib.as_array(input_pointer, shape=(value_count,)).copy()
        dehaze = ctypes.cast(
            dehaze_pointer, ctypes.POINTER(native_renderer._DehazeParams)
        ).contents
        basic = ctypes.cast(
            basic_pointer, ctypes.POINTER(native_renderer._BasicParams)
        ).contents
        self.render_calls.append({
            "width": width,
            "height": height,
            "input": source,
            "dehaze": tuple(getattr(dehaze, name) for name in native_renderer._DEHAZE_FIELDS),
            "basic": tuple(getattr(basic, name) for name in native_renderer._BASIC_FIELDS),
        })
        self.output = source.reshape(height, width, 3)
        self.outputs[self._handle_id(_renderer)] = self.output
        return 0

    def _upload_preview(self, renderer, width, height, input_pointer, value_count):
        source = np.ctypeslib.as_array(input_pointer, shape=(value_count,)).copy()
        handle = self._handle_id(renderer)
        self.upload_calls.append({
            "handle": handle,
            "width": width,
            "height": height,
            "input": source,
        })
        self.preview_images[handle] = source.reshape(height, width, 3)
        return 0

    def _render_preview(self, renderer, level, dehaze_pointer, basic_pointer):
        handle = self._handle_id(renderer)
        source = self.preview_images[handle]
        dehaze = ctypes.cast(
            dehaze_pointer, ctypes.POINTER(native_renderer._DehazeParams)
        ).contents
        basic = ctypes.cast(
            basic_pointer, ctypes.POINTER(native_renderer._BasicParams)
        ).contents
        self.preview_render_calls.append({
            "handle": handle,
            "level": level,
            "dehaze": tuple(getattr(dehaze, name) for name in native_renderer._DEHAZE_FIELDS),
            "basic": tuple(getattr(basic, name) for name in native_renderer._BASIC_FIELDS),
        })
        divisor = 1 << level
        height = max(1, (source.shape[0] + divisor - 1) // divisor)
        width = max(1, (source.shape[1] + divisor - 1) // divisor)
        self.output = np.resize(source, (height, width, 3))
        self.outputs[handle] = self.output
        return 0

    def _get_output_size(self, renderer, width_pointer, height_pointer, count_pointer):
        output = self.outputs[self._handle_id(renderer)]
        height, width, _ = output.shape
        ctypes.cast(width_pointer, ctypes.POINTER(ctypes.c_uint32))[0] = width
        ctypes.cast(height_pointer, ctypes.POINTER(ctypes.c_uint32))[0] = height
        ctypes.cast(count_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = output.size
        return 0

    def _copy_output(self, renderer, output_pointer, value_count):
        destination = np.ctypeslib.as_array(output_pointer, shape=(value_count,))
        destination[:] = self.outputs[self._handle_id(renderer)].reshape(-1)
        return 0


@pytest.fixture
def fake_native(monkeypatch):
    native_renderer._clear_preview_cache()
    library = _FakeLibrary()
    monkeypatch.setattr(native_renderer, "_cached_library", library)
    monkeypatch.setattr(native_renderer, "_cached_library_path", "/test/libimprint_renderer")
    yield library
    native_renderer._clear_preview_cache()


def _dehaze_params(**overrides):
    values = {name: 0.0 for name in native_renderer._DEHAZE_FIELDS}
    values["strength"] = 0.5
    values.update(overrides)
    return values


def _basic_params(**overrides):
    values = {name: 0.0 for name in native_renderer._BASIC_FIELDS}
    values.update(overrides)
    return values


def test_native_dehaze_expands_uint8_and_zeroes_basic_stage(fake_native):
    source = np.array([[[0, 1, 127], [128, 254, 255]]], dtype=np.uint8)
    before = source.copy()

    result = native_renderer.native_dehaze(source, _dehaze_params(local_contrast=0.2))

    assert result.dtype == np.uint8
    assert np.array_equal(result, source)
    assert np.array_equal(source, before)
    call = fake_native.render_calls[0]
    assert call["input"].tolist() == (source.astype(np.uint16) * 257).reshape(-1).tolist()
    assert call["dehaze"][0] == pytest.approx(0.5)
    assert call["dehaze"][3] == pytest.approx(0.2)
    assert call["basic"] == (0.0,) * len(native_renderer._BASIC_FIELDS)
    assert fake_native.destroyed == 1


def test_native_basic_nonzero_exposure_uses_corrected_cpu_fallback(fake_native):
    full = np.arange(2 * 4 * 3, dtype=np.uint16).reshape(2, 4, 3) * 113
    source = full[:, ::2, :]
    before = source.copy()

    result = native_renderer.native_basic(source, _basic_params(exposure=1.0))

    assert result.dtype == np.uint16
    normalized = source.astype(np.float32) / 65535.0
    linear = np.where(
        normalized <= 0.04045,
        normalized / 12.92,
        ((normalized + 0.055) / 1.055) ** 2.4,
    ) * 2.0
    expected = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    expected = np.clip(np.rint(expected * 65535.0), 0, 65535).astype(np.uint16)
    assert np.max(np.abs(result.astype(np.int32) - expected.astype(np.int32))) <= 1
    assert np.array_equal(source, before)
    assert not fake_native.render_calls


def test_native_basic_zero_exposure_keeps_gpu_path(fake_native):
    source = np.array([[[20, 120, 200]]], dtype=np.uint8)
    result = native_renderer.native_basic(source, _basic_params(saturation=-4))

    assert np.array_equal(result, source)
    assert len(fake_native.render_calls) == 1
    assert fake_native.render_calls[0]["basic"][0] == 0.0
    assert fake_native.render_calls[0]["basic"][7] == pytest.approx(-4.0)


def test_status_requires_successful_renderer_creation(fake_native):
    assert native_renderer.get_native_status() == {
        "available": True,
        "backend": "Fake GPU",
    }

    unavailable = _FakeLibrary(create_status=2)
    native_renderer._cached_library = unavailable
    status = native_renderer.get_native_status()
    assert status == {"available": False, "backend": None}


@pytest.mark.parametrize("bad", [np.zeros((2, 2), dtype=np.uint8),
                                   np.zeros((2, 2, 4), dtype=np.uint8),
                                   np.zeros((2, 2, 3), dtype=np.float32)])
def test_rejects_unsupported_input_shapes_and_dtypes(fake_native, bad):
    with pytest.raises((TypeError, ValueError)):
        native_renderer.native_dehaze(bad, _dehaze_params())
    assert not fake_native.render_calls


def test_rejects_invalid_parameters_before_calling_native(fake_native):
    invalid = _basic_params(exposure=float("nan"))
    with pytest.raises(ValueError, match="exposure"):
        native_renderer.native_basic(np.zeros((1, 1, 3), dtype=np.uint8), invalid)
    assert not fake_native.render_calls


def test_cached_native_preview_uploads_once_and_renders_each_level(fake_native):
    source = np.arange(5 * 3 * 3, dtype=np.uint8).reshape(3, 5, 3)
    factory_calls = []

    def image_factory():
        factory_calls.append(True)
        return source, {"width": 5, "height": 3}

    key = ("session-a", "photo-a", 12.5, 2048, 1600)
    l0, metadata = native_renderer.native_dehaze_preview(
        key, image_factory, _dehaze_params(strength=0.1), 0
    )
    l1, metadata_l1 = native_renderer.native_dehaze_preview(
        key, image_factory, _dehaze_params(strength=0.6), 1
    )
    l2, metadata_l2 = native_renderer.native_dehaze_preview(
        key, image_factory, _dehaze_params(strength=0.9), 2
    )

    assert factory_calls == [True]
    assert len(fake_native.upload_calls) == 1
    assert fake_native.upload_calls[0]["input"].tolist() == (
        source.astype(np.uint16) * 257
    ).reshape(-1).tolist()
    assert [call["level"] for call in fake_native.preview_render_calls] == [0, 1, 2]
    assert [call["dehaze"][0] for call in fake_native.preview_render_calls] == pytest.approx(
        [0.1, 0.6, 0.9]
    )
    assert all(call["basic"] == (0.0,) * len(native_renderer._BASIC_FIELDS)
               for call in fake_native.preview_render_calls)
    assert l0.shape == (3, 5, 3)
    assert l1.shape == (2, 3, 3)
    assert l2.shape == (1, 2, 3)
    assert l0.dtype == l1.dtype == l2.dtype == np.uint8
    assert metadata_l1 is metadata_l2 is metadata


def test_cached_native_preview_uses_distinct_handles_and_evicts_lru(fake_native, monkeypatch):
    monkeypatch.setattr(native_renderer, "_PREVIEW_CACHE_LIMIT", 2)
    calls = {"a": 0, "b": 0, "c": 0}

    def factory(name):
        def create():
            calls[name] += 1
            return np.full((2, 2, 3), calls[name], dtype=np.uint16), {"photo": name}
        return create

    key_a, key_b, key_c = (("s", name, 1, 2, 2) for name in ("a", "b", "c"))
    params = _dehaze_params()
    _, meta_a = native_renderer.native_dehaze_preview(key_a, factory("a"), params, 0)
    _, meta_b = native_renderer.native_dehaze_preview(key_b, factory("b"), params, 0)
    native_renderer.native_dehaze_preview(key_a, factory("a"), params, 1)
    native_renderer.native_dehaze_preview(key_c, factory("c"), params, 2)

    assert len(fake_native.upload_calls) == 3
    assert calls == {"a": 1, "b": 1, "c": 1}
    assert fake_native.destroyed == 1
    assert meta_a == {"photo": "a"}
    assert meta_b == {"photo": "b"}

    _, meta_b_again = native_renderer.native_dehaze_preview(
        key_b, factory("b"), params, 0
    )
    assert calls["b"] == 2
    assert len(fake_native.upload_calls) == 4
    assert fake_native.destroyed == 2
    assert meta_b_again == {"photo": "b"}


def test_cached_native_preview_keeps_concurrent_photo_handles_separate(fake_native):
    barrier = threading.Barrier(3)
    factory_calls = {"a": 0, "b": 0}
    params = _dehaze_params(strength=0.4)

    def render(name, value):
        key = ("session", name, 10, 12, 20)

        def image_factory():
            factory_calls[name] += 1
            return np.full((4, 6, 3), value, dtype=np.uint16), {"photo": name}

        barrier.wait()
        return native_renderer.native_dehaze_preview(key, image_factory, params, 0)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(render, "a", 10000),
            pool.submit(render, "a", 20000),
            pool.submit(render, "b", 50000),
        ]
        results = [future.result() for future in futures]

    assert factory_calls == {"a": 1, "b": 1}
    assert len(fake_native.upload_calls) == 2
    # Either same-photo caller may win the first decode; both must reuse it.
    first_value = int(results[0][0][0, 0, 0])
    assert first_value in (10000, 20000)
    assert np.all(results[0][0] == first_value)
    assert np.all(results[1][0] == first_value)
    assert np.all(results[2][0] == 50000)
    assert [metadata["photo"] for _, metadata in results] == ["a", "a", "b"]


def test_slow_first_decode_does_not_block_another_photo(fake_native):
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    source = np.full((4, 6, 3), 12000, dtype=np.uint16)

    def first_factory():
        first_started.set()
        assert release_first.wait(timeout=3)
        return source, {"photo": "first"}

    def second_factory():
        second_started.set()
        return source, {"photo": "second"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(native_renderer.native_dehaze_preview,
                            ("session", "first"), first_factory, _dehaze_params(), 2)
        assert first_started.wait(timeout=3)
        second = pool.submit(native_renderer.native_dehaze_preview,
                             ("session", "second"), second_factory, _dehaze_params(), 2)
        try:
            assert second_started.wait(timeout=1)
            assert second.result(timeout=3)[1] == {"photo": "second"}
        finally:
            release_first.set()
        assert first.result(timeout=3)[1] == {"photo": "first"}


def test_concurrent_first_decode_of_same_photo_is_shared(fake_native):
    first_started = threading.Event()
    release_first = threading.Event()
    calls = []
    source = np.full((4, 6, 3), 12000, dtype=np.uint16)

    def image_factory():
        calls.append(True)
        first_started.set()
        assert release_first.wait(timeout=3)
        return source, {"photo": "shared"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(native_renderer.native_dehaze_preview,
                            ("session", "shared"), image_factory, _dehaze_params(), 2)
        assert first_started.wait(timeout=3)
        second = pool.submit(native_renderer.native_dehaze_preview,
                             ("session", "shared"), image_factory, _dehaze_params(), 2)
        release_first.set()
        assert first.result(timeout=3)[1] == {"photo": "shared"}
        assert second.result(timeout=3)[1] == {"photo": "shared"}
    assert calls == [True]
