from __future__ import annotations

import ctypes
import gc
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_api  # noqa: E402
import native_dense  # noqa: E402
import native_renderer  # noqa: E402
import native_shared  # noqa: E402


class _Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class _SharedLibrary:
    def __init__(self, *, shared_abi=True):
        self.buffers = {}
        self.destroyed_buffers = []
        self.destroyed_renderers = []
        self.next_handle = 100
        self.allocations = []
        self.transfer_calls = []
        self.transfer_inputs = []
        self.profile_shared_source_calls = []
        self.profile_copy_source_calls = 0
        self.profile_render_count = 0
        self.im_renderer_destroy = _Function(self._destroy_renderer)
        if shared_abi:
            self.im_renderer_create_shared_buffer = _Function(self._create_shared)
            self.im_shared_buffer_data = _Function(self._shared_data)
            self.im_shared_buffer_size = _Function(self._shared_size)
            self.im_shared_buffer_destroy = _Function(self._destroy_shared)
        self.im_renderer_set_camera_profile_transfer_source = _Function(lambda *_args: 0)
        self.im_renderer_supports_camera_profile_transfer = _Function(lambda _renderer: 1)
        self.im_renderer_render_camera_profile_transfer = _Function(self._copy_transfer)
        self.im_renderer_render_camera_profile_transfer_shared = _Function(self._shared_transfer)
        self.im_renderer_set_camera_profile_source = _Function(self._profile_copy_source)
        self.im_renderer_set_camera_profile_source_shared = _Function(self._profile_shared_source)
        self.im_renderer_render_camera_profile = _Function(self._profile_copy_render)
        self.im_renderer_render_camera_profile_shared = _Function(self._profile_shared_render)

    @staticmethod
    def _id(handle):
        return handle.value if isinstance(handle, ctypes.c_void_p) else int(handle)

    def _destroy_renderer(self, renderer):
        self.destroyed_renderers.append(self._id(renderer))

    def _create_shared(self, _renderer, byte_count, output):
        self.next_handle += 1
        handle = self.next_handle
        storage = ctypes.create_string_buffer(byte_count)
        self.buffers[handle] = storage
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(handle)
        self.allocations.append((handle, byte_count))
        return 0

    def _shared_data(self, handle):
        return ctypes.addressof(self.buffers[self._id(handle)])

    def _shared_size(self, handle):
        return ctypes.sizeof(self.buffers[self._id(handle)])

    def _destroy_shared(self, handle):
        handle_id = self._id(handle)
        self.destroyed_buffers.append(handle_id)
        self.buffers.pop(handle_id, None)

    def _copy_transfer(self, _renderer, _processed, _count, _inverse, _inverse_count,
                       output, output_count):
        target = np.ctypeslib.as_array(output, shape=(output_count,))
        target.fill(123)
        return 0

    def _shared_transfer(self, _renderer, processed, _inverse, _inverse_count, output):
        processed_id = self._id(processed)
        output_id = self._id(output)
        self.transfer_calls.append((processed_id, output_id))
        self.transfer_inputs.append(
            np.ndarray((2, 3, 3), dtype=np.uint16,
                       buffer=self.buffers[processed_id]).copy()
        )
        target = np.ndarray((2, 3, 3), dtype=np.uint16, buffer=self.buffers[output_id])
        target.fill(len(self.transfer_calls) * 1000 + 7)
        return 0

    def _profile_copy_source(self, *_args):
        self.profile_copy_source_calls += 1
        return 0

    def _profile_shared_source(self, _renderer, _width, _height, source):
        self.profile_shared_source_calls.append(self._id(source))
        return 0

    def _profile_copy_render(self, _renderer, _constants, _constant_count, _look, _look_count,
                             _hue, _saturation, _value, _encoding, _tone, _tone_count,
                             output, output_count):
        np.ctypeslib.as_array(output, shape=(output_count,)).fill(19)
        return 0

    def _profile_shared_render(self, _renderer, _constants, _constant_count, _look, _look_count,
                               _hue, _saturation, _value, _encoding, _tone, _tone_count, output):
        self.profile_render_count += 1
        output_id = self._id(output)
        output_bytes = ctypes.sizeof(self.buffers[output_id])
        target = np.ctypeslib.as_array(
            ctypes.cast(self._shared_data(output), ctypes.POINTER(ctypes.c_uint8)),
            shape=(output_bytes,),
        )
        target.fill(30 + self.profile_render_count)
        return 0


@pytest.fixture
def shared_host(monkeypatch):
    library = _SharedLibrary()
    monkeypatch.setattr(native_shared.sys, "platform", "darwin")
    monkeypatch.setattr(native_renderer.sys, "platform", "darwin")
    monkeypatch.setattr(native_renderer, "_get_library", lambda: (library, "fake"))
    monkeypatch.setattr(native_renderer, "_create_renderer",
                        lambda _library: (ctypes.c_void_p(77), "Metal"))
    monkeypatch.setenv("IMPRINT_NATIVE_SHARED_MEMORY", "1")
    monkeypatch.setenv("IMPRINT_NATIVE_CAMERA_PROFILE_TRANSFER", "1")
    native_renderer.clear_camera_profile_gpu_cache()
    native_shared.clear_shared_allocator_cache()
    yield library
    native_renderer.clear_camera_profile_gpu_cache()


def test_policy_and_missing_or_disabled_abi(monkeypatch):
    monkeypatch.setattr(native_shared.sys, "platform", "darwin")
    monkeypatch.setenv("IMPRINT_NATIVE_SHARED_MEMORY", "0")
    assert not native_shared.shared_memory_enabled()
    assert not native_shared.shared_memory_eligible((2, 3, 3))
    with pytest.raises(native_renderer.NativeRendererError, match="disabled"):
        native_shared.allocate_shared_rgb((2, 3, 3))

    monkeypatch.setenv("IMPRINT_NATIVE_SHARED_MEMORY", "1")
    assert native_shared.shared_memory_eligible((2, 3, 3))
    library = _SharedLibrary(shared_abi=False)
    monkeypatch.setattr(native_renderer, "_get_library", lambda: (library, "fake"))
    monkeypatch.setattr(native_renderer, "_create_renderer",
                        lambda _library: (ctypes.c_void_p(88), "Metal"))
    with pytest.raises(native_renderer.NativeRendererError, match="ABI is unavailable"):
        native_shared.allocate_shared_rgb((2, 3, 3))


def test_default_size_threshold_and_env_small_image_override(monkeypatch):
    monkeypatch.setattr(native_shared.sys, "platform", "darwin")
    monkeypatch.setenv("IMPRINT_NATIVE_SHARED_MEMORY", "")
    assert not native_shared.shared_memory_eligible((999, 1000, 3))
    assert native_shared.shared_memory_eligible((1000, 1000, 3))
    monkeypatch.setenv("IMPRINT_NATIVE_SHARED_MEMORY", "1")
    assert native_shared.shared_memory_eligible((3, 4, 3))
    monkeypatch.setattr(native_shared.sys, "platform", "win32")
    assert not native_shared.shared_memory_eligible((3, 4, 3))


def test_owner_lifetime_survives_slices_and_allocator_clear(shared_host):
    array = native_shared.allocate_shared_rgb((2, 3, 3))
    owner = native_shared.shared_buffer_owner(array)
    handle = owner.handle.value
    array[...] = np.arange(array.size, dtype=np.uint16).reshape(array.shape)
    partial = array[1:, :, :]
    readonly = array.view()
    readonly.setflags(write=False)
    assert native_shared.shared_buffer_owner(readonly) is owner
    assert native_shared.shared_buffer_owner(partial) is None
    assert native_shared.shared_buffer_owner(np.zeros_like(array)) is None

    native_shared.clear_shared_allocator_cache()
    assert shared_host.destroyed_renderers == [77]
    assert np.array_equal(partial, array[1:, :, :])
    del array, readonly, owner
    gc.collect()
    assert handle not in shared_host.destroyed_buffers
    del partial
    gc.collect()
    assert shared_host.destroyed_buffers.count(handle) == 1


def test_transfer_uses_owned_buffers_and_returns_distinct_outputs(shared_host):
    camera = np.zeros((2, 3, 3), dtype=np.uint16)
    reference = np.ones_like(camera)
    processed = np.full_like(camera, 23)

    first, _ = native_renderer.native_camera_profile_transfer(
        camera, reference, processed, np.eye(3, dtype=np.float32))
    first_snapshot = first.copy()
    second, _ = native_renderer.native_camera_profile_transfer(
        camera, reference, processed + 1, np.eye(3, dtype=np.float32))

    assert native_shared.shared_buffer_owner(first) is not None
    assert native_shared.shared_buffer_owner(second) is not None
    assert first.ctypes.data != second.ctypes.data
    assert np.all(first == 1007)
    assert np.all(second == 2007)
    assert np.array_equal(first, first_snapshot)
    assert len(shared_host.transfer_calls) == 2
    first_input, first_output = shared_host.transfer_calls[0]
    assert first_input != first_output
    assert first_output in shared_host.buffers
    assert np.array_equal(shared_host.transfer_inputs[0], processed)


def test_existing_shared_transfer_input_is_bound_directly(shared_host):
    camera = np.zeros((2, 3, 3), dtype=np.uint16)
    reference = np.ones_like(camera)
    processed = native_shared.allocate_shared_rgb(camera.shape)
    processed.fill(37)
    expected_handle = native_shared.shared_buffer_owner(processed).handle.value

    output, _ = native_renderer.native_camera_profile_transfer(
        camera, reference, processed, np.eye(3, dtype=np.float32))

    assert shared_host.transfer_calls[0][0] == expected_handle
    assert native_shared.shared_buffer_owner(output) is not None


def test_profile_binds_shared_source_and_keeps_outputs_independent(shared_host):
    source = native_shared.allocate_shared_rgb((2, 3, 3))
    source.fill(1200)
    source.setflags(write=False)
    source_handle = native_shared.shared_buffer_owner(source).handle.value
    matrix = np.eye(3, dtype=np.float32)
    white = np.ones(3, dtype=np.float32)
    look = np.zeros((2, 4, 3, 3), dtype=np.float32)
    tone = np.array([[0., 0.], [1., 1.]], dtype=np.float32)

    first, _ = native_renderer.native_camera_profile(
        source, matrix, white, 1.0, look, (4, 3, 2), 0, tone, matrix)
    first_snapshot = first.copy()
    second, _ = native_renderer.native_camera_profile(
        source, matrix, white, 1.0, look, (4, 3, 2), 0, tone, matrix)

    assert shared_host.profile_shared_source_calls == [source_handle]
    assert shared_host.profile_copy_source_calls == 0
    assert np.all(first == 31)
    assert np.all(second == 32)
    assert first.ctypes.data != second.ctypes.data
    assert np.array_equal(first, first_snapshot)

    source.setflags(write=True)
    third, _ = native_renderer.native_camera_profile(
        source, matrix, white, 1.0, look, (4, 3, 2), 0, tone, matrix)
    assert shared_host.profile_shared_source_calls == [source_handle, source_handle]
    assert np.all(third == 33)

    native_renderer.clear_camera_profile_gpu_cache()
    assert native_shared.shared_buffer_owner(source) is not None
    assert np.all(source == 1200)
    assert np.all(first == 31) and np.all(second == 32) and np.all(third == 33)


def test_partial_shared_input_is_copied_before_transfer_binding(shared_host):
    camera = np.zeros((2, 3, 3), dtype=np.uint16)
    reference = np.ones_like(camera)
    backing = native_shared.allocate_shared_rgb((4, 3, 3))
    backing[:2] = 45
    partial = backing[:2]
    assert native_shared.shared_buffer_owner(partial) is None

    native_renderer.native_camera_profile_transfer(
        camera, reference, partial, np.eye(3, dtype=np.float32))

    bound_handle, _ = shared_host.transfer_calls[0]
    backing_owner = native_shared.shared_buffer_owner(backing)
    assert bound_handle != backing_owner.handle.value
    assert np.array_equal(shared_host.transfer_inputs[0], partial)


def test_dense_destinations_are_validated_and_returned(monkeypatch):
    source = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    expected = np.rint(source * 65535.0).astype(np.uint16)

    def quantize(_width, _height, source_pointer, count, output_pointer, _output_count):
        source_values = np.ctypeslib.as_array(source_pointer, shape=(count,)).copy()
        target = np.ctypeslib.as_array(output_pointer, shape=(count,))
        target[:] = np.clip(np.rint(source_values * 65535.0), 0, 65535).astype(np.uint16)
        return 0

    monkeypatch.setattr(native_dense, "_function", lambda *_args: _Function(quantize))
    destination = np.empty(source.shape, dtype=np.uint16)
    result = native_dense.linear_float_to_rgb16(source, destination=destination)
    assert result is destination
    assert np.array_equal(result, expected)

    with pytest.raises(TypeError, match="uint16"):
        native_dense.linear_float_to_rgb16(source, destination=np.empty(source.shape, dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        native_dense.linear_float_to_rgb16(source, destination=np.empty((2, 3, 3), dtype=np.uint16))
    readonly = np.empty(source.shape, dtype=np.uint16)
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        native_dense.linear_float_to_rgb16(source, destination=readonly)


def test_lens_remap_destination_is_validated_and_returned(monkeypatch):
    source = np.arange(3 * 4 * 3, dtype=np.uint16).reshape(3, 4, 3)
    maps = np.zeros((2, 4, 2), dtype=np.float32)

    def remap(_w, _h, _source, _source_count, _maps, _map_count, _strip_height,
              _channels, output_pointer, output_count, _mode):
        np.ctypeslib.as_array(output_pointer, shape=(output_count,)).fill(91)
        return 0

    monkeypatch.setattr(native_dense, "_function", lambda *_args: _Function(remap))
    destination = np.empty((2, 4, 3), dtype=np.uint16)
    result = native_dense.lens_remap(source, maps, destination=destination)
    assert result is destination
    assert np.all(result == 91)
    with pytest.raises(ValueError, match="shape"):
        native_dense.lens_remap(source, maps, destination=np.empty((1, 4, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="C-contiguous"):
        native_dense.lens_remap(source, maps,
                                destination=np.empty((2, 4, 6), dtype=np.uint16)[..., ::2])


def test_cpu_quantization_fallback_keeps_exact_rounding(monkeypatch):
    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    source = np.array([[[0.0, 0.5 / 65535.0, 1.5 / 65535.0],
                        [0.25, 0.5, 65534.5 / 65535.0],
                        [1.0, -0.25, 1.25]]], dtype=np.float32)
    before = source.copy()

    result = app_api._linear_float_to_uint16(source, backend="cpu")

    expected = np.clip(np.rint(source * 65535.0), 0, 65535).astype(np.uint16)
    assert np.array_equal(result, expected)
    assert np.array_equal(source, before)


def test_preview_lens_entry_keeps_existing_uint16_storage(monkeypatch):
    import app_api
    from types import SimpleNamespace

    image = np.zeros((3, 4, 3), dtype=np.uint16)
    image.setflags(write=False)
    marker = object()

    def lens(source, metadata, **kwargs):
        assert source is image
        assert not source.flags.writeable
        return source.copy(), marker

    def unexpected_conversion(_source):
        raise AssertionError('uint16 storage must reach Lensfun without another copy')

    monkeypatch.setattr(app_api, 'apply_lens_correction', lens)
    monkeypatch.setattr(app_api, 'to_uint16', unexpected_conversion)
    output, result, gain = app_api._correct_enhanced_raw(
        image, SimpleNamespace(source_kind='raw'), Path('fixture.NEF'), preview=True,
    )
    assert result is marker and not gain
    assert np.array_equal(output, image) and not np.shares_memory(output, image)
