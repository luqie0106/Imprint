"""Lifetime-safe NumPy views over native Metal shared buffers.

Shared allocations are created by a cached, renderer-only allocator. Each
NumPy buffer owns its opaque native handle independently, so clearing the
allocator renderer cannot invalidate arrays that are still in use.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np


_MIN_SHARED_PIXELS = 1_000_000
_MAX_IMAGE_VALUES = 1 << 29
_allocator_lock = threading.Lock()
_allocator_state: tuple[object, ctypes.c_void_p] | None = None


def _renderer_api():
    # Import lazily: native_renderer also calls into this module from its
    # rendering entrypoints.
    from native_renderer import NativeRendererError, _create_renderer, _get_library

    return NativeRendererError, _create_renderer, _get_library


def shared_memory_enabled() -> bool:
    """Whether shared allocations are enabled on this host."""
    setting = os.environ.get("IMPRINT_NATIVE_SHARED_MEMORY")
    return sys.platform == "darwin" and setting != "0"


def shared_memory_eligible(shape, dtype=None) -> bool:
    """Check an HxWx3 RGB shape against the opt-in shared-memory policy."""
    if not shared_memory_enabled():
        return False
    try:
        dims = tuple(int(value) for value in shape)
    except (TypeError, ValueError, OverflowError):
        return False
    if len(dims) != 3 or dims[0] <= 0 or dims[1] <= 0 or dims[2] != 3:
        return False
    if dims[0] > 65535 or dims[1] > 65535 or dims[0] * dims[1] * 3 > _MAX_IMAGE_VALUES:
        return False
    if dtype is not None and np.dtype(dtype) not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        return False
    return (os.environ.get("IMPRINT_NATIVE_SHARED_MEMORY") == "1"
            or dims[0] * dims[1] >= _MIN_SHARED_PIXELS)


class _SharedBufferOwner:
    """Strong owner for an opaque native shared buffer allocation."""

    __slots__ = ("library", "handle", "data_address", "nbytes", "dtype", "shape", "_destroy")

    def __init__(self, library, handle, data_address: int, nbytes: int,
                 dtype: np.dtype, shape: tuple[int, ...]):
        self.library = library
        self.handle = ctypes.c_void_p(handle.value if isinstance(handle, ctypes.c_void_p) else handle)
        self.data_address = int(data_address)
        self.nbytes = int(nbytes)
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape)
        self._destroy = library.im_shared_buffer_destroy

    def __del__(self):
        handle = getattr(self, "handle", None)
        destroy = getattr(self, "_destroy", None)
        if handle is not None and handle.value and destroy is not None:
            self.handle = ctypes.c_void_p()
            try:
                destroy(handle)
            except Exception:
                # Destructors must not surface exceptions during garbage
                # collection or interpreter shutdown.
                pass


def _find_owner(array: np.ndarray):
    current = array
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        owner = getattr(current, "_imprint_shared_owner", None)
        if owner is not None:
            return owner
        current = getattr(current, "base", None)
    return None


def shared_buffer_owner(array: np.ndarray):
    """Return the owner only for a complete, contiguous view of its allocation.

    Slices and offset views remain lifetime-safe through NumPy's base chain,
    but are intentionally ineligible for direct GPU binding.
    """
    if not isinstance(array, np.ndarray):
        return None
    owner = _find_owner(array)
    if owner is None:
        return None
    if (array.dtype != owner.dtype or not array.flags.c_contiguous
            or array.nbytes != owner.nbytes or array.ctypes.data != owner.data_address):
        return None
    return owner


def clear_shared_allocator_cache() -> None:
    """Destroy only the renderer used to allocate buffers; extant arrays live on."""
    global _allocator_state
    with _allocator_lock:
        state = _allocator_state
        _allocator_state = None
    if state is not None:
        library, renderer = state
        library.im_renderer_destroy(renderer)


def allocate_shared_rgb(shape, dtype=np.uint16) -> np.ndarray:
    """Allocate a zero-copy NumPy RGB array backed by a native shared buffer."""
    NativeRendererError, create_renderer, get_library = _renderer_api()
    try:
        dims = tuple(int(value) for value in shape)
        sample_dtype = np.dtype(dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("shared RGB shape and dtype are invalid") from exc
    if sample_dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise TypeError("shared RGB storage supports uint8 or uint16")
    if len(dims) != 3 or dims[2] != 3 or any(value <= 0 for value in dims):
        raise ValueError("shared RGB shape must be HxWx3")
    if not shared_memory_eligible(dims, sample_dtype):
        raise NativeRendererError("Native shared memory is disabled or the RGB image is not eligible")
    byte_count = int(np.prod(dims, dtype=np.int64)) * sample_dtype.itemsize
    if byte_count > np.iinfo(np.uintp).max:
        raise ValueError("shared RGB allocation exceeds addressable memory")

    library, _ = get_library()
    create = getattr(library, "im_renderer_create_shared_buffer", None)
    data = getattr(library, "im_shared_buffer_data", None)
    size = getattr(library, "im_shared_buffer_size", None)
    destroy_buffer = getattr(library, "im_shared_buffer_destroy", None)
    if create is None or data is None or size is None or destroy_buffer is None:
        raise NativeRendererError("Native shared-buffer ABI is unavailable")

    renderer_pointer = ctypes.c_void_p
    shared_pointer = ctypes.c_void_p
    create.argtypes = [renderer_pointer, ctypes.c_size_t, ctypes.POINTER(shared_pointer)]
    create.restype = ctypes.c_int
    data.argtypes = [shared_pointer]
    data.restype = ctypes.c_void_p
    size.argtypes = [shared_pointer]
    size.restype = ctypes.c_size_t
    destroy_buffer.argtypes = [shared_pointer]
    destroy_buffer.restype = None

    global _allocator_state
    with _allocator_lock:
        if _allocator_state is not None and _allocator_state[0] is not library:
            old_library, old_renderer = _allocator_state
            _allocator_state = None
            old_library.im_renderer_destroy(old_renderer)
        if _allocator_state is None:
            renderer, backend = create_renderer(library)
            if backend.lower() != "metal":
                library.im_renderer_destroy(renderer)
                raise NativeRendererError("Native shared buffers require the Metal backend")
            _allocator_state = (library, renderer)
        _, renderer = _allocator_state
        handle = shared_pointer()
        status = create(renderer, byte_count, ctypes.byref(handle))
        if status != 0 or not handle.value:
            if handle.value:
                destroy_buffer(handle)
            raise NativeRendererError(f"Native shared-buffer allocation failed (status {status})")
        actual_size = int(size(handle))
        address = data(handle)
        address_value = int(address or 0)
        if actual_size != byte_count or not address_value:
            destroy_buffer(handle)
            raise NativeRendererError("Native shared buffer returned an invalid data region")

    owner = _SharedBufferOwner(library, handle, address_value, byte_count, sample_dtype, dims)
    storage_type = ctypes.c_ubyte * byte_count
    storage = storage_type.from_address(address_value)
    # ndarray(buffer=...) retains this ctypes exporter, which in turn retains
    # the owner for every derived view and slice.
    storage._imprint_shared_owner = owner
    return np.ndarray(dims, dtype=sample_dtype, buffer=storage, order="C")


try:
    import atexit

    atexit.register(clear_shared_allocator_cache)
except Exception:  # pragma: no cover - atexit is available in normal runtimes
    pass
