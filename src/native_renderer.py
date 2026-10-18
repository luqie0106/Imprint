"""Safe ctypes bridge to Imprint's optional native renderer.

The bridge deliberately owns only one processing stage per call. Python remains
responsible for decoding, color management, lens correction, XMP, and export.
"""

from __future__ import annotations

import ctypes
import atexit
from collections import OrderedDict
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import sys
import threading
from typing import Callable, Mapping
from weakref import WeakValueDictionary

import numpy as np


_DEHAZE_FIELDS = (
    "strength",
    "naturalness",
    "fog_retention",
    "local_contrast",
    "color_recovery",
    "color_protection",
    "highlight_protection",
    "shadow_protection",
    "brightness_protection",
)
_BASIC_FIELDS = (
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "vibrance",
    "saturation",
)
_BASIC_LIMITS = ((-5.0, 5.0), *((-100.0, 100.0),) * 7)
_MAX_IMAGE_VALUES = 1 << 29
_RICOH_LUT_EDGE = 65
_RICOH_LUT_CACHE_LIMIT = 4
_ricoh_lut_cache: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
_ricoh_lut_lock = threading.Lock()
_PREVIEW_CACHE_LIMIT = 4


class NativeRendererError(RuntimeError):
    """The native renderer is missing, unavailable, or rejected a render."""


class _DehazeParams(ctypes.Structure):
    _fields_ = [(field, ctypes.c_float) for field in _DEHAZE_FIELDS]


class _BasicParams(ctypes.Structure):
    _fields_ = [(field, ctypes.c_float) for field in _BASIC_FIELDS]


@dataclass
class _PreviewCacheEntry:
    library: object
    renderer: ctypes.c_void_p
    metadata: object
    width: int
    height: int
    was_uint8: bool
    lock: threading.Lock = field(default_factory=threading.Lock)


_preview_cache: OrderedDict[tuple[object, ...], _PreviewCacheEntry] = OrderedDict()
_preview_cache_lock = threading.RLock()
_preview_creation_locks: WeakValueDictionary[tuple[object, ...], threading.Lock] = WeakValueDictionary()


_load_lock = threading.Lock()
_cached_library: ctypes.CDLL | None = None
_cached_library_path: str | None = None
_last_native_physical_backend: str | None = None


def _float_range_valid(array: np.ndarray, minimum: float = 0.0,
                       maximum: float = 1.0) -> bool:
    """Validate float pixels without allocating three full-image masks."""
    if array.dtype == np.float32 and array.size:
        try:
            from native_dense import float_range_valid as native_valid_float

            return native_valid_float(array, minimum, maximum)
        except (ImportError, NativeRendererError):
            pass
    return bool(np.isfinite(array).all()
                and not np.any(array < minimum) and not np.any(array > maximum))

# One camera source on the GPU, with no retained Python full-image reference.
# Preview cache eviction explicitly destroys this renderer and its allocations.
_camera_profile_gpu_lock = threading.Lock()
_camera_profile_gpu_state: tuple | None = None


def clear_camera_profile_gpu_cache() -> None:
    global _camera_profile_gpu_state
    with _camera_profile_gpu_lock:
        if _camera_profile_gpu_state is not None:
            library, renderer, *_ = _camera_profile_gpu_state
            _camera_profile_gpu_state = None
            library.im_renderer_destroy(renderer)
    # Keep lock ordering independent and release the transfer renderer too.
    clear_camera_profile_transfer_gpu_cache()
    # Shared arrays own their handles independently of this allocator renderer.
    from native_shared import clear_shared_allocator_cache
    clear_shared_allocator_cache()


def _native_profile_look(
    rgb: np.ndarray, look: np.ndarray, dims, encoding: int, *, recovered: bool,
    bulk: bool = False, lookup_gain: float = 1.0, amount: float = 1.0,
) -> np.ndarray:
    import operator

    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
            or max(rgb.shape[:2]) > 65535 or rgb.size > (1 << 29)):
        raise ValueError("Profile look requires nonempty float32 HxWx3 RGB")
    try:
        if len(dims) != 3 or any(isinstance(v, (bool, np.bool_)) for v in dims):
            raise ValueError("Profile look requires three integer dimensions")
        hue, saturation, value = (operator.index(v) for v in dims)
        if isinstance(encoding, (bool, np.bool_)):
            raise ValueError("Profile look encoding must be 0 or 1")
        mode = operator.index(encoding)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Profile look dimensions and encoding must be integers") from exc
    if (hue < 1 or saturation < 2 or value < (1 if recovered else 2)
            or hue * saturation * value > 1000000
            or mode not in (0, 1)):
        raise ValueError("Profile look dimensions or encoding are outside the native domain")
    if recovered and rgb.shape[0] * rgb.shape[1] > (1 << 26):
        raise ValueError("ACR profile look input exceeds the prepared pixel limit")
    if bulk:
        try:
            gain_value, amount_value = float(lookup_gain), float(amount)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ACR bulk profile controls must be finite numbers") from exc
        if (not math.isfinite(gain_value) or not 0 <= gain_value <= 65536
                or not math.isfinite(amount_value) or not 0 <= amount_value <= 2):
            raise ValueError("ACR bulk profile controls are outside the prepared domain")
    table = np.ascontiguousarray(look, dtype=np.float32)
    if table.shape != (value, hue, saturation, 3):
        raise ValueError("Profile look table does not match its dimensions")
    source = np.ascontiguousarray(rgb)
    output = np.empty_like(source)
    library, _ = _get_library()
    symbol = "im_native_acr_profile_look_rgbf32" if recovered else "im_native_profile_look_rgbf32"
    if bulk:
        symbol = "im_native_acr_profile_stage_sdr_rgbf32"
    function = getattr(library, symbol, None)
    if function is None:
        raise NativeRendererError("Native profile LookTable ABI is unavailable")
    fp = ctypes.POINTER(ctypes.c_float)
    argument_types = [ctypes.c_uint32, ctypes.c_uint32, fp, ctypes.c_size_t,
                      fp, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32,
                      ctypes.c_uint32, ctypes.c_uint32]
    if bulk:
        argument_types += [ctypes.c_double, ctypes.c_double]
    function.argtypes = argument_types + [fp, ctypes.c_size_t]
    function.restype = ctypes.c_int
    arguments = [source.shape[1], source.shape[0], source.ctypes.data_as(fp), source.size,
                 table.ctypes.data_as(fp), table.size, hue, saturation, value, mode]
    if bulk:
        arguments += [gain_value, amount_value]
    status = function(*arguments, output.ctypes.data_as(fp), output.size)
    if status == 1:
        raise ValueError("Native profile LookTable rejected input or table")
    if status:
        raise NativeRendererError(f"Native profile LookTable failed (status {status})")
    return output


def native_profile_look(rgb: np.ndarray, look: np.ndarray, dims, encoding: int) -> np.ndarray:
    """Apply the existing Imprint profile LookTable compatibility operation."""
    return _native_profile_look(rgb, look, dims, encoding, recovered=False)


def native_acr_profile_look(rgb: np.ndarray, look: np.ndarray, dims, encoding: int) -> np.ndarray:
    """Apply the recovered single-map helper (012ebae4), with C++ resources."""
    return _native_profile_look(rgb, look, dims, encoding, recovered=True)


def native_acr_profile_stage_sdr(
    rgb: np.ndarray, look: np.ndarray, dims, encoding: int, *,
    lookup_gain: float = 1.0, amount: float = 1.0,
) -> np.ndarray:
    """Recovered baseline bulk SDR stage; all resource/pixel work is in C++."""
    return _native_profile_look(rgb, look, dims, encoding, recovered=True,
                               bulk=True, lookup_gain=lookup_gain, amount=amount)


def native_acr_prepare_huesatmap(first, second, dims, white_xy, *,
                                endpoint_kelvin=(2850.0, 6500.0)):
    """Select/blend a prepared HueSatMap in C++; no Python-generated resource."""
    import operator

    try:
        if len(dims) != 3 or any(isinstance(v, (bool, np.bool_)) for v in dims):
            raise ValueError("HueSatMap requires three integer dimensions")
        hue, saturation, value = (operator.index(v) for v in dims)
    except (TypeError, OverflowError) as exc:
        raise ValueError("HueSatMap dimensions must be integers") from exc
    if hue < 1 or saturation < 2 or value < 1 or hue*saturation*value > 1_000_000:
        raise ValueError("HueSatMap dimensions exceed the native domain")
    shape = (value, hue, saturation, 3)
    table1 = np.ascontiguousarray(first, dtype=np.float32)
    table2 = None if second is None else np.ascontiguousarray(second, dtype=np.float32)
    if table1.shape != shape or (table2 is not None and table2.shape != shape):
        raise ValueError("HueSatMap data does not match its dimensions")
    xy = np.asarray(white_xy, dtype=np.float64)
    endpoints = np.asarray(endpoint_kelvin, dtype=np.float64)
    if xy.shape != (2,) or endpoints.shape != (2,):
        raise ValueError("HueSatMap requires white xy and two endpoint temperatures")
    controls = np.ascontiguousarray(np.concatenate((xy, endpoints)), dtype=np.float64)
    output = np.empty(shape, dtype=np.float32)
    metadata = np.empty(3, dtype=np.float64)
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_prepare_huesatmap", None)
    if function is None:
        raise NativeRendererError("Native HueSatMap preparation ABI is unavailable")
    fp, dp = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)
    function.argtypes = [fp, ctypes.c_size_t, fp, ctypes.c_size_t,
                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                          dp, ctypes.c_size_t, fp, ctypes.c_size_t, dp, ctypes.c_size_t]
    function.restype = ctypes.c_int
    status = function(table1.ctypes.data_as(fp), table1.size,
        None if table2 is None else table2.ctypes.data_as(fp),
        0 if table2 is None else table2.size, hue, saturation, value,
        controls.ctypes.data_as(dp), controls.size,
        output.ctypes.data_as(fp), output.size, metadata.ctypes.data_as(dp), metadata.size)
    if status == 1:
        raise ValueError("Native HueSatMap preparation rejected the inputs")
    if status:
        raise NativeRendererError(f"Native HueSatMap preparation failed (status {status})")
    return output, metadata


def native_acr_prepare_huesatmap_gain(process_version, controls):
    """Resolve HSM gains from explicit prepared state; no metadata inference."""
    import operator

    try:
        if isinstance(process_version, (bool, np.bool_)):
            raise TypeError("boolean is not a process version")
        version = operator.index(process_version)
    except (TypeError, OverflowError) as exc:
        raise ValueError("process_version must be uint32") from exc
    if version < 0 or version > 0xffffffff:
        raise ValueError("process_version must be uint32")
    values = np.ascontiguousarray(controls, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("HSM gain requires three finite prepared controls")
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_prepare_huesatmap_gain", None)
    if function is None:
        raise NativeRendererError("Native HSM gain preparation ABI is unavailable")
    dp = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [ctypes.c_uint32, dp, ctypes.c_size_t, dp, ctypes.c_size_t]
    function.restype = ctypes.c_int
    output = np.empty(2, dtype=np.float64)
    status = function(version, values.ctypes.data_as(dp), values.size,
                      output.ctypes.data_as(dp), output.size)
    if status == 1:
        raise ValueError("Native HSM gain preparation rejected the inputs")
    if status:
        raise NativeRendererError(f"Native HSM gain preparation failed (status {status})")
    return output


def native_camera_profile(source, matrix, white, gain, look, dims, encoding, tone, projection):
    """Exact camera profile on Metal/D3D12, reusing an immutable source upload.

    Mutable arrays are uploaded on every call. Cached read-only arrays must
    remain immutable for their lifetime, as with the existing preview caches.
    Returns (display RGB8, actual backend), or raises for CPU fallback.
    """
    global _camera_profile_gpu_state
    import weakref
    from native_shared import allocate_shared_rgb, shared_buffer_owner, shared_memory_eligible
    from native_dense import _camera_profile_arguments

    if os.environ.get("IMPRINT_NATIVE_CAMERA_PROFILE") == "0":
        raise NativeRendererError("Camera profile GPU operator is disabled")
    image, width, height, constants, table, (hue, saturation, value), curve = _camera_profile_arguments(
        source, matrix, white, gain, look, dims, encoding, tone, projection)
    source_owner = (shared_buffer_owner(image) if
                    shared_memory_eligible(image.shape, np.uint16) else None)
    library, _ = _get_library()
    upload = getattr(library, "im_renderer_set_camera_profile_source", None)
    render = getattr(library, "im_renderer_render_camera_profile", None)
    upload_shared = getattr(library, "im_renderer_set_camera_profile_source_shared", None)
    render_shared = getattr(library, "im_renderer_render_camera_profile_shared", None)
    if render is None or (upload is None and (source_owner is None or upload_shared is None)):
        raise NativeRendererError("Camera profile GPU ABI is unavailable")
    u16, u8, fp = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint8),
                   ctypes.POINTER(ctypes.c_float))
    if upload is not None:
        upload.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, u16, ctypes.c_size_t]
        upload.restype = ctypes.c_int
    if upload_shared is not None:
        upload_shared.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p]
        upload_shared.restype = ctypes.c_int
    render.argtypes = [ctypes.c_void_p, fp, ctypes.c_size_t, fp, ctypes.c_size_t,
                      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                      fp, ctypes.c_size_t, u8, ctypes.c_size_t]
    render.restype = ctypes.c_int
    if render_shared is not None:
        render_shared.argtypes = [
            ctypes.c_void_p, fp, ctypes.c_size_t, fp, ctypes.c_size_t,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            fp, ctypes.c_size_t, ctypes.c_void_p,
        ]
        render_shared.restype = ctypes.c_int

    with _camera_profile_gpu_lock:
        if _camera_profile_gpu_state is None:
            renderer, backend = _create_renderer(library)
            enabled = os.environ.get("IMPRINT_NATIVE_CAMERA_PROFILE", "1" if backend == "Metal" else "0") == "1"
            if backend not in ("Metal", "D3D12") or not enabled:
                library.im_renderer_destroy(renderer)
                raise NativeRendererError("Camera profile GPU backend is unavailable or disabled")
            _camera_profile_gpu_state = (library, renderer, backend, None, None)
        cached_library, renderer, backend, uploaded, uploaded_mode = _camera_profile_gpu_state
        if cached_library is not library:
            raise NativeRendererError("Camera profile renderer library changed; clear the GPU cache first")
        try:
            source_mode = "shared" if source_owner is not None and upload_shared is not None else "copy"
            if (image.flags.writeable or uploaded is None or uploaded() is not image
                    or uploaded_mode != source_mode):
                if source_mode == "shared":
                    status = upload_shared(renderer, width, height, source_owner.handle)
                else:
                    if upload is None:
                        raise NativeRendererError("Camera profile source upload ABI is unavailable")
                    status = upload(renderer, width, height, image.ctypes.data_as(u16), image.size)
                if status != 0:
                    raise NativeRendererError(f"Camera profile upload failed: {_native_error(library, renderer)}")
                _camera_profile_gpu_state = (library, renderer, backend, weakref.ref(image), source_mode)
            output_owner = None
            if render_shared is not None and shared_memory_eligible(image.shape, np.uint8):
                try:
                    output = allocate_shared_rgb(image.shape, np.uint8)
                    output_owner = shared_buffer_owner(output)
                except NativeRendererError:
                    output = None
            else:
                output = None
            if output_owner is not None:
                status = render_shared(
                    renderer, constants.ctypes.data_as(fp), constants.size,
                    table.ctypes.data_as(fp), table.size, hue, saturation, value, int(encoding),
                    curve.ctypes.data_as(fp), curve.size, output_owner.handle,
                )
            else:
                output = np.empty(image.shape, dtype=np.uint8)
                status = render(renderer, constants.ctypes.data_as(fp), constants.size,
                                table.ctypes.data_as(fp), table.size, hue, saturation, value, int(encoding),
                                curve.ctypes.data_as(fp), curve.size, output.ctypes.data_as(u8), output.size)
            if status != 0:
                raise NativeRendererError(f"Camera profile GPU render failed: {_native_error(library, renderer)}")
            return output, backend.lower()
        except Exception:
            _camera_profile_gpu_state = None
            library.im_renderer_destroy(renderer)
            raise


atexit.register(clear_camera_profile_gpu_cache)


_camera_profile_transfer_gpu_lock = threading.Lock()
_camera_profile_transfer_gpu_state: tuple | None = None


def clear_camera_profile_transfer_gpu_cache() -> None:
    """Release the transfer renderer and all Metal buffers it owns."""
    global _camera_profile_transfer_gpu_state
    with _camera_profile_transfer_gpu_lock:
        if _camera_profile_transfer_gpu_state is not None:
            library, renderer, *_ = _camera_profile_transfer_gpu_state
            _camera_profile_transfer_gpu_state = None
            library.im_renderer_destroy(renderer)


def native_camera_profile_transfer(camera, reference, processed, inverse_matrix):
    """Run camera-space enhancement transfer on a supported GPU backend.

    Read-only camera/reference arrays are retained by weak identity and reused.
    Mutable sources are uploaded on every invocation to prevent stale pixels.
    The returned uint16 array owns its storage; inputs remain alive until the
    synchronous native call and GPU command have completed.
    """
    global _camera_profile_transfer_gpu_state
    transfer_setting = os.environ.get("IMPRINT_NATIVE_CAMERA_PROFILE_TRANSFER")
    if transfer_setting == "0" or (sys.platform != "darwin" and transfer_setting != "1"):
        raise NativeRendererError("Camera profile GPU transfer is disabled")
    if not isinstance(camera, np.ndarray) or not isinstance(reference, np.ndarray) or \
            not isinstance(processed, np.ndarray):
        raise TypeError("Camera profile GPU transfer sources must be NumPy arrays")
    images = (camera, reference, processed)
    if any(image.dtype != np.dtype(np.uint16) or image.ndim != 3 or image.shape[2] != 3
           for image in images):
        raise ValueError("Camera profile GPU transfer requires uint16 RGB images")
    if camera.shape != reference.shape or camera.shape != processed.shape:
        raise ValueError("Camera profile GPU transfer images must have identical shapes")
    height, width, _ = camera.shape
    if not height or not width or width > 65535 or height > 65535 or camera.size > _MAX_IMAGE_VALUES:
        raise ValueError("Camera profile GPU transfer dimensions exceed the ABI limits")
    inverse = np.asarray(inverse_matrix, dtype=np.float32)
    if inverse.shape != (3, 3) or not np.isfinite(inverse).all() or np.any(np.abs(inverse) > 1e6):
        raise ValueError("Camera profile GPU transfer matrix must be finite 3x3 RGB")

    camera_image = np.ascontiguousarray(camera)
    reference_image = np.ascontiguousarray(reference)
    processed_image = np.ascontiguousarray(processed)
    inverse_image = np.ascontiguousarray(inverse)
    library, _ = _get_library()
    upload = getattr(library, "im_renderer_set_camera_profile_transfer_source", None)
    render = getattr(library, "im_renderer_render_camera_profile_transfer", None)
    render_shared = getattr(library, "im_renderer_render_camera_profile_transfer_shared", None)
    supports = getattr(library, "im_renderer_supports_camera_profile_transfer", None)
    if upload is None or supports is None:
        raise NativeRendererError("Camera profile GPU transfer ABI is unavailable")
    u16 = ctypes.POINTER(ctypes.c_uint16)
    fp = ctypes.POINTER(ctypes.c_float)
    upload.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                       u16, u16, ctypes.c_size_t]
    upload.restype = ctypes.c_int
    if render is not None:
        render.argtypes = [ctypes.c_void_p, u16, ctypes.c_size_t,
                           fp, ctypes.c_size_t, u16, ctypes.c_size_t]
        render.restype = ctypes.c_int
    supports.argtypes = [ctypes.c_void_p]
    supports.restype = ctypes.c_int

    from native_shared import (
        allocate_shared_rgb, shared_buffer_owner, shared_memory_eligible,
    )
    use_shared = False
    shared_processed = None
    shared_processed_owner = None
    shared_output = None
    shared_output_owner = None
    if render_shared is not None and shared_memory_eligible(processed_image.shape, np.uint16):
        render_shared.argtypes = [ctypes.c_void_p, ctypes.c_void_p, fp,
                                  ctypes.c_size_t, ctypes.c_void_p]
        render_shared.restype = ctypes.c_int
        try:
            # Reuse an already-owned input. Ordinary NumPy inputs require one
            # explicit copy into a shareable allocation before the GPU call.
            shared_processed_owner = shared_buffer_owner(processed_image)
            if shared_processed_owner is None:
                shared_processed = allocate_shared_rgb(processed_image.shape, np.uint16)
                np.copyto(shared_processed, processed_image)
                shared_processed_owner = shared_buffer_owner(shared_processed)
            shared_output = allocate_shared_rgb(camera_image.shape, np.uint16)
            shared_output_owner = shared_buffer_owner(shared_output)
            use_shared = shared_processed_owner is not None and shared_output_owner is not None
        except NativeRendererError:
            use_shared = False
            shared_processed = None
            shared_processed_owner = None
            shared_output = None
            shared_output_owner = None
    if not use_shared and render is None:
        raise NativeRendererError("Camera profile GPU transfer ABI is unavailable")

    import weakref
    with _camera_profile_transfer_gpu_lock:
        if (_camera_profile_transfer_gpu_state is not None and
                _camera_profile_transfer_gpu_state[0] is not library):
            old_library, old_renderer, *_ = _camera_profile_transfer_gpu_state
            _camera_profile_transfer_gpu_state = None
            old_library.im_renderer_destroy(old_renderer)
        if _camera_profile_transfer_gpu_state is None:
            renderer, backend = _create_renderer(library)
            if backend.lower() != "metal" or not supports(renderer):
                library.im_renderer_destroy(renderer)
                raise NativeRendererError("Camera profile GPU transfer is unavailable on this backend")
            _camera_profile_transfer_gpu_state = (library, renderer, backend, None, None, width, height)

        cached_library, renderer, backend, camera_ref, reference_ref, cached_width, cached_height = (
            _camera_profile_transfer_gpu_state
        )
        try:
            cacheable = not camera_image.flags.writeable and not reference_image.flags.writeable
            cached_sources_match = (
                cacheable and cached_width == width and cached_height == height and
                camera_ref is not None and reference_ref is not None and
                camera_ref() is camera_image and reference_ref() is reference_image
            )
            if not cached_sources_match:
                status = upload(renderer, width, height,
                                camera_image.ctypes.data_as(u16),
                                reference_image.ctypes.data_as(u16), camera_image.size)
                if status != 0:
                    raise NativeRendererError(
                        f"Camera profile GPU transfer source upload failed: "
                        f"{_native_error(library, renderer)}"
                    )
                _camera_profile_transfer_gpu_state = (
                    library, renderer, backend,
                    weakref.ref(camera_image) if cacheable else None,
                    weakref.ref(reference_image) if cacheable else None,
                    width, height,
                )
            if use_shared:
                output = shared_output
                status = render_shared(renderer, shared_processed_owner.handle,
                                       inverse_image.ctypes.data_as(fp), inverse_image.size,
                                       shared_output_owner.handle)
            else:
                output = np.empty(camera_image.shape, dtype=np.uint16)
                status = render(renderer, processed_image.ctypes.data_as(u16), processed_image.size,
                                inverse_image.ctypes.data_as(fp), inverse_image.size,
                                output.ctypes.data_as(u16), output.size)
            if status != 0:
                raise NativeRendererError(
                    f"Camera profile GPU transfer failed: {_native_error(library, renderer)}"
                )
            return output, backend.lower()
        except Exception:
            _camera_profile_transfer_gpu_state = None
            library.im_renderer_destroy(renderer)
            raise


atexit.register(clear_camera_profile_transfer_gpu_cache)


_warp_gpu_lock = threading.Lock()
_warp_gpu_state: tuple | None = None


def clear_native_warp_cache() -> None:
    global _warp_gpu_state
    with _warp_gpu_lock:
        if _warp_gpu_state is not None:
            library, renderer, _ = _warp_gpu_state
            _warp_gpu_state = None
            library.im_renderer_destroy(renderer)


def native_warp_rectilinear(source, constants):
    """GPU-first stateless RGB16 warp; no source arrays are retained."""
    global _warp_gpu_state
    from native_dense import _profile_rgb16
    from dng_warp import _validate_constants
    image, width, height = _profile_rgb16(source)
    values = _validate_constants(constants)
    library, _ = _get_library()
    run = getattr(library, "im_renderer_warp_rectilinear_rgb16", None)
    supports = getattr(library, "im_renderer_supports_warp_rectilinear", None)
    if run is None or supports is None:
        raise NativeRendererError("GPU WarpRectilinear ABI is unavailable")
    u16, fp = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float)
    run.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, u16,
                   ctypes.c_size_t, fp, ctypes.c_size_t, u16, ctypes.c_size_t]
    run.restype = ctypes.c_int
    supports.argtypes = [ctypes.c_void_p]
    supports.restype = ctypes.c_int
    with _warp_gpu_lock:
        if _warp_gpu_state is not None and _warp_gpu_state[0] is not library:
            old_library, old_renderer, _ = _warp_gpu_state
            _warp_gpu_state = None
            old_library.im_renderer_destroy(old_renderer)
        if _warp_gpu_state is None:
            renderer, backend = _create_renderer(library)
            if backend not in ("Metal", "D3D12") or not supports(renderer):
                library.im_renderer_destroy(renderer)
                raise NativeRendererError("GPU WarpRectilinear operator is unavailable")
            _warp_gpu_state = (library, renderer, backend)
        _, renderer, backend = _warp_gpu_state
        try:
            output = np.empty_like(image)
            status = run(renderer, width, height, image.ctypes.data_as(u16), image.size,
                         values.ctypes.data_as(fp), values.size,
                         output.ctypes.data_as(u16), output.size)
            if status != 0:
                raise NativeRendererError(f"GPU WarpRectilinear failed: {_native_error(library, renderer)}")
            return output, backend
        except Exception:
            _warp_gpu_state = None
            library.im_renderer_destroy(renderer)
            raise


atexit.register(clear_native_warp_cache)


def _library_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("imprint_renderer.dll", "libimprint_renderer.dll")
    if sys.platform == "darwin":
        return ("libimprint_renderer.dylib", "imprint_renderer.dylib")
    return ("libimprint_renderer.so", "imprint_renderer.so")


def _library_candidates() -> list[Path]:
    """Return only an operator-configured path and known installation locations."""
    explicit = os.environ.get("IMPRINT_NATIVE_RENDERER_LIB")
    if explicit:
        return [Path(explicit).expanduser()]

    names = _library_names()
    project = Path(__file__).resolve().parents[1]
    roots = [
        Path(sys.executable).resolve().parent,
        Path(sys.executable).resolve().parent / "native-renderer",
        project / "native-renderer" / "build",
        project / "native-renderer" / "cmake-build-release",
        project / "native-renderer" / "cmake-build-debug",
        project / "build" / "native-renderer",
        project / "build",
    ]
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        bundle_path = Path(bundle_root).resolve()
        roots[0:0] = [bundle_path / "_internal", bundle_path]
    candidates: list[Path] = []
    for root in roots:
        for name in names:
            candidates.extend((root / name, root / "Release" / name, root / "Debug" / name))
    # Keep order while avoiding repeated executable/project directories.
    return list(dict.fromkeys(candidates))


def _configure_library(library: ctypes.CDLL) -> None:
    acr_flare = getattr(library, "im_native_acr_flare_u16_rgb", None)
    if acr_flare is not None:
        acr_flare.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        acr_flare.restype = ctypes.c_int
    acr_project = getattr(library, "im_native_acr_project_srgb_rgb8", None)
    if acr_project is not None:
        acr_project.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                               ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                               ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        acr_project.restype = ctypes.c_int
    acr_scale = getattr(library, "im_native_acr_scale_sdr_rgbf32", None)
    if acr_scale is not None:
        acr_scale.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                             ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        acr_scale.restype = ctypes.c_int
    profile_camera_prepare = getattr(library, "im_native_acr_prepare_profile_camera", None)
    if profile_camera_prepare is not None:
        profile_camera_prepare.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                          ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        profile_camera_prepare.restype = ctypes.c_int
    neutral_prepare = getattr(library, "im_native_acr_prepare_camera_neutral", None)
    if neutral_prepare is not None:
        neutral_prepare.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        neutral_prepare.restype = ctypes.c_int
    analog_prepare = getattr(library, "im_native_acr_prepare_analog_balance", None)
    if analog_prepare is not None:
        analog_prepare.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        analog_prepare.restype = ctypes.c_int
    calibration_prepare = getattr(library, "im_native_acr_prepare_camera_calibration", None)
    if calibration_prepare is not None:
        calibration_prepare.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                       ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        calibration_prepare.restype = ctypes.c_int
    white_resolve = getattr(library, "im_native_acr_resolve_white", None)
    if white_resolve is not None:
        white_resolve.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        white_resolve.restype = ctypes.c_int
    colorspec_prepare = getattr(library, "im_native_acr_prepare_colorspec", None)
    if colorspec_prepare is not None:
        colorspec_prepare.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                     ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        colorspec_prepare.restype = ctypes.c_int
    camera_prepare = getattr(library, "im_native_acr_prepare_camera_stage", None)
    if camera_prepare is not None:
        camera_prepare.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        camera_prepare.restype = ctypes.c_int
    exposure_prepare = getattr(library, "im_native_acr_prepare_exposure", None)
    if exposure_prepare is not None:
        exposure_prepare.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                     ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        exposure_prepare.restype = ctypes.c_int
    exposure_limit_log = getattr(library, "im_native_acr_exposure_limit_log_rgbf32", None)
    if exposure_limit_log is not None:
        exposure_limit_log.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        exposure_limit_log.restype = ctypes.c_int
    exposure_from_anchor = getattr(
        library, "im_native_acr_exposure_from_anchor_rgbf32", None)
    if exposure_from_anchor is not None:
        float_pointer = ctypes.POINTER(ctypes.c_float)
        double_pointer = ctypes.POINTER(ctypes.c_double)
        exposure_from_anchor.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            float_pointer, ctypes.c_size_t,
            float_pointer, ctypes.c_size_t,
            double_pointer, ctypes.c_size_t,
            float_pointer, ctypes.c_size_t,
            double_pointer, ctypes.c_size_t,
        ]
        exposure_from_anchor.restype = ctypes.c_int
    gray_log = getattr(library, "im_native_acr_gray_log_rgbf32", None)
    if gray_log is not None:
        float_pointer = ctypes.POINTER(ctypes.c_float)
        double_pointer = ctypes.POINTER(ctypes.c_double)
        gray_log.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            float_pointer, ctypes.c_size_t,
            double_pointer, ctypes.c_size_t,
            ctypes.c_double,
            float_pointer, ctypes.c_size_t,
            double_pointer, ctypes.c_size_t,
        ]
        gray_log.restype = ctypes.c_int
    gray_u16_log = getattr(library, "im_native_acr_gray_u16_log_rgbf32", None)
    if gray_u16_log is not None:
        float_pointer = ctypes.POINTER(ctypes.c_float)
        double_pointer = ctypes.POINTER(ctypes.c_double)
        gray_u16_log.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            float_pointer, ctypes.c_size_t,
            double_pointer, ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32,
            float_pointer, ctypes.c_size_t,
            double_pointer, ctypes.c_size_t,
        ]
        gray_u16_log.restype = ctypes.c_int
    source_limit_u16 = getattr(library, "im_native_acr_source_limit_scene_u16_rgb", None)
    if source_limit_u16 is not None:
        source_limit_u16.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        source_limit_u16.restype = ctypes.c_int
    profile_tone_exposure = getattr(library, "im_native_acr_profile_tone_exposure_rgbf32", None)
    if profile_tone_exposure is not None:
        profile_tone_exposure.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        profile_tone_exposure.restype = ctypes.c_int
    profile_tone = getattr(library, "im_native_acr_profile_tone_rgbf32", None)
    if profile_tone is not None:
        profile_tone.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ]
        profile_tone.restype = ctypes.c_int
    white_temperature = getattr(library, "im_native_acr_white_temperature", None)
    if white_temperature is not None:
        white_temperature.argtypes = [ctypes.c_double, ctypes.c_double,
                                      ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        white_temperature.restype = ctypes.c_int
    for symbol in ("im_native_acr_hsl_sdr_rgbf32", "im_native_acr_grading_sdr_rgbf32",
                   "im_native_acr_rendered_hsl_sdr_rgbf32",
                   "im_native_acr_camera_float_rgbf32"):
        function = getattr(library, symbol, None)
        if function is not None:
            function.argtypes = [
                ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ]
            function.restype = ctypes.c_int
    tone_prepared = getattr(library, "im_native_acr_tone_prepared_sdr_rgbf32", None)
    if tone_prepared is not None:
        tone_prepared.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ]
        tone_prepared.restype = ctypes.c_int
    tone_prepared_mode = getattr(
        library, "im_native_acr_tone_prepared_sdr_mode_rgbf32", None)
    if tone_prepared_mode is not None:
        tone_prepared_mode.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ]
        tone_prepared_mode.restype = ctypes.c_int
    renderer = ctypes.c_void_p
    uint16_pointer = ctypes.POINTER(ctypes.c_uint16)
    library.im_renderer_create.argtypes = [ctypes.c_int, ctypes.POINTER(renderer)]
    library.im_renderer_create.restype = ctypes.c_int
    library.im_renderer_destroy.argtypes = [renderer]
    library.im_renderer_destroy.restype = None
    library.im_renderer_last_error.argtypes = [renderer]
    library.im_renderer_last_error.restype = ctypes.c_char_p
    library.im_renderer_backend_name.argtypes = [renderer]
    library.im_renderer_backend_name.restype = ctypes.c_char_p
    lens_support = getattr(library, "im_renderer_supports_lens_remap", None)
    lens_begin = getattr(library, "im_renderer_lens_remap_begin", None)
    lens_strip = getattr(library, "im_renderer_lens_remap_strip", None)
    lens_finish = getattr(library, "im_renderer_lens_remap_finish", None)
    if all(symbol is not None for symbol in (lens_support, lens_begin, lens_strip, lens_finish)):
        lens_support.argtypes = [renderer]
        lens_support.restype = ctypes.c_int
        lens_begin.argtypes = [renderer, ctypes.c_uint32, ctypes.c_uint32,
                               uint16_pointer, ctypes.c_size_t]
        lens_begin.restype = ctypes.c_int
        lens_strip.argtypes = [renderer, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.c_uint32]
        lens_strip.restype = ctypes.c_int
        lens_finish.argtypes = [renderer, uint16_pointer, ctypes.c_size_t]
        lens_finish.restype = ctypes.c_int
    physical_support = getattr(library, "im_renderer_supports_physical_float", None)
    if physical_support is not None:
        physical_support.argtypes = [renderer]
        physical_support.restype = ctypes.c_int
    library.im_renderer_render_full.argtypes = [
        renderer,
        ctypes.c_uint32,
        ctypes.c_uint32,
        uint16_pointer,
        ctypes.c_size_t,
        ctypes.POINTER(_DehazeParams),
        ctypes.POINTER(_BasicParams),
    ]
    library.im_renderer_render_full.restype = ctypes.c_int
    library.im_renderer_upload_preview_image.argtypes = [
        renderer, ctypes.c_uint32, ctypes.c_uint32, uint16_pointer, ctypes.c_size_t
    ]
    library.im_renderer_upload_preview_image.restype = ctypes.c_int
    library.im_renderer_render.argtypes = [
        renderer, ctypes.c_int, ctypes.POINTER(_DehazeParams), ctypes.POINTER(_BasicParams)
    ]
    library.im_renderer_render.restype = ctypes.c_int
    library.im_renderer_render_ricoh_full.argtypes = [
        renderer,
        ctypes.c_uint32,
        ctypes.c_uint32,
        uint16_pointer,
        ctypes.c_size_t,
        uint16_pointer,
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    library.im_renderer_render_ricoh_full.restype = ctypes.c_int
    library.im_renderer_get_output_size.argtypes = [
        renderer,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.im_renderer_get_output_size.restype = ctypes.c_int
    library.im_renderer_copy_output.argtypes = [renderer, uint16_pointer, ctypes.c_size_t]
    library.im_renderer_copy_output.restype = ctypes.c_int
    # Optional in older packaged renderers. The spatial path checks for the
    # symbol before use and retains the Python reference as its fallback.
    spatial = getattr(library, "im_native_dehaze_spatial_run", None)
    if spatial is not None:
        spatial.argtypes = [
            uint16_pointer, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(_DehazeParams),
            uint16_pointer, ctypes.c_size_t,
        ]
        spatial.restype = ctypes.c_int
    gpu_spatial = getattr(library, "im_renderer_render_spatial_full", None)
    if gpu_spatial is not None:
        gpu_spatial.argtypes = [
            renderer, ctypes.c_uint32, ctypes.c_uint32, uint16_pointer, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(_DehazeParams),
            uint16_pointer, ctypes.c_size_t,
        ]
        gpu_spatial.restype = ctypes.c_int
    physical_float = getattr(library, "im_renderer_render_physical_float", None)
    if physical_float is not None:
        float_pointer = ctypes.POINTER(ctypes.c_float)
        physical_float.argtypes = [
            renderer, ctypes.c_uint32, ctypes.c_uint32,
            float_pointer, ctypes.c_size_t,
            float_pointer, ctypes.c_size_t,
            float_pointer, ctypes.POINTER(_DehazeParams),
            float_pointer, ctypes.c_size_t,
        ]
        physical_float.restype = ctypes.c_int

    guarded = getattr(library, "im_renderer_render_physical_guarded_float", None)
    if guarded is not None:
        float_pointer = ctypes.POINTER(ctypes.c_float)
        guarded.argtypes = [renderer, ctypes.c_uint32, ctypes.c_uint32,
            float_pointer, ctypes.c_size_t, float_pointer, ctypes.c_size_t,
            float_pointer, ctypes.POINTER(_DehazeParams), ctypes.c_float,
            float_pointer, ctypes.c_size_t]
        guarded.restype = ctypes.c_int
    supports_guarded = getattr(library, "im_renderer_supports_physical_guarded_float", None)
    if supports_guarded is not None:
        supports_guarded.argtypes = [renderer]
        supports_guarded.restype = ctypes.c_int

    # Camera-profile transfer is optional in older packaged renderer builds.
    transfer_upload = getattr(library, "im_renderer_set_camera_profile_transfer_source", None)
    transfer_render = getattr(library, "im_renderer_render_camera_profile_transfer", None)
    transfer_supports = getattr(library, "im_renderer_supports_camera_profile_transfer", None)
    if transfer_upload is not None:
        uint16_pointer = ctypes.POINTER(ctypes.c_uint16)
        transfer_upload.argtypes = [renderer, ctypes.c_uint32, ctypes.c_uint32,
                                    uint16_pointer, uint16_pointer, ctypes.c_size_t]
        transfer_upload.restype = ctypes.c_int
    if transfer_render is not None:
        uint16_pointer = ctypes.POINTER(ctypes.c_uint16)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        transfer_render.argtypes = [renderer, uint16_pointer, ctypes.c_size_t,
                                    float_pointer, ctypes.c_size_t,
                                    uint16_pointer, ctypes.c_size_t]
        transfer_render.restype = ctypes.c_int
    if transfer_supports is not None:
        transfer_supports.argtypes = [renderer]
        transfer_supports.restype = ctypes.c_int


def _supports_physical_float(library: object, renderer: object, backend: str) -> bool:
    query = getattr(library, "im_renderer_supports_physical_float", None)
    if query is not None:
        return bool(query(renderer))
    # Older libraries exported the float ABI with only a Metal implementation.
    return backend.lower() == "metal"


_physical_context = threading.local()


def get_last_native_physical_guarded() -> bool:
    return bool(getattr(_physical_context, "guarded", False))


def native_physical_dehaze(
    image: np.ndarray, params: object, transmission: np.ndarray, atmosphere: np.ndarray,
    *, dark_floor: float | None = None,
) -> np.ndarray:
    """Run the optional linear-float physical operator on Metal or D3D12."""
    global _last_native_physical_backend
    _last_native_physical_backend = None
    _physical_context.guarded = False
    if dark_floor is not None and (not math.isfinite(dark_floor) or not 0 <= dark_floor <= 2):
        raise ValueError("Dark background floor must be finite and within [0, 2]")
    if not isinstance(image, np.ndarray):
        raise TypeError("Physical dehaze source must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Physical dehaze source must have shape (height, width, 3) RGB")
    if image.dtype != np.dtype(np.float32):
        raise TypeError("Physical dehaze source must use normalized float32 RGB")
    height, width, _ = image.shape
    if not height or not width or width > 65535 or height > 65535:
        raise ValueError("Physical dehaze dimensions must be within 1..65535")
    value_count = int(image.size)
    if value_count > _MAX_IMAGE_VALUES:
        raise ValueError("Physical dehaze source exceeds the C ABI size limit")
    if not _float_range_valid(image):
        raise ValueError("Physical dehaze source must be finite and within [0, 1]")

    values = _values(params, _DEHAZE_FIELDS, ((0.0, 1.0),) * len(_DEHAZE_FIELDS), "dehaze")
    if not isinstance(transmission, np.ndarray) or transmission.shape != (height, width):
        raise ValueError("Physical dehaze transmission must match image dimensions")
    if not np.issubdtype(transmission.dtype, np.floating):
        raise TypeError("Physical dehaze transmission must use a floating-point dtype")
    if not isinstance(atmosphere, np.ndarray) or atmosphere.shape != (3,):
        raise ValueError("Physical dehaze airlight must contain three channels")
    if not np.issubdtype(atmosphere.dtype, np.floating):
        raise TypeError("Physical dehaze airlight must use a floating-point dtype")
    if not _float_range_valid(transmission):
        raise ValueError("Physical dehaze transmission must be finite and within [0, 1]")
    if not _float_range_valid(atmosphere):
        raise ValueError("Physical dehaze airlight must be finite and within [0, 1]")

    # Own the buffers passed through the const C ABI, even if the caller's
    # arrays are already contiguous. The input photo must remain untouched.
    source32 = np.array(image, dtype=np.float32, order="C", copy=True)
    transmission32 = np.array(transmission, dtype=np.float32, order="C", copy=True)
    atmosphere32 = np.array(atmosphere, dtype=np.float32, order="C", copy=True)
    output = np.empty_like(source32)
    library, _ = _get_library()
    render = getattr(library, "im_renderer_render_physical_float", None)
    if render is None:
        raise NativeRendererError("Native physical float dehaze ABI is unavailable")
    renderer, backend = _create_renderer(library)
    try:
        if not _supports_physical_float(library, renderer, backend):
            raise NativeRendererError(
                f"Native physical float dehaze is unavailable on this backend ({backend})"
            )
        native_params = _DehazeParams(*values)
        guarded_render = getattr(library, "im_renderer_render_physical_guarded_float", None)
        supports_guarded = getattr(library, "im_renderer_supports_physical_guarded_float", None)
        # Metal passed full-resolution Python parity and hardware timing gates.
        # D3D12 remains opt-in pending the same checks on Windows hardware.
        use_guarded = (dark_floor is not None and dark_floor > 1e-8
            and os.environ.get("IMPRINT_NATIVE_GUARDED_FLOAT",
                               "1" if backend == "Metal" else "0") == "1"
            and guarded_render is not None and supports_guarded is not None
            and bool(supports_guarded(renderer)))
        if use_guarded:
            status = guarded_render(renderer, width, height,
                source32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), source32.size,
                transmission32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), transmission32.size,
                atmosphere32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.byref(native_params),
                dark_floor, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size)
        else:
            status = render(
                renderer, width, height,
                source32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), source32.size,
                transmission32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), transmission32.size,
                atmosphere32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(native_params),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size,
            )
        if status != 0:
            raise NativeRendererError(
                f"Native physical float dehaze failed (status {status}): "
                f"{_native_error(library, renderer)}"
            )
        _last_native_physical_backend = backend
        _physical_context.guarded = use_guarded
    finally:
        library.im_renderer_destroy(renderer)
    return output


def get_last_native_physical_backend() -> str | None:
    """Return the backend that completed the latest physical float render."""
    return _last_native_physical_backend


def get_native_physical_status() -> dict[str, object]:
    """Report GPU physical support and the independent C++ CPU fallback."""
    gpu_available = cpu_available = False
    physical_backend = None
    try:
        library, _ = _get_library()
        cpu_available = (os.environ.get("IMPRINT_NATIVE_DENSE", "1") != "0"
                         and getattr(library, "im_native_physical_float_run", None) is not None)
        if getattr(library, "im_renderer_render_physical_float", None) is not None:
            renderer = None
            try:
                renderer, backend = _create_renderer(library)
                gpu_available = _supports_physical_float(library, renderer, backend)
                physical_backend = backend if gpu_available else None
            finally:
                if renderer is not None:
                    library.im_renderer_destroy(renderer)
    except Exception:
        # A missing GPU must not hide the independent CPU kernels.
        pass
    return {"available": gpu_available or cpu_available,
            "backend": physical_backend or ("cpp_cpu" if cpu_available else None),
            "gpu_available": gpu_available, "cpu_available": cpu_available}


def native_gpu_spatial_dehaze(
    image: np.ndarray, params: object, transmission: np.ndarray, atmosphere: np.ndarray,
) -> np.ndarray:
    """Render spatial dehaze on the platform GPU, or raise for CPU fallback."""
    rgb16, width, height, was_uint8 = _prepare_image(image)
    values = _values(params, _DEHAZE_FIELDS, ((0.0, 1.0),) * len(_DEHAZE_FIELDS), "dehaze")
    if not isinstance(transmission, np.ndarray) or transmission.shape != (height, width):
        raise ValueError("Spatial transmission must match image dimensions")
    if not isinstance(atmosphere, np.ndarray) or atmosphere.shape != (3,):
        raise ValueError("Spatial airlight must contain three channels")
    if not np.isfinite(transmission).all() or not np.isfinite(atmosphere).all():
        raise ValueError("Spatial transmission and airlight must be finite")
    transmission32 = np.ascontiguousarray(transmission, dtype=np.float32)
    atmosphere32 = np.ascontiguousarray(atmosphere, dtype=np.float32)
    library, _ = _get_library()
    render = getattr(library, "im_renderer_render_spatial_full", None)
    if render is None:
        raise NativeRendererError("GPU spatial dehaze operator is unavailable")
    renderer, backend = _create_renderer(library)
    try:
        if backend.lower() not in ("metal", "d3d12"):
            raise NativeRendererError(f"Spatial GPU renderer is unavailable ({backend})")
        output16 = np.empty_like(rgb16)
        status = render(
            renderer, width, height,
            rgb16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), rgb16.size,
            transmission32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), transmission32.size,
            atmosphere32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(_DehazeParams(*values)),
            output16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), output16.size,
        )
        if status != 0:
            raise NativeRendererError(
                f"GPU spatial dehaze failed (status {status}): {_native_error(library, renderer)}"
            )
    finally:
        library.im_renderer_destroy(renderer)
    if was_uint8:
        return _compact_native_rgb8(output16)
    return output16


def native_spatial_dehaze(
    image: np.ndarray,
    params: object,
    transmission: np.ndarray,
    atmosphere: np.ndarray,
) -> np.ndarray:
    """Run the bounded spatial map through the native C++ pixel operator."""
    rgb16, width, height, was_uint8 = _prepare_image(image)
    values = _values(params, _DEHAZE_FIELDS, ((0.0, 1.0),) * len(_DEHAZE_FIELDS), "dehaze")
    if not isinstance(transmission, np.ndarray) or transmission.shape != (height, width):
        raise ValueError("Spatial transmission must match image dimensions")
    if not isinstance(atmosphere, np.ndarray) or atmosphere.shape != (3,):
        raise ValueError("Spatial airlight must contain three channels")
    if not np.isfinite(transmission).all() or not np.isfinite(atmosphere).all():
        raise ValueError("Spatial transmission and airlight must be finite")
    transmission32 = np.ascontiguousarray(transmission, dtype=np.float32)
    atmosphere32 = np.ascontiguousarray(atmosphere, dtype=np.float32)
    library, _ = _get_library()
    spatial = getattr(library, "im_native_dehaze_spatial_run", None)
    if spatial is None:
        raise NativeRendererError("Native spatial dehaze operator is unavailable")
    output16 = np.empty_like(rgb16)
    status = spatial(
        rgb16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), width, height,
        transmission32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), transmission32.size,
        atmosphere32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(_DehazeParams(*values)),
        output16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), output16.size,
    )
    if status != 0:
        raise NativeRendererError(f"Native spatial dehaze failed (status {status})")
    if was_uint8:
        return _compact_native_rgb8(output16)
    return output16


def get_native_spatial_status() -> dict[str, object]:
    """Report actual spatial CPU and GPU availability separately."""
    gpu_backend: str | None = None
    try:
        library, _ = _get_library()
        available = getattr(library, "im_native_dehaze_spatial_run", None) is not None
        if getattr(library, "im_renderer_render_spatial_full", None) is not None:
            try:
                renderer, backend = _create_renderer(library)
                try:
                    if backend.lower() in ("metal", "d3d12"):
                        gpu_backend = backend
                finally:
                    library.im_renderer_destroy(renderer)
            except NativeRendererError:
                pass
    except NativeRendererError:
        available = False
    return {"available": available or gpu_backend is not None,
            "backend": gpu_backend or ("C++ CPU" if available else None),
            "gpu_available": gpu_backend is not None,
            "cpu_available": available}


def _get_library() -> tuple[ctypes.CDLL, str]:
    global _cached_library, _cached_library_path
    with _load_lock:
        if _cached_library is not None and _cached_library_path is not None:
            return _cached_library, _cached_library_path
        candidates = _library_candidates()
        explicit = bool(os.environ.get("IMPRINT_NATIVE_RENDERER_LIB"))
        for candidate in candidates:
            try:
                if not candidate.is_file():
                    if explicit:
                        raise NativeRendererError(
                            "Configured native renderer library does not exist or is not a file"
                        )
                    continue
                library = ctypes.CDLL(str(candidate))
                _configure_library(library)
                _cached_library = library
                _cached_library_path = str(candidate.resolve())
                return library, _cached_library_path
            except NativeRendererError:
                raise
            except (OSError, AttributeError):
                continue

        if explicit:
            raise NativeRendererError("Could not load the configured native renderer library")
        raise NativeRendererError("Native renderer library was not found or could not be loaded")


def _native_error(library: ctypes.CDLL, renderer: ctypes.c_void_p | None) -> str:
    try:
        raw = library.im_renderer_last_error(renderer)
    except Exception:
        raw = None
    return raw.decode("utf-8", errors="replace") if raw else "no native error detail"


def _create_renderer(library: ctypes.CDLL, backend_kind: int = 0) -> tuple[ctypes.c_void_p, str]:
    renderer = ctypes.c_void_p()
    # IM_BACKEND_AUTO asks the native factory to choose an available backend.
    status = library.im_renderer_create(backend_kind, ctypes.byref(renderer))
    if status != 0 or not renderer.value:
        detail = _native_error(library, None)
        if renderer.value:
            library.im_renderer_destroy(renderer)
        raise NativeRendererError(
            f"Could not create a native renderer (status {status}): {detail}"
        )
    name = library.im_renderer_backend_name(renderer)
    backend = name.decode("utf-8", errors="replace") if name else "unknown"
    return renderer, backend


class NativeLensRemapper:
    """Per-operation streaming Metal RGB16 Lensfun resampler.

    The full source is uploaded once. Call :meth:`remap` with absolute source
    coordinate strips and their destination y offsets, then :meth:`finish` to
    read the complete result. The object owns only its renderer handle; it does
    not retain the NumPy source or coordinate arrays.
    """

    def __init__(self, source, interpolation_mode: int):
        source = np.asarray(source)
        if source.dtype != np.uint16 or source.ndim != 3 or source.shape[2] != 3:
            raise NativeRendererError("Native lens remap source must be RGB uint16")
        if not source.flags.c_contiguous:
            raise NativeRendererError("Native lens remap source must be C-contiguous")
        if interpolation_mode not in (0, 1):
            raise NativeRendererError("Native lens remap interpolation mode must be 0 or 1")
        height, width, _ = source.shape
        if width < 1 or height < 1:
            raise NativeRendererError("Native lens remap source must have nonzero dimensions")

        self._library = None
        self._renderer = None
        self._width = int(width)
        self._height = int(height)
        self._mode = int(interpolation_mode)
        self._finished = False

        library, _ = _get_library()
        required = (
            "im_renderer_supports_lens_remap",
            "im_renderer_lens_remap_begin",
            "im_renderer_lens_remap_strip",
            "im_renderer_lens_remap_finish",
        )
        if any(getattr(library, name, None) is None for name in required):
            raise NativeRendererError("Streaming GPU lens remap ABI is unavailable in this library")
        _configure_library(library)
        renderer, backend = _create_renderer(library, backend_kind=1)  # IM_BACKEND_METAL
        if backend.lower() != "metal" or not library.im_renderer_supports_lens_remap(renderer):
            library.im_renderer_destroy(renderer)
            raise NativeRendererError("Streaming RGB16 lens remap is unavailable on Metal")
        self._library = library
        self._renderer = renderer
        status = library.im_renderer_lens_remap_begin(
            renderer, self._width, self._height,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), source.size,
        )
        if status != 0:
            detail = _native_error(library, renderer)
            self.close()
            raise NativeRendererError(f"Could not upload streaming lens remap source: {detail}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._height, self._width, 3)

    def _require_open(self) -> tuple[ctypes.CDLL, ctypes.c_void_p]:
        if self._renderer is None or self._library is None or self._finished:
            raise NativeRendererError("Native lens remapper is closed or already finished")
        return self._library, self._renderer

    def remap(self, maps, y: int) -> None:
        library, renderer = self._require_open()
        maps = np.asarray(maps)
        if maps.dtype != np.float32:
            raise NativeRendererError("Native lens remap coordinate strips must be float32")
        expected_shared = (None, self._width, 2)
        expected_channel = (None, self._width, 3, 2)
        if (maps.ndim == 3 and maps.shape[1:] == expected_shared[1:]):
            map_channels = 1
        elif maps.ndim == 4 and maps.shape[1:] == expected_channel[1:]:
            map_channels = 3
        else:
            raise NativeRendererError("Native lens remap strip must have shape (h,w,2) or (h,w,3,2)")
        strip_height = int(maps.shape[0])
        if strip_height <= 0 or not isinstance(y, (int, np.integer)):
            raise NativeRendererError("Native lens remap strip offset and height are invalid")
        strip_y = int(y)
        if strip_y < 0 or strip_y + strip_height > self._height:
            raise NativeRendererError("Native lens remap strip lies outside the output image")
        if not maps.flags.c_contiguous:
            maps = np.ascontiguousarray(maps)
        status = library.im_renderer_lens_remap_strip(
            renderer, maps.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), maps.size,
            strip_y, strip_height, map_channels, self._mode,
        )
        if status != 0:
            raise NativeRendererError(f"Streaming lens remap strip failed: {_native_error(library, renderer)}")

    def finish(self, destination) -> None:
        library, renderer = self._require_open()
        if (not isinstance(destination, np.ndarray) or destination.dtype != np.uint16 or
                destination.shape != self.shape or not destination.flags.c_contiguous or
                not destination.flags.writeable):
            raise NativeRendererError("Native lens remap destination must be writable contiguous RGB uint16")
        status = library.im_renderer_lens_remap_finish(
            renderer, destination.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            destination.size,
        )
        if status != 0:
            raise NativeRendererError(f"Streaming lens remap readback failed: {_native_error(library, renderer)}")
        self._finished = True

    def close(self) -> None:
        renderer = self._renderer
        library = self._library
        self._renderer = None
        self._library = None
        if renderer is not None and library is not None:
            library.im_renderer_destroy(renderer)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _values(params: object, fields: tuple[str, ...], limits: tuple[tuple[float, float], ...], label: str):
    if isinstance(params, Mapping):
        if set(params) != set(fields):
            raise ValueError(f"{label} parameters must contain exactly {', '.join(fields)}")
        raw_values = [params[field] for field in fields]
    else:
        try:
            raw_values = [getattr(params, field) for field in fields]
        except AttributeError as exc:
            raise TypeError(f"{label} parameters must be a mapping or expose all fields") from exc

    values: list[float] = []
    for field, raw, (minimum, maximum) in zip(fields, raw_values, limits):
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label}.{field} must be a finite number") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{label}.{field} must be between {minimum:g} and {maximum:g}")
        values.append(value)
    return values


def _compact_native_rgb8(output16: np.ndarray) -> np.ndarray:
    """Pack GPU output in one CPU pass, with the original integer fallback."""
    try:
        from native_dense import compact_rgb16_to_rgb8

        return compact_rgb16_to_rgb8(output16)
    except (ImportError, NativeRendererError):
        return ((output16.astype(np.uint32) + 128) // 257).astype(np.uint8)


def _prepare_image(image: object) -> tuple[np.ndarray, int, int, bool]:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Native renderer input must have shape (height, width, 3) RGB")
    if array.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise TypeError("Native renderer input must use uint8 or uint16 samples")
    height, width, _ = array.shape
    if height == 0 or width == 0 or width > 65535 or height > 65535:
        raise ValueError("Native renderer input dimensions must be within 1..65535")
    value_count = int(array.size)
    if value_count > _MAX_IMAGE_VALUES:
        raise ValueError("Native renderer input exceeds the C ABI size limit")

    was_uint8 = array.dtype == np.dtype(np.uint8)
    if was_uint8:
        try:
            from native_dense import expand_rgb8_to_rgb16

            # The fused conversion already owns contiguous storage. Keep the
            # caller protected without copying this fresh allocation again.
            rgb16 = expand_rgb8_to_rgb16(array)
            return rgb16, int(width), int(height), was_uint8
        except (ImportError, NativeRendererError):
            pass
        # Exact full-range expansion: 0 -> 0 and 255 -> 65535.
        rgb16 = array.astype(np.uint16) * np.uint16(257)
    else:
        rgb16 = array
    # Give the const C ABI its own contiguous storage. A native bug cannot
    # mutate the caller's image, including when the source is already contiguous.
    rgb16 = np.array(rgb16, dtype=np.uint16, order="C", copy=True)
    return rgb16, int(width), int(height), was_uint8


def _render_one_stage(image: object, params: object, *, stage: str) -> np.ndarray:
    rgb16, width, height, was_uint8 = _prepare_image(image)
    if stage == "dehaze":
        dehaze_values = _values(
            params, _DEHAZE_FIELDS, ((0.0, 1.0),) * len(_DEHAZE_FIELDS), "dehaze"
        )
        basic_values = [0.0] * len(_BASIC_FIELDS)
    elif stage == "basic":
        dehaze_values = [0.0] * len(_DEHAZE_FIELDS)
        basic_values = _values(params, _BASIC_FIELDS, _BASIC_LIMITS, "basic")
    else:  # Internal guard.
        raise ValueError(f"Unknown native renderer stage: {stage}")

    library, _ = _get_library()
    renderer, _ = _create_renderer(library)
    try:
        dehaze = _DehazeParams(*dehaze_values)
        basic = _BasicParams(*basic_values)
        pointer = rgb16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        status = library.im_renderer_render_full(
            renderer,
            width,
            height,
            pointer,
            rgb16.size,
            ctypes.byref(dehaze),
            ctypes.byref(basic),
        )
        if status != 0:
            raise NativeRendererError(
                f"Native {stage} render failed (status {status}): "
                f"{_native_error(library, renderer)}"
            )

        output_width = ctypes.c_uint32()
        output_height = ctypes.c_uint32()
        output_count = ctypes.c_size_t()
        status = library.im_renderer_get_output_size(
            renderer,
            ctypes.byref(output_width),
            ctypes.byref(output_height),
            ctypes.byref(output_count),
        )
        if status != 0:
            raise NativeRendererError(
                f"Could not query native output size (status {status}): "
                f"{_native_error(library, renderer)}"
            )
        if (
            output_width.value != width
            or output_height.value != height
            or output_count.value != rgb16.size
        ):
            raise NativeRendererError(
                "Native renderer returned an output size that does not match its input"
            )

        output16 = np.empty((height, width, 3), dtype=np.uint16)
        output_pointer = output16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        status = library.im_renderer_copy_output(renderer, output_pointer, output16.size)
        if status != 0:
            raise NativeRendererError(
                f"Could not copy native output (status {status}): "
                f"{_native_error(library, renderer)}"
            )
    finally:
        library.im_renderer_destroy(renderer)

    if was_uint8:
        # Integer round-to-nearest while mapping the native 16-bit full range
        # back to all 256 uint8 codes.
        return _compact_native_rgb8(output16)
    return output16


def native_dehaze(image: object, params: object) -> np.ndarray:
    """Apply only native dehaze to an RGB uint8/uint16 image.

    Output dtype matches input. Raises :class:`NativeRendererError` when the
    library/backend is unavailable so the caller can use its Python fallback.
    """
    return _render_one_stage(image, params, stage="dehaze")


def _destroy_preview_entry(entry: _PreviewCacheEntry) -> None:
    entry.library.im_renderer_destroy(entry.renderer)


def _clear_preview_cache() -> None:
    """Destroy cached preview handles. Intended for orderly shutdown and tests."""
    with _preview_cache_lock:
        entries = list(_preview_cache.values())
        _preview_cache.clear()
        for entry in entries:
            with entry.lock:
                _destroy_preview_entry(entry)


def _render_preview_entry(
    entry: _PreviewCacheEntry,
    params: object,
    level: int,
) -> np.ndarray:
    dehaze_values = _values(
        params, _DEHAZE_FIELDS, ((0.0, 1.0),) * len(_DEHAZE_FIELDS), "dehaze"
    )
    dehaze = _DehazeParams(*dehaze_values)
    basic = _BasicParams(*([0.0] * len(_BASIC_FIELDS)))
    library = entry.library
    renderer = entry.renderer

    status = library.im_renderer_render(
        renderer, level, ctypes.byref(dehaze), ctypes.byref(basic)
    )
    if status != 0:
        raise NativeRendererError(
            f"Native dehaze preview render failed (status {status}): "
            f"{_native_error(library, renderer)}"
        )

    output_width = ctypes.c_uint32()
    output_height = ctypes.c_uint32()
    output_count = ctypes.c_size_t()
    status = library.im_renderer_get_output_size(
        renderer,
        ctypes.byref(output_width),
        ctypes.byref(output_height),
        ctypes.byref(output_count),
    )
    if status != 0:
        raise NativeRendererError(
            f"Could not query native preview output size (status {status}): "
            f"{_native_error(library, renderer)}"
        )
    expected_count = int(output_width.value) * int(output_height.value) * 3
    if (
        output_width.value == 0
        or output_height.value == 0
        or output_width.value > 65535
        or output_height.value > 65535
        or expected_count > _MAX_IMAGE_VALUES
        or output_count.value != expected_count
    ):
        raise NativeRendererError("Native preview renderer returned an invalid output size")

    output16 = np.empty((output_height.value, output_width.value, 3), dtype=np.uint16)
    output_pointer = output16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    status = library.im_renderer_copy_output(renderer, output_pointer, output16.size)
    if status != 0:
        raise NativeRendererError(
            f"Could not copy native preview output (status {status}): "
            f"{_native_error(library, renderer)}"
        )
    if entry.was_uint8:
        return _compact_native_rgb8(output16)
    return output16


def native_dehaze_preview(
    cache_key: tuple[object, ...],
    image_factory: Callable[[], tuple[np.ndarray, object]],
    params: object,
    level: int,
) -> tuple[np.ndarray, object]:
    """Render a cached preview image with native dehaze at L0, L1, or L2.

    ``cache_key`` must identify the preview source and its decoding dimensions.
    ``image_factory`` is called only for a cache miss and returns an RGB uint8 or
    uint16 image plus lightweight metadata. The returned ndarray has the input
    dtype; the metadata is the value returned by the factory on the first miss.
    Native failures raise :class:`NativeRendererError` so callers can fall back.
    """
    if not isinstance(cache_key, tuple):
        raise TypeError("Native preview cache key must be a tuple")
    try:
        hash(cache_key)
    except TypeError as exc:
        raise TypeError("Native preview cache key values must be hashable") from exc
    if isinstance(level, bool) or not isinstance(level, int) or level not in (0, 1, 2):
        raise ValueError("Native preview level must be 0, 1, or 2")
    if not callable(image_factory):
        raise TypeError("Native preview image_factory must be callable")

    # Validate parameters before creating or touching a renderer. Basic controls
    # intentionally remain zero: the existing API merges them with Ricoh state.
    _values(params, _DEHAZE_FIELDS, ((0.0, 1.0),) * len(_DEHAZE_FIELDS), "dehaze")

    with _preview_cache_lock:
        entry = _preview_cache.get(cache_key)
        if entry is not None:
            _preview_cache.move_to_end(cache_key)
            entry.lock.acquire()
        else:
            creation_lock = _preview_creation_locks.get(cache_key)
            if creation_lock is None:
                creation_lock = threading.Lock()
                _preview_creation_locks[cache_key] = creation_lock
    if entry is None:
        # Calls for one photo share a first decode, while another photo can
        # decode without waiting for it. The weak lock table stays bounded by
        # requests in flight rather than all photos ever previewed.
        with creation_lock:
            with _preview_cache_lock:
                entry = _preview_cache.get(cache_key)
                if entry is not None:
                    _preview_cache.move_to_end(cache_key)
                    entry.lock.acquire()
            if entry is None:
                library, _ = _get_library()
                renderer, _ = _create_renderer(library)
                try:
                    factory_result = image_factory()
                    if not isinstance(factory_result, tuple) or len(factory_result) != 2:
                        raise TypeError("Native preview image_factory must return (image, metadata)")
                    image, metadata = factory_result
                    rgb16, width, height, was_uint8 = _prepare_image(image)
                    pointer = rgb16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
                    status = library.im_renderer_upload_preview_image(
                        renderer, width, height, pointer, rgb16.size
                    )
                    if status != 0:
                        raise NativeRendererError(
                            f"Could not upload native preview image (status {status}): "
                            f"{_native_error(library, renderer)}"
                        )
                    entry = _PreviewCacheEntry(
                        library=library,
                        renderer=renderer,
                        metadata=metadata,
                        width=width,
                        height=height,
                        was_uint8=was_uint8,
                    )
                    entry.lock.acquire()
                    with _preview_cache_lock:
                        _preview_cache[cache_key] = entry
                        while len(_preview_cache) > _PREVIEW_CACHE_LIMIT:
                            evicted_key, evicted = _preview_cache.popitem(last=False)
                            if evicted_key == cache_key:
                                _preview_cache[evicted_key] = evicted
                                raise RuntimeError("Native preview cache evicted its new entry")
                            with evicted.lock:
                                _destroy_preview_entry(evicted)
                except BaseException:
                    library.im_renderer_destroy(renderer)
                    raise

    try:
        result = _render_preview_entry(entry, params, level)
        return result, entry.metadata
    finally:
        entry.lock.release()


def native_basic(image: object, params: object) -> np.ndarray:
    """Apply basic adjustments, using the corrected CPU path for nonzero EV.

    The GPU shader ABI predates the linear-light exposure correction. Zero-EV
    calls retain the GPU path; nonzero exposure uses the v2 CPU kernel or its
    NumPy fallback.
    """
    values = _values(params, _BASIC_FIELDS, _BASIC_LIMITS, "basic")
    if values[0] != 0.0:
        from ricoh_filter import apply_basic_preview_effect

        return apply_basic_preview_effect(image, dict(zip(_BASIC_FIELDS, values)))
    return _render_one_stage(image, params, stage="basic")


def _ricoh_basic_values(basic_params: object) -> dict[str, float]:
    if basic_params is None:
        return dict.fromkeys(_BASIC_FIELDS, 0.0)
    if not isinstance(basic_params, Mapping):
        raise TypeError("Ricoh basic parameters must be a mapping or None")
    # Reuse the same validation and defaults as the Python Ricoh preview path.
    from ricoh_filter import validate_basic_params

    return validate_basic_params(dict(basic_params))


def _ricoh_lut(preset_id: str, basic_params: object, use_measured_color: bool) -> np.ndarray:
    if not isinstance(preset_id, str):
        raise TypeError("Ricoh preset id must be a string")
    basic = _ricoh_basic_values(basic_params)
    measured = bool(use_measured_color)
    edge = _RICOH_LUT_EDGE
    key = (preset_id, tuple(float(basic[field]) for field in _BASIC_FIELDS), measured, edge)
    with _ricoh_lut_lock:
        cached = _ricoh_lut_cache.get(key)
        if cached is not None:
            _ricoh_lut_cache.move_to_end(key)
            return cached

    # Sampling is performed by the Python preview effect itself. The grid's
    # flattened order matches the native ABI: ((r * edge + g) * edge + b).
    import numpy as np
    from ricoh_filter import apply_ricoh_preview_effect

    coordinates = np.rint(np.linspace(0.0, 65535.0, edge, dtype=np.float64)).astype(np.uint16)
    samples = np.empty((edge * edge, edge, 3), dtype=np.uint16)
    flat_samples = samples.reshape(-1, 3)
    flat_samples[:, 0] = np.repeat(coordinates, edge * edge)
    flat_samples[:, 1] = np.tile(np.repeat(coordinates, edge), edge)
    flat_samples[:, 2] = np.tile(coordinates, edge * edge)
    rendered = apply_ricoh_preview_effect(
        samples, preset_id, basic, use_measured_color=measured
    )
    lut = np.ascontiguousarray(rendered.reshape(-1), dtype=np.uint16)
    lut.setflags(write=False)

    with _ricoh_lut_lock:
        cached = _ricoh_lut_cache.get(key)
        if cached is not None:
            _ricoh_lut_cache.move_to_end(key)
            return cached
        _ricoh_lut_cache[key] = lut
        while len(_ricoh_lut_cache) > _RICOH_LUT_CACHE_LIMIT:
            _ricoh_lut_cache.popitem(last=False)
    return lut


def native_ricoh(
    image: object,
    preset_id: str,
    basic_params: object = None,
    use_measured_color: bool = False,
) -> np.ndarray:
    """Apply the Python Ricoh preview effect through a cached 3D LUT.

    Python remains responsible for interpreting the preset. The native renderer
    performs RGB16 pixel interpolation and is an approximation of that effect.
    It raises :class:`NativeRendererError` when native execution fails, allowing
    the caller to use the existing Python fallback.
    """
    rgb16, width, height, was_uint8 = _prepare_image(image)
    if not isinstance(preset_id, str):
        raise TypeError("Ricoh preset id must be a string")
    basic = _ricoh_basic_values(basic_params)
    measured = bool(use_measured_color)
    from ricoh_filter import list_ricoh_presets

    if preset_id not in {preset["id"] for preset in list_ricoh_presets()}:
        raise KeyError(preset_id)
    library, _ = _get_library()
    renderer, _ = _create_renderer(library)
    try:
        # Avoid generating a 65^3 Python grid on hosts without a usable GPU.
        # The handle is still destroyed if LUT preparation raises.
        lut = _ricoh_lut(preset_id, basic, measured)
        image_pointer = rgb16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        lut_pointer = lut.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        status = library.im_renderer_render_ricoh_full(
            renderer,
            width,
            height,
            image_pointer,
            rgb16.size,
            lut_pointer,
            lut.size,
            _RICOH_LUT_EDGE,
        )
        if status != 0:
            raise NativeRendererError(
                f"Native Ricoh render failed (status {status}): "
                f"{_native_error(library, renderer)}"
            )

        output_width = ctypes.c_uint32()
        output_height = ctypes.c_uint32()
        output_count = ctypes.c_size_t()
        status = library.im_renderer_get_output_size(
            renderer,
            ctypes.byref(output_width),
            ctypes.byref(output_height),
            ctypes.byref(output_count),
        )
        if status != 0:
            raise NativeRendererError(
                f"Could not query native Ricoh output size (status {status}): "
                f"{_native_error(library, renderer)}"
            )
        if output_width.value != width or output_height.value != height or output_count.value != rgb16.size:
            raise NativeRendererError("Native Ricoh output size does not match its input")

        output16 = np.empty((height, width, 3), dtype=np.uint16)
        output_pointer = output16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        status = library.im_renderer_copy_output(renderer, output_pointer, output16.size)
        if status != 0:
            raise NativeRendererError(
                f"Could not copy native Ricoh output (status {status}): "
                f"{_native_error(library, renderer)}"
            )
    finally:
        library.im_renderer_destroy(renderer)

    if was_uint8:
        return _compact_native_rgb8(output16)
    return output16


def _native_acr_color_sdr(rgb: np.ndarray, controls: object, symbol: str,
                          control_count: int | tuple[int, ...]) -> np.ndarray:
    """Run one recovered CPU color stage; input/output is linear ProPhoto.

    This entry point does not select a profile, apply tone/display conversion,
    or certify Adobe output. Older native libraries fail explicitly.
    """
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
            or rgb.size > _MAX_IMAGE_VALUES):
        raise ValueError("ACR color input must be a nonempty float32 HxWx3 array")
    values = np.ascontiguousarray(controls, dtype=np.float64)
    counts = (control_count,) if isinstance(control_count, int) else control_count
    if values.ndim != 1 or values.size not in counts or not np.isfinite(values).all():
        raise ValueError(f"ACR color controls must contain {control_count} finite values")
    source = np.ascontiguousarray(rgb)
    library, _ = _get_library()
    function = getattr(library, symbol, None)
    if function is None:
        raise NativeRendererError("Native library does not provide recovered SDR color operators")
    output = np.empty_like(source)
    height, width = source.shape[:2]
    status = function(width, height, source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      source.size, values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      values.size, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size)
    if status == 1:
        raise ValueError("Native SDR color operator rejected input range or controls")
    if status:
        raise NativeRendererError(f"Native SDR color operator failed (status {status})")
    return output


def native_acr_white_temperature(xy: object) -> tuple[float, float]:
    """Recover temperature/tint with the C++ CPU fixed-table projection."""
    values = np.asarray(xy, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("white chromaticity must contain two finite values")
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_white_temperature", None)
    if function is None:
        raise NativeRendererError("Native library does not provide recovered white projection")
    result = np.empty(2, dtype=np.float64)
    status = function(float(values[0]), float(values[1]),
                      result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2)
    if status == 1:
        raise ValueError("Native white projection rejected chromaticity")
    if status:
        raise NativeRendererError(f"Native white projection failed (status {status})")
    return float(result[0]), float(result[1])


def native_acr_illuminant_temperatures(codes: object) -> tuple[float, float]:
    """Resolve two ACR illuminant codes through the recovered native lookup."""
    try:
        values = tuple(codes)
    except TypeError as exc:
        raise ValueError("illuminant codes must contain exactly two integers") from exc
    if len(values) != 2:
        raise ValueError("illuminant codes must contain exactly two integers")
    checked: list[int] = []
    for value in values:
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer))):
            raise ValueError("illuminant codes must be non-boolean integers")
        code = int(value)
        if not 0 <= code <= 65535 or code == 255:
            raise ValueError("illuminant code is outside the supported range")
        checked.append(code)

    library, _ = _get_library()
    function = getattr(library, "im_native_acr_illuminant_temperatures", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR illuminant lookup")
    function.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    function.restype = ctypes.c_int
    code_array = (ctypes.c_uint32 * 2)(*checked)
    output = np.empty(2, dtype=np.float64)
    status = function(code_array, 2,
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2)
    if status == 1:
        raise ValueError("Native ACR illuminant lookup rejected input")
    if status:
        raise NativeRendererError(f"Native ACR illuminant lookup failed (status {status})")
    return float(output[0]), float(output[1])


def native_acr_hsl_sdr(rgb: np.ndarray, controls: object) -> np.ndarray:
    """Apply H/S/L jointly, with optional 25th Vibrance, using C++ CPU."""
    return _native_acr_color_sdr(rgb, controls, "im_native_acr_hsl_sdr_rgbf32", (24, 25))


def native_acr_grading_sdr(rgb: np.ndarray, controls: object) -> np.ndarray:
    """Apply four color wheels, Blending and Balance using C++ CPU."""
    return _native_acr_color_sdr(rgb, controls, "im_native_acr_grading_sdr_rgbf32", 14)


def native_acr_flare_u16(rgb: np.ndarray, controls: object) -> np.ndarray:
    """Compute the explicit uint16 edge2 source-pyramid flare subset in C++."""
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint16
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
            or rgb.size > _MAX_IMAGE_VALUES):
        raise ValueError("ACR flare requires nonempty uint16 HxWx3 input")
    values = np.ascontiguousarray(controls, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("ACR flare requires seven finite prepared controls")
    source = np.ascontiguousarray(rgb)
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_flare_u16_rgb", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR source flare")
    output = np.empty(6, dtype=np.float64)
    height, width = source.shape[:2]
    status = function(width, height, source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                      source.size, values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      values.size, output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), output.size)
    if status == 1:
        raise ValueError("Native ACR source flare rejected input or controls")
    if status:
        raise NativeRendererError(f"Native ACR source flare failed (status {status})")
    return output


def native_acr_project_srgb(rgb: np.ndarray, prophoto_to_pcs: object) -> np.ndarray:
    """Project an explicit native working-space matrix to Imprint display sRGB8."""
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
            or rgb.size > _MAX_IMAGE_VALUES):
        raise ValueError("ACR display projection requires nonempty float32 HxWx3 input")
    matrix = np.ascontiguousarray(prophoto_to_pcs, dtype=np.float64).reshape(-1)
    if matrix.shape != (9,) or not np.isfinite(matrix).all():
        raise ValueError("ACR display projection requires nine finite matrix values")
    source = np.ascontiguousarray(rgb)
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_project_srgb_rgb8", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR display projection")
    output = np.empty(source.shape, dtype=np.uint8)
    height, width = source.shape[:2]
    status = function(width, height, source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      source.size, matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      matrix.size, output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), output.size)
    if status == 1:
        raise ValueError("Native ACR display projection rejected input")
    if status:
        raise NativeRendererError(f"Native ACR display projection failed (status {status})")
    return output


def native_acr_scale_sdr(rgb: np.ndarray, gain: float) -> np.ndarray:
    """Preserve Imprint's existing clipped linear gain using native CPU pixels."""
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
            or rgb.size > _MAX_IMAGE_VALUES or not np.isfinite(gain) or gain < 0):
        raise ValueError("ACR SDR scaling requires nonempty float32 RGB and finite nonnegative gain")
    source = np.ascontiguousarray(rgb)
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_scale_sdr_rgbf32", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR SDR scaling")
    output = np.empty_like(source)
    height, width = source.shape[:2]
    status = function(width, height, source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      source.size, float(gain), output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      output.size)
    if status == 1:
        raise ValueError("Native ACR SDR scaling rejected input or gain")
    if status:
        raise NativeRendererError(f"Native ACR SDR scaling failed (status {status})")
    return output


def native_acr_prepare_profile_camera(controls: object) -> np.ndarray:
    """Compose native neutral, ColorSpec and camera parameter factories."""
    values = np.ascontiguousarray(controls, dtype=np.float64)
    if values.shape != (75,) or not np.isfinite(values).all():
        raise ValueError("ACR profile-camera preparation requires 75 finite controls")
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_prepare_profile_camera", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR profile-camera preparation")
    output = np.empty(65, dtype=np.float64)
    status = function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), output.size)
    if status == 1:
        raise ValueError("Native ACR profile-camera preparation rejected controls")
    if status:
        raise NativeRendererError(f"Native ACR profile-camera preparation failed (status {status})")
    return output


def native_acr_prepare_analog_balance(balance: object | None) -> np.ndarray:
    """Prepare raw source AnalogBalance in C++, preserving its absence default."""
    values = np.ascontiguousarray([] if balance is None else balance, dtype=np.float64)
    if values.shape not in ((0,), (3,)) or not np.isfinite(values).all():
        raise ValueError("ACR analog balance requires absent or three finite values")
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_prepare_analog_balance", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR analog balance preparation")
    output = np.empty(3, dtype=np.float64)
    status = function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), output.size)
    if status == 1:
        raise ValueError("Native ACR analog balance preparation rejected values")
    if status:
        raise NativeRendererError(f"Native ACR analog balance preparation failed (status {status})")
    return output


def native_acr_prepare_camera_calibration(matrix: object) -> np.ndarray:
    """Quantize one source RGB calibration matrix using the native source reader."""
    values = np.ascontiguousarray(matrix, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise ValueError("ACR camera calibration requires a finite 3x3 matrix")
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_prepare_camera_calibration", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR camera calibration preparation")
    output = np.empty((3, 3), dtype=np.float64)
    status = function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), output.size)
    if status == 1:
        raise ValueError("Native ACR camera calibration preparation rejected matrix")
    if status:
        raise NativeRendererError(f"Native ACR camera calibration preparation failed (status {status})")
    return output


def native_acr_prepare_camera_neutral(neutral: object) -> np.ndarray:
    """Prepare three source CameraNeutral fields using the native scalar setter."""
    values = np.ascontiguousarray(neutral, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("ACR neutral preparation requires three finite values")
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_prepare_camera_neutral", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR neutral preparation")
    output = np.empty(3, dtype=np.float64)
    status = function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), output.size)
    if status == 1:
        raise ValueError("Native ACR neutral preparation rejected values")
    if status:
        raise NativeRendererError(f"Native ACR neutral preparation failed (status {status})")
    return output


def native_acr_resolve_white(controls: object) -> np.ndarray:
    """Resolve a prepared source neutral and explicit mode using C++ CPU.

    The 68-input/26-output layout is documented in the native header. This
    adapter does not normalize metadata, infer WB or generate algorithm output.
    """
    values = np.ascontiguousarray(controls, dtype=np.float64)
    if values.shape != (68,) or not np.isfinite(values).all():
        raise ValueError("ACR white resolution requires 68 finite prepared controls")
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_resolve_white", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR white resolution")
    output = np.empty(26, dtype=np.float64)
    status = function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), output.size)
    if status == 1:
        raise ValueError("Native ACR white resolution rejected controls")
    if status:
        raise NativeRendererError(f"Native ACR white resolution failed (status {status})")
    return output


def native_acr_prepare_colorspec(controls: object) -> np.ndarray:
    """Construct a finite RGB ColorSpec in C++ from explicit profile and white data.

    Caller resolves white xy, illuminant endpoints and compatible calibration.
    The 64-input/22-output layout is documented in the native header.
    """
    values = np.ascontiguousarray(controls, dtype=np.float64)
    if values.shape != (64,) or not np.isfinite(values).all():
        raise ValueError("ACR ColorSpec requires 64 finite prepared controls")
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_prepare_colorspec", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR ColorSpec preparation")
    output = np.empty(22, dtype=np.float64)
    status = function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), output.size)
    if status == 1:
        raise ValueError("Native ACR ColorSpec preparation rejected controls")
    if status:
        raise NativeRendererError(f"Native ACR ColorSpec preparation failed (status {status})")
    return output


def native_acr_prepare_camera_stage(controls: object) -> np.ndarray:
    """Prepare camera matrices, gains and compensation in C++ CPU.

    The 31 inputs must already contain resolved ColorSpec and AnalogBalance
    data. See the native header for ordering; no metadata inference or Python
    numerical fallback is performed here.
    """
    values = np.ascontiguousarray(controls, dtype=np.float64)
    if values.shape != (31,) or not np.isfinite(values).all():
        raise ValueError("ACR camera preparation requires 31 finite prepared controls")
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_prepare_camera_stage", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR camera preparation")
    output = np.empty(30, dtype=np.float64)
    status = function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), output.size)
    if status == 1:
        raise ValueError("Native ACR camera preparation rejected controls")
    if status:
        raise NativeRendererError(f"Native ACR camera preparation failed (status {status})")
    return output


def native_acr_source_limit_scene_u16(rgb: np.ndarray, controls: object) -> np.ndarray:
    """Native zero-pedestal scene/no-filter source pyramid through limitEV88.

    The caller establishes this explicit source domain and supplies the 22
    prepared controls. This is not an automatic RAW/metadata adapter.
    """
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint16
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
            or rgb.shape[0] * rgb.shape[1] > (1 << 26)):
        raise ValueError("ACR scene source limit requires nonempty uint16 HxWx3")
    values = np.ascontiguousarray(controls, dtype=np.float64)
    if values.shape != (22,) or not np.isfinite(values).all():
        raise ValueError("ACR scene source limit requires 22 finite prepared controls")
    source = np.ascontiguousarray(rgb)
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_source_limit_scene_u16_rgb", None)
    if function is None:
        raise NativeRendererError("Native library does not provide scene uint16 source limit")
    output = np.empty(24, dtype=np.float64)
    status = function(source.shape[1], source.shape[0],
                      source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), source.size,
                      values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), output.size)
    if status == 1:
        raise ValueError("Native ACR scene source limit rejected prepared input")
    if status:
        raise NativeRendererError(f"Native ACR scene source limit failed (status {status})")
    return output


def native_acr_exposure_limit_log(rgb: np.ndarray, controls: object, epsilon: float) -> np.ndarray:
    """Compute per-plane high statistics and limitEV88 from prepared log pixels.

    This buffer adapter does not prepare a RAW source, infer metadata, or
    substitute display RGB/gray for source camera-plane log samples.
    See the native header for the 17 controls and 15 output lanes.
    """
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
            or rgb.shape[0] * rgb.shape[1] > (1 << 26)):
        raise ValueError("ACR exposure limit requires nonempty prepared float32 log HxWx3")
    values = np.ascontiguousarray(controls, dtype=np.float64)
    if values.shape != (17,) or not np.isfinite(values).all():
        raise ValueError("ACR exposure limit requires 17 finite prepared controls")
    if not np.isfinite(epsilon) or epsilon <= 0 or epsilon > 2.0 ** -12:
        raise ValueError("ACR exposure limit requires source epsilon in (0,2^-12]")
    source = np.ascontiguousarray(rgb)
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_exposure_limit_log_rgbf32", None)
    if function is None:
        raise NativeRendererError("Native library does not provide prepared log exposure limit")
    output = np.empty(15, dtype=np.float64)
    status = function(source.shape[1], source.shape[0],
                      source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), source.size,
                      values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
                      float(epsilon), output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      output.size)
    if status == 1:
        raise ValueError("Native ACR exposure limit rejected prepared input")
    if status:
        raise NativeRendererError(f"Native ACR exposure limit failed (status {status})")
    return output


def native_acr_prepare_exposure(controls: object) -> np.ndarray:
    """Prepare modern EV lanes in C++; source inputs must already be resolved.

    See im_native_acr_prepare_exposure's header contract for input/output order.
    This adapter performs no metadata inference or numerical algorithm fallback.
    """
    values = np.ascontiguousarray(controls, dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("ACR exposure requires ten finite prepared controls")
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_prepare_exposure", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR exposure preparation")
    output = np.empty(19, dtype=np.float64)
    status = function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), output.size)
    if status == 1:
        raise ValueError("Native ACR exposure preparation rejected controls")
    if status:
        raise NativeRendererError(f"Native ACR exposure preparation failed (status {status})")
    return output


def native_acr_profile_tone(rgb: np.ndarray, points: object, *, modern: bool = True,
                            curve_ev18: float = 0.0) -> np.ndarray:
    """Profile natural spline/MMHG, optionally preceded by prepared negative curveEV18.

    curve_ev18 is a resolved native context lane, not manual Exposure.
    """
    if not np.isfinite(curve_ev18) or curve_ev18 < -120 or curve_ev18 > 0:
        raise ValueError("ACR profile negative curveEV18 must be finite in [-120,0]")
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
            or rgb.size > _MAX_IMAGE_VALUES):
        raise ValueError("ACR profile tone requires nonempty float32 HxWx3 input")
    curve = np.ascontiguousarray(points, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 2 or not 2 <= curve.shape[0] <= 65536:
        raise ValueError("ACR profile tone requires 2..65536 x,y points")
    source = np.ascontiguousarray(rgb)
    library, _ = _get_library()
    symbol = ("im_native_acr_profile_tone_rgbf32" if curve_ev18 == 0 else
              "im_native_acr_profile_tone_exposure_rgbf32")
    function = getattr(library, symbol, None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR profile tone")
    output = np.empty_like(source)
    height, width = source.shape[:2]
    arguments = (width, height, source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 source.size, curve.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                 curve.shape[0], int(bool(modern)))
    if curve_ev18 != 0:
        arguments += (float(curve_ev18),)
    status = function(*arguments, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size)
    if status == 1:
        raise ValueError("Native ACR profile tone rejected input or curve")
    if status:
        raise NativeRendererError(f"Native ACR profile tone failed (status {status})")
    return output


def native_acr_rendered_hsl_sdr(rgb: np.ndarray, controls: object) -> np.ndarray:
    """Apply inverse/forward default-tone normalization around HSL/Vibrance."""
    return _native_acr_color_sdr(rgb, controls, "im_native_acr_rendered_hsl_sdr_rgbf32", (24, 25))


def native_acr_camera_float(rgb: np.ndarray, controls: object) -> np.ndarray:
    """Run the recovered CPU camera component on gain-sorted camera RGB.

    Input channels must already be ordered by descending gain and may exceed
    one. The result is interleaved RGB with fixed ProPhoto channel semantics;
    this function does not select a profile or run later rendering stages.
    """
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
            or rgb.size > _MAX_IMAGE_VALUES):
        raise ValueError("ACR camera input must be a nonempty float32 HxWx3 array")
    if not np.isfinite(rgb).all() or np.any(rgb < 0.0):
        raise ValueError("ACR camera input must be finite and nonnegative")
    values = np.asarray(controls, dtype=np.float64)
    if values.shape != (13,) or not np.isfinite(values).all():
        raise ValueError("ACR camera controls must contain 13 finite values")
    if np.any(np.abs(values) > np.finfo(np.float32).max):
        raise ValueError("ACR camera controls must be representable as float32")
    source = np.ascontiguousarray(rgb)
    parameters = np.ascontiguousarray(values)
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_camera_float_rgbf32", None)
    if function is None:
        raise NativeRendererError("Native library does not provide the recovered camera operator")
    output = np.empty_like(source)
    height, width = source.shape[:2]
    status = function(
        width, height, source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), source.size,
        parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), parameters.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size,
    )
    if status == 1:
        raise ValueError("Native camera operator rejected controls or input")
    if status:
        raise NativeRendererError(f"Native camera operator failed (status {status})")
    return output


def native_acr_tone_prepared_sdr(guide_rgb: np.ndarray, tone_rgb: np.ndarray,
                                 controls: object) -> np.ndarray:
    """Run the recovered CPU tone subset on two already prepared RGB domains.

    ``guide_rgb`` is the image domain entering MakeGrayLog after camera and
    earlier preparation; ``tone_rgb`` is the image domain at the enabled tone
    stage. They must share geometry, but can contain different pixel values.
    Neither input is treated as arbitrary sRGB. Controls are source epsilon,
    EV90/98/100, Shadows2012 UI, Highlights2012 UI, nonnegative Clarity2012 UI,
    and integer sigma mode (0..2). Negative clarity and GPU execution are not
    part of this recovered operator.
    """
    for name, image in (("guide", guide_rgb), ("tone", tone_rgb)):
        if (not isinstance(image, np.ndarray) or image.dtype != np.float32
                or image.ndim != 3 or image.shape[2] != 3
                or min(image.shape[:2]) <= 0 or image.size > _MAX_IMAGE_VALUES):
            raise ValueError(f"ACR tone {name} input must be a nonempty float32 HxWx3 array")
        if not np.isfinite(image).all() or np.any(image < 0.0):
            raise ValueError(f"ACR tone {name} input must be finite and nonnegative")
    if guide_rgb.shape != tone_rgb.shape:
        raise ValueError("ACR tone guide and tone inputs must have identical geometry")

    values = np.asarray(controls, dtype=np.float64)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("ACR tone controls must contain 8 finite values")
    if (values[0] <= 0.0 or values[0] > 2.0 ** -12
            or np.float32(values[0]) == 0.0
            or np.any(values[1:4] < -120.0) or np.any(values[1:4] > 120.0)
            or not -100.0 <= values[4] <= 100.0
            or not -100.0 <= values[5] <= 100.0
            or not 0.0 <= values[6] <= 100.0
            or values[7] not in (0.0, 1.0, 2.0)):
        raise ValueError("ACR tone controls are outside the recovered operator domain")

    guide = np.ascontiguousarray(guide_rgb)
    tone = np.ascontiguousarray(tone_rgb)
    parameters = np.ascontiguousarray(values)
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_tone_prepared_sdr_rgbf32", None)
    if function is None:
        raise NativeRendererError("Native library does not provide the recovered tone operator")
    output = np.empty_like(tone)
    height, width = tone.shape[:2]
    status = function(
        width, height,
        guide.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tone.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), tone.size,
        parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), parameters.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size,
    )
    if status == 1:
        raise ValueError("Native tone operator rejected input or controls")
    if status:
        raise NativeRendererError(f"Native tone operator failed (status {status})")
    return output


def native_acr_tone_prepared_sdr_mode(guide_rgb: np.ndarray, tone_rgb: np.ndarray,
                                      controls: object) -> np.ndarray:
    """Run the prepared SDR tone subset with an explicit output mode.

    Controls are the existing eight values followed by output mode (0 keeps
    RGB ratios; 1 writes clamped mapped linear gray). Mode 1 assumes the tone
    stage has already been selected by the caller; this operator does not
    decide whether a stage exists or prepare normalized anchor inputs.
    """
    for name, image in (("guide", guide_rgb), ("tone", tone_rgb)):
        if (not isinstance(image, np.ndarray) or image.dtype != np.float32
                or image.ndim != 3 or image.shape[2] != 3
                or min(image.shape[:2]) <= 0 or image.size > _MAX_IMAGE_VALUES):
            raise ValueError(f"ACR tone {name} input must be a nonempty float32 HxWx3 array")
        if not np.isfinite(image).all() or np.any(image < 0.0):
            raise ValueError(f"ACR tone {name} input must be finite and nonnegative")
    if guide_rgb.shape != tone_rgb.shape:
        raise ValueError("ACR tone guide and tone inputs must have identical geometry")

    values = np.asarray(controls, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("ACR tone mode controls must contain 9 finite values")
    if (values[0] <= 0.0 or values[0] > 2.0 ** -12
            or np.float32(values[0]) == 0.0
            or np.any(values[1:4] < -120.0) or np.any(values[1:4] > 120.0)
            or not -100.0 <= values[4] <= 100.0
            or not -100.0 <= values[5] <= 100.0
            or not 0.0 <= values[6] <= 100.0
            or values[7] not in (0.0, 1.0, 2.0)
            or values[8] not in (0.0, 1.0)):
        raise ValueError("ACR tone mode controls are outside the recovered operator domain")

    guide = np.ascontiguousarray(guide_rgb)
    tone = np.ascontiguousarray(tone_rgb)
    parameters = np.ascontiguousarray(values)
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_tone_prepared_sdr_mode_rgbf32", None)
    if function is None:
        raise NativeRendererError("Native library does not provide the tone output mode operator")
    output = np.empty_like(tone)
    height, width = tone.shape[:2]
    status = function(
        width, height,
        guide.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tone.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), tone.size,
        parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), parameters.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size,
    )
    if status == 1:
        raise ValueError("Native tone output mode operator rejected input or controls")
    if status:
        raise NativeRendererError(f"Native tone output mode operator failed (status {status})")
    return output


def native_acr_exposure_from_anchor(
    rgb: np.ndarray, anchor_log: np.ndarray, controls: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the prepared SDR exposure operator using caller-supplied anchor samples.

    ``anchor_log`` must already contain the correct normalized single-plane log
    samples. This adapter does not assemble RAW metadata or a guide image, and
    it does not choose the stage at which the operator should run.
    Controls are ordered as context98, contextA0, amount58, grayDiffB8,
    WhitesEV80, sourceEpsilon, and finalPixelEpsilon.
    """
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32
            or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) <= 0
            or int(rgb.shape[0]) * int(rgb.shape[1]) > (1 << 26)):
        raise ValueError("ACR anchor exposure requires nonempty float32 HxWx3 RGB")
    if not np.isfinite(rgb).all() or np.any(rgb < 0.0):
        raise ValueError("ACR anchor exposure RGB must be finite and nonnegative")
    if (not isinstance(anchor_log, np.ndarray) or anchor_log.dtype != np.float32
            or anchor_log.ndim not in (1, 2) or anchor_log.size == 0
            or anchor_log.size > (1 << 26)):
        raise ValueError("ACR exposure anchor must be a nonempty 1D or 2D float32 array")
    if not np.isfinite(anchor_log).all():
        raise ValueError("ACR exposure anchor must contain only finite log samples")

    values = np.ascontiguousarray(controls, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("ACR anchor exposure requires 7 finite controls")
    if (np.any(values[:4] < -120.0) or np.any(values[:4] > 120.0)
            or not -2.0 <= values[4] <= 2.0
            or not 0.0 < values[5] <= 2.0 ** -12
            or not 0.0 < values[6] <= 2.0 ** -16
            or np.float32(values[6]) == 0.0):
        raise ValueError("ACR anchor exposure controls are outside the native operator domain")

    source = np.ascontiguousarray(rgb)
    anchor = np.ascontiguousarray(anchor_log).reshape(-1)
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_exposure_from_anchor_rgbf32", None)
    if function is None:
        raise NativeRendererError("Native library does not provide ACR anchor exposure")
    output = np.empty_like(source)
    diagnostics = np.empty(10, dtype=np.float64)
    height, width = source.shape[:2]
    status = function(
        width, height,
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), source.size,
        anchor.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), anchor.size,
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size,
        diagnostics.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), diagnostics.size,
    )
    if status == 1:
        raise ValueError("Native ACR anchor exposure rejected input or controls")
    if status:
        raise NativeRendererError(f"Native ACR anchor exposure failed (status {status})")
    return output, diagnostics


def _prepare_native_acr_gray_log_arguments(
    rgb: np.ndarray, coefficients: object, source_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32
            or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) <= 0
            or int(rgb.shape[0]) * int(rgb.shape[1]) > (1 << 26)):
        raise ValueError("ACR gray log requires nonempty float32 HxWx3 RGB")
    if not np.isfinite(rgb).all() or np.any(rgb < 0.0):
        raise ValueError("ACR gray log RGB must be finite and nonnegative")

    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("ACR gray log requires 3 finite coefficients")
    if np.any(values < -120.0) or np.any(values > 120.0):
        raise ValueError("ACR gray log coefficients are outside [-120, 120]")
    try:
        epsilon = float(source_epsilon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ACR gray log source epsilon must be a finite scalar") from exc
    if (not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 2.0 ** -12
            or np.float32(epsilon) == 0.0):
        raise ValueError("ACR gray log source epsilon is outside the native domain")

    source = np.ascontiguousarray(rgb)
    parameters = np.ascontiguousarray(values)
    return source, parameters, epsilon


def native_acr_gray_log(
    rgb: np.ndarray, coefficients: object, source_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply caller-prepared RGB-to-gray and source-log operations.

    ``rgb`` must already be in the RGB domain expected by the caller, and
    ``coefficients`` must be the resolved row for that domain. This float
    variant returns an HxW float32 source-log plane plus six float64 coefficient
    diagnostics. It does not construct a true RAW anchor, select a stage,
    assemble source/EV inputs, apply a transfer curve, perform the normalized
    render's uint16 store/reload, or Pin overrange values; those remain caller
    responsibilities.
    """
    source, parameters, epsilon = _prepare_native_acr_gray_log_arguments(
        rgb, coefficients, source_epsilon
    )
    library, _ = _get_library()
    function = getattr(library, "im_native_acr_gray_log_rgbf32", None)
    if function is None:
        raise NativeRendererError("Native library does not provide the ACR gray-log operator")
    height, width = source.shape[:2]
    pixel_count = int(height) * int(width)
    output = np.empty((height, width), dtype=np.float32)
    diagnostics = np.empty(6, dtype=np.float64)
    status = function(
        width, height,
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), source.size,
        parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), parameters.size,
        epsilon,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), pixel_count,
        diagnostics.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), diagnostics.size,
    )
    if status == 1:
        raise ValueError("Native ACR gray-log operator rejected input or coefficients")
    if status:
        raise NativeRendererError(f"Native ACR gray-log operator failed (status {status})")
    return output, diagnostics


def native_acr_gray_u16_log(
    rgb: np.ndarray, coefficients: object, source_epsilon: float, *,
    origin_x: int = 0, origin_y: int = 0, dither: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply prepared gray conversion, uint16 store/reload, and source log.

    The caller must provide already-prepared RGB, upstream Pin/stage/source/
    guide/EV decisions, and absolute image origin. The native operation rejects
    derived gray outside [0,1]; this adapter does not clamp or quantize pixels
    in Python. It does not generate a true RAW anchor or claim a complete ACR
    render. Origins are signed int32 coordinates used for the 128x128 dither
    phase; ``dither`` accepts only bool or numpy.bool_.
    """
    source, parameters, epsilon = _prepare_native_acr_gray_log_arguments(
        rgb, coefficients, source_epsilon
    )
    origins: list[int] = []
    for name, value in (("origin_x", origin_x), ("origin_y", origin_y)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))):
            raise ValueError(f"ACR gray uint16 log {name} must be a non-bool integer")
        coordinate = int(value)
        if not -(1 << 31) <= coordinate < (1 << 31):
            raise ValueError(f"ACR gray uint16 log {name} must fit signed int32")
        origins.append(coordinate)
    if not isinstance(dither, (bool, np.bool_)):
        raise ValueError("ACR gray uint16 log dither must be bool or numpy.bool_")

    library, _ = _get_library()
    function = getattr(library, "im_native_acr_gray_u16_log_rgbf32", None)
    if function is None:
        raise NativeRendererError("Native library does not provide the gray uint16-log operator")
    height, width = source.shape[:2]
    pixel_count = int(height) * int(width)
    output = np.empty((height, width), dtype=np.float32)
    diagnostics = np.empty(6, dtype=np.float64)
    status = function(
        width, height,
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), source.size,
        parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), parameters.size,
        epsilon,
        origins[0], origins[1], int(bool(dither)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), pixel_count,
        diagnostics.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), diagnostics.size,
    )
    if status == 1:
        raise ValueError("Native ACR gray uint16-log operator rejected prepared input")
    if status:
        raise NativeRendererError(f"Native ACR gray uint16-log operator failed (status {status})")
    return output, diagnostics


def get_native_status() -> dict[str, object]:
    """Probe actual renderer creation without exposing installation paths."""
    try:
        library, _ = _get_library()
        renderer, backend = _create_renderer(library)
        library.im_renderer_destroy(renderer)
        return {"available": True, "backend": backend}
    except NativeRendererError:
        return {"available": False, "backend": None}
