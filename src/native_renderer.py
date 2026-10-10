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

# One camera source on the GPU, with no retained Python full-image reference.
# Preview cache eviction explicitly destroys this renderer and its allocations.
_camera_profile_gpu_lock = threading.Lock()
_camera_profile_gpu_state: tuple | None = None


def clear_camera_profile_gpu_cache() -> None:
    global _camera_profile_gpu_state
    with _camera_profile_gpu_lock:
        if _camera_profile_gpu_state is not None:
            library, renderer, _, _ = _camera_profile_gpu_state
            _camera_profile_gpu_state = None
            library.im_renderer_destroy(renderer)


def native_camera_profile(source, matrix, white, gain, look, dims, encoding, tone, projection):
    """Exact camera profile on Metal/D3D12, reusing an immutable source upload.

    Mutable arrays are uploaded on every call. Cached read-only arrays must
    remain immutable for their lifetime, as with the existing preview caches.
    Returns (display RGB8, actual backend), or raises for CPU fallback.
    """
    global _camera_profile_gpu_state
    import weakref
    from native_dense import _camera_profile_arguments

    if os.environ.get("IMPRINT_NATIVE_CAMERA_PROFILE") == "0":
        raise NativeRendererError("Camera profile GPU operator is disabled")
    image, width, height, constants, table, (hue, saturation, value), curve = _camera_profile_arguments(
        source, matrix, white, gain, look, dims, encoding, tone, projection)
    library, _ = _get_library()
    upload = getattr(library, "im_renderer_set_camera_profile_source", None)
    render = getattr(library, "im_renderer_render_camera_profile", None)
    if upload is None or render is None:
        raise NativeRendererError("Camera profile GPU ABI is unavailable")
    u16, u8, fp = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint8),
                   ctypes.POINTER(ctypes.c_float))
    upload.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, u16, ctypes.c_size_t]
    upload.restype = ctypes.c_int
    render.argtypes = [ctypes.c_void_p, fp, ctypes.c_size_t, fp, ctypes.c_size_t,
                      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                      fp, ctypes.c_size_t, u8, ctypes.c_size_t]
    render.restype = ctypes.c_int
    with _camera_profile_gpu_lock:
        if _camera_profile_gpu_state is None:
            renderer, backend = _create_renderer(library)
            enabled = os.environ.get("IMPRINT_NATIVE_CAMERA_PROFILE", "1" if backend == "Metal" else "0") == "1"
            if backend not in ("Metal", "D3D12") or not enabled:
                library.im_renderer_destroy(renderer)
                raise NativeRendererError("Camera profile GPU backend is unavailable or disabled")
            _camera_profile_gpu_state = (library, renderer, backend, None)
        cached_library, renderer, backend, uploaded = _camera_profile_gpu_state
        if cached_library is not library:
            raise NativeRendererError("Camera profile renderer library changed; clear the GPU cache first")
        try:
            if image.flags.writeable or uploaded is None or uploaded() is not image:
                status = upload(renderer, width, height, image.ctypes.data_as(u16), image.size)
                if status != 0:
                    raise NativeRendererError(f"Camera profile upload failed: {_native_error(library, renderer)}")
                _camera_profile_gpu_state = (library, renderer, backend, weakref.ref(image))
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
    if not np.isfinite(image).all() or np.any(image < 0.0) or np.any(image > 1.0):
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
    if (not np.isfinite(transmission).all() or np.any(transmission < 0.0) or
            np.any(transmission > 1.0)):
        raise ValueError("Physical dehaze transmission must be finite and within [0, 1]")
    if (not np.isfinite(atmosphere).all() or np.any(atmosphere < 0.0) or
            np.any(atmosphere > 1.0)):
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
        return ((output16.astype(np.uint32) + 128) // 257).astype(np.uint8)
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
        return ((output16.astype(np.uint32) + 128) // 257).astype(np.uint8)
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


def _create_renderer(library: ctypes.CDLL) -> tuple[ctypes.c_void_p, str]:
    renderer = ctypes.c_void_p()
    # IM_BACKEND_AUTO asks the native factory to choose an available backend.
    status = library.im_renderer_create(0, ctypes.byref(renderer))
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
        return ((output16.astype(np.uint32) + 128) // 257).astype(np.uint8)
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
        return ((output16.astype(np.uint32) + 128) // 257).astype(np.uint8)
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
        return ((output16.astype(np.uint32) + 128) // 257).astype(np.uint8)
    return output16


def get_native_status() -> dict[str, object]:
    """Probe actual renderer creation without exposing installation paths."""
    try:
        library, _ = _get_library()
        renderer, backend = _create_renderer(library)
        library.im_renderer_destroy(renderer)
        return {"available": True, "backend": backend}
    except NativeRendererError:
        return {"available": False, "backend": None}
