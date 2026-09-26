"""Safe ctypes bridge to Imprint's optional native renderer.

The bridge deliberately owns only one processing stage per call. Python remains
responsible for decoding, color management, lens correction, XMP, and export.
"""

from __future__ import annotations

import ctypes
from collections import OrderedDict
import math
import os
from pathlib import Path
import sys
import threading
from typing import Mapping

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


class NativeRendererError(RuntimeError):
    """The native renderer is missing, unavailable, or rejected a render."""


class _DehazeParams(ctypes.Structure):
    _fields_ = [(field, ctypes.c_float) for field in _DEHAZE_FIELDS]


class _BasicParams(ctypes.Structure):
    _fields_ = [(field, ctypes.c_float) for field in _BASIC_FIELDS]


_load_lock = threading.Lock()
_cached_library: ctypes.CDLL | None = None
_cached_library_path: str | None = None


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


def native_basic(image: object, params: object) -> np.ndarray:
    """Apply only native basic adjustments to RGB uint8/uint16 input.

    The current native basic implementation can differ from the Python preview
    implementation; this bridge does not claim pixel-for-pixel equivalence.
    """
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
