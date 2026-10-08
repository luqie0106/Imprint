"""Optional C++ CPU kernels; loading them never creates a GPU renderer.

IMPRINT_NATIVE_DENSE=0 keeps the NumPy reference available for comparisons.
Older installed libraries may omit these symbols and callers then use Python.
"""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
import os
import threading

import numpy as np
import cv2
from native_renderer import (
    NativeRendererError, _get_library, _DehazeParams, _BasicParams, _values,
    _DEHAZE_FIELDS, _BASIC_FIELDS, _BASIC_LIMITS,
)

_F = ct.POINTER(ct.c_float)
_D = ct.POINTER(ct.c_double)
_IMAGE = [ct.c_uint32, ct.c_uint32, _F, ct.c_size_t]
_lock = threading.Lock()


def _function(name: str, arguments: list):
    if os.environ.get("IMPRINT_NATIVE_DENSE", "1") == "0":
        raise NativeRendererError("C++ dense kernels disabled")
    library, _ = _get_library()
    fn = getattr(library, name, None)
    if fn is None:
        raise NativeRendererError(f"C++ kernel unavailable: {name}")
    with _lock:
        fn.argtypes = arguments
        fn.restype = ct.c_int
    return fn


def _source(image):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("source must have shape HxWx3")
    if image.dtype != np.float32:
        raise TypeError("source must be float32")
    h, w = image.shape[:2]
    if not h or not w or h > 65535 or w > 65535 or image.size > (1 << 29):
        raise ValueError("invalid source dimensions")
    return np.ascontiguousarray(image), w, h


def _pointer(array):
    return array.ctypes.data_as(_D if array.dtype == np.float64 else _F)


def _check(status, name):
    if status:
        raise NativeRendererError(f"C++ {name} rejected input or failed (status {status})")


def physical_pixels(source, transmission, atmosphere, params):
    image, w, h = _source(source)
    t = np.ascontiguousarray(transmission, dtype=np.float32)
    air = np.ascontiguousarray(atmosphere, dtype=np.float32)
    if t.shape != (h, w) or air.shape != (3,):
        raise ValueError("transmission/airlight shape mismatch")
    values = _values(params, _DEHAZE_FIELDS, ((0., 1.),) * len(_DEHAZE_FIELDS), "dehaze")
    fn = _function("im_native_physical_float_run", _IMAGE + [_F, ct.c_size_t, _F,
                   ct.POINTER(_DehazeParams), _F, ct.c_size_t])
    out = np.empty_like(image)
    _check(fn(w, h, _pointer(image), image.size, _pointer(t), t.size, _pointer(air),
              ct.byref(_DehazeParams(*values)), _pointer(out), out.size), "physical pixels")
    return out


def dark_guard(source, result, floor):
    image, w, h = _source(source)
    rendered = np.ascontiguousarray(result, dtype=np.float32)
    if rendered.shape != image.shape:
        raise ValueError("result shape mismatch")
    fn = _function("im_native_dark_guard_run", _IMAGE + [_F, ct.c_size_t, ct.c_float, _F, ct.c_size_t])
    out = np.empty_like(image)
    _check(fn(w, h, _pointer(image), image.size, _pointer(rendered), rendered.size,
              floor, _pointer(out), out.size), "dark guard")
    return out


def scalar_lut(source, lut):
    image, w, h = _source(source)
    table = np.ascontiguousarray(lut, dtype=np.float32)
    if table.ndim != 3 or len(set(table.shape)) != 1 or not 2 <= table.shape[0] <= 129:
        raise ValueError("LUT must be cubic with edge in [2, 129]")
    fn = _function("im_native_scalar_lut_run", _IMAGE + [_F, ct.c_size_t, ct.c_uint32, _F, ct.c_size_t])
    out = np.empty((h, w), dtype=np.float32)
    _check(fn(w, h, _pointer(image), image.size, _pointer(table), table.size,
              table.shape[0], _pointer(out), out.size), "scalar LUT")
    return out


def lut_splat(source, depth, confidence, edge, depth_min, depth_max):
    image, w, h = _source(source)
    payload = np.ascontiguousarray(depth, dtype=np.float32)
    weights = np.ascontiguousarray(confidence, dtype=np.float32)
    if payload.shape != (h, w) or weights.shape != (h, w):
        raise ValueError("depth/confidence shape mismatch")
    if not 2 <= edge <= 129:
        raise ValueError("invalid LUT edge")
    fn = _function("im_native_lut_splat_run", _IMAGE + [_F, ct.c_size_t, _F, ct.c_size_t,
                   ct.c_uint32, ct.c_float, ct.c_float, _D, _D, ct.c_size_t])
    total = np.empty((edge,) * 3, dtype=np.float64)
    mass = np.empty_like(total)
    _check(fn(w, h, _pointer(image), image.size, _pointer(payload), payload.size,
              _pointer(weights), weights.size, edge, depth_min, depth_max,
              _pointer(total), _pointer(mass), total.size), "LUT splat")
    return total, mass


@lru_cache(maxsize=4)
def _lens_interpolation_mode(optimized: bool, ipp_enabled: bool) -> int:
    """Match OpenCV's installed float-map path (IPP may skip 1/32 quantization)."""
    y, x = np.mgrid[:32, :32].astype(np.float32)
    source = (x * 1000 + y * 700).astype(np.uint16)
    maps = np.stack((x[:8] + np.float32(.113), y[:8] + np.float32(.219)), axis=-1)
    actual = cv2.remap(source, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT101)
    fixed_maps = cv2.convertMaps(maps, None, cv2.CV_16SC2)
    quantized = cv2.remap(source, *fixed_maps, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)
    # Exclude the reflected edge when comparing against the planar gradient.
    continuous = np.rint(maps[..., 0] * 1000 + maps[..., 1] * 700).astype(np.uint16)
    interior = (slice(None), slice(None, -1))
    if np.max(np.abs(actual[interior].astype(np.int32) - quantized[interior].astype(np.int32))) <= 1:
        return 0
    if np.max(np.abs(actual[interior].astype(np.int32) - continuous[interior].astype(np.int32))) <= 1:
        return 1
    raise NativeRendererError("Unsupported OpenCV lens interpolation semantics")


def lens_remap(source, maps):
    if not isinstance(source, np.ndarray) or source.dtype != np.uint16 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("lens source must be RGB uint16")
    image = np.ascontiguousarray(source)
    h, w = image.shape[:2]
    coordinates = np.ascontiguousarray(maps, dtype=np.float32)
    if coordinates.ndim == 3 and coordinates.shape[1:] == (w, 2):
        channels = 1
    elif coordinates.ndim == 4 and coordinates.shape[1:] == (w, 3, 2):
        channels = 3
    else:
        raise ValueError("lens map must have shape strip_height x width x [3 x] 2")
    if not 0 < coordinates.shape[0] <= h or not 0 < w < 32767 or not 0 < h < 32767 or image.size > (1 << 29):
        raise ValueError("invalid lens dimensions")
    u16 = ct.POINTER(ct.c_uint16)
    fn = _function("im_native_lens_remap_rgb16", [ct.c_uint32, ct.c_uint32, u16, ct.c_size_t,
                   _F, ct.c_size_t, ct.c_uint32, ct.c_uint32, u16, ct.c_size_t, ct.c_uint32])
    out = np.empty((coordinates.shape[0], w, 3), dtype=np.uint16)
    _check(fn(w, h, image.ctypes.data_as(u16), image.size, _pointer(coordinates), coordinates.size,
              coordinates.shape[0], channels, out.ctypes.data_as(u16), out.size,
              _lens_interpolation_mode(cv2.useOptimized(),
                  bool(getattr(getattr(cv2, "ipp", None), "useIPP", lambda: False)()))), "lens remap")
    return out


def basic_pixels(source, params):
    """Apply the eight existing basic controls without full-size float temporaries."""
    if not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("basic source must be RGB")
    if source.dtype not in (np.uint8, np.uint16):
        raise TypeError("basic source must use uint8 or uint16")
    image = np.ascontiguousarray(source)
    h, w = image.shape[:2]
    if not h or not w or h > 65535 or w > 65535 or image.size > (1 << 29):
        raise ValueError("invalid basic dimensions")
    values = _values(params, _BASIC_FIELDS, _BASIC_LIMITS, "basic")
    sample = ct.c_uint8 if image.dtype == np.uint8 else ct.c_uint16
    pointer = ct.POINTER(sample)
    suffix = "rgb8" if image.dtype == np.uint8 else "rgb16"
    fn = _function("im_native_basic_" + suffix, [ct.c_uint32, ct.c_uint32, pointer,
                   ct.c_size_t, ct.POINTER(_BasicParams), pointer, ct.c_size_t])
    output = np.empty_like(image)
    _check(fn(w, h, image.ctypes.data_as(pointer), image.size, ct.byref(_BasicParams(*values)),
              output.ctypes.data_as(pointer), output.size), "basic pixels")
    return output
