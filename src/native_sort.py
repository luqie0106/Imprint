"""Validated ctypes bridge to the optional native burst-sort scorer."""

from __future__ import annotations

import ctypes
import operator
from pathlib import Path
import sys
import threading

import numpy as np


class NativeSortError(RuntimeError):
    """The native sort library is missing or rejected a scoring request."""


class _NativeROI(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint32),
        ("y", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
    ]


_load_lock = threading.Lock()
_cached_library: ctypes.CDLL | None = None


def _library_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("imprint_sort.dll", "libimprint_sort.dll")
    if sys.platform == "darwin":
        return ("libimprint_sort.dylib", "imprint_sort.dylib")
    return ("libimprint_sort.so", "imprint_sort.so")


def _library_candidates() -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    roots: list[Path] = [project_root / "native-sort" / "build"]
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        base = Path(bundle_root).resolve()
        roots[0:0] = [
            base,
            base / "native-sort",
            base / "native-sort" / "build",
            base / "_internal",
            base / "_internal" / "native-sort",
        ]

    candidates: list[Path] = []
    for root in roots:
        for name in _library_names():
            candidates.extend((root / name, root / "Release" / name, root / "Debug" / name))
    return list(dict.fromkeys(candidates))


def _configure_library(library: ctypes.CDLL) -> None:
    byte_pointer = ctypes.POINTER(ctypes.c_uint8)
    library.im_sort_region_sharpness.argtypes = [
        byte_pointer,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.im_sort_region_sharpness.restype = ctypes.c_int
    library.im_sort_region_sharpness_batch.argtypes = [
        byte_pointer,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(_NativeROI),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.im_sort_region_sharpness_batch.restype = ctypes.c_int
    library.im_sort_exposure_score.argtypes = [
        byte_pointer,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.im_sort_exposure_score.restype = ctypes.c_int


def _get_library() -> ctypes.CDLL:
    global _cached_library
    with _load_lock:
        if _cached_library is not None:
            return _cached_library
        for candidate in _library_candidates():
            if not candidate.is_file():
                continue
            try:
                library = ctypes.CDLL(str(candidate))
                _configure_library(library)
                _cached_library = library
                return library
            except (OSError, AttributeError):
                continue
        raise NativeSortError("Native sort library was not found or could not be loaded")


def get_native_sort_status() -> dict[str, bool | str | None]:
    """Return only load availability and the CPU backend name."""
    try:
        _get_library()
    except NativeSortError:
        return {"available": False, "backend": None}
    return {"available": True, "backend": "cpu"}


def _prepare_gray(image: object) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("Native sort input must be a 2D grayscale image")
    if array.dtype != np.dtype(np.uint8):
        raise TypeError("Native sort input must use uint8 samples")
    if not array.flags.c_contiguous:
        raise ValueError("Native sort input must be C-contiguous")
    height, width = array.shape
    if height == 0 or width == 0 or width > 0xFFFFFFFF or height > 0xFFFFFFFF:
        raise ValueError("Native sort image dimensions must fit positive uint32 values")
    # The C ABI reads a private buffer, so an unexpected native write cannot
    # mutate the caller's image.
    return np.array(array, dtype=np.uint8, order="C", copy=True)


def _pointer(gray: np.ndarray):
    return gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def _check_status(status: int, operation: str) -> None:
    if status != 0:
        raise NativeSortError(f"Native sort {operation} failed with status {status}")


def _region_coordinate(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"ROI {name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"ROI {name} must be an integer") from exc
    if result < 0 or result > 0xFFFFFFFF:
        raise ValueError(f"ROI {name} must fit uint32")
    return result


def native_region_sharpness(
    image: object, x: object, y: object, width: object, height: object
) -> float:
    """Compute Laplacian variance plus Sobel energy for a grayscale ROI."""
    return native_region_sharpness_many(image, [(x, y, width, height)])[0]


def native_region_sharpness_many(
    image: object, regions: object
) -> list[float]:
    """Score multiple ROIs with one protected grayscale copy and one C ABI call."""
    gray = _prepare_gray(image)
    try:
        raw_regions = list(regions)
    except TypeError as exc:
        raise TypeError("Native sort regions must be an iterable of ROI tuples") from exc
    if not raw_regions:
        return []
    image_height, image_width = gray.shape
    validated: list[_NativeROI] = []
    for region in raw_regions:
        try:
            x, y, width, height = region
        except (TypeError, ValueError) as exc:
            raise ValueError("Each native sort ROI must contain x, y, width, height") from exc
        roi_x = _region_coordinate(x, "x")
        roi_y = _region_coordinate(y, "y")
        roi_width = _region_coordinate(width, "width")
        roi_height = _region_coordinate(height, "height")
        if (
            roi_width == 0
            or roi_height == 0
            or roi_x >= image_width
            or roi_y >= image_height
            or roi_width > image_width - roi_x
            or roi_height > image_height - roi_y
        ):
            raise ValueError("Native sort ROI must be a non-empty rectangle inside the image")
        validated.append(_NativeROI(roi_x, roi_y, roi_width, roi_height))

    library = _get_library()
    rois = (_NativeROI * len(validated))(*validated)
    output = (ctypes.c_double * len(validated))()
    status = library.im_sort_region_sharpness_batch(
        _pointer(gray),
        image_width,
        image_height,
        rois,
        len(validated),
        output,
    )
    _check_status(status, "sharpness")
    values = [float(value) for value in output]
    if not np.all(np.isfinite(values)):
        raise NativeSortError("Native sort sharpness returned a non-finite value")
    return values


def native_exposure_score(image: object) -> float:
    """Compute the current exposure score from uint8 grayscale threshold counts."""
    gray = _prepare_gray(image)
    height, width = gray.shape
    library = _get_library()
    output = ctypes.c_double()
    status = library.im_sort_exposure_score(
        _pointer(gray), width, height, ctypes.byref(output)
    )
    _check_status(status, "exposure")
    if not np.isfinite(output.value):
        raise NativeSortError("Native sort exposure returned a non-finite value")
    return float(output.value)
