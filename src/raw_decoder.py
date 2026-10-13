"""Small, opt-in bridge for Nikon HE files missing from the installed rawpy.

Normal RAW files continue to use :func:`rawpy.imread` directly. This module
only selects the native Imprint decoder after rawpy reports its specific
unsupported-file exception for a NEF; it never patches rawpy globally.
"""

from __future__ import annotations

import ctypes
from datetime import datetime
import math
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import rawpy


class NativeRawError(RuntimeError):
    """The native Nikon HE fallback could not open or decode a NEF."""


class NativeRawLibraryUnavailableError(NativeRawError):
    """The packaged/developer Imprint native RAW library could not be loaded."""


class _CRawSizes(ctypes.Structure):
    _fields_ = [
        ("raw_height", ctypes.c_uint32),
        ("raw_width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("top_margin", ctypes.c_uint32),
        ("left_margin", ctypes.c_uint32),
        ("iheight", ctypes.c_uint32),
        ("iwidth", ctypes.c_uint32),
        ("pixel_aspect", ctypes.c_double),
        ("flip", ctypes.c_int32),
        ("crop_left_margin", ctypes.c_uint32),
        ("crop_top_margin", ctypes.c_uint32),
        ("crop_width", ctypes.c_uint32),
        ("crop_height", ctypes.c_uint32),
        ("raw_pitch_bytes", ctypes.c_uint32),
    ]


class _CRawInfo(ctypes.Structure):
    _fields_ = [
        ("sizes", _CRawSizes),
        ("compression", ctypes.c_uint32),
        ("filters", ctypes.c_uint32),
        ("num_colors", ctypes.c_uint32),
        ("raw_pattern_size", ctypes.c_uint32),
        ("raw_pattern", ctypes.c_uint8 * 36),
        ("color_desc", ctypes.c_char * 5),
        ("make", ctypes.c_char * 64),
        ("model", ctypes.c_char * 64),
        ("lens_make", ctypes.c_char * 128),
        ("lens_model", ctypes.c_char * 128),
        ("artist", ctypes.c_char * 128),
        ("camera_whitebalance", ctypes.c_float * 4),
        ("daylight_whitebalance", ctypes.c_float * 4),
        ("cam_xyz", ctypes.c_float * 12),
        ("black_level_per_channel", ctypes.c_uint32 * 4),
        ("white_level", ctypes.c_uint32),
        ("has_camera_white_level_per_channel", ctypes.c_int32),
        ("camera_white_level_per_channel", ctypes.c_uint32 * 4),
        ("iso_speed", ctypes.c_double),
        ("shutter_speed", ctypes.c_double),
        ("aperture", ctypes.c_double),
        ("focal_length", ctypes.c_double),
        ("timestamp", ctypes.c_int64),
        ("shot_order", ctypes.c_int32),
        ("lens_min_focal", ctypes.c_double),
        ("lens_max_focal", ctypes.c_double),
    ]


class _CPostprocessParams(ctypes.Structure):
    _fields_ = [
        ("half_size", ctypes.c_int32),
        ("use_camera_wb", ctypes.c_int32),
        ("no_auto_bright", ctypes.c_int32),
        ("output_bps", ctypes.c_int32),
        ("output_color", ctypes.c_int32),
        ("gamma", ctypes.c_double * 2),
        ("has_user_wb", ctypes.c_int32),
        ("user_wb", ctypes.c_float * 4),
        ("has_user_flip", ctypes.c_int32),
        ("user_flip", ctypes.c_int32),
    ]


_C_ERROR_CAPACITY = 1024
_UINT8_POINTER = ctypes.POINTER(ctypes.c_uint8)
_UINT16_POINTER = ctypes.POINTER(ctypes.c_uint16)
_NATIVE_LIBRARY_CACHE: dict[str, Any] = {}
_NATIVE_LIBRARY_CACHE_LOCK = threading.Lock()


def _candidate_libraries() -> list[Path]:
    """Return the supported local and PyInstaller library locations."""
    override = os.environ.get("IMPRINT_NATIVE_RAW_LIB")
    if override:
        return [Path(override).expanduser()]

    roots: list[Path] = [Path(__file__).resolve().parents[1]]
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        roots.insert(0, Path(frozen_root))

    names = (
        Path("native-raw/build/libimprint_raw.dylib"),
        Path("native-raw/build/libimprint_raw.so"),
        Path("native-raw/build/Release/imprint_raw.dll"),
        Path("native-raw/build/Release/libimprint_raw.dll"),
        Path("native-raw/build/imprint_raw.dll"),
        Path("native-raw/build/libimprint_raw.dll"),
        Path("native-raw/lib/libimprint_raw.dylib"),
        Path("native-raw/lib/libimprint_raw.so"),
        Path("native-raw/lib/imprint_raw.dll"),
        Path("native-raw/lib/libimprint_raw.dll"),
        Path("libimprint_raw.dylib"),
        Path("libimprint_raw.so"),
        Path("imprint_raw.dll"),
        Path("libimprint_raw.dll"),
    )
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _bind_library(library: Any) -> Any:
    """Set the stable C ABI declarations before the first call."""
    signatures = {
        "imprint_raw_probe_nikon_he": (
            [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
             ctypes.POINTER(ctypes.c_char), ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "imprint_raw_open": (
            [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_CRawInfo),
             ctypes.POINTER(ctypes.c_char), ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "imprint_raw_close": ([ctypes.c_void_p], None),
        "imprint_raw_copy_visible": (
            [ctypes.c_void_p, _UINT16_POINTER, ctypes.c_size_t,
             ctypes.POINTER(ctypes.c_char), ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "imprint_raw_postprocess": (
            [ctypes.c_void_p, ctypes.POINTER(_CPostprocessParams),
             ctypes.POINTER(_UINT8_POINTER), ctypes.POINTER(ctypes.c_size_t),
             ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
             ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
             ctypes.POINTER(ctypes.c_char), ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "imprint_raw_extract_thumb": (
            [ctypes.c_void_p, ctypes.POINTER(_UINT8_POINTER), ctypes.POINTER(ctypes.c_size_t),
             ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32),
             ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
             ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_char), ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "imprint_raw_decoder_backend": ([ctypes.c_void_p], ctypes.c_char_p),
    }
    try:
        for name, (argtypes, restype) in signatures.items():
            function = getattr(library, name)
            function.argtypes = argtypes
            function.restype = restype
    except (AttributeError, TypeError) as exc:
        raise NativeRawError(f"native RAW library has an incompatible C ABI: {exc}") from exc
    return library


def _load_native_library() -> Any:
    candidates = _candidate_libraries()
    errors: list[str] = []
    with _NATIVE_LIBRARY_CACHE_LOCK:
        for candidate in candidates:
            if not candidate.is_file():
                continue
            cache_key = str(candidate.resolve())
            if cache_key in _NATIVE_LIBRARY_CACHE:
                return _NATIVE_LIBRARY_CACHE[cache_key]
            try:
                library = _bind_library(ctypes.CDLL(str(candidate)))
                _NATIVE_LIBRARY_CACHE[cache_key] = library
                return library
            except (OSError, NativeRawError) as exc:
                errors.append(f"{candidate}: {exc}")
    if os.environ.get("IMPRINT_NATIVE_RAW_LIB"):
        searched = str(candidates[0])
    else:
        searched = ", ".join(str(path) for path in candidates)
    detail = "; ".join(errors) if errors else "no candidate library file exists"
    raise NativeRawLibraryUnavailableError(
        "Nikon HE needs the Imprint native RAW library; "
        f"searched {searched}. Set IMPRINT_NATIVE_RAW_LIB to its full path ({detail})"
    )


def _decode_c_string(value: bytes | bytearray) -> str:
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


def _c_error(buffer: ctypes.Array[ctypes.c_char]) -> str:
    return buffer.value.decode("utf-8", errors="replace") or "native RAW operation failed"


def _require_success(code: int, buffer: ctypes.Array[ctypes.c_char], operation: str) -> None:
    if code != 0:
        raise NativeRawError(f"{operation}: {_c_error(buffer)}")


def _probe_nikon_he(library: Any, path: Path) -> bool | None:
    """Inspect the Nikon maker note and selected loader without unpacking."""
    is_he = ctypes.c_int32()
    error = ctypes.create_string_buffer(_C_ERROR_CAPACITY)
    try:
        code = library.imprint_raw_probe_nikon_he(
            os.fsencode(path), ctypes.byref(is_he), error, ctypes.sizeof(error)
        )
    except Exception:
        return None
    if code != 0:
        return None
    return bool(is_he.value)


class _RawPyDeferredFallback:
    """Keep ordinary NEFs on rawpy while improving deferred unsupported errors."""

    def __init__(self, raw: Any, path: Path, library_error: NativeRawLibraryUnavailableError):
        self._raw = raw
        self._path = path
        self._library_error = library_error

    def __enter__(self) -> "_RawPyDeferredFallback":
        enter = getattr(self._raw, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        exit_method = getattr(self._raw, "__exit__", None)
        if exit_method is not None:
            return exit_method(exc_type, exc, traceback)
        self.close()
        return False

    def close(self) -> None:
        close_method = getattr(self._raw, "close", None)
        if close_method is not None:
            close_method()

    def __getattr__(self, name: str) -> Any:
        try:
            value = getattr(self._raw, name)
        except rawpy.LibRawFileUnsupportedError as original_error:
            raise self._missing_library_error(original_error) from original_error
        if not callable(value):
            return value

        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return value(*args, **kwargs)
            except rawpy.LibRawFileUnsupportedError as original_error:
                raise self._missing_library_error(original_error) from original_error

        return call

    def _missing_library_error(self, original_error: BaseException) -> NativeRawLibraryUnavailableError:
        return NativeRawLibraryUnavailableError(
            f"rawpy deferred an unsupported RAW error while reading {self._path.name}; "
            f"the Nikon HE native fallback is unavailable: {self._library_error}"
        )


class NativeRaw:
    """rawpy-compatible context manager backed by the C++ LibRaw bridge.

    The C++ layer copies the unpacked visible CFA and RGB/thumb results into
    memory owned by this Python object. No NumPy-side entropy decoding or
    image reconstruction occurs here.
    """

    def __init__(self, path: str | os.PathLike[str], *, _library: Any = None) -> None:
        self.path = Path(path)
        if self.path.suffix.lower() != ".nef":
            raise NativeRawError("the native Nikon HE decoder accepts NEF files only")
        self._lock = threading.RLock()
        self._handle = ctypes.c_void_p()
        self._library = _library if _library is not None else _load_native_library()
        info = _CRawInfo()
        error = ctypes.create_string_buffer(_C_ERROR_CAPACITY)
        try:
            code = self._library.imprint_raw_open(
                os.fsencode(self.path), ctypes.byref(self._handle), ctypes.byref(info),
                error, ctypes.sizeof(error),
            )
            _require_success(code, error, "native Nikon HE open failed")
            if not self._handle.value:
                raise NativeRawError("native RAW library returned success without a handle")
            self._info = info
            self._initialize_rawpy_properties()
            backend = self._library.imprint_raw_decoder_backend(self._handle)
            if isinstance(backend, bytes):
                backend = backend.decode("ascii", errors="replace")
            self.decoder_backend = str(backend or "none")
            if self.decoder_backend not in ("c++/metal", "c++/cpu"):
                raise NativeRawError(
                    f"native TicoRAW decoder returned an invalid backend: {self.decoder_backend!r}"
                )
        except Exception:
            self.close()
            raise

    def _initialize_rawpy_properties(self) -> None:
        info = self._info
        source_sizes = info.sizes
        self.sizes = SimpleNamespace(
            raw_height=int(source_sizes.raw_height),
            raw_width=int(source_sizes.raw_width),
            height=int(source_sizes.height),
            width=int(source_sizes.width),
            top_margin=int(source_sizes.top_margin),
            left_margin=int(source_sizes.left_margin),
            iheight=int(source_sizes.iheight),
            iwidth=int(source_sizes.iwidth),
            pixel_aspect=float(source_sizes.pixel_aspect),
            flip=int(source_sizes.flip),
            crop_left_margin=int(source_sizes.crop_left_margin),
            crop_top_margin=int(source_sizes.crop_top_margin),
            crop_width=int(source_sizes.crop_width),
            crop_height=int(source_sizes.crop_height),
            raw_pitch=int(source_sizes.raw_pitch_bytes),
        )
        size = int(info.raw_pattern_size)
        if size < 1 or size > 6:
            raise NativeRawError(f"native RAW library returned invalid CFA pattern size {size}")
        pattern = np.ctypeslib.as_array(info.raw_pattern).copy()[: size * size]
        self._raw_pattern = pattern.reshape(size, size)
        self.make = _decode_c_string(info.make)
        self.model = _decode_c_string(info.model)
        self.color_desc = _decode_c_string(info.color_desc).encode("utf-8")
        self.num_colors = int(info.num_colors)
        self.filters = int(info.filters)
        self.camera_whitebalance = [float(value) for value in info.camera_whitebalance]
        self.daylight_whitebalance = [float(value) for value in info.daylight_whitebalance]
        self.rgb_xyz_matrix = np.asarray(info.cam_xyz, dtype=np.float32).reshape(4, 3).copy()
        self.black_level_per_channel = [int(value) for value in info.black_level_per_channel]
        self.white_level = int(info.white_level)
        self.camera_white_level_per_channel = (
            [int(value) for value in info.camera_white_level_per_channel]
            if info.has_camera_white_level_per_channel else None
        )
        self.lens = SimpleNamespace(
            make=_decode_c_string(info.lens_make),
            model=_decode_c_string(info.lens_model),
            min_focal=float(info.lens_min_focal),
            max_focal=float(info.lens_max_focal),
        )
        try:
            timestamp = datetime.fromtimestamp(int(info.timestamp))
        except (OverflowError, OSError, ValueError):
            timestamp = None
        self.other = SimpleNamespace(
            iso_speed=float(info.iso_speed),
            shutter_speed=float(info.shutter_speed),
            aperture=float(info.aperture),
            focal_length=float(info.focal_length),
            timestamp=timestamp,
            shot_order=int(info.shot_order),
            artist=_decode_c_string(info.artist),
        )

    def __enter__(self) -> "NativeRaw":
        if not self._handle.value:
            raise NativeRawError("native RAW handle is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        with getattr(self, "_lock", threading.RLock()):
            handle = getattr(self, "_handle", None)
            library = getattr(self, "_library", None)
            if handle is not None and handle.value and library is not None:
                library.imprint_raw_close(handle)
                handle.value = None

    def _ensure_open(self) -> None:
        if not self._handle.value:
            raise NativeRawError("native RAW handle is closed")

    @property
    def raw_pattern(self) -> np.ndarray:
        self._ensure_open()
        return self._raw_pattern.copy()

    @property
    def raw_image_visible(self) -> np.ndarray:
        with self._lock:
            self._ensure_open()
            image = np.empty((self.sizes.height, self.sizes.width), dtype=np.uint16)
            error = ctypes.create_string_buffer(_C_ERROR_CAPACITY)
            code = self._library.imprint_raw_copy_visible(
                self._handle, image.ctypes.data_as(_UINT16_POINTER), image.size,
                error, ctypes.sizeof(error),
            )
            _require_success(code, error, "native visible CFA copy failed")
            return image

    def extract_thumb(self) -> Any:
        with self._lock:
            self._ensure_open()
            data_pointer = _UINT8_POINTER()
            data_size = ctypes.c_size_t()
            image_format = ctypes.c_int32()
            width = ctypes.c_uint32()
            height = ctypes.c_uint32()
            channels = ctypes.c_uint32()
            bits = ctypes.c_uint32()
            error = ctypes.create_string_buffer(_C_ERROR_CAPACITY)
            code = self._library.imprint_raw_extract_thumb(
                self._handle, ctypes.byref(data_pointer), ctypes.byref(data_size),
                ctypes.byref(image_format), ctypes.byref(width), ctypes.byref(height),
                ctypes.byref(channels), ctypes.byref(bits), error, ctypes.sizeof(error),
            )
            _require_success(code, error, "native thumbnail extraction failed")
            payload = ctypes.string_at(data_pointer, data_size.value)
            if image_format.value == 1:
                return SimpleNamespace(format=rawpy.ThumbFormat.JPEG, data=payload)
            if image_format.value == 2:
                dtype = np.uint8 if bits.value == 8 else np.uint16 if bits.value == 16 else None
                if dtype is None or channels.value not in (1, 3, 4):
                    raise NativeRawError("native thumbnail has unsupported bitmap dimensions or depth")
                pixels = np.frombuffer(payload, dtype=dtype).copy()
                expected = int(width.value * height.value * channels.value)
                if pixels.size != expected:
                    raise NativeRawError("native bitmap thumbnail size does not match its metadata")
                shape = (int(height.value), int(width.value))
                if channels.value > 1:
                    shape += (int(channels.value),)
                return SimpleNamespace(format=rawpy.ThumbFormat.BITMAP, data=pixels.reshape(shape))
            raise NativeRawError(f"native thumbnail returned unknown format {image_format.value}")

    def postprocess(self, **kwargs: Any) -> np.ndarray:
        allowed = {
            "half_size", "use_camera_wb", "no_auto_bright", "output_bps",
            "output_color", "gamma", "user_wb", "user_flip",
        }
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(f"unsupported native RAW postprocess parameter(s): {', '.join(unknown)}")

        def boolean(name: str, default: bool) -> bool:
            value = kwargs.get(name, default)
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
                return bool(value)
            raise TypeError(f"{name} must be a boolean")

        output_bps = kwargs.get("output_bps", 8)
        if isinstance(output_bps, bool) or not isinstance(output_bps, (int, np.integer)):
            raise TypeError("output_bps must be 8 or 16")
        if int(output_bps) not in (8, 16):
            raise ValueError("output_bps must be 8 or 16")

        output_color = kwargs.get("output_color", rawpy.ColorSpace.sRGB)
        if hasattr(output_color, "value"):
            output_color = output_color.value
        try:
            output_color = int(output_color)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("output_color must be rawpy.ColorSpace.raw or sRGB") from exc
        raw_color = int(rawpy.ColorSpace.raw.value)
        srgb_color = int(rawpy.ColorSpace.sRGB.value)
        if output_color not in (raw_color, srgb_color):
            raise ValueError("native RAW supports only raw and sRGB output colors")

        gamma = kwargs.get("gamma", (2.222, 4.5))
        if gamma is None:
            gamma = (2.222, 4.5)
        try:
            if len(gamma) != 2:
                raise ValueError
            gamma_values = (float(gamma[0]), float(gamma[1]))
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError("gamma must contain two positive finite numbers") from exc
        if not all(math.isfinite(value) and value > 0 for value in gamma_values):
            raise ValueError("gamma must contain two positive finite numbers")

        user_wb = kwargs.get("user_wb")
        if user_wb is None:
            user_wb_values = (0.0, 0.0, 0.0, 0.0)
            has_user_wb = 0
        else:
            try:
                user_wb_values = tuple(float(value) for value in user_wb)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("user_wb must contain four positive finite multipliers") from exc
            if len(user_wb_values) != 4 or not all(
                math.isfinite(value) and value > 0 for value in user_wb_values
            ):
                raise ValueError("user_wb must contain four positive finite multipliers")
            has_user_wb = 1

        user_flip = kwargs.get("user_flip")
        if user_flip is None:
            has_user_flip = 0
            user_flip_value = 0
        else:
            if isinstance(user_flip, bool) or not isinstance(user_flip, (int, np.integer)):
                raise TypeError("user_flip must be an integer from 0 through 7 or None")
            user_flip_value = int(user_flip)
            if user_flip_value < 0 or user_flip_value > 7:
                raise ValueError("user_flip must be an integer from 0 through 7 or None")
            has_user_flip = 1

        params = _CPostprocessParams(
            int(boolean("half_size", False)),
            int(boolean("use_camera_wb", False)),
            int(boolean("no_auto_bright", False)),
            int(output_bps),
            output_color,
            (ctypes.c_double * 2)(*gamma_values),
            has_user_wb,
            (ctypes.c_float * 4)(*user_wb_values),
            has_user_flip,
            user_flip_value,
        )
        with self._lock:
            self._ensure_open()
            data_pointer = _UINT8_POINTER()
            data_size = ctypes.c_size_t()
            width = ctypes.c_uint32()
            height = ctypes.c_uint32()
            channels = ctypes.c_uint32()
            bits = ctypes.c_uint32()
            error = ctypes.create_string_buffer(_C_ERROR_CAPACITY)
            code = self._library.imprint_raw_postprocess(
                self._handle, ctypes.byref(params), ctypes.byref(data_pointer),
                ctypes.byref(data_size), ctypes.byref(width), ctypes.byref(height),
                ctypes.byref(channels), ctypes.byref(bits), error, ctypes.sizeof(error),
            )
            _require_success(code, error, "native RAW postprocess failed")
            if bits.value not in (8, 16) or channels.value != 3:
                raise NativeRawError("native RAW postprocess returned unexpected RGB metadata")
            expected_bytes = int(width.value * height.value * channels.value * (bits.value // 8))
            if data_size.value != expected_bytes:
                raise NativeRawError("native RAW RGB buffer size does not match its metadata")
            raw_bytes = ctypes.string_at(data_pointer, data_size.value)
            dtype = np.uint8 if bits.value == 8 else np.uint16
            result = np.frombuffer(raw_bytes, dtype=dtype).copy()
            return result.reshape((int(height.value), int(width.value), int(channels.value)))


def open_raw(path: str | os.PathLike[str]) -> Any:
    """Open a RAW through rawpy, with a narrow native fallback for Nikon HE."""
    source = Path(path)
    native_library = None
    native_library_error = None
    if source.suffix.lower() == ".nef":
        try:
            native_library = _load_native_library()
        except NativeRawLibraryUnavailableError as exc:
            native_library_error = exc
        if native_library is not None and _probe_nikon_he(native_library, source):
            return NativeRaw(source, _library=native_library)
    try:
        raw = rawpy.imread(str(source))
    except rawpy.LibRawFileUnsupportedError as original_error:
        if source.suffix.lower() != ".nef":
            raise
        if native_library is None:
            detail = str(native_library_error or "native RAW library could not be loaded")
            raise NativeRawLibraryUnavailableError(
                f"rawpy rejected {source.name}; Nikon HE fallback is unavailable: {detail}"
            ) from original_error
        try:
            return NativeRaw(source, _library=native_library)
        except NativeRawError as native_error:
            raise type(native_error)(
                f"rawpy rejected {source.name}; Nikon HE fallback failed: {native_error}"
            ) from original_error
    if (
        source.suffix.lower() == ".nef"
        and native_library is None
        and type(raw) is rawpy.RawPy
    ):
        return _RawPyDeferredFallback(raw, source, native_library_error)
    return raw

