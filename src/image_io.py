"""Unified preview and full-resolution RGB loading for the enhancement module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps
import rawpy

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    import pillow_jxl  # noqa: F401
except Exception:
    pass


RAW_SUFFIXES = {
    ".nef", ".nrw", ".arw", ".srf", ".sr2", ".cr2", ".cr3", ".crw",
    ".rw2", ".raw", ".dng", ".raf", ".orf", ".ori", ".pef", ".ptx",
    ".3fr", ".fff", ".iiq", ".srw", ".x3f", ".mrw", ".gpr", ".erf",
    ".mef", ".mos",
}
STANDARD_SUFFIXES = {
    ".jpg", ".jpeg", ".jpe", ".jxl", ".hif", ".heif", ".heic", ".png",
    ".webp", ".tiff", ".tif", ".bmp",
}
SUPPORTED_SUFFIXES = RAW_SUFFIXES | STANDARD_SUFFIXES
OUTPUT_DIR_NAME = "去朦胧输出"

# These are the IFD pointers defined by TIFF/Exif.  Pillow's ``Exif`` object
# exposes the pointer values in IFD0, while the pointed-to IFDs need an
# explicit ``get_ifd`` call.  Keeping this list explicit also means that we do
# not accidentally walk a vendor-specific MakerNote or profile blob.
_EXIF_IFD_POINTERS = {34665, 34853, 40965}  # ExifIFD, GPS IFD, Interop IFD
_BINARY_EXIF_TAGS = {
    "MakerNote", "ICC_Profile", "InterColorProfile", "PrintIM", "Padding",
}


@dataclass
class ImageMetadata:
    width: int
    height: int
    bit_depth: int
    color_space: str = "sRGB"
    source_kind: str = "rgb"
    exif: dict[str, Any] = field(default_factory=dict)


def scan_photo_directory(directory: Path) -> list[Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))
    files: list[Path] = []
    for path in directory.rglob("*"):
        if OUTPUT_DIR_NAME in path.parts:
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(files, key=lambda item: str(item).casefold())


def _normalise_exif_value(value: Any) -> Any:
    """Return a small, JSON-compatible representation of a Pillow EXIF value.

    ``IFDRational`` and similar Pillow scalar classes are intentionally
    converted to ordinary Python numbers.  Binary payloads are not metadata
    useful to the enhancement pipeline (and MakerNotes commonly contain
    offsets into the original file), so only decodable text is retained.
    Returning ``None`` asks the caller to omit a malformed/unrepresentable
    value rather than making an unsafe claim about it.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            text = bytes(value).rstrip(b"\0").decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = bytes(value).rstrip(b"\0").decode("ascii")
            except UnicodeDecodeError:
                return None
        return text
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    # Pillow's IFDRational has numerator/denominator attributes.  Converting
    # it before the sequence case avoids exposing a non-JSON scalar subtype.
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            numerator = int(value.numerator)
            denominator = int(value.denominator)
            if denominator == 0:
                return None
            number = numerator / denominator
            return number if math.isfinite(number) else None
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, (tuple, list)):
        normalised = []
        for item in value:
            item_value = _normalise_exif_value(item)
            if item_value is None:
                return None
            normalised.append(item_value)
        # Tuples are kept for compatibility with the existing CameraWhiteBalance
        # value, while still being accepted by json.dumps as arrays.
        return tuple(normalised) if isinstance(value, tuple) else normalised
    # numpy scalar values are not expected from Pillow, but converting objects
    # with a real integer/float protocol is harmless and keeps this helper safe
    # for test doubles and newer Pillow releases.
    try:
        if hasattr(value, "__int__") and not hasattr(value, "__float__"):
            return int(value)
        if hasattr(value, "__float__"):
            number = float(value)
            return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _exif_tag_name(key: int, *, gps: bool = False) -> str:
    table = ExifTags.GPSTAGS if gps else ExifTags.TAGS
    return table.get(key, str(key))


def _exif_from_pil(image: Image.Image) -> dict[str, Any]:
    """Read IFD0, ExifIFD, GPS IFD and Interop IFD without binary blobs."""
    result: dict[str, Any] = {}
    try:
        exif = image.getexif()
    except Exception:
        return result

    visited: set[int] = set()

    def walk(ifd: Any, *, gps: bool = False) -> None:
        if not hasattr(ifd, "items"):
            return
        # Pillow can return the same IFD object for malformed cyclic pointers.
        marker = id(ifd)
        if marker in visited:
            return
        visited.add(marker)
        try:
            items = tuple(ifd.items())
        except Exception:
            return
        for key, value in items:
            try:
                numeric_key = int(key)
            except (TypeError, ValueError):
                continue
            if not gps and numeric_key in _EXIF_IFD_POINTERS:
                try:
                    child = exif.get_ifd(numeric_key)
                except Exception:
                    continue
                walk(child, gps=numeric_key == 34853)
                continue
            name = _exif_tag_name(numeric_key, gps=gps)
            if name in _BINARY_EXIF_TAGS:
                continue
            normalised = _normalise_exif_value(value)
            if normalised is not None:
                result[name] = normalised

    walk(exif)
    return result


def _safe_exif(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            return _exif_from_pil(image)
    except Exception:
        return {}


def _set_missing(exif: dict[str, Any], key: str, value: Any) -> None:
    if key not in exif and value is not None:
        normalised = _normalise_exif_value(value)
        if normalised is not None:
            exif[key] = normalised


def _find_exiftool() -> str | None:
    """Locate an optional local ExifTool without making it a hard dependency."""
    discovered = shutil.which("exiftool")
    if discovered:
        return discovered
    for candidate in ("/opt/homebrew/bin/exiftool", "/usr/local/bin/exiftool"):
        if Path(candidate).is_file():
            return candidate
    return None


def _lens_metadata_from_exiftool_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map vendor lens tags to portable standard Exif fields."""
    result: dict[str, Any] = {}
    lens_model = next(
        (
            record.get(key)
            for key in ("LensModel", "LensID", "LensSpec", "Lens")
            if _metadata_candidate(record.get(key)) is not None
        ),
        None,
    )
    if lens_model is not None:
        result["LensModel"] = _metadata_candidate(lens_model)
    lens_make = _metadata_candidate(record.get("LensMake"))
    if lens_make is None and "nikon" in str(record.get("Make", "")).casefold():
        lens_make = "Nikon"
    if lens_make is not None:
        result["LensMake"] = lens_make
    lens_serial = _metadata_candidate(record.get("LensSerialNumber"))
    if lens_serial is not None:
        result["LensSerialNumber"] = lens_serial
    lens_specification = record.get("LensSpecification") or record.get("LensInfo")
    if isinstance(lens_specification, (tuple, list)) and len(lens_specification) == 4:
        normalised_specification = tuple(
            _normalise_exif_value(value) for value in lens_specification
        )
        if all(value is not None for value in normalised_specification):
            result["LensSpecification"] = normalised_specification
    if "LensSpecification" not in result:
        parsed_specification = _parse_lens_specification(
            lens_specification or record.get("Lens") or result.get("LensModel")
        )
        if parsed_specification is not None:
            result["LensSpecification"] = parsed_specification
    return result


def _metadata_candidate(value: Any) -> Any:
    normalised = _normalise_exif_value(value)
    if normalised is None or (isinstance(normalised, str) and not normalised.strip()):
        return None
    return normalised


def _parse_lens_specification(value: Any) -> tuple[float, float, float, float] | None:
    """Parse common lens names such as ``14-24mm f/2.8`` into EXIF values."""
    text = str(value or "")
    match = re.search(
        r"(?P<min>\d+(?:\.\d+)?)\s*(?:-\s*(?P<max>\d+(?:\.\d+)?))?\s*mm"
        r"(?:\s+f\s*/\s*(?P<ap_min>\d+(?:\.\d+)?)"
        r"(?:\s*-\s*(?P<ap_max>\d+(?:\.\d+)?))?)?",
        text,
        re.IGNORECASE,
    )
    if match is None or match.group("ap_min") is None:
        return None
    min_focal = float(match.group("min"))
    max_focal = float(match.group("max") or min_focal)
    min_aperture = float(match.group("ap_min"))
    max_aperture = float(match.group("ap_max") or min_aperture)
    if min_focal <= 0 or max_focal < min_focal or min_aperture <= 0 or max_aperture <= 0:
        return None
    return min_focal, max_focal, min_aperture, max_aperture


def _fill_lens_specification(exif: dict[str, Any]) -> None:
    if "LensSpecification" in exif:
        return
    specification = _parse_lens_specification(exif.get("LensModel"))
    if specification is not None:
        exif["LensSpecification"] = specification


def _exiftool_lens_metadata(path: Path) -> dict[str, Any]:
    """Read Nikon/Sony/etc. private lens identity using optional ExifTool."""
    executable = _find_exiftool()
    if executable is None:
        return {}
    try:
        completed = subprocess.run(
            [
                executable,
                "-json",
                "-Make",
                "-LensMake",
                "-LensModel",
                "-LensID",
                "-LensSpec",
                "-LensInfo",
                "-LensSpecification",
                "-Lens",
                "-LensSerialNumber",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
        records = json.loads(completed.stdout)
        if not isinstance(records, list) or not records or not isinstance(records[0], dict):
            return {}
        return _lens_metadata_from_exiftool_record(records[0])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, UnicodeError):
        return {}


def _raw_metadata(raw: Any, exif: dict[str, Any]) -> None:
    """Fill standard fields absent from the embedded RAW thumbnail EXIF.

    rawpy 0.27 exposes LibRaw's ``other`` and ``lens`` named tuples.  Access is
    deliberately defensive because individual RAW formats may not provide a
    field, and because this function must not make decoding fail for metadata
    alone.
    """
    try:
        other = raw.other
    except Exception:
        other = None
    try:
        lens = raw.lens
    except Exception:
        lens = None

    if other is not None:
        _set_missing(exif, "ISO", getattr(other, "iso_speed", None))
        _set_missing(exif, "ExposureTime", getattr(other, "shutter_speed", None))
        _set_missing(exif, "FNumber", getattr(other, "aperture", None))
        _set_missing(exif, "FocalLength", getattr(other, "focal_length", None))
        _set_missing(exif, "Artist", getattr(other, "artist", None))
        timestamp = getattr(other, "timestamp", None)
        if isinstance(timestamp, (datetime, date)):
            captured = timestamp.strftime("%Y:%m:%d %H:%M:%S")
            _set_missing(exif, "DateTimeOriginal", captured)
            _set_missing(exif, "DateTimeDigitized", captured)
            _set_missing(exif, "DateTime", captured)

    if lens is not None:
        _set_missing(exif, "LensMake", getattr(lens, "make", None))
        _set_missing(exif, "LensModel", getattr(lens, "model", None))
        if "LensSpecification" not in exif:
            min_focal = getattr(lens, "min_focal", None)
            max_focal = getattr(lens, "max_focal", None)
            min_aperture = getattr(lens, "max_aperture_at_min_focal", None)
            max_aperture = getattr(lens, "max_aperture_at_max_focal", None)
            lens_specification = (min_focal, max_focal, min_aperture, max_aperture)
            if all(_normalise_exif_value(value) is not None for value in lens_specification):
                exif["LensSpecification"] = tuple(
                    _normalise_exif_value(value) for value in lens_specification
                )

    # Depending on the LibRaw build, camera identity may be exposed as
    # optional attributes.  Do not rely on private/vendor fields, but use these
    # public names if a future rawpy release provides them.
    _set_missing(exif, "Make", getattr(raw, "make", None))
    _set_missing(exif, "Model", getattr(raw, "model", None))

    for attr, key in (
        ("daylight_whitebalance", "DaylightWhiteBalance"),
        ("auto_whitebalance", "AutoWhiteBalance"),
    ):
        try:
            values = getattr(raw, attr)
        except Exception:
            values = None
        _set_missing(exif, key, values)


def _raw_bit_depth(raw: Any) -> int:
    """Infer the source RAW sample depth from LibRaw's sensor white level.

    The decoded RGB buffer is still 16-bit so enhancement math keeps its
    precision.  This value describes the source samples and is used only when
    the processed pixels are quantized for the final Linear DNG.
    """
    candidates: list[int] = []
    try:
        candidates.append(int(raw.white_level))
    except (AttributeError, TypeError, ValueError, OverflowError):
        pass
    try:
        candidates.extend(
            int(value)
            for value in raw.camera_white_level_per_channel
            if value is not None
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        pass
    white_level = max((value for value in candidates if value > 0), default=65535)
    detected = max(1, white_level.bit_length())
    # Common still-photo sample widths.  Rounding upward avoids clipping when
    # a camera's calibrated white level is slightly below its integer maximum.
    for supported in (8, 10, 12, 14, 16):
        if detected <= supported:
            return supported
    return 16


def _read_raw(path: Path, preview: bool) -> tuple[np.ndarray, ImageMetadata]:
    with rawpy.imread(str(path)) as raw:
        source_bit_depth = _raw_bit_depth(raw)
        kwargs: dict[str, Any] = {
            "use_camera_wb": True,
            "output_bps": 8 if preview else 16,
            "no_auto_bright": not preview,
            "output_color": rawpy.ColorSpace.sRGB,
            "gamma": (2.222, 4.5) if preview else (1.0, 1.0),
            "half_size": preview,
        }
        rgb = raw.postprocess(**kwargs)
        camera = getattr(raw, "camera_whitebalance", None)
        # Nikon and other TIFF-based RAW files commonly keep LensModel and
        # LensSpecification in the RAW container but omit them from the
        # embedded JPEG.  Read the container first, then use the thumbnail and
        # LibRaw only to fill fields that are genuinely absent.
        exif: dict[str, Any] = _safe_exif(path)
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                with Image.open(BytesIO(thumb.data)) as thumb_image:
                    for key, value in _exif_from_pil(thumb_image).items():
                        _set_missing(exif, key, value)
        except Exception:
            pass
        _raw_metadata(raw, exif)
        if (
            _metadata_candidate(exif.get("LensModel")) is None
            or "LensSpecification" not in exif
        ):
            for key, value in _exiftool_lens_metadata(path).items():
                _set_missing(exif, key, value)
        _fill_lens_specification(exif)
        if camera is not None:
            exif["CameraWhiteBalance"] = tuple(camera)
    metadata = ImageMetadata(
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        bit_depth=source_bit_depth,
        color_space="sRGB" if preview else "Linear sRGB",
        source_kind="raw",
        exif=exif,
    )
    return np.ascontiguousarray(rgb), metadata


def _read_standard(path: Path) -> tuple[np.ndarray, ImageMetadata]:
    # OpenCV preserves 16-bit PNG/TIFF samples. Pillow is retained as a fallback for
    # HEIC/JXL and for applying EXIF orientation consistently.
    exif = _safe_exif(path)
    payload = np.fromfile(str(path), dtype=np.uint8)
    decoded = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED) if payload.size else None
    if decoded is not None and decoded.ndim in (2, 3):
        if decoded.ndim == 2:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
        elif decoded.shape[2] == 4:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
        else:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        if decoded.dtype not in (np.uint8, np.uint16):
            decoded = np.clip(decoded, 0, 255).astype(np.uint8)
        orientation = int(exif.get("Orientation", 1) or 1)
        if orientation == 2:
            decoded = cv2.flip(decoded, 1)
        elif orientation == 3:
            decoded = cv2.rotate(decoded, cv2.ROTATE_180)
        elif orientation == 4:
            decoded = cv2.flip(decoded, 0)
        elif orientation == 5:
            decoded = cv2.transpose(decoded)
        elif orientation == 6:
            decoded = cv2.rotate(decoded, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 7:
            decoded = cv2.flip(cv2.transpose(decoded), -1)
        elif orientation == 8:
            decoded = cv2.rotate(decoded, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb = decoded
    else:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8).copy()
    return np.ascontiguousarray(rgb), ImageMetadata(
        width=int(rgb.shape[1]), height=int(rgb.shape[0]),
        bit_depth=16 if rgb.dtype == np.uint16 else 8,
        color_space="sRGB", source_kind="rgb", exif=exif,
    )


def read_image(path: str | Path, *, preview: bool = False, max_edge: int = 2048) -> tuple[np.ndarray, ImageMetadata]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持或不存在的照片: {source.name}")
    image, metadata = _read_raw(source, preview) if source.suffix.lower() in RAW_SUFFIXES else _read_standard(source)
    if preview and max(image.shape[:2]) > max_edge:
        scale = max_edge / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    metadata.width, metadata.height = int(image.shape[1]), int(image.shape[0])
    return image, metadata


def to_uint16(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint16:
        return image.copy()
    if image.dtype == np.uint8:
        return (image.astype(np.uint16) * 257).astype(np.uint16)
    raise TypeError("Only uint8 and uint16 RGB images are supported")
