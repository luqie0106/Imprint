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


def _metadata_number(value: Any) -> int | float | None:
    """Convert common EXIF numeric representations to finite JSON numbers."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator = _normalise_exif_value(value[0])
        denominator = _normalise_exif_value(value[1])
        try:
            if denominator == 0:
                return None
            number = float(numerator) / float(denominator)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        normalised = _normalise_exif_value(value)
        try:
            if isinstance(normalised, str):
                number = float(normalised.strip())
            elif isinstance(normalised, (int, float)) and not isinstance(normalised, bool):
                number = float(normalised)
            else:
                return None
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(number) or number <= 0:
        return None
    return int(number) if number.is_integer() else number


def read_photo_metadata(path: str | Path) -> dict[str, Any]:
    """Read compact capture metadata without decoding image pixels.

    Standard images are inspected through Pillow's header and EXIF readers.
    RAW files are opened through LibRaw for its dimensions and camera fields;
    this deliberately never calls ``postprocess`` or accesses the pixel array.
    """
    source = Path(path)
    stat = source.stat()
    width: int | None = None
    height: int | None = None

    if source.suffix.lower() in RAW_SUFFIXES:
        exif = _safe_exif(source)
        try:
            with rawpy.imread(str(source)) as raw:
                sizes = getattr(raw, "sizes", None)
                width_value = getattr(sizes, "width", None)
                height_value = getattr(sizes, "height", None)
                if width_value is not None and height_value is not None:
                    width, height = int(width_value), int(height_value)
                _raw_metadata(raw, exif)
        except Exception:
            # Filename and file size are still useful for a RAW file whose
            # metadata cannot be parsed by this LibRaw build.
            pass
    else:
        exif = {}
        try:
            with Image.open(source) as image:
                width, height = (int(image.width), int(image.height))
                exif = _exif_from_pil(image)
        except Exception:
            pass

    try:
        orientation = int(exif.get("Orientation", 1) or 1)
    except (TypeError, ValueError, OverflowError):
        orientation = 1
    if width is not None and height is not None and orientation in (5, 6, 7, 8):
        width, height = height, width

    camera_model = exif.get("Model")
    if not isinstance(camera_model, str) or not camera_model.strip():
        camera_model = exif.get("Make")
    if not isinstance(camera_model, str) or not camera_model.strip():
        camera_model = None
    else:
        camera_model = camera_model.strip()

    return {
        "filename": source.name,
        "size_bytes": int(stat.st_size),
        "width": width,
        "height": height,
        "iso": _metadata_number(
            exif.get("ISO", exif.get("ISOSpeedRatings", exif.get("PhotographicSensitivity")))
        ),
        "aperture": _metadata_number(exif.get("FNumber")),
        "exposure_time": _metadata_number(exif.get("ExposureTime")),
        "focal_length": _metadata_number(exif.get("FocalLength")),
        "camera_model": camera_model,
    }


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
        # Previews only use the decoded pixels, their dimensions, and color
        # space. The source bit depth and camera metadata are needed for the
        # full-resolution export path, where preserving them is important.
        source_bit_depth = 16 if preview else _raw_bit_depth(raw)
        kwargs: dict[str, Any] = {
            "use_camera_wb": True,
            # Both the quick preview and final export use the same linear
            # exposure scale.  Display gamma is applied only when encoding a
            # JPEG for the UI; LibRaw auto-bright would make the two paths
            # disagree even before dehazing.
            "output_bps": 16,
            "no_auto_bright": True,
            "output_color": rawpy.ColorSpace.sRGB,
            "gamma": (1.0, 1.0),
            "half_size": preview,
        }
        rgb = raw.postprocess(**kwargs)
        exif: dict[str, Any] = {}
        if not preview:
            camera = getattr(raw, "camera_whitebalance", None)
            # Nikon and other TIFF-based RAW files commonly keep LensModel and
            # LensSpecification in the RAW container but omit them from the
            # embedded JPEG. Read the container first, then use the thumbnail
            # and LibRaw only to fill fields that are genuinely absent.
            exif = _safe_exif(path)
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
        color_space="Linear sRGB",
        source_kind="raw",
        exif=exif,
    )
    return np.ascontiguousarray(rgb), metadata


class EnhancedDNGColorError(ValueError):
    """Source RAW data cannot be represented as a camera-space enhanced DNG."""


def enhanced_dng_source_data(
    processed_rgb16: np.ndarray,
    reference_rgb16: np.ndarray,
    source_path: str | Path,
    source_exif: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], int]:
    """Prepare CFA and processed RGB in the same camera-native, sensor orientation.

    LibRaw's camera-space rendering and the original linear-sRGB rendering are
    sampled at matching full-resolution pixels to recover its color conversion.
    The processed result is then mapped through the inverse conversion.  This
    data must be paired with the CFA in an Enhanced Image Data DNG, never
    written as a standalone camera-native linear DNG.
    """
    if (processed_rgb16.dtype != np.uint16 or reference_rgb16.dtype != np.uint16
            or processed_rgb16.shape != reference_rgb16.shape
            or processed_rgb16.ndim != 3 or processed_rgb16.shape[2] != 3):
        raise EnhancedDNGColorError("enhanced DNG requires aligned uint16 RGB images")
    with rawpy.imread(str(source_path)) as raw:
        mosaic = np.ascontiguousarray(raw.raw_image_visible.copy())
        pattern = np.asarray(raw.raw_pattern)
        color_desc = bytes(raw.color_desc)
        if pattern.shape != (2, 2) or mosaic.dtype != np.uint16:
            raise EnhancedDNGColorError("unsupported CFA layout")
        try:
            cfa_pattern = np.array(
                [[b"RGB".index(color_desc[index:index + 1]) for index in row] for row in pattern],
                dtype=np.uint8,
            )
        except ValueError as exc:
            raise EnhancedDNGColorError("unsupported CFA colors") from exc
        camera = raw.postprocess(
            half_size=False, output_bps=16, no_auto_bright=True,
            gamma=(1.0, 1.0), output_color=rawpy.ColorSpace.raw,
            user_wb=[1.0, 1.0, 1.0, 1.0],
        )
        matrix = np.asarray(raw.rgb_xyz_matrix[:3, :], dtype=np.float64)
        white_balance = np.asarray(raw.camera_whitebalance[:3], dtype=np.float64)
        black_levels = tuple(int(value) for value in raw.black_level_per_channel[:4])
        white_level = int(raw.white_level)
        sizes = raw.sizes if hasattr(raw, "sizes") else None
        crop_origin = (
            int(getattr(sizes, "crop_left_margin", 0)),
            int(getattr(sizes, "crop_top_margin", 0)),
        )
        crop_size = (
            int(getattr(sizes, "crop_width", mosaic.shape[1])),
            int(getattr(sizes, "crop_height", mosaic.shape[0])),
        )
    if camera.shape != reference_rgb16.shape:
        raise EnhancedDNGColorError("camera-space and processed image dimensions differ")
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-6:
        raise EnhancedDNGColorError("camera color matrix unavailable")
    if not np.isfinite(white_balance).all() or np.any(white_balance <= 0):
        raise EnhancedDNGColorError("camera white balance unavailable")

    source_samples = camera[::16, ::16].reshape(-1, 3).astype(np.float64) / 65535.0
    reference_samples = reference_rgb16[::16, ::16].reshape(-1, 3).astype(np.float64) / 65535.0
    positive = (
        (source_samples.min(axis=1) > 0.01)
        & (reference_samples.min(axis=1) > 0.01)
    )
    camera_to_srgb = None
    # LibRaw's highlight handling is not a single linear transform.  Fit the
    # camera matrix from midtones first, broadening the range only when a dark
    # image does not supply enough samples.  This keeps clipped highlights
    # from falsely rejecting an otherwise sound conversion.
    for upper in (0.30, 0.50, 0.75, 0.95):
        usable = (
            positive
            & (source_samples.max(axis=1) < upper)
            & (reference_samples.max(axis=1) < upper)
        )
        if np.count_nonzero(usable) < 100:
            continue
        fitted, _, rank, _ = np.linalg.lstsq(
            source_samples[usable], reference_samples[usable], rcond=None,
        )
        if rank != 3 or np.linalg.cond(fitted) > 100:
            continue
        fit_error = np.mean(np.abs(source_samples[usable] @ fitted - reference_samples[usable]))
        if fit_error <= 0.003:
            camera_to_srgb = fitted
            break
    if camera_to_srgb is None:
        raise EnhancedDNGColorError("camera color transform does not fit")
    srgb_to_camera = np.linalg.inv(camera_to_srgb).astype(np.float32)
    del camera

    camera_rgb = np.empty_like(processed_rgb16)
    for start in range(0, processed_rgb16.shape[0], 128):
        end = min(start + 128, processed_rgb16.shape[0])
        chunk = processed_rgb16[start:end].astype(np.float32) @ srgb_to_camera
        camera_rgb[start:end] = np.clip(np.rint(chunk), 0, 65535).astype(np.uint16)

    orientation = int(source_exif.get("Orientation") or 1)
    if orientation == 8:
        camera_rgb = np.ascontiguousarray(np.rot90(camera_rgb, -1))
    elif orientation == 6:
        camera_rgb = np.ascontiguousarray(np.rot90(camera_rgb, 1))
    elif orientation == 3:
        camera_rgb = np.ascontiguousarray(np.rot90(camera_rgb, 2))
    elif orientation != 1:
        raise EnhancedDNGColorError("unsupported RAW orientation")
    if camera_rgb.shape[:2] != mosaic.shape:
        raise EnhancedDNGColorError("enhanced and CFA sensor dimensions differ")
    if (crop_origin[0] < 0 or crop_origin[1] < 0
            or crop_size[0] < 1 or crop_size[1] < 1
            or crop_origin[0] + crop_size[0] > mosaic.shape[1]
            or crop_origin[1] + crop_size[1] > mosaic.shape[0]):
        raise EnhancedDNGColorError("invalid RAW crop")

    make = str(source_exif.get("Make") or "").strip()
    model = str(source_exif.get("Model") or "").strip()
    if not model:
        raise EnhancedDNGColorError("camera model unavailable")
    make_prefix = make.split()[0] if make else ""
    unique_model = model if not make_prefix or model.casefold().startswith(make_prefix.casefold()) else f"{make_prefix} {model}"
    # Adobe's Nikon Z III DNG uses this spaced camera identity.  Keep its
    # spelling when the NEF's EXIF model uses Nikon's compact underscore form.
    nikon_z_model = re.fullmatch(r"(?:NIKON\s+)?Z(\d+)_(\d+)", model, flags=re.IGNORECASE)
    if make_prefix.casefold() == "nikon" and nikon_z_model:
        unique_model = f"Nikon Z {nikon_z_model.group(1)} {nikon_z_model.group(2)}"
    neutral = 1.0 / white_balance
    neutral /= neutral[1]
    profile = {
        "DNGColorMatrix1": tuple(float(value) for value in matrix.flat),
        "DNGAsShotNeutral": tuple(float(value) for value in neutral),
        "DNGUniqueCameraModel": unique_model,
        "DNGBlackLevel": black_levels,
        "DNGWhiteLevel": white_level,
        "DNGDefaultCropOrigin": crop_origin,
        "DNGDefaultCropSize": crop_size,
    }
    return camera_rgb, mosaic, cfa_pattern, profile, orientation


def camera_profile_names(source_path: str | Path) -> dict[str, str]:
    """Read source camera identity and profile names without changing the RAW."""
    executable = _find_exiftool()
    if executable is None:
        return {}
    try:
        completed = subprocess.run(
            [executable, "-json", "-CameraProfile", "-PictureControlName", "-Make", "-Model", str(source_path)],
            check=True, capture_output=True, text=True, timeout=8,
        )
        records = json.loads(completed.stdout)
        record = records[0] if isinstance(records, list) and records else {}
        if not isinstance(record, dict):
            return {}
        result: dict[str, str] = {}
        for source, target in (
            ("CameraProfile", "SourceCameraProfileName"),
            ("PictureControlName", "NikonPictureControlName"),
            ("Make", "Make"),
            ("Model", "Model"),
        ):
            value = _metadata_candidate(record.get(source))
            if isinstance(value, str):
                result[target] = value
        return result
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, UnicodeError):
        return {}


def matching_embedded_profile_dng(
    source_path: str | Path,
    source_info: dict[str, Any],
) -> Path | None:
    """Find a neighboring ACR DNG with the identical Nikon Picture Control.

    A matching profile name alone is insufficient: two customized Flexible
    Color recipes can share a name.  Compare the original Nikon maker-note
    PictureControlData bytes, camera identity, profile name and the DNG's
    explicit copying policy before using its embedded profile as a reference.
    """
    executable = _find_exiftool()
    selected = _metadata_candidate(source_info.get("SourceCameraProfileName"))
    make = _metadata_candidate(source_info.get("Make"))
    model = _metadata_candidate(source_info.get("Model"))
    if executable is None or not all(isinstance(x, str) for x in (selected, make, model)):
        return None
    source = Path(source_path)
    try:
        source_control = subprocess.run(
            [executable, "-b", "-Nikon:PictureControlData", str(source)],
            check=True, capture_output=True, timeout=10,
        ).stdout
        if not source_control:
            return None
        source_curve = subprocess.run(
            [executable, "-b", "-Nikon:ContrastCurve", str(source)],
            check=True, capture_output=True, timeout=10,
        ).stdout
        candidates = sorted(
            (path for path in source.parent.iterdir() if path.is_file() and path.suffix.lower() == ".dng"),
            key=lambda path: path.name,
        )
        for candidate in candidates:
            try:
                details = subprocess.run(
                    [executable, "-json", "-n", "-Make", "-Model", "-ProfileName", "-ProfileEmbedPolicy", str(candidate)],
                    check=True, capture_output=True, text=True, timeout=10,
                )
                records = json.loads(details.stdout)
                record = records[0] if isinstance(records, list) and records else {}
                if not isinstance(record, dict):
                    continue
                if (
                    _metadata_candidate(record.get("Make")) != make
                    or _metadata_candidate(record.get("Model")) != model
                    or _metadata_candidate(record.get("ProfileName")) != selected
                    or str(record.get("ProfileEmbedPolicy")) != "0"
                ):
                    continue
                candidate_control = subprocess.run(
                    [executable, "-b", "-Nikon:PictureControlData", str(candidate)],
                    check=True, capture_output=True, timeout=10,
                ).stdout
                candidate_curve = subprocess.run(
                    [executable, "-b", "-Nikon:ContrastCurve", str(candidate)],
                    check=True, capture_output=True, timeout=10,
                ).stdout
                if candidate_control == source_control and candidate_curve == source_curve:
                    return candidate
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError, UnicodeError):
                continue
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, UnicodeError):
        return None
    return None


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
