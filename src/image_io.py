"""Unified preview and full-resolution RGB loading for the enhancement module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO
from collections import OrderedDict
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import threading
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
_PROFILE_DNG_CACHE_LIMIT = 32
_profile_dng_cache: OrderedDict[tuple[Any, ...], Path | None] = OrderedDict()
_profile_dng_cache_lock = threading.Lock()

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
        # Preview lens correction needs the same camera/lens EXIF as export.
        # Only source bit depth can be skipped on the quick path.
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


@dataclass(frozen=True)
class EnhancedDNGSourceCache:
    """Validated, full-resolution camera data shared by preview and DNG export."""

    source_identity: tuple[Any, ...]
    source_shape: tuple[int, ...]
    camera_rgb16: np.ndarray
    reference_rgb16: np.ndarray
    srgb_to_camera: np.ndarray
    camera_lens_result: Any
    reference_lens_result: Any
    correction_version: Any
    calibration_signature: tuple[Any, ...] = ()


def enhanced_dng_source_identity(path: str | Path) -> tuple[Any, ...]:
    """Return a stable filesystem identity for an enhanced-DNG source RAW."""
    absolute_path = Path(path).expanduser().resolve(strict=True)
    stat = absolute_path.stat()
    return (
        str(absolute_path), int(stat.st_dev), int(stat.st_ino), int(stat.st_size),
        int(stat.st_mtime_ns), int(stat.st_ctime_ns),
    )


def _camera_profile_transfer_numpy(
    camera: np.ndarray,
    reference: np.ndarray,
    processed: np.ndarray,
    srgb_to_camera: np.ndarray,
) -> np.ndarray:
    """NumPy reference for transferring an enhancement onto camera RGB."""
    output = np.empty_like(camera)
    luminance = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    for start in range(0, processed.shape[0], 128):
        end = min(start + 128, processed.shape[0])
        processed_chunk = processed[start:end].astype(np.float32)
        reference_chunk = reference[start:end].astype(np.float32)
        camera_chunk = camera[start:end].astype(np.float32)
        processed_luma = np.maximum(processed_chunk @ luminance, 0.0)
        reference_luma = np.maximum(reference_chunk @ luminance, 0.0)
        ratio = processed_luma / np.maximum(reference_luma, 1.0)
        residual = processed_chunk - reference_chunk * ratio[:, :, None]
        dark = reference_luma < 1.0
        residual[dark] = 0.0
        ratio[dark] = 0.0
        chunk = camera_chunk * ratio[:, :, None] + residual @ srgb_to_camera
        output[start:end] = np.clip(np.rint(chunk), 0, 65535).astype(np.uint16)
    return output


def _fit_srgb_to_camera(
    uncorrected_camera: np.ndarray,
    uncorrected_reference: np.ndarray,
) -> np.ndarray:
    """Fit the export transform from uncorrected camera and reference pixels."""
    source_samples = uncorrected_camera[::16, ::16].reshape(-1, 3).astype(np.float64) / 65535.0
    reference_samples = uncorrected_reference[::16, ::16].reshape(-1, 3).astype(np.float64) / 65535.0
    positive = (
        (source_samples.min(axis=1) > 0.01)
        & (reference_samples.min(axis=1) > 0.01)
    )
    camera_to_srgb = None
    # LibRaw's highlight handling is not a single linear transform. Fit the
    # camera matrix from midtones first, broadening the range only when a dark
    # image does not supply enough samples.
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
    # Some DNG camera renderings have a narrow usable midtone interval even
    # when the established ranges above include enough samples. Try narrower
    # intervals only after every established candidate has failed.
    if camera_to_srgb is None:
        for upper in (0.25, 0.20, 0.15, 0.10):
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
    return np.ascontiguousarray(np.linalg.inv(camera_to_srgb).astype(np.float32))


def _lens_signature(result: Any) -> tuple[Any, ...]:
    return (
        bool(result.applied), str(result.camera_name or ""), str(result.lens_name or ""),
        bool(result.distortion_applied), bool(result.tca_applied),
        bool(result.vignetting_applied),
    )


def _lens_cache_results_equal(camera_result: Any, reference_result: Any) -> bool:
    try:
        return (
            _lens_signature(camera_result) == _lens_signature(reference_result)
            and float(getattr(camera_result, "scale", 1.0))
            == float(getattr(reference_result, "scale", 1.0))
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _lens_result_matches_metadata(
    camera_result: Any,
    reference_result: Any,
    source_exif: dict[str, Any],
    *,
    require_geometry: bool,
) -> bool:
    try:
        camera_signature = _lens_signature(camera_result)
        reference_signature = _lens_signature(reference_result)
        operation_names = {
            name for name, applied in (
                ("distortion", camera_result.distortion_applied),
                ("tca", camera_result.tca_applied),
                ("vignetting", camera_result.vignetting_applied),
            ) if applied
        }
        recorded_operations = {
            item.strip() for item in str(source_exif.get("LensCorrectionOperations") or "").split(",")
            if item.strip()
        }
        geometry_applied = bool(
            camera_result.distortion_applied or camera_result.tca_applied
        )
        return bool(
            source_exif.get("LensCorrectionApplied") is True
            and camera_signature == reference_signature
            and camera_result.applied
            and (not require_geometry or geometry_applied)
            and operation_names == recorded_operations
            and str(source_exif.get("LensCorrectionEngine") or "") == "Lensfun/lensfunpy"
            and str(source_exif.get("LensCorrectionCamera") or "") == str(camera_result.camera_name or "")
            and str(source_exif.get("LensCorrectionLens") or "") == str(camera_result.lens_name or "")
        )
    except (AttributeError, TypeError, ValueError):
        return False


def build_enhanced_dng_source_cache(
    source_path: str | Path,
    uncorrected_camera: np.ndarray,
    uncorrected_reference: np.ndarray,
    corrected_camera: np.ndarray,
    corrected_reference: np.ndarray,
    camera_lens_result: Any,
    reference_lens_result: Any,
    *,
    correction_version: Any,
    camera_gain_applied: bool = False,
    reference_gain_applied: bool = False,
) -> EnhancedDNGSourceCache | None:
    """Build a no-copy cache only for matching, geometry-corrected RGB data."""
    try:
        arrays = (
            uncorrected_camera, uncorrected_reference,
            corrected_camera, corrected_reference,
        )
        if any(not isinstance(array, np.ndarray) for array in arrays):
            return None
        shape = uncorrected_camera.shape
        if (
            len(shape) != 3 or shape[2] != 3
            or any(array.dtype != np.uint16 or array.shape != shape for array in arrays)
            or not corrected_camera.flags.c_contiguous
            or not corrected_reference.flags.c_contiguous
            or camera_gain_applied or reference_gain_applied
            or getattr(camera_lens_result, "engine", "Lensfun/lensfunpy") != "Lensfun/lensfunpy"
            or getattr(reference_lens_result, "engine", "Lensfun/lensfunpy") != "Lensfun/lensfunpy"
            or not _lens_cache_results_equal(camera_lens_result, reference_lens_result)
            or not _lens_result_matches_metadata(
                camera_lens_result, reference_lens_result,
                {
                    "LensCorrectionApplied": True,
                    "LensCorrectionEngine": "Lensfun/lensfunpy",
                    "LensCorrectionCamera": str(camera_lens_result.camera_name or ""),
                    "LensCorrectionLens": str(camera_lens_result.lens_name or ""),
                    "LensCorrectionOperations": ",".join(
                        name for name, applied in (
                            ("distortion", camera_lens_result.distortion_applied),
                            ("tca", camera_lens_result.tca_applied),
                            ("vignetting", camera_lens_result.vignetting_applied),
                        ) if applied
                    ),
                },
                require_geometry=True,
            )
        ):
            return None
        srgb_to_camera = _fit_srgb_to_camera(uncorrected_camera, uncorrected_reference)
        from lens_correction import _calibration_signature

        calibration_signature = tuple(_calibration_signature())
        identity = enhanced_dng_source_identity(source_path)
        # Keep the caller's corrected pixel buffers shared with preview while
        # making the snapshots read-only for the duration of export.
        corrected_camera.flags.writeable = False
        corrected_reference.flags.writeable = False
        srgb_to_camera.flags.writeable = False
        return EnhancedDNGSourceCache(
            source_identity=identity,
            source_shape=tuple(int(size) for size in shape),
            camera_rgb16=corrected_camera,
            reference_rgb16=corrected_reference,
            srgb_to_camera=srgb_to_camera,
            camera_lens_result=camera_lens_result,
            reference_lens_result=reference_lens_result,
            correction_version=correction_version,
            calibration_signature=calibration_signature,
        )
    except Exception:
        # Cache creation is opportunistic. A caller can always export through
        # the established full-resolution reconstruction path.
        return None


def _dng_exiftool_camera_profile(source_path: str | Path) -> tuple[np.ndarray, tuple[float, ...], str] | None:
    """Read a DNG's explicit D65 color profile when LibRaw has no matrix."""
    executable = _find_exiftool()
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable, "-json", "-n", "-ColorMatrix1", "-ColorMatrix2",
                "-CalibrationIlluminant1", "-CalibrationIlluminant2",
                "-AsShotNeutral", "-UniqueCameraModel", str(source_path),
            ],
            check=True, capture_output=True, text=True, timeout=8,
        )
        records = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, UnicodeError):
        return None
    record = records[0] if isinstance(records, list) and records else None
    if not isinstance(record, dict):
        return None

    def numeric_values(value: Any) -> np.ndarray | None:
        if isinstance(value, str):
            values: Any = value.split()
        else:
            values = value
        try:
            return np.asarray(values, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            return None

    neutral = numeric_values(record.get("AsShotNeutral"))
    if neutral is None:
        return None
    model = record.get("UniqueCameraModel")
    if (neutral.shape != (3,) or not np.isfinite(neutral).all()
            or np.any(neutral <= 0) or not isinstance(model, str) or not model.strip()):
        return None
    for index in (2, 1):
        try:
            d65 = float(record.get(f"CalibrationIlluminant{index}")) == 21.0
        except (TypeError, ValueError, OverflowError):
            d65 = False
        if not d65:
            continue
        matrix = numeric_values(record.get(f"ColorMatrix{index}"))
        if matrix is None:
            continue
        if matrix.size != 9 or not np.isfinite(matrix).all():
            continue
        matrix = matrix.reshape(3, 3)
        if abs(float(np.linalg.det(matrix))) < 1e-6:
            continue
        return matrix, tuple(float(value) for value in neutral), model.strip()
    return None


def _enhanced_dng_cache_matches(
    source_cache: EnhancedDNGSourceCache | None,
    processed_rgb16: np.ndarray,
    reference_rgb16: np.ndarray,
    source_exif: dict[str, Any],
    correction_version: Any,
) -> bool:
    if not isinstance(source_cache, EnhancedDNGSourceCache):
        return False
    if correction_version is None or source_cache.correction_version != correction_version:
        return False
    if source_exif.get("DNGGainMapApplied") is True:
        return False
    try:
        from lens_correction import _calibration_signature

        if tuple(_calibration_signature()) != source_cache.calibration_signature:
            return False
    except Exception:
        return False
    shape = tuple(int(size) for size in processed_rgb16.shape)
    if source_cache.source_shape != shape or reference_rgb16.shape != processed_rgb16.shape:
        return False
    for array in (source_cache.camera_rgb16, source_cache.reference_rgb16):
        if (
            not isinstance(array, np.ndarray) or array.dtype != np.uint16
            or array.shape != shape or not array.flags.c_contiguous
            or array.flags.writeable
        ):
            return False
    matrix = source_cache.srgb_to_camera
    if (
        not isinstance(matrix, np.ndarray) or matrix.dtype != np.float32
        or matrix.shape != (3, 3) or not matrix.flags.c_contiguous
        or matrix.flags.writeable or not np.isfinite(matrix).all()
        or abs(float(np.linalg.det(matrix))) < 1e-6
    ):
        return False
    return _lens_result_matches_metadata(
        source_cache.camera_lens_result,
        source_cache.reference_lens_result,
        source_exif,
        require_geometry=True,
    ) and _lens_cache_results_equal(
        source_cache.camera_lens_result, source_cache.reference_lens_result,
    )


def enhanced_dng_source_data(
    processed_rgb16: np.ndarray,
    reference_rgb16: np.ndarray,
    source_path: str | Path,
    source_exif: dict[str, Any],
    *,
    source_cache: EnhancedDNGSourceCache | None = None,
    correction_version: Any = None,
    cache_diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], int]:
    """Prepare CFA and processed RGB in the same camera-native, sensor orientation.

    A fitted camera-to-sRGB transform maps only the processed color residual
    onto the native camera rendering. The aligned linear luminance ratio carries
    tonal changes without replacing camera colors with an inverse of LibRaw's
    possibly clipped or nonlinear sRGB rendering. This data must be paired with
    the CFA in an Enhanced Image Data DNG.
    """
    if (processed_rgb16.dtype != np.uint16 or reference_rgb16.dtype != np.uint16
            or processed_rgb16.shape != reference_rgb16.shape
            or processed_rgb16.ndim != 3 or processed_rgb16.shape[2] != 3):
        raise EnhancedDNGColorError("enhanced DNG requires aligned uint16 RGB images")
    cache_candidate = _enhanced_dng_cache_matches(
        source_cache, processed_rgb16, reference_rgb16, source_exif, correction_version,
    )
    if cache_diagnostics is not None:
        cache_diagnostics["camera"] = "miss"
    before_read_identity = None
    if cache_candidate:
        try:
            before_read_identity = enhanced_dng_source_identity(source_path)
        except (OSError, RuntimeError, ValueError):
            before_read_identity = None
        if (
            before_read_identity is None
            or before_read_identity != source_cache.source_identity
        ):
            cache_candidate = False

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
        matrix = np.asarray(raw.rgb_xyz_matrix[:3, :], dtype=np.float64)
        camera_whitebalance = getattr(raw, "camera_whitebalance", None)
        white_balance = (
            np.asarray(camera_whitebalance[:3], dtype=np.float64)
            if camera_whitebalance is not None else np.asarray([], dtype=np.float64)
        )
        black_levels = tuple(int(value) for value in raw.black_level_per_channel[:4])
        white_level = int(raw.white_level)
        sizes = raw.sizes if hasattr(raw, "sizes") else None
        crop_origin = (
            int(getattr(sizes, "crop_left_margin", 0)),
            int(getattr(sizes, "crop_top_margin", 0)),
        )
        crop_width = int(getattr(sizes, "crop_width", mosaic.shape[1]))
        crop_height = int(getattr(sizes, "crop_height", mosaic.shape[0]))
        if crop_width == 0:
            crop_width = mosaic.shape[1] - crop_origin[0]
        if crop_height == 0:
            crop_height = mosaic.shape[0] - crop_origin[1]
        crop_size = (crop_width, crop_height)
        if cache_candidate:
            orientation = int(source_exif.get("Orientation") or 1)
            expected_camera_shape = (
                (mosaic.shape[1], mosaic.shape[0], 3)
                if orientation in (6, 8) else (mosaic.shape[0], mosaic.shape[1], 3)
            )
            if source_cache.source_shape != expected_camera_shape:
                cache_candidate = False
        if cache_candidate:
            try:
                after_read_identity = enhanced_dng_source_identity(source_path)
            except (OSError, RuntimeError, ValueError):
                after_read_identity = None
            if (
                after_read_identity is None
                or after_read_identity != source_cache.source_identity
                or after_read_identity != before_read_identity
            ):
                cache_candidate = False
        if cache_candidate:
            camera = source_cache.camera_rgb16
            reference_base = source_cache.reference_rgb16
            srgb_to_camera = source_cache.srgb_to_camera
            lens_result = source_cache.camera_lens_result
        else:
            camera = raw.postprocess(
                half_size=False, output_bps=16, no_auto_bright=True,
                gamma=(1.0, 1.0), output_color=rawpy.ColorSpace.raw,
                user_wb=[1.0, 1.0, 1.0, 1.0],
            )
            lens_result = None
    if cache_diagnostics is not None:
        cache_diagnostics["camera"] = "hit" if cache_candidate else "miss"
    if camera.shape != reference_rgb16.shape:
        raise EnhancedDNGColorError("camera-space and processed image dimensions differ")
    if camera.dtype != np.uint16:
        raise EnhancedDNGColorError("camera-space rendering must use uint16 samples")
    dng_profile_override = None
    matrix_valid = (
        matrix.shape == (3, 3) and np.isfinite(matrix).all()
        and abs(float(np.linalg.det(matrix))) >= 1e-6
    )
    if not matrix_valid:
        if Path(source_path).suffix.lower() == ".dng":
            dng_profile_override = _dng_exiftool_camera_profile(source_path)
        if dng_profile_override is None:
            raise EnhancedDNGColorError("camera color matrix unavailable")
        matrix, _, _ = dng_profile_override
    elif (white_balance.shape != (3,) or not np.isfinite(white_balance).all()
          or np.any(white_balance <= 0)):
        raise EnhancedDNGColorError("camera white balance unavailable")

    if not cache_candidate:
        srgb_to_camera = _fit_srgb_to_camera(camera, reference_rgb16)

    # The enhancement reference and processed image already include these
    # geometric operations. Reapply them to both transfer inputs so their
    # pixels still refer to the same scene coordinates as the processed image.
    camera_base = camera
    if cache_candidate:
        camera_base = source_cache.camera_rgb16
        reference_base = source_cache.reference_rgb16
    else:
        reference_base = reference_rgb16
    embedded_applied = False
    if (not cache_candidate and source_exif.get("LensCorrectionApplied") is True
            and source_exif.get("LensCorrectionEngine") == "DNG/WarpRectilinear"):
        try:
            from dng_warp import apply_dng_warp_correction
            from lens_correction import LensCorrectionResult

            diagnostics: dict[str, Any] = {}
            # Both layers use the same working-space highlight shoulder. The
            # helper executes GainMap and Warp in their source opcode order.
            camera_base, camera_warp, camera_gain = apply_dng_warp_correction(
                camera_base, source_path, highlight_reference=reference_rgb16,
                diagnostics=diagnostics,
            )
            reference_base, reference_warp, reference_gain = apply_dng_warp_correction(
                reference_base, source_path, highlight_reference=reference_rgb16,
            )
            operations = {"distortion"}
            if diagnostics.get("tca_applied"):
                operations.add("tca")
            recorded = {
                value.strip() for value in
                str(source_exif.get("LensCorrectionOperations") or "").split(",")
                if value.strip()
            }
            if (not camera_warp or not reference_warp or camera_gain != reference_gain
                    or camera_gain != bool(source_exif.get("DNGGainMapApplied"))
                    or operations != recorded
                    or source_exif.get("LensCorrectionLens") != "DNG WarpRectilinear"
                    or str(source_exif.get("LensCorrectionCamera") or "") !=
                    str(source_exif.get("Model") or "")):
                raise EnhancedDNGColorError("embedded correction metadata does not match the source")
            lens_result = LensCorrectionResult(
                True, str(source_exif.get("Model") or ""), "DNG WarpRectilinear",
                True, bool(diagnostics.get("tca_applied")), False,
                engine="DNG/WarpRectilinear", backend=diagnostics.get("backend"),
            )
            embedded_applied = True
        except Exception as exc:
            raise EnhancedDNGColorError("embedded DNG correction cannot be reproduced for export") from exc
    elif not cache_candidate and source_exif.get("LensCorrectionApplied") is True:
        try:
            from lens_correction import apply_lens_correction

            camera_base, camera_lens_result = apply_lens_correction(
                camera_base, source_exif, require_correction=True,
            )
            reference_base, reference_lens_result = apply_lens_correction(
                reference_base, source_exif, require_correction=True,
            )
        except Exception as exc:
            raise EnhancedDNGColorError("lens correction cannot be reproduced for DNG export") from exc

        if not _lens_result_matches_metadata(
            camera_lens_result, reference_lens_result, source_exif,
            require_geometry=False,
        ):
            raise EnhancedDNGColorError("lens correction metadata does not match the source correction")
        lens_result = camera_lens_result

    # app_api skips the DNG opcode map when Lensfun already baked vignetting.
    # Respect that choice even if stale metadata happens to carry both flags.
    if not embedded_applied and source_exif.get("DNGGainMapApplied") is True and not bool(
        lens_result is not None and lens_result.vignetting_applied
    ):
        try:
            from dng_gainmap import apply_dng_gain_map

            gain_options = {}
            if source_exif.get("DNGGainMapHighlightProtection") is True:
                # The shoulder is defined in working sRGB. Reuse its source
                # brightness for the camera layer rather than making a second
                # exposure decision in a different colour space.
                gain_options = {"preserve_highlights": True,
                                "highlight_reference": reference_base}
            camera_base, camera_gain_applied = apply_dng_gain_map(
                camera_base, source_path, **gain_options,
            )
            reference_base, reference_gain_applied = apply_dng_gain_map(
                reference_base, source_path, **gain_options,
            )
        except Exception as exc:
            raise EnhancedDNGColorError("DNG gain map cannot be reproduced for export") from exc
        if not camera_gain_applied or camera_gain_applied != reference_gain_applied:
            raise EnhancedDNGColorError("DNG gain map metadata does not match the source correction")

    if camera_base.shape != reference_base.shape or processed_rgb16.shape != reference_base.shape:
        raise EnhancedDNGColorError("corrected camera and reference dimensions differ")
    # Preserve the native camera rendering exactly for identity operations,
    # including dark pixels where a luminance ratio is intentionally undefined.
    if np.array_equal(processed_rgb16, reference_base):
        camera_rgb = np.ascontiguousarray(camera_base.copy())
    else:
        from native_dense import camera_profile_transfer
        from native_renderer import NativeRendererError

        try:
            camera_rgb = camera_profile_transfer(
                camera_base, reference_base, processed_rgb16, srgb_to_camera,
            )
        except NativeRendererError:
            camera_rgb = _camera_profile_transfer_numpy(
                camera_base, reference_base, processed_rgb16, srgb_to_camera,
            )

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

    if dng_profile_override is not None:
        _, neutral, unique_model = dng_profile_override
    else:
        make = str(source_exif.get("Make") or "").strip()
        model = str(source_exif.get("Model") or "").strip()
        if not model:
            raise EnhancedDNGColorError("camera model unavailable")
        make_prefix = make.split()[0] if make else ""
        unique_model = model if not make_prefix or model.casefold().startswith(make_prefix.casefold()) else f"{make_prefix} {model}"
        # Adobe's Nikon Z III DNG uses this spaced camera identity. Keep its
        # spelling when the NEF's EXIF model uses Nikon's compact underscore form.
        nikon_z_model = re.fullmatch(r"(?:NIKON\s+)?Z(\d+)_(\d+)", model, flags=re.IGNORECASE)
        if make_prefix.casefold() == "nikon" and nikon_z_model:
            unique_model = f"Nikon Z {nikon_z_model.group(1)} {nikon_z_model.group(2)}"
        neutral_array = 1.0 / white_balance
        neutral_array /= neutral_array[1]
        neutral = tuple(float(value) for value in neutral_array)
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


def _profile_dng_file_identity(path: Path) -> tuple[Any, ...]:
    absolute_path = path.expanduser().resolve(strict=True)
    stat = absolute_path.stat()
    return (
        str(absolute_path), int(stat.st_dev), int(stat.st_ino), int(stat.st_size),
        int(stat.st_mtime_ns), int(stat.st_ctime_ns),
    )


def _profile_dng_cache_snapshot(
    source: Path,
    executable: str,
    selected: str,
    make: str,
    model: str,
) -> tuple[list[Path] | None, tuple[Any, ...] | None]:
    """Capture neighboring DNG identities without holding the cache lock."""
    try:
        candidates = sorted(
            (
                path for path in source.parent.iterdir()
                if path.is_file() and path.suffix.lower() == ".dng"
            ),
            key=lambda path: path.name,
        )
    except OSError:
        return None, None
    try:
        key = (
            str(executable), selected, make, model,
            _profile_dng_file_identity(source),
            tuple(_profile_dng_file_identity(path) for path in candidates),
        )
    except (OSError, RuntimeError):
        return candidates, None
    return candidates, key


_PROFILE_DNG_CACHE_MISS = object()


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
    candidates, cache_key = _profile_dng_cache_snapshot(
        source, executable, selected, make, model,
    )
    if cache_key is not None:
        with _profile_dng_cache_lock:
            cached = _profile_dng_cache.get(cache_key, _PROFILE_DNG_CACHE_MISS)
            if cached is not _PROFILE_DNG_CACHE_MISS:
                _profile_dng_cache.move_to_end(cache_key)
                return cached

    cacheable = True
    match: Path | None = None
    try:
        source_control = subprocess.run(
            [executable, "-b", "-Nikon:PictureControlData", str(source)],
            check=True, capture_output=True, timeout=10,
        ).stdout
        if not source_control:
            result: Path | None = None
        else:
            source_curve = subprocess.run(
                [executable, "-b", "-Nikon:ContrastCurve", str(source)],
                check=True, capture_output=True, timeout=10,
            ).stdout
            if candidates is None:
                candidates = sorted(
                    (
                        path for path in source.parent.iterdir()
                        if path.is_file() and path.suffix.lower() == ".dng"
                    ),
                    key=lambda path: path.name,
                )
            for candidate in candidates:
                try:
                    details = subprocess.run(
                        [executable, "-json", "-n", "-Make", "-Model", "-ProfileName", "-ProfileEmbedPolicy", str(candidate)],
                        check=True, capture_output=True, text=True, timeout=10,
                    )
                    records = json.loads(details.stdout)
                    if (
                        not isinstance(records, list) or not records
                        or not isinstance(records[0], dict)
                    ):
                        cacheable = False
                        continue
                    record = records[0]
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
                        match = candidate
                        break
                except (OSError, subprocess.SubprocessError, json.JSONDecodeError, UnicodeError):
                    cacheable = False
            result = match
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, UnicodeError):
        return None
    if cacheable and cache_key is not None:
        with _profile_dng_cache_lock:
            _profile_dng_cache[cache_key] = result
            _profile_dng_cache.move_to_end(cache_key)
            while len(_profile_dng_cache) > _PROFILE_DNG_CACHE_LIMIT:
                _profile_dng_cache.popitem(last=False)
    return result


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
