"""Unified preview and full-resolution RGB loading for the enhancement module."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
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


def _safe_exif(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        with Image.open(path) as image:
            for key, value in image.getexif().items():
                name = ExifTags.TAGS.get(key, str(key))
                if isinstance(value, (str, int, float, bytes, tuple)):
                    result[name] = value
    except Exception:
        pass
    return result


def _exif_from_pil(image: Image.Image) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        for key, value in image.getexif().items():
            name = ExifTags.TAGS.get(key, str(key))
            if isinstance(value, (str, int, float, bytes, tuple)):
                result[name] = value
    except Exception:
        pass
    return result


def _read_raw(path: Path, preview: bool) -> tuple[np.ndarray, ImageMetadata]:
    with rawpy.imread(str(path)) as raw:
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
        exif: dict[str, Any] = {}
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                with Image.open(BytesIO(thumb.data)) as thumb_image:
                    exif = _exif_from_pil(thumb_image)
        except Exception:
            pass
        if camera is not None:
            exif["CameraWhiteBalance"] = tuple(camera)
    metadata = ImageMetadata(
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        bit_depth=8 if preview else 16,
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
