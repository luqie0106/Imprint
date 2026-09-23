"""Ricoh Camera Raw preset catalog and XMP sidecar export helpers."""

from __future__ import annotations

import os
import sys
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from image_io import OUTPUT_DIR_NAME, SUPPORTED_SUFFIXES


_CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_MAX_PHOTOS = 500
_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class RicohPreset:
    id: str
    model: str
    name: str
    description: str
    resource: str


# Keep this list explicit: only reviewed GR2/GR3 CameraRaw presets are made
# available to the API, regardless of what other files happen to be packaged.
_PRESETS = (
    RicohPreset(
        "gr2_positive_film", "GR2", "GR2 正片 Positive Film",
        "深绿、青色阴影、暖色高光与克制的饱和度。",
        "Ricoh_GR2_CameraRaw_Presets/Presets/GR2_Positive_Film.xmp",
    ),
    RicohPreset(
        "gr2_hi_bw", "GR2", "GR2 高对比黑白 Hi-BW",
        "浓郁黑位、清晰高光、强对比与可见颗粒。",
        "Ricoh_GR2_CameraRaw_Presets/Presets/GR2_Hi_BW.xmp",
    ),
    RicohPreset(
        "gr2_negative_film", "GR2", "GR2 负片 Negative Film",
        "抬高黑位、柔和反差、低饱和色彩与暖色中间调。",
        "Ricoh_GR2_CameraRaw_Presets/Presets/GR2_Negative_Film.xmp",
    ),
    RicohPreset(
        "gr2_street_positive", "GR2", "GR2 街头正片 Street Positive",
        "更强反差、清晰细节、克制色彩与微冷阴影。",
        "Ricoh_GR2_CameraRaw_Presets/Presets/GR2_Street_Positive.xmp",
    ),
    RicohPreset(
        "gr3_positive_film", "GR3", "GR3 正片 Positive Film",
        "深蓝、克制的绿色、青色阴影与暖色高光。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_Positive_Film.xmp",
    ),
    RicohPreset(
        "gr3_negative_film", "GR3", "GR3 负片 Negative Film",
        "抬高黑位、柔和反差、低饱和色彩与柔润高光。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_Negative_Film.xmp",
    ),
    RicohPreset(
        "gr3_vivid_street", "GR3", "GR3 鲜明街头 Vivid Street",
        "强反差、鲜明蓝色、清晰细节与受控肤色。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_Vivid_Street.xmp",
    ),
    RicohPreset(
        "gr3_standard", "GR3", "GR3 标准 Standard",
        "均衡反差、克制饱和度、冷色阴影与微暖高光。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_Standard.xmp",
    ),
    RicohPreset(
        "gr3_high_contrast_bw", "GR3", "GR3 高反差黑白 High Contrast B&W",
        "浓郁黑位、明亮白色与可见颗粒。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_High_Contrast_BW.xmp",
    ),
    RicohPreset(
        "gr3_bleach_bypass", "GR3", "GR3 漂白负冲 Bleach Bypass",
        "高反差、低饱和度与金属灰色调。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_Bleach_Bypass.xmp",
    ),
)
_PRESETS_BY_ID = {preset.id: preset for preset in _PRESETS}

# Preset identity and UI metadata must not be copied into a photo's sidecar.
# Camera Raw processing settings (including curve RDF structures) remain.
_PRESET_IDENTITY_FIELDS = {
    "presettype", "presetid", "presetname", "presetgroup", "presetgroupid",
    "uuid", "id", "cluster", "name", "shortname", "group", "groupname",
    "description", "sortname", "copyright", "contactinfo", "isfavorite",
    "isdefault", "isuserpreset",
}


class RicohBatchLimitError(ValueError):
    """Raised when a request resolves to more photos than the API permits."""


def _resource_root() -> Path:
    """Find bundled XMP files in development and in a PyInstaller package."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    app_root = Path(frozen_root) if frozen_root else Path(__file__).resolve().parents[1]
    return app_root / "assets" / "ricoh"


def list_ricoh_presets() -> list[dict[str, str]]:
    """Return the stable, public catalog without exposing resource paths."""
    return [
        {
            "id": preset.id,
            "model": preset.model,
            "name": preset.name,
            "description": preset.description,
        }
        for preset in _PRESETS
    ]


def _preset_payload(preset_id: str) -> bytes:
    preset = _PRESETS_BY_ID.get(preset_id)
    if preset is None:
        raise KeyError(preset_id)

    source = _resource_root() / preset.resource
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError("preset resource unavailable") from exc

    for element in root.iter():
        for attribute in tuple(element.attrib):
            namespace, separator, local_name = attribute[1:].partition("}") if attribute.startswith("{") else ("", "", attribute)
            if separator and namespace == _CRS_NS:
                normalized = local_name.casefold()
                if (
                    normalized in _PRESET_IDENTITY_FIELDS
                    or normalized.startswith("supports")
                    or normalized.startswith("requires")
                ):
                    del element.attrib[attribute]

    for parent in root.iter():
        for child in tuple(parent):
            if child.tag.startswith("{" + _CRS_NS + "}"):
                local_name = child.tag.rsplit("}", 1)[-1].casefold()
                if (
                    local_name in _PRESET_IDENTITY_FIELDS
                    or local_name.startswith("preset")
                ):
                    parent.remove(child)

    ET.register_namespace("x", "adobe:ns:meta/")
    ET.register_namespace("rdf", _RDF_NS)
    ET.register_namespace("crs", _CRS_NS)
    body = ET.tostring(root, encoding="utf-8")
    return (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + body
        + b'\n<?xpacket end="w"?>'
    )


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "文件或目录不存在"
    if isinstance(exc, PermissionError):
        return "没有读取或写入权限"
    if isinstance(exc, IsADirectoryError):
        return "输入不是图片文件"
    if isinstance(exc, FileExistsError):
        return "已存在同名 XMP，未覆盖"
    if isinstance(exc, ValueError):
        return "输入路径无效"
    if isinstance(exc, RuntimeError):
        return "内置预设资源不可用"
    return "文件访问或写入失败"


def _input_name(raw_path: str) -> str:
    try:
        name = Path(raw_path).name
        return name or "输入路径"
    except (TypeError, ValueError):
        return "输入路径"


def _collect_photos(paths: Iterable[str]) -> tuple[list[Path], list[dict[str, str]]]:
    photos: dict[tuple[str, str], Path] = {}
    results: list[dict[str, str]] = []

    def add_photo(path: Path) -> None:
        # A RAW and its companion JPEG share one sidecar stem. Emit one result
        # and write one XMP for that stem, while keeping separate directories
        # independent.
        absolute = Path(os.path.abspath(path))
        key = (str(absolute.parent), absolute.stem.casefold())
        photos.setdefault(key, absolute)

    for raw_path in paths:
        name = _input_name(raw_path)
        try:
            if not raw_path or not raw_path.strip():
                raise ValueError("empty path")
            path = Path(raw_path).expanduser()
            if path.is_dir():
                if OUTPUT_DIR_NAME in path.parts:
                    continue
                try:
                    for candidate in path.rglob("*"):
                        if (
                            OUTPUT_DIR_NAME not in candidate.parts
                            and candidate.is_file()
                            and candidate.suffix.lower() in SUPPORTED_SUFFIXES
                        ):
                            add_photo(candidate)
                except OSError as exc:
                    results.append({"name": name, "status": "failed", "error": _safe_error(exc)})
                continue
            if not path.exists():
                raise FileNotFoundError
            if not path.is_file():
                raise IsADirectoryError
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                results.append({"name": name, "status": "failed", "error": "不支持的图片格式"})
                continue
            add_photo(path)
        except (OSError, ValueError) as exc:
            results.append({"name": name, "status": "failed", "error": _safe_error(exc)})

    return list(photos.values()), results


def _sidecar_exists_case_insensitive(directory: Path, stem: str) -> bool:
    wanted = (stem + ".xmp").casefold()
    try:
        return any(entry.name.casefold() == wanted for entry in directory.iterdir())
    except OSError:
        raise


def _write_sidecar(photo: Path, payload: bytes) -> None:
    directory = photo.parent
    target = directory / (photo.stem + ".xmp")
    with _WRITE_LOCK:
        if _sidecar_exists_case_insensitive(directory, photo.stem):
            raise FileExistsError
        descriptor: int | None = None
        created = False
        try:
            # O_EXCL prevents replacing an existing file and works on common
            # removable filesystems that do not support hard links.
            descriptor = os.open(
                target,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o666,
            )
            created = True
            with os.fdopen(descriptor, "wb") as output:
                descriptor = None
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if created:
                try:
                    target.unlink()
                except OSError:
                    pass
            raise


def _summarize(files: list[dict[str, str]]) -> dict[str, object]:
    counts = {status: sum(item["status"] == status for item in files) for status in ("written", "skipped", "failed")}
    return {
        "total": len(files),
        "written": counts["written"],
        "skipped": counts["skipped"],
        "failed": counts["failed"],
        "files": files,
    }


def apply_ricoh_preset(paths: list[str], preset_id: str) -> dict[str, object]:
    """Write a preset's Camera Raw settings as safe, non-overwriting sidecars."""
    if preset_id not in _PRESETS_BY_ID:
        raise KeyError(preset_id)

    photos, files = _collect_photos(paths)
    if len(photos) + len(files) > _MAX_PHOTOS:
        raise RicohBatchLimitError(f"最多处理 {_MAX_PHOTOS} 张照片")
    if not photos:
        return _summarize(files)
    try:
        payload = _preset_payload(preset_id)
    except RuntimeError as exc:
        files.extend(
            {"name": photo.name, "status": "failed", "error": _safe_error(exc)}
            for photo in photos
        )
        return _summarize(files)

    for photo in photos:
        sidecar_name = photo.stem + ".xmp"
        try:
            _write_sidecar(photo, payload)
            files.append({"name": sidecar_name, "status": "written"})
        except FileExistsError as exc:
            files.append({"name": sidecar_name, "status": "skipped", "error": _safe_error(exc)})
        except (OSError, RuntimeError, ValueError) as exc:
            files.append({"name": sidecar_name, "status": "failed", "error": _safe_error(exc)})

    return _summarize(files)
