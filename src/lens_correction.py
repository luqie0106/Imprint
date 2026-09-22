"""Lensfun based correction of linear RGB images.

The public entry point in this module deliberately works on already decoded
pixels.  It does not move or rewrite the source photograph.  Lensfun supplies
the calibration and coordinate maps, while OpenCV performs the interpolation
needed to bake the correction into the output pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


STRIP_HEIGHT = 128
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


class LensCorrectionError(RuntimeError):
    """Base class for failures that must be visible to the caller."""


class LensfunUnavailableError(LensCorrectionError):
    """Raised when lensfunpy is not installed or cannot be initialized."""


class LensMatchError(LensCorrectionError):
    """Raised when camera/lens metadata cannot be matched safely."""


class LensCorrectionNotAppliedError(LensCorrectionError):
    """Raised when a match exists but no geometric correction was applied."""


@dataclass(frozen=True)
class LensCorrectionResult:
    """What was actually applied to a corrected image."""

    applied: bool
    camera_name: str | None
    lens_name: str | None
    distortion_applied: bool
    tca_applied: bool
    vignetting_applied: bool
    scale: float = 1.0

    # These aliases make the result convenient for API code without requiring
    # callers to know the exact spelling used by the dataclass fields.
    @property
    def camera_match(self) -> str | None:
        return self.camera_name

    @property
    def lens_match(self) -> str | None:
        return self.lens_name

    @property
    def geometry_applied(self) -> bool:
        return self.distortion_applied or self.tca_applied


def _result_unapplied() -> LensCorrectionResult:
    return LensCorrectionResult(
        applied=False,
        camera_name=None,
        lens_name=None,
        distortion_applied=False,
        tca_applied=False,
        vignetting_applied=False,
    )


def _metadata_value(metadata: Any, key: str) -> Any:
    """Read a field from a mapping, ImageMetadata, or a metadata-like object."""

    if metadata is None:
        return None
    candidates = [key]
    # A few EXIF readers use these alternate names.
    candidates.extend(
        {
            "Make": ("CameraMake", "camera_make"),
            "Model": ("CameraModel", "camera_model"),
            "LensMake": ("lens_make",),
            "LensModel": ("LensName", "lens_model"),
            "FocalLength": ("focal_length",),
            "FNumber": ("Aperture", "aperture", "f_number"),
            "SubjectDistance": ("subject_distance", "FocusDistance"),
        }.get(key, ())
    )

    containers: list[Any] = [metadata]
    for nested_name in ("exif", "metadata", "tags"):
        try:
            nested = getattr(metadata, nested_name)
        except Exception:
            nested = None
        if nested is not None and nested is not metadata:
            containers.append(nested)

    for container in containers:
        if isinstance(container, Mapping):
            # Prefer exact spelling, then a case-insensitive lookup.
            for candidate in candidates:
                if candidate in container and container[candidate] is not None:
                    return container[candidate]
            lower = {str(k).lower(): v for k, v in container.items()}
            for candidate in candidates:
                value = lower.get(candidate.lower())
                if value is not None:
                    return value
        else:
            for candidate in candidates:
                try:
                    value = getattr(container, candidate)
                except Exception:
                    continue
                if value is not None:
                    return value
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip().strip("\x00")
    return str(value).strip().strip("\x00")


def _number(value: Any) -> float | None:
    """Convert EXIF numbers and common textual forms to a finite float."""

    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            numerator = float(value[0])
            denominator = float(value[1])
            if denominator != 0:
                value = numerator / denominator
        elif hasattr(value, "numerator") and hasattr(value, "denominator"):
            value = float(value.numerator) / float(value.denominator)
        elif isinstance(value, str):
            text = value.strip()
            if "/" in text:
                parts = text.split("/", 1)
                try:
                    value = float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    match = _NUMBER_RE.search(text)
                    value = float(match.group(0)) if match else None
            else:
                match = _NUMBER_RE.search(text)
                value = float(match.group(0)) if match else None
        value = float(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def _normal_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _text(value).lower())


def _name_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _text(value).lower()))


def _similarity(expected: Any, actual: Any) -> float:
    expected_text = _normal_name(expected)
    actual_text = _normal_name(actual)
    if not expected_text or not actual_text:
        return 0.0
    if expected_text == actual_text:
        return 1.0
    if expected_text in actual_text or actual_text in expected_text:
        return 0.86
    expected_tokens = _name_tokens(expected)
    actual_tokens = _name_tokens(actual)
    if not expected_tokens or not actual_tokens:
        return 0.0
    overlap = len(expected_tokens & actual_tokens)
    return overlap / max(len(expected_tokens), len(actual_tokens))


def _object_text(obj: Any, *names: str) -> str:
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if value is not None and _text(value):
            return _text(value)
    return ""


def _camera_score(camera: Any, make: str, model: str) -> float:
    candidate_make = _object_text(camera, "maker", "make", "camera_make")
    candidate_model = _object_text(camera, "model", "name", "camera_model")
    model_score = _similarity(model, candidate_model)
    if not model_score:
        return 0.0
    make_score = _similarity(make, candidate_make) if make else 0.5
    return model_score * 0.78 + make_score * 0.22


def _lens_score(lens: Any, make: str, model: str) -> float:
    candidate_make = _object_text(lens, "maker", "make", "lens_make")
    candidate_model = _object_text(lens, "model", "name", "lens_model")
    model_score = _similarity(model, candidate_model)
    if not model_score:
        return 0.0
    make_score = _similarity(make, candidate_make) if make else 0.5
    return model_score * 0.82 + make_score * 0.18


def _best_match(items: Iterable[Any], score_fn: Any, *, minimum: float) -> Any | None:
    best_item = None
    best_score = -1.0
    for item in items:
        score = float(score_fn(item))
        if score > best_score:
            best_item, best_score = item, score
    return best_item if best_score >= minimum else None


def _database_xml_files() -> list[str]:
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(str(meipass)))
    module_root = Path(__file__).resolve().parent.parent
    roots.append(module_root)

    files: list[str] = []
    seen: set[str] = set()
    for root in roots:
        directory = root / "third_party" / "lensfun-db"
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.xml")):
            key = str(path.resolve())
            if key not in seen and path.is_file():
                seen.add(key)
                files.append(key)
    return files


def _load_database(lensfunpy: Any) -> Any:
    files = _database_xml_files()
    if files:
        try:
            # lensfunpy 1.18 accepts XML files, not the containing directory.
            return lensfunpy.Database(paths=files, load_bundled=True)
        except TypeError:
            # Small compatibility fallback for fake/older bindings used by
            # downstream applications; the normal 1.18 path is above.
            try:
                return lensfunpy.Database(paths=files)
            except TypeError:
                pass
    try:
        return lensfunpy.Database(load_bundled=True)
    except TypeError:
        return lensfunpy.Database()


def _find_camera(database: Any, make: str, model: str) -> Any | None:
    if not model:
        raise LensMatchError("相机 Model 缺失，无法安全匹配 Lensfun 配置")
    queries: list[tuple[Any, Any]] = []
    for query_make in (make, "", None):
        query = (query_make, model)
        if query not in queries:
            queries.append(query)
    candidates: list[Any] = []
    for query_make, query_model in queries:
        try:
            found = database.find_cameras(query_make, query_model, loose_search=True)
        except TypeError:
            try:
                found = database.find_cameras(query_make, query_model)
            except (TypeError, ValueError, AttributeError):
                continue
        except (TypeError, ValueError, AttributeError):
            continue
        if found:
            candidates.extend(found)
            # A maker-qualified exact result is preferable to broad fallback
            # results, but still rank all returned candidates below.
            if make and query_make == make:
                break
    camera = _best_match(candidates, lambda item: _camera_score(item, make, model), minimum=0.34)
    if camera is None:
        raise LensMatchError(f"未找到可靠的相机匹配: {_text(make)} {_text(model)}".strip())
    return camera


def _find_lens(database: Any, camera: Any, make: str, model: str) -> Any | None:
    if not model:
        raise LensMatchError("LensModel 缺失，无法安全匹配 Lensfun 配置")
    queries: list[Any] = []
    for query_make in (make, "", None):
        if query_make not in queries:
            queries.append(query_make)
    candidates: list[Any] = []
    for query_make in queries:
        try:
            found = database.find_lenses(camera, query_make, model, loose_search=True)
        except TypeError:
            try:
                found = database.find_lenses(camera, query_make, model)
            except (TypeError, ValueError, AttributeError):
                continue
        except (ValueError, AttributeError):
            continue
        if found:
            candidates.extend(found)
            if make and query_make == make:
                break
    lens = _best_match(candidates, lambda item: _lens_score(item, make, model), minimum=0.34)
    if lens is None:
        raise LensMatchError(f"未找到可靠的镜头匹配: {_text(make)} {_text(model)}".strip())
    return lens


def _flag(lensfunpy: Any, name: str) -> int:
    enum = getattr(lensfunpy, "ModifyFlags", None)
    value = getattr(enum, name, None) if enum is not None else None
    if value is None:
        value = getattr(lensfunpy, name, None)
    if value is None:
        value = getattr(lensfunpy, f"MODIFY_{name}", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise LensCorrectionError(f"lensfunpy 缺少 {name} 修正标志") from None


def _initialize_modifier(
    modifier: Any,
    focal_length: float,
    aperture: float,
    distance: float,
    image_dtype: np.dtype[Any],
    flags: int,
    scale: float,
) -> Any:
    """Call both the current and older lensfunpy initialize signatures."""

    try:
        return modifier.initialize(
            focal_length,
            aperture,
            distance,
            pixel_format=image_dtype,
            flags=flags,
            scale=scale,
        )
    except TypeError:
        try:
            return modifier.initialize(focal_length, aperture, distance, flags, scale)
        except TypeError:
            return modifier.initialize(focal_length, aperture, distance, flags)


def _enabled_flags(modifier: Any, initialized: Any, requested: int) -> int:
    values: list[Any] = [initialized]
    for name in ("flags", "modify_flags", "enabled_flags", "mod_flags"):
        value = getattr(modifier, name, None)
        if value is not None:
            values.append(value)
    getter = getattr(modifier, "get_mod_flags", None)
    if callable(getter):
        try:
            values.append(getter())
        except Exception:
            pass
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        # ``initialize`` is documented inconsistently across lensfunpy
        # builds: some report an integer flag mask, while others return a
        # floating-point scale or None.  Never reinterpret a float scale as a
        # bit mask (for example 1.0 must not accidentally enable one flag).
        if not isinstance(value, (int, np.integer)):
            continue
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if integer:
            return integer & requested
    # The released lensfunpy binding returns None from initialize and exposes
    # successful work through its API calls, so requested flags are the best
    # available state in that case.
    return requested


def _call_map(method: Any, y: int, width: int, height: int) -> Any:
    """Ask Lensfun for one output strip, with a no-argument fake fallback."""

    try:
        return method(0.0, float(y), int(width), int(height))
    except TypeError:
        # Some test doubles and old wrappers only expose a no-argument method.
        # Do not manufacture a full-image map for the production binding.
        return method()


def _combined_map(value: Any, height: int, width: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.size != height * width * 3 * 2:
        return None
    if array.shape == (height, width, 3, 2):
        result = array
    elif array.shape == (height, width, 2, 3):
        result = array.transpose(0, 1, 3, 2)
    else:
        result = array.reshape(height, width, 3, 2)
    return result if np.isfinite(result).all() else None


def _geometry_map(value: Any, height: int, width: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.size != height * width * 2:
        return None
    if array.shape == (height, width, 2):
        result = array
    else:
        result = array.reshape(height, width, 2)
    return result if np.isfinite(result).all() else None


def _apply_geometry(
    corrected: np.ndarray,
    modifier: Any,
    *,
    width: int,
    height: int,
    geometry_enabled: bool,
    tca_enabled: bool,
    distortion_enabled: bool,
) -> tuple[bool, bool, bool]:
    if not geometry_enabled:
        return False, False, False
    combined_method = getattr(modifier, "apply_subpixel_geometry_distortion", None)
    geometry_method = getattr(modifier, "apply_geometry_distortion", None)
    if not callable(combined_method) and not callable(geometry_method):
        return False, False, False

    # Mapping strips are remapped from a stable source copy.  Without that
    # copy, an earlier strip could alter samples needed by a later map.
    source = corrected.copy()
    combined_applied = False
    geometry_applied = False
    for y in range(0, height, STRIP_HEIGHT):
        strip_height = min(STRIP_HEIGHT, height - y)
        maps = None
        if callable(combined_method):
            maps = _combined_map(_call_map(combined_method, y, width, strip_height), strip_height, width)
        if maps is not None:
            for channel in range(3):
                remapped = cv2.remap(
                    source[..., channel],
                    maps[..., channel, 0],
                    maps[..., channel, 1],
                    interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT101,
                )
                corrected[y : y + strip_height, :, channel] = remapped
            combined_applied = True
            geometry_applied = True
            continue

        if not callable(geometry_method):
            continue
        coordinate_map = _geometry_map(
            _call_map(geometry_method, y, width, strip_height), strip_height, width
        )
        if coordinate_map is None:
            continue
        for channel in range(3):
            remapped = cv2.remap(
                source[..., channel],
                coordinate_map[..., 0],
                coordinate_map[..., 1],
                interpolation=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT101,
            )
            corrected[y : y + strip_height, :, channel] = remapped
        geometry_applied = True

    if not geometry_applied:
        return False, False, False
    # A combined map is capable of carrying both effects.  Respect the
    # initialize result when it explicitly disabled one of them.
    return (
        geometry_applied and distortion_enabled,
        combined_applied and tca_enabled,
        geometry_applied,
    )


def _failure(
    error: LensCorrectionError,
    image: np.ndarray,
    require_correction: bool,
) -> tuple[np.ndarray, LensCorrectionResult]:
    if require_correction:
        raise error
    return image.copy(), _result_unapplied()


def apply_lens_correction(
    image_rgb16: np.ndarray,
    metadata: Any,
    *,
    require_correction: bool = True,
) -> tuple[np.ndarray, LensCorrectionResult]:
    """Bake Lensfun correction into a copy of a 16-bit linear RGB image.

    ``metadata`` may be an EXIF mapping or an object containing an ``exif``
    mapping.  The source array is never modified.  With the default strict
    mode, missing dependencies, unsafe matches, invalid optical parameters,
    and a vignetting-only calibration all raise a typed exception.
    """

    if not isinstance(image_rgb16, np.ndarray):
        raise TypeError("image_rgb16 must be a numpy array")
    if image_rgb16.ndim != 3 or image_rgb16.shape[2] != 3:
        raise ValueError("image_rgb16 must have shape (height, width, 3)")
    if image_rgb16.dtype != np.uint16:
        raise TypeError("image_rgb16 must use uint16 samples")
    if image_rgb16.size == 0:
        raise ValueError("image_rgb16 must not be empty")

    make = _text(_metadata_value(metadata, "Make"))
    model = _text(_metadata_value(metadata, "Model"))
    lens_make = _text(_metadata_value(metadata, "LensMake"))
    lens_model = _text(_metadata_value(metadata, "LensModel"))
    focal_length = _number(_metadata_value(metadata, "FocalLength"))
    aperture = _number(_metadata_value(metadata, "FNumber"))
    distance = _number(_metadata_value(metadata, "SubjectDistance"))
    if focal_length is None or focal_length <= 0.0:
        return _failure(LensCorrectionError("FocalLength 无效，无法初始化 Lensfun"), image_rgb16, require_correction)
    if aperture is None or aperture <= 0.0:
        return _failure(LensCorrectionError("FNumber 无效，无法初始化 Lensfun"), image_rgb16, require_correction)
    # Lensfun accepts a very distant subject as a stable default.  EXIF often
    # omits SubjectDistance; only a present, finite positive value overrides it.
    if distance is None or distance <= 0.0:
        distance = 1000.0

    try:
        import lensfunpy  # type: ignore[import-not-found]
    except Exception as exc:
        return _failure(LensfunUnavailableError("未安装或无法导入 lensfunpy"), image_rgb16, require_correction)

    try:
        database = _load_database(lensfunpy)
        camera = _find_camera(database, make, model)
        lens = _find_lens(database, camera, lens_make, lens_model)
        crop_factor = _number(getattr(camera, "crop_factor", 1.0)) or 1.0
        if crop_factor <= 0.0:
            crop_factor = 1.0
        # lensfunpy 1.18's color-modification binding does not accept uint16
        # images (the export path intentionally stays uint16 to avoid a full
        # resolution float32 allocation).  Request only the coordinate stages
        # that can be safely and directly baked into the uint16 output.
        flags_by_name = {name: _flag(lensfunpy, name) for name in ("TCA", "DISTORTION", "SCALE")}
        requested_flags = 0
        for value in flags_by_name.values():
            requested_flags |= value
        height, width = int(image_rgb16.shape[0]), int(image_rgb16.shape[1])
        modifier = lensfunpy.Modifier(lens, float(crop_factor), width, height)
        initialized = _initialize_modifier(
            modifier,
            float(focal_length),
            float(aperture),
            float(distance),
            image_rgb16.dtype.type,
            requested_flags,
            0.0,
        )
        enabled = _enabled_flags(modifier, initialized, requested_flags)
        # lensfunpy uses scale=0 for its built-in automatic scaling/crop.  The
        # Python binding does not expose Lensfun's C++ GetAutoScale method.
        scale = _number(getattr(modifier, "scale", None))
        if scale is None:
            scale = 0.0

        corrected = image_rgb16.copy()
        # Vignetting is deliberately not requested here: lensfunpy's
        # apply_color_modification currently accepts float/uint8 buffers but
        # raises for uint16, and converting a full-resolution export to float
        # would be an unnecessary large allocation.  The result field remains
        # explicit so callers can report that only geometry/TCA was baked.
        vignetting_applied = False

        # lensfunpy 1.18 returns None from initialize, so requested flags alone
        # cannot prove a calibration exists.  Require the matched profile to
        # contain the corresponding calibration records before reporting that
        # correction as applied.
        distortion_enabled = bool(enabled & flags_by_name["DISTORTION"]) and bool(
            getattr(lens, "calib_distortion", ())
        )
        tca_enabled = bool(enabled & flags_by_name["TCA"]) and bool(
            getattr(lens, "calib_tca", ())
        )
        distortion_applied, tca_applied, geometry_applied = _apply_geometry(
            corrected,
            modifier,
            width=width,
            height=height,
            geometry_enabled=distortion_enabled or tca_enabled,
            tca_enabled=tca_enabled,
            distortion_enabled=distortion_enabled,
        )
        if not geometry_applied:
            error = LensCorrectionNotAppliedError(
                "Lensfun 匹配仅提供暗角或未提供可用坐标，未满足像素级 distortion/TCA 校正"
            )
            return _failure(error, image_rgb16, require_correction)
        result = LensCorrectionResult(
            applied=True,
            camera_name=_object_text(camera, "model", "name", "camera_model") or model or None,
            lens_name=_object_text(lens, "model", "name", "lens_model") or lens_model or None,
            distortion_applied=distortion_applied,
            tca_applied=tca_applied,
            vignetting_applied=vignetting_applied,
            scale=float(scale),
        )
        return corrected, result
    except LensCorrectionError as exc:
        return _failure(exc, image_rgb16, require_correction)
    except Exception as exc:
        return _failure(LensCorrectionError(f"Lensfun 校正失败: {type(exc).__name__}"), image_rgb16, require_correction)


__all__ = [
    "LensCorrectionError",
    "LensCorrectionNotAppliedError",
    "LensCorrectionResult",
    "LensMatchError",
    "LensfunUnavailableError",
    "apply_lens_correction",
]
