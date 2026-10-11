"""Careful rendering of supported DNG camera profiles for preview images.

This module implements the small, verified subset of the DNG profile pipeline
used by the offline ACR profile trial. It deliberately declines profiles whose
calibration or HueSatMap behavior is outside that subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import struct
import sys
import threading
from typing import Any

import cv2
import numpy as np
import rawpy

import native_dense
from native_renderer import (
    NativeRendererError, native_camera_profile, native_camera_profile_transfer,
    clear_camera_profile_gpu_cache,
)
from dng_writer import _embedded_profile_tags
from image_io import RAW_SUFFIXES, camera_profile_names, matching_embedded_profile_dng


PROFILE_PREVIEW_VERSION = "camera-profile-preview-v3-gpu-stages"
_backend_context = threading.local()


def get_last_profile_backend() -> str:
    return getattr(_backend_context, "profile", "python")


def get_last_transfer_backend() -> str:
    return getattr(_backend_context, "transfer", "python")

PROPHOTO_TO_XYZ50 = np.array(
    [[0.7977, 0.1352, 0.0313], [0.2880, 0.7119, 0.0001], [0.0, 0.0, 0.8249]],
    dtype=np.float64,
)
SRGB_TO_XYZ50 = np.array(
    [[0.4361, 0.3851, 0.1431], [0.2225, 0.7169, 0.0606], [0.0139, 0.0971, 0.7141]],
    dtype=np.float64,
)
_PROPHOTO_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ50) @ PROPHOTO_TO_XYZ50
_PROFILE_CACHE_SIZE = 24


@dataclass(frozen=True)
class Profile:
    """A validated camera profile and the file identity used to load it."""

    name: str
    fingerprint: tuple[Any, ...]
    color_matrix1: np.ndarray
    color_matrix2: np.ndarray
    forward_matrix1: np.ndarray
    forward_matrix2: np.ndarray
    tone_curve: np.ndarray
    look_table: np.ndarray
    look_dims: tuple[int, int, int]
    look_encoding: int
    baseline_exposure_offset: float


def _file_signature(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
        return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        return None


def _directory_dng_signature(path: Path) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    try:
        candidates = sorted(
            (item for item in path.parent.iterdir() if item.is_file() and item.suffix.lower() == ".dng"),
            key=lambda item: item.name.casefold(),
        )
    except OSError:
        return ()
    result = []
    for candidate in candidates:
        signature = _file_signature(candidate)
        if signature is not None:
            result.append((str(candidate.resolve()), signature))
    return tuple(result)


def _decode_tag(tag: Any) -> np.ndarray:
    """Decode numeric values returned by dng_writer's bounded TIFF reader."""
    kind = int(tag.kind)
    data = bytes(tag.data)
    if kind == 11:
        values = np.frombuffer(data, dtype="<f4").astype(np.float64)
    elif kind == 12:
        values = np.frombuffer(data, dtype="<f8").astype(np.float64)
    elif kind == 3:
        values = np.frombuffer(data, dtype="<u2").astype(np.float64)
    elif kind == 4:
        values = np.frombuffer(data, dtype="<u4").astype(np.float64)
    elif kind == 9:
        values = np.frombuffer(data, dtype="<i4").astype(np.float64)
    elif kind in (5, 10):
        numerator_dtype = "<i4" if kind == 10 else "<u4"
        parts = np.frombuffer(data, dtype=numerator_dtype).reshape(-1, 2)
        if np.any(parts[:, 1] == 0):
            raise ValueError("profile contains a zero rational denominator")
        values = parts[:, 0].astype(np.float64) / parts[:, 1].astype(np.float64)
    else:
        raise ValueError("profile contains a non-numeric tag")
    if values.size != int(tag.count) or not np.isfinite(values).all():
        raise ValueError("profile contains an incomplete or non-finite numeric tag")
    return values


def _tag_map(tags: list[Any]) -> dict[int, Any]:
    mapped = {int(tag.code): tag for tag in tags}
    if len(mapped) != len(tags):
        raise ValueError("profile contains duplicate tags")
    return mapped


def _require_values(tags: dict[int, Any], code: int, count: int) -> np.ndarray:
    tag = tags.get(code)
    if tag is None or int(tag.count) != count:
        raise ValueError(f"profile tag {code} is missing or has the wrong size")
    values = _decode_tag(tag)
    if values.size != count:
        raise ValueError(f"profile tag {code} is incomplete")
    return values


def _validate_identity_calibration(tags: dict[int, Any]) -> None:
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    for code in (50723, 50724):
        tag = tags.get(code)
        if tag is None:
            continue
        if int(tag.count) != 9 or not np.allclose(_decode_tag(tag), identity, rtol=0.0, atol=1e-6):
            raise ValueError("non-identity camera calibration is not supported")


def _validate_identity_huesat_maps(tags: dict[int, Any]) -> None:
    map_codes = {50937, 50938, 50939}
    present = map_codes.intersection(tags)
    if not present:
        return
    if present != map_codes:
        raise ValueError("incomplete HueSatMap is not supported")
    dims_tag = tags[50937]
    if int(dims_tag.count) != 3:
        raise ValueError("invalid HueSatMap dimensions")
    dims = _decode_tag(dims_tag)
    if not np.equal(dims, np.floor(dims)).all():
        raise ValueError("invalid HueSatMap dimensions")
    hue, saturation, value = (int(part) for part in dims)
    entries = hue * saturation * value
    if hue < 1 or saturation < 1 or value < 1 or entries > 1_000_000:
        raise ValueError("invalid HueSatMap dimensions")
    identity_entry = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    for code in (50938, 50939):
        data = _require_values(tags, code, entries * 3).reshape(-1, 3)
        if not np.allclose(data, identity_entry, rtol=0.0, atol=1e-6):
            raise ValueError("non-identity HueSatMap is not supported")


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _parse_profile(tags_list: list[Any], expected_name: str, fingerprint: tuple[Any, ...]) -> Profile:
    tags = _tag_map(tags_list)
    name_tag = tags.get(50936)
    if name_tag is None or int(name_tag.kind) != 2:
        raise ValueError("profile name is missing")
    name = bytes(name_tag.data).rstrip(b"\0").decode("ascii", "strict")
    if name != expected_name:
        raise ValueError("profile name does not match the selected camera profile")

    # Camera matrices, ForwardMatrices and their illuminants must form the
    # supported two-illuminant profile pair (D50 warm / D65 daylight).
    illuminants = []
    for code in (50778, 50779):
        value = _require_values(tags, code, 1)
        if int(value[0]) != value[0]:
            raise ValueError("invalid calibration illuminant")
        illuminants.append(int(value[0]))
    if illuminants != [17, 21]:
        raise ValueError("only calibration illuminants 17 and 21 are supported")

    matrices = []
    for code in (50721, 50722, 50964, 50965):
        matrix = _require_values(tags, code, 9).reshape(3, 3)
        if abs(float(np.linalg.det(matrix))) < 1e-8:
            raise ValueError("profile matrix is singular")
        matrices.append(_readonly(matrix))

    _validate_identity_calibration(tags)
    if 50727 in tags and not np.allclose(_require_values(tags, 50727, 3), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("non-identity analog balance is not supported")
    _validate_identity_huesat_maps(tags)

    curve = _require_values(tags, 50940, int(tags.get(50940).count) if tags.get(50940) else 0)
    if curve.size < 4 or curve.size % 2:
        raise ValueError("invalid profile tone curve")
    curve = curve.reshape(-1, 2)
    if (curve.shape[0] < 2 or abs(float(curve[0, 0])) > 1e-6
            or abs(float(curve[-1, 0]) - 1.0) > 1e-6
            or abs(float(curve[0, 1])) > 1e-6
            or abs(float(curve[-1, 1]) - 1.0) > 1e-6
            or np.any(np.diff(curve[:, 0]) <= 0.0)
            or np.any(np.diff(curve[:, 1]) < -1e-7)
            or np.any(curve < 0.0) or np.any(curve > 1.0)):
        raise ValueError("profile tone curve is not a finite monotonic unit curve")

    dims = _require_values(tags, 50981, 3)
    if not np.equal(dims, np.floor(dims)).all():
        raise ValueError("invalid profile look-table dimensions")
    hue, saturation, value = (int(part) for part in dims)
    if hue < 1 or saturation < 2 or value < 2 or hue * saturation * value > 1_000_000:
        raise ValueError("invalid profile look-table dimensions")
    look = _require_values(tags, 50982, hue * saturation * value * 3)
    look = look.reshape(value, hue, saturation, 3).astype(np.float32)
    # DNG LookTable hue deltas are degrees and may be negative. Saturation and
    # value are non-negative multipliers; Adobe's Camera Standard has measured
    # saturation values above 8, so keep a bounded but useful supported range.
    if (not np.isfinite(look).all() or np.any(look[..., 0] < -360.0)
            or np.any(look[..., 0] > 360.0)
            or np.any(look[..., 1:] < 0.0) or np.any(look[..., 1:] > 16.0)):
        raise ValueError("profile look table contains unsupported values")

    encoding = _require_values(tags, 51108, 1)
    if int(encoding[0]) != encoding[0] or int(encoding[0]) not in (0, 1):
        raise ValueError("unsupported profile look-table encoding")
    offset_tag = tags.get(51109)
    offset = float(_decode_tag(offset_tag)[0]) if offset_tag is not None else 0.0
    if not math.isfinite(offset) or abs(offset) > 8.0:
        raise ValueError("invalid profile baseline exposure offset")

    return Profile(
        name=name,
        fingerprint=fingerprint,
        color_matrix1=matrices[0],
        color_matrix2=matrices[1],
        forward_matrix1=matrices[2],
        forward_matrix2=matrices[3],
        tone_curve=_readonly(curve.astype(np.float32)),
        look_table=_readonly(look),
        look_dims=(hue, saturation, value),
        look_encoding=int(encoding[0]),
        baseline_exposure_offset=offset,
    )


def _fingerprint(
    path: Path,
    stat_signature: tuple[int, int, int, int],
    tags: list[Any],
    source_name: str,
    source_signature: tuple[int, int, int, int],
    neighbor_signature: tuple[tuple[str, tuple[int, int, int, int]], ...],
) -> tuple[Any, ...]:
    digest = hashlib.sha256()
    for tag in sorted(tags, key=lambda item: int(item.code)):
        digest.update(struct.pack("<HI", int(tag.code), int(tag.count)))
        digest.update(bytes(tag.data))
    return (
        str(path.resolve()), stat_signature, digest.hexdigest(),
        source_name, source_signature, neighbor_signature,
    )


def _profile_name(source_info: dict[str, Any]) -> str | None:
    raw = source_info.get("SourceCameraProfileName")
    if raw is None:
        raw = source_info.get("CameraProfile")
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if value.startswith("Group: "):
        value = value[len("Group: "):].strip()
    return value or None


def _has_enhanced_rgb_subifd(path: Path) -> bool:
    """Return True when a classic TIFF DNG contains an Enhanced Image Data IFD."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(8)
            if len(header) != 8 or header[:2] not in (b"II", b"MM"):
                return True
            endian = "<" if header[:2] == b"II" else ">"
            if struct.unpack_from(endian + "H", header, 2)[0] != 42:
                return True
            root_offset = struct.unpack_from(endian + "I", header, 4)[0]

            def read_at(offset: int, count: int) -> bytes:
                if offset < 0 or count < 0 or offset + count > file_size:
                    raise ValueError("TIFF value points outside the file")
                handle.seek(offset)
                result = handle.read(count)
                if len(result) != count:
                    raise ValueError("truncated TIFF value")
                return result

            def read_ifd(offset: int) -> dict[int, tuple[int, int, bytes | int]]:
                count_data = read_at(offset, 2)
                count = struct.unpack(endian + "H", count_data)[0]
                if count > 4096:
                    raise ValueError("invalid IFD size")
                entries = read_at(offset + 2, count * 12)
                read_at(offset + 2 + count * 12, 4)  # Validate the next-IFD field.
                result: dict[int, tuple[int, int, bytes | int]] = {}
                type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
                              8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
                for index in range(count):
                    entry = entries[index * 12:(index + 1) * 12]
                    code, kind, items, value = struct.unpack_from(endian + "HHII", entry)
                    if kind not in type_sizes:
                        raise ValueError("invalid TIFF type")
                    byte_count = type_sizes[kind] * items
                    if byte_count <= 4:
                        raw_value: bytes | int = entry[8:8 + byte_count]
                    else:
                        raw_value = value
                    result[code] = (kind, items, raw_value)
                return result

            root = read_ifd(root_offset)
            subifds = root.get(330)
            if subifds is None:
                return False
            kind, count, raw_value = subifds
            if kind not in (4, 13) or count > 128:
                return True
            if isinstance(raw_value, bytes):
                raw = raw_value
            else:
                raw = read_at(raw_value, count * 4)
            if len(raw) != count * 4:
                return True
            child_offsets = struct.unpack(endian + "I" * count, raw)
            for child_offset in child_offsets:
                child = read_ifd(int(child_offset))
                entry = child.get(254)
                if entry is None:
                    continue
                child_kind, child_count, child_value = entry
                if child_count != 1 or not isinstance(child_value, bytes):
                    return True
                if child_kind == 4:
                    child_type = struct.unpack(endian + "I", child_value)[0]
                elif child_kind == 3:
                    child_type = struct.unpack(endian + "H", child_value)[0]
                else:
                    return True
                if child_type & 16:
                    return True
            return False
    except (OSError, ValueError, struct.error, OverflowError):
        # Unknown or malformed DNG IFD layouts are not a safe CFA source.
        return True


@lru_cache(maxsize=_PROFILE_CACHE_SIZE)
def _resolve_cached(
    source_name: str,
    source_signature: tuple[int, int, int, int],
    neighbor_signature: tuple[tuple[str, tuple[int, int, int, int]], ...],
    source_info_key: tuple[tuple[str, str], ...],
) -> Profile | None:
    source = Path(source_name)
    info = dict(source_info_key)
    if source.suffix.lower() == ".dng" and _has_enhanced_rgb_subifd(source):
        return None

    names = camera_profile_names(source)
    # ExifTool's direct camera profile name is authoritative, while caller
    # metadata supplies already-read Make/Model when the external tool is absent.
    combined = dict(info)
    combined.update(names)
    expected_name = _profile_name(combined)
    if not expected_name:
        return None

    if source.suffix.lower() == ".dng":
        reference = source
    else:
        reference = matching_embedded_profile_dng(source, combined)
        if reference is None:
            return None
    reference = Path(reference).expanduser().resolve()
    reference_signature = _file_signature(reference)
    if reference_signature is None:
        return None
    try:
        tags = _embedded_profile_tags(reference, expected_name)
        return _parse_profile(
            tags, expected_name,
            _fingerprint(reference, reference_signature, tags, source_name, source_signature, neighbor_signature),
        )
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, np.linalg.LinAlgError):
        return None


def resolve_profile(source_path: str | Path, source_info: dict[str, Any]) -> Profile | None:
    """Return only a validated, strictly matching adjacent or embedded profile."""
    source = Path(source_path).expanduser().resolve()
    if source.suffix.lower() not in RAW_SUFFIXES:
        return None
    source_signature = _file_signature(source)
    if source_signature is None or not isinstance(source_info, dict):
        return None
    combined = dict(source_info)
    compact = tuple(sorted(
        (key, value.strip()) for key, value in combined.items()
        if key in {"SourceCameraProfileName", "CameraProfile", "Make", "Model", "NikonPictureControlName"}
        and isinstance(value, str)
    ))
    neighbors = _directory_dng_signature(source)
    try:
        return _resolve_cached(str(source), source_signature, neighbors, compact)
    except (OSError, ValueError, TypeError):
        return None


def load_camera_rgb(
    source_path: str | Path,
    target_shape: tuple[int, ...],
    preview: bool,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Decode unbalanced camera-space RGB and match the requested image size."""
    if len(target_shape) not in (2, 3):
        raise ValueError("target_shape must contain height and width")
    height, width = int(target_shape[0]), int(target_shape[1])
    if height < 1 or width < 1 or (len(target_shape) == 3 and int(target_shape[2]) != 3):
        raise ValueError("target_shape must describe a non-empty RGB image")
    with rawpy.imread(str(Path(source_path))) as raw:
        camera_wb = np.asarray(raw.camera_whitebalance[:3], dtype=np.float64)
        if camera_wb.shape != (3,) or not np.isfinite(camera_wb).all() or np.any(camera_wb <= 0):
            raise ValueError("camera white balance is unavailable")
        neutral = 1.0 / camera_wb
        neutral /= neutral[1]
        rgb = raw.postprocess(
            half_size=bool(preview), output_bps=16, no_auto_bright=True,
            gamma=(1.0, 1.0), output_color=rawpy.ColorSpace.raw,
            user_wb=[1.0, 1.0, 1.0, 1.0],
        )
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint16:
        raise ValueError("LibRaw did not return camera-space uint16 RGB")
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb), tuple(float(value) for value in neutral)


def _fit_transfer_matrix(camera_u16: np.ndarray, reference_u16: np.ndarray) -> np.ndarray:
    camera_samples = camera_u16[::16, ::16].reshape(-1, 3).astype(np.float64) / 65535.0
    reference_samples = reference_u16[::16, ::16].reshape(-1, 3).astype(np.float64) / 65535.0
    positive = (camera_samples.min(axis=1) > 0.01) & (reference_samples.min(axis=1) > 0.01)
    for upper in (0.30, 0.50, 0.75, 0.95):
        usable = (
            positive & (camera_samples.max(axis=1) < upper)
            & (reference_samples.max(axis=1) < upper)
        )
        if np.count_nonzero(usable) < 100:
            continue
        fitted, _, rank, _ = np.linalg.lstsq(camera_samples[usable], reference_samples[usable], rcond=None)
        if rank != 3 or np.linalg.cond(fitted) > 100:
            continue
        error = np.mean(np.abs(camera_samples[usable] @ fitted - reference_samples[usable]))
        if error <= 0.003:
            return np.linalg.inv(fitted).astype(np.float32)
    raise ValueError("camera-to-reference color transform does not fit")


def transfer_enhancement(
    camera_u16: np.ndarray,
    reference_u16: np.ndarray,
    processed_u16: np.ndarray,
    *,
    backend: str = "cpu",
) -> np.ndarray:
    """Transfer luma and color residual changes onto camera-native RGB."""
    images = (camera_u16, reference_u16, processed_u16)
    if any(image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 3 for image in images):
        raise ValueError("transfer requires aligned uint16 RGB images")
    if camera_u16.shape != reference_u16.shape or camera_u16.shape != processed_u16.shape:
        raise ValueError("transfer images must have identical shapes")
    _backend_context.transfer = "identity"
    if np.array_equal(reference_u16, processed_u16):
        return np.ascontiguousarray(camera_u16.copy())

    srgb_to_camera = _fit_transfer_matrix(camera_u16, reference_u16)
    transfer_setting = os.environ.get("IMPRINT_NATIVE_CAMERA_PROFILE_TRANSFER")
    use_gpu_transfer = (
        backend in ("auto", "native") and transfer_setting != "0" and
        (sys.platform == "darwin" or transfer_setting == "1") and
        (transfer_setting == "1" or camera_u16.shape[0] * camera_u16.shape[1] >= 1_000_000)
    )
    if use_gpu_transfer:
        try:
            output, actual_backend = native_camera_profile_transfer(
                camera_u16, reference_u16, processed_u16, srgb_to_camera,
            )
            _backend_context.transfer = actual_backend
            return output
        except NativeRendererError:
            # Keep the existing CPU ABI and NumPy routes available when the
            # selected GPU or its optional transfer symbols are unavailable.
            pass
    try:
        output = native_dense.camera_profile_transfer(
            camera_u16, reference_u16, processed_u16, srgb_to_camera)
        _backend_context.transfer = "cpp_cpu"
        return output
    except NativeRendererError:
        _backend_context.transfer = "python"
    output = np.empty_like(camera_u16)
    luminance = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    for start in range(0, camera_u16.shape[0], 128):
        end = min(start + 128, camera_u16.shape[0])
        processed = processed_u16[start:end].astype(np.float32)
        reference = reference_u16[start:end].astype(np.float32)
        camera = camera_u16[start:end].astype(np.float32)
        processed_luma = np.maximum(processed @ luminance, 0.0)
        reference_luma = np.maximum(reference @ luminance, 0.0)
        ratio = processed_luma / np.maximum(reference_luma, 1.0)
        residual = processed - reference * ratio[:, :, None]
        dark = reference_luma < 1.0
        residual[dark] = 0.0
        ratio[dark] = 0.0
        chunk = camera * ratio[:, :, None] + residual @ srgb_to_camera
        output[start:end] = np.clip(np.rint(chunk), 0, 65535).astype(np.uint16)
    return output


def _estimate_cct(xy: np.ndarray) -> float:
    denominator = float(xy[1] - 0.1858)
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        raise ValueError("invalid white chromaticity")
    n = (float(xy[0]) - 0.3320) / denominator
    cct = -449.0 * n**3 + 3525.0 * n**2 - 6823.3 * n + 5520.33
    return float(np.clip(cct, 1667.0, 25000.0))


def _camera_transform(profile: Profile, neutral: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray((0.3457, 0.3585), dtype=np.float64)
    weight = 0.5
    for _ in range(30):
        cct = _estimate_cct(xy)
        weight = float(np.clip((1.0 / cct - 1.0 / 6500.0) / (1.0 / 2850.0 - 1.0 / 6500.0), 0.0, 1.0))
        color = profile.color_matrix1 * weight + profile.color_matrix2 * (1.0 - weight)
        xyz = np.linalg.solve(color, neutral)
        total = float(xyz.sum())
        if not np.isfinite(xyz).all() or abs(total) < 1e-12:
            raise ValueError("camera neutral cannot be mapped to a white point")
        next_xy = xyz[:2] / total
        if np.abs(next_xy - xy).sum() < 1e-7:
            xy = next_xy
            break
        xy = next_xy
    forward = profile.forward_matrix1 * weight + profile.forward_matrix2 * (1.0 - weight)
    white = neutral / float(np.max(neutral))
    matrix = np.linalg.inv(PROPHOTO_TO_XYZ50) @ forward @ np.diag(1.0 / white)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-8:
        raise ValueError("profile camera transform is invalid")
    return matrix.astype(np.float32), white.astype(np.float32)


def _encode_srgb(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1.0 / 2.4) - 0.055)


def _decode_srgb(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0.04045, values / 12.92, np.power((values + 0.055) / 1.055, 2.4))


def _apply_look(rgb: np.ndarray, profile: Profile) -> np.ndarray:
    hue_count, saturation_count, value_count = profile.look_dims
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2HSV)
    encoded = _encode_srgb(hsv[..., 2]) if profile.look_encoding == 1 else hsv[..., 2]
    hue_coord = (hsv[..., 0] % 360.0) * (hue_count / 360.0)
    sat_coord = np.clip(hsv[..., 1], 0.0, 1.0) * (saturation_count - 1)
    value_coord = np.clip(encoded, 0.0, 1.0) * (value_count - 1)
    h0 = np.floor(hue_coord).astype(np.int32)
    h1 = (h0 + 1) % hue_count
    s0 = np.minimum(np.floor(sat_coord).astype(np.int32), saturation_count - 2)
    v0 = np.minimum(np.floor(value_coord).astype(np.int32), value_count - 2)
    hf, sf, vf = hue_coord - h0, sat_coord - s0, value_coord - v0
    mods = np.zeros_like(rgb, dtype=np.float32)
    for dh in (0, 1):
        hw = hf if dh else 1.0 - hf
        hi = h1 if dh else h0
        for ds in (0, 1):
            sw = sf if ds else 1.0 - sf
            for dv in (0, 1):
                vw = vf if dv else 1.0 - vf
                mods += (hw * sw * vw)[..., None] * profile.look_table[v0 + dv, hi, s0 + ds]
    hsv[..., 0] = (hsv[..., 0] + mods[..., 0]) % 360.0
    hsv[..., 1] = np.clip(hsv[..., 1] * mods[..., 1], 0.0, 1.0)
    new_value = np.clip(encoded * mods[..., 2], 0.0, 1.0)
    hsv[..., 2] = _decode_srgb(new_value) if profile.look_encoding == 1 else new_value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _apply_tone(rgb: np.ndarray, curve: np.ndarray) -> np.ndarray:
    low = rgb.min(axis=-1)
    high = rgb.max(axis=-1)
    new_low = np.interp(low, curve[:, 0], curve[:, 1])
    new_high = np.interp(high, curve[:, 0], curve[:, 1])
    span = high - low
    position = np.divide(rgb - low[..., None], span[..., None], out=np.zeros_like(rgb), where=span[..., None] > 1e-12)
    return new_low[..., None] + (new_high - new_low)[..., None] * position


def _project_srgb(rgb: np.ndarray) -> np.ndarray:
    srgb = rgb @ _PROPHOTO_TO_SRGB.astype(np.float32).T
    encoded = _encode_srgb(srgb)
    return np.clip(np.rint(encoded * 255.0), 0.0, 255.0).astype(np.uint8)


def render_profile(
    camera_u16: np.ndarray,
    neutral: tuple[float, float, float] | np.ndarray,
    profile: Profile,
    exposure_ev: float,
    *, backend: str = "cpu",
) -> np.ndarray:
    """Render camera-space uint16 RGB through a validated DNG profile."""
    if camera_u16.dtype != np.uint16 or camera_u16.ndim != 3 or camera_u16.shape[2] != 3:
        raise ValueError("profile renderer requires uint16 RGB")
    if not isinstance(profile, Profile):
        raise TypeError("profile must be a validated Profile")
    neutral_array = np.asarray(neutral, dtype=np.float64)
    if neutral_array.shape != (3,) or not np.isfinite(neutral_array).all() or np.any(neutral_array <= 0):
        raise ValueError("neutral must contain three finite positive values")
    if not math.isfinite(float(exposure_ev)) or abs(float(exposure_ev)) > 32:
        raise ValueError("exposure_ev must be finite and within a supported range")
    matrix, white = _camera_transform(profile, neutral_array)
    exposure_scale = np.float32(2.0 ** (float(exposure_ev) + profile.baseline_exposure_offset))
    _backend_context.profile = "python"
    if backend in ("auto", "native") and os.environ.get("IMPRINT_NATIVE_DENSE", "1") != "0":
        try:
            output, actual = native_camera_profile(
                camera_u16, matrix, white, exposure_scale, profile.look_table,
                profile.look_dims, profile.look_encoding, profile.tone_curve, _PROPHOTO_TO_SRGB)
            _backend_context.profile = actual
            return output
        except NativeRendererError:
            pass
    try:
        output = native_dense.camera_profile_pixels(
            camera_u16, matrix, white, exposure_scale, profile.look_table,
            profile.look_dims, profile.look_encoding, profile.tone_curve,
            _PROPHOTO_TO_SRGB)
        _backend_context.profile = "cpp_cpu"
        return output
    except NativeRendererError:
        pass
    output = np.empty(camera_u16.shape, dtype=np.uint8)
    for start in range(0, camera_u16.shape[0], 128):
        end = min(start + 128, camera_u16.shape[0])
        camera = camera_u16[start:end].astype(np.float32) / np.float32(65535.0)
        base = np.clip(np.minimum(camera, white) @ matrix.T, 0.0, 1.0)
        exposed = np.clip(base * exposure_scale, 0.0, 1.0)
        looked = _apply_look(exposed, profile)
        toned = _apply_tone(looked, profile.tone_curve)
        output[start:end] = _project_srgb(toned)
    if not np.isfinite(output).all():
        raise ValueError("profile renderer generated non-finite output")
    return output
