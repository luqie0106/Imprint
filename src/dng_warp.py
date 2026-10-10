"""Bounded parsing and application of DNG OpcodeList3 WarpRectilinear.

Only full active-area, ClassicTIFF DNG warp metadata is accepted. Unsupported
metadata returns ``None`` from :func:`parse_dng_warp`; malformed metadata is a
hard error so callers cannot silently route a broken file through another
correction path.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import struct
import threading
from typing import BinaryIO

import numpy as np

from dng_gainmap import (
    ACTIVE_AREA_TAG,
    DNGGainMapError,
    IMAGE_HEIGHT_TAG,
    IMAGE_WIDTH_TAG,
    OPCODE_LIST3_TAG,
    ORIENTATION_TAG,
    SUB_IFDS_TAG,
    _Entry,
    _TIFF_TYPE_SIZE,
    _entry_bytes,
    _entry_integers,
    _parse_gainmap_payload,
    _read_at,
    _read_ifd,
)


WARP_RECTILINEAR_OPCODE_ID = 1
GAIN_MAP_OPCODE_ID = 9
DEFAULT_SCALE_TAG = 50718
DEFAULT_CROP_ORIGIN_TAG = 50719
DEFAULT_CROP_SIZE_TAG = 50720
_OPTIONAL_OPCODE_FLAG = 1
_SUPPORTED_OPCODE_VERSION = 0x01030000
_MAX_IFDS = 32
_MAX_IFD_ENTRIES = 4096
_MAX_SUB_IFDS = 32
_MAX_OPCODE_COUNT = 64
_MAX_OPCODE_LIST_BYTES = 16 * 1024 * 1024
_MAX_VALUES = 1 << 29
_STRIP_HEIGHT = 32
_METADATA_CACHE_LIMIT = 8


class DNGWarpError(DNGGainMapError):
    """Raised when a DNG WarpRectilinear TIFF or opcode is malformed."""


@dataclass(frozen=True)
class DNGWarpRectilinear:
    """A supported stage-3 warp in sensor-coordinate order."""

    planes: int
    coefficients: np.ndarray
    center_x: float
    center_y: float
    pixel_scale_v: float
    image_width: int
    image_height: int
    active_area: tuple[int, int, int, int]
    orientation: int
    gain_map_before_warp: bool
    gain_map_present: bool
    default_crop_origin: tuple[float, float] | None
    default_crop_size: tuple[float, float] | None
    fingerprint: tuple[object, ...]

    @property
    def tca_applied(self) -> bool:
        """Whether the three RGB planes carry different warp coefficients."""
        return not bool(np.array_equal(self.coefficients[0], self.coefficients[1])
                        and np.array_equal(self.coefficients[1], self.coefficients[2]))


_metadata_cache: OrderedDict[tuple[object, ...], DNGWarpRectilinear | None] = OrderedDict()
_metadata_cache_lock = threading.Lock()


def _fingerprint(path: Path) -> tuple[object, ...]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise DNGWarpError("Unable to read DNG metadata") from exc
    try:
        resolved = str(path.resolve(strict=True))
    except OSError as exc:
        raise DNGWarpError("Unable to read DNG metadata") from exc
    return (resolved, stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _entry_numeric_pair(
    file: BinaryIO,
    entry: _Entry,
    order: str,
    file_size: int,
    *,
    tag_name: str,
) -> tuple[float, float] | None:
    """Read a two-value DNG scale/crop tag without accepting unknown encodings."""
    if entry.count != 2 or entry.kind not in (3, 4, 5, 11, 12):
        return None
    try:
        raw = _entry_bytes(file, entry, order, file_size, limit=16)
    except DNGGainMapError as exc:
        raise DNGWarpError(f"{tag_name} field is malformed") from exc
    if entry.kind in (3, 4):
        values = struct.unpack(order + ("2H" if entry.kind == 3 else "2I"), raw)
    elif entry.kind == 5:
        numerators_denominators = struct.unpack(order + "4I", raw)
        if numerators_denominators[1] == 0 or numerators_denominators[3] == 0:
            raise DNGWarpError(f"{tag_name} contains a zero denominator")
        values = (numerators_denominators[0] / numerators_denominators[1],
                  numerators_denominators[2] / numerators_denominators[3])
    elif entry.kind == 11:
        values = struct.unpack(order + "2f", raw)
    else:
        values = struct.unpack(order + "2d", raw)
    if not all(math.isfinite(float(value)) for value in values):
        raise DNGWarpError(f"{tag_name} values must be finite")
    return float(values[0]), float(values[1])


def _parse_warp_payload(payload: bytes) -> tuple[int, np.ndarray, float, float]:
    if len(payload) < 4:
        raise DNGWarpError("WarpRectilinear payload is truncated")
    planes = struct.unpack_from(">I", payload, 0)[0]
    if planes not in (1, 3):
        raise DNGWarpError("WarpRectilinear plane count must be one or three")
    expected_parameter_bytes = 4 + planes * 6 * 8 + 2 * 8
    if len(payload) != expected_parameter_bytes:
        raise DNGWarpError("WarpRectilinear payload size does not match its plane count")

    values = struct.unpack_from(">" + "d" * (planes * 6 + 2), payload, 4)
    coefficient_values = values[:planes * 6]
    center_x, center_y = values[-2:]
    if not all(math.isfinite(value) for value in values):
        raise DNGWarpError("WarpRectilinear parameters must be finite")
    if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
        raise DNGWarpError("WarpRectilinear center must be within [0, 1]")

    coefficients = np.asarray(coefficient_values, dtype=np.float64).reshape(planes, 6)
    if planes == 1:
        coefficients = np.broadcast_to(coefficients, (3, 6)).copy()
    else:
        coefficients = coefficients.copy()
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float32)
    if not np.isfinite(coefficients).all():
        raise DNGWarpError("WarpRectilinear parameters exceed float32 range")
    coefficients.setflags(write=False)
    return planes, coefficients, float(center_x), float(center_y)


def _parse_opcode_list(payload: bytes):
    if len(payload) < 4:
        raise DNGWarpError("OpcodeList3 is missing its opcode count")
    opcode_count = struct.unpack_from(">I", payload, 0)[0]
    if opcode_count > _MAX_OPCODE_COUNT:
        raise DNGWarpError("OpcodeList3 contains too many opcodes")
    position = 4
    warps: list[tuple[int, np.ndarray, float, float]] = []
    gain_positions: list[int] = []
    unsupported_mandatory = False
    for opcode_index in range(opcode_count):
        if len(payload) - position < 16:
            raise DNGWarpError("OpcodeList3 contains a truncated opcode header")
        opcode_id, version, flags, data_size = struct.unpack_from(">4I", payload, position)
        position += 16
        if data_size > len(payload) - position:
            raise DNGWarpError("Opcode data extends beyond OpcodeList3")
        data = payload[position:position + data_size]
        position += data_size

        if opcode_id == WARP_RECTILINEAR_OPCODE_ID:
            if version != _SUPPORTED_OPCODE_VERSION:
                return None
            if flags & ~_OPTIONAL_OPCODE_FLAG:
                return None
            warps.append(_parse_warp_payload(data))
        elif opcode_id == GAIN_MAP_OPCODE_ID:
            if version != _SUPPORTED_OPCODE_VERSION:
                return None
            if flags & ~_OPTIONAL_OPCODE_FLAG:
                return None
            try:
                _parse_gainmap_payload(data)
            except DNGGainMapError as exc:
                raise DNGWarpError(str(exc)) from exc
            gain_positions.append(opcode_index)
        elif not (flags & _OPTIONAL_OPCODE_FLAG):
            unsupported_mandatory = True

    if position != len(payload):
        raise DNGWarpError("OpcodeList3 has trailing bytes")
    if unsupported_mandatory or len(warps) > 1 or len(gain_positions) > 1:
        return None
    if not warps:
        return None
    return warps[0], bool(gain_positions), bool(gain_positions and gain_positions[0] < next(
        index for index, opcode_id in enumerate(_opcode_ids(payload))
        if opcode_id == WARP_RECTILINEAR_OPCODE_ID
    ))


def _opcode_ids(payload: bytes) -> list[int]:
    """Return opcode IDs after the primary parser has checked all boundaries."""
    ids: list[int] = []
    count = struct.unpack_from(">I", payload, 0)[0]
    position = 4
    for _ in range(count):
        opcode_id, _version, _flags, data_size = struct.unpack_from(">4I", payload, position)
        ids.append(opcode_id)
        position += 16 + data_size
    return ids


def _read_ifd_image_metadata(file: BinaryIO, entries: dict[int, _Entry], order: str,
                             file_size: int, inherited_orientation: int):
    active_area = None
    if ACTIVE_AREA_TAG in entries:
        try:
            area = _entry_integers(file, entries[ACTIVE_AREA_TAG], order, file_size,
                                   accepted_types=(4,), max_count=4)
        except DNGGainMapError as exc:
            raise DNGWarpError("DNG ActiveArea tag is malformed") from exc
        if len(area) != 4 or area[2] <= area[0] or area[3] <= area[1]:
            raise DNGWarpError("DNG ActiveArea tag is invalid")
        active_area = tuple(int(value) for value in area)

    dimensions: dict[int, int] = {}
    for tag in (IMAGE_WIDTH_TAG, IMAGE_HEIGHT_TAG):
        if tag in entries:
            try:
                value = _entry_integers(file, entries[tag], order, file_size,
                                        accepted_types=(3, 4), max_count=1)
            except DNGGainMapError as exc:
                raise DNGWarpError("TIFF image dimensions are malformed") from exc
            if len(value) != 1 or value[0] <= 0:
                raise DNGWarpError("TIFF image dimensions are invalid")
            dimensions[tag] = int(value[0])

    orientation = inherited_orientation
    if ORIENTATION_TAG in entries:
        try:
            value = _entry_integers(file, entries[ORIENTATION_TAG], order, file_size,
                                    accepted_types=(3,), max_count=1)
        except DNGGainMapError as exc:
            raise DNGWarpError("TIFF Orientation tag is malformed") from exc
        if len(value) != 1 or not 1 <= value[0] <= 8:
            raise DNGWarpError("TIFF Orientation tag is invalid")
        orientation = int(value[0])

    scale_pair = None
    if DEFAULT_SCALE_TAG in entries:
        scale_pair = _entry_numeric_pair(file, entries[DEFAULT_SCALE_TAG], order, file_size,
                                         tag_name="DefaultScale")
        if scale_pair is None:
            return None
        if scale_pair[0] <= 0.0 or scale_pair[1] <= 0.0:
            raise DNGWarpError("DefaultScale values must be positive")

    crop_origin = None
    if DEFAULT_CROP_ORIGIN_TAG in entries:
        crop_origin = _entry_numeric_pair(file, entries[DEFAULT_CROP_ORIGIN_TAG], order,
                                          file_size, tag_name="DefaultCropOrigin")
        if crop_origin is None:
            return None
    crop_size = None
    if DEFAULT_CROP_SIZE_TAG in entries:
        crop_size = _entry_numeric_pair(file, entries[DEFAULT_CROP_SIZE_TAG], order, file_size,
                                        tag_name="DefaultCropSize")
        if crop_size is None:
            return None
        if crop_size[0] <= 0.0 or crop_size[1] <= 0.0:
            raise DNGWarpError("DefaultCropSize values must be positive")
    if (crop_origin is None) != (crop_size is None):
        return None

    if active_area is not None and IMAGE_WIDTH_TAG in dimensions and IMAGE_HEIGHT_TAG in dimensions:
        top, left, bottom, right = active_area
        if bottom > dimensions[IMAGE_HEIGHT_TAG] or right > dimensions[IMAGE_WIDTH_TAG]:
            raise DNGWarpError("DNG ActiveArea extends beyond its image")

    return (dimensions.get(IMAGE_WIDTH_TAG), dimensions.get(IMAGE_HEIGHT_TAG), active_area,
            orientation, scale_pair, crop_origin, crop_size)


def _parse_file(path: Path, fingerprint: tuple[object, ...]) -> DNGWarpRectilinear | None:
    try:
        file = path.open("rb")
    except OSError as exc:
        raise DNGWarpError("Unable to read DNG metadata") from exc
    with file:
        file_size = os.fstat(file.fileno()).st_size
        if file_size < 8:
            return None
        try:
            header = _read_at(file, 0, 8, file_size)
        except DNGGainMapError as exc:
            raise DNGWarpError(str(exc)) from exc
        if header[:2] == b"II":
            order = "<"
        elif header[:2] == b"MM":
            order = ">"
        else:
            return None
        magic = struct.unpack(order + "H", header[2:4])[0]
        if magic != 42:
            return None
        root_offset = struct.unpack(order + "I", header[4:8])[0]
        if root_offset == 0:
            raise DNGWarpError("TIFF has no IFD0")

        pending = [root_offset]
        visited: set[int] = set()
        root_orientation = 1
        records: list[tuple[object, ...]] = []
        while pending:
            if len(visited) >= _MAX_IFDS:
                raise DNGWarpError("TIFF contains too many directories")
            offset = pending.pop(0)
            if offset in visited:
                raise DNGWarpError("TIFF directory offsets contain a cycle")
            visited.add(offset)
            try:
                entries, next_ifd = _read_ifd(file, offset, order, file_size)
            except DNGGainMapError as exc:
                raise DNGWarpError(str(exc)) from exc

            inherited = root_orientation
            metadata = _read_ifd_image_metadata(file, entries, order, file_size, inherited)
            if metadata is None:
                # A present but unfamiliar DefaultScale/Crop encoding makes
                # this directory unsafe for normalized sensor-space warping.
                metadata = (None, None, None, inherited, None, None, None)
            if offset == root_offset:
                root_orientation = metadata[3]

            if OPCODE_LIST3_TAG in entries:
                opcode_entry = entries[OPCODE_LIST3_TAG]
                if opcode_entry.kind not in (1, 7):
                    raise DNGWarpError("OpcodeList3 tag is not a byte field")
                try:
                    opcode_bytes = _entry_bytes(file, opcode_entry, order, file_size,
                                                limit=_MAX_OPCODE_LIST_BYTES)
                except DNGGainMapError as exc:
                    raise DNGWarpError(str(exc)) from exc
                parsed = _parse_opcode_list(opcode_bytes)
                if parsed is not None:
                    warp_data, gain_present, gain_before = parsed
                    records.append((*warp_data, gain_present, gain_before, *metadata))

            if SUB_IFDS_TAG in entries:
                try:
                    offsets = _entry_integers(file, entries[SUB_IFDS_TAG], order, file_size,
                                              accepted_types=(4, 13), max_count=_MAX_SUB_IFDS)
                except DNGGainMapError as exc:
                    raise DNGWarpError("TIFF SubIFDs tag is malformed") from exc
                for child in offsets:
                    if child:
                        pending.append(child)
            if next_ifd:
                pending.append(next_ifd)

        if not records:
            return None
        if len(records) != 1:
            return None
        (planes, coefficients, center_x, center_y, gain_present, gain_before,
         image_width, image_height, active_area, orientation, scale_pair,
         crop_origin, crop_size) = records[0]
        if orientation not in (1, 3, 6, 8):
            return None
        if image_width is None or image_height is None:
            return None
        if active_area is None:
            active_area = (0, 0, int(image_height), int(image_width))
        top, left, bottom, right = active_area
        area_width = right - left
        area_height = bottom - top
        if area_width <= 0 or area_height <= 0:
            raise DNGWarpError("DNG ActiveArea is empty")

        pixel_scale_v = 1.0 if scale_pair is None else scale_pair[1] / scale_pair[0]
        if not math.isfinite(pixel_scale_v) or pixel_scale_v <= 0.0:
            raise DNGWarpError("DefaultScale produces an invalid vertical pixel scale")
        pixel_scale32 = np.float32(pixel_scale_v)
        if not np.isfinite(pixel_scale32) or pixel_scale32 <= 0:
            raise DNGWarpError("DefaultScale exceeds the supported numeric range")

        return DNGWarpRectilinear(
            planes=int(planes), coefficients=coefficients, center_x=center_x,
            center_y=center_y, pixel_scale_v=float(pixel_scale32),
            image_width=int(image_width), image_height=int(image_height),
            active_area=tuple(int(value) for value in active_area),
            orientation=int(orientation), gain_map_before_warp=bool(gain_before),
            gain_map_present=bool(gain_present), default_crop_origin=crop_origin,
            default_crop_size=crop_size, fingerprint=fingerprint,
        )


def parse_dng_warp(source_path: str | os.PathLike[str]) -> DNGWarpRectilinear | None:
    """Read one supported WarpRectilinear opcode from a bounded TIFF/DNG."""
    path = Path(source_path)
    fingerprint = _fingerprint(path)
    with _metadata_cache_lock:
        cached = _metadata_cache.get(fingerprint, _MISSING)
        if cached is not _MISSING:
            _metadata_cache.move_to_end(fingerprint)
            return cached
    parsed = _parse_file(path, fingerprint)  # Failed parses are never cached.
    with _metadata_cache_lock:
        _metadata_cache[fingerprint] = parsed
        _metadata_cache.move_to_end(fingerprint)
        while len(_metadata_cache) > _METADATA_CACHE_LIMIT:
            _metadata_cache.popitem(last=False)
    return parsed


_MISSING = object()


def _frame_scale(image_width: int, image_height: int, frame_width: float,
                 frame_height: float) -> float | None:
    if image_width <= 0 or image_height <= 0 or frame_width <= 0 or frame_height <= 0:
        return None
    scale_x = image_width / frame_width
    scale_y = image_height / frame_height
    scale = (scale_x + scale_y) * 0.5
    if scale <= 0 or abs(scale_x - scale_y) * max(frame_width, frame_height) > 2.0:
        return None
    return scale


def _sensor_orientation_view(image: np.ndarray, warp: DNGWarpRectilinear, *,
                             full_frame: bool = False) -> tuple[np.ndarray, int] | None:
    # EXIF orientations 6/8 swap displayed dimensions; rotate back to the
    # stored sensor frame before applying stage-3 geometry.
    to_sensor = {1: 0, 3: 2, 6: 1, 8: 3}.get(warp.orientation)
    if to_sensor is None:
        return None
    sensor = np.rot90(image, k=to_sensor) if to_sensor else image
    height, width = sensor.shape[:2]
    top, left, bottom, right = warp.active_area
    area_width = right - left
    area_height = bottom - top
    if (height, width) == (area_height, area_width):
        return sensor, to_sensor
    if _frame_scale(width, height, area_width, area_height) is None:
        return None

    # DefaultCrop can describe a smaller view whose aspect ratio is almost
    # indistinguishable from the active area. Only accept a reduced frame when
    # its dimensions identify the full active area unambiguously.
    if not full_frame and warp.default_crop_origin is not None and warp.default_crop_size is not None:
        crop_w, crop_h = warp.default_crop_size
        origin_x, origin_y = warp.default_crop_origin
        cropped = (origin_x > 0.0 or origin_y > 0.0 or
                   not math.isclose(crop_w, area_width, rel_tol=0.0, abs_tol=0.5) or
                   not math.isclose(crop_h, area_height, rel_tol=0.0, abs_tol=0.5))
        if cropped and _frame_scale(width, height, crop_w, crop_h) is not None:
            return None
    return sensor, to_sensor


def _validate_rgb16(image: object, *, label: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array")
    if image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label} must have shape (height, width, 3) and dtype uint16")
    height, width = image.shape[:2]
    if not height or not width or height > 65535 or width > 65535 or image.size > _MAX_VALUES:
        raise ValueError(f"{label} dimensions exceed the supported range")
    return image


def _validate_constants(constants: object) -> np.ndarray:
    values = np.asarray(constants, dtype=np.float32)
    if values.shape != (21,) or not np.isfinite(values).all():
        raise ValueError("WarpRectilinear constants must contain 21 finite float32 values")
    if not 0.0 <= float(values[18]) <= 1.0 or not 0.0 <= float(values[19]) <= 1.0:
        raise ValueError("WarpRectilinear center must be within [0, 1]")
    if not float(values[20]) > 0.0:
        raise ValueError("WarpRectilinear pixel scale must be positive")
    return np.ascontiguousarray(values)


def _warp_constants(warp: DNGWarpRectilinear) -> np.ndarray:
    return _validate_constants(np.concatenate((warp.coefficients.reshape(-1),
                                                np.asarray((warp.center_x, warp.center_y,
                                                            warp.pixel_scale_v), dtype=np.float32))))


def _warp_reference(image_rgb16: np.ndarray, constants: np.ndarray) -> np.ndarray:
    """Strip-wise float bilinear reference with clamped edges and ties-to-even."""
    source = _validate_rgb16(image_rgb16, label="image_rgb16")
    constants = _validate_constants(constants)
    height, width = source.shape[:2]
    f = np.float32
    center_x = f(width) * constants[18]
    center_y = f(height) * constants[19]
    pixel_scale_v = constants[20]
    scaled_height = f(np.floor(f(height) * pixel_scale_v + f(0.5)))
    scaled_center_y = scaled_height * constants[19]
    horizontal = max(center_x, f(width) - center_x)
    vertical = max(scaled_center_y, scaled_height - scaled_center_y)
    radius = f(np.sqrt(horizontal * horizontal + vertical * vertical))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("WarpRectilinear normalization radius is invalid")
    output = np.empty_like(source)
    dx = np.arange(width, dtype=np.float32) - center_x
    nx = dx / radius
    nx2 = nx * nx
    for top in range(0, height, _STRIP_HEIGHT):
        bottom = min(top + _STRIP_HEIGHT, height)
        dy = np.arange(top, bottom, dtype=np.float32)[:, None] - center_y
        ny = (dy * pixel_scale_v) / radius
        ny2 = ny * ny
        r2 = np.minimum(nx2[None, :] + ny2, f(1))
        for plane, coefficients in enumerate(constants[:18].reshape(3, 6)):
            k0, k1, k2, k3, t0, t1 = coefficients
            ratio = k0 + r2 * (k1 + r2 * (k2 + r2 * k3))
            tx = t1 * (r2 + f(2) * nx2[None, :]) + f(2) * t0 * nx[None, :] * ny
            ty = t0 * (r2 + f(2) * ny2) + f(2) * t1 * nx[None, :] * ny
            sx = center_x + dx[None, :] * ratio + radius * tx
            sy = center_y + dy * ratio + radius * ty / pixel_scale_v
            np.clip(sx, 0, width - 1, out=sx)
            np.clip(sy, 0, height - 1, out=sy)
            x0, y0 = np.floor(sx).astype(np.intp), np.floor(sy).astype(np.intp)
            x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
            fx, fy = sx - x0.astype(np.float32), sy - y0.astype(np.float32)
            p00, p10 = source[y0, x0, plane].astype(np.float32), source[y0, x1, plane].astype(np.float32)
            p01, p11 = source[y1, x0, plane].astype(np.float32), source[y1, x1, plane].astype(np.float32)
            upper = p00 * (f(1) - fx) + p10 * fx
            lower = p01 * (f(1) - fx) + p11 * fx
            values = upper * (f(1) - fy) + lower * fy
            output[top:bottom, :, plane] = np.rint(np.clip(values, 0, 65535)).astype(np.uint16)
    return output


def _apply_native_or_reference(image: np.ndarray, constants: np.ndarray, backend: str):
    if backend in ("auto", "gpu", "metal", "d3d12"):
        try:
            from native_renderer import native_warp_rectilinear
            return native_warp_rectilinear(image, constants)
        except Exception as exc:
            from native_renderer import NativeRendererError
            if not isinstance(exc, NativeRendererError):
                raise
    try:
        from native_dense import warp_rectilinear
        return warp_rectilinear(image, constants), "cpp_cpu"
    except Exception as exc:
        from native_renderer import NativeRendererError
        if not isinstance(exc, NativeRendererError):
            raise
    return _warp_reference(image, constants), "python"


def apply_dng_warp_correction(
    image_rgb16: np.ndarray,
    source_path: str | os.PathLike[str],
    *,
    backend: str = "auto",
    preserve_highlights: bool = True,
    highlight_reference: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
    full_frame: bool = False,
) -> tuple[np.ndarray, bool, bool]:
    """Apply stage-3 operations in file order.

    ``full_frame`` identifies previews resized from the decoded ActiveArea,
    before DefaultCrop. Other callers retain conservative crop checks.
    """
    source = _validate_rgb16(image_rgb16, label="image_rgb16")
    if backend not in ("auto", "gpu", "metal", "d3d12", "cpu"):
        raise ValueError("backend must be auto, gpu, metal, d3d12, or cpu")
    if not isinstance(preserve_highlights, bool):
        raise TypeError("preserve_highlights must be a bool")
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise TypeError("diagnostics must be a dict")
    if highlight_reference is not None:
        reference = _validate_rgb16(highlight_reference, label="highlight_reference")
        if reference.shape != source.shape:
            raise ValueError("highlight_reference must match image_rgb16 shape")
    else:
        reference = None

    warp = parse_dng_warp(source_path)
    if warp is None:
        if diagnostics is not None:
            diagnostics.update(backend=None, tca_applied=False,
                               gain_map_before_warp=False, fingerprint=None)
        return image_rgb16, False, False

    oriented_result = _sensor_orientation_view(source, warp, full_frame=full_frame)
    if oriented_result is None:
        if diagnostics is not None:
            diagnostics.update(backend=None, tca_applied=warp.tca_applied,
                               gain_map_before_warp=warp.gain_map_before_warp,
                               fingerprint=warp.fingerprint)
        return image_rgb16, False, False
    sensor_source, to_sensor = oriented_result
    sensor_reference = None
    if reference is not None:
        reference_result = _sensor_orientation_view(reference, warp, full_frame=full_frame)
        if reference_result is None or reference_result[0].shape != sensor_source.shape:
            raise ValueError("highlight_reference does not match the active sensor frame")
        sensor_reference = reference_result[0]

    constants = _warp_constants(warp)
    gain_map = None
    if warp.gain_map_present:
        from dng_gainmap import parse_dng_gain_map, _is_supported_full_frame
        gain_map = parse_dng_gain_map(source_path)
        # Arrays have already been rotated back to sensor orientation.
        if gain_map is not None:
            gain_map = replace(gain_map, orientation=1)
        if gain_map is None or not _is_supported_full_frame(gain_map, *sensor_source.shape[:2]):
            return image_rgb16, False, False
    actual_backend = "python"
    gain_applied = False
    try:
        from dng_gainmap import _apply_gain_map

        if warp.gain_map_present and warp.gain_map_before_warp:
            gained = _apply_gain_map(
                sensor_source, gain_map, preserve_highlights=preserve_highlights,
                highlight_reference=sensor_reference,
            )
            gain_applied = True
            warped, actual_backend = _apply_native_or_reference(gained, constants, backend)
        else:
            warped, actual_backend = _apply_native_or_reference(sensor_source, constants, backend)
            if warp.gain_map_present:
                warped_reference = None
                if sensor_reference is not None:
                    warped_reference, _ = _apply_native_or_reference(sensor_reference, constants, backend)
                warped = _apply_gain_map(
                    warped, gain_map, preserve_highlights=preserve_highlights,
                    highlight_reference=warped_reference,
                )
                gain_applied = True
    except DNGGainMapError as exc:
        raise DNGWarpError(str(exc)) from exc

    if to_sensor:
        warped = np.rot90(warped, k=-to_sensor)
    if diagnostics is not None:
        diagnostics.update(backend=actual_backend, tca_applied=warp.tca_applied,
                           gain_map_before_warp=warp.gain_map_before_warp,
                           fingerprint=warp.fingerprint)
    return np.ascontiguousarray(warped), True, bool(gain_applied)
