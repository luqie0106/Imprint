"""Safely read and apply the DNG OpcodeList3 GainMap opcode.

This module intentionally supports only a full active-area, three-channel
linear RGB gain map. Other opcodes, including WarpRectilinear, are left to the
caller or ignored here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import struct
from typing import BinaryIO

import cv2
import numpy as np


# DNG tag 51022 (0xC74E) is OpcodeList3. Tag 51041 is NoiseProfile.
OPCODE_LIST3_TAG = 51022
SUB_IFDS_TAG = 330
ACTIVE_AREA_TAG = 50829
IMAGE_WIDTH_TAG = 256
IMAGE_HEIGHT_TAG = 257
ORIENTATION_TAG = 274
GAIN_MAP_OPCODE_ID = 9

_MAX_IFDS = 32
_MAX_IFD_ENTRIES = 4096
_MAX_SUB_IFDS = 32
_MAX_OPCODE_COUNT = 64
_MAX_OPCODE_LIST_BYTES = 16 * 1024 * 1024
_MAX_GAIN_MAP_POINTS = 512
_MAX_GAIN_MAP_VALUES = 1_000_000
_STRIP_HEIGHT = 128


class DNGGainMapError(ValueError):
    """Raised when TIFF or GainMap bytes are malformed or truncated."""


@dataclass(frozen=True)
class DNGGainMap:
    """Parsed GainMap opcode values, in vertical/horizontal/plane order."""

    top: int
    left: int
    bottom: int
    right: int
    plane: int
    planes: int
    row_pitch: int
    col_pitch: int
    map_points_v: int
    map_points_h: int
    map_spacing_v: float
    map_spacing_h: float
    map_origin_v: float
    map_origin_h: float
    map_planes: int
    gains: np.ndarray
    active_area: tuple[int, int, int, int] | None
    image_width: int | None
    image_height: int | None
    orientation: int


@dataclass(frozen=True)
class _Entry:
    kind: int
    count: int
    value_field: bytes


_TIFF_TYPE_SIZE = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    6: 1,  # SBYTE
    7: 1,  # UNDEFINED
    8: 2,  # SSHORT
    9: 4,  # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
    13: 4,  # IFD (Classic TIFF)
}


def _read_at(file: BinaryIO, offset: int, size: int, file_size: int, *, limit: int | None = None) -> bytes:
    if offset < 0 or size < 0 or (limit is not None and size > limit):
        raise DNGGainMapError("TIFF offset or field size is invalid")
    if offset > file_size or size > file_size - offset:
        raise DNGGainMapError("TIFF field extends beyond the file")
    file.seek(offset)
    data = file.read(size)
    if len(data) != size:
        raise DNGGainMapError("TIFF field is truncated")
    return data


def _entry_bytes(file: BinaryIO, entry: _Entry, order: str, file_size: int, *, limit: int) -> bytes:
    type_size = _TIFF_TYPE_SIZE.get(entry.kind)
    if type_size is None:
        raise DNGGainMapError("TIFF tag uses an unsupported type")
    size = entry.count * type_size
    if size > limit:
        raise DNGGainMapError("TIFF tag exceeds the supported size limit")
    if size <= 4:
        return entry.value_field[:size]
    value_offset = struct.unpack(order + "I", entry.value_field)[0]
    return _read_at(file, value_offset, size, file_size, limit=limit)


def _entry_integers(
    file: BinaryIO,
    entry: _Entry,
    order: str,
    file_size: int,
    *,
    accepted_types: tuple[int, ...],
    max_count: int,
) -> tuple[int, ...]:
    if entry.kind not in accepted_types or entry.count < 1 or entry.count > max_count:
        raise DNGGainMapError("TIFF integer tag has an unsupported type or count")
    raw = _entry_bytes(file, entry, order, file_size, limit=max_count * 4)
    code = {3: "H", 4: "I", 13: "I"}.get(entry.kind)
    if code is None:
        raise DNGGainMapError("TIFF integer tag type is unsupported")
    return struct.unpack(order + code * entry.count, raw)


def _read_ifd(
    file: BinaryIO,
    offset: int,
    order: str,
    file_size: int,
) -> tuple[dict[int, _Entry], int]:
    count = struct.unpack(order + "H", _read_at(file, offset, 2, file_size))[0]
    if count > _MAX_IFD_ENTRIES:
        raise DNGGainMapError("TIFF IFD contains too many entries")
    entries_size = count * 12
    raw = _read_at(file, offset + 2, entries_size + 4, file_size, limit=_MAX_IFD_ENTRIES * 12 + 4)
    entries: dict[int, _Entry] = {}
    for index in range(count):
        tag, kind, value_count = struct.unpack_from(order + "HHI", raw, index * 12)
        field = raw[index * 12 + 8:index * 12 + 12]
        if tag in entries:
            raise DNGGainMapError("TIFF IFD contains a duplicate tag")
        entries[tag] = _Entry(kind, value_count, field)
    next_ifd = struct.unpack_from(order + "I", raw, entries_size)[0]
    return entries, next_ifd


def _parse_gainmap_payload(payload: bytes) -> DNGGainMap:
    fixed_size = 10 * 4 + 4 * 8 + 4
    if len(payload) < fixed_size:
        raise DNGGainMapError("GainMap opcode is shorter than its fixed parameters")
    fields = struct.unpack_from(">10I4dI", payload, 0)
    (
        top,
        left,
        bottom,
        right,
        plane,
        planes,
        row_pitch,
        col_pitch,
        points_v,
        points_h,
        spacing_v,
        spacing_h,
        origin_v,
        origin_h,
        map_planes,
    ) = fields
    if not points_v or not points_h or not map_planes:
        raise DNGGainMapError("GainMap dimensions must be non-zero")
    if points_v > _MAX_GAIN_MAP_POINTS or points_h > _MAX_GAIN_MAP_POINTS:
        raise DNGGainMapError("GainMap dimensions exceed the supported limit")
    value_count = points_v * points_h * map_planes
    if value_count > _MAX_GAIN_MAP_VALUES:
        raise DNGGainMapError("GainMap contains too many values")
    expected_size = fixed_size + value_count * 4
    if len(payload) != expected_size:
        raise DNGGainMapError("GainMap payload size does not match its dimensions")
    if not all(math.isfinite(value) for value in (spacing_v, spacing_h, origin_v, origin_h)):
        raise DNGGainMapError("GainMap coordinate parameters are not finite")
    if spacing_v <= 0.0 or spacing_h <= 0.0 or bottom <= top or right <= left:
        raise DNGGainMapError("GainMap bounds or spacing are invalid")

    gains = np.frombuffer(payload, dtype=">f4", count=value_count, offset=fixed_size)
    if not np.isfinite(gains).all() or (gains < 0).any():
        raise DNGGainMapError("GainMap contains an invalid gain value")
    gains = gains.astype(np.float32, copy=True).reshape(points_v, points_h, map_planes)
    return DNGGainMap(
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        plane=plane,
        planes=planes,
        row_pitch=row_pitch,
        col_pitch=col_pitch,
        map_points_v=points_v,
        map_points_h=points_h,
        map_spacing_v=spacing_v,
        map_spacing_h=spacing_h,
        map_origin_v=origin_v,
        map_origin_h=origin_h,
        map_planes=map_planes,
        gains=gains,
        active_area=None,
        image_width=None,
        image_height=None,
        orientation=1,
    )


def _parse_opcode_list(payload: bytes) -> list[DNGGainMap]:
    if len(payload) < 4:
        raise DNGGainMapError("OpcodeList3 is missing its opcode count")
    opcode_count = struct.unpack_from(">I", payload, 0)[0]
    if opcode_count > _MAX_OPCODE_COUNT:
        raise DNGGainMapError("OpcodeList3 contains too many opcodes")
    position = 4
    maps: list[DNGGainMap] = []
    for _ in range(opcode_count):
        if len(payload) - position < 16:
            raise DNGGainMapError("OpcodeList3 contains a truncated opcode header")
        opcode_id, _version, _flags, data_size = struct.unpack_from(">4I", payload, position)
        position += 16
        if data_size > len(payload) - position:
            raise DNGGainMapError("Opcode data extends beyond OpcodeList3")
        data = payload[position:position + data_size]
        position += data_size
        if opcode_id == GAIN_MAP_OPCODE_ID:
            maps.append(_parse_gainmap_payload(data))
    if position != len(payload):
        raise DNGGainMapError("OpcodeList3 has trailing bytes")
    return maps


def parse_dng_gain_map(source_path: str | os.PathLike[str]) -> DNGGainMap | None:
    """Parse the single GainMap found in a Classic TIFF/DNG OpcodeList3.

    Returns ``None`` for non-TIFF files, TIFFs without OpcodeList3, and lists
    without a GainMap. Malformed TIFF or opcode data raises
    :class:`DNGGainMapError`. BigTIFF is deliberately unsupported.
    """

    path = Path(source_path)
    try:
        file = path.open("rb")
    except OSError as exc:
        raise DNGGainMapError("Unable to read DNG metadata") from exc

    with file:
        file_size = os.fstat(file.fileno()).st_size
        if file_size < 8:
            return None
        header = _read_at(file, 0, 8, file_size)
        if header[:2] == b"II":
            order = "<"
        elif header[:2] == b"MM":
            order = ">"
        else:
            return None
        magic = struct.unpack(order + "H", header[2:4])[0]
        if magic != 42:  # BigTIFF magic 43 is intentionally not supported.
            return None
        root_offset = struct.unpack(order + "I", header[4:8])[0]
        if root_offset == 0:
            raise DNGGainMapError("TIFF has no IFD0")

        pending = [root_offset]
        visited: set[int] = set()
        found_maps: list[DNGGainMap] = []
        while pending:
            if len(visited) >= _MAX_IFDS:
                raise DNGGainMapError("TIFF contains too many directories")
            offset = pending.pop(0)
            if offset in visited:
                raise DNGGainMapError("TIFF directory offsets contain a cycle")
            visited.add(offset)
            entries, _next_ifd = _read_ifd(file, offset, order, file_size)

            active_area: tuple[int, int, int, int] | None = None
            if ACTIVE_AREA_TAG in entries:
                values = _entry_integers(
                    file,
                    entries[ACTIVE_AREA_TAG],
                    order,
                    file_size,
                    accepted_types=(4,),
                    max_count=4,
                )
                if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
                    raise DNGGainMapError("DNG ActiveArea tag is invalid")
                active_area = values  # type: ignore[assignment]

            image_width = image_height = None
            for tag, name in ((IMAGE_WIDTH_TAG, "width"), (IMAGE_HEIGHT_TAG, "height")):
                if tag in entries:
                    values = _entry_integers(
                        file,
                        entries[tag],
                        order,
                        file_size,
                        accepted_types=(3, 4),
                        max_count=1,
                    )
                    if len(values) != 1 or values[0] <= 0:
                        raise DNGGainMapError(f"TIFF image {name} is invalid")
                    if name == "width":
                        image_width = values[0]
                    else:
                        image_height = values[0]

            orientation = 1
            if ORIENTATION_TAG in entries:
                values = _entry_integers(
                    file,
                    entries[ORIENTATION_TAG],
                    order,
                    file_size,
                    accepted_types=(3,),
                    max_count=1,
                )
                orientation = values[0]

            if OPCODE_LIST3_TAG in entries:
                opcode_entry = entries[OPCODE_LIST3_TAG]
                if opcode_entry.kind not in (1, 7):
                    raise DNGGainMapError("OpcodeList3 tag is not a byte field")
                opcode_bytes = _entry_bytes(
                    file,
                    opcode_entry,
                    order,
                    file_size,
                    limit=_MAX_OPCODE_LIST_BYTES,
                )
                parsed_maps = _parse_opcode_list(opcode_bytes)
                for gain_map in parsed_maps:
                    found_maps.append(
                        DNGGainMap(
                            **{
                                **gain_map.__dict__,
                                "active_area": active_area,
                                "image_width": image_width,
                                "image_height": image_height,
                                "orientation": orientation,
                            }
                        )
                    )

            if SUB_IFDS_TAG in entries:
                subifd_entry = entries[SUB_IFDS_TAG]
                offsets = _entry_integers(
                    file,
                    subifd_entry,
                    order,
                    file_size,
                    accepted_types=(4, 13),
                    max_count=_MAX_SUB_IFDS,
                )
                for subifd in offsets:
                    if subifd:
                        pending.append(subifd)
        if not found_maps:
            return None
        if len(found_maps) != 1:
            raise DNGGainMapError("Multiple GainMap opcodes are not supported")
        return found_maps[0]


def _is_supported_full_frame(gain_map: DNGGainMap, image_height: int, image_width: int) -> bool:
    if (
        gain_map.plane != 0
        or gain_map.planes != 3
        or gain_map.map_planes != 3
        or gain_map.row_pitch != 1
        or gain_map.col_pitch != 1
        or gain_map.orientation != 1
        or gain_map.top != 0
        or gain_map.left != 0
    ):
        return False
    area_height = gain_map.bottom - gain_map.top
    area_width = gain_map.right - gain_map.left
    if gain_map.active_area is not None and gain_map.active_area != (
        gain_map.top,
        gain_map.left,
        gain_map.bottom,
        gain_map.right,
    ):
        return False
    if gain_map.active_area is None:
        if gain_map.image_width is None or gain_map.image_height is None:
            return False
        if gain_map.bottom != gain_map.image_height or gain_map.right != gain_map.image_width:
            return False
    if area_height <= 0 or area_width <= 0:
        return False
    expected_ratio = area_width / area_height
    actual_ratio = image_width / image_height
    # Reduced previews can be rounded by a pixel while preserving the frame.
    return math.isclose(actual_ratio, expected_ratio, rel_tol=0.005, abs_tol=0.00001)


def _apply_gain_map(image_rgb16: np.ndarray, gain_map: DNGGainMap) -> np.ndarray:
    height, width = image_rgb16.shape[:2]
    output = image_rgb16.copy()
    x_relative = (np.arange(width, dtype=np.float64) + 0.5) / width
    x_map = (x_relative - gain_map.map_origin_h) / gain_map.map_spacing_h
    x_map = np.clip(x_map, 0.0, gain_map.map_points_h - 1).astype(np.float32)

    for top in range(0, height, _STRIP_HEIGHT):
        bottom = min(top + _STRIP_HEIGHT, height)
        y_relative = (np.arange(top, bottom, dtype=np.float64) + 0.5) / height
        y_map = (y_relative - gain_map.map_origin_v) / gain_map.map_spacing_v
        y_map = np.clip(y_map, 0.0, gain_map.map_points_v - 1).astype(np.float32)
        map_x = np.broadcast_to(x_map[None, :], (bottom - top, width)).copy()
        map_y = np.broadcast_to(y_map[:, None], (bottom - top, width)).copy()
        gain_strip = cv2.remap(
            gain_map.gains,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        pixels = image_rgb16[top:bottom].astype(np.float32)
        np.multiply(pixels, gain_strip, out=pixels)
        np.clip(pixels, 0.0, 65535.0, out=pixels)
        np.rint(pixels, out=pixels)
        output[top:bottom] = pixels.astype(np.uint16)
    return output


def apply_dng_gain_map(
    image_rgb16: np.ndarray,
    source_path: str | os.PathLike[str],
) -> tuple[np.ndarray, bool]:
    """Apply a compatible OpcodeList3 GainMap to linear uint16 RGB pixels.

    The source array is never modified. Reduced preview images are supported
    when they preserve the full active-area aspect ratio. Non-DNG files and
    incompatible opcodes return the original array with ``False``; malformed
    TIFF/opcode data raises :class:`DNGGainMapError`.
    """

    if not isinstance(image_rgb16, np.ndarray):
        raise TypeError("image_rgb16 must be a NumPy array")
    if image_rgb16.dtype != np.uint16 or image_rgb16.ndim != 3 or image_rgb16.shape[2] != 3:
        raise ValueError("image_rgb16 must have shape (height, width, 3) and dtype uint16")
    height, width = image_rgb16.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("image_rgb16 must not be empty")

    gain_map = parse_dng_gain_map(source_path)
    if gain_map is None or not _is_supported_full_frame(gain_map, height, width):
        return image_rgb16, False
    return _apply_gain_map(image_rgb16, gain_map), True
