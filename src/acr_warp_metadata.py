"""Read prepared WarpRectilinear metadata without reducing payload precision.

This experiment-facing reader keeps the opcode's original per-plane doubles
and the metadata from the same TIFF IFD. It does not infer runtime geometry or
bind an aspect ratio for a native warp implementation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np

from dng_gainmap import (
    DNGGainMapError,
    _Entry,
    OPCODE_LIST3_TAG,
    SUB_IFDS_TAG,
    _entry_bytes,
    _entry_integers,
    _parse_gainmap_payload,
    _read_at,
    _read_ifd,
)
from dng_warp import DNGWarpError, _read_ifd_image_metadata


WARP_RECTILINEAR_OPCODE_ID = 1
GAIN_MAP_OPCODE_ID = 9
SUPPORTED_OPCODE_VERSION = 0x01030000
OPTIONAL_OPCODE_FLAG = 1
MAX_IFDS = 32
MAX_SUB_IFDS = 32
MAX_OPCODE_COUNT = 64
MAX_OPCODE_LIST_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class PreparedWarpMetadata:
    """One WarpRectilinear and its source sensor metadata from a single IFD."""

    planes: int
    coefficients: np.ndarray
    center_x: float
    center_y: float
    default_scale_h: float
    default_scale_v: float
    active_area: tuple[int, int, int, int]
    image_width: int
    image_height: int
    orientation: int
    gain_map_present: bool
    gain_map_before_warp: bool
    fingerprint: tuple[object, ...]
    ifd_offset: int
    opcode_ids: tuple[int, ...]


@dataclass(frozen=True)
class _WarpPayload:
    planes: int
    coefficients: np.ndarray
    center_x: float
    center_y: float


@dataclass(frozen=True)
class _OpcodeList:
    opcode_ids: tuple[int, ...]
    has_warp: bool
    warp: _WarpPayload | None
    gain_map_present: bool
    gain_map_before_warp: bool
    supported: bool


def _parse_warp_payload(payload: bytes) -> _WarpPayload:
    if len(payload) < 4:
        raise DNGWarpError("WarpRectilinear payload is truncated")
    planes = struct.unpack_from(">I", payload, 0)[0]
    if not 1 <= planes <= 4:
        raise DNGWarpError("WarpRectilinear plane count must be between one and four")
    expected_size = 20 + 48 * planes
    if len(payload) != expected_size:
        raise DNGWarpError("WarpRectilinear payload size does not match its plane count")

    values = struct.unpack_from(">" + "d" * (planes * 6 + 2), payload, 4)
    if not all(math.isfinite(value) for value in values):
        raise DNGWarpError("WarpRectilinear parameters must be finite")
    center_x, center_y = values[-2:]
    if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
        raise DNGWarpError("WarpRectilinear center must be within [0, 1]")

    # Reading from the big-endian wire representation and converting directly
    # to native float64 preserves the original IEEE-754 double values.
    coefficients = np.frombuffer(
        payload, dtype=">f8", count=planes * 6, offset=4
    ).astype(np.float64, copy=True).reshape(planes, 6)
    coefficients.setflags(write=False)
    return _WarpPayload(planes, coefficients, float(center_x), float(center_y))


def _parse_opcode_list(payload: bytes) -> _OpcodeList:
    if len(payload) < 4:
        raise DNGWarpError("OpcodeList3 is missing its opcode count")
    opcode_count = struct.unpack_from(">I", payload, 0)[0]
    if opcode_count > MAX_OPCODE_COUNT:
        raise DNGWarpError("OpcodeList3 contains too many opcodes")

    position = 4
    opcode_ids: list[int] = []
    warps: list[_WarpPayload] = []
    gain_positions: list[int] = []
    supported = True
    for opcode_index in range(opcode_count):
        if len(payload) - position < 16:
            raise DNGWarpError("OpcodeList3 contains a truncated opcode header")
        opcode_id, version, flags, data_size = struct.unpack_from(">4I", payload, position)
        position += 16
        if data_size > len(payload) - position:
            raise DNGWarpError("Opcode data extends beyond OpcodeList3")
        data = payload[position:position + data_size]
        position += data_size
        opcode_ids.append(opcode_id)

        if opcode_id == WARP_RECTILINEAR_OPCODE_ID:
            if version != SUPPORTED_OPCODE_VERSION or flags not in (0, OPTIONAL_OPCODE_FLAG):
                supported = False
                continue
            warps.append(_parse_warp_payload(data))
        elif opcode_id == GAIN_MAP_OPCODE_ID:
            if version != SUPPORTED_OPCODE_VERSION or flags not in (0, OPTIONAL_OPCODE_FLAG):
                supported = False
                continue
            try:
                _parse_gainmap_payload(data)
            except DNGGainMapError as exc:
                raise DNGWarpError(str(exc)) from exc
            gain_positions.append(opcode_index)
        elif not (flags & OPTIONAL_OPCODE_FLAG):
            supported = False

    if position != len(payload):
        raise DNGWarpError("OpcodeList3 has trailing bytes")

    has_warp = WARP_RECTILINEAR_OPCODE_ID in opcode_ids
    if len(warps) > 1 or len(gain_positions) > 1:
        supported = False
    warp_position = next(
        (index for index, opcode_id in enumerate(opcode_ids)
         if opcode_id == WARP_RECTILINEAR_OPCODE_ID),
        None,
    )
    gain_before = bool(gain_positions and warp_position is not None
                       and gain_positions[0] < warp_position)
    return _OpcodeList(
        tuple(opcode_ids), has_warp, warps[0] if len(warps) == 1 else None,
        bool(gain_positions), gain_before, supported,
    )


def _read_opcode_list(file: BinaryIO, entry: _Entry, order: str, file_size: int) -> bytes:
    if entry.kind not in (1, 7):
        raise DNGWarpError("OpcodeList3 tag is not a byte field")
    try:
        return _entry_bytes(file, entry, order, file_size, limit=MAX_OPCODE_LIST_BYTES)
    except DNGGainMapError as exc:
        raise DNGWarpError(str(exc)) from exc


def _ifd_graph_has_cycle(edges: dict[int, tuple[int, ...]]) -> bool:
    state: dict[int, int] = {}

    def visit(offset: int) -> bool:
        current = state.get(offset, 0)
        if current == 1:
            return True
        if current == 2:
            return False
        state[offset] = 1
        for child in edges[offset]:
            if visit(child):
                return True
        state[offset] = 2
        return False

    return any(visit(offset) for offset in edges if state.get(offset, 0) == 0)


def _read_directories(
    file: BinaryIO, root_offset: int, order: str, file_size: int
) -> tuple[list[int], dict[int, dict[int, _Entry]], dict[int, tuple[int, ...]]]:
    offsets: list[int] = []
    entries_by_offset: dict[int, dict[int, _Entry]] = {}
    edges: dict[int, tuple[int, ...]] = {}
    pending = deque([root_offset])
    while pending:
        offset = pending.popleft()
        if offset in entries_by_offset:
            continue
        if len(entries_by_offset) >= MAX_IFDS:
            raise DNGWarpError("TIFF contains too many directories")
        try:
            entries, next_ifd = _read_ifd(file, offset, order, file_size)
        except DNGGainMapError as exc:
            raise DNGWarpError(str(exc)) from exc

        children: list[int] = []
        if SUB_IFDS_TAG in entries:
            try:
                child_offsets = _entry_integers(
                    file, entries[SUB_IFDS_TAG], order, file_size,
                    accepted_types=(4, 13), max_count=MAX_SUB_IFDS,
                )
            except DNGGainMapError as exc:
                raise DNGWarpError("TIFF SubIFDs tag is malformed") from exc
            children.extend(int(child) for child in child_offsets if child)
        if next_ifd:
            children.append(int(next_ifd))

        offsets.append(offset)
        entries_by_offset[offset] = entries
        edges[offset] = tuple(children)
        pending.extend(child for child in children if child not in entries_by_offset)

    if _ifd_graph_has_cycle(edges):
        raise DNGWarpError("TIFF directory offsets contain a cycle")
    return offsets, entries_by_offset, edges


def _fingerprint(path: Path, file: BinaryIO) -> tuple[object, ...]:
    try:
        stat = os.fstat(file.fileno())
        resolved = str(path.resolve(strict=True))
    except OSError as exc:
        raise DNGWarpError("Unable to read DNG metadata") from exc
    return (resolved, stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _parse_file(path: Path, file: BinaryIO) -> PreparedWarpMetadata | None:
    try:
        file_size = os.fstat(file.fileno()).st_size
    except OSError as exc:
        raise DNGWarpError("Unable to read DNG metadata") from exc
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
    if struct.unpack(order + "H", header[2:4])[0] != 42:
        return None
    root_offset = struct.unpack(order + "I", header[4:8])[0]
    if root_offset == 0:
        raise DNGWarpError("TIFF has no IFD0")

    offsets, entries_by_offset, _edges = _read_directories(
        file, root_offset, order, file_size
    )
    fingerprint = _fingerprint(path, file)

    root_metadata = _read_ifd_image_metadata(
        file, entries_by_offset[root_offset], order, file_size, 1
    )
    root_orientation = 1 if root_metadata is None else root_metadata[3]

    candidates: list[PreparedWarpMetadata] = []
    for offset in offsets:
        entries = entries_by_offset[offset]
        try:
            metadata = _read_ifd_image_metadata(
                file, entries, order, file_size, root_orientation
            )
        except DNGWarpError:
            raise
        if OPCODE_LIST3_TAG not in entries:
            continue

        opcode_bytes = _read_opcode_list(
            file, entries[OPCODE_LIST3_TAG], order, file_size
        )
        parsed = _parse_opcode_list(opcode_bytes)
        if not parsed.has_warp:
            continue
        if not parsed.supported or parsed.warp is None:
            return None
        if metadata is None:
            return None

        image_width, image_height, active_area, orientation, scale_pair, _crop_origin, _crop_size = metadata
        if image_width is None or image_height is None:
            return None
        if scale_pair is None:
            scale_h = scale_v = 1.0
        else:
            scale_h, scale_v = scale_pair
            if scale_h <= 0.0 or scale_v <= 0.0:
                raise DNGWarpError("DefaultScale values must be positive")
        if active_area is None:
            active_area = (0, 0, int(image_height), int(image_width))
        top, left, bottom, right = active_area
        if bottom > image_height or right > image_width:
            raise DNGWarpError("DNG ActiveArea extends beyond its image")
        if bottom <= top or right <= left:
            raise DNGWarpError("DNG ActiveArea is empty")

        warp = parsed.warp
        candidates.append(PreparedWarpMetadata(
            planes=warp.planes,
            coefficients=warp.coefficients,
            center_x=warp.center_x,
            center_y=warp.center_y,
            default_scale_h=float(scale_h),
            default_scale_v=float(scale_v),
            active_area=tuple(int(value) for value in active_area),
            image_width=int(image_width),
            image_height=int(image_height),
            orientation=int(orientation),
            gain_map_present=parsed.gain_map_present,
            gain_map_before_warp=parsed.gain_map_before_warp,
            fingerprint=fingerprint,
            ifd_offset=offset,
            opcode_ids=parsed.opcode_ids,
        ))

    if len(candidates) != 1:
        return None
    return candidates[0]


def parse_acr_warp_prepared(path: str | os.PathLike[str]) -> PreparedWarpMetadata | None:
    """Read one supported prepared WarpRectilinear from a bounded ClassicTIFF.

    Non-TIFF files and unsupported opcode pipelines return ``None``. Malformed
    TIFF structures or payloads raise :class:`DNGWarpError`.
    """
    source = Path(path)
    try:
        file = source.open("rb")
    except OSError as exc:
        raise DNGWarpError("Unable to read DNG metadata") from exc
    with file:
        return _parse_file(source, file)
