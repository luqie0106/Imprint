"""Minimal standards-based writer for 16-bit Linear/Demosaiced DNG files.

The output is a TIFF/EP container with DNG tags, LinearRaw photometric data,
lossless Adobe Deflate strips, and a reduced JPEG preview IFD. It never writes
directly to the final filename.

This product includes DNG technology under license by Adobe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import io
import os
from pathlib import Path
import struct
import tempfile
import threading
from typing import Any, Iterable
import zlib

import numpy as np
from PIL import Image


BYTE, ASCII, SHORT, LONG, RATIONAL, UNDEFINED, SLONG, SRATIONAL = 1, 2, 3, 4, 5, 7, 9, 10
TYPE_SIZES = {BYTE: 1, ASCII: 1, SHORT: 2, LONG: 4, RATIONAL: 8, UNDEFINED: 1, SLONG: 4, SRATIONAL: 8}
_OUTPUT_LOCK = threading.Lock()


@dataclass
class _Tag:
    code: int
    kind: int
    count: int
    data: bytes


def _ascii(value: Any) -> bytes:
    return str(value).encode("utf-8", "replace") + b"\0"


def _shorts(values: Iterable[int]) -> bytes:
    values = tuple(values)
    return struct.pack("<" + "H" * len(values), *values)


def _longs(values: Iterable[int]) -> bytes:
    values = tuple(values)
    return struct.pack("<" + "I" * len(values), *values)


def _rationals(values: Iterable[tuple[int, int]], signed: bool = False) -> bytes:
    values = tuple(values)
    fmt = "<" + ("ii" if signed else "II") * len(values)
    flat = [part for value in values for part in value]
    return struct.pack(fmt, *flat)


def _tag(code: int, kind: int, data: bytes, count: int | None = None) -> _Tag:
    return _Tag(code, kind, count if count is not None else len(data) // TYPE_SIZES[kind], data)


def _build_ifd(tags: list[_Tag], offset: int, next_ifd: int = 0) -> tuple[bytes, dict[int, int]]:
    tags = sorted(tags, key=lambda item: item.code)
    table_size = 2 + len(tags) * 12 + 4
    extra_offset = offset + table_size
    entries = bytearray(struct.pack("<H", len(tags)))
    extras = bytearray()
    data_offsets: dict[int, int] = {}
    for item in tags:
        entries += struct.pack("<HHI", item.code, item.kind, item.count)
        if len(item.data) <= 4:
            entries += item.data.ljust(4, b"\0")
        else:
            if extra_offset % 2:
                extras += b"\0"
                extra_offset += 1
            entries += struct.pack("<I", extra_offset)
            data_offsets[item.code] = extra_offset
            extras += item.data
            extra_offset += len(item.data)
    entries += struct.pack("<I", next_ifd)
    return bytes(entries + extras), data_offsets


def _predictor_bytes(strip: np.ndarray) -> bytes:
    predicted = strip.copy()
    predicted[:, 1:, :] = (predicted[:, 1:, :] - predicted[:, :-1, :]).astype(np.uint16)
    return predicted.astype("<u2", copy=False).tobytes(order="C")


def _jpeg_preview(image: np.ndarray, max_edge: int = 1024) -> tuple[bytes, int, int]:
    preview = Image.fromarray((image >> 8).astype(np.uint8), mode="RGB")
    preview.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    preview.save(stream, "JPEG", quality=88, optimize=True)
    return stream.getvalue(), preview.width, preview.height


def _unique_output_path(output_dir: Path, source: Path) -> Path:
    candidate = output_dir / f"{source.stem}_dehaze.dng"
    index = 2
    while candidate.exists():
        candidate = output_dir / f"{source.stem}_dehaze_{index}.dng"
        index += 1
    return candidate


def write_linear_dng(
    image_rgb16: np.ndarray,
    source_path: str | Path,
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically create a lossless 16-bit RGB Linear DNG and return its path."""
    if image_rgb16.dtype != np.uint16 or image_rgb16.ndim != 3 or image_rgb16.shape[2] != 3:
        raise ValueError("Linear DNG input must be a uint16 RGB array")
    source = Path(source_path)
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = _unique_output_path(destination_dir, source)
    height, width = image_rgb16.shape[:2]
    rows_per_strip = max(1, min(height, max(1, 1024 * 1024 // max(1, width * 6))))
    strips = [
        zlib.compress(_predictor_bytes(image_rgb16[row:row + rows_per_strip]), level=6)
        for row in range(0, height, rows_per_strip)
    ]
    preview, preview_width, preview_height = _jpeg_preview(image_rgb16)
    info = metadata or {}
    make = info.get("Make", "Imprint")
    model = info.get("Model", "RGB Linear DNG")
    captured = info.get("DateTimeOriginal", info.get("DateTime", datetime.now().strftime("%Y:%m:%d %H:%M:%S")))

    # Layout: TIFF header, primary IFD, full-res strips, preview IFD, JPEG preview.
    primary_tags = [
        _tag(254, LONG, _longs([0])), _tag(256, LONG, _longs([width])),
        _tag(257, LONG, _longs([height])), _tag(258, SHORT, _shorts([16, 16, 16])),
        _tag(259, SHORT, _shorts([8])), _tag(262, SHORT, _shorts([34892])),
        _tag(271, ASCII, _ascii(make)), _tag(272, ASCII, _ascii(model)),
        _tag(273, LONG, _longs([0] * len(strips))), _tag(274, SHORT, _shorts([1])),
        _tag(277, SHORT, _shorts([3])), _tag(278, LONG, _longs([rows_per_strip])),
        _tag(279, LONG, _longs([len(item) for item in strips])),
        _tag(284, SHORT, _shorts([1])), _tag(305, ASCII, _ascii("Imprint Natural Dehaze")),
        _tag(306, ASCII, _ascii(captured)), _tag(317, SHORT, _shorts([2])),
        _tag(339, SHORT, _shorts([1, 1, 1])),
        _tag(50706, BYTE, bytes([1, 4, 0, 0]), 4), _tag(50707, BYTE, bytes([1, 4, 0, 0]), 4),
        _tag(50708, ASCII, _ascii(f"{make} {model}")), _tag(50717, LONG, _longs([65535, 65535, 65535])),
        _tag(50719, LONG, _longs([0, 0])), _tag(50720, LONG, _longs([width, height])),
        # XYZ D50 to linear-sRGB. The pixels are exported in this declared working
        # space, rather than pretending that processed RGB remains camera-native.
        _tag(50721, SRATIONAL, _rationals([(3133856, 1000000), (-1616867, 1000000), (-490615, 1000000), (-978768, 1000000), (1916142, 1000000), (33454, 1000000), (71945, 1000000), (-228991, 1000000), (1405243, 1000000)], True)),
        _tag(50728, RATIONAL, _rationals([(1, 1), (1, 1), (1, 1)])),
        _tag(50730, SRATIONAL, _rationals([(0, 10000)], True)),
        _tag(50734, RATIONAL, _rationals([(1, 1)])), _tag(50778, SHORT, _shorts([21])),
        _tag(50780, RATIONAL, _rationals([(1, 1)])), _tag(50829, LONG, _longs([0, 0, height, width])),
    ]
    primary_placeholder, _ = _build_ifd(primary_tags, 8)
    strip_start = 8 + len(primary_placeholder)
    strip_offsets = []
    cursor = strip_start
    for item in strips:
        strip_offsets.append(cursor)
        cursor += len(item)

    preview_ifd_offset = cursor
    preview_tags = [
        _tag(254, LONG, _longs([1])), _tag(256, LONG, _longs([preview_width])),
        _tag(257, LONG, _longs([preview_height])), _tag(258, SHORT, _shorts([8, 8, 8])),
        _tag(259, SHORT, _shorts([7])), _tag(262, SHORT, _shorts([6])),
        _tag(273, LONG, _longs([0])), _tag(277, SHORT, _shorts([3])),
        _tag(278, LONG, _longs([preview_height])), _tag(279, LONG, _longs([len(preview)])),
        _tag(284, SHORT, _shorts([1])),
    ]
    preview_placeholder, _ = _build_ifd(preview_tags, preview_ifd_offset)
    preview_data_offset = preview_ifd_offset + len(preview_placeholder)
    preview_tags = [item if item.code != 273 else _tag(273, LONG, _longs([preview_data_offset])) for item in preview_tags]
    preview_ifd, _ = _build_ifd(preview_tags, preview_ifd_offset)
    primary_tags = [item if item.code != 273 else _tag(273, LONG, _longs(strip_offsets)) for item in primary_tags]
    primary_ifd, _ = _build_ifd(primary_tags, 8, preview_ifd_offset)

    fd, temporary_name = tempfile.mkstemp(prefix=f".{final_path.stem}-", suffix=".tmp", dir=destination_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"II*\x00" + struct.pack("<I", 8))
            handle.write(primary_ifd)
            for item in strips:
                handle.write(item)
            handle.write(preview_ifd)
            handle.write(preview)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("rb") as check:
            signature = check.read(4)
        if temporary.stat().st_size < 1024 or signature != b"II*\x00":
            raise OSError("DNG validation failed before commit")
        with _OUTPUT_LOCK:
            # Re-check inside the process-wide commit lock so simultaneous jobs can
            # never replace one another's output.
            final_path = _unique_output_path(destination_dir, source)
            os.replace(temporary, final_path)
        return final_path
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
