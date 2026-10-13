#!/usr/bin/env python3
"""Read-only, bounded Nikon NEF/TicoRAW header probe.

This script parses only Classic TIFF directory metadata and a small prefix of
the selected RAW strip. It does not decode or copy strip/image data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any


TIFF_TYPE_SIZES = {
    1: 1,   # BYTE
    2: 1,   # ASCII
    3: 2,   # SHORT
    4: 4,   # LONG
    5: 8,   # RATIONAL
    6: 1,   # SBYTE
    7: 1,   # UNDEFINED
    8: 2,   # SSHORT
    9: 4,   # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
    13: 4,  # IFD
}
NUMERIC_TIFF_TYPES = {1, 3, 4, 6, 8, 9, 13}
METADATA_TAGS = {
    256: "width",
    257: "height",
    258: "bits_per_sample",
    259: "compression",
    273: "strip_offsets",
    279: "strip_byte_counts",
    330: "sub_ifds",
}
EXPECTED_RAW_WIDTH = 6064
EXPECTED_RAW_HEIGHT = 4040
EXPECTED_RAW_COMPRESSION = 34713
EXPECTED_RAW_BITS = 14
EXTENSION_MARKER = b"CONTACT_INTOPIX_"


class ProbeError(ValueError):
    """Input is not suitable for the bounded probe."""


class ClassicTiff:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.file = path.open("rb")
        except OSError as exc:
            raise ProbeError(f"cannot open input {path}: {exc}") from exc
        try:
            self.file_size = os.fstat(self.file.fileno()).st_size
            header = self.read_at(0, 8, "Classic TIFF header")
            if header[:2] == b"II":
                self.byte_order = "little"
                self.endian = "<"
            elif header[:2] == b"MM":
                self.byte_order = "big"
                self.endian = ">"
            else:
                raise ProbeError("invalid TIFF byte-order marker (expected II or MM)")
            magic = struct.unpack(self.endian + "H", header[2:4])[0]
            if magic != 42:
                raise ProbeError(f"unsupported TIFF magic {magic}; expected Classic TIFF magic 42")
            self.first_ifd_offset = struct.unpack(self.endian + "I", header[4:8])[0]
            if self.first_ifd_offset == 0:
                raise ProbeError("TIFF header has a zero first-IFD offset")
        except Exception:
            self.file.close()
            raise

    def close(self) -> None:
        self.file.close()

    def read_at(self, offset: int, size: int, description: str) -> bytes:
        if offset < 0 or size < 0 or offset > self.file_size or size > self.file_size - offset:
            raise ProbeError(
                f"truncated or out-of-bounds {description}: offset={offset}, "
                f"size={size}, file_size={self.file_size}"
            )
        self.file.seek(offset)
        data = self.file.read(size)
        if len(data) != size:
            raise ProbeError(f"short read for {description}: expected {size}, got {len(data)}")
        return data

    def read_numeric_values(self, field_type: int, count: int, raw_value: bytes, tag: int) -> list[int]:
        if field_type not in NUMERIC_TIFF_TYPES:
            raise ProbeError(f"TIFF tag {tag} uses unsupported numeric field type {field_type}")
        type_size = TIFF_TYPE_SIZES[field_type]
        total_size = type_size * count
        if total_size <= 4:
            payload = raw_value[:total_size]
        else:
            value_offset = struct.unpack(self.endian + "I", raw_value)[0]
            payload = self.read_at(value_offset, total_size, f"TIFF tag {tag} values")

        formats = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 13: "I"}
        code = formats[field_type]
        return [item[0] for item in struct.iter_unpack(self.endian + code, payload)]

    def parse_ifd(self, offset: int) -> dict[str, Any]:
        count_data = self.read_at(offset, 2, f"IFD entry count at {offset}")
        entry_count = struct.unpack(self.endian + "H", count_data)[0]
        entries_size = entry_count * 12
        entries_offset = offset + 2
        entries = self.read_at(entries_offset, entries_size, f"IFD entries at {offset}")
        next_offset_data = self.read_at(entries_offset + entries_size, 4, f"next IFD pointer at {offset}")
        next_ifd_offset = struct.unpack(self.endian + "I", next_offset_data)[0]

        metadata: dict[str, Any] = {}
        for index in range(entry_count):
            start = index * 12
            tag, field_type, count = struct.unpack(self.endian + "HHI", entries[start:start + 8])
            if tag not in METADATA_TAGS:
                continue
            raw_value = entries[start + 8:start + 12]
            values = self.read_numeric_values(field_type, count, raw_value, tag)
            metadata[METADATA_TAGS[tag]] = values

        return {
            "offset": offset,
            "next_ifd_offset": next_ifd_offset,
            "metadata": metadata,
        }

    def walk_ifds(self) -> list[dict[str, Any]]:
        pending: list[tuple[int, str]] = [(self.first_ifd_offset, "IFD0")]
        seen: set[int] = set()
        queued: set[int] = {self.first_ifd_offset}
        directories: list[dict[str, Any]] = []

        while pending:
            offset, path = pending.pop(0)
            if offset == 0 or offset in seen:
                continue
            if len(directories) >= 4096:
                raise ProbeError("IFD traversal exceeded the safety limit of 4096 directories")
            seen.add(offset)
            directory = self.parse_ifd(offset)
            directory["path"] = path
            directories.append(directory)

            sub_ifds = directory["metadata"].get("sub_ifds", [])
            for index, child_offset in enumerate(sub_ifds):
                if child_offset and child_offset not in seen and child_offset not in queued:
                    pending.append((child_offset, f"{path}/SubIFD[{index}]"))
                    queued.add(child_offset)
            next_offset = directory["next_ifd_offset"]
            if next_offset and next_offset not in seen and next_offset not in queued:
                pending.append((next_offset, f"{path}/next"))
                queued.add(next_offset)

        return directories


def scalar(metadata: dict[str, Any], key: str) -> int | None:
    values = metadata.get(key)
    return values[0] if isinstance(values, list) and len(values) == 1 else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProbeError(f"cannot hash input {path}: {exc}") from exc
    return digest.hexdigest()


def find_raw_ifd(directories: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for directory in directories:
        metadata = directory["metadata"]
        width = scalar(metadata, "width")
        height = scalar(metadata, "height")
        compression = scalar(metadata, "compression")
        bits = metadata.get("bits_per_sample", [])
        if (
            width is not None
            and height is not None
            and compression == EXPECTED_RAW_COMPRESSION
            and bits
            and all(value == EXPECTED_RAW_BITS for value in bits)
        ):
            candidates.append(directory)

    if not candidates:
        raise ProbeError(
            "no IFD found with Nikon RAW compression 34713 and 14-bit samples"
        )

    selected = max(
        candidates,
        key=lambda item: scalar(item["metadata"], "width")
        * scalar(item["metadata"], "height"),
    )
    metadata = selected["metadata"]
    width = scalar(metadata, "width")
    height = scalar(metadata, "height")
    if (width, height) != (EXPECTED_RAW_WIDTH, EXPECTED_RAW_HEIGHT):
        raise ProbeError(
            "largest Nikon RAW IFD has unexpected dimensions "
            f"{width}x{height}; expected {EXPECTED_RAW_WIDTH}x{EXPECTED_RAW_HEIGHT}"
        )
    return selected


def read_strip_bounds(
    tiff: ClassicTiff, metadata: dict[str, Any]
) -> list[dict[str, int]]:
    offsets = metadata.get("strip_offsets")
    byte_counts = metadata.get("strip_byte_counts")
    if not offsets or not byte_counts:
        raise ProbeError("selected RAW IFD has no StripOffsets/StripByteCounts")
    if len(offsets) != len(byte_counts):
        raise ProbeError(
            "selected RAW IFD has different StripOffsets and StripByteCounts lengths"
        )

    bounds = []
    for index, (offset, byte_count) in enumerate(zip(offsets, byte_counts)):
        if byte_count <= 0:
            raise ProbeError(f"RAW strip {index} has a non-positive byte count {byte_count}")
        if offset < 0 or offset > tiff.file_size or byte_count > tiff.file_size - offset:
            raise ProbeError(
                f"RAW strip {index} is outside the input file: offset={offset}, "
                f"byte_count={byte_count}, file_size={tiff.file_size}"
            )
        bounds.append(
            {
                "index": index,
                "offset": offset,
                "byte_count": byte_count,
                "end_exclusive": offset + byte_count,
            }
        )
    return bounds


def probe_marker_header(strip_prefix: bytes, strip_size: int) -> dict[str, Any]:
    if len(strip_prefix) < 6:
        raise ProbeError("first RAW strip is too short for the TicoRAW prefix")
    if strip_prefix[0:2] != b"\xff\x10":
        raise ProbeError("first RAW strip does not begin with marker ff10")
    if strip_prefix[2:4] != b"\xff\x50":
        raise ProbeError("first RAW strip does not contain marker ff50 at bytes 2:4")

    n = int.from_bytes(strip_prefix[4:6], "big")
    if n < 2:
        raise ProbeError(f"TicoRAW header length n={n} is below the required minimum 2")
    if n + 32 > strip_size:
        raise ProbeError(
            f"TicoRAW header length n={n} requires {n + 32} bytes, "
            f"but the first strip has {strip_size}"
        )

    # Only read the bounded header area; no strip copy is made.
    header = strip_prefix if len(strip_prefix) >= n + 32 else None
    # The caller supplies the first 128 bytes only, so fetch a larger marker
    # window only when n places ff12 beyond that prefix.
    if header is None:
        raise ProbeError("internal probe error: bounded marker header was not read")

    extension = header[6:22]
    if any(extension) and extension != EXTENSION_MARKER:
        raise ProbeError(
            "nonzero TicoRAW extension block does not match CONTACT_INTOPIX_"
        )

    marker_offset = n + 4
    if header[marker_offset:marker_offset + 2] != b"\xff\x12":
        raise ProbeError(f"TicoRAW marker ff12 missing at strip offset {marker_offset}")
    payload_offset = n + 8
    payload = header[payload_offset:payload_offset + 20]
    if len(payload) != 20:
        raise ProbeError("TicoRAW payload header is truncated")

    width = int.from_bytes(payload[8:10], "big")
    height = int.from_bytes(payload[10:12], "big")
    zero_field = int.from_bytes(payload[12:14], "big")
    if zero_field != 0:
        raise ProbeError(f"TicoRAW payload p+12 field is {zero_field}, expected 0")
    if payload[17] != 4 or payload[18] != 8:
        raise ProbeError(
            f"TicoRAW payload p+17/p+18 are {payload[17]}/{payload[18]}, expected 4/8"
        )
    if payload[19] > 20:
        raise ProbeError(
            f"TicoRAW payload p+19 value {payload[19]} exceeds the bound 20"
        )

    return {
        "prefix_markers": ["ff10", "ff50"],
        "declared_n": n,
        "extension_marker": EXTENSION_MARKER.decode("ascii") if any(extension) else None,
        "ff12_marker_offset": marker_offset,
        "ff12_marker": "ff12",
        "payload_offset": payload_offset,
        "payload_header_hex": payload.hex(),
        "width": width,
        "height": height,
        "zero_field_p_plus_12": zero_field,
        "field_p_plus_17": payload[17],
        "field_p_plus_18": payload[18],
        "field_p_plus_19": payload[19],
        "unknown_format_byte": payload[16],
        "scope": "bounded layout and bounds check only; not a decoder or full format validation",
    }


def probe(input_path: Path) -> dict[str, Any]:
    # Keep the TIFF handle open while metadata and bounded strip prefixes are read.
    tiff = ClassicTiff(input_path)
    try:
        directories = tiff.walk_ifds()
        selected = find_raw_ifd(directories)
        strip_bounds = read_strip_bounds(tiff, selected["metadata"])
        first_strip = strip_bounds[0]
        strip_prefix = tiff.read_at(
            first_strip["offset"], min(128, first_strip["byte_count"]), "first 128 RAW strip bytes"
        )
        # If the ff12 marker is farther than byte 127, read only the rest of
        # the bounded header, not the RAW strip.
        n = int.from_bytes(strip_prefix[4:6], "big") if len(strip_prefix) >= 6 else 0
        required_header_size = n + 32
        if required_header_size > len(strip_prefix) and required_header_size <= first_strip["byte_count"]:
            strip_prefix += tiff.read_at(
                first_strip["offset"] + len(strip_prefix),
                required_header_size - len(strip_prefix),
                "bounded TicoRAW marker header",
            )
        marker_header = probe_marker_header(strip_prefix, first_strip["byte_count"])
        if (marker_header["width"], marker_header["height"]) != (
            EXPECTED_RAW_WIDTH,
            EXPECTED_RAW_HEIGHT,
        ):
            raise ProbeError(
                "TicoRAW payload dimensions "
                f"{marker_header['width']}x{marker_header['height']} do not match "
                f"the selected RAW IFD {EXPECTED_RAW_WIDTH}x{EXPECTED_RAW_HEIGHT}"
            )
        sample_hash = sha256_file(input_path)

        serialized_ifds = []
        for directory in directories:
            metadata = directory["metadata"]
            serialized_ifds.append(
                {
                    "path": directory["path"],
                    "offset": directory["offset"],
                    "next_ifd_offset": directory["next_ifd_offset"],
                    "metadata": metadata,
                }
            )

        return {
            "probe": "nikon_ticoraw_header_probe_v1",
            "status": "partial_header_probe",
            "sample": {
                "name": input_path.name,
                "size_bytes": tiff.file_size,
                "sha256": sample_hash,
                "tiff_byte_order": tiff.byte_order,
                "first_ifd_offset": tiff.first_ifd_offset,
            },
            "ifds": serialized_ifds,
            "selected_raw_ifd": {
                "path": selected["path"],
                "offset": selected["offset"],
                "metadata": selected["metadata"],
                "selection": "largest area among IFDs with compression 34713 and 14-bit samples",
            },
            "strip_bounds": strip_bounds,
            "first_strip_probe": {
                "first_128_bytes_hex": strip_prefix[:128].hex(),
                "marker_header": marker_header,
            },
            "limitations": [
                "Only Classic TIFF metadata and a bounded RAW-strip header were read.",
                "No image or complete strip was decoded or copied.",
                "Header checks are not complete format validation.",
            ],
        }
    finally:
        tiff.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="Nikon NEF sample to inspect read-only")
    parser.add_argument("--output", type=Path, help="write JSON evidence to this path (default: stdout)")
    args = parser.parse_args(argv)

    try:
        result = probe(args.sample)
        serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            input_resolved = args.sample.resolve()
            output_resolved = args.output.resolve()
            if input_resolved == output_resolved:
                raise ProbeError("output path must not be the input sample")
            if args.output.exists() and os.path.samefile(args.sample, args.output):
                raise ProbeError("output path refers to the input sample")
            try:
                args.output.write_text(serialized, encoding="utf-8")
            except OSError as exc:
                raise ProbeError(f"cannot write JSON output {args.output}: {exc}") from exc
        else:
            sys.stdout.write(serialized)
    except (ProbeError, OSError) as exc:
        print(f"probe_ticoraw_header: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
