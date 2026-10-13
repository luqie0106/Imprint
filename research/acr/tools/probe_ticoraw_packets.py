#!/usr/bin/env python3
"""Strictly inventory JPEG XS-style packet boundaries in a Nikon TicoRAW strip.

The probe reads Classic TIFF metadata and the selected compressed RAW strip. It
does not decode precinct payloads or include compressed image data in its JSON
output. Only segment fields and precinct headers/boundaries are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from probe_ticoraw_header import (
    EXPECTED_RAW_HEIGHT,
    EXPECTED_RAW_WIDTH,
    ClassicTiff,
    ProbeError,
    find_raw_ifd,
    read_strip_bounds,
    sha256_file,
)


MARKER_NAMES = {
    0xFF11: "EOC",
    0xFF12: "PIH",
    0xFF13: "CDT",
    0xFF14: "WGT",
    0xFF20: "SLH",
    0xFF50: "CAP",
}
PRECINCTS_PER_SLICE = 16
PRECINCT_HEADER_SIZE = 12
SLICE_MARKER = b"\xff\x20\x00\x04"
VENDOR_TAIL_SIZE = 13


def be16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 2], "big")


def be24(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 3], "big")


def marker_hex(marker: int) -> str:
    return f"{marker:04x}"


def parse_pih(body: bytes, marker_offset: int) -> dict[str, Any]:
    if len(body) < 20:
        raise ProbeError(
            f"PIH at strip offset {marker_offset} has {len(body)} body bytes; at least 20 required"
        )

    fields: dict[str, Any] = {
        "Lcod": int.from_bytes(body[0:4], "big"),
        "width": be16(body, 8),
        "height": be16(body, 10),
        "precinct_width": be16(body, 12),
        "Hsl": be16(body, 14),
        "component_count": body[16],
        "coefficient_group": body[17],
        "significance_group": body[18],
        "Bw": body[19],
    }

    if len(body) >= 24 + VENDOR_TAIL_SIZE:
        tail = body[24:24 + VENDOR_TAIL_SIZE]
        fields["vendor_tail"] = {
            "body_offset": 24,
            "length_bytes": len(tail),
            "hex": tail.hex(),
            "derived_fields": {
                "B": ((tail[3] & 0x0F) << 8) | tail[4],
                "A": ((tail[7] & 0x03) << 16) | (tail[8] << 8) | tail[9],
                "C": (tail[10] << 8) | tail[11],
            },
            "interpretation": (
                "B/A/C use the public Nikon HE/JPEG XS implementation interpretation "
                "recorded for this research. LibRaw independently confirms the 13-byte "
                "tail location; Adobe Camera Raw behavior remains unverified."
            ),
        }
    else:
        fields["vendor_tail"] = None
    return fields


def parse_header(strip: bytes) -> tuple[int, list[dict[str, Any]], dict[str, Any], list[dict[str, int]]]:
    if len(strip) < 2 or strip[:2] != b"\xff\x10":
        raise ProbeError("selected RAW strip does not begin with SOC marker ff10")

    segments: list[dict[str, Any]] = []
    parsed: dict[str, Any] = {}
    offset = 2
    initial_slice: dict[str, int] | None = None

    while True:
        if offset + 4 > len(strip):
            raise ProbeError(f"truncated header segment prefix at strip offset {offset}")
        marker = be16(strip, offset)
        if strip[offset] != 0xFF:
            raise ProbeError(f"invalid marker prefix at strip offset {offset}")
        if marker == 0xFF11:
            raise ProbeError(f"EOC occurs before the initial SLH at strip offset {offset}")

        lseg = be16(strip, offset + 2)
        if lseg < 2:
            raise ProbeError(f"marker {marker_hex(marker)} at {offset} has invalid Lseg={lseg}")
        end = offset + 2 + lseg
        if end > len(strip):
            raise ProbeError(
                f"marker {marker_hex(marker)} at {offset} ends at {end}, beyond strip size {len(strip)}"
            )
        body = strip[offset + 4:end]
        name = MARKER_NAMES.get(marker, "unknown")
        segment: dict[str, Any] = {
            "marker": marker_hex(marker),
            "name": name,
            "offset": offset,
            "Lseg": lseg,
            "body_length": len(body),
            "end_exclusive": end,
            "body_hex": body.hex(),
        }

        if marker == 0xFF12:
            if "PIH" in parsed:
                raise ProbeError(f"duplicate PIH marker at strip offset {offset}")
            parsed["PIH"] = parse_pih(body, offset)
        elif marker == 0xFF14:
            if "WGT" in parsed:
                raise ProbeError(f"duplicate WGT marker at strip offset {offset}")
            if not body or len(body) % 2:
                raise ProbeError(
                    f"WGT at strip offset {offset} must contain nonempty gain/priority pairs"
                )
            parsed["WGT"] = {
                "band_count": len(body) // 2,
                "bands": [
                    {"gain": body[index], "priority": body[index + 1]}
                    for index in range(0, len(body), 2)
                ],
            }

        if marker == 0xFF20:
            if lseg != 4 or len(body) != 2:
                raise ProbeError(
                    f"initial SLH at {offset} must have Lseg=4 and a two-byte slice index"
                )
            slice_index = int.from_bytes(body, "big")
            if slice_index != 0:
                raise ProbeError(f"initial SLH at {offset} has slice index {slice_index}; expected 0")
            segment["slice_index"] = slice_index
            initial_slice = {
                "slice_index": slice_index,
                "marker_offset": offset,
                "precinct_count_before": 0,
                "first_precinct_offset": end,
            }

        segments.append(segment)
        offset = end
        if marker == 0xFF20:
            break

    if "PIH" not in parsed or "WGT" not in parsed:
        missing = [name for name in ("PIH", "WGT") if name not in parsed]
        raise ProbeError(f"header is missing required segment(s): {', '.join(missing)}")

    pih = parsed["PIH"]
    if (pih["width"], pih["height"]) != (EXPECTED_RAW_WIDTH, EXPECTED_RAW_HEIGHT):
        raise ProbeError(
            f"PIH dimensions {pih['width']}x{pih['height']} do not match selected RAW IFD "
            f"{EXPECTED_RAW_WIDTH}x{EXPECTED_RAW_HEIGHT}"
        )
    if pih["Lcod"] != len(strip):
        raise ProbeError(f"PIH Lcod={pih['Lcod']} does not match strip byte count {len(strip)}")

    assert initial_slice is not None
    return offset, segments, parsed, [initial_slice]


def parse_precinct_stream(
    strip: bytes, offset: int, slice_boundaries: list[dict[str, int]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    precincts: list[dict[str, Any]] = []
    next_slice_index = 1

    while True:
        # An EOC is accepted only at a precinct boundary and only when it is the
        # final two bytes of the strip. The current sample has no tail padding.
        if offset + 2 <= len(strip) and strip[offset:offset + 2] == b"\xff\x11":
            if offset + 2 != len(strip):
                trailing = len(strip) - (offset + 2)
                raise ProbeError(
                    f"EOC at strip offset {offset} is followed by {trailing} trailing byte(s); "
                    "this probe accepts no unvalidated tail padding"
                )
            return precincts, {
                "marker": "ff11",
                "name": "EOC",
                "offset": offset,
                "end_exclusive": offset + 2,
                "trailing_padding_bytes": 0,
                "strip_end_exclusive": len(strip),
            }

        if precincts and len(precincts) % PRECINCTS_PER_SLICE == 0:
            if offset + 6 > len(strip):
                raise ProbeError(f"truncated SLH after precinct {len(precincts) - 1} at {offset}")
            if strip[offset:offset + 4] != SLICE_MARKER:
                raise ProbeError(
                    f"expected SLH ff20 0004 after {len(precincts)} precincts at strip offset {offset}; "
                    f"found {strip[offset:offset + 4].hex()}"
                )
            slice_index = be16(strip, offset + 4)
            if slice_index != next_slice_index:
                raise ProbeError(
                    f"SLH at {offset} has slice index {slice_index}; expected {next_slice_index}"
                )
            slice_boundaries.append(
                {
                    "slice_index": slice_index,
                    "marker_offset": offset,
                    "precinct_count_before": len(precincts),
                    "first_precinct_offset": offset + 6,
                }
            )
            offset += 6
            next_slice_index += 1

        if offset + PRECINCT_HEADER_SIZE > len(strip):
            raise ProbeError(
                f"truncated precinct header at strip offset {offset}: "
                f"need {PRECINCT_HEADER_SIZE}, have {len(strip) - offset}"
            )
        declared_size = be24(strip, offset)
        full_size = declared_size + PRECINCT_HEADER_SIZE
        end = offset + full_size
        if end > len(strip):
            raise ProbeError(
                f"precinct {len(precincts)} at {offset} declares size {declared_size} "
                f"(full length {full_size}) ending at {end}, beyond strip size {len(strip)}"
            )

        precincts.append(
            {
                "index": len(precincts),
                "offset": offset,
                "declared_size": declared_size,
                "total_size": full_size,
                "Bp": strip[offset + 3],
                "Br": strip[offset + 4],
                "Dpb_hex": strip[offset + 5:offset + 12].hex(),
                "end_exclusive": end,
            }
        )
        offset = end


def probe(input_path: Path) -> dict[str, Any]:
    tiff = ClassicTiff(input_path)
    try:
        directories = tiff.walk_ifds()
        selected = find_raw_ifd(directories)
        strips = read_strip_bounds(tiff, selected["metadata"])
        if not strips:
            raise ProbeError("selected RAW IFD has no strips")
        # This sample is expected to use one strip. Refuse to imply a single
        # continuous packet stream if the selected IFD has multiple strips.
        if len(strips) != 1:
            raise ProbeError(
                f"selected RAW IFD has {len(strips)} strips; packet traversal requires one strip"
            )
        raw_strip_info = strips[0]
        strip = tiff.read_at(
            raw_strip_info["offset"], raw_strip_info["byte_count"], "complete selected RAW strip"
        )
        stream_offset, segments, parsed_header, slice_boundaries = parse_header(strip)
        precincts, termination = parse_precinct_stream(strip, stream_offset, slice_boundaries)

        slice_counts = []
        for index, boundary in enumerate(slice_boundaries):
            start = boundary["precinct_count_before"]
            stop = (
                slice_boundaries[index + 1]["precinct_count_before"]
                if index + 1 < len(slice_boundaries)
                else len(precincts)
            )
            slice_counts.append(
                {
                    "slice_index": boundary["slice_index"],
                    "first_precinct_index": start,
                    "precinct_count": stop - start,
                }
            )

        return {
            "probe": "nikon_ticoraw_packet_probe_v1",
            "status": "strict_packet_boundary_inventory",
            "sample": {
                "name": input_path.name,
                "size_bytes": tiff.file_size,
                "sha256": sha256_file(input_path),
                "tiff_byte_order": tiff.byte_order,
                "first_ifd_offset": tiff.first_ifd_offset,
            },
            "selected_raw_ifd": {
                "path": selected["path"],
                "offset": selected["offset"],
                "width": EXPECTED_RAW_WIDTH,
                "height": EXPECTED_RAW_HEIGHT,
                "compression": 34713,
                "bits_per_sample": 14,
                "selection": "largest area among IFDs with compression 34713 and 14-bit samples",
            },
            "strip": {
                **raw_strip_info,
                "sha256": hashlib.sha256(strip).hexdigest(),
                "byte_count_matches_input_bounds": (
                    raw_strip_info["end_exclusive"] - raw_strip_info["offset"] == len(strip)
                ),
            },
            "header": {
                "soc": {"marker": "ff10", "offset": 0},
                "segments": segments,
                "parsed": parsed_header,
                "precinct_stream_offset": stream_offset,
                "segment_length_rule": "Lseg includes its own 2-byte length field; excludes the 2-byte marker",
            },
            "slice_boundaries": slice_boundaries,
            "slice_counts": slice_counts,
            "precinct_count": len(precincts),
            "precincts": precincts,
            "termination": termination,
            "validation": {
                "precinct_size_rule": "total_size = 3-byte BE declared_size + 12-byte precinct header",
                "slice_rule": (
                    f"After each {PRECINCTS_PER_SLICE} precincts, require exactly "
                    "ff20 0004 followed by the next sequential BE16 slice index."
                ),
                "eoc_rule": "Accept ff11 only at a precinct boundary and only as the final two strip bytes.",
                "trailing_padding_bytes": termination["trailing_padding_bytes"],
                "compressed_precinct_payloads_included": False,
                "vendor_tail_source": (
                    "Public LibRaw Nikon HE/JPEG XS implementation: "
                    "src/decoders/nikon_he/nikon_he_picture_header.h and .cpp; "
                    "the implementation declares a 13-byte tail and copies it from PIH body offset 24."
                ),
                "limitations": [
                    "This inventories compressed packet boundaries and selected header fields; it does not decode image pixels.",
                    "Vendor-tail B/A/C bit formulas follow the public implementation and are not validated against Adobe Camera Raw behavior.",
                    "The current sample had no bytes after EOC; nonempty tail padding is rejected rather than inferred.",
                ],
            },
        }
    finally:
        tiff.close()


def write_output(output_path: Path, input_path: Path, serialized: str) -> None:
    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    if input_resolved == output_resolved:
        raise ProbeError("output path must not be the input sample")
    if output_path.exists():
        try:
            if os.path.samefile(input_path, output_path):
                raise ProbeError("output path refers to the input sample")
        except OSError:
            pass
        raise ProbeError(f"output already exists; refusing to overwrite: {output_path}")
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(serialized)
    except FileExistsError as exc:
        raise ProbeError(f"output already exists; refusing to overwrite: {output_path}") from exc
    except OSError as exc:
        raise ProbeError(f"cannot write JSON output {output_path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="Nikon NEF sample to inspect read-only")
    parser.add_argument("--output", type=Path, help="write JSON evidence to a new path (default: stdout)")
    args = parser.parse_args(argv)

    try:
        result = probe(args.sample)
        serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            write_output(args.output, args.sample, serialized)
        else:
            sys.stdout.write(serialized)
    except (ProbeError, OSError) as exc:
        print(f"probe_ticoraw_packets: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
