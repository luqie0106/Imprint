"""Read bounded Adobe camera-profile metadata from a DNG root IFD.

This reader is for preview profile parsing. Export has a stricter copy policy
and continues to use :func:`dng_writer._embedded_profile_tags`.
"""

from __future__ import annotations

from pathlib import Path
import struct

import dng_writer


_MAX_IFD_ENTRIES = 4096
_MAX_TAG_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_TAG_BYTES = 64 * 1024 * 1024

_ACR_PROFILE_TAGS = {
    50721, 50722, 50723, 50724, 50727, 50778, 50779,
    50931, 50932, 50936, 50937, 50938, 50939, 50940,
    50941, 50942, 50964, 50965, 50981, 50982, 51107,
    51108, 51109,
    # Preserve third-illuminant markers so the two-illuminant preview parser
    # can reject the unsupported profile instead of silently dropping them.
    52529, 52530, 52531, 52532, 52535, 52537,
}


def _read_acr_profile_tags(
    reference: Path,
    expected_name: str | None,
) -> list[dng_writer._Tag]:
    """Read selected profile tags, optionally checking an external profile name."""
    with reference.open("rb") as handle:
        file_size = reference.stat().st_size
        header = handle.read(8)
        if len(header) != 8 or header[:4] != b"II*\x00":
            raise ValueError("profile reference must be a little-endian classic TIFF")

        ifd_offset = struct.unpack_from("<I", header, 4)[0]
        if ifd_offset < 8 or ifd_offset > file_size - 2:
            raise ValueError("profile reference has an invalid root IFD")

        handle.seek(ifd_offset)
        count_bytes = handle.read(2)
        if len(count_bytes) != 2:
            raise ValueError("profile reference root IFD is truncated")
        entry_count = struct.unpack("<H", count_bytes)[0]
        if entry_count > _MAX_IFD_ENTRIES:
            raise ValueError("profile reference root IFD has too many entries")

        entry_bytes_count = 12 * entry_count
        if ifd_offset + 2 + entry_bytes_count + 4 > file_size:
            raise ValueError("profile reference root IFD is truncated")
        entry_bytes = handle.read(entry_bytes_count)
        if len(entry_bytes) != entry_bytes_count:
            raise ValueError("profile reference root IFD is truncated")

        # First validate all selected entry declarations and aggregate sizes.
        # This bounds memory before any out-of-line payload is read.
        declared: list[tuple[int, int, int, int, bytes, int]] = []
        seen: set[int] = set()
        total_payload_bytes = 0
        for index in range(entry_count):
            entry = entry_bytes[index * 12:(index + 1) * 12]
            code, kind, item_count, value_offset = struct.unpack("<HHII", entry)
            if code not in _ACR_PROFILE_TAGS:
                continue
            if code in seen:
                raise ValueError("profile reference contains a duplicate profile tag")
            seen.add(code)

            type_size = dng_writer.TYPE_SIZES.get(kind)
            if type_size is None:
                raise ValueError("profile reference contains an unsupported TIFF type")
            if item_count == 0:
                raise ValueError("profile reference contains an empty profile tag")
            byte_count = type_size * item_count
            if byte_count > _MAX_TAG_BYTES:
                raise ValueError("profile reference profile tag exceeds the size limit")
            total_payload_bytes += byte_count
            if total_payload_bytes > _MAX_TOTAL_TAG_BYTES:
                raise ValueError("profile reference profile data exceeds the total size limit")
            declared.append((code, kind, item_count, byte_count, entry[8:12], value_offset))

        tags: list[dng_writer._Tag] = []
        for code, kind, item_count, byte_count, inline_value, value_offset in declared:
            if byte_count <= 4:
                data = inline_value[:byte_count]
            else:
                if value_offset > file_size or byte_count > file_size - value_offset:
                    raise ValueError("profile reference tag points outside the file")
                handle.seek(value_offset)
                data = handle.read(byte_count)
                if len(data) != byte_count:
                    raise ValueError("profile reference tag is truncated")
            tags.append(dng_writer._Tag(code, kind, item_count, data))

    name_tag = next((tag for tag in tags if tag.code == 50936), None)
    if name_tag is None or name_tag.kind != dng_writer.ASCII:
        raise ValueError("profile reference has no ASCII profile name")
    try:
        profile_name = name_tag.data.rstrip(b"\x00").decode("ascii", "strict")
    except UnicodeDecodeError as exc:
        raise ValueError("profile reference profile name is not ASCII") from exc
    if not profile_name:
        raise ValueError("profile reference profile name is empty")
    if expected_name is not None and profile_name != expected_name:
        raise ValueError("profile reference name does not match the source")

    return tags


def read_acr_profile_tags(reference: Path, expected_name: str) -> list[dng_writer._Tag]:
    """Read selected profile tags and strictly match an external profile name.

    The function copies only selected tag payloads into memory. It does not
    follow other IFDs, inspect copy policy, require a LookTable, or modify the
    source file.
    """
    if not isinstance(expected_name, str):
        raise TypeError("expected_name must be a string")
    return _read_acr_profile_tags(reference, expected_name)


def read_embedded_acr_profile_tags(reference: Path) -> list[dng_writer._Tag]:
    """Read selected profile tags using the valid root-IFD profile name.

    This is for a source DNG with no externally selected profile name. The
    profile name stored in the root IFD is still required to be non-empty ASCII.
    """
    return _read_acr_profile_tags(reference, None)
