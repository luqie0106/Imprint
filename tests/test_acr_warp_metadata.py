"""Prepared WarpRectilinear metadata reader contracts."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import acr_warp_metadata as prepared
from dng_warp import DNGWarpError


def _warp_payload(planes: int = 3, coefficients=None, center=(0.5, 0.5)) -> bytes:
    if coefficients is None:
        coefficients = [
            value
            for plane in range(planes)
            for value in (1.0 + plane / 7.0, 0.1, -0.02, 0.003, 0.0, 0.0)
        ]
    return struct.pack(">I", planes) + struct.pack(
        ">" + "d" * (6 * planes + 2), *coefficients, *center
    )


def _opcode(opcode_id: int, data: bytes, *, version=0x01030000, flags=0) -> bytes:
    return struct.pack(">4I", opcode_id, version, flags, len(data)) + data


def _opcode_list(*opcodes: bytes, trailing=b"") -> bytes:
    return struct.pack(">I", len(opcodes)) + b"".join(opcodes) + trailing


def _valid_gainmap_payload() -> bytes:
    fixed = struct.pack(">10I4dI", 0, 0, 2, 2, 0, 3, 1, 1, 1, 1,
                        1.0, 1.0, 0.0, 0.0, 1)
    return fixed + struct.pack(">f", 1.0)


def _metadata_tags(
    endian: str,
    *,
    width=100,
    height=80,
    orientation=1,
    active_area=(2, 3, 78, 97),
    scale=(3, 2, 5, 4),
    opcode_list: bytes | None = None,
):
    tags = [
        (256, 4, 1, struct.pack(endian + "I", width)),
        (257, 4, 1, struct.pack(endian + "I", height)),
        (274, 3, 1, struct.pack(endian + "H", orientation)),
        (50829, 4, 4, struct.pack(endian + "4I", *active_area)),
        (50718, 5, 2, struct.pack(endian + "4I", *scale)),
    ]
    if opcode_list is not None:
        tags.append((51022, 7, len(opcode_list), opcode_list))
    return tags


def _write_tiff(tmp_path, directories, *, endian="<", name="fixture.dng"):
    """Write small ClassicTIFF IFD graphs with optional next/SubIFD links."""
    byte_order = "II" if endian == "<" else "MM"
    directory_offsets = []
    cursor = 8
    for directory in directories:
        tags = list(directory["tags"])
        if directory.get("sub_ifds"):
            tags.append((330, 4, len(directory["sub_ifds"]), None))
        tags.sort(key=lambda item: item[0])
        directory_offsets.append(cursor)
        cursor += 2 + 12 * len(tags) + 4

    external_offset = cursor
    external_data = bytearray()
    encoded_directories = []
    for index, directory in enumerate(directories):
        tags = list(directory["tags"])
        if directory.get("sub_ifds"):
            children = [directory_offsets[child] for child in directory["sub_ifds"]]
            tags.append((330, 4, len(children), struct.pack(endian + "I" * len(children), *children)))
        tags.sort(key=lambda item: item[0])
        entries = []
        for tag, kind, count, raw in tags:
            if raw is None:
                raise AssertionError("unexpected empty TIFF tag data")
            if len(raw) <= 4:
                value_field = raw.ljust(4, b"\0")
            else:
                value_field = struct.pack(endian + "I", external_offset + len(external_data))
                external_data.extend(raw)
            entries.append(struct.pack(endian + "HHI4s", tag, kind, count, value_field))
        next_index = directory.get("next")
        next_offset = 0 if next_index is None else directory_offsets[next_index]
        encoded = (
            struct.pack(endian + "H", len(entries))
            + b"".join(entries)
            + struct.pack(endian + "I", next_offset)
        )
        encoded_directories.append(encoded)

    root_offset = directory_offsets[0]
    data = (
        byte_order.encode("ascii")
        + struct.pack(endian + "HI", 42, root_offset)
        + b"".join(encoded_directories)
        + external_data
    )
    path = tmp_path / name
    path.write_bytes(data)
    return path, tuple(directory_offsets)


def _simple_tiff(tmp_path, opcodes, *, endian="<", name="fixture.dng", **metadata):
    listing = _opcode_list(*opcodes)
    directory = {"tags": _metadata_tags(endian, opcode_list=listing, **metadata)}
    return _write_tiff(tmp_path, [directory], endian=endian, name=name)[0]


@pytest.mark.parametrize("endian", ["<", ">"])
def test_reads_big_and_little_endian_classic_tiff_and_preserves_same_ifd_scale(endian, tmp_path):
    path = _simple_tiff(
        tmp_path,
        [_opcode(1, _warp_payload())],
        endian=endian,
        scale=(3, 2, 5, 4),
    )

    result = prepared.parse_acr_warp_prepared(path)

    assert result is not None
    assert (result.image_width, result.image_height) == (100, 80)
    assert result.active_area == (2, 3, 78, 97)
    assert (result.default_scale_h, result.default_scale_v) == (1.5, 1.25)
    assert result.opcode_ids == (1,)
    assert len(result.fingerprint) == 6


@pytest.mark.parametrize("planes", [1, 2, 3, 4])
def test_keeps_each_supported_plane_as_readonly_float64_without_broadcast(planes, tmp_path):
    values = [float(index) / 13.0 for index in range(planes * 6)]
    path = _simple_tiff(tmp_path, [_opcode(1, _warp_payload(planes, values))])

    result = prepared.parse_acr_warp_prepared(path)

    assert result is not None
    assert result.planes == planes
    assert result.coefficients.shape == (planes, 6)
    assert result.coefficients.dtype == np.float64
    assert not result.coefficients.flags.writeable
    np.testing.assert_array_equal(result.coefficients, np.asarray(values).reshape(planes, 6))


def test_payload_double_values_do_not_round_trip_through_float32(tmp_path):
    original = [0.1, 1.0000000000000002, -0.3456789012345678, 0.0, 0.0, 0.0]
    path = _simple_tiff(tmp_path, [_opcode(1, _warp_payload(1, original))])

    result = prepared.parse_acr_warp_prepared(path)

    assert result is not None
    assert result.coefficients[0, 0] == np.float64(original[0])
    assert result.coefficients[0, 1] == np.float64(original[1])
    assert result.coefficients[0, 1] != np.float64(np.float32(original[1]))


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        struct.pack(">I", 0),
        struct.pack(">I", 5),
        _warp_payload(1)[:-1],
        _warp_payload(1, [float("nan"), 0, 0, 0, 0, 0]),
        _warp_payload(1, center=(1.01, 0.5)),
        _warp_payload(2) + b"x",
    ],
)
def test_malformed_warp_payload_raises(payload, tmp_path):
    path = _simple_tiff(tmp_path, [_opcode(1, payload)])
    with pytest.raises(DNGWarpError):
        prepared.parse_acr_warp_prepared(path)


def test_optional_unknown_opcode_is_skipped(tmp_path):
    path = _simple_tiff(tmp_path, [
        _opcode(999, b"opaque", flags=1),
        _opcode(1, _warp_payload(2)),
    ])

    result = prepared.parse_acr_warp_prepared(path)

    assert result is not None
    assert result.opcode_ids == (999, 1)
    assert result.planes == 2


@pytest.mark.parametrize("flags", [0, 1])
def test_recognized_opcode_accepts_only_supported_optional_flag_values(flags, tmp_path):
    result = prepared.parse_acr_warp_prepared(
        _simple_tiff(tmp_path, [_opcode(1, _warp_payload(), flags=flags)])
    )
    assert result is not None


@pytest.mark.parametrize(
    "opcodes",
    [
        [_opcode(1, _warp_payload(), version=0x01020000)],
        [_opcode(1, _warp_payload(), flags=2)],
        [_opcode(999, b"opaque"), _opcode(1, _warp_payload())],
        [_opcode(1, _warp_payload()), _opcode(1, _warp_payload())],
        [_opcode(9, _valid_gainmap_payload()), _opcode(9, _valid_gainmap_payload()),
         _opcode(1, _warp_payload())],
    ],
)
def test_unsupported_or_duplicate_opcode_pipeline_returns_none(opcodes, tmp_path):
    assert prepared.parse_acr_warp_prepared(_simple_tiff(tmp_path, opcodes)) is None


def test_gainmap_order_is_reported_and_payload_is_validated(tmp_path):
    gain = _opcode(9, _valid_gainmap_payload())
    warp = _opcode(1, _warp_payload())
    before = prepared.parse_acr_warp_prepared(_simple_tiff(tmp_path, [gain, warp]))
    after = prepared.parse_acr_warp_prepared(_simple_tiff(tmp_path, [warp, gain]))

    assert before is not None and before.gain_map_present and before.gain_map_before_warp
    assert after is not None and after.gain_map_present and not after.gain_map_before_warp
    with pytest.raises(DNGWarpError):
        prepared.parse_acr_warp_prepared(
            _simple_tiff(tmp_path, [_opcode(9, b"bad"), warp], name="bad-gain.dng")
        )


@pytest.mark.parametrize(
    "listing",
    [
        struct.pack(">I", 1) + b"\0" * 3,
        struct.pack(">4I", 1, 0x01030000, 0, 20) + b"short",
        _opcode_list(_opcode(1, _warp_payload()), trailing=b"x"),
    ],
)
def test_truncated_or_trailing_opcode_list_raises(listing, tmp_path):
    directory = {"tags": _metadata_tags("<", opcode_list=listing)}
    path, _offsets = _write_tiff(tmp_path, [directory])
    with pytest.raises(DNGWarpError):
        prepared.parse_acr_warp_prepared(path)


def test_metadata_is_taken_from_the_ifd_containing_the_opcode(tmp_path):
    endian = "<"
    root = {"tags": _metadata_tags(endian, width=40, height=30, active_area=(0, 0, 30, 40)),
            "next": 1}
    child = {
        "tags": _metadata_tags(
            endian,
            width=100,
            height=80,
            orientation=6,
            active_area=(1, 2, 79, 99),
            scale=(7, 3, 11, 4),
            opcode_list=_opcode_list(_opcode(1, _warp_payload(4))),
        )
    }
    path, offsets = _write_tiff(tmp_path, [root, child])

    result = prepared.parse_acr_warp_prepared(path)

    assert result is not None
    assert result.ifd_offset == offsets[1]
    assert (result.image_width, result.image_height) == (100, 80)
    assert result.active_area == (1, 2, 79, 99)
    assert result.orientation == 6
    assert (result.default_scale_h, result.default_scale_v) == pytest.approx((7 / 3, 11 / 4))
    assert result.planes == 4


def test_subifd_is_scanned_and_cycle_is_rejected(tmp_path):
    root = {"tags": _metadata_tags("<", width=40, height=30,
                                    active_area=(0, 0, 30, 40)), "sub_ifds": [1]}
    child = {"tags": _metadata_tags("<", opcode_list=_opcode_list(_opcode(1, _warp_payload())))}
    path, offsets = _write_tiff(tmp_path, [root, child])
    result = prepared.parse_acr_warp_prepared(path)
    assert result is not None and result.ifd_offset == offsets[1]

    cyclic = {"tags": _metadata_tags("<", opcode_list=_opcode_list(_opcode(1, _warp_payload()))),
              "next": 0}
    cyclic_path, _offsets = _write_tiff(tmp_path, [cyclic], name="cycle.dng")
    with pytest.raises(DNGWarpError, match="cycle"):
        prepared.parse_acr_warp_prepared(cyclic_path)


def test_ifd_scan_is_bounded_to_32_directories(tmp_path):
    directories = [{"tags": [], "next": index + 1 if index < 32 else None}
                   for index in range(33)]
    path, _offsets = _write_tiff(tmp_path, directories)
    with pytest.raises(DNGWarpError, match="too many directories"):
        prepared.parse_acr_warp_prepared(path)


def test_unreadable_file_raises_and_non_tiff_returns_none(tmp_path):
    with pytest.raises(DNGWarpError, match="Unable to read"):
        prepared.parse_acr_warp_prepared(tmp_path / "missing.dng")
    non_tiff = tmp_path / "not-tiff.dng"
    non_tiff.write_bytes(b"not a tiff")
    assert prepared.parse_acr_warp_prepared(non_tiff) is None
