from __future__ import annotations

from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import acr_profile_metadata as metadata
import dng_writer


Entry = tuple[int, int, int, bytes | int]


def _entry(code: int, kind: int, data: bytes, *, count: int | None = None) -> Entry:
    return code, kind, len(data) // dng_writer.TYPE_SIZES[kind] if count is None else count, data


def _build_tiff(
    entries: list[Entry],
    *,
    header: bytes = b"II*\x00",
    root_offset: int = 8,
    declared_count: int | None = None,
    truncate_after_root_count: bool = False,
) -> bytes:
    """Build a tiny classic-TIFF container; integer values are forced offsets."""
    count = len(entries) if declared_count is None else declared_count
    ifd_end = root_offset + 2 + 12 * len(entries) + 4
    payload_offset = ifd_end
    external = bytearray()
    encoded_entries = bytearray()
    for code, kind, item_count, value in entries:
        if isinstance(value, int):
            value_field = struct.pack("<I", value)
        elif len(value) <= 4:
            value_field = value.ljust(4, b"\x00")
        else:
            value_field = struct.pack("<I", payload_offset)
            external.extend(value)
            payload_offset += len(value)
        encoded_entries.extend(struct.pack("<HHI", code, kind, item_count))
        encoded_entries.extend(value_field)

    prefix = header[:4].ljust(4, b"\x00")
    result = bytearray(max(8, root_offset + 2))
    result[:4] = prefix
    result[4:8] = struct.pack("<I", root_offset)
    if truncate_after_root_count:
        result[root_offset:root_offset + 2] = struct.pack("<H", count)
        return bytes(result)

    directory = struct.pack("<H", count) + encoded_entries + struct.pack("<I", 0)
    result.extend(b"\x00" * max(0, root_offset - len(result)))
    result[root_offset:root_offset + len(directory)] = directory
    result.extend(external)
    return bytes(result)


def _profile_name(name: bytes = b"Camera HSM") -> Entry:
    return 50936, dng_writer.ASCII, len(name) + 1, name + b"\x00"


def _header_with_root_offset(offset: int) -> bytes:
    return b"II*\x00" + struct.pack("<I", offset)


def _codes(tags: list[dng_writer._Tag]) -> set[int]:
    return {tag.code for tag in tags}


def test_reads_hue_sat_map_without_look_table_or_forward_matrix(tmp_path: Path) -> None:
    payload = _build_tiff([
        _profile_name(),
        _entry(50937, dng_writer.LONG, struct.pack("<III", 2, 2, 2)),
        _entry(50938, dng_writer.FLOAT, struct.pack("<fff", 0.0, 1.0, 1.0)),
        _entry(52537, dng_writer.UNDEFINED, b"\x02\x00\x01"),
    ])
    reference = tmp_path / "hsm-only.dng"
    reference.write_bytes(payload)

    tags = metadata.read_acr_profile_tags(reference, "Camera HSM")

    assert _codes(tags) == {50936, 50937, 50938, 52537}
    assert next(tag for tag in tags if tag.code == 50938).data == struct.pack("<fff", 0.0, 1.0, 1.0)


@pytest.mark.parametrize("code", [52529, 52530, 52531, 52532, 52535, 52537])
def test_third_illuminant_markers_reach_the_profile_validator(tmp_path: Path, code: int) -> None:
    reference = tmp_path / "third-illuminant-marker.dng"
    reference.write_bytes(_build_tiff([
        _profile_name(), _entry(code, dng_writer.UNDEFINED, b"\x01"),
    ]))
    tags = metadata.read_acr_profile_tags(reference, "Camera HSM")
    assert code in _codes(tags)
    assert next(tag for tag in tags if tag.code == code).data == b"\x01"


@pytest.mark.parametrize("policy", [None, 0, 1, 2, 0xFFFFFFFF])
def test_copy_policy_value_or_absence_does_not_gate_metadata_reading(
    tmp_path: Path, policy: int | None,
) -> None:
    entries = [_profile_name(), _entry(50721, dng_writer.FLOAT, struct.pack("<f", 1.0))]
    if policy is not None:
        entries.append(_entry(50941, dng_writer.LONG, struct.pack("<I", policy)))
    reference = tmp_path / f"policy-{policy}.dng"
    reference.write_bytes(_build_tiff(entries))

    tags = metadata.read_acr_profile_tags(reference, "Camera HSM")

    assert {50936, 50721} <= _codes(tags)
    if policy is None:
        assert 50941 not in _codes(tags)
    else:
        policy_tag = next(tag for tag in tags if tag.code == 50941)
        assert policy_tag.kind == dng_writer.LONG
        assert policy_tag.data == struct.pack("<I", policy)


def test_profile_name_must_match_and_be_ascii(tmp_path: Path) -> None:
    wrong_name = tmp_path / "wrong-name.dng"
    wrong_name.write_bytes(_build_tiff([_profile_name(b"Camera Other")]))
    with pytest.raises(ValueError, match="does not match"):
        metadata.read_acr_profile_tags(wrong_name, "Camera HSM")

    invalid_ascii = tmp_path / "invalid-name.dng"
    invalid_ascii.write_bytes(_build_tiff([(50936, dng_writer.ASCII, 2, b"\xff\x00")]))
    with pytest.raises(ValueError, match="not ASCII"):
        metadata.read_acr_profile_tags(invalid_ascii, "Camera HSM")


def test_embedded_reader_accepts_its_valid_root_profile_name(tmp_path: Path) -> None:
    reference = tmp_path / "embedded-name.dng"
    reference.write_bytes(_build_tiff([
        _profile_name(b"Embedded"),
        _entry(50937, dng_writer.LONG, struct.pack("<III", 2, 2, 2)),
    ]))

    tags = metadata.read_embedded_acr_profile_tags(reference)

    name_tag = next(tag for tag in tags if tag.code == 50936)
    assert name_tag.data.rstrip(b"\x00").decode("ascii") == "Embedded"
    assert _codes(tags) == {50936, 50937}


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([], "no ASCII profile name"),
        ([(50936, dng_writer.ASCII, 2, b"\xff\x00")], "not ASCII"),
        ([(50936, dng_writer.ASCII, 1, b"\x00")], "name is empty"),
    ],
)
def test_embedded_reader_rejects_missing_or_invalid_root_profile_name(
    tmp_path: Path, entries: list[Entry], message: str,
) -> None:
    reference = tmp_path / "bad-embedded-name.dng"
    reference.write_bytes(_build_tiff(entries))

    with pytest.raises(ValueError, match=message):
        metadata.read_embedded_acr_profile_tags(reference)


@pytest.mark.parametrize("expected_name", [None, 123])
def test_named_reader_rejects_non_string_expected_name(
    tmp_path: Path, expected_name: object,
) -> None:
    reference = tmp_path / "embedded-name.dng"
    reference.write_bytes(_build_tiff([_profile_name(b"Embedded")]))

    with pytest.raises(TypeError, match="expected_name must be a string"):
        metadata.read_acr_profile_tags(reference, expected_name)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_build_tiff([_profile_name()], header=b"MM*\x00"), "little-endian"),
        (_build_tiff([_profile_name()], header=b"II+\x00"), "little-endian"),
        (_header_with_root_offset(4), "root IFD"),
        (_header_with_root_offset(0xFFFFFFF0), "root IFD"),
        (_build_tiff([], declared_count=1, truncate_after_root_count=True), "truncated"),
        (_build_tiff([
            _profile_name(), _entry(50721, dng_writer.BYTE, b"a"),
            _entry(50721, dng_writer.BYTE, b"b"),
        ]), "duplicate"),
        (_build_tiff([_profile_name(), (50721, 65000, 1, b"\x00\x00\x00\x00")]), "unsupported TIFF type"),
        (_build_tiff([_profile_name(), (50721, dng_writer.BYTE, 0, b"\x00")]), "empty profile tag"),
        (_build_tiff([_profile_name(), (50721, dng_writer.BYTE, 5, 0xFFFFFFF0)]), "outside the file"),
    ],
)
def test_rejects_unsupported_or_malformed_root_ifd(payload: bytes, message: str, tmp_path: Path) -> None:
    reference = tmp_path / "malformed.dng"
    reference.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        metadata.read_acr_profile_tags(reference, "Camera HSM")


def test_non_target_entry_type_and_offset_are_not_parsed(tmp_path: Path) -> None:
    reference = tmp_path / "unselected-tag.dng"
    reference.write_bytes(_build_tiff([
        _profile_name(),
        (65000, 65000, 0xFFFFFFFF, 0xFFFFFFF0),
    ]))

    tags = metadata.read_acr_profile_tags(reference, "Camera HSM")

    assert _codes(tags) == {50936}


def test_rejects_more_than_4096_root_entries(tmp_path: Path) -> None:
    reference = tmp_path / "too-many-entries.dng"
    reference.write_bytes(_header_with_root_offset(8) + struct.pack("<H", 4097))

    with pytest.raises(ValueError, match="too many entries"):
        metadata.read_acr_profile_tags(reference, "Camera HSM")


def test_rejects_short_out_of_line_payload_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = tmp_path / "short-payload.dng"
    complete = _build_tiff([
        _profile_name(), _entry(50721, dng_writer.BYTE, b"payload"),
    ])
    reference.write_bytes(complete[:-2])
    real_stat = Path.stat

    def report_original_size(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == reference:
            return SimpleNamespace(st_size=result.st_size + 2)
        return result

    monkeypatch.setattr(Path, "stat", report_original_size)

    with pytest.raises(ValueError, match="tag is truncated"):
        metadata.read_acr_profile_tags(reference, "Camera HSM")


def test_rejects_single_target_payload_above_16_mib_without_reading_it(tmp_path: Path) -> None:
    reference = tmp_path / "oversized-tag.dng"
    reference.write_bytes(_build_tiff([
        _profile_name(),
        (50721, dng_writer.BYTE, 16 * 1024 * 1024 + 1, 0xFFFFFFF0),
    ]))

    with pytest.raises(ValueError, match="exceeds the size limit"):
        metadata.read_acr_profile_tags(reference, "Camera HSM")


def test_rejects_total_target_payload_above_64_mib_before_payload_reads(tmp_path: Path) -> None:
    large_count = 16 * 1024 * 1024
    entries = [_profile_name()]
    entries.extend(
        (code, dng_writer.BYTE, large_count, 0xFFFFFFF0)
        for code in (50721, 50722, 50723, 50724)
    )
    entries.append((50727, dng_writer.BYTE, 1, 0xFFFFFFF0))
    reference = tmp_path / "oversized-total.dng"
    reference.write_bytes(_build_tiff(entries))

    with pytest.raises(ValueError, match="total size limit"):
        metadata.read_acr_profile_tags(reference, "Camera HSM")


def test_read_leaves_source_bytes_and_file_metadata_unchanged(tmp_path: Path) -> None:
    reference = tmp_path / "unchanged.dng"
    source = _build_tiff([
        _profile_name(), _entry(50721, dng_writer.BYTE, b"\x01"),
    ])
    reference.write_bytes(source)
    before_stat = reference.stat()
    before_fingerprint = (
        before_stat.st_dev, before_stat.st_ino, before_stat.st_mode,
        before_stat.st_size, before_stat.st_mtime_ns, before_stat.st_ctime_ns,
    )

    metadata.read_acr_profile_tags(reference, "Camera HSM")

    after_stat = reference.stat()
    after_fingerprint = (
        after_stat.st_dev, after_stat.st_ino, after_stat.st_mode,
        after_stat.st_size, after_stat.st_mtime_ns, after_stat.st_ctime_ns,
    )
    assert reference.read_bytes() == source
    assert after_fingerprint == before_fingerprint
