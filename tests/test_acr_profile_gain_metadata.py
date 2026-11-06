from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import camera_profile as cp
import dng_writer


class Tag:
    def __init__(self, code: int, kind: int, count: int, data: bytes):
        self.code = code
        self.kind = kind
        self.count = count
        self.data = data


def _entry(code: int, kind: int, data: bytes, *, count: int | None = None):
    if count is None:
        count = len(data) // dng_writer.TYPE_SIZES[kind]
    return code, kind, count, data


def _build_tiff(entries: list[tuple[int, int, int, bytes]]) -> bytes:
    """Build a small little-endian classic TIFF with root-IFD entries."""
    root_offset = 8
    payload_offset = root_offset + 2 + 12 * len(entries) + 4
    encoded_entries = bytearray()
    external = bytearray()
    for code, kind, count, data in entries:
        if len(data) <= 4:
            value = data.ljust(4, b"\x00")
        else:
            value = struct.pack("<I", payload_offset)
            external.extend(data)
            payload_offset += len(data)
        encoded_entries.extend(struct.pack("<HHI", code, kind, count))
        encoded_entries.extend(value)
    directory = struct.pack("<H", len(entries)) + encoded_entries + struct.pack("<I", 0)
    return b"II*\x00" + struct.pack("<I", root_offset) + directory + external


def _base_profile_tags() -> list[Tag]:
    name = b"Selected Camera Profile\x00"
    identity = np.eye(3, dtype="<f4").reshape(-1).tobytes()
    curve = np.asarray([0.0, 0.0, 1.0, 1.0], dtype="<f4").tobytes()
    look = np.tile(np.asarray([0.0, 1.0, 1.0], dtype="<f4"), 4 * 2 * 2).tobytes()
    return [
        Tag(50936, dng_writer.ASCII, len(name), name),
        Tag(50721, dng_writer.FLOAT, 9, identity),
        Tag(50722, dng_writer.FLOAT, 9, identity),
        Tag(50964, dng_writer.FLOAT, 9, identity),
        Tag(50965, dng_writer.FLOAT, 9, identity),
        Tag(50778, dng_writer.SHORT, 1, struct.pack("<H", 17)),
        Tag(50779, dng_writer.SHORT, 1, struct.pack("<H", 21)),
        Tag(50940, dng_writer.FLOAT, 4, curve),
        Tag(50981, dng_writer.LONG, 3, struct.pack("<III", 4, 2, 2)),
        Tag(50982, dng_writer.FLOAT, 48, look),
        Tag(51108, dng_writer.LONG, 1, struct.pack("<I", 0)),
    ]


def _map_tag(payload: bytes, *, kind: int = dng_writer.UNDEFINED,
             count: int | None = None) -> Tag:
    return Tag(52544, kind, len(payload) if count is None else count, payload)


def _valid_map_payload() -> bytes:
    header = struct.pack(
        "<II4dI5fI3f", 1, 1, 1.0, 1.0, 0.0, 0.0, 1,
        0.03, 0.07, 0.02, 0.01, 0.04, 3, 1.0, 0.25, 2.25,
    )
    return header + struct.pack("<f", 1.0)


def _write_profile_reference(path: Path, payload: bytes) -> None:
    entries = [
        _entry(tag.code, tag.kind, tag.data, count=tag.count)
        for tag in _base_profile_tags()
    ]
    entries.append(_entry(52544, dng_writer.UNDEFINED, payload))
    path.write_bytes(_build_tiff(entries))


def test_bounded_root_ifd_reads_selected_profile_map2_only_and_preserves_source(tmp_path: Path) -> None:
    profile_name = b"Selected Camera Profile\x00"
    payload = bytes(range(1, 33))
    source_map = b"source-ifd-map"
    source = tmp_path / "selected-profile.dng"
    original = _build_tiff([
        _entry(50936, dng_writer.ASCII, profile_name),
        _entry(52544, dng_writer.UNDEFINED, payload),
        _entry(52525, dng_writer.UNDEFINED, source_map),
    ])
    source.write_bytes(original)
    before = source.stat()

    tags = cp.read_acr_profile_tags(source, "Selected Camera Profile")

    assert {tag.code for tag in tags} == {50936, 52544}
    map_tag = next(tag for tag in tags if tag.code == 52544)
    assert map_tag.kind == dng_writer.UNDEFINED
    assert map_tag.count == len(payload)
    assert map_tag.data == payload
    assert source.read_bytes() == original
    after = source.stat()
    assert (after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
        before.st_size, before.st_mtime_ns, before.st_ctime_ns,
    )


def test_acr_extended_profile_retains_immutable_payload_after_native_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"serialized selected map payload"
    observed: list[tuple[bytes, int, int]] = []
    monkeypatch.setattr(
        cp, "native_acr_validate_profile_gain_payload",
        lambda data, *, tag, byte_order: observed.append((data, tag, byte_order)),
    )

    profile = cp._parse_profile(
        [*_base_profile_tags(), _map_tag(payload)], "Selected Camera Profile", ("test",),
        _acr_extended=True,
    )

    assert observed == [(payload, 52544, 0)]
    assert profile.profile_gain_table_map2_payload == payload
    assert type(profile.profile_gain_table_map2_payload) is bytes
    with pytest.raises(FrozenInstanceError):
        profile.profile_gain_table_map2_payload = b"changed"  # type: ignore[misc]


def test_no_map_does_not_call_native_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cp, "native_acr_validate_profile_gain_payload",
        lambda *_args, **_kwargs: pytest.fail("validator called without tag 52544"),
    )

    profile = cp._parse_profile(
        _base_profile_tags(), "Selected Camera Profile", ("test",), _acr_extended=True,
    )

    assert profile.profile_gain_table_map2_payload is None


@pytest.mark.parametrize(
    ("tag", "message"),
    [
        (_map_tag(b"bad kind", kind=dng_writer.BYTE), "invalid selected-profile"),
        (_map_tag(b"wrong count", count=1), "invalid selected-profile"),
        (_map_tag(b"", count=0), "invalid selected-profile"),
    ],
)
def test_invalid_profile_map_tag_is_rejected_before_native_validation(
    monkeypatch: pytest.MonkeyPatch, tag: Tag, message: str,
) -> None:
    monkeypatch.setattr(
        cp, "native_acr_validate_profile_gain_payload",
        lambda *_args, **_kwargs: pytest.fail("invalid tag reached native validator"),
    )

    with pytest.raises(ValueError, match=message):
        cp._parse_profile(
            [*_base_profile_tags(), tag], "Selected Camera Profile", ("test",),
            _acr_extended=True,
        )


def test_native_payload_rejection_is_not_bypassed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"\x00" * 80

    def reject(data: bytes, *, tag: int, byte_order: int) -> None:
        assert (data, tag, byte_order) == (payload, 52544, 0)
        raise ValueError("malformed serialized map")

    monkeypatch.setattr(cp, "native_acr_validate_profile_gain_payload", reject)

    with pytest.raises(ValueError, match="malformed serialized map"):
        cp._parse_profile(
            [*_base_profile_tags(), _map_tag(payload)], "Selected Camera Profile", ("test",),
            _acr_extended=True,
        )


def test_legacy_profile_path_does_not_enable_or_validate_map_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"legacy should ignore this"
    monkeypatch.setattr(
        cp, "native_acr_validate_profile_gain_payload",
        lambda *_args, **_kwargs: pytest.fail("legacy path called native validator"),
    )

    profile = cp._parse_profile(
        [*_base_profile_tags(), _map_tag(payload)], "Selected Camera Profile", ("test",),
        _acr_extended=False,
    )

    assert not profile.acr_only
    assert profile.profile_gain_table_map2_payload is None


def test_profile_default_construction_keeps_optional_payload_empty() -> None:
    identity = np.eye(3, dtype=np.float64)
    profile = cp.Profile(
        name="Selected Camera Profile", fingerprint=("test",),
        color_matrix1=identity, color_matrix2=identity,
        forward_matrix1=identity, forward_matrix2=identity,
        tone_curve=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        look_table=None, look_dims=(0, 0, 0), look_encoding=0,
        baseline_exposure_offset=0.0,
    )
    assert profile.profile_gain_table_map2_payload is None


def test_bounded_tiff_profile_payload_is_validated_by_cxx_and_malformed_payload_rejected(
    tmp_path: Path,
) -> None:
    profile_name = b"Selected Camera Profile\x00"
    payload = _valid_map_payload()
    reference = tmp_path / "native-validation.dng"
    reference.write_bytes(_build_tiff([
        _entry(50936, dng_writer.ASCII, profile_name),
        _entry(52544, dng_writer.UNDEFINED, payload),
    ]))
    read_tags = cp.read_acr_profile_tags(reference, "Selected Camera Profile")
    profile_tags = [tag for tag in _base_profile_tags() if tag.code != 50936]

    profile = cp._parse_profile(
        [*profile_tags, *read_tags], "Selected Camera Profile", ("native-validation",),
        _acr_extended=True,
    )

    assert profile.profile_gain_table_map2_payload == payload
    with pytest.raises(ValueError, match="rejected"):
        cp._parse_profile(
            [*_base_profile_tags(), _map_tag(b"\x00" * 80)],
            "Selected Camera Profile", ("malformed-native-validation",), _acr_extended=True,
        )


def test_optional_resolver_falls_back_on_native_error_but_propagates_other_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nef"
    reference = tmp_path / "selected-profile.dng"
    source.write_bytes(b"raw fixture")
    _write_profile_reference(reference, _valid_map_payload())
    monkeypatch.setattr(cp, "camera_profile_names", lambda _path: {})
    monkeypatch.setattr(cp, "matching_embedded_profile_dng", lambda *_args: reference)
    source_info = {"SourceCameraProfileName": "Selected Camera Profile"}
    reference_tags = cp.read_acr_profile_tags(reference, "Selected Camera Profile")

    def missing_native_symbol(*_args, **_kwargs):
        raise cp.NativeRendererError("Native library does not provide PGTM payload validation")

    monkeypatch.setattr(cp, "native_acr_validate_profile_gain_payload", missing_native_symbol)
    with pytest.raises(cp.NativeRendererError, match="does not provide"):
        cp._parse_profile(
            reference_tags, "Selected Camera Profile", ("direct-parse",), _acr_extended=True,
        )

    cp._resolve_cached.cache_clear()
    try:
        assert cp.resolve_acr_profile(source, source_info) is None

        cp._resolve_cached.cache_clear()

        def unexpected_runtime_error(*_args, **_kwargs):
            raise RuntimeError("unexpected test runtime error")

        monkeypatch.setattr(cp, "native_acr_validate_profile_gain_payload", unexpected_runtime_error)
        with pytest.raises(RuntimeError, match="unexpected test runtime error"):
            cp.resolve_acr_profile(source, source_info)
    finally:
        cp._resolve_cached.cache_clear()
