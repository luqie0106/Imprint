from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import acr_profile_metadata as metadata
import camera_profile as cp
import dng_writer
from test_acr_huesatmap_profile import (
    PROFILE_NAME, _append_full_profile_parts, _hsm_tags, _minimal_profile_dng, _tag,
)
from test_acr_profile_metadata import _build_tiff, _entry, _profile_name


NEUTRAL_PAIRS = ((123456789, 987654321), (4000000001, 3999999999), (17, 11))


def _neutral_tag(pairs=NEUTRAL_PAIRS, *, kind=5):
    data = struct.pack("<" + ("i" if kind == 10 else "I") * (2 * len(pairs)),
                       *(value for pair in pairs for value in pair))
    return dng_writer._Tag(50728, kind, len(pairs), data)


def _parse(extra=(), *, source=True, extended=True):
    tags = _hsm_tags()
    if not extended:
        tags = _append_full_profile_parts([
            tag for tag in tags if tag.code not in (50937, 50938, 50939)
        ])
    return cp._parse_profile(
        tags + list(extra), PROFILE_NAME, ("source-neutral-test",),
        _acr_extended=extended, _source_dng=source,
    )


@pytest.fixture
def mock_preparation(monkeypatch):
    # Metadata binding tests exercise the 75 inputs without a native library.
    monkeypatch.setattr(cp, "native_acr_illuminant_temperatures",
                        lambda _codes: np.asarray([2850.0, 6500.0]))
    monkeypatch.setattr(cp, "native_acr_prepare_analog_balance",
                        lambda raw: np.ones(3) if raw is None else raw.copy())
    monkeypatch.setattr(cp, "native_acr_prepare_camera_calibration", lambda raw: raw.copy())


@pytest.mark.parametrize("signatures_match", [False, True])
def test_source_neutral_preserves_rational_doubles_and_owns_native_inputs(
    mock_preparation, signatures_match,
):
    neutral_tag = _neutral_tag()
    signatures = [
        dng_writer._Tag(50931, 2, 2, b"A\0"),
        dng_writer._Tag(50932, 2, 2, b"A\0" if signatures_match else b"B\0"),
    ]
    profile = _parse([neutral_tag, *signatures])
    expected = np.asarray([numerator / denominator for numerator, denominator in NEUTRAL_PAIRS])
    assert profile.source_calibration_enabled
    assert profile.source_calibration_matches == signatures_match
    assert profile.source_camera_neutral.dtype == np.float64
    assert not profile.source_camera_neutral.flags.writeable
    np.testing.assert_array_equal(profile.source_camera_neutral, expected)
    with pytest.raises(ValueError, match="read-only"):
        profile.source_camera_neutral[0] = 1.0

    caller = np.asarray([0.7, 1.0, 0.3], dtype=np.float64)
    before = caller.copy()
    controls = cp._acr_profile_camera_controls(profile, caller)

    assert controls.shape == (75,)
    np.testing.assert_array_equal(controls[64:67], expected)
    np.testing.assert_array_equal(controls[67:75], np.zeros(8))
    np.testing.assert_array_equal(caller, before)
    np.testing.assert_array_equal(profile.source_camera_neutral, expected)
    assert neutral_tag.data == _neutral_tag().data
    assert controls[66] > 1.0  # Neither max normalization nor 1e6 rounding in Python.
    assert not np.array_equal(expected, np.round(expected, 6))


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("extended", [False, True])
def test_only_extended_same_file_source_parses_neutral(source, extended):
    profile = _parse([_neutral_tag()], source=source, extended=extended)
    assert profile.source_calibration_enabled == (source and extended)
    assert (profile.source_camera_neutral is not None) == (source and extended)


@pytest.mark.parametrize("source", [False, True])
def test_missing_neutral_keeps_caller_fallback_without_using_white_xy(mock_preparation, source):
    # AsShotWhiteXY is not a substitute for the neutral-only native factory.
    profile = _parse([_tag(50729, [0.3127, 0.3290])], source=source)
    assert profile.source_camera_neutral is None
    caller = np.asarray([0.3, 1.7, 0.45])
    controls = cp._acr_profile_camera_controls(profile, caller)
    np.testing.assert_array_equal(controls[64:67], caller)


def test_adjacent_neutral_and_ungated_field_cannot_replace_caller(mock_preparation):
    caller = np.asarray([0.8, 1.25, 0.6])
    profile = _parse([_neutral_tag(), _tag(50879, [1], kind=3)], source=False)
    assert profile.source_camera_neutral is None
    np.testing.assert_array_equal(cp._acr_profile_camera_controls(profile, caller)[64:67], caller)

    # Even a manually supplied field must honor the source calibration contract.
    profile = replace(profile, source_camera_neutral=np.asarray([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(cp._acr_profile_camera_controls(profile, caller)[64:67], caller)


INVALID_NEUTRALS = [
    pytest.param(_tag(50728, [1, 1, 1], kind=4), "RATIONAL", id="wrong-kind"),
    pytest.param(_neutral_tag(((1, 1), (2, 1))), "wrong size", id="wrong-count"),
    pytest.param(_neutral_tag(()), "wrong size", id="empty-count"),
    pytest.param(dng_writer._Tag(50728, 5, 3, b"\0" * 23), None, id="truncated-payload"),
    pytest.param(dng_writer._Tag(50728, 5, 3, _neutral_tag().data[:16]),
                 "incomplete", id="incomplete-payload"),
    pytest.param(dng_writer._Tag(50728, 5, 3, _neutral_tag().data + struct.pack("<II", 1, 1)),
                 "incomplete", id="extra-payload"),
    pytest.param(_neutral_tag(((1, 1), (1, 0), (1, 1))), "zero rational denominator", id="zero-denom"),
    pytest.param(_neutral_tag(((1, 1), (0, 1), (1, 1))), "positive", id="zero-value"),
    # Negative/non-finite numbers cannot be encoded as unsigned RATIONALs;
    # their actual TIFF types must be rejected instead of silently coerced.
    pytest.param(_neutral_tag(((1, 1), (-1, 1), (1, 1)), kind=10), "RATIONAL", id="negative-srational"),
    pytest.param(_tag(50728, [1, np.nan, 1]), "RATIONAL", id="nan-float"),
    pytest.param(_tag(50728, [1, np.inf, 1]), "RATIONAL", id="infinite-float"),
]


@pytest.mark.parametrize("tag,message", INVALID_NEUTRALS)
def test_invalid_neutral_rejected_only_as_source_metadata(tag, message):
    with pytest.raises(ValueError, match=message):
        _parse([tag])
    assert _parse([tag], source=False).source_camera_neutral is None
    assert _parse([tag], extended=False).source_camera_neutral is None


INVALID_REFERENCES = [
    pytest.param(_tag(50879, [1], kind=3), "scene-referred", id="output-referred"),
    pytest.param(_tag(50879, [65535], kind=3), "scene-referred", id="unknown-value"),
    pytest.param(_tag(50879, [0], kind=4), "SHORT", id="wrong-kind"),
    pytest.param(_tag(50879, [0, 0], kind=3), "wrong size", id="wrong-count"),
    pytest.param(_tag(50879, [], kind=3), "wrong size", id="empty-count"),
    pytest.param(dng_writer._Tag(50879, 3, 1, b"\0"), None, id="truncated-payload"),
    pytest.param(dng_writer._Tag(50879, 3, 1, b"\0" * 4), "incomplete", id="extra-payload"),
    pytest.param(_tag(50879, [np.nan]), "SHORT", id="nan-float"),
]


@pytest.mark.parametrize("tag,message", INVALID_REFERENCES)
@pytest.mark.parametrize("has_neutral", [False, True])
def test_source_requires_scene_reference_even_without_neutral(tag, message, has_neutral):
    tags = [tag] + ([_neutral_tag()] if has_neutral else [])
    with pytest.raises(ValueError, match=message):
        _parse(tags)
    assert _parse(tags, source=False).source_camera_neutral is None
    assert _parse(tags, extended=False).source_camera_neutral is None


@pytest.mark.parametrize("tags,source_valid", [
    ([], True),
    ([_neutral_tag()], True),
    ([_neutral_tag(), _tag(50879, [0], kind=3)], True),
    ([_neutral_tag(), _tag(50879, [1], kind=3)], False),
    ([_tag(50879, [1], kind=3)], False),
    ([_tag(50879, [0], kind=4)], False),
    ([_neutral_tag(((1, 1), (1, 0), (1, 1)))], False),
    ([_neutral_tag(((1, 1), (0, 1), (1, 1)))], False),
    ([_neutral_tag(((1, 1), (1, 1)))], False),
    ([_tag(50728, [1, np.nan, 1])], False),
])
def test_resolver_binds_same_file_and_preserves_nef_reference_route(
    tmp_path, monkeypatch, tags, source_valid,
):
    source = tmp_path / "source.DNG"
    source.write_bytes(_minimal_profile_dng(_hsm_tags() + tags))
    raw = tmp_path / "source.NEF"
    raw.write_bytes(b"synthetic source identity")
    monkeypatch.setattr(cp, "camera_profile_names", lambda _path: {"CameraProfile": PROFILE_NAME})
    reference_calls = []

    def reference(path, _info):
        reference_calls.append(path)
        return source

    monkeypatch.setattr(cp, "matching_embedded_profile_dng", reference)
    profile = cp.resolve_acr_profile(source, {})
    assert (profile is not None) == source_valid
    assert reference_calls == []
    if profile is not None:
        assert profile.source_calibration_enabled
        assert (profile.source_camera_neutral is not None) == any(tag.code == 50728 for tag in tags)
        assert profile.fingerprint[0] == "acr-huesatmap-v5-source-neutral"

    adjacent = cp.resolve_acr_profile(raw, {})
    assert adjacent is not None
    assert adjacent.acr_only
    assert not adjacent.source_calibration_enabled
    assert adjacent.source_camera_neutral is None
    assert reference_calls == [raw.resolve()]


def test_embedded_no_name_resolution_keeps_source_neutral(tmp_path, monkeypatch):
    source = tmp_path / "embedded.dng"
    source.write_bytes(_minimal_profile_dng(_hsm_tags() + [_neutral_tag()]))
    monkeypatch.setattr(cp, "camera_profile_names", lambda _path: {})
    profile = cp.resolve_acr_profile(source, {})
    assert profile is not None and profile.source_calibration_enabled
    np.testing.assert_array_equal(profile.source_camera_neutral,
                                  [numerator / denominator for numerator, denominator in NEUTRAL_PAIRS])
    assert "acr-huesatmap-v5-source-neutral" in cp.PROFILE_PREVIEW_VERSION


@pytest.mark.parametrize("code", [50728, 50879])
def test_reader_retains_source_tag_bytes_in_fingerprint(tmp_path, code):
    path = tmp_path / "fingerprint.dng"
    tags = ([_neutral_tag(((1, 2), (1, 1), (3, 4))), _neutral_tag(((2, 4), (2, 2), (6, 8)))]
            if code == 50728 else [_tag(50879, [0], kind=3), _tag(50879, [1], kind=3)])
    fingerprints = []
    for tag in tags:
        path.write_bytes(_minimal_profile_dng(_hsm_tags() + [tag]))
        loaded = metadata.read_acr_profile_tags(path, PROFILE_NAME)
        loaded_tag = next(item for item in loaded if item.code == code)
        assert (loaded_tag.kind, loaded_tag.count, loaded_tag.data) == (tag.kind, tag.count, tag.data)
        # Hold every file identity fixed: the raw tag payload must affect the digest.
        fingerprints.append(cp._fingerprint(path, (1, 2, 3, 4), loaded, str(path), (1, 2, 3, 4), ()))
    assert fingerprints[0][2] != fingerprints[1][2]


def test_reader_does_not_follow_other_ifds_or_read_white_xy(tmp_path):
    path = tmp_path / "root-only.dng"
    entries = [
        _profile_name(),
        _entry(330, dng_writer.LONG, struct.pack("<I", 0xFFFFFFF0)),
        _entry(34665, dng_writer.LONG, struct.pack("<I", 0xFFFFFFF0)),
        (50729, 65000, 0xFFFFFFFF, 0xFFFFFFF0),
    ]
    payload = bytearray(_build_tiff(entries))
    struct.pack_into("<I", payload, 8 + 2 + 12 * len(entries), 0xFFFFFFF0)
    path.write_bytes(payload)
    assert {tag.code for tag in metadata.read_embedded_acr_profile_tags(path)} == {50936}


@pytest.mark.parametrize("code,kind,count", [(50728, 5, 3), (50879, 3, 3)])
@pytest.mark.parametrize("failure", ["duplicate", "bounds", "size"])
def test_source_tags_keep_reader_duplicate_bounds_and_size_checks(tmp_path, code, kind, count, failure):
    if failure == "duplicate":
        entry = _entry(code, kind, b"\0" * (dng_writer.TYPE_SIZES[kind] * count))
        entries = [entry, entry]
        message = "duplicate"
    elif failure == "bounds":
        entries = [(code, kind, count, 0xFFFFFFF0)]
        message = "outside the file"
    else:
        entries = [(code, kind, 16 * 1024 * 1024, 0xFFFFFFF0)]
        message = "size limit"
    path = tmp_path / "bad-source-tag.dng"
    path.write_bytes(_build_tiff([_profile_name(), *entries]))
    with pytest.raises(ValueError, match=message):
        metadata.read_embedded_acr_profile_tags(path)
