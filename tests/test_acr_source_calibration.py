from __future__ import annotations

from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import camera_profile as cp
import acr_profile_metadata as metadata
from dng_writer import _Tag

NAME = 'Source calibration test'


def numeric(code, values, kind=11):
    values = np.asarray(values).reshape(-1)
    if kind in (5, 10):
        parts = np.column_stack((np.rint(values * 1000000).astype(np.int64),
                                 np.full(values.size, 1000000, dtype=np.int64)))
        data = parts.astype('<i4' if kind == 10 else '<u4').tobytes()
    else:
        data = values.astype('<f4').tobytes()
    return _Tag(code, kind, values.size, data)


def ascii_tag(code, value):
    return _Tag(code, 2, len(value), value)


def tags():
    return [ascii_tag(50936, NAME.encode() + b'\0'),
            numeric(50721, np.eye(3)),
            numeric(50940, [0., 0., 1., 1.]),
            numeric(50727, [2., 3., 6.], 5),
            numeric(50723, [[1., .012345, 0.], [0., 1., 0.], [0., 0., 1.]], 10)]


def parse(values):
    return cp._parse_profile(values, NAME, ('test',), _acr_extended=True, _source_dng=True)


@pytest.mark.parametrize('source_sig,profile_sig,match', [
    (None, None, True), (b'', None, True), (b'\0', b'', True),
    (b'Camera-A\0', b'Camera-A\0', True),
    (b'Camera-A\0ignored', b'Camera-A', True),
    (b'Camera-A', b'camera-a', False), (b'Camera-A ', b'Camera-A', False),
    (None, b'Camera-A', False), (b'Camera-A', None, False),
])
def test_source_signature_controls_calibration(source_sig, profile_sig, match):
    values = tags()
    if source_sig is not None:
        values.append(ascii_tag(50931, source_sig))
    if profile_sig is not None:
        values.append(ascii_tag(50932, profile_sig))
    profile = parse(values)
    assert profile.source_calibration_enabled
    assert profile.source_calibration1 is not None
    assert profile.source_calibration_matches == match
    assert profile.source_calibration2 is None
    assert not profile.source_analog_balance.flags.writeable
    np.testing.assert_array_equal(profile.source_analog_balance, [2., 3., 6.])
    assert not profile.source_calibration1.flags.writeable
    assert profile.source_calibration1[0, 1] == .012345


def test_reference_or_legacy_cannot_supply_nonidentity_source_metadata():
    for extended in (False, True):
        with pytest.raises(ValueError):
            cp._parse_profile(tags(), NAME, ('reference',), _acr_extended=extended)


def test_native_source_preparation_owns_math(monkeypatch):
    profile = parse(tags())
    seen = []
    def analog(raw):
        np.testing.assert_array_equal(raw, [2., 3., 6.])
        seen.append('analog')
        return np.array([1., 1.5, 3.])
    def calibration(raw):
        assert raw[0, 1] == .012345  # remains unquantized in metadata
        seen.append('calibration')
        result = np.eye(3)
        result[0, 1] = .0123
        return result
    monkeypatch.setattr(cp, 'native_acr_prepare_analog_balance', analog)
    monkeypatch.setattr(cp, 'native_acr_prepare_camera_calibration', calibration)
    controls = cp._acr_profile_camera_controls(profile, np.ones(3), np.array([0., 0.]))
    assert seen == ['analog', 'calibration']
    np.testing.assert_array_equal(controls[:3], [1., 1.5, 3.])
    assert controls[22] == .0123
    np.testing.assert_array_equal(controls[30:39], np.eye(3).reshape(-1))


def test_mismatched_matrix_is_quantized_before_signature_selection(monkeypatch):
    profile = parse(tags() + [ascii_tag(50931, b'A'), ascii_tag(50932, b'B')])
    seen = []
    monkeypatch.setattr(cp, 'native_acr_prepare_analog_balance', lambda _raw: np.ones(3))
    def quantize(raw):
        seen.append(raw.copy())
        return np.eye(3) * 2
    monkeypatch.setattr(cp, 'native_acr_prepare_camera_calibration', quantize)
    controls = cp._acr_profile_camera_controls(profile, np.ones(3), np.array([0., 0.]))
    assert len(seen) == 1 and seen[0][0, 1] == .012345
    np.testing.assert_array_equal(controls[21:39], np.tile(np.eye(3).reshape(-1), 2))


@pytest.mark.parametrize('code', [50723, 50727])
def test_source_metadata_wrong_numeric_type_rejected(code):
    values = [tag for tag in tags() if tag.code != code]
    values.append(numeric(code, np.eye(3) if code == 50723 else [2, 3, 6]))
    with pytest.raises(ValueError):
        parse(values)


def test_nonascii_signature_not_silently_transcoded():
    with pytest.raises(ValueError, match='non-ASCII'):
        parse(tags() + [ascii_tag(50931, b'\xff')])


def write_tiff(path, values):
    ordered = sorted(values, key=lambda tag: tag.code)
    payload = bytearray()
    entries = bytearray()
    start = 8 + 2 + 12 * len(ordered) + 4
    for tag in ordered:
        data = bytes(tag.data)
        value = data.ljust(4, b'\0') if len(data) <= 4 else struct.pack('<I', start + len(payload))
        entries.extend(struct.pack('<HHI', tag.code, tag.kind, tag.count) + value)
        if len(data) > 4:
            payload.extend(data)
    path.write_bytes(b'II*\0' + struct.pack('<I', 8) + struct.pack('<H', len(ordered))
                     + entries + b'\0' * 4 + payload)


def test_reader_retains_empty_calibration_signatures(tmp_path):
    path = tmp_path / 'source.dng'
    write_tiff(path, tags() + [ascii_tag(50931, b''), ascii_tag(50932, b'\0')])
    loaded = metadata.read_acr_profile_tags(path, NAME)
    signatures = {tag.code: tag.data for tag in loaded if tag.code in (50931, 50932)}
    assert signatures == {50931: b'', 50932: b'\0'}
    assert parse(loaded).source_calibration1 is not None


def test_resolver_enables_only_source_dng_calibration(tmp_path, monkeypatch):
    source = tmp_path / 'source.dng'
    write_tiff(source, tags())
    monkeypatch.setattr(cp, 'camera_profile_names', lambda _path: {'CameraProfile': NAME})
    profile = cp.resolve_acr_profile(source, {})
    assert profile is not None and profile.source_calibration_enabled
    raw = tmp_path / 'other.NEF'
    raw.write_bytes(b'not an actual photo')
    monkeypatch.setattr(cp, 'matching_embedded_profile_dng', lambda *_args: source)
    assert cp.resolve_acr_profile(raw, {}) is None
