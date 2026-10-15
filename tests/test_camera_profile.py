from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import camera_profile as cp


@dataclass
class Tag:
    code: int
    kind: int
    count: int
    data: bytes


def _tag(code: int, values, *, kind: int = 11) -> Tag:
    array = np.asarray(values)
    if kind == 11:
        data = array.astype("<f4").tobytes()
    elif kind == 4:
        data = array.astype("<u4").tobytes()
    elif kind == 3:
        data = array.astype("<u2").tobytes()
    elif kind == 2:
        data = bytes(values)
    else:
        raise AssertionError(kind)
    return Tag(code, kind, int(array.size) if kind != 2 else len(data), data)


def _valid_tags(name: str = "Camera Standard") -> list[Tag]:
    identity = np.eye(3, dtype=np.float32).reshape(-1)
    forward = cp.PROPHOTO_TO_XYZ50.astype(np.float32).reshape(-1)
    curve = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32).reshape(-1)
    dims = np.asarray([4, 2, 2], dtype=np.uint32)
    look = np.tile(np.asarray([0.0, 1.0, 1.0], dtype=np.float32), 4 * 2 * 2)
    return [
        Tag(50936, 2, len(name) + 1, name.encode("ascii") + b"\0"),
        _tag(50721, identity), _tag(50722, identity),
        _tag(50964, forward), _tag(50965, forward),
        _tag(50778, [17], kind=3), _tag(50779, [21], kind=3),
        _tag(50940, curve), _tag(50981, dims, kind=4), _tag(50982, look),
        _tag(51108, [0], kind=4), _tag(51109, [0.25]),
        _tag(50723, identity), _tag(50724, identity),
    ]


def _profile(**overrides) -> cp.Profile:
    tags = {tag.code: tag for tag in _valid_tags()}
    return cp._parse_profile(list(tags.values()), "Camera Standard", ("test",))


def test_transfer_identity_preserves_camera_exactly() -> None:
    camera = np.arange(6 * 7 * 3, dtype=np.uint16).reshape(6, 7, 3)
    reference = np.full_like(camera, 25000)
    transferred = cp.transfer_enhancement(camera, reference, reference.copy())
    assert np.array_equal(transferred, camera)
    assert transferred is not camera


def test_transfer_enhancement_checks_aligned_uint16_rgb() -> None:
    camera = np.zeros((2, 3, 3), dtype=np.uint16)
    with pytest.raises(ValueError, match="identical shapes"):
        cp.transfer_enhancement(camera, camera[:, :2], camera)
    with pytest.raises(ValueError, match="uint16 RGB"):
        cp.transfer_enhancement(camera.astype(np.float32), camera, camera)


def test_profile_exposure_and_baseline_offset_are_applied_once() -> None:
    profile = _profile()
    sample = np.full((3, 4, 3), 16384, dtype=np.uint16)
    output = cp.render_profile(sample, (1.0, 1.0, 1.0), profile, exposure_ev=1.0)
    linear = (16384.0 / 65535.0) * 2.0 ** (1.0 + profile.baseline_exposure_offset)
    expected = np.clip(np.rint(cp._encode_srgb(np.asarray(linear)) * 255.0), 0, 255).astype(np.uint8)
    assert np.all(output == expected)
    assert output.dtype == np.uint8


def test_look_table_keeps_gray_neutral_and_handles_color_boundaries() -> None:
    base = _profile()
    changed_look = np.array(base.look_table, copy=True)
    changed_look[..., 0] = 83.0
    changed_look[..., 1] = 0.4
    profile = cp.Profile(
        name=base.name, fingerprint=base.fingerprint,
        color_matrix1=base.color_matrix1, color_matrix2=base.color_matrix2,
        forward_matrix1=base.forward_matrix1, forward_matrix2=base.forward_matrix2,
        tone_curve=base.tone_curve, look_table=changed_look,
        look_dims=base.look_dims, look_encoding=base.look_encoding,
        baseline_exposure_offset=base.baseline_exposure_offset,
    )
    colors = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]],
        dtype=np.float32,
    )
    rendered = cp._apply_look(colors, profile)
    assert np.isfinite(rendered).all()
    assert np.allclose(rendered[0, 1], rendered[0, 1, 0], atol=1e-6)
    assert np.allclose(rendered[0, 0], 0.0, atol=1e-6)
    assert np.all((rendered >= 0.0) & (rendered <= 1.0))


def test_renderer_rejects_nonfinite_neutral_and_preserves_input() -> None:
    profile = _profile()
    camera = np.full((2, 2, 3), 8192, dtype=np.uint16)
    before = camera.copy()
    with pytest.raises(ValueError, match="neutral"):
        cp.render_profile(camera, (1.0, float("nan"), 1.0), profile, exposure_ev=0.0)
    assert np.array_equal(camera, before)
    with pytest.raises(ValueError, match="exposure_ev"):
        cp.render_profile(camera, (1.0, 1.0, 1.0), profile, exposure_ev=float("inf"))


@pytest.mark.parametrize("invalid_kind", ["calibration", "huesat", "illuminant", "tone"])
def test_unsupported_profile_falls_back_to_none(tmp_path, monkeypatch, invalid_kind) -> None:
    source = tmp_path / "source.nef"
    candidate = tmp_path / "camera.dng"
    source.write_bytes(b"raw")
    candidate.write_bytes(b"profile")
    tags = _valid_tags()
    if invalid_kind == "calibration":
        tags = [tag for tag in tags if tag.code != 50723]
        tags.append(_tag(50723, [1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
    elif invalid_kind == "huesat":
        tags.extend([
            _tag(50937, [1, 1, 1], kind=4),
            _tag(50938, [0.25, 1.0, 1.0]),
            _tag(50939, [0.0, 1.0, 1.0]),
        ])
    elif invalid_kind == "illuminant":
        tags = [tag for tag in tags if tag.code != 50779]
        tags.append(_tag(50779, [23], kind=3))
    else:
        tags = [tag for tag in tags if tag.code != 50940]
        tags.append(_tag(50940, [0.0, 0.2, 1.0, 1.0]))

    monkeypatch.setattr(cp, "camera_profile_names", lambda _path: {
        "SourceCameraProfileName": "Camera Standard",
    })
    monkeypatch.setattr(cp, "matching_embedded_profile_dng", lambda *_args: candidate)
    monkeypatch.setattr(cp, "_embedded_profile_tags", lambda *_args: tags)
    assert cp.resolve_profile(source, {"Make": "Nikon", "Model": "Z"}) is None


def test_enhanced_rgb_subifd_is_not_accepted_as_camera_raw(tmp_path) -> None:
    source = tmp_path / "enhanced.dng"
    # Classic little-endian TIFF: IFD0 points to one SubIFD whose NewSubfileType
    # is 16 (Enhanced Image Data). Pixel strips are intentionally absent.
    root = struct.pack("<H", 1) + struct.pack("<HHII", 330, 4, 1, 26) + struct.pack("<I", 0)
    child = struct.pack("<H", 1) + struct.pack("<HHII", 254, 4, 1, 16) + struct.pack("<I", 0)
    source.write_bytes(b"II*\0" + struct.pack("<I", 8) + root + b"\0" * (26 - 26) + child)
    assert cp._has_enhanced_rgb_subifd(source)


def test_resolve_profile_rejects_enhanced_dng_before_metadata_lookup(tmp_path, monkeypatch) -> None:
    source = tmp_path / "enhanced.dng"
    root = struct.pack("<H", 1) + struct.pack("<HHII", 330, 4, 1, 26) + struct.pack("<I", 0)
    child = struct.pack("<H", 1) + struct.pack("<HHII", 254, 4, 1, 16) + struct.pack("<I", 0)
    source.write_bytes(b"II*\0" + struct.pack("<I", 8) + root + child)
    monkeypatch.setattr(cp, "camera_profile_names", lambda _path: pytest.fail("must reject first"))
    assert cp.resolve_profile(source, {}) is None


def test_real_working_small_matches_profile_trial_when_available() -> None:
    repo = Path(__file__).resolve().parents[1]
    fixture = repo / "artifacts/acr_tone_dng_20261009/real_photo/working_small.npz"
    reference_dng = repo / "test_images/LCR_0166adobe.DNG"
    source_raw = repo / "test_images/LCR_0166.NEF"
    if not (fixture.is_file() and reference_dng.is_file() and source_raw.is_file()):
        pytest.skip("real profile trial data and matching RAW are not available")

    spec = importlib.util.spec_from_file_location(
        "camera_profile_trial",
        repo / "artifacts/acr_tone_dng_20261009/profile_trial/profile_ablation.py",
    )
    if spec is None or spec.loader is None:
        pytest.skip("offline profile trial source is not available")
    trial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trial)

    from dng_writer import _embedded_profile_tags
    import rawpy

    tags = _embedded_profile_tags(reference_dng, "Camera Standard")
    sample = np.load(fixture)
    camera = sample["enhanced_camera_rgb_u16"]
    with rawpy.imread(str(source_raw)) as raw:
        neutral = 1.0 / np.asarray(raw.camera_whitebalance[:3], dtype=np.float64)
        neutral /= neutral[1]
    prof = cp._parse_profile(tags, "Camera Standard", ("real",))
    expected_matrix, white, _ = trial.camera_transform({tag.code: tag for tag in tags}, neutral)
    base = np.clip(np.minimum(camera.astype(np.float32) / 65535.0, white) @ expected_matrix.T, 0.0, 1.0).astype(np.float32)
    offset = float(trial.numeric({tag.code: tag for tag in tags}[51109])[0])
    exposed = np.clip(base * 2.0 ** (2.3 + offset), 0.0, 1.0)
    expected = trial.project(trial.apply_tone(trial.apply_look(exposed, {tag.code: tag for tag in tags}), trial.numeric({tag.code: tag for tag in tags}[50940]).reshape(-1, 2)))
    actual = cp.render_profile(camera, neutral, prof, exposure_ev=2.3)
    error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert float(error.mean()) <= 0.02
    assert int(error.max()) <= 1
