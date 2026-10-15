from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import image_io


def _read_raw_metadata(monkeypatch, tmp_path: Path, exif: dict, supplement=None):
    source = tmp_path / "capture.nef"
    source.write_bytes(b"raw metadata fixture")
    monkeypatch.setattr(image_io, "_safe_exif", lambda _path: dict(exif))

    def fail_libraw(_path):
        raise RuntimeError("metadata-only fixture")

    monkeypatch.setattr(image_io.rawpy, "imread", fail_libraw)
    if supplement is not None:
        monkeypatch.setattr(image_io, "_exiftool_lens_metadata", supplement)
    return image_io.read_photo_metadata(source)


def test_directory_scan_is_top_level_sorted_and_excludes_output(tmp_path: Path):
    root = tmp_path / "photos"
    root.mkdir()
    expected = [root / "A.NEF", root / "b.jpg"]
    for photo in expected:
        photo.touch()
    nested = root / "nested"
    nested.mkdir()
    (nested / "nested.png").touch()
    output = root / image_io.OUTPUT_DIR_NAME
    output.mkdir()
    (output / "generated.dng").touch()
    (root / "notes.txt").touch()

    assert image_io.scan_photo_directory(root) == expected


def test_apsc_range_uses_positive_equivalent_focal_length_without_changing_original(
    monkeypatch, tmp_path: Path,
):
    exif = {
        "FocalLength": 16,
        "FocalLengthIn35mmFilm": 24,
        "LensSpecification": (16, 50, 3.5, 5.6),
    }
    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("valid embedded equivalent should avoid fallback")

    monkeypatch.setattr(image_io, "_exiftool_lens_metadata", unexpected_fallback)
    metadata = _read_raw_metadata(monkeypatch, tmp_path, exif)

    assert metadata["focal_length"] == 16
    assert metadata["lens_focal_range"] == [16, 50]
    assert metadata["focal_length_35mm"] == 24
    assert metadata["lens_focal_range_35mm"] == [24, 75]


@pytest.mark.parametrize("recorded_value", [None, 0, float("nan")], ids=["missing", "zero", "nan"])
def test_invalid_embedded_equivalent_focal_length_uses_raw_exiftool_supplement(
    monkeypatch, tmp_path: Path, recorded_value,
):
    exif = {
        "FocalLength": 16,
        "LensSpecification": (16, 50, 3.5, 5.6),
    }
    if recorded_value is not None:
        exif["FocalLengthIn35mmFilm"] = recorded_value
    calls = []

    def supplement(path, *, include_focal_length_35mm=False):
        calls.append((path, include_focal_length_35mm))
        return {"FocalLengthIn35mmFilm": 24}

    metadata = _read_raw_metadata(monkeypatch, tmp_path, exif, supplement)

    assert calls == [(tmp_path / "capture.nef", True)]
    assert metadata["focal_length"] == 16
    assert metadata["lens_focal_range"] == [16, 50]
    assert metadata["focal_length_35mm"] == 24
    assert metadata["lens_focal_range_35mm"] == [24, 75]


def test_equivalent_range_is_unknown_without_metadata_and_for_dji_pre_equivalent_lens_info(
    monkeypatch, tmp_path: Path,
):
    monkeypatch.setattr(image_io, "_exiftool_lens_metadata", lambda *_args, **_kwargs: {})
    absent = _read_raw_metadata(
        monkeypatch,
        tmp_path,
        {"FocalLength": 16, "LensSpecification": (16, 50, 3.5, 5.6)},
    )
    assert absent["focal_length_35mm"] is None
    assert absent["lens_focal_range"] == [16, 50]
    assert absent["lens_focal_range_35mm"] is None

    drone = _read_raw_metadata(
        monkeypatch,
        tmp_path,
        {
            "Make": "DJI",
            "FocalLength": 12.29,
            "FocalLengthIn35mmFilm": 24,
            "LensInfo": (24, 24, 2.8, 2.8),
        },
    )
    assert drone["focal_length"] == 12.29
    assert drone["focal_length_35mm"] == 24
    assert drone["lens_focal_range"] == [24, 24]
    assert drone["lens_focal_range_35mm"] is None


def test_near_one_equivalent_multiplier_keeps_the_original_lens_range():
    assert image_io._photo_lens_focal_range_35mm(50, 51, [18, 55]) == [18, 55]


def test_exiftool_equivalent_tag_maps_to_standard_field_and_is_requested_numerically(
    monkeypatch, tmp_path: Path,
):
    mapped = image_io._lens_metadata_from_exiftool_record(
        {"FocalLengthIn35mmFormat": 24},
    )
    assert mapped == {"FocalLengthIn35mmFilm": 24}

    source = tmp_path / "capture.nef"
    source.touch()
    calls = []
    monkeypatch.setattr(image_io, "_find_exiftool", lambda: "exiftool")

    def run(arguments, **kwargs):
        calls.append(arguments)
        return type("Completed", (), {"stdout": json.dumps([{"FocalLengthIn35mmFormat": 24}])})()

    monkeypatch.setattr(image_io.subprocess, "run", run)
    assert image_io._exiftool_lens_metadata(
        source, include_focal_length_35mm=True,
    ) == {"FocalLengthIn35mmFilm": 24}
    assert "-n" in calls[0]
    assert "-FocalLengthIn35mmFormat" in calls[0]
