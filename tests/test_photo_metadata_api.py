from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app_api
import image_io


def test_standard_metadata_is_compact_numeric_and_orientation_aware(monkeypatch, tmp_path: Path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (640, 480)).save(path, format="JPEG")
    monkeypatch.setattr(image_io, "_exif_from_pil", lambda _image: {
        "Orientation": 6,
        "Model": "Example Camera",
        "ISOSpeedRatings": 400,
        "FNumber": (28, 10),
        "ExposureTime": (1, 250),
        "FocalLength": (50, 1),
        "LensSpecification": ((24, 1), (70, 1), (28, 10), (28, 10)),
        "GPSLatitude": ((22, 1), (32, 1), (0, 1)),
        "MakerNote": "private data",
    })

    metadata = image_io.read_photo_metadata(path)

    assert metadata == {
        "filename": "sample.jpg",
        "size_bytes": path.stat().st_size,
        "width": 480,
        "height": 640,
        "iso": 400,
        "aperture": 2.8,
        "exposure_time": 0.004,
        "focal_length": 50,
        "lens_focal_range": [24, 70],
        "focal_length_35mm": None,
        "lens_focal_range_35mm": None,
        "camera_model": "Example Camera",
    }
    assert "GPSLatitude" not in metadata
    assert "MakerNote" not in metadata


@pytest.mark.parametrize("exif, expected", [
    ({"LensSpecification": (35, 35, 1.8, 1.8)}, [35, 35]),
    ({"LensModel": "Example 24-70mm F2.8"}, [24, 70]),
    ({"LensModel": "Example 35mm F1.8"}, [35, 35]),
    ({"FocalLength": 24}, None),
    ({"LensSpecification": (0, 0, 0, 0)}, None),
    ({"LensSpecification": (70, 24, 2.8, 2.8)}, None),
])
def test_lens_range_uses_lens_metadata_not_current_focal_length(exif, expected):
    assert image_io._photo_lens_focal_range(exif) == expected


def test_raw_metadata_opens_libraw_without_postprocessing(monkeypatch, tmp_path: Path):
    path = tmp_path / "sample.nef"
    path.write_bytes(b"raw-placeholder")

    class FakeRaw:
        sizes = SimpleNamespace(width=6000, height=4000)
        other = SimpleNamespace(
            iso_speed=800,
            shutter_speed=1 / 125,
            aperture=4.0,
            focal_length=85.0,
        )
        lens = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def postprocess(self, **_kwargs):
            raise AssertionError("metadata lookup must not decode RAW pixels")

    monkeypatch.setattr(image_io.rawpy, "imread", lambda _path: FakeRaw())
    monkeypatch.setattr(image_io, "_exiftool_lens_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(image_io, "_safe_exif", lambda _path: {
        "Orientation": 1,
        "Make": "Example",
        "Model": "Example X",
    })

    metadata = image_io.read_photo_metadata(path)

    assert metadata["width"] == 6000
    assert metadata["height"] == 4000
    assert metadata["iso"] == 800
    assert metadata["aperture"] == 4
    assert metadata["exposure_time"] == 0.008
    assert metadata["focal_length"] == 85
    assert metadata["focal_length_35mm"] is None
    assert metadata["lens_focal_range_35mm"] is None
    assert metadata["camera_model"] == "Example X"


def test_enhance_metadata_requires_active_session_and_photo_ids(monkeypatch, tmp_path: Path):
    session_id = "metadata-session"
    photo_id = "photo-opaque-id"
    path = tmp_path / "private-photo.jpg"
    path.write_bytes(b"image")
    app_api._ENHANCE_SESSIONS[session_id] = {
        "files": {photo_id: path},
        "created": 0.0,
    }
    returned = {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "width": 1,
        "height": 1,
        "iso": None,
        "aperture": None,
        "exposure_time": None,
        "focal_length": None,
        "lens_focal_range": None,
        "focal_length_35mm": None,
        "lens_focal_range_35mm": None,
        "camera_model": None,
    }
    seen_paths: list[Path] = []
    monkeypatch.setattr(
        app_api,
        "read_photo_metadata",
        lambda current_path: seen_paths.append(Path(current_path)) or returned,
    )
    try:
        response = app_api.get_enhance_photo_metadata(session_id, photo_id)

        assert response == returned
        assert seen_paths == [path.resolve()]
        assert "path" not in response
        assert str(path) not in repr(response)
        assert app_api.get_enhance_photo_metadata(session_id, "unknown-photo").status_code == 404
        assert app_api.get_enhance_photo_metadata("unknown-session", photo_id).status_code == 404
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
