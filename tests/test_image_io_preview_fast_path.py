from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src import image_io


class _FakeRaw:
    white_level = 4095
    camera_white_level_per_channel = [4095, 4095, 4095, 4095]
    camera_whitebalance = (2.0, 1.0, 1.5, 1.0)
    other = SimpleNamespace(
        iso_speed=200,
        shutter_speed=0.01,
        aperture=2.8,
        focal_length=35.0,
    )
    lens = SimpleNamespace(
        make="Example",
        model="Example Lens",
        min_focal=24.0,
        max_focal=70.0,
        max_aperture_at_min_focal=2.8,
        max_aperture_at_max_focal=4.0,
    )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def postprocess(self, **_kwargs):
        return np.zeros((13, 21, 3), dtype=np.uint16)


def test_raw_preview_reads_lens_metadata_without_source_bit_depth(monkeypatch):
    monkeypatch.setattr(image_io.rawpy, "imread", lambda _path: _FakeRaw())

    def unexpected(*_args, **_kwargs):
        raise AssertionError("preview should not read source bit depth")

    monkeypatch.setattr(image_io, "_raw_bit_depth", unexpected)
    monkeypatch.setattr(image_io, "_safe_exif", lambda _path: {"Make": "Example", "Model": "Camera"})

    image, metadata = image_io._read_raw(Path("preview.dng"), preview=True)

    assert image.shape == (13, 21, 3)
    assert metadata.width == 21
    assert metadata.height == 13
    assert metadata.color_space == "Linear sRGB"
    assert metadata.bit_depth == 16
    assert metadata.exif["Make"] == "Example"
    assert metadata.exif["LensModel"] == "Example Lens"
    assert metadata.exif["FocalLength"] == 35.0


def test_raw_export_keeps_source_bit_depth_and_camera_metadata(monkeypatch):
    monkeypatch.setattr(image_io.rawpy, "imread", lambda _path: _FakeRaw())
    monkeypatch.setattr(
        image_io,
        "_safe_exif",
        lambda _path: {
            "LensModel": "Example Lens",
            "LensSpecification": (24.0, 70.0, 2.8, 4.0),
        },
    )

    image, metadata = image_io._read_raw(Path("export.dng"), preview=False)

    assert image.shape == (13, 21, 3)
    assert metadata.bit_depth == 12
    assert metadata.exif["ISO"] == 200
    assert metadata.exif["LensModel"] == "Example Lens"
    assert metadata.exif["CameraWhiteBalance"] == (2.0, 1.0, 1.5, 1.0)
