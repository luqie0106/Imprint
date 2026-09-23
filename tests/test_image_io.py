from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import rawpy



sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_io import (
    _exif_from_pil,
    _fill_lens_specification,
    _lens_metadata_from_exiftool_record,
    _parse_lens_specification,
    _raw_bit_depth,
    _read_raw,
    _raw_metadata,
    camera_profile_names,
    enhanced_dng_source_data,
    matching_embedded_profile_dng,
)


def test_raw_preview_uses_the_export_linear_exposure_scale(monkeypatch):
    calls = []

    class FakeRaw:
        camera_whitebalance = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def postprocess(self, **kwargs):
            calls.append(kwargs)
            return np.full((4, 6, 3), 32768, dtype=np.uint16)

        def extract_thumb(self):
            raise RuntimeError("no thumbnail")

    monkeypatch.setattr("image_io.rawpy.imread", lambda _path: FakeRaw())
    monkeypatch.setattr("image_io._raw_bit_depth", lambda _raw: 14)
    monkeypatch.setattr("image_io._safe_exif", lambda _path: {
        "LensModel": "Test Lens", "LensSpecification": (35, 35, 2, 2),
    })
    monkeypatch.setattr("image_io._raw_metadata", lambda *_args: None)
    monkeypatch.setattr("image_io._fill_lens_specification", lambda _exif: None)

    image, metadata = _read_raw(Path("sample.nef"), preview=True)
    assert image.dtype == np.uint16
    assert metadata.color_space == "Linear sRGB"
    assert calls[0]["half_size"] is True
    assert calls[0]["no_auto_bright"] is True
    assert calls[0]["output_bps"] == 16
    assert calls[0]["gamma"] == (1.0, 1.0)


class _FakeExif(dict):
    def __init__(self, root: dict, children: dict[int, dict]):
        super().__init__(root)
        self._children = children

    def get_ifd(self, tag: int):
        return self._children.get(tag, {})


class _FakeImage:
    def __init__(self, exif: _FakeExif):
        self._exif = exif

    def getexif(self):
        return self._exif


def test_exif_from_pil_walks_exif_and_gps_ifds_but_skips_makernote():
    exif = _FakeExif(
        {
            271: "NIKON CORPORATION",
            272: "NIKON Z 8",
            34665: 100,
            34853: 200,
        },
        {
            34665: {
                33434: (1, 125),
                33437: 2.8,
                34855: 400,
                36867: "2026:09:22 10:11:12",
                37500: b"vendor-private-offsets",
                42036: "NIKKOR Z 50mm f/1.8 S",
            },
            34853: {
                1: "N",
                2: ((22, 1), (32, 1), (123, 10)),
                3: "E",
                4: ((114, 1), (3, 1), (456, 10)),
            },
        },
    )

    metadata = _exif_from_pil(_FakeImage(exif))

    assert metadata["Make"] == "NIKON CORPORATION"
    assert metadata["Model"] == "NIKON Z 8"
    assert metadata["ExposureTime"] == (1, 125)
    assert metadata["FNumber"] == 2.8
    assert metadata["ISOSpeedRatings"] == 400
    assert metadata["DateTimeOriginal"] == "2026:09:22 10:11:12"
    assert metadata["LensModel"] == "NIKKOR Z 50mm f/1.8 S"
    assert metadata["GPSLatitudeRef"] == "N"
    assert metadata["GPSLatitude"] == ((22, 1), (32, 1), (123, 10))
    assert "MakerNote" not in metadata


def test_raw_metadata_fills_standard_fields_without_overwriting_embedded_exif():
    raw = SimpleNamespace(
        other=SimpleNamespace(
            iso_speed=800,
            shutter_speed=1 / 60,
            aperture=4.0,
            focal_length=85.0,
            timestamp=datetime(2026, 9, 22, 8, 9, 10),
            artist="Photographer",
        ),
        lens=SimpleNamespace(
            make="NIKON",
            model="NIKKOR Z 85mm f/1.8 S",
        ),
        daylight_whitebalance=[2.0, 1.0, 1.5, 1.0],
        auto_whitebalance=None,
    )
    metadata = {"ISO": 200, "LensModel": "Embedded lens name"}

    _raw_metadata(raw, metadata)

    assert metadata["ISO"] == 200
    assert metadata["LensModel"] == "Embedded lens name"
    assert metadata["ExposureTime"] == 1 / 60
    assert metadata["FNumber"] == 4.0
    assert metadata["FocalLength"] == 85.0
    assert metadata["DateTimeOriginal"] == "2026:09:22 08:09:10"
    assert metadata["DateTimeDigitized"] == "2026:09:22 08:09:10"
    assert metadata["Artist"] == "Photographer"
    assert metadata["LensMake"] == "NIKON"
    assert metadata["DaylightWhiteBalance"] == [2.0, 1.0, 1.5, 1.0]


def test_raw_bit_depth_uses_sensor_white_level_and_supported_widths():
    assert _raw_bit_depth(SimpleNamespace(white_level=4095, camera_white_level_per_channel=None)) == 12
    assert _raw_bit_depth(SimpleNamespace(white_level=15520, camera_white_level_per_channel=None)) == 14
    assert _raw_bit_depth(SimpleNamespace(white_level=65535, camera_white_level_per_channel=None)) == 16


def test_private_lens_id_is_mapped_to_portable_lens_fields():
    metadata = _lens_metadata_from_exiftool_record(
        {
            "Make": "NIKON CORPORATION",
            "LensID": "Nikkor Z 14-24mm f/2.8 S",
            "Lens": "14-24mm f/2.8",
            "LensSerialNumber": "12345678",
        }
    )

    assert metadata == {
        "LensModel": "Nikkor Z 14-24mm f/2.8 S",
        "LensMake": "Nikon",
        "LensSerialNumber": "12345678",
        "LensSpecification": (14.0, 24.0, 2.8, 2.8),
    }


def test_lens_name_supplies_missing_lens_specification():
    assert _parse_lens_specification("NIKKOR Z 14-24mm f/2.8 S") == (14.0, 24.0, 2.8, 2.8)
    assert _parse_lens_specification("NIKKOR Z 24-200mm f/4-6.3 VR") == (24.0, 200.0, 4.0, 6.3)
    metadata = {"LensModel": "NIKKOR Z 50mm f/1.8 S"}
    _fill_lens_specification(metadata)
    assert metadata["LensSpecification"] == (50.0, 50.0, 1.8, 1.8)


def test_camera_profile_names_reads_only_whitelisted_names(monkeypatch):
    monkeypatch.setattr("image_io._find_exiftool", lambda: "exiftool")
    monkeypatch.setattr(
        "image_io.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout='[{"CameraProfile":"Camera Flexible Color","PictureControlName":"Negative_Cy_01a","SerialNumber":"secret"}]'
        ),
    )
    assert camera_profile_names("sample.nef") == {
        "SourceCameraProfileName": "Camera Flexible Color",
        "NikonPictureControlName": "Negative_Cy_01a",
    }


def test_enhanced_source_data_keeps_cfa_and_rotates_rgb_to_sensor_orientation(monkeypatch):
    rng = np.random.default_rng(34)
    sensor_rgb = rng.integers(8000, 26000, (256, 320, 3), dtype=np.uint16)
    reference = np.ascontiguousarray(np.rot90(sensor_rgb))
    mosaic = np.full((256, 320), 1200, dtype=np.uint16)

    class FakeRaw:
        raw_image_visible = mosaic
        raw_pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        color_desc = b"RGBG"
        rgb_xyz_matrix = np.array([[1.0, 0.1, 0], [0, 1.0, 0.1], [0.1, 0, 1.0], [0, 0, 0]])
        camera_whitebalance = [2.0, 1.0, 1.25, 1.0]
        black_level_per_channel = [100, 100, 100, 100]
        white_level = 16383

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def postprocess(self, *, output_color, **_kwargs):
            assert output_color == rawpy.ColorSpace.raw
            return reference

    monkeypatch.setattr("image_io.rawpy.imread", lambda _path: FakeRaw())
    enhanced, cfa, pattern, profile, orientation = enhanced_dng_source_data(
        reference, reference, "sample.nef", {"Make": "Nikon", "Model": "Z6_3", "Orientation": 8},
    )
    assert enhanced.shape == cfa.shape + (3,)
    assert np.array_equal(enhanced, sensor_rgb)
    assert pattern.tolist() == [[0, 1], [1, 2]]
    assert profile["DNGUniqueCameraModel"] == "Nikon Z 6 3"
    assert profile["DNGBlackLevel"] == (100, 100, 100, 100)
    assert profile["DNGDefaultCropOrigin"] == (0, 0)
    assert profile["DNGDefaultCropSize"] == (320, 256)
    assert orientation == 8


def test_enhanced_source_data_uses_midtones_when_highlights_are_nonlinear(monkeypatch):
    rng = np.random.default_rng(8707)
    camera = rng.integers(5000, 18000, (256, 256, 3), dtype=np.uint16)
    camera[128:] = rng.integers(40000, 55000, (128, 256, 3), dtype=np.uint16)
    reference = camera.copy()
    bright = camera.max(axis=2) > 22000
    reference[bright, 0] = np.minimum(
        reference[bright, 0].astype(np.uint32) + 5000, 65535,
    ).astype(np.uint16)

    class FakeRaw:
        raw_image_visible = np.full((256, 256), 1200, dtype=np.uint16)
        raw_pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        color_desc = b"RGBG"
        rgb_xyz_matrix = np.eye(4, 3)
        camera_whitebalance = [1.4, 1.0, 1.6, 1.0]
        black_level_per_channel = [100, 100, 100, 100]
        white_level = 16383

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def postprocess(self, **_kwargs):
            return camera

    monkeypatch.setattr("image_io.rawpy.imread", lambda _path: FakeRaw())
    enhanced, _, _, _, _ = enhanced_dng_source_data(
        reference, reference, "bright.nef", {"Make": "Nikon", "Model": "Z6_3"},
    )
    np.testing.assert_array_equal(enhanced, reference)


def test_profile_reference_requires_identical_picture_control_bytes(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.nef"
    reference = tmp_path / "reference.dng"
    source.touch()
    reference.touch()
    controls = {str(source): b"nikon-flexible-color", str(reference): b"nikon-flexible-color"}
    curves = {str(source): b"tone-curve", str(reference): b"tone-curve"}

    def fake_run(arguments, **_kwargs):
        path = arguments[-1]
        if "-json" in arguments:
            return SimpleNamespace(stdout='[{"Make":"NIKON CORPORATION","Model":"NIKON Z6_3","ProfileName":"Camera Flexible Color","ProfileEmbedPolicy":0}]')
        if "-Nikon:ContrastCurve" in arguments:
            return SimpleNamespace(stdout=curves[path])
        return SimpleNamespace(stdout=controls[path])

    monkeypatch.setattr("image_io._find_exiftool", lambda: "exiftool")
    monkeypatch.setattr("image_io.subprocess.run", fake_run)
    info = {
        "Make": "NIKON CORPORATION", "Model": "NIKON Z6_3",
        "SourceCameraProfileName": "Camera Flexible Color",
    }
    assert matching_embedded_profile_dng(source, info) == reference
    curves[str(reference)] = b"different tone curve"
    assert matching_embedded_profile_dng(source, info) is None
    curves[str(reference)] = curves[str(source)]
    controls[str(reference)] = b"a different camera recipe"
    assert matching_embedded_profile_dng(source, info) is None
