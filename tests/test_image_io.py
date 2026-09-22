from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_io import (
    _exif_from_pil,
    _fill_lens_specification,
    _lens_metadata_from_exiftool_record,
    _parse_lens_specification,
    _raw_bit_depth,
    _raw_metadata,
)


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
