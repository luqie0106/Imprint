from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dng_writer import _packed_strip_bytes, _quantize_samples, write_linear_dng


def _primary_tag(payload: bytes, wanted: int) -> tuple[int, int, int, bytes]:
    ifd_offset = struct.unpack_from("<I", payload, 4)[0]
    count = struct.unpack_from("<H", payload, ifd_offset)[0]
    for index in range(count):
        entry = ifd_offset + 2 + index * 12
        code, kind, item_count = struct.unpack_from("<HHI", payload, entry)
        if code == wanted:
            return code, kind, item_count, payload[entry + 8:entry + 12]
    raise AssertionError(f"missing TIFF tag {wanted}")


def test_linear_dng_signature_tags_and_non_overwrite(tmp_path: Path):
    image = np.linspace(0, 65535, 96 * 128 * 3, dtype=np.uint16).reshape(96, 128, 3)
    first = write_linear_dng(image, "scene.jpg", tmp_path, {"Make": "Test", "Model": "Fixture"})
    second = write_linear_dng(image, "scene.jpg", tmp_path, {"Make": "Test", "Model": "Fixture"})
    assert first != second and first.exists() and second.exists()
    payload = first.read_bytes()
    assert payload[:4] == b"II*\x00"
    ifd_offset = struct.unpack_from("<I", payload, 4)[0]
    count = struct.unpack_from("<H", payload, ifd_offset)[0]
    tags = {struct.unpack_from("<H", payload, ifd_offset + 2 + index * 12)[0] for index in range(count)}
    assert {256, 257, 258, 259, 262, 50706, 50707, 50721}.issubset(tags)


def test_dng_contains_exif_ifd_and_standard_fields(tmp_path: Path):
    image = np.full((24, 32, 3), 12000, dtype=np.uint16)
    output = write_linear_dng(
        image,
        "scene.nef",
        tmp_path,
        {
            "Make": "Test Make",
            "Model": "Test Body",
            "DateTimeOriginal": "2024:05:06 07:08:09",
            "DateTimeDigitized": "2024:05:06 07:08:10",
            "ExposureTime": 1 / 125,
            "FNumber": 2.8,
            "ISO": 400,
            "ExposureBiasValue": -1 / 3,
            "MeteringMode": 5,
            "Flash": 0,
            "FocalLength": 50,
            "LensMake": "Test Lens Co",
            "LensModel": "Prime 50",
            "LensSpecification": (50, 50, 1.8, 1.8),
            "BodySerialNumber": "body-1",
            "LensSerialNumber": "lens-1",
            "ExposureProgram": 3,
            "WhiteBalance": 0,
            "OffsetTimeOriginal": "+08:00",
            "Artist": "Photographer",
            "Copyright": "Copyright 2024",
            "GPSLatitudeRef": "N",
            "GPSLatitude": (40, 30, 0),
            "GPSLongitudeRef": "W",
            "GPSLongitude": (73, 59, 0),
        },
    )
    payload = output.read_bytes()
    primary_offset = struct.unpack_from("<I", payload, 4)[0]
    primary_count = struct.unpack_from("<H", payload, primary_offset)[0]
    primary = {}
    for index in range(primary_count):
        entry = primary_offset + 2 + index * 12
        code, kind, count = struct.unpack_from("<HHI", payload, entry)
        if kind == 4 and count == 1:
            value = struct.unpack_from("<I", payload, entry + 8)[0]
        else:
            value = None
        primary[code] = value
    assert primary[34665] > 0  # ExifIFD pointer
    assert primary[34853] > 0  # GPS IFD pointer
    assert 700 in primary  # embedded XMP packet with Camera Raw lens settings
    assert 50736 in primary  # DNG LensInfo used by raw converters for matching

    exif_offset = primary[34665]
    exif_count = struct.unpack_from("<H", payload, exif_offset)[0]
    exif_codes = {
        struct.unpack_from("<H", payload, exif_offset + 2 + index * 12)[0]
        for index in range(exif_count)
    }
    assert {
        33434, 33437, 34850, 34855, 36864, 36867, 36868, 36881,
        37380, 37383, 37385, 37386, 40961, 40962, 40963,
        41987, 42033, 42034, 42035, 42036, 42037,
    }.issubset(exif_codes)


def test_exiftool_accepts_dng_when_available(tmp_path: Path):
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return
    path = write_linear_dng(np.full((64, 96, 3), 28000, np.uint16), "raw.nef", tmp_path)
    result = subprocess.run(
        [exiftool, "-validate", "-warning", "-error", "-PhotometricInterpretation", "-Make", "-Model", str(path)],
        check=True, capture_output=True, text=True,
    )
    assert "Validate" in result.stdout and "OK" in result.stdout
    assert "Linear Raw" in result.stdout
    assert "Imprint" in result.stdout and "RGB Linear DNG" in result.stdout


def test_source_bit_depth_is_written_and_reduces_payload(tmp_path: Path):
    rng = np.random.default_rng(42)
    image = rng.integers(0, 65536, size=(96, 127, 3), dtype=np.uint16)
    paths = {
        bits: write_linear_dng(image, f"raw-{bits}.nef", tmp_path, bits_per_sample=bits)
        for bits in (12, 14, 16)
    }
    for bits, path in paths.items():
        payload = path.read_bytes()
        _, kind, count, value_or_offset = _primary_tag(payload, 258)
        assert kind == 3 and count == 3
        offset = struct.unpack("<I", value_or_offset)[0]
        assert struct.unpack_from("<HHH", payload, offset) == (bits, bits, bits)
        _, compression_kind, compression_count, compression_value = _primary_tag(payload, 259)
        assert compression_kind == 3 and compression_count == 1
        assert struct.unpack("<H", compression_value[:2])[0] == 1
    assert paths[12].stat().st_size < paths[14].stat().st_size < paths[16].stat().st_size


def test_packed_samples_round_trip_with_strip_padding():
    image = np.array(
        [
            [[0, 1000, 65535], [32768, 12345, 54321], [1, 2, 3]],
            [[65535, 50000, 40000], [30000, 20000, 10000], [9, 8, 7]],
        ],
        dtype=np.uint16,
    )
    for bits_per_sample in (10, 12, 14):
        quantized = _quantize_samples(image, bits_per_sample)
        packed = _packed_strip_bytes(quantized, bits_per_sample)
        sample_bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[
            :image.size * bits_per_sample
        ]
        samples = sample_bits.reshape(-1, bits_per_sample)
        weights = 1 << np.arange(bits_per_sample - 1, -1, -1, dtype=np.uint32)
        decoded = (samples.astype(np.uint32) @ weights).astype(np.uint16).reshape(image.shape)
        np.testing.assert_array_equal(decoded, quantized)


def test_exiftool_validates_packed_bit_depths_when_available(tmp_path: Path):
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return
    image = np.linspace(0, 65535, 65 * 97 * 3, dtype=np.uint16).reshape(65, 97, 3)
    for bits in (12, 14):
        path = write_linear_dng(image, f"raw-{bits}.nef", tmp_path, bits_per_sample=bits)
        result = subprocess.run(
            [exiftool, "-validate", "-warning", "-error", "-BitsPerSample", str(path)],
            check=True, capture_output=True, text=True,
        )
        assert "Validate" in result.stdout and "OK" in result.stdout
        assert f"{bits} {bits} {bits}" in result.stdout


def test_exiftool_reads_transferred_metadata_when_available(tmp_path: Path):
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return
    path = write_linear_dng(
        np.full((32, 48, 3), 28000, np.uint16),
        "raw.nef",
        tmp_path,
        {
            "Make": "Test Make",
            "Model": "Test Body",
            "DateTimeOriginal": "2024:05:06 07:08:09",
            "ExposureTime": 1 / 125,
            "FNumber": 2.8,
            "ISO": 400,
            "FocalLength": 50,
            "LensModel": "Prime 50",
            "LensSpecification": (50, 50, 1.8, 1.8),
            "LensSerialNumber": "lens-1",
            "Artist": "Photographer",
        },
    )
    result = subprocess.run(
        [
            exiftool,
            "-validate",
            "-warning",
            "-error",
            "-s",
            "-G1",
            "-a",
            "-Make",
            "-Model",
            "-DateTimeOriginal",
            "-ExposureTime",
            "-FNumber",
            "-ISO",
            "-FocalLength",
            "-LensModel",
            "-Artist",
            "-XMP-crs:HasSettings",
            "-XMP-crs:LensProfileEnable",
            "-XMP-crs:LensProfileSetup",
            "-XMP-crs:LensProfileDistortionScale",
            "-XMP-crs:LensProfileVignettingScale",
            "-XMP-crs:RawFileName",
            "-XMP-aux:Lens",
            "-XMP-aux:LensInfo",
            "-EXIF:LensInfo",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Validate                        : OK" in result.stdout
    assert "Test Make" in result.stdout
    assert "Test Body" in result.stdout
    assert "2024:05:06 07:08:09" in result.stdout
    assert "Prime 50" in result.stdout
    assert "LensProfileEnable               : 1" in result.stdout
    assert "LensProfileSetup                : Auto" in result.stdout
    assert "LensProfileDistortionScale      : 0" in result.stdout
    assert "LensProfileVignettingScale      : 100" in result.stdout
    assert "RawFileName                     : raw.nef" in result.stdout
    assert result.stdout.count("LensInfo") >= 2
    assert "50mm f/1.8" in result.stdout


def test_baked_lens_correction_disables_second_acr_profile_pass(tmp_path: Path):
    path = write_linear_dng(
        np.full((24, 32, 3), 24000, np.uint16),
        "corrected.nef",
        tmp_path,
        {
            "Make": "Nikon",
            "Model": "Z6_3",
            "LensModel": "NIKKOR Z 14-24mm f/2.8 S",
            "LensCorrectionApplied": True,
            "LensCorrectionEngine": "Lensfun 0.3.4",
            "LensCorrectionCamera": "Nikon Z6_3",
            "LensCorrectionLens": "Nikkor Z 14-24mm f/2.8 S",
            "LensCorrectionOperations": "distortion,tca,vignetting",
        },
    )
    payload = path.read_bytes()
    assert b'crs:LensProfileEnable="0"' in payload
    assert b'crs:LensProfileSetup="Custom"' in payload
    assert b'crs:LensProfileVignettingScale="0"' in payload
    assert b'imprint:LensCorrectionApplied="True"' in payload
    assert b'imprint:LensCorrectionOperations="distortion,tca,vignetting"' in payload
    assert b"Imprint baked Lensfun correction into pixels" in payload


def test_failure_leaves_no_dng(tmp_path: Path):
    bad = np.zeros((10, 10), dtype=np.uint16)
    try:
        write_linear_dng(bad, "broken.jpg", tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input should fail")
    assert not list(tmp_path.glob("*.dng"))
