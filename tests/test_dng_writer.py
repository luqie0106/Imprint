from __future__ import annotations

import io
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dng_writer
from dng_writer import _packed_strip_bytes, _quantize_samples, write_enhanced_dng, write_linear_dng


def _primary_tag(payload: bytes, wanted: int) -> tuple[int, int, int, bytes]:
    ifd_offset = struct.unpack_from("<I", payload, 4)[0]
    count = struct.unpack_from("<H", payload, ifd_offset)[0]
    for index in range(count):
        entry = ifd_offset + 2 + index * 12
        code, kind, item_count = struct.unpack_from("<HHI", payload, entry)
        if code == wanted:
            return code, kind, item_count, payload[entry + 8:entry + 12]
    raise AssertionError(f"missing TIFF tag {wanted}")


def _primary_tag_data(payload: bytes, wanted: int) -> tuple[int, int, bytes]:
    ifd_offset = struct.unpack_from("<I", payload, 4)[0]
    count = struct.unpack_from("<H", payload, ifd_offset)[0]
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8, 11: 4, 12: 8}
    for index in range(count):
        entry = ifd_offset + 2 + index * 12
        code, kind, item_count = struct.unpack_from("<HHI", payload, entry)
        if code != wanted:
            continue
        byte_count = sizes[kind] * item_count
        value_or_offset = payload[entry + 8:entry + 12]
        if byte_count <= 4:
            return kind, item_count, value_or_offset[:byte_count]
        data_offset = struct.unpack("<I", value_or_offset)[0]
        return kind, item_count, payload[data_offset:data_offset + byte_count]
    raise AssertionError(f"missing TIFF tag {wanted}")


def _ifd_entries(payload: bytes, ifd_offset: int) -> dict[int, tuple[int, int, bytes]]:
    count = struct.unpack_from("<H", payload, ifd_offset)[0]
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8, 11: 4, 12: 8}
    entries: dict[int, tuple[int, int, bytes]] = {}
    for index in range(count):
        entry = ifd_offset + 2 + index * 12
        code, kind, item_count = struct.unpack_from("<HHI", payload, entry)
        byte_count = sizes[kind] * item_count
        value_or_offset = payload[entry + 8:entry + 12]
        if byte_count <= 4:
            data = value_or_offset[:byte_count]
        else:
            data_offset = struct.unpack("<I", value_or_offset)[0]
            data = payload[data_offset:data_offset + byte_count]
        entries[code] = (kind, item_count, data)
    return entries


def _ifd_next_offset(payload: bytes, ifd_offset: int) -> int:
    count = struct.unpack_from("<H", payload, ifd_offset)[0]
    return struct.unpack_from("<I", payload, ifd_offset + 2 + count * 12)[0]


def _decode_lossless_jpeg(encoded: bytes) -> np.ndarray:
    try:
        import imagecodecs
    except ImportError:
        imagecodecs = None
    if imagecodecs is not None and callable(getattr(imagecodecs, "ljpeg_decode", None)):
        try:
            return imagecodecs.ljpeg_decode(encoded)
        except ImportError:
            pass
    djpeg = shutil.which("djpeg")
    if not djpeg:
        pytest.skip("lossless JPEG decoder is unavailable")
    with tempfile.TemporaryDirectory(prefix="imprint-test-ljpeg-") as work_dir:
        source = Path(work_dir) / "strip.jpg"
        decoded = Path(work_dir) / "strip.pgm"
        source.write_bytes(encoded)
        subprocess.run([djpeg, "-outfile", str(decoded), str(source)], check=True, capture_output=True)
        return _read_pnm16(decoded.read_bytes(), 1)


def _decode_jpeg_xl(encoded: bytes) -> np.ndarray:
    try:
        import imagecodecs
    except ImportError:
        imagecodecs = None
    if imagecodecs is not None and callable(getattr(imagecodecs, "jpegxl_decode", None)):
        try:
            return imagecodecs.jpegxl_decode(encoded)
        except ImportError:
            pass
    djxl = shutil.which("djxl")
    if not djxl:
        pytest.skip("JPEG XL decoder is unavailable")
    with tempfile.TemporaryDirectory(prefix="imprint-test-jxl-") as work_dir:
        source = Path(work_dir) / "strip.jxl"
        decoded = Path(work_dir) / "strip.ppm"
        source.write_bytes(encoded)
        subprocess.run([djxl, str(source), str(decoded), "--quiet"], check=True, capture_output=True)
        return _read_pnm16(decoded.read_bytes(), 3)


def _read_pnm16(payload: bytes, channels: int) -> np.ndarray:
    header = payload.split(b"\n", 3)
    assert len(header) == 4
    magic, dimensions, maximum, pixels = header
    assert magic == (b"P5" if channels == 1 else b"P6")
    assert maximum == b"65535"
    width, height = (int(part) for part in dimensions.split())
    result = np.frombuffer(pixels, dtype=">u2").astype(np.uint16)
    shape = (height, width) if channels == 1 else (height, width, channels)
    return result.reshape(shape)


def _profile_reference(tmp_path: Path, *, policy: int = 0) -> Path:
    tags = [
        dng_writer._tag(50936, dng_writer.ASCII, dng_writer._ascii("Camera Flexible Color")),
        dng_writer._tag(50941, dng_writer.LONG, dng_writer._longs([policy])),
        dng_writer._tag(50981, dng_writer.LONG, dng_writer._longs([2, 2, 1])),
        dng_writer._tag(50982, dng_writer.FLOAT, struct.pack("<" + "f" * 12, *([1.0] * 12))),
        dng_writer._tag(50940, dng_writer.FLOAT, struct.pack("<ffff", 0.0, 0.0, 1.0, 1.0)),
    ]
    ifd, _ = dng_writer._build_ifd(tags, 8)
    reference = tmp_path / "adobe-reference.dng"
    reference.write_bytes(b"II*\0" + struct.pack("<I", 8) + ifd)
    return reference


def test_enhanced_dng_embeds_matching_copy_permitted_camera_profile(tmp_path: Path):
    reference = _profile_reference(tmp_path)
    metadata = {
        "Make": "NIKON CORPORATION", "Model": "NIKON Z6_3",
        "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "DNGAsShotNeutral": [0.5, 1, 0.8],
        "DNGUniqueCameraModel": "Nikon Z 6 3",
        "DNGBlackLevel": [100, 100, 100, 100],
        "DNGWhiteLevel": 16383,
        "SourceCameraProfileName": "Camera Flexible Color",
        "DNGProfileReferencePath": reference,
    }
    output = write_enhanced_dng(
        np.full((12, 16, 3), 10000, np.uint16),
        np.full((12, 16), 1000, np.uint16),
        [[0, 1], [1, 2]], "scene.nef", tmp_path, metadata,
    )
    root = _ifd_entries(output.read_bytes(), 8)
    assert root[50936][2] == b"Camera Flexible Color\0"
    assert root[50934][2] == b"Camera Flexible Color\0"
    assert root[50981][2] == struct.pack("<III", 2, 2, 1)
    assert root[50982][1] == 12
    assert b'crs:CameraProfile="Camera Flexible Color"' in root[700][2]
    assert metadata.get("DNGEmbeddedProfileName") is None


def test_enhanced_dng_rejects_profile_that_forbids_copying(tmp_path: Path):
    reference = _profile_reference(tmp_path, policy=1)
    metadata = {
        "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "DNGAsShotNeutral": [1, 1, 1],
        "DNGUniqueCameraModel": "Camera",
        "DNGBlackLevel": [100, 100, 100, 100],
        "DNGWhiteLevel": 16383,
        "SourceCameraProfileName": "Camera Flexible Color",
        "DNGProfileReferencePath": reference,
    }
    with pytest.raises(ValueError, match="does not permit copying"):
        write_enhanced_dng(
            np.full((12, 16, 3), 10000, np.uint16),
            np.full((12, 16), 1000, np.uint16),
            [[0, 1], [1, 2]], "scene.nef", tmp_path, metadata,
        )
    assert not list(tmp_path.glob("*_dehaze*.dng"))


def test_enhanced_dng_contains_cfa_and_linear_rgb_subifds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Force several strips so the encoded bytecounts and offsets are exercised.
    monkeypatch.setattr(dng_writer, "_DNG_CODEC_STRIP_BYTES", 300)
    height, width = 12, 16
    raw = np.arange(height * width, dtype=np.uint16).reshape(height, width) + 80
    enhanced = np.arange(height * width * 3, dtype=np.uint16).reshape(height, width, 3) + 300
    preview = np.full((300, 400, 3), (70, 125, 195), dtype=np.uint8)
    metadata = {
        "Make": "Test Make",
        "Model": "Test Camera",
        "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "DNGAsShotNeutral": [1, 1, 1],
        "DNGUniqueCameraModel": "Test Camera Native",
        "DNGBlackLevel": [64, 65, 66, 67],
        "DNGWhiteLevel": 4095,
        "DNGDefaultCropOrigin": (2, 4),
        "DNGDefaultCropSize": (12, 8),
        "DateTimeOriginal": "2024:05:06 07:08:09",
        "ExposureTime": "1/125",
        "ISO": 400,
        "LensModel": "Test Lens",
        "LensCorrectionApplied": True,
        "LensCorrectionEngine": "Lensfun fixture",
        "LensCorrectionOperations": "distortion,vignetting",
    }
    output = write_enhanced_dng(
        enhanced,
        raw,
        np.array([[0, 1], [1, 2]], dtype=np.uint8),
        "scene.nef",
        tmp_path,
        metadata,
        orientation=6,
        preview_rgb16=preview,
    )
    payload = output.read_bytes()
    assert payload[:4] == b"II*\x00"
    assert not list(tmp_path.glob("*.tmp"))

    root_offset = struct.unpack_from("<I", payload, 4)[0]
    root = _ifd_entries(payload, root_offset)
    assert root[254][2] == struct.pack("<I", 1)
    assert root[258][1] == 3 and root[258][2] == struct.pack("<HHH", 8, 8, 8)
    assert root[259][2] == struct.pack("<H", 1)
    assert root[262][2] == struct.pack("<H", 2)
    assert root[274][2] == struct.pack("<H", 6)
    assert root[50706][2] == bytes((1, 7, 0, 0))
    assert root[50707][2] == bytes((1, 7, 0, 0))
    assert root[50708][2] == b"Test Camera Native\0"
    assert root[50778][2] == struct.pack("<H", 21)
    assert root[50879][2] == struct.pack("<H", 0)
    assert root[50970][0] == 4 and root[50970][2] == struct.pack("<I", 2)
    assert 34665 in root and 700 in root
    assert b"RawFileName=\"scene.nef\"" in root[700][2]
    assert b"imprint:LensCorrectionApplied=\"True\"" in root[700][2]
    thumb_width = struct.unpack("<I", root[256][2])[0]
    thumb_height = struct.unpack("<I", root[257][2])[0]
    thumb_offset = struct.unpack("<I", root[273][2])[0]
    thumb_size = struct.unpack("<I", root[279][2])[0]
    assert (thumb_width, thumb_height) == (256, 192)
    assert thumb_size == thumb_width * thumb_height * 3
    thumbnail = np.frombuffer(payload[thumb_offset:thumb_offset + thumb_size], dtype=np.uint8)
    thumbnail = thumbnail.reshape(thumb_height, thumb_width, 3)
    np.testing.assert_array_equal(thumbnail[96, 128], np.array([70, 125, 195], dtype=np.uint8))

    preview_ifd_offset = _ifd_next_offset(payload, root_offset)
    assert preview_ifd_offset > root_offset
    preview_ifd = _ifd_entries(payload, preview_ifd_offset)
    assert preview_ifd[254][2] == struct.pack("<I", 1)
    assert preview_ifd[256][2] == struct.pack("<I", 400)
    assert preview_ifd[257][2] == struct.pack("<I", 300)
    assert preview_ifd[259][2] == struct.pack("<H", 7)
    assert preview_ifd[262][2] == struct.pack("<H", 6)
    assert preview_ifd[274][2] == struct.pack("<H", 6)
    assert preview_ifd[50970][0] == 4
    assert preview_ifd[50970][2] == struct.pack("<I", 2)
    preview_offset = struct.unpack("<I", preview_ifd[273][2])[0]
    preview_size = struct.unpack("<I", preview_ifd[279][2])[0]
    preview_bytes = payload[preview_offset:preview_offset + preview_size]
    assert b"\xff\xc0" in preview_bytes and b"\xff\xc2" not in preview_bytes
    with Image.open(io.BytesIO(preview_bytes)) as decoded_preview:
        assert decoded_preview.size == (400, 300)
        pixel = np.asarray(decoded_preview.convert("RGB"))[3, 4].astype(int)
        assert np.all(np.abs(pixel - np.array([70, 125, 195])) < 12)

    subifd_kind, subifd_count, subifd_data = root[330]
    assert subifd_kind == 4 and subifd_count == 2
    raw_ifd_offset, enhanced_ifd_offset = struct.unpack("<II", subifd_data)
    raw_ifd = _ifd_entries(payload, raw_ifd_offset)
    enhanced_ifd = _ifd_entries(payload, enhanced_ifd_offset)
    assert raw_ifd[254][2] == struct.pack("<I", 0)
    assert raw_ifd[262][2] == struct.pack("<H", 32803)
    assert raw_ifd[258][2] == struct.pack("<H", 16)
    assert raw_ifd[259][2] == struct.pack("<H", 1)
    assert raw_ifd[278][2] == struct.pack("<I", 9)
    assert raw_ifd[33421][2] == struct.pack("<HH", 2, 2)
    assert raw_ifd[33422][2] == bytes((0, 1, 1, 2))
    assert raw_ifd[50711][2] == struct.pack("<H", 1)
    assert struct.unpack("<IIIIIIII", raw_ifd[50714][2]) == (64, 1, 65, 1, 66, 1, 67, 1)
    assert struct.unpack("<I", raw_ifd[50717][2])[0] == 4095
    assert raw_ifd[50719][2] == struct.pack("<II", 2, 4)
    assert raw_ifd[50720][2] == struct.pack("<II", 12, 8)
    raw_offsets = struct.unpack("<" + "I" * raw_ifd[273][1], raw_ifd[273][2])
    raw_counts = struct.unpack("<" + "I" * raw_ifd[279][1], raw_ifd[279][2])
    raw_rows_per_strip = struct.unpack("<I", raw_ifd[278][2])[0]
    raw_pixels = []
    for index, (offset, count) in enumerate(zip(raw_offsets, raw_counts)):
        start = index * raw_rows_per_strip
        end = min(height, start + raw_rows_per_strip)
        expected = raw[start:end].astype("<u2", copy=False).tobytes(order="C")
        assert count == len(expected) == (end - start) * width * 2
        assert payload[offset:offset + count] == expected
        raw_pixels.append(np.frombuffer(payload[offset:offset + count], dtype="<u2").reshape(end - start, width))
    raw_pixels = np.concatenate(raw_pixels, axis=0)
    np.testing.assert_array_equal(raw_pixels, raw)

    assert enhanced_ifd[254][2] == struct.pack("<I", 16)
    assert enhanced_ifd[262][2] == struct.pack("<H", 34892)
    assert enhanced_ifd[258][1] == 3 and enhanced_ifd[258][2] == struct.pack("<HHH", 16, 16, 16)
    assert enhanced_ifd[259][2] == struct.pack("<H", 52546)
    assert enhanced_ifd[278][2] == struct.pack("<I", 3)
    assert enhanced_ifd[51182][2] == b"Imprint Dehaze\0"
    assert struct.unpack("<IIIIII", enhanced_ifd[50714][2]) == (0, 1, 0, 1, 0, 1)
    assert struct.unpack("<III", enhanced_ifd[50717][2]) == (65535, 65535, 65535)
    rgb_offsets = struct.unpack("<" + "I" * enhanced_ifd[273][1], enhanced_ifd[273][2])
    rgb_counts = struct.unpack("<" + "I" * enhanced_ifd[279][1], enhanced_ifd[279][2])
    rgb_pixels = np.concatenate(
        [
            _decode_jpeg_xl(payload[offset:offset + count])
            for offset, count in zip(rgb_offsets, rgb_counts)
        ],
        axis=0,
    )
    assert rgb_pixels.shape == enhanced.shape
    assert rgb_pixels.dtype == enhanced.dtype == np.uint16
    rgb_error = np.abs(rgb_pixels.astype(np.int32) - enhanced.astype(np.int32))
    assert np.quantile(rgb_error, 0.99) <= 4
    assert int(rgb_error.max()) <= 8

    exif_offset = struct.unpack("<I", root[34665][2])[0]
    exif = _ifd_entries(payload, exif_offset)
    assert exif[36867][2] == b"2024:05:06 07:08:09\0"
    assert exif[33434][1] == 1


def test_enhanced_preview_is_capped_at_4096_and_uint16_input_still_works(tmp_path: Path):
    enhanced = np.full((8, 10, 3), 12000, np.uint16)
    raw = np.full((8, 10), 1000, np.uint16)
    preview = np.zeros((11, 5000, 3), dtype=np.uint16)
    preview[..., 0] = 65535
    path = write_enhanced_dng(
        enhanced,
        raw,
        [[0, 1], [1, 2]],
        "bounded.nef",
        tmp_path,
        {
            "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "DNGAsShotNeutral": [1, 1, 1],
            "DNGUniqueCameraModel": "Fixture Camera",
            "DNGBlackLevel": [64, 64, 64, 64],
            "DNGWhiteLevel": 4095,
        },
        preview_rgb16=preview,
    )
    payload = path.read_bytes()
    preview_ifd_offset = _ifd_next_offset(payload, 8)
    preview_ifd = _ifd_entries(payload, preview_ifd_offset)
    assert struct.unpack("<I", preview_ifd[256][2])[0] == 4096
    preview_offset = struct.unpack("<I", preview_ifd[273][2])[0]
    preview_size = struct.unpack("<I", preview_ifd[279][2])[0]
    with Image.open(io.BytesIO(payload[preview_offset:preview_offset + preview_size])) as image:
        assert image.width == 4096 and image.height <= 4096


def test_enhanced_dng_does_not_overwrite_same_name(tmp_path: Path):
    enhanced = np.full((8, 10, 3), 20000, dtype=np.uint16)
    raw = np.full((8, 10), 1000, dtype=np.uint16)
    metadata = {
        "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "DNGAsShotNeutral": [1, 1, 1],
        "DNGUniqueCameraModel": "Fixture Camera",
        "DNGBlackLevel": [64, 64, 64, 64],
        "DNGWhiteLevel": 4095,
    }
    arguments = (enhanced, raw, [[0, 1], [1, 2]], "same.nef", tmp_path, metadata)
    first = write_enhanced_dng(*arguments)
    first_bytes = first.read_bytes()
    second = write_enhanced_dng(*arguments)
    assert first != second
    assert first.read_bytes() == first_bytes
    assert second.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_enhanced_dng_write_failure_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    enhanced = np.full((8, 10, 3), 20000, dtype=np.uint16)
    raw = np.full((8, 10), 1000, dtype=np.uint16)
    metadata = {
        "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "DNGAsShotNeutral": [1, 1, 1],
        "DNGUniqueCameraModel": "Fixture Camera",
        "DNGBlackLevel": [64, 64, 64, 64],
        "DNGWhiteLevel": 4095,
    }

    def fail_fsync(_fd: int) -> None:
        raise OSError("injected DNG flush failure")

    monkeypatch.setattr(dng_writer.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected DNG flush failure"):
        write_enhanced_dng(
            enhanced, raw, [[0, 1], [1, 2]], "broken.nef", tmp_path, metadata
        )
    assert not list(tmp_path.glob("*.dng"))
    assert not list(tmp_path.glob("*.tmp"))


def test_enhanced_dng_codec_failure_closes_spool_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    enhanced = np.full((8, 10, 3), 20000, dtype=np.uint16)
    raw = np.full((8, 10), 1000, dtype=np.uint16)
    metadata = {
        "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "DNGAsShotNeutral": [1, 1, 1],
        "DNGUniqueCameraModel": "Fixture Camera",
        "DNGBlackLevel": [64, 64, 64, 64],
        "DNGWhiteLevel": 4095,
    }
    real_temporary_file = tempfile.TemporaryFile
    spools = []

    def track_spool(*args, **kwargs):
        spool = real_temporary_file(*args, **kwargs)
        spools.append(spool)
        return spool

    def fail_jpeg_xl(_strip: np.ndarray) -> bytes:
        raise RuntimeError("injected JPEG XL failure")

    monkeypatch.setattr(dng_writer.tempfile, "TemporaryFile", track_spool)
    monkeypatch.setattr(dng_writer, "_encode_jpeg_xl", fail_jpeg_xl)
    with pytest.raises(RuntimeError, match="injected JPEG XL failure"):
        write_enhanced_dng(
            enhanced, raw, [[0, 1], [1, 2]], "broken.nef", tmp_path, metadata
        )
    assert len(spools) == 1 and spools[0].closed
    assert not list(tmp_path.iterdir())


def test_jpeg_xl_encoder_uses_adobe_camera_raw_parameters(monkeypatch: pytest.MonkeyPatch):
    image = np.zeros((4, 6, 3), dtype=np.uint16)
    calls = []

    def fake_encoder(strip: np.ndarray, **options: object) -> bytes:
        calls.append((strip, options))
        return b"\xff\x0a\x00"

    def fake_imagecodecs_encoder(name: str):
        assert name == "jpegxl_encode"
        return fake_encoder

    monkeypatch.setattr(dng_writer, "_imagecodecs_encoder", fake_imagecodecs_encoder)
    assert dng_writer._encode_jpeg_xl(image) == b"\xff\x0a\x00"
    assert len(calls) == 1
    assert calls[0][0] is image
    assert calls[0][1] == {
        "distance": 0.01,
        "lossless": False,
        "effort": 7,
        "usecontainer": False,
    }


def test_exiftool_validates_enhanced_dng_when_available(tmp_path: Path):
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return
    enhanced = np.full((32, 48, 3), 28000, dtype=np.uint16)
    raw = np.full((32, 48), 3000, dtype=np.uint16)
    path = write_enhanced_dng(
        enhanced,
        raw,
        [[0, 1], [1, 2]],
        "validation.nef",
        tmp_path,
        {
            "Make": "Fixture",
            "Model": "Fixture Camera",
            "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "DNGAsShotNeutral": [1, 1, 1],
            "DNGUniqueCameraModel": "Fixture Camera",
            "DNGBlackLevel": [64, 65, 66, 67],
            "DNGWhiteLevel": 4095,
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
            "-DNGVersion",
            "-SubfileType",
            "-PhotometricInterpretation",
            "-EnhanceParams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Validate                        : OK" in result.stdout
    assert "Imprint Dehaze" in result.stdout


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
    kind, neutral_count, neutral_data = _primary_tag_data(payload, 50728)
    assert kind == 5 and neutral_count == 3
    assert struct.unpack("<IIIIII", neutral_data) == (1, 1, 1, 1, 1, 1)
    kind, model_count, model_data = _primary_tag_data(payload, 50708)
    assert kind == 2 and model_count == len(b"Test Fixture\0")
    assert model_data == b"Test Fixture\0"
    kind, matrix_count, matrix_data = _primary_tag_data(payload, 50721)
    assert kind == 10 and matrix_count == 9
    matrix_parts = struct.unpack("<" + "ii" * 9, matrix_data)
    assert list(zip(matrix_parts[::2], matrix_parts[1::2])) == [
        (3133856, 1000000), (-1616867, 1000000), (-490615, 1000000),
        (-978768, 1000000), (1916142, 1000000), (33454, 1000000),
        (71945, 1000000), (-228991, 1000000), (1405243, 1000000),
    ]
    kind, _, illuminant_data = _primary_tag_data(payload, 50778)
    assert kind == 3 and struct.unpack("<H", illuminant_data)[0] == 21


def test_camera_native_color_profile_keeps_source_names_as_imprint_metadata(tmp_path: Path):
    image = np.full((12, 16, 3), 18000, dtype=np.uint16)
    metadata = {
        "Make": "Nikon",
        "Model": "Z 8",
        "DNGColorMatrix1": [2, "1/10", 0, 0, 3, 0, 0, 0, 4],
        "DNGAsShotNeutral": ["1/2", 1, "2/3"],
        "DNGUniqueCameraModel": "NIKON Z 8",
        "SourceCameraProfileName": "Camera Flexible Color",
        "NikonPictureControlName": "Negative_Cy_01a",
        # Legacy callers may still supply this key; it must never select a profile.
        "DNGAsShotProfileName": "Legacy Fake Selection",
    }
    path = write_linear_dng(image, "scene.nef", tmp_path, metadata)
    payload = path.read_bytes()

    kind, count, matrix_data = _primary_tag_data(payload, 50721)
    assert kind == 10 and count == 9
    matrix_parts = struct.unpack("<" + "ii" * 9, matrix_data)
    assert list(zip(matrix_parts[::2], matrix_parts[1::2])) == [
        (2, 1), (1, 10), (0, 1), (0, 1), (3, 1), (0, 1),
        (0, 1), (0, 1), (4, 1),
    ]

    kind, count, neutral_data = _primary_tag_data(payload, 50728)
    assert kind == 5 and count == 3
    assert struct.unpack("<IIIIII", neutral_data) == (1, 2, 1, 1, 2, 3)
    assert _primary_tag_data(payload, 50708) == (2, len(b"NIKON Z 8\0"), b"NIKON Z 8\0")
    ifd_offset = struct.unpack_from("<I", payload, 4)[0]
    ifd_count = struct.unpack_from("<H", payload, ifd_offset)[0]
    tag_codes = {
        struct.unpack_from("<H", payload, ifd_offset + 2 + index * 12)[0]
        for index in range(ifd_count)
    }
    assert 50934 not in tag_codes
    assert b'imprint:SourceCameraProfileName="Camera Flexible Color"' in payload
    assert b'imprint:NikonPictureControlName="Negative_Cy_01a"' in payload
    assert b"crs:CameraProfile=" not in payload
    assert b"Legacy Fake Selection" not in payload
    assert _primary_tag_data(payload, 50778) == (3, 1, struct.pack("<H", 21))


@pytest.mark.parametrize(
    "metadata",
    [
        {"DNGColorMatrix1": [1] * 9},  # incomplete color triplet
        {
            "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "DNGAsShotNeutral": [1, 1, 1],
        },  # missing unique model
        {
            "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1, 2],
            "DNGAsShotNeutral": [1, 1, 1],
            "DNGUniqueCameraModel": "Camera",
        },  # wrong matrix size
        {
            "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, float("nan")],
            "DNGAsShotNeutral": [1, 1, 1],
            "DNGUniqueCameraModel": "Camera",
        },  # non-finite matrix
        {
            "DNGColorMatrix1": [1, 2, 3, 2, 4, 6, 1, 0, 1],
            "DNGAsShotNeutral": [1, 1, 1],
            "DNGUniqueCameraModel": "Camera",
        },  # singular matrix
        {
            "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "DNGAsShotNeutral": [1, 0, 1],
            "DNGUniqueCameraModel": "Camera",
        },  # invalid white balance
        {
            "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "DNGAsShotNeutral": [1, float("inf"), 1],
            "DNGUniqueCameraModel": "Camera",
        },  # non-finite white balance
        {
            "DNGColorMatrix1": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "DNGAsShotNeutral": [1, 1, 1],
            "DNGUniqueCameraModel": "  ",
        },  # empty camera identity
    ],
)
def test_invalid_or_incomplete_camera_color_metadata_fails_without_output(
    tmp_path: Path, metadata: dict[str, object]
):
    with pytest.raises(ValueError):
        write_linear_dng(np.zeros((8, 8, 3), np.uint16), "bad.nef", tmp_path, metadata)
    assert not list(tmp_path.glob("*.dng"))
    assert not list(tmp_path.glob("*.tmp"))


def test_nikon_picture_control_is_safely_recorded_in_xmp(tmp_path: Path):
    value = 'Negative_Cy_01a & <custom> "safe"'
    path = write_linear_dng(
        np.full((8, 8, 3), 10000, np.uint16),
        "scene.nef",
        tmp_path,
        {"NikonPictureControlName": value},
    )
    assert (
        b'imprint:NikonPictureControlName="Negative_Cy_01a &amp; &lt;custom&gt; &quot;safe&quot;"'
        in path.read_bytes()
    )


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


def test_exiftool_validates_camera_native_color_profile_when_available(tmp_path: Path):
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return
    path = write_linear_dng(
        np.full((32, 48, 3), 28000, np.uint16),
        "native.nef",
        tmp_path,
        {
            "DNGColorMatrix1": [2, "1/10", 0, 0, 3, 0, 0, 0, 4],
            "DNGAsShotNeutral": ["1/2", 1, "2/3"],
            "DNGUniqueCameraModel": "NIKON Z 8",
            "SourceCameraProfileName": "Camera Flexible Color",
            "NikonPictureControlName": "Negative_Cy_01a",
        },
    )
    result = subprocess.run(
        [
            exiftool,
            "-validate",
            "-warning",
            "-error",
            "-ColorMatrix1",
            "-AsShotNeutral",
            "-CalibrationIlluminant1",
            "-UniqueCameraModel",
            "-AsShotProfileName",
            "-XMP-crs:CameraProfile",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Validate" in result.stdout and "OK" in result.stdout
    assert "NIKON Z 8" in result.stdout
    assert "AsShotProfileName" not in result.stdout
    assert "CameraProfile" not in result.stdout


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


def test_write_failure_removes_temporary_dng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fail_pack(*args: object, **kwargs: object) -> bytes:
        raise OSError("injected strip write failure")

    monkeypatch.setattr(dng_writer, "_packed_strip_bytes", fail_pack)
    with pytest.raises(OSError, match="injected strip write failure"):
        write_linear_dng(np.zeros((16, 16, 3), np.uint16), "broken.nef", tmp_path)
    assert not list(tmp_path.glob("*.dng"))
    assert not list(tmp_path.glob("*.tmp"))
