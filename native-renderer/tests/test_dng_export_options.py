from __future__ import annotations

import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import dng_writer


_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8, 11: 4, 12: 8}


def _read_ifd(path: Path, offset: int | None = None) -> tuple[dict[int, tuple[int, ...] | bytes], int]:
    payload = path.read_bytes()
    if offset is None:
        offset = struct.unpack_from("<I", payload, 4)[0]
    count = struct.unpack_from("<H", payload, offset)[0]
    tags: dict[int, tuple[int, ...] | bytes] = {}
    for index in range(count):
        entry_offset = offset + 2 + index * 12
        code, kind, item_count, value_offset = struct.unpack_from("<HHII", payload, entry_offset)
        byte_count = _TYPE_SIZES[kind] * item_count
        data = (
            payload[entry_offset + 8:entry_offset + 8 + byte_count]
            if byte_count <= 4
            else payload[value_offset:value_offset + byte_count]
        )
        if kind in {1, 2, 5, 7, 9, 10, 11, 12}:
            tags[code] = data
        elif kind == 3:
            tags[code] = struct.unpack("<" + "H" * item_count, data)
        elif kind == 4:
            tags[code] = struct.unpack("<" + "I" * item_count, data)
        else:
            raise AssertionError(f"test IFD reader does not support TIFF type {kind}")
    next_offset = struct.unpack_from("<I", payload, offset + 2 + count * 12)[0]
    return tags, next_offset


def _ifd_number(tags: dict[int, tuple[int, ...] | bytes], code: int) -> int:
    value = tags[code]
    assert isinstance(value, tuple)
    assert len(value) == 1
    return value[0]


def _decode_packed(data: bytes, sample_count: int, bits_per_sample: int) -> np.ndarray:
    if bits_per_sample == 16:
        return np.frombuffer(data, dtype="<u2", count=sample_count).copy()
    if bits_per_sample == 8:
        return np.frombuffer(data, dtype=np.uint8, count=sample_count).astype(np.uint16)
    bit_values = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:sample_count * bits_per_sample]
    bit_values = bit_values.reshape(sample_count, bits_per_sample)
    weights = 1 << np.arange(bits_per_sample - 1, -1, -1, dtype=np.uint16)
    return (bit_values @ weights).astype(np.uint16)


def _decode_pnm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    position = 0
    tokens: list[bytes] = []
    while len(tokens) < 4:
        while data[position:position + 1] in (b" ", b"\t", b"\r", b"\n"):
            position += 1
        if data[position:position + 1] == b"#":
            position = data.index(b"\n", position) + 1
            continue
        start = position
        while data[position:position + 1] not in (b" ", b"\t", b"\r", b"\n"):
            position += 1
        tokens.append(data[start:position])
    if data[position:position + 2] == b"\r\n":
        position += 2
    else:
        position += 1
    magic, width, height, maximum = tokens
    channels = 1 if magic == b"P5" else 3
    dtype = np.uint8 if int(maximum) <= 255 else ">u2"
    return np.frombuffer(data[position:], dtype=dtype).astype(np.uint16).reshape(
        int(height), int(width), channels
    )


def _decode_lossless_jpeg(data: bytes, shape: tuple[int, ...], tmp_path: Path) -> np.ndarray:
    decoder = shutil.which("djpeg")
    if decoder is None:
        pytest.skip("lossless JPEG decode check requires djpeg")
    jpeg_path = tmp_path / "strip.jpg"
    pnm_path = tmp_path / "strip.pnm"
    jpeg_path.write_bytes(data)
    subprocess.run(
        [decoder, "-pnm", "-outfile", str(pnm_path), str(jpeg_path)],
        check=True,
        capture_output=True,
    )
    decoded = _decode_pnm(pnm_path)
    if len(shape) == 2:
        decoded = decoded[:, :, 0]
    assert decoded.shape == shape
    return decoded


def _decode_strips(path: Path, tags: dict[int, tuple[int, ...] | bytes], samples_per_pixel: int) -> np.ndarray:
    bits_per_sample = tags[258]
    assert isinstance(bits_per_sample, tuple)
    assert len(set(bits_per_sample)) == 1
    bits = bits_per_sample[0]
    width = _ifd_number(tags, 256)
    height = _ifd_number(tags, 257)
    offsets = tags[273]
    counts = tags[279]
    assert isinstance(offsets, tuple) and isinstance(counts, tuple)
    payload = path.read_bytes()
    rows_per_strip = _ifd_number(tags, 278)
    decoded: list[np.ndarray] = []
    rows_left = height
    for offset, count in zip(offsets, counts):
        rows = min(rows_per_strip, rows_left)
        shape = (rows, width) if samples_per_pixel == 1 else (rows, width, samples_per_pixel)
        sample_count = rows * width * samples_per_pixel
        strip_bytes = payload[offset:offset + count]
        if _ifd_number(tags, 259) == 7:
            strip = _decode_lossless_jpeg(strip_bytes, shape, path.parent)
        else:
            strip = _decode_packed(strip_bytes, sample_count, bits).reshape(shape)
        decoded.append(strip)
        rows_left -= rows
    assert rows_left == 0
    return np.concatenate(decoded, axis=0)


def _strip_payloads(path: Path, tags: dict[int, tuple[int, ...] | bytes]) -> list[bytes]:
    offsets = tags[273]
    counts = tags[279]
    assert isinstance(offsets, tuple) and isinstance(counts, tuple)
    payload = path.read_bytes()
    return [payload[offset:offset + count] for offset, count in zip(offsets, counts)]


def _metadata(white_level: int = 4095) -> dict[str, object]:
    return {
        "DNGColorMatrix1": [0.7, -0.2, -0.1, -0.3, 1.1, 0.2, 0.1, -0.5, 0.8],
        "DNGAsShotNeutral": [1.0, 1.0, 1.0],
        "DNGUniqueCameraModel": "Test Camera",
        "DNGBlackLevel": [64, 64, 64, 64],
        "DNGWhiteLevel": white_level,
        "DNGDefaultCropOrigin": (1, 2),
        "DNGDefaultCropSize": (29, 21),
        "Make": "Imprint Test",
        "Model": "RGB output",
        "DateTimeOriginal": "2026:10:06 12:34:56",
        "Artist": "Imprint test suite",
    }


def _images() -> tuple[np.ndarray, np.ndarray]:
    height, width = 24, 32
    rgb = np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3)
    rgb = ((rgb * 97 + 113) & 0xFFFF).astype(np.uint16)
    cfa = (np.arange(height * width, dtype=np.uint16).reshape(height, width) * 7 + 65)
    return rgb, cfa


def _quantized(image: np.ndarray, bits: int) -> np.ndarray:
    maximum = (1 << bits) - 1
    return ((image.astype(np.uint32) * maximum + 32767) // 65535).astype(np.uint16)


def _check_exiftool(path: Path) -> None:
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        return
    result = subprocess.run(
        [exiftool, "-validate", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert re.search(r"Validate\s*:\s*OK", result.stdout)


def test_linear_dng_packs_selected_depth_and_preserves_preview_exif(tmp_path: Path) -> None:
    rgb, _ = _images()
    first = dng_writer.write_linear_dng(
        rgb, "sample.nef", tmp_path, _metadata(), bits_per_sample=10
    )
    second = dng_writer.write_linear_dng(
        rgb, "sample.nef", tmp_path, _metadata(), bits_per_sample=10
    )
    assert first.name == "sample_dehaze.dng"
    assert second.name == "sample_dehaze_2.dng"

    tags, preview_offset = _read_ifd(first)
    assert tags[258] == (10, 10, 10)
    assert _ifd_number(tags, 259) == 1
    assert tags[50717] == (1023, 1023, 1023)
    np.testing.assert_array_equal(_decode_strips(first, tags, 3), _quantized(rgb, 10))
    assert tags[34665]
    exif_offset = _ifd_number(tags, 34665)
    exif, _ = _read_ifd(first, exif_offset)
    assert exif[36867] == b"2026:10:06 12:34:56\0"
    assert preview_offset > 0
    preview, _ = _read_ifd(first, preview_offset)
    assert _ifd_number(preview, 259) == 7
    _check_exiftool(first)


def test_linear_dng_lossless_jpeg_matches_quantized_rgb(tmp_path: Path) -> None:
    rgb, _ = _images()
    path = dng_writer.write_linear_dng(
        rgb, "sample.nef", tmp_path, bits_per_sample=10, compression="lossless_jpeg"
    )
    tags, _ = _read_ifd(path)
    assert _ifd_number(tags, 259) == 7
    np.testing.assert_array_equal(_decode_strips(path, tags, 3), _quantized(rgb, 10))
    _check_exiftool(path)


def test_enhanced_dng_keeps_cfa_exact_and_quantizes_rgb(tmp_path: Path) -> None:
    rgb, cfa = _images()
    cfa = cfa % 4096
    path = dng_writer.write_enhanced_dng(
        rgb,
        cfa,
        np.array([[0, 1], [1, 2]], dtype=np.uint8),
        "sample.nef",
        tmp_path,
        _metadata(),
        bits_per_sample=10,
        raw_bits_per_sample=12,
        compression="none",
    )
    root, preview_offset = _read_ifd(path)
    assert isinstance(root[330], tuple)
    raw_tags, _ = _read_ifd(path, root[330][0])
    rgb_tags, _ = _read_ifd(path, root[330][1])
    assert raw_tags[258] == (12,)
    assert _ifd_number(raw_tags, 259) == 1
    assert raw_tags[50717] == (4095,)
    assert rgb_tags[258] == (10, 10, 10)
    assert _ifd_number(rgb_tags, 259) == 1
    assert rgb_tags[50717] == (1023, 1023, 1023)
    np.testing.assert_array_equal(_decode_strips(path, raw_tags, 1), cfa)
    np.testing.assert_array_equal(_decode_strips(path, rgb_tags, 3), _quantized(rgb, 10))
    assert preview_offset > 0
    assert _ifd_number(_read_ifd(path, preview_offset)[0], 259) == 7
    assert _ifd_number(root, 34665) > 0
    _check_exiftool(path)


def test_enhanced_dng_lossless_jpeg_matches_quantized_rgb(tmp_path: Path) -> None:
    rgb, cfa = _images()
    cfa = cfa % 4096
    path = dng_writer.write_enhanced_dng(
        rgb,
        cfa,
        np.array([[0, 1], [1, 2]], dtype=np.uint8),
        "sample.nef",
        tmp_path,
        _metadata(),
        bits_per_sample=12,
        raw_bits_per_sample=12,
        compression="lossless_jpeg",
    )
    root, _ = _read_ifd(path)
    assert isinstance(root[330], tuple)
    raw_tags, _ = _read_ifd(path, root[330][0])
    rgb_tags, _ = _read_ifd(path, root[330][1])
    assert _ifd_number(rgb_tags, 259) == 7
    np.testing.assert_array_equal(_decode_strips(path, raw_tags, 1), cfa)
    np.testing.assert_array_equal(_decode_strips(path, rgb_tags, 3), _quantized(rgb, 12))
    _check_exiftool(path)


def test_jpegxl_rejects_unrepresentable_low_bit_depth(tmp_path: Path) -> None:
    rgb, cfa = _images()
    with pytest.raises(ValueError, match="requires 16-bit"):
        dng_writer.write_enhanced_dng(
            rgb,
            cfa,
            np.array([[0, 1], [1, 2]], dtype=np.uint8),
            "sample.nef",
            tmp_path,
            _metadata(white_level=65535),
            bits_per_sample=12,
            raw_bits_per_sample=16,
            compression="jpegxl",
        )
    assert list(tmp_path.iterdir()) == []


def test_jpegxl_tags_match_16_bit_codestreams_for_both_writers(tmp_path: Path) -> None:
    rgb, cfa = _images()
    linear = dng_writer.write_linear_dng(
        rgb, "linear.nef", tmp_path, compression="jpegxl"
    )
    enhanced = dng_writer.write_enhanced_dng(
        rgb,
        cfa,
        np.array([[0, 1], [1, 2]], dtype=np.uint8),
        "enhanced.nef",
        tmp_path,
        _metadata(),
        raw_bits_per_sample=16,
        compression="jpegxl",
    )
    linear_tags, _ = _read_ifd(linear)
    root, _ = _read_ifd(enhanced)
    assert _ifd_number(linear_tags, 259) == 52546
    assert linear_tags[258] == (16, 16, 16)
    assert linear_tags[50706] == bytes((1, 7, 0, 0))
    assert isinstance(root[330], tuple)
    enhanced_tags, _ = _read_ifd(enhanced, root[330][1])
    assert _ifd_number(enhanced_tags, 259) == 52546
    assert enhanced_tags[258] == (16, 16, 16)

    djxl = shutil.which("djxl")
    for path, tags in ((linear, linear_tags), (enhanced, enhanced_tags)):
        strips = _strip_payloads(path, tags)
        assert all(strip.startswith(b"\xff\x0a") for strip in strips)
        if djxl is not None:
            decoded = tmp_path / f"{path.stem}.ppm"
            jxl = tmp_path / f"{path.stem}.jxl"
            jxl.write_bytes(strips[0])
            subprocess.run([djxl, str(jxl), str(decoded)], check=True, capture_output=True)
            expected = _ifd_number(tags, 257), _ifd_number(tags, 256), 3
            assert _decode_pnm(decoded).shape == expected
        _check_exiftool(path)


def test_failed_write_leaves_no_dng_or_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rgb, _ = _images()

    def fail_packing(*_args: object, **_kwargs: object) -> bytes:
        raise RuntimeError("injected strip packing failure")

    monkeypatch.setattr(dng_writer, "_packed_strip_bytes", fail_packing)
    with pytest.raises(RuntimeError, match="injected strip packing failure"):
        dng_writer.write_linear_dng(rgb, "sample.nef", tmp_path, bits_per_sample=10)
    assert list(tmp_path.iterdir()) == []
