"""Synthetic DNG round-trip and output-commit regressions."""

from __future__ import annotations

from io import BytesIO
import multiprocessing
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image
import pytest
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dng_writer


def _profile(white_level=65535):
    return {
        "Make": "Imprint",
        "Model": "Synthetic Camera",
        "DateTimeOriginal": "2026:11:07 10:20:30",
        "ExposureTime": (1, 125),
        "ISO": 640,
        "DNGColorMatrix1": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        "DNGAsShotNeutral": (1.0, 1.0, 1.0),
        "DNGUniqueCameraModel": "Imprint Synthetic Camera",
        "DNGBlackLevel": (0, 0, 0, 0),
        "DNGWhiteLevel": white_level,
    }


def _ifd(payload, offset=8):
    assert payload[:4] == b"II*\x00"
    count = struct.unpack_from("<H", payload, offset)[0]
    values = {}
    formats = {1: "B", 3: "H", 4: "I", 5: "II", 7: "B", 10: "ii"}
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 10: 8}
    for index in range(count):
        position = offset + 2 + index * 12
        code, kind, length, pointer = struct.unpack_from("<HHII", payload, position)
        if kind not in sizes:
            continue
        size = sizes[kind] * length
        data_offset = position + 8 if size <= 4 else pointer
        data = payload[data_offset:data_offset + size]
        assert len(data) == size
        if kind == 2:
            values[code] = data.rstrip(b"\0").decode("ascii")
        else:
            values[code] = struct.unpack("<" + formats[kind] * length, data)
    next_ifd = struct.unpack_from("<I", payload, offset + 2 + count * 12)[0]
    return values, next_ifd


def _export(kind, image, source, output_dir, *, bits=16):
    if kind == "enhanced":
        cfa = image[:, :, 0] % ((1 << bits) - 1)
        return dng_writer.write_enhanced_dng(
            image, cfa, [[0, 1], [1, 2]], source, output_dir,
            _profile((1 << bits) - 1), raw_bits_per_sample=bits, compression="none",
        )
    return dng_writer.write_linear_dng(
        image, source, output_dir, _profile(), bits_per_sample=bits,
        name_suffix="_ricoh" if kind == "ricoh" else "_dehaze",
    )


@pytest.mark.parametrize("bits", [8, 10, 12, 14, 16])
@pytest.mark.parametrize("height,width", [(64, 64), (64, 65), (64, 66), (4000, 65)])
def test_linear_dng_round_trip_preserves_row_boundaries(tmp_path, bits, height, width):
    image = np.random.default_rng(23).integers(0, 65536, (height, width, 3), dtype=np.uint16)
    original = image.copy()
    output = _export("linear", image, "sample.png", tmp_path, bits=bits)

    with rawpy.imread(str(output)) as raw:
        actual = raw.raw_image_visible[:, :, :3].copy()
    maximum = (1 << bits) - 1
    expected = ((image.astype(np.uint32) * maximum + 32767) // 65535).astype(np.uint16)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, original)

    tags, _ = _ifd(output.read_bytes())
    expected_size = image.shape[0] * ((width * 3 * bits + 7) // 8)
    assert sum(tags[279]) == expected_size


@pytest.mark.parametrize("bits", [8, 10, 12, 14, 16])
@pytest.mark.parametrize("width", [64, 65, 66])
def test_enhanced_dng_cfa_round_trip_preserves_row_boundaries(tmp_path, bits, width):
    image = np.random.default_rng(29).integers(0, 65536, (64, width, 3), dtype=np.uint16)
    original = image.copy()
    expected = image[:, :, 0] % ((1 << bits) - 1)
    output = _export("enhanced", image, "sample.nef", tmp_path, bits=bits)

    with rawpy.imread(str(output)) as raw:
        np.testing.assert_array_equal(raw.raw_image_visible, expected)
    np.testing.assert_array_equal(image, original)
    payload = output.read_bytes()
    root, _ = _ifd(payload)
    cfa, _ = _ifd(payload, root[330][0])
    assert sum(cfa[279]) == image.shape[0] * ((width * bits + 7) // 8)


@pytest.mark.parametrize("kind", ["linear", "enhanced"])
def test_dng_has_real_image_tags_preview_and_exif(tmp_path, kind):
    image = np.full((64, 66, 3), 24000, dtype=np.uint16)
    output = _export(kind, image, "sample.png", tmp_path)
    payload = output.read_bytes()
    root, preview_offset = _ifd(payload)
    assert root[50706] == ((1, 4, 0, 0) if kind == "linear" else (1, 7, 0, 0))
    assert root[271] == "Imprint"
    assert root[272] == "Synthetic Camera"
    assert root[274] == (1,)
    linear = root
    if kind == "enhanced":
        cfa, _ = _ifd(payload, root[330][0])
        linear, _ = _ifd(payload, root[330][1])
        assert cfa[262] == (32803,)
        assert cfa[258] == (16,)
        assert cfa[259] == (1,)
    assert linear[262] == (34892,)
    assert linear[258] == (16, 16, 16)
    assert linear[259] == (1,)
    exif, _ = _ifd(payload, root[34665][0])
    assert exif[36867] == "2026:11:07 10:20:30"
    assert exif[33434] == (1, 125)
    assert exif[34855] == (640,)
    preview, _ = _ifd(payload, preview_offset)
    assert preview[259] == (7,)
    start, size = preview[273][0], preview[279][0]
    with Image.open(BytesIO(payload[start:start + size])) as jpeg:
        assert jpeg.format == "JPEG"
        assert jpeg.size == (66, 64)
        jpeg.load()
    assert list(tmp_path.glob("*.tmp")) == []


def _concurrent_export(directory, kind, value, barrier, results):
    original_link = dng_writer.os.link
    original_replace = dng_writer.os.replace
    synchronized = False

    def synchronize():
        nonlocal synchronized
        if not synchronized:
            synchronized = True
            barrier.wait(timeout=10)

    def link(*args, **kwargs):
        synchronize()
        return original_link(*args, **kwargs)

    def replace(*args, **kwargs):
        synchronize()
        return original_replace(*args, **kwargs)

    dng_writer.os.link = link
    dng_writer.os.replace = replace
    try:
        image = np.full((64, 64, 3), value, dtype=np.uint16)
        output = _export(kind, image, "sample.png", directory)
        results.put((value, output.name))
    finally:
        dng_writer.os.link = original_link
        dng_writer.os.replace = original_replace


@pytest.mark.parametrize("kind", ["linear", "ricoh", "enhanced"])
def test_concurrent_processes_never_replace_an_output(tmp_path, kind):
    suffix = "_ricoh" if kind == "ricoh" else "_dehaze"
    existing = tmp_path / f"sample{suffix}.dng"
    existing.write_bytes(b"existing output")
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(2), context.Queue()
    processes = [
        context.Process(target=_concurrent_export, args=(str(tmp_path), kind, value, barrier, results))
        for value in (1000, 2000)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        assert [process.exitcode for process in processes] == [0, 0]
        exports = [results.get(timeout=2) for _ in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()
    assert len({name for _, name in exports}) == 2
    assert {name for _, name in exports} == {
        f"sample{suffix}_2.dng", f"sample{suffix}_3.dng",
    }
    for value, name in exports:
        with rawpy.imread(str(tmp_path / name)) as raw:
            actual = raw.raw_image_visible
            if kind != "enhanced":
                actual = actual[:, :, :3]
            assert np.all(actual == value)
    assert existing.read_bytes() == b"existing output"
    assert len(list(tmp_path.iterdir())) == 3


@pytest.mark.parametrize("kind", ["linear", "enhanced"])
@pytest.mark.parametrize("failure", ["packing", "commit"])
def test_failed_export_removes_temporary_files(tmp_path, monkeypatch, kind, failure):
    source = tmp_path / "sample.nef"
    source.write_bytes(b"unchanged source")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / "sample_dehaze.dng"
    existing.write_bytes(b"unchanged output")

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    if failure == "commit":
        monkeypatch.setattr(dng_writer.os, "link", fail)
    else:
        monkeypatch.setattr(dng_writer, "_packed_strip_bytes", fail)
    with pytest.raises(OSError, match="simulated write failure"):
        _export(kind, np.full((64, 64, 3), 2000, np.uint16), source, output_dir)
    assert list(output_dir.iterdir()) == [existing]
    assert existing.read_bytes() == b"unchanged output"
    assert source.read_bytes() == b"unchanged source"
