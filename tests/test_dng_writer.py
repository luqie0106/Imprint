from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dng_writer import write_linear_dng


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


def test_failure_leaves_no_dng(tmp_path: Path):
    bad = np.zeros((10, 10), dtype=np.uint16)
    try:
        write_linear_dng(bad, "broken.jpg", tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input should fail")
    assert not list(tmp_path.glob("*.dng"))
