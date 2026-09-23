from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ricoh_filter
from image_io import OUTPUT_DIR_NAME


def test_catalog_exposes_only_ten_gr2_and_gr3_presets():
    presets = ricoh_filter.list_ricoh_presets()

    assert len(presets) == 10
    assert len({preset["id"] for preset in presets}) == 10
    assert {preset["model"] for preset in presets} == {"GR2", "GR3"}
    assert all(set(preset) == {"id", "model", "name", "description"} for preset in presets)


def test_export_is_valid_xmp_without_preset_identity_and_keeps_curve():
    payload = ricoh_filter._preset_payload("gr3_positive_film")
    root = ET.fromstring(payload)
    crs = "{" + ricoh_filter._CRS_NS + "}"
    values = [element for element in root.iter()]

    assert all(not key.startswith(crs) or key[len(crs):].casefold() not in {
        "presettype", "uuid", "name", "shortname", "group", "description",
        "cluster", "copyright", "contactinfo",
    } for element in values for key in element.attrib)
    assert not any(
        element.tag.startswith(crs)
        and element.tag[len(crs):].casefold() in {"name", "shortname", "group", "description"}
        for element in values
    )
    assert any(key == crs + "Exposure2012" for element in values for key in element.attrib)
    assert any(element.tag == crs + "ToneCurvePV2012" for element in values)


def test_apply_mixed_paths_writes_once_per_stem_and_rejects_unsupported(tmp_path: Path):
    folder = tmp_path / "photos"
    folder.mkdir()
    jpeg = folder / "IMG_001.JPG"
    raw = folder / "IMG_001.NEF"
    png = folder / "IMG_002.png"
    unsupported = tmp_path / "notes.txt"
    generated_dir = folder / OUTPUT_DIR_NAME
    generated_dir.mkdir()
    generated_dng = generated_dir / "generated.dng"
    originals = {path: b"original bytes" for path in (jpeg, raw, png, unsupported, generated_dng)}
    for path, data in originals.items():
        path.write_bytes(data)

    result = ricoh_filter.apply_ricoh_preset([str(folder), str(unsupported)], "gr2_street_positive")

    assert result["total"] == 3
    assert result["written"] == 2
    assert result["skipped"] == 0
    assert result["failed"] == 1
    assert (folder / "IMG_001.xmp").is_file()
    assert (folder / "IMG_002.xmp").is_file()
    assert not (generated_dir / "generated.xmp").exists()
    assert next(item for item in result["files"] if item["name"] == "notes.txt")["error"] == "不支持的图片格式"
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_existing_xmp_is_skipped_case_insensitively_and_never_changed(tmp_path: Path):
    photo = tmp_path / "sample.jpg"
    sidecar = tmp_path / "SAMPLE.XMP"
    photo.write_bytes(b"photo")
    sidecar.write_bytes(b"user sidecar")

    result = ricoh_filter.apply_ricoh_preset([str(photo)], "gr3_standard")

    assert result == {
        "total": 1,
        "written": 0,
        "skipped": 1,
        "failed": 0,
        "files": [{"name": "sample.xmp", "status": "skipped", "error": "已存在同名 XMP，未覆盖"}],
    }
    assert sidecar.read_bytes() == b"user sidecar"
    assert {path.name for path in tmp_path.iterdir() if path.suffix.lower() == ".xmp"} == {"SAMPLE.XMP"}
