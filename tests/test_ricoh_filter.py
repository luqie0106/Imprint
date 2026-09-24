from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ricoh_filter
from image_io import OUTPUT_DIR_NAME
from dehaze import DehazeParams


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


def test_unparseable_existing_xmp_is_reported_and_never_changed(tmp_path: Path):
    photo = tmp_path / "sample.jpg"
    sidecar = tmp_path / "SAMPLE.XMP"
    photo.write_bytes(b"photo")
    sidecar.write_bytes(b"user sidecar")

    result = ricoh_filter.apply_ricoh_preset([str(photo)], "gr3_standard")

    assert result["total"] == 1
    assert result["written"] == 0
    assert result["failed"] == 1
    assert result["files"][0]["status"] == "failed"
    assert result["files"][0]["error"] == "现有 XMP 无法解析，未修改"
    assert sidecar.read_bytes() == b"user sidecar"
    assert {path.name for path in tmp_path.iterdir() if path.suffix.lower() == ".xmp"} == {"SAMPLE.XMP"}


def test_ricoh_and_dehaze_settings_merge_without_losing_unknown_xmp(tmp_path: Path):
    photo = tmp_path / "sample.jpg"
    sidecar = tmp_path / "SAMPLE.XMP"
    photo.write_bytes(b"photo")
    sidecar.write_text(
        '''<?xpacket begin=""?><x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:custom="https://example.test/ns/">
        <rdf:RDF><rdf:Description rdf:about="" custom:KeepMe="retain"/></rdf:RDF></x:xmpmeta><?xpacket end="w"?>''',
        encoding="utf-8",
    )

    dehaze = DehazeParams(strength=0.23, local_contrast=0.61)
    assert ricoh_filter.write_dehaze_settings(photo, dehaze.__dict__) == "SAMPLE.XMP"
    assert ricoh_filter.write_ricoh_preset(photo, "gr3_standard") == "SAMPLE.XMP"

    root = ricoh_filter._parse_xmp(sidecar.read_bytes())
    description = ricoh_filter._description(root)
    assert description is not None
    assert description.attrib["{https://example.test/ns/}KeepMe"] == "retain"
    assert description.attrib["{" + ricoh_filter._CRS_NS + "}Contrast2012"] == "8"
    settings = ricoh_filter.read_photo_settings(photo)
    assert settings["ricoh_preset_id"] == "gr3_standard"
    assert settings["dehaze_params"]["strength"] == 0.23
    assert settings["dehaze_params"]["local_contrast"] == 0.61


def test_switching_preset_clears_fields_absent_from_new_preset(tmp_path: Path, monkeypatch):
    photo = tmp_path / "sample.jpg"
    photo.touch()

    ricoh_filter.write_ricoh_preset(photo, "gr2_positive_film")
    first_root = ricoh_filter._parse_xmp(photo.with_suffix(".xmp").read_bytes())
    first_description = ricoh_filter._description(first_root)
    assert first_description is not None
    assert "{" + ricoh_filter._CRS_NS + "}Temperature" not in first_description.attrib
    # An older sidecar may still contain the invalid placeholder; switching
    # presets must remove it as well.
    first_description.set("{" + ricoh_filter._CRS_NS + "}Temperature", "0")
    photo.with_suffix(".xmp").write_bytes(ricoh_filter._serialize_xmp(first_root))

    old_payload = ricoh_filter._preset_payload
    next_preset = (
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="{ricoh_filter._RDF_NS}" '
        f'xmlns:crs="{ricoh_filter._CRS_NS}"><rdf:RDF><rdf:Description rdf:about="" '
        'crs:Contrast2012="8"/></rdf:RDF></x:xmpmeta>'
    ).encode()
    temperature_key = "{" + ricoh_filter._CRS_NS + "}Temperature"
    monkeypatch.setattr(ricoh_filter, "_PRESET_PROCESSING_KEYS", ({temperature_key}, set()))
    monkeypatch.setattr(
        ricoh_filter,
        "_preset_payload",
        lambda preset_id: next_preset if preset_id == "gr3_positive_film" else old_payload(preset_id),
    )
    merged = ricoh_filter._merge_preset_payload(photo.with_suffix(".xmp").read_bytes(), "gr3_positive_film")
    second_root = ricoh_filter._parse_xmp(merged)
    second_description = ricoh_filter._description(second_root)
    assert second_description is not None
    assert temperature_key not in second_description.attrib
    assert second_description.attrib["{" + ricoh_filter._CRS_NS + "}Contrast2012"] == "8"


def test_basic_controls_write_acr_fields_and_keep_as_shot_white_balance(tmp_path: Path):
    photo = tmp_path / "sample.nef"
    photo.touch()
    basic = {key: 0.0 for key in ricoh_filter._BASIC_FIELDS}
    basic.update(exposure=0.75, highlights=-12, shadows=20)
    ricoh_filter.write_ricoh_preset(photo, "gr3_standard", basic)
    root = ricoh_filter._parse_xmp(photo.with_suffix(".xmp").read_bytes())
    description = ricoh_filter._description(root)
    assert description is not None
    crs = "{" + ricoh_filter._CRS_NS + "}"
    assert description.attrib[crs + "WhiteBalance"] == "As Shot"
    assert crs + "Temperature" not in description.attrib
    assert crs + "Tint" not in description.attrib
    assert description.attrib[crs + "Exposure2012"] == "0.75"
    assert float(description.attrib[crs + "Highlights2012"]) == ricoh_filter._preset_controls("gr3_standard")["highlights"] - 12
    assert ricoh_filter.read_photo_settings(photo)["basic_params"] == basic


def test_malformed_xmp_is_never_overwritten_by_dehaze_write(tmp_path: Path):
    photo = tmp_path / "sample.jpg"
    sidecar = tmp_path / "sample.xmp"
    photo.touch()
    sidecar.write_bytes(b"<broken")

    try:
        ricoh_filter.write_dehaze_settings(photo, DehazeParams().__dict__)
    except Exception as exc:
        assert "sample" not in str(exc)
    else:
        raise AssertionError("invalid sidecar should not be overwritten")
    assert sidecar.read_bytes() == b"<broken"


def test_missing_source_never_creates_orphan_xmp(tmp_path: Path):
    missing = tmp_path / "moved-away.jpg"

    for write in (
        lambda: ricoh_filter.write_dehaze_settings(missing, DehazeParams().__dict__),
        lambda: ricoh_filter.write_ricoh_preset(missing, "gr3_standard"),
    ):
        try:
            write()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing photo should reject stale session writes")
    assert not (tmp_path / "moved-away.xmp").exists()


def test_ricoh_preview_effect_preserves_input_and_supports_bw_preset():
    import numpy as np

    source = np.array([[[20, 100, 220], [240, 120, 10]]], dtype=np.uint8)
    original = source.copy()

    result = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_high_contrast_bw")

    assert source.tolist() == original.tolist()
    assert result.dtype == np.uint8
    assert result.shape == source.shape
    assert np.array_equal(result[..., 0], result[..., 1])
    assert np.array_equal(result[..., 1], result[..., 2])


def _neutral_preview_controls(**updates):
    identity = ((0.0, 0.0), (255.0, 255.0))
    controls = {
        "exposure": 0.0, "contrast": 0.0, "saturation": 0.0, "vibrance": 0.0,
        "temperature": 0.0, "tint": 0.0, "shadows": 0.0, "highlights": 0.0,
        "whites": 0.0, "blacks": 0.0, "grayscale": False,
        "split_shadow_hue": 0.0, "split_shadow_saturation": 0.0,
        "split_highlight_hue": 0.0, "split_highlight_saturation": 0.0,
        "split_balance": 0.0, "grade_shadow_hue": 0.0,
        "grade_shadow_saturation": 0.0, "grade_midtone_hue": 0.0,
        "grade_midtone_saturation": 0.0, "grade_highlight_hue": 0.0,
        "grade_highlight_saturation": 0.0, "grade_global_hue": 0.0,
        "grade_global_saturation": 0.0, "grade_shadow_luminance": 0.0,
        "grade_midtone_luminance": 0.0, "grade_highlight_luminance": 0.0,
        "grade_global_luminance": 0.0, "grade_blending": 50.0,
        "hue_adjustments": {color: 0.0 for color in ricoh_filter._HSL_COLORS},
        "saturation_adjustments": {color: 0.0 for color in ricoh_filter._HSL_COLORS},
        "luminance_adjustments": {color: 0.0 for color in ricoh_filter._HSL_COLORS},
        "tone_curve": identity, "red_curve": identity,
        "green_curve": identity, "blue_curve": identity,
    }
    controls.update(updates)
    return controls


def test_ricoh_preview_applies_xmp_hsl_color_ranges(monkeypatch):
    import numpy as np

    controls = _neutral_preview_controls()
    controls["saturation_adjustments"]["Red"] = 100.0
    controls["luminance_adjustments"]["Red"] = 30.0
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: controls)
    source = np.array([[[200, 80, 80], [80, 200, 80], [128, 128, 128]]], dtype=np.uint8)

    result = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")

    assert result[0, 0, 1] < source[0, 0, 1]
    assert result[0, 0, 2] < source[0, 0, 2]
    assert np.max(np.abs(result[0, 1].astype(np.int16) - source[0, 1])) <= 1
    assert np.array_equal(result[0, 2], source[0, 2])


def test_hsl_weights_are_smooth_and_blue_range_covers_cyan():
    import cv2
    import numpy as np

    controls = _neutral_preview_controls()
    controls["saturation_adjustments"]["Blue"] = 60.0
    hsv = np.array([[[190.0, 0.7, 0.7], [200.0, 0.7, 0.7],
                     [210.0, 0.7, 0.7], [220.0, 0.7, 0.7]]], dtype=np.float32)
    source = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    original = source.copy()

    result = ricoh_filter._apply_hsl_adjustments(source, controls, np)
    result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

    assert np.all(result_hsv[0, 1:4, 1] > hsv[0, 1:4, 1])
    # The Blue center is calibrated at 210 degrees, so its response peaks
    # there while neighboring cyan and blue hues change continuously.
    increases = result_hsv[0, :, 1] - hsv[0, :, 1]
    assert increases[2] > increases[0]
    assert np.max(np.abs(np.diff(increases))) < 0.08
    assert np.array_equal(source, original)


def test_hsl_hue_adjustment_changes_gradually_across_adjacent_hues():
    import cv2
    import numpy as np

    controls = _neutral_preview_controls()
    controls["hue_adjustments"]["Blue"] = 30.0
    hsv = np.array([[[208.0, 0.8, 0.7], [209.0, 0.8, 0.7],
                     [210.0, 0.8, 0.7], [211.0, 0.8, 0.7],
                     [212.0, 0.8, 0.7]]], dtype=np.float32)
    source = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    result = ricoh_filter._apply_hsl_adjustments(source, controls, np)
    result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    hue_deltas = (result_hsv[..., 0] - hsv[..., 0] + 180.0) % 360.0 - 180.0

    assert np.all(hue_deltas > 0.0)
    assert np.max(np.abs(np.diff(hue_deltas[0]))) < 0.6


def test_color_grading_masks_softly_cross_and_honor_blending_and_balance():
    import numpy as np

    controls = _neutral_preview_controls()
    luminance = np.array([[0.15, 0.35, 0.5, 0.65, 0.85]], dtype=np.float32)
    shadow, midtone, highlight = ricoh_filter._grading_masks(luminance, controls, np)

    assert shadow[0, 0] > shadow[0, 1] > shadow[0, 2] > shadow[0, 3]
    assert highlight[0, 1] == highlight[0, 2] == 0.0
    assert 0.0 < highlight[0, 3] < highlight[0, 4]
    assert midtone[0, 2] > midtone[0, 0]
    assert midtone[0, 3] > midtone[0, 4]
    assert 0.0 < shadow[0, 2] < 1.0
    assert 0.0 < highlight[0, 3] < 1.0

    sample = np.array([[0.35, 0.65]], dtype=np.float32)
    narrow = dict(controls, grade_blending=0.0)
    wide = dict(controls, grade_blending=100.0)
    narrow_shadow, _, narrow_highlight = ricoh_filter._grading_masks(sample, narrow, np)
    wide_shadow, _, wide_highlight = ricoh_filter._grading_masks(sample, wide, np)
    default_shadow, _, default_highlight = ricoh_filter._grading_masks(sample, controls, np)
    assert narrow_shadow[0, 0] > default_shadow[0, 0] > wide_shadow[0, 0]
    assert narrow_highlight[0, 1] < default_highlight[0, 1] < wide_highlight[0, 1]

    positive = dict(controls, split_balance=100.0)
    negative = dict(controls, split_balance=-100.0)
    positive_shadow, _, _ = ricoh_filter._grading_masks(
        np.array([[0.5]], dtype=np.float32), positive, np,
    )
    negative_shadow, _, _ = ricoh_filter._grading_masks(
        np.array([[0.5]], dtype=np.float32), negative, np,
    )
    assert positive_shadow[0, 0] < shadow[0, 2] < negative_shadow[0, 0]
    balance_sample = np.array([[0.62]], dtype=np.float32)
    _, _, positive_highlight = ricoh_filter._grading_masks(balance_sample, positive, np)
    _, _, negative_highlight = ricoh_filter._grading_masks(balance_sample, negative, np)
    _, _, default_highlight = ricoh_filter._grading_masks(balance_sample, controls, np)
    assert positive_highlight[0, 0] > default_highlight[0, 0] > negative_highlight[0, 0]


def test_color_grading_zero_is_identity_and_nonfinite_input_is_bounded():
    import numpy as np

    controls = _neutral_preview_controls()
    source = np.array([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
                        [0.2, 0.4, 0.8]]], dtype=np.float32)
    original = source.copy()
    assert np.array_equal(ricoh_filter._apply_color_grading(source, controls, np), source)
    assert np.array_equal(source, original)

    controls.update(grade_global_hue=float("inf"), grade_global_saturation=1000.0)
    contaminated = np.array([[[np.nan, 0.4, 0.7], [np.inf, 0.2, 0.1]]], dtype=np.float32)
    contaminated_original = contaminated.copy()
    result = ricoh_filter._apply_color_grading(contaminated, controls, np)
    assert np.isfinite(result).all()
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert np.array_equal(contaminated, contaminated_original, equal_nan=True)


def test_ricoh_preview_merges_basic_values_before_color_processing(monkeypatch):
    import numpy as np

    controls = _neutral_preview_controls()
    controls["saturation_adjustments"]["Blue"] = -60.0
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: controls.copy())
    source = np.array([[[30, 65, 115]]], dtype=np.uint8)
    adjustments = {key: 0.0 for key in ricoh_filter._BASIC_FIELDS}
    adjustments["exposure"] = 1.0

    merged = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard", adjustments)
    controls["exposure"] = 1.0
    absolute = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")

    assert np.array_equal(merged, absolute)
    assert np.array_equal(source, np.array([[[30, 65, 115]]], dtype=np.uint8))


def test_ricoh_preview_applies_split_toning_and_pv2012_curve(monkeypatch):
    import numpy as np

    controls = _neutral_preview_controls(
        split_shadow_hue=210.0, split_shadow_saturation=100.0,
        split_balance=100.0,
        grade_midtone_hue=35.0, grade_midtone_saturation=100.0,
        tone_curve=((0.0, 8.0), (128.0, 142.0), (255.0, 248.0)),
    )
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: controls)
    source = np.array([[[48, 48, 48], [128, 128, 128], [230, 230, 230]]], dtype=np.uint8)

    result = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")

    assert result[0, 0, 2] > result[0, 0, 0]
    assert result[0, 1, 0] > result[0, 1, 2]
    assert result[0, 0].mean() > source[0, 0].mean()
    assert result[0, 1, 0] > source[0, 1, 0]
    assert np.isfinite(result).all()


def test_ricoh_master_curve_changes_luminance_without_shifting_color_and_keeps_gray(monkeypatch):
    import numpy as np

    curve = ((0.0, 0.0), (32.0, 64.0), (128.0, 128.0), (255.0, 255.0))
    controls = _neutral_preview_controls(tone_curve=curve)
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: controls)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    x_curve = np.asarray([point[0] / 255.0 for point in curve], dtype=np.float32)
    y_curve = np.asarray([point[1] / 255.0 for point in curve], dtype=np.float32)

    for dtype, multiplier in ((np.uint8, 1), (np.uint16, 257)):
        source = np.array([[[20, 15, 70]]], dtype=np.uint8).astype(dtype) * multiplier
        original = source.copy()
        maximum = float(np.iinfo(dtype).max)
        before = source[0, 0].astype(np.float32) / maximum
        before_luminance = float(before @ weights)
        target_luminance = float(np.interp(before_luminance, x_curve, y_curve))
        expected = before * (target_luminance / before_luminance)

        result = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")

        after = result[0, 0].astype(np.float32) / maximum
        assert result.dtype == dtype
        assert target_luminance > before_luminance
        assert abs(float(after @ weights) - target_luminance) <= 2.0 / maximum
        assert np.allclose(after, expected, atol=2.0 / maximum)
        assert np.array_equal(source, original)

        gray = np.full((1, 1, 3), round(40 * maximum / 255), dtype=dtype)
        gray_result = ricoh_filter.apply_ricoh_preview_effect(gray, "gr3_standard")
        assert gray_result[0, 0, 0] == gray_result[0, 0, 1] == gray_result[0, 0, 2]


def test_ricoh_preview_supports_uint16_and_does_not_mutate_input(monkeypatch):
    import numpy as np

    curve = ricoh_filter._preset_controls("gr3_positive_film")["tone_curve"]
    assert len(curve) > 2
    assert ricoh_filter._preset_controls("gr3_positive_film")["hue_adjustments"]["Green"] == 7.0
    controls = _neutral_preview_controls(
        hue_adjustments=ricoh_filter._preset_controls("gr3_positive_film")["hue_adjustments"],
        saturation_adjustments=ricoh_filter._preset_controls("gr3_positive_film")["saturation_adjustments"],
        luminance_adjustments=ricoh_filter._preset_controls("gr3_positive_film")["luminance_adjustments"],
        tone_curve=curve,
        split_shadow_hue=190.0, split_shadow_saturation=10.0,
        split_highlight_hue=43.0, split_highlight_saturation=8.0,
        grade_midtone_hue=36.0, grade_midtone_saturation=2.0,
    )
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: controls)
    source = np.array(
        [[[0, 1000, 5000], [16000, 32000, 50000], [65535, 40000, 20000]]],
        dtype=np.uint16,
    )
    original = source.copy()

    result = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_positive_film")

    assert result.dtype == np.uint16
    assert result.shape == source.shape
    assert np.isfinite(result).all()
    assert np.array_equal(source, original)
    assert not np.array_equal(result, source)


def test_ricoh_preview_strong_shadows_compress_chroma_without_shifting_hue(monkeypatch):
    import numpy as np

    controls = _neutral_preview_controls(shadows=100.0, blacks=100.0)
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: controls)
    source = np.array(
        [[[0, 0, 0], [24, 72, 120], [128, 128, 128], [255, 255, 255]]],
        dtype=np.uint8,
    )
    original = source.copy()

    result = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    before = source[0, 1].astype(np.float32) / 255.0
    before_luminance = float(before @ weights)
    shadow_mask = np.clip((0.62 - before_luminance) / 0.62, 0.0, 1.0) ** 1.5
    target_luminance = before_luminance + 0.30 * shadow_mask
    proportional_lift = before * (target_luminance / before_luminance)
    proportional_lift_luminance = float(proportional_lift @ weights)
    proportional_chroma = proportional_lift - proportional_lift_luminance
    after = result[0, 1].astype(np.float32) / 255.0
    after_luminance = float(after @ weights)
    after_chroma = after - after_luminance
    before_chroma = before - before_luminance

    assert after_luminance > before_luminance
    assert abs(after_luminance - target_luminance) <= 2 / 255
    assert np.linalg.norm(after_chroma) < np.linalg.norm(proportional_chroma)
    assert np.all(np.diff(after) > 0)  # red < green < blue remains true
    assert np.allclose(
        after_chroma / np.linalg.norm(after_chroma),
        before_chroma / np.linalg.norm(before_chroma),
        atol=0.02,
    )

    # Black and gray stay neutral, and white stays at the output boundary.
    assert 0 < result[0, 0, 0] < 255
    assert result[0, 0, 0] == result[0, 0, 1] == result[0, 0, 2]
    assert result[0, 2, 0] == result[0, 2, 1] == result[0, 2, 2]
    assert np.array_equal(result[0, 3], np.array([255, 255, 255], dtype=np.uint8))
    assert result.dtype == source.dtype
    assert result.shape == source.shape
    assert np.array_equal(source, original)


def test_ricoh_preview_zero_luminance_lift_is_identity(monkeypatch):
    import numpy as np

    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: _neutral_preview_controls())
    for dtype in (np.uint8, np.uint16):
        source = np.array([[[24, 72, 120], [128, 128, 128]]], dtype=dtype)
        original = source.copy()

        result = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")

        assert np.array_equal(result, source)
        assert np.array_equal(source, original)
        assert result.dtype == dtype


def test_ricoh_preview_tone_controls_move_shadows_and_highlights_in_expected_direction(monkeypatch):
    import numpy as np

    controls = _neutral_preview_controls(shadows=100.0, blacks=100.0,
                                         highlights=100.0, whites=100.0)
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: controls)
    source = np.array([[[24, 60, 96], [220, 160, 80]]], dtype=np.uint8)

    lifted = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")
    assert lifted[0, 0].mean() > source[0, 0].mean()
    assert lifted[0, 1].mean() > source[0, 1].mean()
    assert np.all(lifted <= 255)
    assert np.all(lifted >= 0)

    controls.update(shadows=-100.0, blacks=-100.0, highlights=-100.0, whites=-100.0)
    lowered = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")
    assert lowered[0, 0].mean() < source[0, 0].mean()
    assert lowered[0, 1].mean() < source[0, 1].mean()
    assert np.all(lowered <= 255)
    assert np.all(lowered >= 0)
