from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import app_api
import ricoh_filter as xmp
from dehaze import DehazeParams

CURVES = (((0, 0), (64, 40), (128, 100), (255, 255)),) * 3


def description(photo):
    return xmp._description(xmp._parse_xmp(photo.with_suffix(".xmp").read_bytes()))


def test_curves_are_adobe_fields_repeatable_and_restore_existing_curves(tmp_path):
    photo = tmp_path / "sample.jpg"
    photo.write_bytes(b"unchanged")
    xmp.write_ricoh_preset(photo, "gr3_positive_film")
    before = xmp._curve_snapshot(description(photo))
    adobe_dehaze = description(photo).get("{" + xmp._CRS_NS + "}Dehaze")
    params = DehazeParams(strength=.8).__dict__
    xmp.write_dehaze_settings(photo, params, compatibility_curves=CURVES)
    first = xmp._curve_snapshot(description(photo))
    assert first != before
    assert description(photo).get("{" + xmp._CRS_NS + "}Dehaze") == adobe_dehaze
    assert len(xmp._read_tone_curve(description(photo), "ToneCurvePV2012Red")) == 4
    xmp.write_dehaze_settings(photo, params, compatibility_curves=CURVES)
    assert xmp._curve_snapshot(description(photo)) == first
    xmp.write_dehaze_settings(photo, DehazeParams(strength=0).__dict__, compatibility_curves=None)
    assert xmp._curve_snapshot(description(photo)) == before
    assert photo.read_bytes() == b"unchanged"


def test_external_curve_edit_is_kept_when_disabling_fit(tmp_path):
    photo = tmp_path / "sample.jpg"
    photo.touch()
    xmp.write_dehaze_settings(photo, DehazeParams().__dict__, compatibility_curves=CURVES)
    root = xmp._parse_xmp(photo.with_suffix(".xmp").read_bytes())
    desc = xmp._description(root)
    red = desc.find("{" + xmp._CRS_NS + "}ToneCurvePV2012Red")
    red.find("{" + xmp._RDF_NS + "}Seq")[1].text = "64, 51"
    desc.set("{https://example.test/vendor}Keep", "yes")
    photo.with_suffix(".xmp").write_bytes(xmp._serialize_xmp(root))
    xmp.write_dehaze_settings(photo, DehazeParams(strength=0).__dict__)
    assert xmp._read_tone_curve(description(photo), "ToneCurvePV2012Red")[1] == (64, 51)
    assert description(photo).get("{https://example.test/vendor}Keep") == "yes"


def test_preset_switch_recomposes_without_accumulation(tmp_path):
    photo = tmp_path / "sample.jpg"
    photo.touch()
    params = DehazeParams().__dict__
    basic = app_api.BasicParamsRequest().values()
    xmp.write_photo_settings(photo, params, basic, "gr3_positive_film", compatibility_curves=CURVES)
    xmp.write_photo_settings(photo, params, basic, "gr3_negative_film", compatibility_curves=CURVES)
    switched = xmp._curve_snapshot(description(photo))
    other = tmp_path / "clean.jpg"
    other.touch()
    xmp.write_photo_settings(other, params, basic, "gr3_negative_film", compatibility_curves=CURVES)
    assert xmp._curve_snapshot(description(other)) == switched
    xmp.write_photo_settings(photo, DehazeParams(strength=0).__dict__, basic, None)
    assert description(photo).find("{" + xmp._CRS_NS + "}ToneCurvePV2012Red") is None


def test_api_writes_fitted_curves_and_decode_failure_keeps_old_packet(tmp_path):
    photo = tmp_path / "valid.png"
    rgb = np.full((48, 64, 3), (160, 180, 200), dtype=np.uint8)
    rgb[24:] = (80, 110, 130)
    Image.fromarray(rgb).save(photo)
    original = photo.read_bytes()
    session_id = "xmp-fit-test"
    app_api._ENHANCE_SESSIONS[session_id] = {"files": {"p": photo}}
    req = app_api.PhotoSettingsRequest(session_id=session_id, photo_id="p",
        dehaze_params=app_api.EnhanceParamsRequest(strength=1),
        basic_params=app_api.BasicParamsRequest(), auto_mode=True)
    try:
        result = app_api.save_photo_settings_snapshot(req)
        assert result["dehaze_compatibility"] == "approximate"
        desc = description(photo)
        assert desc.get("{" + xmp._CRS_NS + "}ProcessVersion") is not None
        assert desc.get("{" + xmp._CRS_NS + "}Version") is not None
        assert desc.get("{" + xmp._CRS_NS + "}HasSettings") == "True"
        assert len(xmp._read_tone_curve(desc, "ToneCurvePV2012Blue")) > 2
        assert photo.read_bytes() == original
        saved = photo.with_suffix(".xmp").read_bytes()
        photo.write_bytes(b"not an image")
        response = app_api.save_photo_settings_snapshot(req)
        assert response.status_code == 500
        assert photo.with_suffix(".xmp").read_bytes() == saved
        req.dehaze_params = app_api.EnhanceParamsRequest(strength=0)
        reset = app_api.save_photo_settings_snapshot(req)
        assert reset["dehaze_compatibility"] == "disabled"
        assert description(photo).find("{" + xmp._CRS_NS + "}ToneCurvePV2012Red") is None
    finally:
        app_api._ENHANCE_SESSIONS.pop(session_id, None)
