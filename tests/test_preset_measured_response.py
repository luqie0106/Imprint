"""Measured ACR response integration in the Ricoh preview pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import measured_response
import ricoh_filter


def test_vivid_street_xmp_matches_acr_tone_controls_and_curve():
    controls = ricoh_filter._preset_controls("gr3_vivid_street")

    assert {key: controls[key] for key in (
        "contrast", "highlights", "shadows", "whites", "blacks",
    )} == {
        "contrast": 20.0, "highlights": -34.0, "shadows": 18.0,
        "whites": 10.0, "blacks": -15.0,
    }
    assert controls["tone_curve"] == (
        (0.0, 0.0), (24.0, 15.0), (72.0, 70.0), (128.0, 137.0),
        (192.0, 204.0), (236.0, 237.0), (255.0, 250.0),
    )


def _neutral_controls(**updates):
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


def test_xmp_preset_applies_measured_color_without_manual_sliders():
    source = np.array([[[200, 80, 80], [40, 120, 190], [150, 170, 80]]], dtype=np.uint8)

    legacy = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_positive_film")
    measured = ricoh_filter.apply_ricoh_preview_effect(
        source, "gr3_positive_film", use_measured_color=True,
    )

    assert measured.shape == source.shape
    assert measured.dtype == source.dtype
    assert np.any(measured != legacy)


def test_measured_hsl_single_slider_matches_the_direct_lut_control(monkeypatch):
    controls = _neutral_controls()
    controls["saturation_adjustments"]["Red"] = 60.0
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: controls)
    source = np.array([[[200, 80, 80], [190, 100, 70]]], dtype=np.uint8)

    preview = ricoh_filter.apply_ricoh_preview_effect(
        source, "gr3_standard", use_measured_color=True,
    )
    direct = measured_response.apply_hsl_single_slider(source, "saturation", "Red", 60.0)

    assert np.allclose(preview, direct, atol=1)


def test_multiple_measured_hsl_controls_return_bounded_float_rgb_without_mutation():
    source = np.array([[[200, 80, 80], [40, 120, 190], [150, 170, 80]]], dtype=np.float32) / 255.0
    original = source.copy()
    adjustments = {
        "hue": {"Red": 35.0, "Blue": -20.0},
        "saturation": {"Orange": -25.0, "Green": 40.0},
        "luminance": {"Red": 15.0},
    }

    result = measured_response.apply_hsl_controls_float(source, adjustments)

    assert result.dtype == np.float32
    assert result.shape == source.shape
    assert np.isfinite(result).all()
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert not np.array_equal(result, source)
    assert np.array_equal(source, original)


def test_measured_grade_wheel_changes_preview(monkeypatch):
    controls = _neutral_controls(
        grade_midtone_hue=205.0,
        grade_midtone_saturation=100.0,
    )
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: controls)
    source = np.array([[[128, 128, 128], [135, 125, 130]]], dtype=np.uint8)

    legacy = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")
    measured = ricoh_filter.apply_ricoh_preview_effect(
        source, "gr3_standard", use_measured_color=True,
    )

    assert np.any(measured != legacy)
    assert np.any(measured[0, 0] != measured[0, 0, 0])
    assert measured.shape == source.shape
    assert measured.dtype == source.dtype


def test_single_default_midtone_wheel_matches_direct_gray_ramp_curve():
    controls = _neutral_controls(grade_midtone_hue=0.0, grade_midtone_saturation=100.0)
    source = np.repeat(
        np.array([[[48], [96], [128], [176], [224]]], dtype=np.float32) / 255.0,
        3,
        axis=2,
    )

    preview = ricoh_filter._apply_measured_color_grading(source, controls, np)
    direct = measured_response.apply_grading_wheels_float(
        source, [("Midtones", 0.0, 100.0, None)],
    )

    assert np.allclose(preview, direct, atol=1e-6)


def test_measured_color_false_keeps_the_existing_preview_path(monkeypatch):
    controls = _neutral_controls(
        hue_adjustments={**{color: 0.0 for color in ricoh_filter._HSL_COLORS}, "Red": 20.0},
        grade_midtone_hue=190.0,
        grade_midtone_saturation=50.0,
    )
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: controls)
    source = np.array([[[200, 80, 80], [128, 128, 128]]], dtype=np.uint8)

    implicit = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")
    explicit = ricoh_filter.apply_ricoh_preview_effect(
        source, "gr3_standard", use_measured_color=False,
    )

    assert np.array_equal(implicit, explicit)


@pytest.mark.parametrize("dtype, maximum", [(np.uint8, 255), (np.uint16, 65535)])
def test_vivid_street_measured_preview_applies_monotone_neutral_luma_curve(
    monkeypatch, dtype, maximum,
):
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: _neutral_controls())
    levels = np.arange(maximum + 1, dtype=dtype)
    source = np.repeat(levels[None, :, None], 3, axis=2)
    original = source.copy()

    result = ricoh_filter.apply_ricoh_preview_effect(
        source, "gr3_vivid_street", use_measured_color=True,
    )

    assert result.dtype == dtype
    assert np.array_equal(source, original)
    assert np.array_equal(result[..., 0], result[..., 1])
    assert np.array_equal(result[..., 1], result[..., 2])
    assert np.all(np.diff(result[0, :, 0].astype(np.int64)) >= 0)
    assert result[0, 0, 0] == 0
    assert result[0, -1, 0] == maximum

    anchor_x = np.array([0, 8, 24, 40, 56, 72, 88, 104, 120, 144, 176, 208, 240, 255])
    anchor_y = np.array([0, 6, 16, 30, 49, 71, 93, 111, 129, 157, 192, 211, 234, 255])
    sample_indices = np.rint(anchor_x * maximum / 255.0).astype(np.int64)
    expected = np.rint(anchor_y * maximum / 255.0)
    assert np.allclose(result[0, sample_indices, 0], expected, atol=2)
    assert result[0, sample_indices[3], 0] < source[0, sample_indices[3], 0]
    assert result[0, sample_indices[6], 0] > source[0, sample_indices[6], 0]


def test_vivid_street_luma_calibration_preserves_color_direction(monkeypatch):
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: _neutral_controls())
    source = np.array([[[180, 90, 45]]], dtype=np.uint8)
    original = source.copy()

    result = ricoh_filter.apply_ricoh_preview_effect(
        source, "gr3_vivid_street", use_measured_color=True,
    )

    source_direction = source[0, 0].astype(np.float32) / float(source[0, 0].max())
    result_direction = result[0, 0].astype(np.float32) / float(result[0, 0].max())
    assert np.allclose(result_direction, source_direction, atol=1 / 255)
    assert np.array_equal(source, original)


def test_vivid_street_luma_calibration_is_scoped_to_measured_preview(monkeypatch):
    monkeypatch.setattr(ricoh_filter, "_preset_controls", lambda _: _neutral_controls())
    source = np.array([[[8, 8, 8], [40, 40, 40], [88, 88, 88], [176, 176, 176]]], dtype=np.uint8)

    measured_other_preset = ricoh_filter.apply_ricoh_preview_effect(
        source, "gr3_standard", use_measured_color=True,
    )
    vivid_legacy = ricoh_filter.apply_ricoh_preview_effect(
        source, "gr3_vivid_street", use_measured_color=False,
    )

    assert np.array_equal(measured_other_preset, source)
    assert np.array_equal(vivid_legacy, source)
