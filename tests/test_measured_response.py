"""Focused checks for the ACR single-slider preview interpolation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import measured_response


def test_zero_slider_does_not_load_model(monkeypatch):
    monkeypatch.setattr(measured_response, "_load_lut", lambda: (_ for _ in ()).throw(AssertionError()))
    image = np.array([[[90, 45, 20]]], dtype=np.uint8)
    result = measured_response.apply_hsl_single_slider(image, "hue", "Red", 0)
    assert np.array_equal(result, image)
    assert result is not image


def test_periodic_hue_and_level_interpolation(monkeypatch):
    levels = np.array([-100, 0, 100], dtype=np.float32)
    sats = np.array([0.1, 0.95], dtype=np.float32)
    lights = np.array([0.05, 0.95], dtype=np.float32)
    lut = np.zeros((3, 8, 3, 2, 2, 360, 3), dtype=np.float32)
    lut[0, 0, 2, :, :, 359, 0] = 0.2
    lut[0, 0, 2, :, :, 0, 0] = 0.2
    monkeypatch.setattr(measured_response, "_load_lut", lambda: (lut, levels, sats, lights))
    image = np.array([[[50000, 25000, 25001], [50000, 25001, 25000]]], dtype=np.uint16)
    result = measured_response.apply_hsl_single_slider(image, "hue", "Red", 50)
    # Both sides of the 0/360 boundary receive the same half-strength red shift.
    assert abs(int(result[0, 0, 0]) - int(result[0, 1, 0])) <= 20
    assert result[0, 0, 0] > image[0, 0, 0]


def test_out_of_domain_neutral_is_unchanged(monkeypatch):
    levels = np.array([-100, 0, 100], dtype=np.float32)
    sats = np.array([0.1, 0.95], dtype=np.float32)
    lights = np.array([0.05, 0.95], dtype=np.float32)
    lut = np.full((3, 8, 3, 2, 2, 360, 3), 0.1, dtype=np.float32)
    monkeypatch.setattr(measured_response, "_load_lut", lambda: (lut, levels, sats, lights))
    image = np.array([[[128, 128, 128]]], dtype=np.uint8)
    result = measured_response.apply_hsl_single_slider(image, "saturation", "Blue", 100)
    assert np.array_equal(result, image)


def test_grading_gray_baseline_and_measured_wheel():
    image = np.array([[[128, 128, 128], [255, 0, 0]]], dtype=np.uint8)
    baseline = measured_response.apply_grading_gray_preview(image, "Shadows", 50, baseline_only=True)
    graded = measured_response.apply_grading_gray_preview(image, "Shadows", 50)
    assert baseline.shape == image.shape
    assert np.max(baseline[0, 0]) - np.min(baseline[0, 0]) <= 2
    assert np.any(graded != baseline)


def test_grading_balance_interpolates_without_nonfinite_output():
    image = np.full((1, 3, 3), [32, 128, 224], dtype=np.uint8)
    result = measured_response.apply_grading_gray_preview(image, "Balance", 25)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
