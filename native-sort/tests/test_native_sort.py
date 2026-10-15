from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import native_sort  # noqa: E402
from burst_filter import RawEvaluator  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def native_library_available():
    if not native_sort.get_native_sort_status()["available"]:
        pytest.skip("Build native-sort before running native parity checks")


@pytest.mark.parametrize("shape,roi", [
    ((2, 2), (0, 0, 2, 2)),
    ((19, 31), (0, 0, 31, 19)),
    ((64, 80), (7, 11, 47, 39)),
])
def test_region_sharpness_matches_opencv(shape, roi):
    height, width = shape
    rng = np.random.default_rng(height * 100 + width)
    gray = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    original = gray.copy()

    native = native_sort.native_region_sharpness(gray, *roi)
    python = RawEvaluator._region_sharpness(gray, *roi)

    assert native == pytest.approx(python, rel=1e-12, abs=1e-10)
    np.testing.assert_array_equal(gray, original)


def test_exposure_score_matches_gray_threshold_formula():
    gray = np.array([[0, 5, 6, 249], [250, 255, 100, 200]], dtype=np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    assert native_sort.native_exposure_score(gray) == RawEvaluator.exposure_score(rgb)


def test_batch_regions_match_python_and_preserve_input():
    rng = np.random.default_rng(709)
    gray = rng.integers(0, 256, size=(33, 41), dtype=np.uint8)
    original = gray.copy()
    regions = [(0, 0, 13, 11), (13, 11, 15, 12), (30, 25, 11, 8)]
    native = native_sort.native_region_sharpness_many(gray, regions)
    python = [RawEvaluator._region_sharpness(gray, *roi) for roi in regions]
    np.testing.assert_allclose(native, python, rtol=1e-12, atol=1e-10)
    np.testing.assert_array_equal(gray, original)


def test_native_profile_failure_falls_back_to_python():
    gray = np.random.default_rng(87).integers(0, 256, size=(45, 63, 3), dtype=np.uint8)
    native_evaluator = RawEvaluator(sort_backend="native")
    python_evaluator = RawEvaluator()
    with patch("burst_filter.native_region_sharpness_many", side_effect=RuntimeError("failed")):
        native_profile = native_evaluator.sharpness_profile(gray)
    python_profile = python_evaluator.sharpness_profile(gray)
    assert native_profile[0] == python_profile[0]
    np.testing.assert_array_equal(native_profile[1], python_profile[1])


def test_native_input_validation_is_strict():
    with pytest.raises(ValueError):
        native_sort.native_exposure_score(np.zeros((3, 4, 1), dtype=np.uint8))
    with pytest.raises(TypeError):
        native_sort.native_exposure_score(np.zeros((3, 4), dtype=np.uint16))
    with pytest.raises(ValueError):
        native_sort.native_exposure_score(np.zeros((3, 8), dtype=np.uint8)[:, ::2])
