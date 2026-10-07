"""Absolute haze scale, WGIF regression window, and actual render diagnostics."""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from dehaze import DehazeParams
from dehaze_nonlocal import _ray_proposals, _weighted_guided_coefficients
from dehaze_physical import apply_physical_dehaze


def test_fully_hazy_line_does_not_mistake_nearest_endpoint_for_clear_air():
    air = np.array([.60, .65, .70], np.float32)
    radiance = np.array([0, .02, .03], np.float32)
    true_t = np.broadcast_to(np.linspace(.15, .60, 96, dtype=np.float32), (64, 96))
    source = air + true_t[..., None] * (radiance - air)
    proposal, confidence, _, rays = _ray_proposals(
        source, air, .45, .004, proposal_kind='transmission', strength=1)
    assert rays > 0 and np.count_nonzero(confidence) > source.shape[0] * 80
    # No clear pixel exists: relative r/r95 alone would report ~1 at the
    # nearest endpoint. Absolute calibration must retain that endpoint's haze.
    assert float(np.median(proposal[:, -8:])) < .67
    assert float(np.mean(np.abs(proposal[:, 8:-8] - true_t[:, 8:-8]))) < .055
    bound = np.maximum(1 - np.min(source / air, axis=2), 0)
    assert np.all(proposal[confidence > 0] >= bound[confidence > 0] - 1e-6)


def test_wgif_uses_same_regression_and_coefficient_window():
    rng = np.random.default_rng(82)
    guide = rng.uniform(.1, .8, (48, 72)).astype(np.float32)
    payload = (guide * .7 + rng.uniform(0, .3, guide.shape)).astype(np.float32)
    radius, regularizer = 4, .002
    box = lambda x, r: cv2.boxFilter(x, -1, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT_101)
    small_variance = np.maximum(box(guide**2, 1) - box(guide, 1)**2, 0)
    weight_variance = small_variance + max((.001 * float(np.ptp(guide)))**2, 1e-12)
    gamma = cv2.GaussianBlur(weight_variance * np.mean(1/weight_variance), (0, 0), 2,
                             borderType=cv2.BORDER_REFLECT_101)
    mean_g, mean_x = box(guide, radius), box(payload, radius)
    variance = np.maximum(box(guide**2, radius) - mean_g**2, 0)
    covariance = box(guide*payload, radius) - mean_g*mean_x
    expected_a = covariance / (variance + regularizer/np.maximum(gamma, 1e-6))
    expected_b = mean_x - expected_a*mean_g
    a, b, _ = _weighted_guided_coefficients(guide, payload, regularizer=regularizer,
                                           coefficient_radius=radius)
    np.testing.assert_allclose(a, box(expected_a, radius), atol=1e-6)
    np.testing.assert_allclose(b, box(expected_b, radius), atol=1e-6)


def test_actual_render_diagnostics_are_local_and_zero_clears_previous_status():
    source = np.full((40, 60, 3), .025, np.float32)
    status = {'nonlocal_active': True, 'stale': True}
    off = apply_physical_dehaze(source, DehazeParams(strength=1), spatial=True,
                                nonlocal_mode='off')
    fallback = apply_physical_dehaze(source, DehazeParams(strength=1), spatial=True,
                                     nonlocal_mode='strong', diagnostics=status)
    np.testing.assert_array_equal(off, fallback)
    assert not status['nonlocal_active'] and status['fallback_reason'] == 'low_airlight'
    assert 'stale' not in status
    result = apply_physical_dehaze(source, DehazeParams(strength=0), spatial=True,
                                   nonlocal_mode='strong', diagnostics=status)
    np.testing.assert_array_equal(result, source)
    assert status == dict(nonlocal_mode='strong', nonlocal_active=False, fallback_reason='zero_strength')
