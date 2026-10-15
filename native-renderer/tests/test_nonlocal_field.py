"""Synthetic scene and edge contracts for the strong transmission field."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dehaze_nonlocal import (  # noqa: E402
    _weighted_guided_coefficients,
    build_nonlocal_transmission,
)


def _ray_column_scene(height=160, width=256):
    atmosphere = np.array((0.55, 0.60, 0.65), dtype=np.float32)
    direction = np.full(3, 1.0 / np.sqrt(3.0), dtype=np.float32)
    source = np.broadcast_to(atmosphere, (height, width, 3)).copy()
    radii = np.linspace(0.045, 0.40, height - 16, dtype=np.float32)
    column = atmosphere[None, :] - radii[:, None] * direction[None, :]
    source[8:-8, width // 2 - 6:width // 2 + 6] = column[:, None, :]
    return source, atmosphere


def test_actual_ray_transmission_recovers_near_far_order_and_bounds():
    source, atmosphere = _ray_column_scene()
    original = source.copy()
    field, stats = build_nonlocal_transmission(source, atmosphere, 0.62, strength=1.0)
    assert field.dtype == np.float32 and field.shape == source.shape[:2]
    assert np.isfinite(field).all()
    assert float(field.min()) >= 0.0 and float(field.max()) <= 1.0
    assert stats["nonlocal_field_active"]
    assert stats["reliable_ray_count"] > 0
    assert stats["solver_converged"] and stats["solver_iterations"] <= 64
    assert np.isfinite(stats["solver_residual"])
    # Larger distance from atmospheric light corresponds to higher t.
    assert float(field[130, 128]) > float(field[25, 128])
    np.testing.assert_array_equal(source, original)

    # Repeated tower colors in one material share one color-LUT transmission.
    assert field[44, 122] == field[44, 130]


def test_same_rgb_sky_is_equal_across_one_sided_tower_boundary():
    source, atmosphere = _ray_column_scene()
    field, stats = build_nonlocal_transmission(source, atmosphere, 0.62)
    assert stats["nonlocal_field_active"]
    sky_color = source[56, 122].copy()
    source[56, 30] = sky_color
    source[56, 220] = sky_color
    # Query a second time with the same source colors present on both sides of
    # the tower. Their table lookup must be position-independent.
    field, _ = build_nonlocal_transmission(source, atmosphere, 0.62)
    assert field[56, 30] == field[56, 220]


def test_symmetric_vignetted_sky_stays_symmetric_after_field_estimation():
    height, width = 96, 192
    atmosphere = np.array((0.55, 0.60, 0.65), dtype=np.float32)
    direction = np.full(3, 1.0 / np.sqrt(3.0), dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    radii = 0.02 + 0.09 * np.abs(x)
    row = atmosphere[None, :] - radii[:, None] * direction[None, :]
    source = np.broadcast_to(row[None, :, :], (height, width, 3)).copy()
    field, stats = build_nonlocal_transmission(source, atmosphere, 0.62)
    assert stats["nonlocal_field_active"]
    np.testing.assert_array_equal(source, source[:, ::-1])
    np.testing.assert_array_equal(field, field[:, ::-1])
    assert float(field[:, :12].mean()) > float(field[:, width // 2 - 6:width // 2 + 6].mean())


def test_zero_strength_is_exact_identity_and_unreliable_scene_uses_baseline():
    source, atmosphere = _ray_column_scene(height=48, width=80)
    zero, zero_stats = build_nonlocal_transmission(source, atmosphere, 0.57, strength=0.0)
    np.testing.assert_array_equal(zero, np.ones(source.shape[:2], dtype=np.float32))
    assert not zero_stats["nonlocal_field_active"]

    constant = np.broadcast_to(
        np.array((0.14, 0.18, 0.22), dtype=np.float32), (40, 60, 3),
    ).copy()
    baseline, stats = build_nonlocal_transmission(constant, atmosphere, 0.57)
    np.testing.assert_array_equal(baseline, np.full((40, 60), 0.57, np.float32))
    assert not stats["nonlocal_field_active"]
    assert stats["fallback_reason"] == "no_reliable_rays"


def test_weighted_guided_filter_keeps_thin_tower_step_inside_input_range():
    height, width = 96, 176
    guide = np.full((height, width), 0.62, dtype=np.float32)
    payload = np.full((height, width), 0.35, dtype=np.float32)
    guide[:, 82:85] = 0.13
    payload[:, 82:85] = 1.65
    slope, intercept, stats = _weighted_guided_coefficients(guide, payload)
    filtered = slope * guide + intercept
    assert np.isfinite(filtered).all()
    assert float(filtered.min()) >= 0.35 - 1e-5
    assert float(filtered.max()) <= 1.65 + 1e-5
    # The sky stays flat well away from the thin tower. Nearby falloff is
    # smooth and has no overshoot beyond the measured depth range.
    np.testing.assert_allclose(filtered[:, :70], 0.35, atol=1e-5, rtol=0.0)
    assert float(np.ptp(filtered[20:76, 70:97])) <= 1.30
    assert stats["wgi_gamma_mean"] > 0.0
    assert stats["wgi_regularizer"] > 0.0


def test_weighted_guided_filter_contrast_step_is_monotone_without_sky_bands():
    height, width = 80, 192
    guide = np.full((height, width), 0.18, dtype=np.float32)
    guide[:, 96:] = 0.72
    payload = np.full((height, width), 0.22, dtype=np.float32)
    payload[:, 96:] = 1.25
    slope, intercept, _ = _weighted_guided_coefficients(guide, payload)
    filtered = slope * guide + intercept
    profile = filtered[height // 2]
    assert np.isfinite(filtered).all()
    assert float(np.min(np.diff(profile))) >= -1e-5
    assert float(profile.min()) >= 0.22 - 1e-5
    assert float(profile.max()) <= 1.25 + 1e-5
    assert float(np.ptp(profile[:72])) < 1e-4
    assert float(np.ptp(profile[120:])) < 1e-4


def test_full_resolution_reconstruction_has_no_256_row_seam():
    source, atmosphere = _ray_column_scene(height=520, width=256)
    field, stats = build_nonlocal_transmission(source, atmosphere, 0.62)
    assert stats["nonlocal_field_active"]
    assert np.isfinite(field).all()
    seam_diffs = np.abs(np.diff(field[:, 128]))
    assert float(np.max(seam_diffs[252:258])) < 0.04
    repeat, _ = build_nonlocal_transmission(source, atmosphere, 0.62)
    np.testing.assert_array_equal(field, repeat)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_source_is_rejected_without_mutation(invalid):
    source, atmosphere = _ray_column_scene(height=24, width=48)
    source[4, 5, 1] = invalid
    before = source.copy()
    with pytest.raises(ValueError, match="finite"):
        build_nonlocal_transmission(source, atmosphere, 0.6)
    np.testing.assert_array_equal(source, before)
