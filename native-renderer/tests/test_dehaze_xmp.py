"""Numerical checks for display-space dehaze curve fitting."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dehaze_xmp import (
    _fit_one_channel_samples,
    _simplify_curve,
    apply_display_curves,
    fit_dehaze_curves,
)


def _identity_curves():
    return fit_dehaze_curves(
        np.zeros((1, 1, 3), dtype=np.float32),
        np.zeros((1, 1, 3), dtype=np.float32),
    )


def test_fit_recovers_monotonic_gamma_and_channel_color_changes():
    rng = np.random.default_rng(20261003)
    source = rng.random((96, 128, 3), dtype=np.float32)
    target = np.empty_like(source)
    target[..., 0] = source[..., 0] ** 0.72
    target[..., 1] = np.clip(source[..., 1] * 0.84 + 0.07, 0.0, 1.0)
    target[..., 2] = np.clip(source[..., 2] ** 0.62 * 0.82 + 0.04, 0.0, 1.0)

    curves = fit_dehaze_curves(source, target)
    simulated = apply_display_curves(source, curves)
    identity_error = float(np.mean((source - target) ** 2))
    fitted_error = float(np.mean((simulated - target) ** 2))

    assert fitted_error < identity_error * 0.35
    assert len(curves) == 3
    assert len({curve for curve in curves}) > 1
    for channel, curve in enumerate(curves):
        assert 2 <= len(curve) <= 16
        assert curve[0] == (0, 0)
        assert curve[-1] == (255, 255)
        assert all(type(x) is int and type(y) is int for x, y in curve)
        assert all(0 <= x <= 255 and 0 <= y <= 255 for x, y in curve)
        assert all(a[0] < b[0] and a[1] <= b[1] for a, b in zip(curve, curve[1:]))
        dense_x, dense_y = _fit_one_channel_samples(source[..., channel], target[..., channel])
        compact_x = np.fromiter((point[0] for point in curve), dtype=np.float64)
        compact_y = np.fromiter((point[1] for point in curve), dtype=np.float64)
        max_error = float(np.max(np.abs(dense_y - np.interp(dense_x, compact_x, compact_y))))
        assert max_error <= 1.0


def test_identity_pair_returns_exact_identity_curves():
    rng = np.random.default_rng(4)
    image = rng.random((12, 17, 3), dtype=np.float32)

    curves = fit_dehaze_curves(image, image.copy())

    assert curves == (_identity_curves()[0],) * 3
    assert all(len(curve) == 2 for curve in curves)
    assert apply_display_curves(image, curves).dtype == np.float32


def test_constant_pair_uses_identity_when_the_histogram_cannot_constrain_a_curve():
    source = np.full((20, 30, 3), 0.4, dtype=np.float32)
    target = np.full((20, 30, 3), 0.8, dtype=np.float32)

    curves = fit_dehaze_curves(source, target)

    assert curves == (_identity_curves()[0],) * 3


def test_adaptive_curve_simplification_preserves_smooth_curve_with_subpixel_error():
    x_values = np.concatenate((np.arange(0, 256, 4), np.array([255]))).astype(np.float64)
    y_values = np.rint(x_values + 0.001 * x_values * (255.0 - x_values)).astype(np.int16)

    compact = _simplify_curve(x_values, y_values)
    compact_x = np.fromiter((point[0] for point in compact), dtype=np.float64)
    compact_y = np.fromiter((point[1] for point in compact), dtype=np.float64)
    errors = np.abs(y_values - np.interp(x_values, compact_x, compact_y))

    assert len(compact) < 16
    assert compact[0] == (0, 0)
    assert compact[-1] == (255, 255)
    assert all(a[0] < b[0] and a[1] <= b[1] for a, b in zip(compact, compact[1:]))
    assert float(np.max(errors)) <= 0.76


@pytest.mark.parametrize(
    ("source", "target", "error"),
    [
        (np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.float32), ValueError),
        (np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32), ValueError),
        (np.zeros((2, 3, 4), dtype=np.float32), np.zeros((2, 3, 4), dtype=np.float32), ValueError),
        (np.zeros((2, 3, 3), dtype=np.float64), np.zeros((2, 3, 3), dtype=np.float64), TypeError),
        (np.zeros((2, 3, 3), dtype=np.float32), np.zeros((3, 2, 3), dtype=np.float32), ValueError),
    ],
)
def test_rejects_empty_bad_shape_wrong_dtype_and_shape_mismatch(source, target, error):
    with pytest.raises(error):
        fit_dehaze_curves(source, target)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, -0.01, 1.01])
@pytest.mark.parametrize("invalid_input", ["source", "target"])
def test_rejects_non_finite_and_out_of_range_inputs(bad_value, invalid_input):
    source = np.full((2, 3, 3), 0.5, dtype=np.float32)
    target = source.copy()
    destination = source if invalid_input == "source" else target
    destination[0, 0, 0] = bad_value

    with pytest.raises(ValueError):
        fit_dehaze_curves(source, target)


def test_read_only_inputs_are_not_modified():
    rng = np.random.default_rng(19)
    source = rng.random((32, 48, 3), dtype=np.float32)
    target = np.clip(source * 0.9 + 0.025, 0.0, 1.0).astype(np.float32)
    source_before = source.copy()
    target_before = target.copy()
    source.setflags(write=False)
    target.setflags(write=False)

    curves = fit_dehaze_curves(source, target)
    simulated = apply_display_curves(source, curves)

    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(target, target_before)
    assert simulated.shape == source.shape
    assert np.isfinite(simulated).all()
