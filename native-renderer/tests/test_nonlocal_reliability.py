"""Focused contracts for conservative non-local transmission relief."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dehaze_nonlocal import (  # noqa: E402
    _wls_pcg,
    build_reliability_lut,
    lookup_transmission_relief,
)


def _ray_samples(height=64, width=256):
    atmosphere = np.array((0.55, 0.60, 0.65), dtype=np.float32)
    direction = np.full(3, 1.0 / np.sqrt(3.0), dtype=np.float32)
    radii = np.linspace(0.045, 0.43, width, dtype=np.float32)
    row = atmosphere[None, :] - radii[:, None] * direction[None, :]
    sample = np.broadcast_to(row[None, :, :], (height, width, 3)).copy()
    return sample, atmosphere


def test_same_rgb_has_same_lut_value_at_edges_and_open_positions():
    sample, atmosphere = _ray_samples()
    lut, stats = build_reliability_lut(sample, atmosphere, baseline_t=0.42)
    assert stats["solver_converged"]
    assert np.isfinite(stats["solver_residual"])

    color = sample[24, 216].copy()
    query = np.broadcast_to(color, (513, 9, 3)).copy()
    original = query.copy()
    relief = lookup_transmission_relief(query, lut)
    assert relief.dtype == np.float32
    assert relief.shape == (513, 9)
    assert np.isfinite(relief).all()
    assert np.all((relief >= 0.0) & (relief <= 1.0))
    np.testing.assert_array_equal(query, original)
    np.testing.assert_array_equal(relief, np.full_like(relief, relief[0, 0]))
    # Row-blocked evaluation must match the same color queried in one block.
    np.testing.assert_array_equal(
        relief,
        np.tile(lookup_transmission_relief(query[:1], lut), (513, 1)),
    )


def test_repeated_same_ray_near_samples_get_more_relief_than_far_samples():
    sample, atmosphere = _ray_samples(height=64, width=256)
    original = sample.copy()
    lut, stats = build_reliability_lut(sample, atmosphere, baseline_t=0.42)
    assert lut.dtype == np.float32
    assert lut.shape == (33, 33, 33)
    assert np.isfinite(lut).all()
    assert float(lut.min()) >= 0.0 and float(lut.max()) <= 1.0
    assert stats["candidate_count"] > 0
    assert stats["reliable_ray_count"] > 0
    assert stats["solver_converged"]
    assert stats["solver_iterations"] <= 64
    assert stats["solver_residual"] <= 1.5e-4
    assert stats["solver_seconds"] >= 0.0 and stats["lut_seconds"] >= 0.0
    assert stats["max_relief"] > 0.01
    relief = lookup_transmission_relief(sample[0:1], lut)[0]
    assert float(relief[-16:].mean()) > float(relief[:32].mean())
    np.testing.assert_array_equal(sample, original)


def test_insufficient_constant_near_air_and_black_samples_are_conservative():
    atmosphere = np.array((0.55, 0.60, 0.65), dtype=np.float32)
    cases = []
    too_few = np.broadcast_to(atmosphere - 0.2, (1, 3, 3)).copy()
    cases.append((too_few, 3))
    constant = np.broadcast_to(
        np.array((0.12, 0.17, 0.21), dtype=np.float32), (40, 50, 3),
    ).copy()
    cases.append((constant, constant.shape[0] * constant.shape[1]))
    direction = np.full(3, 1.0 / np.sqrt(3.0), dtype=np.float32)
    near_air = np.broadcast_to(
        atmosphere - 1e-4 * direction, (20, 30, 3),
    ).copy()
    cases.append((near_air, 0))
    cases.append((np.zeros((32, 40, 3), dtype=np.float32), 32 * 40))

    for sample, expected_candidates in cases:
        lut, stats = build_reliability_lut(sample, atmosphere, baseline_t=0.3)
        assert lut.dtype == np.float32 and lut.shape == (33, 33, 33)
        assert np.count_nonzero(lut) == 0
        assert stats["candidate_count"] == expected_candidates
        assert stats["reliable_ray_count"] == 0


def test_trilinear_lookup_is_continuous_and_rejects_nonfinite_samples_safely():
    axis = np.linspace(0.0, 1.0, 33, dtype=np.float32)
    red, green, blue = np.meshgrid(axis, axis, axis, indexing="ij")
    lut = (0.2 * red + 0.3 * green + 0.5 * blue).astype(np.float32)
    ramp = np.linspace(0.2, 0.8, 513, dtype=np.float32)
    query = np.stack((ramp, 0.5 * ramp + 0.1, 0.8 - 0.4 * ramp), axis=1)[None]
    result = lookup_transmission_relief(query, lut)[0]
    expected = 0.2 * query[0, :, 0] + 0.3 * query[0, :, 1] + 0.5 * query[0, :, 2]
    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=0.0)
    assert float(np.max(np.abs(np.diff(result)))) < 0.001

    query_with_invalid = np.array(
        [[[0.2, 0.3, 0.4], [np.nan, 0.2, 0.3], [0.4, np.inf, 0.2]]],
        dtype=np.float32,
    )
    invalid_result = lookup_transmission_relief(query_with_invalid, lut)
    assert np.isfinite(invalid_result).all()
    assert invalid_result[0, 1] == 0.0 and invalid_result[0, 2] == 0.0
    assert invalid_result[0, 0] > 0.0

    with pytest.raises(ValueError, match="finite"):
        build_reliability_lut(query_with_invalid, np.array((0.6, 0.7, 0.8), np.float32), 0.5)


def test_empty_and_large_inputs_keep_bounded_analysis_shapes():
    atmosphere = np.array((0.55, 0.60, 0.65), dtype=np.float32)
    empty = np.empty((0, 0, 3), dtype=np.float32)
    lut, stats = build_reliability_lut(empty, atmosphere, baseline_t=0.5)
    assert lut.shape == (33, 33, 33) and lut.dtype == np.float32
    assert stats["candidate_count"] == 0
    assert lookup_transmission_relief(empty, lut).shape == (0, 0)

    sample, atmosphere = _ray_samples(height=333, width=347)
    _lut, stats = build_reliability_lut(sample, atmosphere, baseline_t=0.42)
    assert max(stats["analysis_shape"]) <= 320
    assert stats["solver_converged"]


def test_wls_pcg_stays_bounded_at_a_real_color_edge():
    image = np.empty((48, 96, 3), dtype=np.float32)
    image[:, :48] = (0.15, 0.21, 0.28)
    image[:, 48:] = (0.70, 0.73, 0.78)
    proposal = np.zeros((48, 96), dtype=np.float32)
    confidence = np.zeros_like(proposal)
    proposal[8:40, 8:40] = 0.85
    confidence[8:40, 8:40] = 1.0
    solved, stats = _wls_pcg(image, proposal, confidence)
    assert stats["solver_converged"]
    assert stats["solver_iterations"] <= 64
    assert np.isfinite(stats["solver_residual"])
    assert np.isfinite(solved).all()
    assert float(solved.min()) >= 0.0
    assert float(solved.max()) <= 0.85 + 1e-6
    assert float(solved[:, 55:].max()) < 1e-5
