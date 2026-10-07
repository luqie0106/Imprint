"""Experimental non-local transmission estimates for the physical dehazer.

This module builds a small RGB lookup table from color rays around the
estimated atmospheric light. The conservative API only relieves the existing
inverse. The strong API estimates transmission, refines optical depth with
weighted guidance, then projects it into a scene-shared continuous RGB field.

The reliability and edge-aware WLS ideas are mathematically inspired by
Berman et al., ``Non-Local Image Dehazing`` (CVPR 2016), and Li et al.,
``Weighted Guided Image Filtering`` (TIP 2015). This is an independent
implementation using only NumPy and OpenCV; it does not copy the authors'
reference implementation or its color/contrast transforms, 8-bit output, or
top-row sky boundary assumptions.
"""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np


_LUT_EDGE = 33
_AZIMUTH_BINS = 24
_POLAR_BINS = 12
_MAX_ANALYSIS_EDGE = 320
_PCG_MAX_ITERATIONS = 64
_PCG_TOLERANCE = 1e-4
_WLS_LAMBDA = 0.18
_WLS_ZERO_ANCHOR = 0.05
_EDGE_SIGMA = 0.06
_GUIDE_LUMA = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
_WGIF_REGULARIZER = 2e-3
_WGIF_COEFFICIENT_RADIUS = 6
_FIELD_LUT_EDGE = 65
_FIELD_LUT_GAUSSIAN_SIGMA = 0.85
_FIELD_LUT_PRIOR_MASS = 0.25


def _empty_stats(shape: tuple[int, int] = (0, 0)) -> dict[str, float | int | bool | tuple[int, int]]:
    return {
        "analysis_shape": shape,
        "candidate_count": 0,
        "reliable_ray_count": 0,
        "solver_iterations": 0,
        "solver_residual": 0.0,
        "solver_converged": True,
        "solver_seconds": 0.0,
        "lut_seconds": 0.0,
        "total_seconds": 0.0,
        "airlight_norm": 0.0,
        "near_air_threshold": 0.0,
        "max_relief": 0.0,
        "supported_lut_cells": 0,
    }


def _smoothstep01(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _zero_lut() -> np.ndarray:
    return np.zeros((_LUT_EDGE, _LUT_EDGE, _LUT_EDGE), dtype=np.float32)


def _analysis_image(sample: np.ndarray) -> np.ndarray:
    height, width = sample.shape[:2]
    target_height, target_width = _analysis_shape(height, width)
    if (target_height, target_width) == (height, width):
        return np.array(sample, dtype=np.float32, copy=True)
    return cv2.resize(sample, (target_width, target_height), interpolation=cv2.INTER_AREA).astype(
        np.float32, copy=False,
    )


def _analysis_shape(height: int, width: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        return max(0, height), max(0, width)
    scale = min(1.0, _MAX_ANALYSIS_EDGE / max(height, width))
    if scale >= 1.0:
        return height, width
    return (
        max(1, int(round(height * scale))),
        max(1, int(round(width * scale))),
    )


def _ray_proposals(
    sample: np.ndarray,
    atmosphere: np.ndarray,
    baseline_t: float,
    near_air_threshold: float,
    *,
    proposal_kind: str = "relief",
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return per-pixel proposals/confidence using shared ray statistics.

    ``relief`` is the legacy LUT proposal and keeps its original formula.
    ``transmission`` calibrates radius/r95 with absolute endpoint evidence.
    """
    if proposal_kind not in ("relief", "transmission"):
        raise ValueError("proposal_kind must be 'relief' or 'transmission'")
    height, width = sample.shape[:2]
    proposal = np.zeros((height, width), dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)
    # A radius ratio has no absolute scale when every endpoint is still hazy.
    # Calibrate the strong fit with the dark-channel prior used in the WGIF
    # dehazing application (omega=31/32). Conservative relief is unchanged.
    absolute_prior = None
    normalized_min = np.min(sample / np.maximum(atmosphere, 1e-6), axis=2)
    if proposal_kind == "transmission":
        dark = cv2.erode(normalized_min, np.ones((15, 15), dtype=np.uint8))
        absolute_prior = np.clip(1.0 - (31.0 / 32.0) * dark, .1, 1.0)
    finite = np.isfinite(sample).all(axis=2)
    # Values at or above one are likely clipped in normalized linear input.
    unclipped = (sample < (1.0 - 1e-6)).all(axis=2)
    below_air = (sample < atmosphere[None, None, :]).all(axis=2)
    delta = atmosphere[None, None, :] - np.where(finite[:, :, None], sample, 0.0)
    radius = np.linalg.norm(delta, axis=2)
    candidate = finite & unclipped & below_air & (radius > near_air_threshold)
    candidate_count = int(np.count_nonzero(candidate))
    if candidate_count == 0:
        return proposal, confidence, candidate_count, 0

    unit = np.zeros_like(delta, dtype=np.float32)
    unit[candidate] = delta[candidate] / radius[candidate, None]
    azimuth = np.mod(np.arctan2(unit[:, :, 1], unit[:, :, 0]), 2.0 * np.pi)
    polar = np.arccos(np.clip(unit[:, :, 2], -1.0, 1.0))
    az_bin = np.floor(azimuth * (_AZIMUTH_BINS / (2.0 * np.pi))).astype(np.int16)
    polar_bin = np.floor(polar * (_POLAR_BINS / (0.5 * np.pi))).astype(np.int16)
    az_bin = np.clip(az_bin, 0, _AZIMUTH_BINS - 1)
    polar_bin = np.clip(polar_bin, 0, _POLAR_BINS - 1)
    ray_id = polar_bin * _AZIMUTH_BINS + az_bin

    flat_candidate = candidate.ravel()
    flat_ray = ray_id.ravel()[flat_candidate]
    flat_radius = radius.ravel()[flat_candidate]
    flat_proposal = proposal.ravel()
    flat_confidence = confidence.ravel()
    reliable_ray_count = 0

    # Stable sort groups all candidates from a direction bin without keeping
    # a second full-resolution set of masks or Python objects.
    order = np.argsort(flat_ray, kind="stable")
    sorted_ray = flat_ray[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_ray)) + 1, len(sorted_ray)]
    candidate_flat_indices = np.flatnonzero(flat_candidate)[order]
    flat_indices = candidate_flat_indices
    base = float(np.clip(baseline_t, 0.0, 1.0))
    denominator = max(1.0 - base, 1e-6)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        count = int(end - start)
        # A ray needs repeated samples and a meaningful near/far radius span.
        # Both gates rise continuously to avoid hard confidence cliffs.
        if count < 4:
            continue
        radii = flat_radius[order[start:end]]
        r10, r95 = np.percentile(radii, (10.0, 95.0))
        if not np.isfinite(r95) or r95 <= near_air_threshold:
            continue
        span = float((r95 - r10) / max(float(r95), near_air_threshold))
        count_weight = float(_smoothstep01((count - 3.0) / 13.0))
        span_weight = float(_smoothstep01((span - 0.06) / 0.28))
        ray_confidence = count_weight * span_weight
        if ray_confidence <= 1e-5:
            continue
        reliable_ray_count += 1
        indices = flat_indices[start:end]
        t = np.clip(flat_radius[order[start:end]] / float(r95), 0.0, 1.0)
        if absolute_prior is not None:
            # Use repeated endpoint evidence rather than treating the largest
            # observed radius as proof of a clear foreground. Repeated dark
            # endpoints retain t≈1; fully hazy lines keep their absolute haze.
            endpoints = radii >= np.percentile(radii, 90.0)
            anchor = float(np.median(absolute_prior.ravel()[indices[endpoints]]))
            t *= anchor
            # Nonnegative radiance supplies the paper's physical lower bound.
            t = np.maximum(t, np.clip(1.0 - normalized_min.ravel()[indices], 0, 1))
        if proposal_kind == "relief":
            # Keep this branch byte-for-byte equivalent in formula to the
            # original conservative LUT path.
            relief = _smoothstep01((t - base) / denominator).astype(np.float32)
            flat_proposal[indices] = relief
        else:
            flat_proposal[indices] = np.asarray(
                1.0 - float(np.clip(strength, 0.0, 1.0)) * (1.0 - t),
                dtype=np.float32,
            )
        flat_confidence[indices] = np.float32(ray_confidence)

    return proposal, confidence, candidate_count, reliable_ray_count


def _edge_weights(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma2 = _EDGE_SIGMA * _EDGE_SIGMA
    horizontal_delta = image[:, 1:, :] - image[:, :-1, :]
    vertical_delta = image[1:, :, :] - image[:-1, :, :]
    horizontal = np.exp(-np.sum(horizontal_delta * horizontal_delta, axis=2) / sigma2)
    vertical = np.exp(-np.sum(vertical_delta * vertical_delta, axis=2) / sigma2)
    return horizontal.astype(np.float64), vertical.astype(np.float64)


def _wls_pcg(
    image: np.ndarray,
    proposal: np.ndarray,
    confidence: np.ndarray,
    prior: float = 0.0,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve confidence WLS with a scalar prior using matrix-free PCG."""
    data = np.asarray(confidence, dtype=np.float64)
    target = np.asarray(proposal, dtype=np.float64)
    horizontal, vertical = _edge_weights(image)
    lam = _WLS_LAMBDA
    anchor = _WLS_ZERO_ANCHOR
    diagonal = data + anchor
    diagonal[:, :-1] += lam * horizontal
    diagonal[:, 1:] += lam * horizontal
    diagonal[:-1, :] += lam * vertical
    diagonal[1:, :] += lam * vertical
    if prior == 0.0:
        rhs = data * target
    else:
        rhs = data * target + anchor * float(prior)
    rhs_norm = float(np.linalg.norm(rhs.ravel()))

    def apply(vector: np.ndarray) -> np.ndarray:
        result = (data + anchor) * vector
        horizontal_diff = vector[:, :-1] - vector[:, 1:]
        horizontal_flow = lam * horizontal * horizontal_diff
        result[:, :-1] += horizontal_flow
        result[:, 1:] -= horizontal_flow
        vertical_diff = vector[:-1, :] - vector[1:, :]
        vertical_flow = lam * vertical * vertical_diff
        result[:-1, :] += vertical_flow
        result[1:, :] -= vertical_flow
        return result

    x = np.zeros_like(rhs)
    residual = rhs.copy()
    if rhs_norm <= 1e-14:
        x.fill(float(prior))
        return x.astype(np.float32), {
            "solver_iterations": 0,
            "solver_residual": 0.0,
            "solver_converged": True,
        }
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    residual_preconditioned = float(np.sum(residual * preconditioned))
    relative_residual = 1.0
    iterations = 0
    converged = False
    for iteration in range(1, _PCG_MAX_ITERATIONS + 1):
        applied = apply(direction)
        denominator = float(np.sum(direction * applied))
        if not np.isfinite(denominator) or denominator <= 1e-24:
            break
        alpha = residual_preconditioned / denominator
        x += alpha * direction
        residual -= alpha * applied
        relative_residual = float(np.linalg.norm(residual.ravel()) / rhs_norm)
        iterations = iteration
        if not np.isfinite(relative_residual):
            break
        if relative_residual <= _PCG_TOLERANCE:
            converged = True
            break
        next_preconditioned = residual / diagonal
        next_residual_preconditioned = float(np.sum(residual * next_preconditioned))
        if not np.isfinite(next_residual_preconditioned) or residual_preconditioned <= 0.0:
            break
        beta = next_residual_preconditioned / residual_preconditioned
        direction = next_preconditioned + beta * direction
        preconditioned = next_preconditioned
        residual_preconditioned = next_residual_preconditioned

    # Report the true residual of the returned iterate. If the bounded solver
    # does not converge, fail closed rather than exporting a partial estimate.
    true_residual = float(np.linalg.norm((rhs - apply(x)).ravel()) / rhs_norm)
    converged = converged and np.isfinite(true_residual) and true_residual <= _PCG_TOLERANCE * 1.5
    if not converged or not np.isfinite(x).all():
        x.fill(float(prior))
    else:
        np.clip(x, 0.0, 1.0, out=x)
    return x.astype(np.float32), {
        "solver_iterations": iterations,
        "solver_residual": true_residual if np.isfinite(true_residual) else float("inf"),
        "solver_converged": bool(converged),
    }


def _trilinear_splat(
    image: np.ndarray,
    value: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Project supported WLS values into a 33-cubed RGB LUT."""
    total = np.zeros((_LUT_EDGE, _LUT_EDGE, _LUT_EDGE), dtype=np.float64)
    mass = np.zeros_like(total)
    valid = (
        np.isfinite(image).all(axis=2)
        & np.isfinite(value)
        & np.isfinite(confidence)
        & (confidence > 0.0)
    )
    if not np.any(valid):
        return _zero_lut(), 0
    colors = np.clip(image[valid], 0.0, 1.0) * (_LUT_EDGE - 1)
    lower = np.floor(colors).astype(np.int16)
    fraction = colors - lower
    values = np.clip(value[valid], 0.0, 1.0).astype(np.float64)
    confidences = confidence[valid].astype(np.float64)
    for red_bit in (0, 1):
        red_weight = fraction[:, 0] if red_bit else 1.0 - fraction[:, 0]
        red_index = np.minimum(lower[:, 0] + red_bit, _LUT_EDGE - 1)
        for green_bit in (0, 1):
            green_weight = fraction[:, 1] if green_bit else 1.0 - fraction[:, 1]
            green_index = np.minimum(lower[:, 1] + green_bit, _LUT_EDGE - 1)
            for blue_bit in (0, 1):
                blue_weight = fraction[:, 2] if blue_bit else 1.0 - fraction[:, 2]
                blue_index = np.minimum(lower[:, 2] + blue_bit, _LUT_EDGE - 1)
                weights = confidences * red_weight * green_weight * blue_weight
                np.add.at(total, (red_index, green_index, blue_index), weights * values)
                np.add.at(mass, (red_index, green_index, blue_index), weights)

    supported = mass > 1e-12
    # A small face-neighbor blend reduces quantization noise without using a
    # 2D blur on the 3D color cube or extending support into unseen colors.
    smooth_total = 0.70 * total
    smooth_mass = 0.70 * mass
    for axis in range(3):
        for offset in (-1, 1):
            source_slices = [slice(None)] * 3
            target_slices = [slice(None)] * 3
            if offset < 0:
                source_slices[axis] = slice(0, -1)
                target_slices[axis] = slice(1, None)
            else:
                source_slices[axis] = slice(1, None)
                target_slices[axis] = slice(0, -1)
            smooth_total[tuple(target_slices)] += 0.05 * total[tuple(source_slices)]
            smooth_mass[tuple(target_slices)] += 0.05 * mass[tuple(source_slices)]
    lut = _zero_lut().astype(np.float64)
    # A small zero-prior mass keeps isolated splats from becoming full-strength
    # LUT spikes. Truly unsupported cells still stay exactly zero.
    np.divide(
        smooth_total, smooth_mass + 0.01, out=lut,
        where=supported & (smooth_mass > 1e-12),
    )
    np.clip(lut, 0.0, 1.0, out=lut)
    return lut.astype(np.float32), int(np.count_nonzero(supported))


def build_reliability_lut(
    sample: np.ndarray,
    atmosphere: np.ndarray,
    baseline_t: float,
) -> tuple[np.ndarray, dict[str, float | int | bool | tuple[int, int]]]:
    """Build a conservative RGB LUT of reliability-weighted transmission relief.

    ``sample`` is linear RGB float data in HWC order. Analysis is capped at a
    320-pixel longest edge; the returned LUT is always float32 and 33 cubed.
    Invalid or unsupported color regions map to zero relief.
    """
    started = perf_counter()
    if not isinstance(sample, np.ndarray):
        sample = np.asarray(sample)
    if sample.ndim != 3 or sample.shape[2] != 3:
        raise ValueError("sample must have shape HxWx3")
    if not isinstance(atmosphere, np.ndarray):
        atmosphere = np.asarray(atmosphere)
    if atmosphere.shape != (3,):
        raise ValueError("atmosphere must have shape (3,)")

    source = np.asarray(sample, dtype=np.float32)
    air = np.asarray(atmosphere, dtype=np.float32)
    height, width = source.shape[:2]
    stats = _empty_stats((0, 0))
    air_norm = float(np.linalg.norm(air)) if np.isfinite(air).all() else 0.0
    near_air_threshold = max(1e-6, air_norm * 0.01)
    stats["airlight_norm"] = air_norm
    stats["near_air_threshold"] = near_air_threshold
    if not np.isfinite(source).all():
        raise ValueError("sample must contain only finite values")
    if not np.isfinite(air).all():
        raise ValueError("atmosphere must contain only finite values")
    if not np.isfinite(baseline_t):
        raise ValueError("baseline_t must be finite")
    if height == 0 or width == 0 or air_norm < 0.05:
        stats["total_seconds"] = perf_counter() - started
        return _zero_lut(), stats

    analysis = _analysis_image(source)
    stats["analysis_shape"] = analysis.shape[:2]
    if max(analysis.shape[:2]) < 1:
        stats["total_seconds"] = perf_counter() - started
        return _zero_lut(), stats

    proposal, confidence, candidates, reliable_rays = _ray_proposals(
        analysis, air, float(baseline_t), near_air_threshold,
    )
    stats["candidate_count"] = candidates
    stats["reliable_ray_count"] = reliable_rays

    solver_started = perf_counter()
    solved, solver_stats = _wls_pcg(analysis, proposal, confidence)
    stats.update(solver_stats)
    stats["solver_seconds"] = perf_counter() - solver_started

    lut_started = perf_counter()
    if solver_stats["solver_converged"]:
        lut, supported_cells = _trilinear_splat(analysis, solved, confidence)
    else:
        lut, supported_cells = _zero_lut(), 0
    stats["lut_seconds"] = perf_counter() - lut_started
    stats["supported_lut_cells"] = supported_cells
    stats["max_relief"] = float(np.max(lut)) if lut.size else 0.0
    stats["total_seconds"] = perf_counter() - started
    return lut, stats


def _box_mean(source: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * int(radius) + 1
    return cv2.boxFilter(
        source.astype(np.float32, copy=False), ddepth=-1, ksize=(size, size),
        normalize=True, borderType=cv2.BORDER_REFLECT_101,
    )


def _weighted_guided_coefficients(
    guide: np.ndarray,
    log_depth: np.ndarray,
    *,
    regularizer: float = _WGIF_REGULARIZER,
    coefficient_radius: int = _WGIF_COEFFICIENT_RADIUS,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Estimate locally affine log-depth coefficients with weighted guidance.

    Edge weights use a 3x3 variance and a dynamic-range-scaled epsilon.
    Regression moments and coefficient averaging share the filter radius,
    following the weighted guided filter equations independently.
    """
    guide = np.asarray(guide, dtype=np.float32)
    payload = np.asarray(log_depth, dtype=np.float32)
    if guide.ndim != 2 or payload.shape != guide.shape:
        raise ValueError("guide and log_depth must be matching 2D arrays")
    if not np.isfinite(guide).all() or not np.isfinite(payload).all():
        raise ValueError("guide and log_depth must be finite")
    if not np.isfinite(regularizer) or regularizer <= 0.0:
        raise ValueError("regularizer must be a positive finite value")

    # Eq. (5) uses 3x3 variance for the weight, whereas Eqs. (8)-(10)
    # use the chosen filter window for regression AND coefficient averaging.
    small_mean = _box_mean(guide, radius=1)
    small_variance = np.maximum(_box_mean(guide * guide, radius=1) - small_mean ** 2, 0)
    dynamic_range = float(np.ptp(guide))
    variance_floor = small_variance + max((.001 * dynamic_range) ** 2, 1e-12)
    inverse_mean = float(np.mean(1.0 / variance_floor))
    gamma = variance_floor * inverse_mean
    gamma = cv2.GaussianBlur(gamma, (0, 0), 2, borderType=cv2.BORDER_REFLECT_101)
    gamma = np.maximum(gamma, 1e-6)
    mean_guide = _box_mean(guide, coefficient_radius)
    mean_payload = _box_mean(payload, coefficient_radius)
    variance = np.maximum(_box_mean(guide * guide, coefficient_radius) - mean_guide ** 2, 0)
    covariance = _box_mean(guide * payload, coefficient_radius) - mean_guide * mean_payload
    slope = covariance / (variance + float(regularizer) / gamma)
    intercept = mean_payload - slope * mean_guide
    mean_slope = _box_mean(slope, coefficient_radius)
    mean_intercept = _box_mean(intercept, coefficient_radius)
    stats = {
        "wgi_gamma_mean": float(np.mean(gamma)),
        "wgi_gamma_min": float(np.min(gamma)),
        "wgi_gamma_max": float(np.max(gamma)),
        "wgi_variance_mean": float(np.mean(variance)),
        "wgi_regularizer": float(regularizer),
        "wgi_coefficient_radius": int(coefficient_radius),
    }
    return mean_slope.astype(np.float32), mean_intercept.astype(np.float32), stats


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
    return kernel / np.sum(kernel)


def _separable_gaussian_cube(cube: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a small separable 3D Gaussian using NumPy axis slices."""
    kernel = _gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    filtered = np.asarray(cube, dtype=np.float64)
    for axis in range(3):
        pad_width = [(0, 0), (0, 0), (0, 0)]
        pad_width[axis] = (radius, radius)
        padded = np.pad(filtered, pad_width, mode="edge")
        next_filtered = np.zeros_like(filtered)
        for index, weight in enumerate(kernel):
            slices = [slice(None), slice(None), slice(None)]
            slices[axis] = slice(index, index + filtered.shape[axis])
            next_filtered += weight * padded[tuple(slices)]
        filtered = next_filtered
    return filtered


def _project_optical_depth_lut(
    image: np.ndarray,
    optical_depth: np.ndarray,
    confidence: np.ndarray,
    baseline_depth: float,
    depth_min: float,
    depth_max: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Splat reliable refined depth into a scene-shared 65-cubed RGB field."""
    total = np.zeros((_FIELD_LUT_EDGE,) * 3, dtype=np.float64)
    mass = np.zeros_like(total)
    valid = (
        np.isfinite(image).all(axis=2)
        & np.isfinite(optical_depth)
        & np.isfinite(confidence)
        & (confidence > 0.0)
    )
    if not np.any(valid):
        lut = np.full_like(total, float(baseline_depth))
        return lut.astype(np.float32), {
            "field_lut_supported_cells": 0,
            "field_lut_max_mass": 0.0,
            "field_lut_mean_mass": 0.0,
            "field_lut_sigma": _FIELD_LUT_GAUSSIAN_SIGMA,
            "field_lut_prior_mass": _FIELD_LUT_PRIOR_MASS,
        }

    colors = np.clip(image[valid], 0.0, 1.0) * (_FIELD_LUT_EDGE - 1)
    lower = np.floor(colors).astype(np.int16)
    fraction = colors - lower
    depths = np.clip(optical_depth[valid], depth_min, depth_max).astype(np.float64)
    confidences = confidence[valid].astype(np.float64)
    for red_bit in (0, 1):
        red_weight = fraction[:, 0] if red_bit else 1.0 - fraction[:, 0]
        red_index = np.minimum(lower[:, 0] + red_bit, _FIELD_LUT_EDGE - 1)
        for green_bit in (0, 1):
            green_weight = fraction[:, 1] if green_bit else 1.0 - fraction[:, 1]
            green_index = np.minimum(lower[:, 1] + green_bit, _FIELD_LUT_EDGE - 1)
            for blue_bit in (0, 1):
                blue_weight = fraction[:, 2] if blue_bit else 1.0 - fraction[:, 2]
                blue_index = np.minimum(lower[:, 2] + blue_bit, _FIELD_LUT_EDGE - 1)
                weights = confidences * red_weight * green_weight * blue_weight
                np.add.at(total, (red_index, green_index, blue_index), weights * depths)
                np.add.at(mass, (red_index, green_index, blue_index), weights)

    smooth_total = _separable_gaussian_cube(total, _FIELD_LUT_GAUSSIAN_SIGMA)
    smooth_mass = _separable_gaussian_cube(mass, _FIELD_LUT_GAUSSIAN_SIGMA)
    prior_mass = _FIELD_LUT_PRIOR_MASS
    lut = (smooth_total + prior_mass * float(baseline_depth)) / (smooth_mass + prior_mass)
    np.clip(lut, depth_min, depth_max, out=lut)
    stats = {
        "field_lut_supported_cells": int(np.count_nonzero(mass > 1e-12)),
        "field_lut_max_mass": float(np.max(smooth_mass)),
        "field_lut_mean_mass": float(np.mean(smooth_mass)),
        "field_lut_sigma": _FIELD_LUT_GAUSSIAN_SIGMA,
        "field_lut_prior_mass": prior_mass,
    }
    return lut.astype(np.float32), stats


def _trilinear_lookup_block(source: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Query one source block against a cubic LUT with matching edge length."""
    edge = lut.shape[0]
    finite = np.isfinite(source).all(axis=2)
    safe_source = np.where(finite[:, :, None], source, 0.0)
    colors = np.clip(safe_source, 0.0, 1.0) * (edge - 1)
    lower = np.floor(colors).astype(np.int16)
    fraction = colors - lower
    value = np.zeros(source.shape[:2], dtype=np.float32)
    for red_bit in (0, 1):
        red_weight = fraction[:, :, 0] if red_bit else 1.0 - fraction[:, :, 0]
        red_index = np.minimum(lower[:, :, 0] + red_bit, edge - 1)
        for green_bit in (0, 1):
            green_weight = fraction[:, :, 1] if green_bit else 1.0 - fraction[:, :, 1]
            green_index = np.minimum(lower[:, :, 1] + green_bit, edge - 1)
            for blue_bit in (0, 1):
                blue_weight = fraction[:, :, 2] if blue_bit else 1.0 - fraction[:, :, 2]
                blue_index = np.minimum(lower[:, :, 2] + blue_bit, edge - 1)
                value += (
                    red_weight * green_weight * blue_weight
                    * lut[red_index, green_index, blue_index]
                )
    value[~finite] = 0.0
    return value


def _field_stats(shape: tuple[int, int]) -> dict[str, float | int | bool | tuple[int, int] | str]:
    return {
        "analysis_shape": shape,
        "candidate_count": 0,
        "reliable_ray_count": 0,
        "nonlocal_field_active": False,
        "solver_iterations": 0,
        "solver_residual": 0.0,
        "solver_converged": True,
        "solver_seconds": 0.0,
        "wgi_seconds": 0.0,
        "field_lut_seconds": 0.0,
        "lookup_seconds": 0.0,
        "field_lut_edge": _FIELD_LUT_EDGE,
        "total_seconds": 0.0,
        "initial_t_min": 0.0,
        "initial_t_max": 0.0,
        "transmission_min": 0.0,
        "transmission_max": 0.0,
        "fallback_reason": "",
    }


def _constant_transmission(
    shape: tuple[int, int], value: float,
) -> np.ndarray:
    return np.full(shape, np.float32(value), dtype=np.float32)


def build_nonlocal_transmission(
    source: np.ndarray,
    atmosphere: np.ndarray,
    baseline_t: float,
    strength: float = 1.0,
) -> tuple[np.ndarray, dict[str, float | int | bool | tuple[int, int] | str]]:
    """Estimate a spatial transmission field from reliable atmospheric rays.

    The low-resolution non-local ray field is refined in ``-log(t)`` with an
    independently implemented weighted guided filter. The refined field is
    splatted into a scene-shared 65-cubed RGB optical-depth LUT with a continuous
    baseline prior, then queried in 256-row output blocks. Equal RGB values
    therefore receive equal transmission at every image position.
    """
    started = perf_counter()
    image = np.asarray(source, dtype=np.float32)
    air = np.asarray(atmosphere, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("source must have shape HxWx3")
    if air.shape != (3,):
        raise ValueError("atmosphere must have shape (3,)")
    if not np.isfinite(air).all():
        raise ValueError("atmosphere must contain only finite values")
    if not np.isfinite(baseline_t):
        raise ValueError("baseline_t must be finite")
    if not np.isfinite(strength):
        raise ValueError("strength must be finite")
    height, width = image.shape[:2]
    for row_start in range(0, height, 256):
        if not np.isfinite(image[row_start:row_start + 256]).all():
            raise ValueError("source must contain only finite values")

    base = float(np.clip(baseline_t, 1e-4, 1.0))
    effect = float(np.clip(strength, 0.0, 1.0))
    stats = _field_stats((0, 0))
    air_norm = float(np.linalg.norm(air))
    stats["airlight_norm"] = air_norm
    stats["near_air_threshold"] = max(1e-6, air_norm * 0.01)

    if height == 0 or width == 0:
        stats["analysis_shape"] = (height, width)
        stats["fallback_reason"] = "empty_source"
        stats["total_seconds"] = perf_counter() - started
        return np.empty((height, width), dtype=np.float32), stats
    if effect == 0.0:
        result = np.ones((height, width), dtype=np.float32)
        stats["analysis_shape"] = _analysis_shape(height, width)
        stats["initial_t_min"] = stats["initial_t_max"] = 1.0
        stats["transmission_min"] = stats["transmission_max"] = 1.0
        stats["fallback_reason"] = "zero_strength"
        stats["total_seconds"] = perf_counter() - started
        return result, stats

    if air_norm < 0.05:
        result = _constant_transmission((height, width), base)
        stats["fallback_reason"] = "low_airlight_norm"
        stats["initial_t_min"] = stats["initial_t_max"] = base
        stats["transmission_min"] = stats["transmission_max"] = base
        stats["total_seconds"] = perf_counter() - started
        return result, stats

    analysis = _analysis_image(image)
    analysis_height, analysis_width = analysis.shape[:2]
    stats["analysis_shape"] = (analysis_height, analysis_width)
    proposal, confidence, candidate_count, reliable_rays = _ray_proposals(
        analysis, air, base, float(stats["near_air_threshold"]),
        proposal_kind="transmission", strength=effect,
    )
    stats["candidate_count"] = candidate_count
    stats["reliable_ray_count"] = reliable_rays
    if reliable_rays == 0 or not np.any(confidence > 0.0):
        result = _constant_transmission((height, width), base)
        stats["fallback_reason"] = "no_reliable_rays"
        stats["initial_t_min"] = stats["initial_t_max"] = base
        stats["transmission_min"] = stats["transmission_max"] = base
        stats["total_seconds"] = perf_counter() - started
        return result, stats

    solver_started = perf_counter()
    initial_t, solver_stats = _wls_pcg(
        analysis, proposal, confidence, prior=base,
    )
    stats.update(solver_stats)
    stats["solver_seconds"] = perf_counter() - solver_started
    if not solver_stats["solver_converged"]:
        result = _constant_transmission((height, width), base)
        stats["fallback_reason"] = "solver_nonconverged"
        stats["initial_t_min"] = stats["initial_t_max"] = base
        stats["transmission_min"] = stats["transmission_max"] = base
        stats["total_seconds"] = perf_counter() - started
        return result, stats

    initial_t = np.clip(initial_t, 1e-4, 1.0).astype(np.float32, copy=False)
    initial_min = float(np.min(initial_t))
    initial_max = float(np.max(initial_t))
    stats["initial_t_min"] = initial_min
    stats["initial_t_max"] = initial_max
    low_luma = np.einsum("ijk,k->ij", analysis, _GUIDE_LUMA, optimize=True).astype(np.float32)
    low_log_depth = -np.log(initial_t)

    wgi_started = perf_counter()
    slope, intercept, wgi_stats = _weighted_guided_coefficients(
        low_luma, low_log_depth,
    )
    stats.update(wgi_stats)
    stats["wgi_seconds"] = perf_counter() - wgi_started

    guided_depth = slope * low_luma + intercept
    baseline_depth = float(-np.log(base))
    depth_min = min(float(-np.log(initial_max)), baseline_depth)
    depth_max = max(float(-np.log(initial_min)), baseline_depth)
    np.clip(guided_depth, depth_min, depth_max, out=guided_depth)
    lut_started = perf_counter()
    field_lut, field_lut_stats = _project_optical_depth_lut(
        analysis, guided_depth, confidence, baseline_depth, depth_min, depth_max,
    )
    stats.update(field_lut_stats)
    stats["field_lut_seconds"] = perf_counter() - lut_started

    # The color LUT is scene-shared; query each full-resolution row block with
    # the same table and never recompute confidence or smoothing at tile edges.
    result = np.empty((height, width), dtype=np.float32)
    lookup_started = perf_counter()
    for row_start in range(0, height, 256):
        row_end = min(height, row_start + 256)
        optical_depth = _trilinear_lookup_block(
            image[row_start:row_end], field_lut,
        )
        np.clip(optical_depth, depth_min, depth_max, out=optical_depth)
        block = image[row_start:row_end]
        lower_bound = np.clip(1.0 - np.min(block / np.maximum(air, 1e-6), axis=2), 0, 1)
        lower_bound = 1.0 - effect * (1.0 - lower_bound)
        result[row_start:row_end] = np.maximum(np.exp(-optical_depth), lower_bound).astype(np.float32)
    stats["lookup_seconds"] = perf_counter() - lookup_started
    stats["nonlocal_field_active"] = True
    stats["transmission_min"] = float(np.min(result))
    stats["transmission_max"] = float(np.max(result))
    stats["fallback_reason"] = ""
    stats["endpoint_calibration"] = "dark_channel"
    stats["total_seconds"] = perf_counter() - started
    return result, stats


def lookup_transmission_relief(source: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinearly query a reliability LUT in 256-row blocks."""
    image = np.asarray(source, dtype=np.float32)
    table = np.asarray(lut, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("source must have shape HxWx3")
    if table.shape != (_LUT_EDGE, _LUT_EDGE, _LUT_EDGE):
        raise ValueError("lut must have shape 33x33x33")
    if not np.isfinite(table).all():
        raise ValueError("lut must contain only finite values")
    height, width = image.shape[:2]
    result = np.zeros((height, width), dtype=np.float32)
    if height == 0 or width == 0:
        return result
    for row_start in range(0, height, 256):
        row_end = min(height, row_start + 256)
        block = image[row_start:row_end]
        finite = np.isfinite(block).all(axis=2)
        if not np.any(finite):
            continue
        safe_block = np.where(finite[:, :, None], block, 0.0)
        colors = np.clip(safe_block, 0.0, 1.0) * (_LUT_EDGE - 1)
        lower = np.floor(colors).astype(np.int16)
        fraction = colors - lower
        value = np.zeros(block.shape[:2], dtype=np.float32)
        for red_bit in (0, 1):
            red_weight = fraction[:, :, 0] if red_bit else 1.0 - fraction[:, :, 0]
            red_index = np.minimum(lower[:, :, 0] + red_bit, _LUT_EDGE - 1)
            for green_bit in (0, 1):
                green_weight = fraction[:, :, 1] if green_bit else 1.0 - fraction[:, :, 1]
                green_index = np.minimum(lower[:, :, 1] + green_bit, _LUT_EDGE - 1)
                for blue_bit in (0, 1):
                    blue_weight = fraction[:, :, 2] if blue_bit else 1.0 - fraction[:, :, 2]
                    blue_index = np.minimum(lower[:, :, 2] + blue_bit, _LUT_EDGE - 1)
                    value += (
                        red_weight * green_weight * blue_weight
                        * table[red_index, green_index, blue_index]
                    )
        value[~finite] = 0.0
        result[row_start:row_end] = np.clip(value, 0.0, 1.0)
    return result
