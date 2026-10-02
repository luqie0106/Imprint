"""Conservative spatial transmission for automatic photo dehazing.

The scene cue is only a relative haze estimate, not a metric depth map. It is
smoothed with a fast guided filter and faded at strong edges. Confident bright
sky is protected using the full-resolution guide; uncertain sky stays near the
global reference to avoid a skyline halo.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from dehaze import (
    DehazeParams, _apply_dehaze_cpu,
    _global_transmission, _smooth_chroma_caps, _smooth_chroma_gamut,
)


_LUMA = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
_ANALYSIS_EDGE = 768
_TILE_ROWS = 256
_DETAIL_RADIUS_AT_PREVIEW = 8
_DETAIL_REFERENCE_EDGE = 1800
_last_backend = "尚未处理"


def _sample_peak(image: np.ndarray) -> float:
    return 1.0 if image.dtype == np.float32 else float(np.iinfo(image.dtype).max)


def get_last_spatial_backend() -> str:
    """Return the most recent automatic spatial render path for diagnostics."""
    return _last_backend


def _scene_airlight(image_rgb: np.ndarray) -> np.ndarray:
    """Estimate airlight from broad bright upper regions, excluding lamps.

    The RAW-linear brightness of a sky may be far below 0.35. Selecting the
    single brightest dark-channel pixels instead often selects lit windows in
    night scenes. Both mistakes cause the whole image to be darkened.
    """
    peak = float(np.iinfo(image_rgb.dtype).max)
    height, width = image_rgb.shape[:2]
    scale = min(1.0, _ANALYSIS_EDGE / max(height, width))
    if scale < 1.0:
        image_rgb = cv2.resize(
            image_rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    rgb = image_rgb.astype(np.float32) / peak
    upper = rgb[:max(1, round(rgb.shape[0] * 0.45))]
    luma = upper @ _LUMA
    local_mean = _box(luma, 5)
    local_variance = np.maximum(_box(luma * luma, 5) - local_mean * local_mean, 0.0)
    low, high = np.percentile(luma, (70, 96))
    texture_limit = np.percentile(local_variance, 65)
    candidates = (luma >= low) & (luma <= high) & (local_variance <= texture_limit)
    if np.count_nonzero(candidates) < 8:
        candidates = (luma >= low) & (luma <= high)
    if np.count_nonzero(candidates) < 1:
        return np.clip(np.median(upper.reshape(-1, 3), axis=0), 0.01, 1.0).astype(np.float32)
    return np.clip(np.median(upper[candidates], axis=0), 0.01, 1.0).astype(np.float32)


def _smoothstep(low: float, high: float, value: np.ndarray) -> np.ndarray:
    position = np.clip((value - low) / (high - low), 0.0, 1.0)
    return position * position * (3.0 - 2.0 * position)


def _box(image: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.boxFilter(image, cv2.CV_32F, (size, size), normalize=True,
                         borderType=cv2.BORDER_REFLECT_101)


def _guided_coefficients(
    guide: np.ndarray, estimate: np.ndarray, radius: int = 12,
    epsilon: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-resolution linear coefficients of the grayscale guided filter."""
    mean_guide = _box(guide, radius)
    mean_estimate = _box(estimate, radius)
    variance = _box(guide * guide, radius) - mean_guide * mean_guide
    covariance = _box(guide * estimate, radius) - mean_guide * mean_estimate
    slope = covariance / np.maximum(variance + epsilon, 1e-7)
    intercept = mean_estimate - slope * mean_guide
    return _box(slope, radius), _box(intercept, radius)


def _column_sky_mask(
    guide: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a soft skyline independently per column using its upper sky.

    Local references preserve dark, vignetted sky. The first persistent drop
    below the upper-column reference marks the roofline; no image-wide
    connected component can carve a circular boundary into the sky.
    """
    height, width = guide.shape
    top_rows = max(4, round(height * 0.12))
    column_reference = np.percentile(guide[:top_rows], 50, axis=0).astype(np.float32)
    column_reference = cv2.GaussianBlur(
        column_reference[None, :], (0, 0), sigmaX=max(2.0, width / 96.0),
    ).reshape(-1)
    reference = np.maximum(column_reference, 0.025)

    # Slight horizontal smoothing reduces sensor noise and isolated windows
    # without flattening the vertical skyline profile.
    smooth_guide = cv2.GaussianBlur(guide, (0, 0), sigmaX=1.0, sigmaY=0.6)
    skyline = np.full(width, height * 0.94, dtype=np.float32)
    found = np.zeros(width, dtype=bool)
    first_row = max(4, round(height * 0.06))
    last_row = max(first_row + 1, round(height * 0.88))

    # Compare short averages above and below each row. Require the lower
    # region to remain dark, so thin clouds and isolated texture do not become
    # skyline boundaries.
    for row in range(first_row, min(last_row, height - 8)):
        above = np.mean(smooth_guide[row - 3:row], axis=0)
        below = np.mean(smooth_guide[row:row + 4], axis=0)
        persistent = np.mean(smooth_guide[row:row + 8], axis=0)
        above_relative = above / reference
        below_relative = below / reference
        persistent_relative = persistent / reference
        crossed = (
            (~found)
            & (below_relative <= 0.95)
            & (above_relative - below_relative >= 0.025)
            & (persistent_relative <= 0.96)
        )
        skyline[crossed] = row
        found |= crossed

    # Smooth only across neighboring columns. A short kernel keeps narrow
    # towers while removing one-column edge noise and avoiding sky-wide discs.
    skyline = cv2.GaussianBlur(
        skyline[None, :], (0, 0), sigmaX=max(1.0, width / 256.0),
    ).reshape(-1)
    row_position = np.arange(height, dtype=np.float32)[:, None]
    distance_above_roof = skyline[None, :] - row_position
    support = _smoothstep(-3.5, 3.5, distance_above_roof)

    mean = _box(guide, 7)
    variance = np.maximum(_box(guide * guide, 7) - mean * mean, 0.0)
    relative_texture = np.sqrt(variance) / (mean + 0.025)
    normalized_luma = guide / reference[None, :]
    brightness = _smoothstep(0.84, 0.99, normalized_luma)
    # Texture acts as a gentle confidence cue rather than a hard region mask.
    texture_confidence = 0.85 + 0.15 * (
        1.0 - _smoothstep(0.12, 0.38, relative_texture)
    )
    return (
        (support * brightness * texture_confidence).astype(np.float32),
        column_reference,
        skyline,
    )


def _refine_sky_protection(
    support: np.ndarray,
    full_guide: np.ndarray,
    column_reference: np.ndarray,
    edge: np.ndarray,
) -> np.ndarray:
    """Refine the low-resolution skyline into a soft full-resolution mask."""
    height, width = full_guide.shape
    if support.shape != full_guide.shape:
        support = cv2.resize(
            support, (width, height), interpolation=cv2.INTER_LINEAR,
        )
    if column_reference.shape[0] != width:
        column_reference = cv2.resize(
            column_reference[None, :], (width, 1), interpolation=cv2.INTER_LINEAR,
        ).reshape(-1)
    reference = np.maximum(column_reference, 0.025)[None, :]

    # Recheck local image evidence at native resolution so a coarse support
    # mask cannot spill across a roofline or fill a detailed facade.
    radius = max(2, min(9, round(min(height, width) / 96)))
    mean = _box(full_guide, radius)
    variance = np.maximum(_box(full_guide * full_guide, radius) - mean * mean, 0.0)
    relative_texture = np.sqrt(variance) / (mean + 0.025)
    normalized_luma = full_guide / reference
    brightness = _smoothstep(0.84, 0.99, normalized_luma)
    smoothness = 1.0 - _smoothstep(0.10, 0.34, relative_texture)

    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    upper = 1.0 - _smoothstep(0.90, 0.99, np.broadcast_to(y, full_guide.shape))
    protection = np.clip(support, 0.0, 1.0) * brightness * smoothness * upper
    # The edge guard removes protection across the silhouette itself; a small
    # blur on both sides then yields a conservative, continuous handoff.
    protection *= 1.0 - np.clip(edge, 0.0, 1.0)
    protection = cv2.GaussianBlur(protection, (0, 0), 2.0)
    return np.clip(protection, 0.0, 1.0)


def _transmission_map(image_rgb: np.ndarray, params: DehazeParams) -> np.ndarray:
    """Estimate bounded transmission; keep sky and hard silhouettes stable."""
    height, width = image_rgb.shape[:2]
    peak = _sample_peak(image_rgb)
    scale = min(1.0, _ANALYSIS_EDGE / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        reduced = cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)
    else:
        reduced = image_rgb
    rgb = reduced.astype(np.float32) / peak
    guide = rgb @ _LUMA
    low_height, low_width = guide.shape
    global_t, _ = _global_transmission(rgb, params)
    floor = 0.20 + 0.16 * params.fog_retention + 0.08 * params.naturalness

    # The upper image supplies a sky colour/brightness reference. This is a
    # soft prior only: dark foregrounds and artificial lights are excluded by
    # the luminance and local-variation guards below.
    top = guide[:max(1, low_height // 5)]
    sky_reference = max(0.025, float(np.median(top)))
    local_mean = _box(guide, 7)
    local_variance = np.maximum(_box(guide * guide, 7) - local_mean * local_mean, 0.0)
    relative_texture = np.sqrt(local_variance) / (local_mean + 0.025)
    low_texture = 1.0 - _smoothstep(0.08, 0.35, relative_texture)
    yy = np.linspace(0.0, 1.0, low_height, dtype=np.float32)[:, None]
    depth_prior = _smoothstep(0.02, 0.78, 0.86 - yy)
    normalized_dark = np.clip(np.min(rgb, axis=2) / max(sky_reference, 0.05), 0.0, 1.0)
    haze_cue = (
        0.52 * depth_prior
        + 0.28 * _smoothstep(0.08, 0.58, normalized_dark)
        + 0.20 * low_texture
    ).astype(np.float32)

    # RAW-linear daylight can have a sky median below 0.15, while night city
    # lights can be much brighter. Classify the whole scene instead of using
    # an absolute sky threshold, and recognize a broad saturated sun region.
    scene_median = float(np.median(guide))
    upper_peak = float(np.percentile(np.max(rgb[:max(1, low_height // 2)], axis=2), 99))
    daylight_confidence = max(
        float(_smoothstep(0.06, 0.10, np.asarray(scene_median)))
        * float(_smoothstep(0.09, 0.13, np.asarray(sky_reference))),
        float(_smoothstep(0.16, 0.24, np.asarray(sky_reference))),
        float(_smoothstep(0.75, 0.95, np.asarray(upper_peak))),
    )
    if daylight_confidence > 1e-6:
        sky, column_sky_reference, _ = _column_sky_mask(guide)
        sky *= daylight_confidence
    else:
        sky = np.zeros_like(guide)
        column_sky_reference = np.empty(0, dtype=np.float32)
    proposed_offset = (0.25 - haze_cue) * (0.70 * (1.0 - global_t))
    # Dark cloud structure is not a reliable depth cue either. Keep night
    # scenes near one scene-wide transmission instead of digging a dark band
    # into the cloud layer at maximum strength.
    proposed_offset *= (1.0 - sky) * daylight_confidence

    # Guided filtering smooths the relative-haze estimate along image edges.
    # Following Fast Guided Filter, coefficients are computed at the analysis
    # size and upsampled before the final evaluation against full-size luma.
    slope, intercept = _guided_coefficients(guide, proposed_offset)
    if scale < 1.0:
        target_size = (width, height)
        slope = cv2.resize(slope, target_size, interpolation=cv2.INTER_LINEAR)
        intercept = cv2.resize(intercept, target_size, interpolation=cv2.INTER_LINEAR)
        full_guide = (image_rgb.astype(np.float32) / peak) @ _LUMA
        full_sky = cv2.resize(sky, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        full_guide = guide
        full_sky = sky
    offset = slope * full_guide + intercept

    # An explicit boundary guard suppresses residual guided-filter leakage at
    # building/tree silhouettes. Across these pixels, fall back to the stable
    # global transmission instead of creating a bright or dark rim.
    grad_x = cv2.Sobel(full_guide, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(full_guide, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.maximum(np.abs(grad_x), np.abs(grad_y))
    edge = _smoothstep(0.035, 0.14, edge)
    edge = cv2.dilate(edge, np.ones((5, 5), dtype=np.uint8))
    edge = cv2.GaussianBlur(edge, (0, 0), 2.0)
    offset *= (1.0 - edge) * (1.0 - full_sky)
    limit = min(0.18, 0.38 * (1.0 - global_t))
    # At the maximum setting a sunlit hazy midtone should still move visibly.
    # Scale this daylight-only boost by source luminance so silhouettes are
    # not driven farther toward black; bright source pixels retain the
    # separate highlight protection in the renderer.
    sunlight_confidence = float(_smoothstep(0.75, 0.95, np.asarray(upper_peak)))
    daylight_boost = (
        0.10 * params.strength * sunlight_confidence
        * _smoothstep(0.04, 0.22, full_guide)
    )
    transmission = np.clip(
        global_t + np.clip(offset, -limit, limit) - daylight_boost,
        floor, 1.0,
    )

    # Only exceptionally bright, clear sky warrants a near-original
    # transmission. Restoring a hazy RAW-linear sky to transmission 1 removes
    # nearly all of the dehazing the user requested.
    if daylight_confidence > 1e-6:
        sky_keep = _refine_sky_protection(
            full_sky, full_guide, column_sky_reference, edge,
        )
        sky_keep *= float(_smoothstep(0.22, 0.55, np.asarray(sky_reference)))
        transmission = transmission * (1.0 - sky_keep) + sky_keep
    return np.clip(transmission, floor, 1.0).astype(np.float32)


def _brightness_gain(source: np.ndarray, result: np.ndarray, p: DehazeParams) -> float:
    """One scene-wide midtone gain from a regular sample of the finished image."""
    if p.brightness_protection <= 1e-6:
        return 1.0
    height, width = source.shape[:2]
    stride = max(1, int(np.sqrt(height * width / 250_000)))
    peak = float(np.iinfo(source.dtype).max)
    source_luma = (source[::stride, ::stride].astype(np.float32) / peak) @ _LUMA
    result_luma = (result[::stride, ::stride].astype(np.float32) / peak) @ _LUMA
    valid = (source_luma > 0.08) & (source_luma < 0.88)
    if int(np.count_nonzero(valid)) < 8:
        return 1.0
    source_median = float(np.median(source_luma[valid]))
    result_median = float(np.median(result_luma[valid]))
    if source_median <= 1e-5 or result_median <= 1e-5:
        return 1.0
    drop_ev = np.log2(source_median / result_median)
    allowed_ev = 0.08 + 0.22 * p.strength
    compensation_ev = min(0.40, max(0.0, drop_ev - allowed_ev) * p.brightness_protection)
    return float(2.0 ** compensation_ev)


def _apply_global_brightness_gain(
    source: np.ndarray, result: np.ndarray, gain: float,
) -> np.ndarray:
    if gain <= 1.0 + 1e-6:
        return result
    peak = float(np.iinfo(source.dtype).max)
    original = source.astype(np.float32) / peak
    image = result.astype(np.float32) / peak
    luma = image @ _LUMA
    curve = luma * gain / (1.0 + (gain - 1.0) * luma)
    image *= (curve / np.maximum(luma, 1e-6))[..., None]
    image = _smooth_chroma_gamut(np.clip(image, 0.0, 1.0))
    saturation_limit = max(0.0, 0.97 - 1.5 / peak)
    luma_cap = np.where((original @ _LUMA) < 0.97, saturation_limit, 1.0)
    luma_scale = np.minimum(1.0, luma_cap / np.maximum(image @ _LUMA, 1e-4))
    image *= luma_scale[..., None]
    channel_cap = np.where(original < 0.97, saturation_limit, 1.0)
    image = _smooth_chroma_caps(image, channel_cap)
    return np.clip(np.rint(image * peak), 0, peak).astype(source.dtype)


def _shadow_detail_support(source: np.ndarray) -> np.ndarray:
    """Broad texture cue: retain detailed land, not smooth hazy sea/sky."""
    height, width = source.shape[:2]
    scale = min(1.0, _ANALYSIS_EDGE / max(height, width))
    if scale < 1.0:
        reduced = cv2.resize(
            source, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        reduced = source
    luma = (reduced.astype(np.float32) / np.iinfo(source.dtype).max) @ _LUMA
    mean = _box(luma, 7)
    variance = np.maximum(_box(luma * luma, 7) - mean * mean, 0.0)
    texture = np.sqrt(variance) / (mean + 0.025)
    # A single thin bright structure has a high local gradient too. Require
    # texture across a broad area so its edge cannot lift the adjacent sky.
    texture_pixels = _smoothstep(0.025, 0.060, texture)
    density = _box(texture_pixels, 20)
    support = _smoothstep(0.12, 0.45, density)
    support = cv2.GaussianBlur(support, (0, 0), 4.0)
    rows = np.linspace(0.0, 1.0, support.shape[0], dtype=np.float32)[:, None]
    support *= _smoothstep(0.08, 0.20, rows)
    if scale < 1.0:
        support = cv2.resize(support, (width, height), interpolation=cv2.INTER_LINEAR)
    # Sparse structures in an otherwise smooth sea/sky scene must not make
    # the adjacent background eligible for the land-detail guard.
    scene_texture = float(np.mean(support))
    support *= float(_smoothstep(0.10, 0.30, np.asarray(scene_texture)))
    return support.astype(np.float32)


def _structure_detail_support(source: np.ndarray) -> np.ndarray:
    """Find broad, textured land regions and fade out at daylight skylines."""
    height, width = source.shape[:2]
    peak = float(np.iinfo(source.dtype).max)
    scale = min(1.0, _ANALYSIS_EDGE / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        reduced = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
    else:
        reduced = source
    rgb = reduced.astype(np.float32) / peak
    guide = rgb @ _LUMA
    low_height, low_width = guide.shape

    mean = _box(guide, 7)
    variance = np.maximum(_box(guide * guide, 7) - mean * mean, 0.0)
    local_std = np.sqrt(variance)
    relative_texture = local_std / (mean + 0.025)
    # Requiring both normalized and absolute contrast rejects sensor grain and
    # faint water ripples while retaining visible building and tree texture.
    texture = (
        _smoothstep(0.025, 0.060, relative_texture)
        * _smoothstep(0.0012, 0.0045, local_std)
    )
    density = _box(texture, 10)
    support = _smoothstep(0.10, 0.36, density)
    support = cv2.GaussianBlur(support, (0, 0), 1.5)

    # A shoreline or a single building silhouette can have high local
    # variance without carrying usable texture. Favor regions with detail in
    # both axes; rippled water and one-dimensional borders stay subdued.
    horizontal_energy = np.zeros_like(guide)
    vertical_energy = np.zeros_like(guide)
    horizontal_energy[:, 1:] = np.abs(np.diff(guide, axis=1))
    vertical_energy[1:, :] = np.abs(np.diff(guide, axis=0))
    horizontal_energy = _box(horizontal_energy, 4)
    vertical_energy = _box(vertical_energy, 4)
    direction_balance = (
        2.0 * np.minimum(horizontal_energy, vertical_energy)
        / (horizontal_energy + vertical_energy + 1e-5)
    )
    support *= _smoothstep(0.12, 0.60, direction_balance)

    rows = np.arange(low_height, dtype=np.float32)[:, None]
    support *= _smoothstep(0.08 * low_height, 0.20 * low_height, rows)

    top = guide[:max(1, low_height // 5)]
    scene_median = float(np.median(guide))
    sky_reference = float(np.median(top))
    upper_peak = float(np.percentile(np.max(rgb[:max(1, low_height // 2)], axis=2), 99))
    daylight_confidence = max(
        float(_smoothstep(0.06, 0.10, np.asarray(scene_median)))
        * float(_smoothstep(0.09, 0.13, np.asarray(sky_reference))),
        float(_smoothstep(0.16, 0.24, np.asarray(sky_reference))),
        float(_smoothstep(0.75, 0.95, np.asarray(upper_peak))),
    )
    if daylight_confidence > 1e-6:
        _, _, skyline = _column_sky_mask(guide)
        distance_below_skyline = rows - skyline[None, :]
        # Leave a soft safety strip around roofs and the waterline. That keeps
        # the local-contrast pass from emphasizing haze transitions or halos.
        land_confidence = _smoothstep(
            -0.012 * low_height, 0.045 * low_height, distance_below_skyline,
        )
        support *= land_confidence

    if scale < 1.0:
        support = cv2.resize(support, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(support, 0.0, 1.0).astype(np.float32)


def _protect_spatial_details(source: np.ndarray, result: np.ndarray,
                             strength: float, support: np.ndarray,
                             bright_support: np.ndarray | None = None,
                             bright_target_scale: float = 1.0) -> np.ndarray:
    """Keep dark detail and stop bright structures glowing against darkened sky."""
    peak = float(np.iinfo(source.dtype).max)
    original = source.astype(np.float32) / peak
    enhanced = result.astype(np.float32) / peak
    source_y = original @ _LUMA
    result_y = enhanced @ _LUMA
    shadow = 1.0 - _smoothstep(0.06, 0.30, source_y)
    minimum_y = source_y * (1.0 - 0.05 * strength * shadow)
    shadow_recovery = np.clip(
        (minimum_y - result_y) / np.maximum(source_y - result_y, 1e-6),
        0.0, 1.0,
    ) * support
    enhanced += (original - enhanced) * shadow_recovery[..., None]
    # Limit added radiance without making bright pixels darker than their
    # source. The previous sky-derived target (as low as 0.82 * source) and
    # airlight threshold carved an annulus into smooth solar glare. A scalar
    # luminance ceiling has no spatial mask or scene-brightness threshold and
    # cannot expand the source highlight footprint.
    result_y = enhanced @ _LUMA
    scale = np.minimum(1.0, source_y / np.maximum(result_y, 1e-6))
    enhanced *= scale[..., None]
    if bright_support is not None and bright_target_scale < 1.0:
        # Only actual fine bright structures follow their darker background.
        # Broad solar gradients are excluded by the high-pass support below.
        target = original * bright_target_scale
        correction = bright_support * ((enhanced @ _LUMA) > source_y * bright_target_scale)
        enhanced += (target - enhanced) * correction[..., None]
    return np.clip(np.rint(enhanced * peak), 0, peak).astype(source.dtype)


def _bright_structure_support(source: np.ndarray, radius: int) -> np.ndarray:
    """Positive fine detail, excluding smooth glare and saturated highlights."""
    original = source.astype(np.float32) / np.iinfo(source.dtype).max
    luma = original @ _LUMA
    high_pass = luma - _box(luma, radius)
    support = _smoothstep(0.008, 0.035, high_pass)
    support *= 1.0 - _smoothstep(0.55, 0.80, np.max(original, axis=2))
    return support.astype(np.float32)


def _protect_dark_details_in_tiles(source: np.ndarray, result: np.ndarray,
                                   strength: float,
                                   local_contrast: float = 0.25,
                                   naturalness: float = 0.70) -> np.ndarray:
    support = _shadow_detail_support(source)
    sample_step = max(1, round(max(source.shape[:2]) / 768))
    sample = source[::sample_step, ::sample_step].astype(np.float32) / np.iinfo(source.dtype).max
    upper = sample[:max(1, sample.shape[0] // 4)]
    upper_y = upper @ _LUMA
    scene_y = sample @ _LUMA
    upper_peak = float(np.percentile(np.max(upper, axis=2), 99))
    daylight = (
        (float(np.median(upper_y)) > 0.11 and float(np.median(scene_y)) > 0.06)
        or upper_peak > 0.95
    )
    bright_target_scale = 1.0
    if daylight:
        result_upper = result[::sample_step, ::sample_step][:upper.shape[0]]
        result_y = (result_upper.astype(np.float32) / np.iinfo(result.dtype).max) @ _LUMA
        valid = (upper_y > 0.04) & (upper_y < 0.40) & (np.max(upper, axis=2) < 0.70)
        if np.count_nonzero(valid) >= 16:
            bright_target_scale = float(np.clip(np.median(result_y[valid] / upper_y[valid]), 0.82, 1.0))
    radius = max(2, round(4 * max(source.shape[:2]) / _DETAIL_REFERENCE_EDGE))
    for start in range(0, source.shape[0], _TILE_ROWS):
        stop = min(source.shape[0], start + _TILE_ROWS)
        context_start = max(0, start - radius)
        context_stop = min(source.shape[0], stop + radius)
        bright_support = _bright_structure_support(source[context_start:context_stop], radius)
        bright_support = bright_support[start - context_start:stop - context_start]
        result[start:stop] = _protect_spatial_details(
            source[start:stop], result[start:stop], strength,
            support[start:stop], bright_support, bright_target_scale,
        )

    del support
    structure_support = _structure_detail_support(source)
    _enhance_structure_details_in_tiles(
        result, structure_support, strength, local_contrast, naturalness,
    )
    return result


def _enhance_structure_details_in_tiles(
    result: np.ndarray,
    support: np.ndarray,
    strength: float,
    local_contrast: float,
    naturalness: float,
) -> np.ndarray:
    """Apply bounded luma detail gain with a halo-safe, chunked blur."""
    amount = (
        1.65 * float(np.clip(local_contrast, 0.0, 1.0))
        * float(np.clip(strength, 0.0, 1.0))
        * (1.20 - 0.30 * float(np.clip(naturalness, 0.0, 1.0)))
    )
    if amount <= 1e-6:
        return result

    peak = float(np.iinfo(result.dtype).max)
    height = result.shape[0]
    radius = max(
        2,
        round(
            _DETAIL_RADIUS_AT_PREVIEW
            * max(result.shape[:2]) / _DETAIL_REFERENCE_EDGE
        ),
    )
    for start in range(0, height, _TILE_ROWS):
        stop = min(height, start + _TILE_ROWS)
        context_start = max(0, start - radius)
        context_stop = min(height, stop + radius)
        context = result[context_start:context_stop].astype(np.float32) / peak
        luma = context @ _LUMA
        local_base = _box(luma, radius)
        detail = luma - local_base
        core_start = start - context_start
        core_stop = core_start + (stop - start)
        detail = detail[core_start:core_stop]
        core_luma = luma[core_start:core_stop]
        core_support = support[start:stop]

        delta = detail * (amount * core_support)
        # Limit shadow compression in textured regions and bound highlights so
        # the gain can clarify structures without turning details into spots.
        negative_limit = np.minimum(0.04 * core_luma, 0.004)
        positive_limit = np.minimum(0.08 * core_luma, 0.012)
        delta = np.clip(delta, -negative_limit, positive_limit)
        adjusted_luma = np.clip(core_luma + delta, 0.0, 1.0)

        core_rgb = context[core_start:core_stop]
        scale = adjusted_luma / np.maximum(core_luma, 1e-6)
        result[start:stop] = np.clip(
            np.rint(core_rgb * scale[..., None] * peak), 0, peak,
        ).astype(result.dtype)
    return result


def apply_spatial_dehaze(
    image_rgb: np.ndarray, params: DehazeParams, *, backend: str = "cpu",
) -> np.ndarray:
    """Compatibility entry for the unified linear float physical pipeline."""
    global _last_backend
    from dehaze_physical import apply_physical_dehaze, get_last_physical_backend
    result = apply_physical_dehaze(image_rgb, params, backend=backend, spatial=True)
    _last_backend = get_last_physical_backend()
    return result
