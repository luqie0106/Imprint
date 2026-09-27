"""Conservative spatial transmission for automatic photo dehazing.

The scene cue is only a relative haze estimate, not a metric depth map. It is
smoothed with a fast guided filter and faded at strong edges. Bright sky stays
on the global reference transmission because dark-channel priors are unreliable
there; a mistaken sky estimate must not produce a skyline halo.
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


def _transmission_map(image_rgb: np.ndarray, params: DehazeParams) -> np.ndarray:
    """Estimate bounded transmission; keep sky and hard silhouettes stable."""
    height, width = image_rgb.shape[:2]
    peak = float(np.iinfo(image_rgb.dtype).max)
    scale = min(1.0, _ANALYSIS_EDGE / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        reduced = cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)
    else:
        reduced = image_rgb
    rgb = reduced.astype(np.float32) / peak
    guide = rgb @ _LUMA
    low_height, low_width = guide.shape
    global_t, floor = _global_transmission(rgb, params)

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

    # A near-flat, relatively bright upper region is likely sky. HazeFlow's
    # transmission-refinement discussion explicitly warns that dark-channel
    # estimates fail there. The global result is the safe reference for sky.
    upper = 1.0 - _smoothstep(0.58, 0.86, np.broadcast_to(yy, guide.shape))
    sky = (
        _smoothstep(0.50, 0.82, guide / sky_reference)
        * (0.45 + 0.55 * low_texture)
        * upper
    ).astype(np.float32)
    proposed_offset = (0.25 - haze_cue) * (0.70 * (1.0 - global_t))
    proposed_offset *= (1.0 - sky)

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
    limit = min(0.10, 0.25 * (1.0 - global_t))
    return np.clip(global_t + np.clip(offset, -limit, limit), floor, 1.0).astype(np.float32)


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


def apply_spatial_dehaze(
    image_rgb: np.ndarray, params: DehazeParams, *, backend: str = "cpu",
) -> np.ndarray:
    """Apply the existing color/tone pipeline with one bounded t(x) per pixel."""
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a numpy array")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if image_rgb.dtype not in (np.uint8, np.uint16):
        raise TypeError("image_rgb must use uint8 or uint16 samples")
    p = params.normalized()
    if p.strength <= 1e-6 or image_rgb.size == 0:
        return image_rgb.copy()
    transmission = _transmission_map(image_rgb, p)
    height, width = image_rgb.shape[:2]
    scale = min(1.0, _ANALYSIS_EDGE / max(height, width))
    if scale < 1.0:
        sample = cv2.resize(
            image_rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        sample = image_rgb
    atmosphere = _scene_airlight(sample)
    if backend in ("native", "auto"):
        try:
            from native_renderer import native_spatial_dehaze
            return native_spatial_dehaze(image_rgb, p, transmission, atmosphere)
        except Exception:
            # Older bundles lack the optional C++ entry point. Keep preview
            # and export functional with the same reference map and airlight.
            pass
    output = np.empty_like(image_rgb)
    # All tiles use the same atmospheric light and full-image transmission.
    # Brightness protection is applied afterwards with one global gain.
    tile_params = replace(p, brightness_protection=0.0)
    for start in range(0, height, _TILE_ROWS):
        stop = min(height, start + _TILE_ROWS)
        output[start:stop] = _apply_dehaze_cpu(
            image_rgb[start:stop], tile_params, transmission[start:stop], atmosphere,
        )
    gain = _brightness_gain(image_rgb, output, p)
    if gain > 1.0 + 1e-6:
        for start in range(0, height, _TILE_ROWS):
            stop = min(height, start + _TILE_ROWS)
            output[start:stop] = _apply_global_brightness_gain(
                image_rgb[start:stop], output[start:stop], gain,
            )
    return output
