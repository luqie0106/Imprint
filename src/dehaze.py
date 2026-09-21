"""Spatially consistent dehazing for preview and full-resolution RGB images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


ALGORITHM_VERSION = "natural-global-v5-highlight-rolloff"


@dataclass(frozen=True)
class DehazeParams:
    strength: float = 0.45
    naturalness: float = 0.70
    fog_retention: float = 0.55
    local_contrast: float = 0.25
    color_protection: float = 0.80
    highlight_protection: float = 0.75
    shadow_protection: float = 0.75

    def normalized(self) -> "DehazeParams":
        values = {key: float(np.clip(value, 0.0, 1.0)) for key, value in asdict(self).items()}
        return DehazeParams(**values)

    def cache_token(self) -> str:
        payload = {"version": ALGORITHM_VERSION, **asdict(self.normalized())}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _atmospheric_light(image: np.ndarray) -> np.ndarray:
    """Estimate one RGB atmospheric-light vector for the complete image.

    The dark channel is used only to choose global candidates for ``A``.  It
    is deliberately not used to construct a per-pixel transmission map: the
    same input RGB value therefore receives the same dehaze transform at
    every image coordinate.
    """
    flat = image.reshape(-1, 3)
    dark = np.min(flat, axis=1)
    # Keep a small, stable candidate pool on previews while remaining valid
    # for tiny synthetic images used by tests.
    count = min(dark.size, max(16, int(dark.size * 0.001)))
    if count <= 0:
        return np.ones(3, dtype=np.float32)
    indices = np.argpartition(dark, -count)[-count:]
    candidates = flat[indices]
    luminance = candidates @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    brightest = candidates[np.argsort(luminance)[-max(1, count // 8):]]
    return np.clip(np.median(brightest, axis=0), 0.35, 1.0).astype(np.float32)


def apply_dehaze(image_rgb: np.ndarray, params: DehazeParams | None = None) -> np.ndarray:
    """Return a dehazed RGB image while preserving shape, dtype, and the input array."""
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a numpy array")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if image_rgb.dtype not in (np.uint8, np.uint16):
        raise TypeError("image_rgb must use uint8 or uint16 samples")

    p = (params or DehazeParams()).normalized()
    if p.strength <= 1e-6 or image_rgb.size == 0:
        return image_rgb.copy()

    peak = float(np.iinfo(image_rgb.dtype).max)
    source = image_rgb.astype(np.float32) / peak
    atmosphere = _atmospheric_light(source)
    dark_reference = float(np.percentile(np.min(source, axis=2), 75))
    if not np.isfinite(dark_reference):
        dark_reference = 0.5
    # The image statistic is deliberately only a small correction.  In a
    # backlit city, a large dark building area must not turn a strong setting
    # into an almost-identity transform.
    haze_level = float(np.clip((dark_reference - 0.03) / 0.92, 0.12, 0.92))

    # All pixels use this one scalar transmission.  Strength is the primary,
    # monotonic control; fog retention and naturalness soften it.  Image
    # statistics only make a small correction inside ``omega``: multiplying
    # by ``haze_level`` again here would make a dark, backlit city nearly an
    # identity transform even when the strength slider is close to maximum.
    omega = (
        p.strength
        * (0.66 - 0.14 * p.fog_retention)
        * (0.90 + 0.10 * (1.0 - p.naturalness))
        * (0.92 + 0.08 * haze_level)
    )
    min_transmission = 0.27 + 0.21 * p.fog_retention + 0.11 * p.naturalness
    transmission = float(np.clip(1.0 - omega, min_transmission, 1.0))

    neutral_atmosphere = float(np.mean(atmosphere))
    neutral_mix = 0.45 * p.color_protection * (0.65 + 0.35 * p.naturalness)
    atmosphere = atmosphere * (1.0 - neutral_mix) + neutral_atmosphere * neutral_mix
    atmosphere = np.clip(atmosphere, 0.35, 1.0)
    recovered = (source - atmosphere.reshape(1, 1, 3)) / transmission + atmosphere.reshape(1, 1, 3)
    recovered = np.clip(recovered, 0.0, 1.0)

    luminance = source @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    global_blend = p.strength * (0.78 - 0.28 * p.fog_retention) * (0.90 + 0.10 * (1.0 - p.naturalness))
    blend = np.clip(global_blend, 0.0, 0.82)
    natural = source * (1.0 - blend[..., None]) + recovered * blend[..., None]

    # Protect hue/chroma by mixing the enhanced luminance with source chroma.
    enhanced_luma = natural @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    source_luma = np.maximum(luminance, 1e-4)
    luma_only = np.clip(source * (enhanced_luma / source_luma)[..., None], 0.0, 1.0)
    chroma_mix = p.color_protection * (0.72 + 0.28 * p.naturalness)
    natural = natural * (1.0 - chroma_mix) + luma_only * chroma_mix

    if p.local_contrast > 1e-6:
        luma = natural @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        # A fixed-endpoint, global S-curve provides a smooth highlight
        # shoulder.  It uses no neighbourhood statistics and remains
        # monotonic, so equal RGB values map identically at every coordinate.
        contrast_amount = 0.55 * p.local_contrast
        contrast_luma = luma + contrast_amount * (2.0 * luma - 1.0) * luma * (1.0 - luma)
        natural *= (contrast_luma / np.maximum(luma, 1e-4))[..., None]

    # Apply highlight and shadow protection after every recovery, chroma and
    # tone operation.  Masks are smooth functions of the original luminance,
    # not local/edge statistics, so they cannot create spatial seams.
    highlight_position = np.clip((luminance - 0.58) / 0.40, 0.0, 1.0)
    highlight_position = highlight_position * highlight_position * (3.0 - 2.0 * highlight_position)
    highlight_blend = np.clip(highlight_position * p.highlight_protection, 0.0, 1.0)
    natural = natural * (1.0 - highlight_blend[..., None]) + source * highlight_blend[..., None]

    shadow_position = np.clip((0.26 - luminance) / 0.26, 0.0, 1.0)
    shadow_position = shadow_position * shadow_position * (3.0 - 2.0 * shadow_position)
    shadow_blend = np.clip(shadow_position * p.shadow_protection, 0.0, 1.0)
    natural = natural * (1.0 - shadow_blend[..., None]) + source * shadow_blend[..., None]

    # Do not let tone recovery create a new near-saturated solar halo.  The
    # cap is below the comparison threshold by a small dtype-aware margin;
    # pixels already near saturation remain untouched, while neighbouring
    # pixels cannot cross the threshold through rounding.  This is a global
    # per-pixel value guard, not an edge or neighbourhood operation.
    saturation_threshold = 0.97
    saturation_limit = max(0.0, saturation_threshold - 1.5 / peak)
    final_luma = natural @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma_cap = np.where(luminance < saturation_threshold, saturation_limit, 1.0)
    luma_scale = np.minimum(1.0, luma_cap / np.maximum(final_luma, 1e-4))
    natural *= luma_scale[..., None]
    channel_cap = np.where(source < saturation_threshold, saturation_limit, 1.0)
    natural = np.minimum(natural, channel_cap)

    result = np.nan_to_num(natural, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.rint(result * peak), 0, peak).astype(image_rgb.dtype)
