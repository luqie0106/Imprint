"""Confidence-limited atmospheric inversion in linear float RGB.

Decoding/camera white balance precede this stage. Display transfer, creative
exposure and DNG quantization follow it. Scene priors are evidence, not depth.
"""
from __future__ import annotations

import cv2
import numpy as np

from dehaze import DehazeParams, _global_transmission
from dehaze_spatial import _transmission_map, _box, _smoothstep

_LUMA = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
_last_backend = "尚未处理"


def get_last_physical_backend() -> str:
    return _last_backend


def _linear_source(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if image.dtype not in (np.uint8, np.uint16, np.float32):
        raise TypeError("linear RGB must use uint8, uint16 or float32 samples")
    if image.dtype == np.float32:
        if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
            raise ValueError("linear float RGB must be finite and within [0, 1]")
        return image
    return image.astype(np.float32) / float(np.iinfo(image.dtype).max)


def _analysis_sample(source: np.ndarray) -> np.ndarray:
    height, width = source.shape[:2]
    scale = min(1.0, 768 / max(height, width))
    if scale < 1:
        return cv2.resize(source, (max(1, round(width * scale)),
                                  max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    return source


def _estimate_airlight(rgb: np.ndarray, p: DehazeParams) -> tuple[np.ndarray, float]:
    """Broad unsaturated upper candidates, with coverage and agreement checks.

    A bright object is not proof of fog. Saturated glare is excluded from the
    estimate; texture, colour/brightness disagreement and dark scenes reduce
    confidence. A low confidence estimate can only weaken the requested inverse.
    """
    upper = rgb[:max(1, round(rgb.shape[0] * .45))]
    y = upper @ _LUMA
    mean = _box(y, 5)
    variance = np.maximum(_box(y * y, 5) - mean * mean, 0)
    low, high = np.percentile(y, (70, 96))
    candidates = ((y >= low) & (y <= high)
                  & (variance <= np.percentile(variance, 65))
                  & (np.max(upper, axis=2) < .94))
    count = int(np.count_nonzero(candidates))
    if count < 8:
        air = np.median(upper.reshape(-1, 3), axis=0)
        confidence = .15
    else:
        pool = upper[candidates]
        air = np.median(pool, axis=0)
        air_y = float(air @ _LUMA)
        spread = float(np.median(np.linalg.norm(pool - air, axis=1))) / max(air_y, .025)
        agreement = 1.0 - float(_smoothstep(.12, .60, np.asarray(spread)))
        coverage = float(np.clip(count / max(1, upper.shape[0] * upper.shape[1]) / .12, 0, 1))
        # Darkness supplies little reliable haze evidence, even when windows
        # are bright. Retain some manual authority instead of declaring no fog.
        daylight = float(_smoothstep(.025, .12, np.asarray(air_y)))
        confidence = .15 + .85 * coverage * agreement * daylight
    air = np.clip(air, .005, 1).astype(np.float32)
    # Reduce a suspect chromatic airlight; this is not a second white balance.
    neutral = float(air @ _LUMA)
    neutral_mix = p.color_protection * (.35 + .35 * (1.0 - confidence))
    air = air * (1.0 - neutral_mix) + neutral * neutral_mix
    return air.astype(np.float32), float(confidence)


def _estimate_scene(source: np.ndarray, p: DehazeParams, spatial: bool
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    rgb = _analysis_sample(source)
    air, confidence = _estimate_airlight(rgb, p)
    if spatial:
        requested = _transmission_map(source, p)
        y = rgb @ _LUMA
        mean = _box(y, 7)
        variance = np.maximum(_box(y * y, 7) - mean * mean, 0)
        low_texture = 1.0 - _smoothstep(.08, .35, np.sqrt(variance) / (mean + .025))
        lifted_dark = _smoothstep(.08, .60, np.min(rgb, axis=2) / max(float(np.min(air)), .025))
        # Uniform white/snow or sky is ambiguous. These are modest weights,
        # never an instruction to infer large depth from brightness alone.
        local_confidence = .40 + .60 * low_texture * lifted_dark
        if local_confidence.shape != source.shape[:2]:
            local_confidence = cv2.resize(local_confidence, (source.shape[1], source.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
        t = 1.0 - (1.0 - requested) * confidence * local_confidence
    else:
        requested, _ = _global_transmission(rgb, p)
        t = np.full(source.shape[:2], 1.0 - (1.0 - requested) * confidence, dtype=np.float32)
    stats = {"airlight": air.tolist(), "airlight_confidence": confidence,
             "transmission_min": float(np.min(t)), "transmission_median": float(np.median(t)),
             "max_inverse_gain": 1.0 + .8 * p.strength * (1.0 - .35 * p.naturalness)}
    return np.ascontiguousarray(t, dtype=np.float32), air, stats


def _luminance(rgb: np.ndarray) -> np.ndarray:
    # Match the native non-contracted evaluation order. Near a saturated
    # channel, gamut headroom division can magnify a one-ULP luma difference.
    return ((rgb[..., 0] * np.float32(.2126) + rgb[..., 1] * np.float32(.7152))
            + rgb[..., 2] * np.float32(.0722))


def _gamut(rgb: np.ndarray) -> np.ndarray:
    y = _luminance(rgb)
    chroma = rgb - y[..., None]
    # One factor for the whole chroma vector preserves hue at the gamut edge.
    room = np.where(chroma > 0, (1.0 - y[..., None]) / np.maximum(chroma, 1e-7),
                    y[..., None] / np.maximum(-chroma, 1e-7))
    scale = np.clip(np.min(room, axis=2), 0, 1)
    return np.clip(y[..., None] + chroma * scale[..., None], 0, 1)


def _physical_pixels(source: np.ndarray, transmission: np.ndarray,
                     atmosphere: np.ndarray, p: DehazeParams) -> np.ndarray:
    """Float reference for the native operator; no image-level post compensation.

    Constrain t before inverse so negative RGB/shadow noise are not repaired
    after clipping. The maximum inverse gain is <=1.8. The highlight shoulder
    and source-luma ceiling preserve the approved solar glare behaviour.
    """
    y = _luminance(source)
    air = atmosphere.reshape(1, 1, 3)
    gain = 1.0 + .8 * p.strength * (1.0 - .35 * p.naturalness)
    shadow = 1.0 - _smoothstep(.02, .15, y)
    retention = np.minimum(.98, .65 + .25 * p.brightness_protection + .13 * p.shadow_protection * shadow)
    # J >= retention*I implies t >= (A-I)/(A-retention*I).
    needed = np.maximum(air - source, 0) / np.maximum(air - retention[..., None] * source, 1e-6)
    t = np.maximum(np.maximum(transmission, 1.0 / gain), np.max(needed, axis=2))
    highlight = _smoothstep(.55, .95, np.max(source, axis=2)) * p.highlight_protection
    t = t + (1.0 - t) * highlight
    t = np.clip(t, 1e-4, 1)[..., None]
    delta = source - air
    shoulder = t + (1.0 - t) * np.maximum(delta, 0) / np.maximum(1.0 - air, 1e-4)
    recovered = air + delta / shoulder
    recovered_y = _luminance(recovered)
    source_hue = source * (recovered_y / np.maximum(y, 1e-6))[..., None]
    chroma_confidence = _smoothstep(.005, .08, np.linalg.norm(source - y[..., None], axis=2))
    physical_colour = ((1.0 - p.color_protection) * (.28 + .72 * (1.0 - p.naturalness))
                       * chroma_confidence)
    result = recovered * physical_colour[..., None] + source_hue * (1.0 - physical_colour[..., None])

    # Small optional colour/tone changes follow the stable inverse. No local
    # sharpening or repeated mixback; both sliders vanish continuously at zero.
    result_y = _luminance(result)
    chroma = result - result_y[..., None]
    chroma *= (1.0 + .25 * p.color_recovery * p.strength * chroma_confidence)[..., None]
    contrast_delta = .15 * p.local_contrast * p.strength * result_y * (1.0 - result_y) * (2.0 * result_y - 1.0)
    contrast_delta = np.clip(contrast_delta, -.02 * result_y, .02 * (1.0 - result_y))
    tone_y = np.clip(result_y + contrast_delta, 0, 1)
    result = tone_y[..., None] + chroma
    result = _gamut(result)
    # Scalar (not per-channel) solar ceiling leaves no new luminous ring and
    # never changes channel ratios. It is independent of A or sky thresholds.
    result_y = _luminance(result)
    result *= np.minimum(1.0, y / np.maximum(result_y, 1e-6))[..., None]
    return np.clip(result, 0, 1).astype(np.float32)


def physical_diagnostics(image: np.ndarray, params: DehazeParams, *, spatial: bool = False) -> dict:
    source = _linear_source(image)
    if not source.size:
        return {}
    return _estimate_scene(source, params.normalized(), spatial)[2]


def apply_physical_dehaze(image: np.ndarray, params: DehazeParams | None = None,
                          *, backend: str = "cpu", spatial: bool = False) -> np.ndarray:
    global _last_backend
    source = _linear_source(image)
    p = (params or DehazeParams()).normalized()
    if p.strength <= 1e-6 or not source.size:
        _last_backend = "未处理（强度为 0）"
        return image.copy()
    transmission, atmosphere, _ = _estimate_scene(source, p, spatial)
    result = None
    if backend in ("native", "auto", "legacy_gpu", "pytorch"):
        try:
            from native_renderer import native_physical_dehaze, get_last_native_physical_backend
            result = native_physical_dehaze(source, p, transmission, atmosphere)
            _last_backend = f"{get_last_native_physical_backend()} GPU（线性浮点）"
        except Exception:
            # Never fall back to the old integer operator: same mathematics on
            # CPU also handles older installations without the new optional ABI.
            pass
    if result is None:
        result = np.empty_like(source)
        for start in range(0, source.shape[0], 256):
            stop = min(source.shape[0], start + 256)
            result[start:stop] = _physical_pixels(source[start:stop], transmission[start:stop], atmosphere, p)
        _last_backend = "Python CPU（线性浮点）"
    if image.dtype == np.float32:
        return result
    peak = np.iinfo(image.dtype).max
    return np.clip(np.rint(result * peak), 0, peak).astype(image.dtype)
