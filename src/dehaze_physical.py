"""Confidence-limited atmospheric inversion in linear float RGB.

Decoding/camera white balance precede this stage. Display transfer, creative
exposure and DNG quantization follow it. Scene priors are evidence, not depth.
"""
from __future__ import annotations

import cv2
import numpy as np

from dehaze import DehazeParams, _global_transmission
from dehaze_spatial import _transmission_map, _box, _smoothstep, _guided_coefficients

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
        # The spatial estimator already guards uncertain sky and silhouettes.
        # Gating it again by local texture/darkness raises t around buildings
        # and lamps; later smoothing spreads that residual veil into the sky
        # as a luminous border. Keep the image-wide airlight confidence here;
        # black-level protection belongs to the bounded pixel operator.
        t = 1.0 - (1.0 - requested) * confidence
    else:
        requested, _ = _global_transmission(rgb, p)
        t = np.full(source.shape[:2], 1.0 - (1.0 - requested) * confidence, dtype=np.float32)
    optical_scale = _backlit_optical_scale(rgb, confidence, p) if spatial else 1.0
    if optical_scale < 1.0:
        # Weaken the requested inverse before the shared regularizer/operator.
        # One scene-wide factor cannot selectively lift a wall beside texture.
        t = np.exp(np.log(np.clip(t, 1e-4, 1.0)) * optical_scale)
    stats = {"airlight": air.tolist(), "airlight_confidence": confidence,
             "transmission_min": float(np.min(t)), "transmission_median": float(np.median(t)),
             "backlit_optical_scale": optical_scale,
             "max_inverse_gain": 1.0 + .8 * p.strength * (1.0 - .35 * p.naturalness)}
    return np.ascontiguousarray(t, dtype=np.float32), air, stats


def _backlit_optical_scale(rgb: np.ndarray, airlight_confidence: float,
                           p: DehazeParams) -> float:
    """Conservative scene authority for uncertain, extreme solar backlight.

    Limited clipped upper coverage, a bright upper field and very dark lower
    field must agree. A reliable airlight retains the existing inverse. These
    are global statistics, never a texture/skyline mask or regional lift.
    """
    if p.strength <= 1e-6 or p.brightness_protection <= 0 or not rgb.size:
        return 1.0
    uncertain = 1.0 - float(_smoothstep(.65, .85, np.asarray(airlight_confidence)))
    if uncertain <= 0:
        return 1.0
    y = _luminance(rgb)
    upper_rows = max(1, round(rgb.shape[0] * .45))
    upper_y = float(np.median(y[:upper_rows]))
    lower_y = max(float(np.median(y[rgb.shape[0] // 2:])), .001)
    clipped = float(np.mean(np.max(rgb[:upper_rows], axis=2) > .95))
    sun = float(_smoothstep(.005, .015, np.asarray(clipped)))
    sun *= 1.0 - float(_smoothstep(.15, .35, np.asarray(clipped)))
    backlight = (sun * float(_smoothstep(8, 14, np.asarray(upper_y / lower_y)))
                 * float(_smoothstep(.05, .10, np.asarray(upper_y))))
    return 1.0 - .85 * p.brightness_protection * uncertain * backlight


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


def _regularize_transmission(source: np.ndarray, transmission: np.ndarray) -> np.ndarray:
    """Smooth the complete confidence-weighted optical thickness before inverse.

    The scene estimator is unchanged. Filter log(t) after its confidence/sky
    decisions, rather than smoothing a cue that later masks can divide again.
    Coefficients share one image-wide analysis grid; no per-tile statistics or
    regional exposure compensation can make a seam at a processing boundary.
    """
    depth = -np.log(np.clip(transmission, 1e-4, 1.0))
    sample = _analysis_sample(source)
    size = (sample.shape[1], sample.shape[0])
    reduced = cv2.resize(depth, size, interpolation=cv2.INTER_AREA)
    slope, intercept = _guided_coefficients(_luminance(sample), reduced, radius=8)
    if reduced.shape != depth.shape:
        size = (source.shape[1], source.shape[0])
        slope = cv2.resize(slope, size, interpolation=cv2.INTER_LINEAR)
        intercept = cv2.resize(intercept, size, interpolation=cv2.INTER_LINEAR)
    refined = slope * _luminance(source) + intercept
    # A constant field (manual mode included) remains constant. Projection only
    # keeps small guided-filter overshoots inside the original global range.
    refined = np.clip(refined, float(np.min(depth)), float(np.max(depth)))
    return np.ascontiguousarray(np.exp(-refined), dtype=np.float32)


def _inverse_transmission(source: np.ndarray, transmission: np.ndarray,
                          p: DehazeParams) -> np.ndarray:
    gain = 1.0 + .8 * p.strength * (1.0 - .35 * p.naturalness)
    floor = 1.0 / gain
    # C1 majorant of max(t, floor). A strength-scaled knee keeps zero an exact
    # identity, without a new fixed-width transition at tiny slider values.
    width = .015 * (1.0 - floor)
    t = np.maximum(transmission, floor)
    if width > 0:
        t = t + np.maximum(width - np.abs(transmission - floor), 0) ** 2 / (4.0 * width)
    highlight = _smoothstep(.55, .95, np.max(source, axis=2)) * p.highlight_protection
    return np.clip(t + (1.0 - t) * highlight, 1e-4, 1.0)


def _physical_pixels(source: np.ndarray, transmission: np.ndarray,
                     atmosphere: np.ndarray, p: DehazeParams) -> np.ndarray:
    """Float reference for the native operator; no image-level post compensation.

    Gain stays <=1.8. A monotonic C1 toe bounds negative-side atmospheric
    subtraction without treating fog-lifted midtones as black-level shadows.
    Positive-side highlight shoulder and solar luma ceiling are retained.
    """
    y = _luminance(source)
    air = atmosphere.reshape(1, 1, 3)
    t = _inverse_transmission(source, transmission, p)[..., None]
    delta = source - air
    shoulder = t + (1.0 - t) * np.maximum(delta, 0) / np.maximum(1.0 - air, 1e-4)
    positive = air + delta / shoulder
    deficit = np.maximum(-delta, 0)
    # Approach zero subtraction with zero derivative at I=A, where the solar
    # ceiling hands off to the original. This avoids a contrast kink there.
    smooth_deficit = deficit * deficit / (deficit + .025 * air + 1e-8)
    loss = (1.0 / t - 1.0) * smooth_deficit
    retention = .10 + .08 * p.brightness_protection + .035 * p.shadow_protection
    # A fixed retained fraction makes fog-lifted structures hit the shadow
    # limit before their surrounding veil, flattening their contrast. Instead
    # the subtraction budget approaches O(I^2/A) at black and O(I) above the
    # noise toe. This protects real shadows without capping hazy midtones at
    # the same retained brightness. Use the same source/air ratio in linear
    # RGB at every pixel; no regional masks or exposure compensation.
    budget = ((1.0 - retention) * source * source
              / (source + .12 * air + 1e-8) * p.strength)
    fraction = loss / np.maximum(budget, 1e-8)
    # Exact subtraction in the first half of the budget, then an exponential
    # shoulder with matching value/slope. Output stays >= source-budget;
    # finite differences for fixed t remain nonnegative and gain-bounded.
    bounded = np.where(fraction <= .5, fraction,
                       1.0 - .5 * np.exp(-2.0 * np.maximum(fraction - .5, 0)))
    negative = source - budget * bounded
    recovered = np.where(delta < 0, negative, positive)
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
    p = params.normalized()
    transmission, _, stats = _estimate_scene(source, p, spatial)
    transmission = _regularize_transmission(source, transmission)
    effective = _inverse_transmission(source, transmission, p)
    stats.update(operator_transmission_min=float(np.min(effective)),
                 operator_transmission_median=float(np.median(effective)),
                 shadow_retention_floor=1.0 - (.90 - .08 * p.brightness_protection
                                               - .035 * p.shadow_protection) * p.strength,
                 shadow_toe_airlight_ratio=.12)
    return stats


def apply_physical_dehaze(image: np.ndarray, params: DehazeParams | None = None,
                          *, backend: str = "cpu", spatial: bool = False) -> np.ndarray:
    global _last_backend
    source = _linear_source(image)
    p = (params or DehazeParams()).normalized()
    if p.strength <= 1e-6 or not source.size:
        _last_backend = "未处理（强度为 0）"
        return image.copy()
    transmission, atmosphere, _ = _estimate_scene(source, p, spatial)
    transmission = _regularize_transmission(source, transmission)
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
