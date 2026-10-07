"""Confidence-limited atmospheric inversion in linear float RGB.

Decoding/camera white balance precede this stage. Display transfer, creative
exposure and DNG quantization follow it. Scene priors are evidence, not depth.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from dehaze import DehazeParams, _global_transmission, resolve_nonlocal_mode
from dehaze_spatial import _transmission_map, _box, _smoothstep, _guided_coefficients
from dehaze_nonlocal import (build_reliability_lut, lookup_transmission_relief,
                             build_nonlocal_transmission)

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


def _estimate_upper_airlight(rgb: np.ndarray, p: DehazeParams) -> tuple[np.ndarray, float]:
    """Estimate airlight from unsaturated, low-texture upper-frame candidates.

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
        unsaturated = upper[np.max(upper, axis=2) < .94]
        fallback = unsaturated if unsaturated.size else upper.reshape(-1, 3)
        air = np.median(fallback, axis=0)
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


def _estimate_broad_airlight(rgb: np.ndarray, p: DehazeParams
                             ) -> tuple[np.ndarray, float] | None:
    """Estimate airlight from broad dark-channel support across the full frame.

    Eroding the dark channel makes narrow bright details insufficient evidence;
    clipped light and its immediate neighborhood cannot enter the candidate pool.
    ``None`` means there are too few eligible pixels for a stable estimate.
    """
    if not rgb.size:
        return None
    kernel = np.ones((11, 11), dtype=np.uint8)
    dark = cv2.erode(np.min(rgb, axis=2), kernel)
    clipped = cv2.dilate((np.max(rgb, axis=2) >= .94).astype(np.uint8), kernel).astype(bool)
    eligible = ~clipped
    values = dark[eligible]
    if values.size < 8:
        return None

    threshold = float(np.percentile(values, 99))
    candidates = eligible & (dark >= threshold)
    smoothed = cv2.boxFilter(rgb, -1, (11, 11))
    pool = smoothed[candidates]
    if pool.size == 0:
        return None
    air = np.median(pool, axis=0)
    air_luma = float(air @ _LUMA)
    spread = float(np.median(np.linalg.norm(pool - air, axis=1))) / max(air_luma, 1e-6)
    agreement = 1.0 - float(_smoothstep(.12, .60, np.asarray(spread)))
    signal = float(_smoothstep(.001, .006, np.asarray(air_luma)))
    confidence = .15 + .85 * agreement * signal

    neutral_mix = p.color_protection * (.35 + .35 * (1.0 - confidence))
    air = air * (1.0 - neutral_mix) + air_luma * neutral_mix
    return np.clip(air, .00001, 1).astype(np.float32), float(confidence)


def _estimate_airlight(rgb: np.ndarray, p: DehazeParams) -> tuple[np.ndarray, float]:
    """Blend the upper-frame prior with full-frame broad dark-channel evidence."""
    upper_air, upper_confidence = _estimate_upper_airlight(rgb, p)
    upper_luma = float(upper_air @ _LUMA)
    broad_weight = 1.0 - float(_smoothstep(.025, .12, np.asarray(upper_luma)))
    if broad_weight <= 0:
        return upper_air, upper_confidence

    broad = _estimate_broad_airlight(rgb, p)
    if broad is None:
        return upper_air, upper_confidence
    broad_air, broad_confidence = broad
    air = upper_air * (1.0 - broad_weight) + broad_air * broad_weight
    confidence = (upper_confidence * (1.0 - broad_weight)
                  + broad_confidence * broad_weight)
    return np.clip(air, .00001, 1).astype(np.float32), float(confidence)


def _estimate_scene(source: np.ndarray, p: DehazeParams, spatial: bool,
                    nonlocal_mode: str | None = None,
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    nonlocal_mode = resolve_nonlocal_mode(nonlocal_mode)
    rgb = _analysis_sample(source)
    air, confidence = _estimate_airlight(rgb, p)
    reference, _ = _global_transmission(rgb, p)
    if spatial:
        # 天空分类只用于估计整幅场景的强度，不再把分类边界当作深度边界。
        # 相同 RGB 在太阳、暗角和屋顶附近使用相同变换，避免平滑遮罩造成光圈。
        # 明亮天空的保护值不能让天空占比大的照片整体停止去朦胧。
        requested = float(np.median(np.minimum(_transmission_map(rgb, p), reference)))
    else:
        requested = reference
    t = np.full(source.shape[:2], 1.0 - (1.0 - requested) * confidence, dtype=np.float32)
    optical_scale = _backlit_optical_scale(rgb, confidence, p) if spatial else 1.0
    if optical_scale < 1.0:
        # Weaken the requested inverse before the shared regularizer/operator.
        # One scene-wide factor cannot selectively lift a wall beside texture.
        t = np.exp(np.log(np.clip(t, 1e-4, 1.0)) * optical_scale)
    nonlocal_stats = {"nonlocal_active": False, "fallback_reason": ""}
    eligible = (spatial and p.strength > 1e-6 and confidence >= .85
                and float(air @ _LUMA) >= .12 and optical_scale == 1.0
                and float(np.mean(np.max(rgb, axis=2) >= .95)) < .002)
    if nonlocal_mode != "off":
        if p.strength <= 1e-6:
            nonlocal_stats["fallback_reason"] = "zero_strength"
        elif not spatial:
            nonlocal_stats["fallback_reason"] = "manual_mode"
        elif float(air @ _LUMA) < (.20 if nonlocal_mode == "strong" else .12):
            nonlocal_stats["fallback_reason"] = "low_airlight"
        elif confidence < .85:
            nonlocal_stats["fallback_reason"] = "uncertain_airlight"
        elif optical_scale != 1.0:
            nonlocal_stats["fallback_reason"] = "backlit_scene"
        elif float(np.mean(np.max(rgb, axis=2) >= .95)) >= .002:
            nonlocal_stats["fallback_reason"] = "clipped_highlights"
    # The stronger ray fit flattens low-airlight water/shading in the RAW
    # turbine trial. Keep those already-acceptable scenes on the original path.
    if (nonlocal_mode == "strong" and eligible
            and float(air @ _LUMA) >= .20):
        field, nonlocal_stats = build_nonlocal_transmission(
            source, air, float(t.flat[0]), p.strength)
        nonlocal_stats["nonlocal_active"] = bool(nonlocal_stats.get("nonlocal_field_active"))
        if nonlocal_stats["nonlocal_active"]:
            t = field
    # Reliability is allowed to relieve an over-darkened foreground, never to
    # increase the existing inverse. Keep uncertain airlight, night scenes and
    # clipped solar backlight on their established operator. No sky/roof masks.
    # Opt-in until distant-detail acceptance improves: the initial RAW trial
    # preserves colour/edges but can trade some distant contrast for relief.
    if nonlocal_mode == "conservative" and eligible:
        base = max(float(t.flat[0]),
                   1.0 / (1.0 + .8 * p.strength * (1.0 - .35 * p.naturalness)))
        lut, nonlocal_stats = build_reliability_lut(rgb, air, base)
        nonlocal_stats["nonlocal_active"] = bool(np.any(lut > 0))
        nonlocal_stats["fallback_reason"] = ("" if nonlocal_stats["nonlocal_active"] else
            "no_reliable_rays" if nonlocal_stats.get("solver_converged", True) else "solver_nonconverged")
        if nonlocal_stats["nonlocal_active"]:
            relief = lookup_transmission_relief(source, lut)
            # The RGB lookup retains equal treatment at every coordinate.
            # Reapplying a spatial guided filter would break that invariant.
            # Reject the ill-conditioned neighbourhood of I=A continuously.
            for start in range(0, source.shape[0], 256):
                stop = min(source.shape[0], start + 256)
                deficit = air - source[start:stop]
                radius = np.linalg.norm(deficit, axis=2) / max(float(np.linalg.norm(air)), 1e-6)
                below_air = np.min(deficit / np.maximum(air, 1e-6), axis=2)
                support = (_smoothstep(.10, .25, radius)
                           * _smoothstep(0.0, .05, below_air))
                knee = .015 * (1.0 - 1.0 / (
                    1.0 + .8 * p.strength * (1.0 - .35 * p.naturalness)))
                target = base - knee + (1.0 - base) * .5 * relief[start:stop] * support
                t[start:stop] = np.maximum(t[start:stop], target)
    stats = {"airlight": air.tolist(), "airlight_confidence": confidence,
             "transmission_min": float(np.min(t)), "transmission_median": float(np.median(t)),
             "backlit_optical_scale": optical_scale,
             "max_inverse_gain": 1.0 + .8 * p.strength * (1.0 - .35 * p.naturalness),
             "nonlocal_mode": nonlocal_mode,
             **nonlocal_stats}
    return np.ascontiguousarray(t, dtype=np.float32), air, stats


def _operator_params(p: DehazeParams, stats: dict) -> DehazeParams:
    """Stronger inversion only after a reliable experimental field succeeds.

    Keep the source chroma direction and established highlight protections.
    Failed fits and solar/night fallback use the original parameters exactly.
    """
    if not stats.get("nonlocal_field_active", False):
        return p
    return replace(p, naturalness=p.naturalness * .25,
                   brightness_protection=p.brightness_protection ** 3,
                   shadow_protection=p.shadow_protection ** 2,
                   color_protection=1.0, color_recovery=0.0)


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
    if float(np.min(transmission)) == float(np.max(transmission)):
        return np.ascontiguousarray(transmission, dtype=np.float32)
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


def _shadow_retention(p: DehazeParams) -> float:
    # 亮度保护限制明显减光，暗部保护提供额外余量；轻微去雾仍走精确反演。
    return .10 + .45 * p.brightness_protection + .10 * p.shadow_protection


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
    retention = _shadow_retention(p)
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
    return _source_colour_ceiling(source, result)


def _source_colour_ceiling(source: np.ndarray, result: np.ndarray) -> np.ndarray:
    """Bound added colour at fixed luminance, keeping the solar core intact."""
    y = _luminance(source)
    result_y = np.minimum(_luminance(result), y)
    base = source * (result_y / np.maximum(y, 1e-20))[..., None]
    deviation = result - base
    upper_room = np.maximum(source - base, 0) / np.maximum(deviation, 1e-20)
    lower_room = base / np.maximum(-deviation, 1e-20)
    room = np.where(deviation > 0, upper_room, lower_room)
    colour_scale = np.clip(np.min(room, axis=2), 0, 1)
    result = base + deviation * colour_scale[..., None]
    return np.clip(result, 0, 1).astype(np.float32)


def _dark_background_floor(source: np.ndarray, atmosphere: np.ndarray,
                           p: DehazeParams) -> float:
    """A dark-sky exposure reference, not an estimate of smoke radiance."""
    authority = 1.0 - float(_smoothstep(.065, .09, _luminance(atmosphere)))
    if authority <= 0 or p.brightness_protection <= 0:
        return 0.0
    sample = _analysis_sample(source)
    upper_y = _luminance(sample[:max(1, round(sample.shape[0] * .30))])
    background = float(np.percentile(upper_y, 90))
    return background * 1.25 * (p.brightness_protection / .70) * authority


def _protect_dark_background(source: np.ndarray, result: np.ndarray,
                             floor_level: float) -> np.ndarray:
    """Keep dark background exposure without lifting recovered smoke/lights.

    A monotonic source-value envelope and C1 handoff replace the image-wide
    gain. Above the envelope the complete inverse remains exactly unchanged.
    One image-wide reference is shared across row blocks, with no spatial mask.
    """
    if floor_level <= 1e-8:
        return result
    source_y = _luminance(source)
    result_y = _luminance(result)
    floor = floor_level * (-np.expm1(-source_y / floor_level))
    width = .02 * floor_level
    delta = result_y - floor
    target_y = np.maximum(result_y, floor)
    target_y += np.maximum(width - np.abs(delta), 0) ** 2 / (4.0 * width)
    lift = np.maximum(target_y - result_y, 0)
    # Add luminance on the original hue line, never a colour/atmosphere lift.
    candidate = result + source * (lift / np.maximum(source_y, 1e-20))[..., None]
    protected = _source_colour_ceiling(source, candidate)
    return np.where((lift > 0)[..., None], protected, result)


def physical_diagnostics(image: np.ndarray, params: DehazeParams, *, spatial: bool = False,
                         nonlocal_mode: str | None = None) -> dict:
    source = _linear_source(image)
    if not source.size:
        return {}
    p = params.normalized()
    transmission, atmosphere, stats = (_estimate_scene(source, p, spatial)
        if nonlocal_mode is None else _estimate_scene(source, p, spatial, nonlocal_mode))
    p = _operator_params(p, stats)
    stats["max_inverse_gain"] = 1.0 + .8 * p.strength * (1.0 - .35 * p.naturalness)
    if not stats.get("nonlocal_active", False):
        transmission = _regularize_transmission(source, transmission)
    effective = _inverse_transmission(source, transmission, p)
    stats.update(operator_transmission_min=float(np.min(effective)),
                 operator_transmission_median=float(np.median(effective)),
                 shadow_retention_floor=1.0 - (1.0 - _shadow_retention(p)) * p.strength,
                 shadow_toe_airlight_ratio=.12,
                 dark_background_floor=_dark_background_floor(source, atmosphere, p))
    return stats


def apply_physical_dehaze(image: np.ndarray, params: DehazeParams | None = None,
                          *, backend: str = "cpu", spatial: bool = False,
                          nonlocal_mode: str | None = None,
                          diagnostics: dict | None = None) -> np.ndarray:
    global _last_backend
    source = _linear_source(image)
    p = (params or DehazeParams()).normalized()
    if p.strength <= 1e-6 or not source.size:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(nonlocal_mode=resolve_nonlocal_mode(nonlocal_mode),
                               nonlocal_active=False,
                               fallback_reason="zero_strength" if p.strength <= 1e-6 else "empty_source")
        _last_backend = "未处理（强度为 0）"
        return image.copy()
    transmission, atmosphere, stats = (_estimate_scene(source, p, spatial)
        if nonlocal_mode is None else _estimate_scene(source, p, spatial, nonlocal_mode))
    p = _operator_params(p, stats)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(stats)
        diagnostics["max_inverse_gain"] = 1.0 + .8 * p.strength * (1.0 - .35 * p.naturalness)
    if not stats.get("nonlocal_active", False):
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
    # The shared dark-background guard leaves recovered smoke and lights
    # unchanged; it never applies an exposure gain to the complete image.
    floor_level = _dark_background_floor(source, atmosphere, p)
    if floor_level > 1e-8:
        for start in range(0, source.shape[0], 256):
            stop = min(source.shape[0], start + 256)
            result[start:stop] = _protect_dark_background(
                source[start:stop], result[start:stop], floor_level)
    if image.dtype == np.float32:
        return result
    peak = np.iinfo(image.dtype).max
    return np.clip(np.rint(result * peak), 0, peak).astype(image.dtype)
