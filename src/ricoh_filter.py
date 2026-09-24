"""Ricoh Camera Raw preset catalog and XMP sidecar export helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from image_io import OUTPUT_DIR_NAME, SUPPORTED_SUFFIXES


_CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_IMPRINT_NS = "https://imprint.local/ns/photo-settings/1.0/"
_MAX_PHOTOS = 500
_WRITE_LOCK = threading.RLock()

# Empirical residual luminance calibration for the GR3 Vivid Street measured
# preview. It comes from aligned captures of one photo using Adobe's Standard
# profile, the Vivid Street XMP preset, and Exposure +0.85. This sample-based
# correction improves the preview match; it cannot establish full ACR parity.
_GR3_VIVID_STREET_LUMA_X = tuple(
    value / 255.0 for value in (0, 8, 24, 40, 56, 72, 88, 104, 120, 144, 176, 208, 240, 255)
)
_GR3_VIVID_STREET_LUMA_Y = tuple(
    value / 255.0 for value in (0, 6, 16, 30, 49, 71, 93, 111, 129, 157, 192, 211, 234, 255)
)


@dataclass(frozen=True)
class RicohPreset:
    id: str
    model: str
    name: str
    description: str
    resource: str


# Keep this list explicit: only reviewed GR2/GR3 CameraRaw presets are made
# available to the API, regardless of what other files happen to be packaged.
_PRESETS = (
    RicohPreset(
        "gr2_positive_film", "GR2", "GR2 正片 Positive Film",
        "深绿、青色阴影、暖色高光与克制的饱和度。",
        "Ricoh_GR2_CameraRaw_Presets/Presets/GR2_Positive_Film.xmp",
    ),
    RicohPreset(
        "gr2_hi_bw", "GR2", "GR2 高对比黑白 Hi-BW",
        "浓郁黑位、清晰高光、强对比与可见颗粒。",
        "Ricoh_GR2_CameraRaw_Presets/Presets/GR2_Hi_BW.xmp",
    ),
    RicohPreset(
        "gr2_negative_film", "GR2", "GR2 负片 Negative Film",
        "抬高黑位、柔和反差、低饱和色彩与暖色中间调。",
        "Ricoh_GR2_CameraRaw_Presets/Presets/GR2_Negative_Film.xmp",
    ),
    RicohPreset(
        "gr2_street_positive", "GR2", "GR2 街头正片 Street Positive",
        "更强反差、清晰细节、克制色彩与微冷阴影。",
        "Ricoh_GR2_CameraRaw_Presets/Presets/GR2_Street_Positive.xmp",
    ),
    RicohPreset(
        "gr3_positive_film", "GR3", "GR3 正片 Positive Film",
        "深蓝、克制的绿色、青色阴影与暖色高光。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_Positive_Film.xmp",
    ),
    RicohPreset(
        "gr3_negative_film", "GR3", "GR3 负片 Negative Film",
        "抬高黑位、柔和反差、低饱和色彩与柔润高光。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_Negative_Film.xmp",
    ),
    RicohPreset(
        "gr3_vivid_street", "GR3", "GR3 鲜明街头 Vivid Street",
        "强反差、鲜明蓝色、清晰细节与受控肤色。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_Vivid_Street.xmp",
    ),
    RicohPreset(
        "gr3_standard", "GR3", "GR3 标准 Standard",
        "均衡反差、克制饱和度、冷色阴影与微暖高光。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_Standard.xmp",
    ),
    RicohPreset(
        "gr3_high_contrast_bw", "GR3", "GR3 高反差黑白 High Contrast B&W",
        "浓郁黑位、明亮白色与可见颗粒。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_High_Contrast_BW.xmp",
    ),
    RicohPreset(
        "gr3_bleach_bypass", "GR3", "GR3 漂白负冲 Bleach Bypass",
        "高反差、低饱和度与金属灰色调。",
        "Ricoh_GR3_CameraRaw_Presets/Presets/GR3_Bleach_Bypass.xmp",
    ),
)
_PRESETS_BY_ID = {preset.id: preset for preset in _PRESETS}

# Preset identity and UI metadata must not be copied into a photo's sidecar.
# Camera Raw processing settings (including curve RDF structures) remain.
_PRESET_IDENTITY_FIELDS = {
    "presettype", "presetid", "presetname", "presetgroup", "presetgroupid",
    "uuid", "id", "cluster", "name", "shortname", "group", "groupname",
    "description", "sortname", "copyright", "contactinfo", "isfavorite",
    "isdefault", "isuserpreset",
}
_PRESET_PROCESSING_KEYS: tuple[set[str], set[str]] | None = None


class RicohBatchLimitError(ValueError):
    """Raised when a request resolves to more photos than the API permits."""


def _resource_root() -> Path:
    """Find bundled XMP files in development and in a PyInstaller package."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    app_root = Path(frozen_root) if frozen_root else Path(__file__).resolve().parents[1]
    return app_root / "assets" / "ricoh"


def list_ricoh_presets() -> list[dict[str, str]]:
    """Return the stable, public catalog without exposing resource paths."""
    return [
        {
            "id": preset.id,
            "model": preset.model,
            "name": preset.name,
            "description": preset.description,
        }
        for preset in _PRESETS
    ]


_HSL_COLORS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
_HSL_HUE_CENTERS = (0.0, 30.0, 60.0, 120.0, 180.0, 210.0, 275.0, 315.0)


def _read_tone_curve(description: ET.Element, name: str) -> tuple[tuple[float, float], ...]:
    """Read an XMP curve's RDF sequence, falling back to a straight line."""
    identity = ((0.0, 0.0), (255.0, 255.0))
    element = next((item for item in description.iter("{" + _CRS_NS + "}" + name)), None)
    if element is None:
        return identity
    points: list[tuple[float, float]] = []
    for item in element.iter("{" + _RDF_NS + "}li"):
        value = (item.text or "").strip()
        pair = value.split(",") if "," in value else value.split()
        if len(pair) != 2:
            continue
        try:
            x, y = float(pair[0]), float(pair[1])
        except ValueError:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        points.append((min(255.0, max(0.0, x)), min(255.0, max(0.0, y))))
    if len(points) < 2:
        return identity
    points.sort(key=lambda point: point[0])
    # np.interp expects strictly increasing x positions. If an XMP contains
    # duplicate positions, keep the last point at that coordinate.
    deduplicated: dict[float, float] = {}
    for x, y in points:
        deduplicated[x] = y
    if len(deduplicated) < 2:
        return identity
    return tuple(sorted(deduplicated.items()))


def _preset_controls(preset_id: str) -> dict[str, object]:
    if preset_id not in _PRESETS_BY_ID:
        raise KeyError(preset_id)
    root = _parse_xmp(_preset_payload(preset_id))
    description = _description(root)
    if description is None:
        raise RuntimeError("preset resource unavailable")

    def number(name: str) -> float:
        try:
            value = float(_find_simple(description, _CRS_NS, name) or 0.0)
            return value if math.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    grayscale = (_find_simple(description, _CRS_NS, "ConvertToGrayscale") or "False").casefold() == "true"
    controls: dict[str, object] = {
        "exposure": number("Exposure2012"),
        "contrast": number("Contrast2012"),
        "saturation": number("Saturation"),
        "vibrance": number("Vibrance"),
        "temperature": number("Temperature"),
        "tint": number("Tint"),
        "shadows": number("Shadows2012"),
        "highlights": number("Highlights2012"),
        "whites": number("Whites2012"),
        "blacks": number("Blacks2012"),
        "grayscale": grayscale,
        "split_shadow_hue": number("SplitToningShadowHue"),
        "split_shadow_saturation": number("SplitToningShadowSaturation"),
        "split_highlight_hue": number("SplitToningHighlightHue"),
        "split_highlight_saturation": number("SplitToningHighlightSaturation"),
        "split_balance": number("SplitToningBalance"),
        "grade_shadow_hue": number("ColorGradeShadowHue"),
        "grade_shadow_saturation": number("ColorGradeShadowSat"),
        "grade_midtone_hue": number("ColorGradeMidtoneHue"),
        "grade_midtone_saturation": number("ColorGradeMidtoneSat"),
        "grade_highlight_hue": number("ColorGradeHighlightHue"),
        "grade_highlight_saturation": number("ColorGradeHighlightSat"),
        "grade_global_hue": number("ColorGradeGlobalHue"),
        "grade_global_saturation": number("ColorGradeGlobalSat"),
        "grade_shadow_luminance": number("ColorGradeShadowLum"),
        "grade_midtone_luminance": number("ColorGradeMidtoneLum"),
        "grade_highlight_luminance": number("ColorGradeHighlightLum"),
        "grade_global_luminance": number("ColorGradeGlobalLum"),
        "grade_blending": number("ColorGradeBlending"),
        "hue_adjustments": {color: number("HueAdjustment" + color) for color in _HSL_COLORS},
        "saturation_adjustments": {color: number("SaturationAdjustment" + color) for color in _HSL_COLORS},
        "luminance_adjustments": {color: number("LuminanceAdjustment" + color) for color in _HSL_COLORS},
        "tone_curve": _read_tone_curve(description, "ToneCurvePV2012"),
        "red_curve": _read_tone_curve(description, "ToneCurvePV2012Red"),
        "green_curve": _read_tone_curve(description, "ToneCurvePV2012Green"),
        "blue_curve": _read_tone_curve(description, "ToneCurvePV2012Blue"),
    }
    return controls


def _apply_hsl_adjustments(rgb, controls: dict[str, object], np):
    """Approximate Adobe's eight color ranges using overlapping HSV weights."""
    hue_adjustments = controls["hue_adjustments"]
    saturation_adjustments = controls["saturation_adjustments"]
    luminance_adjustments = controls["luminance_adjustments"]
    values = [
        _finite_control_value(mapping[color])
        for mapping in (hue_adjustments, saturation_adjustments, luminance_adjustments)
        for color in _HSL_COLORS
    ]
    if not any(math.isfinite(value) and value != 0.0 for value in values):
        return rgb

    import cv2

    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.float32), cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    weights = []
    for center in _HSL_HUE_CENTERS:
        distance = np.abs((hue - center + 180.0) % 360.0 - 180.0)
        weights.append(np.exp(-0.5 * (distance / 18.0) ** 2))
    total_weight = np.maximum(np.sum(weights, axis=0), 1e-7)
    weighted_hue = np.zeros_like(hue)
    weighted_saturation = np.zeros_like(hue)
    weighted_luminance = np.zeros_like(hue)
    for idx, color in enumerate(_HSL_COLORS):
        weight = weights[idx] / total_weight
        weighted_hue += weight * _finite_control_value(hue_adjustments[color])
        weighted_saturation += weight * _finite_control_value(saturation_adjustments[color])
        weighted_luminance += weight * _finite_control_value(luminance_adjustments[color])

    original_saturation = saturation
    hsv[..., 0] = (hue + weighted_hue) % 360.0
    hsv[..., 1] = np.clip(saturation * (1.0 + 1.75 * weighted_saturation / 100.0), 0.0, 1.0)
    hsv[..., 2] = np.clip(
        value + 0.012 * weighted_luminance / 100.0
        * np.clip(original_saturation * 4.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _finite_control_value(value: object) -> float:
    """Return a finite scalar for a parsed or externally supplied XMP control."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _smoothstep(edge0, edge1, value, np):
    amount = np.clip((value - edge0) / np.maximum(edge1 - edge0, 1e-6), 0.0, 1.0)
    return amount * amount * (3.0 - 2.0 * amount)


def _grading_masks(lab_luminance, controls: dict[str, object], np):
    """Build continuous Lightroom-style tone masks from normalized Lab L*."""
    balance = np.clip(_finite_control_value(controls["split_balance"]), -100.0, 100.0)
    blending = np.clip(_finite_control_value(controls["grade_blending"]), 0.0, 100.0) / 100.0

    # At the common blending value of 50, shadow and highlight transitions
    # span about [.25, .75] and [.55, .90]. Blending widens or narrows them;
    # positive split balance moves both centers darker to favor highlights.
    shift = 0.2 * balance / 100.0
    shadow_width = 0.2 + 0.6 * blending
    shadow_center = 0.5 - shift
    shadow = 1.0 - _smoothstep(
        shadow_center - shadow_width / 2.0,
        shadow_center + shadow_width / 2.0,
        lab_luminance,
        np,
    )
    highlight_width = 0.15 + 0.4 * blending
    highlight_center = 0.725 - shift
    highlight = _smoothstep(
        highlight_center - highlight_width / 2.0,
        highlight_center + highlight_width / 2.0,
        lab_luminance,
        np,
    )
    midtone = (1.0 - shadow) * (1.0 - highlight)
    return shadow, midtone, highlight


def _lab_to_gamut_safe_rgb(lab, cv2, np):
    """Reduce out-of-gamut Lab chroma while preserving the requested L*."""
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    out_of_gamut = np.any((rgb < 0.0) | (rgb > 1.0), axis=2)
    if not np.any(out_of_gamut):
        return rgb

    selected_lab = lab[out_of_gamut]
    target_chroma = selected_lab[:, 1:3].copy()
    low = np.zeros(len(selected_lab), dtype=np.float32)
    high = np.ones(len(selected_lab), dtype=np.float32)
    # Find the largest in-gamut fraction of the requested opponent-color
    # vector. Only pixels that need compression take part in these conversions.
    for _ in range(8):
        scale = (low + high) * 0.5
        candidate_lab = selected_lab.copy()
        candidate_lab[:, 1:3] = target_chroma * scale[:, None]
        candidate_rgb = cv2.cvtColor(candidate_lab.reshape(-1, 1, 3), cv2.COLOR_Lab2RGB).reshape(-1, 3)
        valid = np.all((candidate_rgb >= 0.0) & (candidate_rgb <= 1.0), axis=1)
        low = np.where(valid, scale, low)
        high = np.where(valid, high, scale)

    selected_lab[:, 1:3] = target_chroma * low[:, None] * 0.995
    lab[out_of_gamut] = selected_lab
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


def _apply_color_grading(rgb, controls: dict[str, object], np):
    """Approximate color wheels as one soft, additive Lab opponent-color shift.

    The chroma amplitudes below are calibrated preview approximations per 100
    saturation units, not Adobe Camera Raw's proprietary grading transform.
    """
    groups = (
        ("split_shadow_hue", "split_shadow_saturation", "shadow", 14.0),
        ("split_highlight_hue", "split_highlight_saturation", "highlight", 10.0),
        ("grade_shadow_hue", "grade_shadow_saturation", "shadow", 14.0),
        ("grade_midtone_hue", "grade_midtone_saturation", "midtone", 6.0),
        ("grade_highlight_hue", "grade_highlight_saturation", "highlight", 10.0),
        ("grade_global_hue", "grade_global_saturation", "global", 14.0),
    )
    active = any(
        abs(_finite_control_value(controls[saturation_key])) > 1e-8
        for _, saturation_key, _, _ in groups
    )
    if not active:
        return rgb

    import cv2

    safe_rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    lab = cv2.cvtColor(np.asarray(safe_rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)
    normalized_luminance = lab[..., 0] / 100.0
    shadow, midtone, highlight = _grading_masks(normalized_luminance, controls, np)
    masks = {"shadow": shadow, "midtone": midtone, "highlight": highlight,
             "global": np.ones_like(normalized_luminance)}
    offset_a = np.zeros_like(normalized_luminance)
    offset_b = np.zeros_like(normalized_luminance)
    for hue_key, saturation_key, mask_key, amplitude in groups:
        saturation = np.clip(_finite_control_value(controls[saturation_key]), -100.0, 100.0)
        hue = _finite_control_value(controls[hue_key])
        if abs(saturation) <= 1e-8:
            continue
        angle = np.deg2rad(hue)
        amount = masks[mask_key] * (saturation / 100.0) * amplitude
        offset_a += amount * np.cos(angle)
        offset_b += amount * np.sin(angle)

    lab[..., 1] += offset_a
    lab[..., 2] += offset_b
    graded = _lab_to_gamut_safe_rgb(lab, cv2, np)
    return np.clip(np.nan_to_num(graded, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _apply_measured_color_grading(rgb, controls: dict[str, object], np):
    """Use gray-ramp measured wheel responses with inferred tone masks.

    ACR wheel responses are measured on gray ramps at Hue 0/Sat 100. Hue
    rotation and saturation scaling follow the XMP controls. Applying these
    responses to colored inputs, multiple wheels, or Balance/Blending-adjusted
    masks is an approximation because those combinations were not measured.
    """
    color_keys = (
        "split_shadow_saturation", "split_highlight_saturation",
        "grade_shadow_saturation", "grade_midtone_saturation",
        "grade_highlight_saturation", "grade_global_saturation",
    )
    if not any(abs(_finite_control_value(controls[key])) > 1e-8 for key in color_keys):
        return rgb

    import cv2
    from measured_response import apply_grading_wheels_float

    lab_luminance = cv2.cvtColor(
        np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0), cv2.COLOR_RGB2Lab,
    )[..., 0] / 100.0
    shadow, midtone, highlight = _grading_masks(lab_luminance, controls, np)
    reference_controls = dict(controls, split_balance=0.0, grade_blending=50.0)
    reference_shadow, reference_midtone, reference_highlight = _grading_masks(
        lab_luminance, reference_controls, np,
    )

    def relative_mask(current, reference):
        # The measured wheel curves already contain the default tonal mask.
        # Apply only a bounded ratio for changed Balance/Blending settings.
        floor = 0.05
        ratio = current / np.maximum(reference, floor)
        ratio = np.where(
            reference < floor,
            np.where(current < floor, 1.0, current / floor),
            ratio,
        )
        return np.clip(ratio, 0.0, 3.0)

    shadow = relative_mask(shadow, reference_shadow)
    midtone = relative_mask(midtone, reference_midtone)
    highlight = relative_mask(highlight, reference_highlight)
    wheels = [
        ("Shadows", _finite_control_value(controls["split_shadow_hue"]),
         _finite_control_value(controls["split_shadow_saturation"]), shadow),
        ("Highlights", _finite_control_value(controls["split_highlight_hue"]),
         _finite_control_value(controls["split_highlight_saturation"]), highlight),
        ("Shadows", _finite_control_value(controls["grade_shadow_hue"]),
         _finite_control_value(controls["grade_shadow_saturation"]), shadow),
        ("Midtones", _finite_control_value(controls["grade_midtone_hue"]),
         _finite_control_value(controls["grade_midtone_saturation"]), midtone),
        ("Highlights", _finite_control_value(controls["grade_highlight_hue"]),
         _finite_control_value(controls["grade_highlight_saturation"]), highlight),
    ]
    graded = apply_grading_wheels_float(rgb, wheels)

    # The measurement set has no global wheel. Retain the previous restrained
    # approximation for that one control without reapplying measured wheels.
    if abs(_finite_control_value(controls["grade_global_saturation"])) > 1e-8:
        global_controls = dict(controls)
        for key in (
            "split_shadow_saturation", "split_highlight_saturation",
            "grade_shadow_saturation", "grade_midtone_saturation",
            "grade_highlight_saturation",
        ):
            global_controls[key] = 0.0
        graded = _apply_color_grading(graded, global_controls, np)
    return np.clip(graded, 0.0, 1.0)


def _apply_curves(rgb, controls: dict[str, object], *, grayscale: bool, np):
    """Apply master and optional per-channel PV2012 curves to normalized RGB."""
    channel_curves = (controls["red_curve"], controls["green_curve"], controls["blue_curve"])
    x_master = np.asarray([pair[0] / 255.0 for pair in controls["tone_curve"]], dtype=np.float32)
    y_master = np.asarray([pair[1] / 255.0 for pair in controls["tone_curve"]], dtype=np.float32)
    for channel in range(3):
        if not grayscale:
            points = channel_curves[channel]
            x = np.asarray([pair[0] / 255.0 for pair in points], dtype=np.float32)
            y = np.asarray([pair[1] / 255.0 for pair in points], dtype=np.float32)
            rgb[..., channel] = np.interp(rgb[..., channel], x, y)
    luminance_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luminance = rgb @ luminance_weights
    target_luminance = np.interp(luminance, x_master, y_master)
    return _adjust_luminance_preserving_color(rgb, luminance, target_luminance, np)


def _adjust_luminance_preserving_color(rgb, luminance, target_luminance, np):
    """Move RGB samples along their existing channel ratios to a new luminance.

    Scaling all channels by one value retains hue and relative chroma for
    in-gamut colors. A zero pixel has no hue to preserve, so lifting it yields
    neutral gray. The scale is capped at the first channel's gamut boundary;
    this avoids clipping one channel and shifting hue on saturated colors.
    """
    maximum_channel = np.max(rgb, axis=2)
    safe_luminance = np.where(luminance > 1e-8, luminance, 1.0)
    requested_scale = target_luminance / safe_luminance
    safe_maximum_channel = np.where(maximum_channel > 1e-8, maximum_channel, 1.0)
    gamut_scale = 1.0 / safe_maximum_channel
    scale = np.minimum(requested_scale, gamut_scale)
    adjusted = rgb * scale[..., None]
    # A zero input is achromatic, so use the requested neutral lift directly.
    return np.where((luminance <= 1e-8)[..., None], target_luminance[..., None], adjusted)


def _apply_gr3_vivid_street_luminance_calibration(rgb, np):
    """Apply the measured preview's residual luminance curve, preserving RGB ratios."""
    luminance_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luminance = rgb @ luminance_weights
    target_luminance = np.interp(
        luminance, _GR3_VIVID_STREET_LUMA_X, _GR3_VIVID_STREET_LUMA_Y,
    )
    return _adjust_luminance_preserving_color(rgb, luminance, target_luminance, np)


def apply_ricoh_preview_effect(image: "object", preset_id: str,
                               basic_params: dict[str, float] | None = None,
                               use_measured_color: bool = False):
    """Approximate XMP Camera Raw controls on an RGB preview.

    Processing is row-chunked to bound temporary memory for full-resolution
    exports. Adobe's camera profile and internal color transforms remain
    proprietary, so this preview does not claim pixel parity with Camera Raw.
    When ``use_measured_color`` is true, the HSL stage adds independently
    measured first-order RGB residuals and Color Grading adds gray-ramp measured
    Lab wheel residuals. Multi-slider HSL combinations, multiple color wheels,
    colored inputs, and Balance/Blending masks are approximations because they
    were not measured together. DNG/export callers leave this false by default.
    """
    import numpy as np

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("preview image must be RGB")
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError("preview image must use 8-bit or 16-bit samples")
    controls = _preset_controls(preset_id)
    # ACR receives the merged absolute crs:* values. Apply the same merged
    # controls before HSL, grading and curves instead of editing rendered RGB.
    if basic_params is not None:
        adjustments = validate_basic_params(basic_params)
        for key, (_, low, high) in _BASIC_FIELDS.items():
            controls[key] = max(low, min(high, float(controls[key]) + adjustments[key]))
    maximum = float(np.iinfo(image.dtype).max)
    temperature = float(controls["temperature"])
    tint = float(controls["tint"])
    is_grayscale = bool(controls["grayscale"])
    output = np.empty_like(image)
    luminance_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    contrast = float(controls["contrast"])
    shadows = float(controls["shadows"]) / 100.0
    highlights = float(controls["highlights"]) / 100.0
    whites = float(controls["whites"]) / 100.0
    blacks = float(controls["blacks"]) / 100.0
    saturation = float(controls["saturation"]) / 100.0
    vibrance = float(controls["vibrance"]) / 100.0
    for row in range(0, image.shape[0], 64):
        end = min(image.shape[0], row + 64)
        rgb = image[row:end].astype(np.float32) / maximum
        # White balance controls in these reviewed presets are generally small;
        # model them as restrained channel gains around neutral.
        rgb[..., 0] *= max(0.6, 1.0 + temperature / 10000.0 + tint / 20000.0)
        rgb[..., 1] *= max(0.6, 1.0 + tint / 10000.0)
        rgb[..., 2] *= max(0.6, 1.0 - temperature / 10000.0 + tint / 20000.0)
        rgb *= float(2.0 ** float(controls["exposure"]))

        rgb = (rgb - 0.5) * max(0.2, 1.0 + contrast / 100.0) + 0.5
        # Work from a valid display RGB base, then apply tone controls as a
        # luminance remap. Multiplying each pixel by one common scale keeps its
        # channel ratios (and therefore its hue/saturation) intact until the
        # requested lift reaches the sRGB gamut boundary.
        rgb = np.clip(rgb, 0.0, 1.0)
        luminance = rgb @ luminance_weights
        shadow_mask = np.clip((0.62 - luminance) / 0.62, 0.0, 1.0) ** 1.5
        highlight_mask = np.clip((luminance - 0.38) / 0.62, 0.0, 1.0) ** 1.5
        target_luminance = np.clip(
            luminance
            + shadows * 0.22 * shadow_mask
            + highlights * 0.18 * highlight_mask
            + whites * 0.08 * highlight_mask
            + blacks * 0.08 * shadow_mask,
            0.0,
            1.0,
        )
        rgb = _adjust_luminance_preserving_color(rgb, luminance, target_luminance, np)

        # Match the calibrated shadow rendering by reducing chroma only where
        # the tone controls raised luminance. Mixing toward the new luma keeps
        # neutral pixels neutral and preserves the color direction.
        lift_strength = np.clip(
            np.maximum(target_luminance - luminance, 0.0)
            / np.maximum(target_luminance, 0.05),
            0.0,
            1.0,
        )
        lifted_luminance = rgb @ luminance_weights
        rgb = lifted_luminance[..., None] + (rgb - lifted_luminance[..., None]) * (
            1.0 - 0.6 * lift_strength[..., None]
        )
        rgb = np.clip(rgb, 0.0, 1.0)

        if is_grayscale:
            luminance = rgb @ luminance_weights
            rgb = np.repeat(luminance[..., None], 3, axis=2)
        else:
            if use_measured_color:
                from measured_response import apply_hsl_controls_float

                rgb = apply_hsl_controls_float(rgb, {
                    "hue": controls["hue_adjustments"],
                    "saturation": controls["saturation_adjustments"],
                    "luminance": controls["luminance_adjustments"],
                })
            else:
                rgb = _apply_hsl_adjustments(rgb, controls, np)
            luminance = rgb @ luminance_weights
            chroma = rgb - luminance[..., None]
            chroma_level = np.max(np.abs(chroma), axis=2)
            vibrance_factor = 1.0 + vibrance * np.clip(1.0 - chroma_level, 0.0, 1.0)
            rgb = luminance[..., None] + chroma * (1.0 + saturation) * vibrance_factor[..., None]
            rgb = np.clip(rgb, 0.0, 1.0)

            # Split Toning and Color Grading coexist in ACR. Combine their
            # wheel offsets once in Lab so each control contributes smoothly.
            if use_measured_color:
                rgb = _apply_measured_color_grading(rgb, controls, np)
            else:
                rgb = _apply_color_grading(rgb, controls, np)

        rgb = _apply_curves(rgb, controls, grayscale=is_grayscale, np=np)
        # Keep the XMP tone curve above. This is only its measured-preview
        # residual correction, applied after the complete preset processing.
        if use_measured_color and preset_id == "gr3_vivid_street":
            rgb = _apply_gr3_vivid_street_luminance_calibration(rgb, np)
        output[row:end] = np.clip(np.rint(rgb * maximum), 0.0, maximum).astype(image.dtype)
    return output


def _preset_payload(preset_id: str) -> bytes:
    preset = _PRESETS_BY_ID.get(preset_id)
    if preset is None:
        raise KeyError(preset_id)

    source = _resource_root() / preset.resource
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError("preset resource unavailable") from exc

    for element in root.iter():
        for attribute in tuple(element.attrib):
            namespace, separator, local_name = attribute[1:].partition("}") if attribute.startswith("{") else ("", "", attribute)
            if separator and namespace == _CRS_NS:
                normalized = local_name.casefold()
                if (
                    normalized in _PRESET_IDENTITY_FIELDS
                    or normalized.startswith("supports")
                    or normalized.startswith("requires")
                ):
                    del element.attrib[attribute]

    for parent in root.iter():
        for child in tuple(parent):
            if child.tag.startswith("{" + _CRS_NS + "}"):
                local_name = child.tag.rsplit("}", 1)[-1].casefold()
                if (
                    local_name in _PRESET_IDENTITY_FIELDS
                    or local_name.startswith("preset")
                ):
                    parent.remove(child)

    ET.register_namespace("x", "adobe:ns:meta/")
    ET.register_namespace("rdf", _RDF_NS)
    ET.register_namespace("crs", _CRS_NS)
    body = ET.tostring(root, encoding="utf-8")
    return (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + body
        + b'\n<?xpacket end="w"?>'
    )


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "文件或目录不存在"
    if isinstance(exc, PermissionError):
        return "没有读取或写入权限"
    if isinstance(exc, IsADirectoryError):
        return "输入不是图片文件"
    if isinstance(exc, FileExistsError):
        return "已存在同名 XMP，未覆盖"
    if isinstance(exc, ValueError):
        return "输入路径无效"
    if isinstance(exc, RuntimeError):
        return "内置预设资源不可用"
    if isinstance(exc, ET.ParseError):
        return "现有 XMP 无法解析，未修改"
    return "文件访问或写入失败"


def _input_name(raw_path: str) -> str:
    try:
        name = Path(raw_path).name
        return name or "输入路径"
    except (TypeError, ValueError):
        return "输入路径"


def _collect_photos(paths: Iterable[str]) -> tuple[list[Path], list[dict[str, str]]]:
    photos: dict[tuple[str, str], Path] = {}
    results: list[dict[str, str]] = []

    def add_photo(path: Path) -> None:
        # A RAW and its companion JPEG share one sidecar stem. Emit one result
        # and write one XMP for that stem, while keeping separate directories
        # independent.
        absolute = Path(os.path.abspath(path))
        key = (str(absolute.parent), absolute.stem.casefold())
        photos.setdefault(key, absolute)

    for raw_path in paths:
        name = _input_name(raw_path)
        try:
            if not raw_path or not raw_path.strip():
                raise ValueError("empty path")
            path = Path(raw_path).expanduser()
            if path.is_dir():
                if OUTPUT_DIR_NAME in path.parts:
                    continue
                try:
                    for candidate in path.rglob("*"):
                        if (
                            OUTPUT_DIR_NAME not in candidate.parts
                            and candidate.is_file()
                            and candidate.suffix.lower() in SUPPORTED_SUFFIXES
                        ):
                            add_photo(candidate)
                except OSError as exc:
                    results.append({"name": name, "status": "failed", "error": _safe_error(exc)})
                continue
            if not path.exists():
                raise FileNotFoundError
            if not path.is_file():
                raise IsADirectoryError
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                results.append({"name": name, "status": "failed", "error": "不支持的图片格式"})
                continue
            add_photo(path)
        except (OSError, ValueError) as exc:
            results.append({"name": name, "status": "failed", "error": _safe_error(exc)})

    return list(photos.values()), results


def _sidecar_exists_case_insensitive(directory: Path, stem: str) -> bool:
    wanted = (stem + ".xmp").casefold()
    try:
        return any(entry.name.casefold() == wanted for entry in directory.iterdir())
    except OSError:
        raise


def _sidecar_path(photo: Path) -> Path | None:
    wanted = (photo.stem + ".xmp").casefold()
    try:
        return next((entry for entry in photo.parent.iterdir() if entry.name.casefold() == wanted), None)
    except OSError:
        raise


def _description(root: ET.Element, *, create: bool = False) -> ET.Element | None:
    rdf = "{" + _RDF_NS + "}"
    for element in root.iter(rdf + "Description"):
        return element
    if not create:
        return None
    rdf_root = next(root.iter(rdf + "RDF"), None)
    if rdf_root is None:
        rdf_root = ET.SubElement(root, rdf + "RDF")
    return ET.SubElement(rdf_root, rdf + "Description", {rdf + "about": ""})


def _parse_xmp(payload: bytes) -> ET.Element:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(payload, parser=parser)
    if not any(element.tag == "{" + _RDF_NS + "}RDF" for element in root.iter()):
        raise ET.ParseError("XMP has no RDF packet")
    return root


def _find_simple(description: ET.Element, namespace: str, name: str) -> str | None:
    key = "{" + namespace + "}" + name
    value = description.attrib.get(key)
    if value is not None:
        return value
    child = next((element for element in description if element.tag == key), None)
    if child is not None and child.text is not None:
        return child.text.strip()
    return None


_DEHAZE_FIELDS = (
    "strength", "naturalness", "fog_retention", "local_contrast", "color_recovery",
    "color_protection", "highlight_protection", "shadow_protection", "brightness_protection",
)

_BASIC_FIELDS = {
    "exposure": ("Exposure2012", -5.0, 5.0),
    "contrast": ("Contrast2012", -100.0, 100.0),
    "highlights": ("Highlights2012", -100.0, 100.0),
    "shadows": ("Shadows2012", -100.0, 100.0),
    "whites": ("Whites2012", -100.0, 100.0),
    "blacks": ("Blacks2012", -100.0, 100.0),
    "vibrance": ("Vibrance", -100.0, 100.0),
    "saturation": ("Saturation", -100.0, 100.0),
}


def validate_basic_params(params: dict[str, float]) -> dict[str, float]:
    if set(params) != set(_BASIC_FIELDS):
        raise ValueError("invalid basic settings")
    import math
    values = {key: float(value) for key, value in params.items()}
    if any(not math.isfinite(values[key]) or not low <= values[key] <= high
           for key, (_, low, high) in _BASIC_FIELDS.items()):
        raise ValueError("invalid basic settings")
    return values


def apply_basic_preview_effect(image: "object", params: dict[str, float]):
    """Approximate Camera Raw's basic controls on an RGB display preview."""
    import numpy as np
    values = validate_basic_params(params)
    maximum = float(np.iinfo(image.dtype).max)
    rgb = image.astype(np.float32) / maximum
    rgb *= 2.0 ** values["exposure"]
    rgb = (rgb - 0.5) * (1.0 + values["contrast"] / 100.0) + 0.5
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    shadow = np.clip((0.62 - luminance) / 0.62, 0, 1) ** 1.5
    highlight = np.clip((luminance - 0.38) / 0.62, 0, 1) ** 1.5
    rgb += (values["shadows"] * 0.0022 + values["blacks"] * 0.0008) * shadow[..., None]
    rgb += (values["highlights"] * 0.0018 + values["whites"] * 0.0008) * highlight[..., None]
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    chroma = rgb - luminance[..., None]
    chroma_level = np.max(np.abs(chroma), axis=2)
    vibrance_factor = 1.0 + values["vibrance"] / 100.0 * np.clip(1.0 - chroma_level, 0, 1)
    rgb = luminance[..., None] + chroma * (1.0 + values["saturation"] / 100.0) * vibrance_factor[..., None]
    return np.clip(np.rint(rgb * maximum), 0, maximum).astype(image.dtype)


def read_photo_settings(photo: str | Path) -> dict[str, object]:
    """Read Imprint's own photo settings, tolerating absent or invalid sidecars."""
    path = Path(photo)
    sidecar = _sidecar_path(path)
    result: dict[str, object] = {"dehaze_params": None, "ricoh_preset_id": None,
                                 "basic_params": {key: 0.0 for key in _BASIC_FIELDS}}
    if sidecar is None:
        return result
    try:
        root = _parse_xmp(sidecar.read_bytes())
        description = _description(root)
        if description is None:
            return result
        values: dict[str, float] = {}
        for field_name in _DEHAZE_FIELDS:
            raw = _find_simple(description, _IMPRINT_NS, "Dehaze" + "".join(part.title() for part in field_name.split("_")))
            if raw is not None:
                value = float(raw)
                if not (0.0 <= value <= 1.0):
                    raise ValueError("invalid dehaze setting")
                values[field_name] = value
        if values:
            if len(values) != len(_DEHAZE_FIELDS):
                from dehaze import DehazeParams
                values = {**DehazeParams().__dict__, **values}
            result["dehaze_params"] = values
        preset_id = _find_simple(description, _IMPRINT_NS, "RicohPresetId")
        result["ricoh_preset_id"] = preset_id if preset_id in _PRESETS_BY_ID else None
        basic = {}
        for key in _BASIC_FIELDS:
            raw = _find_simple(description, _IMPRINT_NS, "Basic" + key.title())
            basic[key] = float(raw) if raw is not None else 0.0
        result["basic_params"] = validate_basic_params(basic)
    except (OSError, ET.ParseError, ValueError, TypeError):
        return result
    return result


def _serialize_xmp(root: ET.Element) -> bytes:
    ET.register_namespace("x", "adobe:ns:meta/")
    ET.register_namespace("rdf", _RDF_NS)
    ET.register_namespace("crs", _CRS_NS)
    ET.register_namespace("imprint", _IMPRINT_NS)
    body = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    return b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n' + body + b'\n<?xpacket end="w"?>'


def _atomic_write_sidecar(photo: Path, payload: bytes, *, create_only: bool = False) -> None:
    """Commit an XMP packet atomically; first writes never replace a sidecar."""
    photo = Path(os.path.abspath(photo))
    if not photo.is_file():
        raise FileNotFoundError
    directory = photo.parent
    directory.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        existing = _sidecar_path(photo)
        target = existing or directory / (photo.stem + ".xmp")
        if existing is not None and create_only:
            raise FileExistsError
        if existing is None and _sidecar_exists_case_insensitive(directory, photo.stem):
            raise FileExistsError
        descriptor, temporary_name = tempfile.mkstemp(prefix=".imprint-xmp-", suffix=".tmp", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            if not photo.is_file():
                raise FileNotFoundError
            if existing is not None:
                os.replace(temporary, target)
            else:
                # link() is an atomic no-replace commit on filesystems that
                # support hard links. Fall back to O_EXCL for removable media.
                try:
                    os.link(temporary, target)
                    temporary.unlink()
                except FileExistsError:
                    raise
                except OSError:
                    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
                    try:
                        with os.fdopen(fd, "wb") as output:
                            fd = -1
                            output.write(payload)
                            output.flush()
                            os.fsync(output.fileno())
                    except BaseException:
                        if fd >= 0:
                            os.close(fd)
                        target.unlink(missing_ok=True)
                        raise
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _merge_preset_payload(existing_payload: bytes | None, preset_id: str,
                          basic_params: dict[str, float] | None = None) -> bytes:
    preset_root = _parse_xmp(_preset_payload(preset_id))
    source = _description(preset_root)
    if source is None:
        raise RuntimeError("invalid preset resource")
    root = _parse_xmp(existing_payload) if existing_payload is not None else ET.Element("{adobe:ns:meta/}xmpmeta")
    destination = _description(root, create=True)
    assert destination is not None
    global _PRESET_PROCESSING_KEYS
    if _PRESET_PROCESSING_KEYS is None:
        attributes: set[str] = set()
        elements: set[str] = set()
        for candidate in _PRESETS:
            candidate_root = _parse_xmp(_preset_payload(candidate.id))
            candidate_description = _description(candidate_root)
            if candidate_description is None:
                continue
            attributes.update(
                key for key in candidate_description.attrib
                if key.startswith("{" + _CRS_NS + "}")
            )
            elements.update(
                child.tag for child in candidate_description
                if child.tag.startswith("{" + _CRS_NS + "}")
            )
        _PRESET_PROCESSING_KEYS = attributes, elements
    attribute_keys, element_keys = _PRESET_PROCESSING_KEYS
    # Remove all processing fields owned by any built-in preset first. This
    # prevents a previous preset's extra settings from leaking into a switch.
    for key in attribute_keys:
        destination.attrib.pop(key, None)
    for child in tuple(destination):
        if child.tag in element_keys:
            destination.remove(child)
    # Replace only Camera Raw processing properties supplied by this preset.
    # Everything else, including custom Imprint values and vendor metadata,
    # stays in the existing sidecar.
    for key, value in source.attrib.items():
        if key.startswith("{" + _CRS_NS + "}"):
            destination.set(key, value)
    for element in source:
        if element.tag.startswith("{" + _CRS_NS + "}"):
            key = element.tag
            for previous in tuple(destination):
                if previous.tag == key:
                    destination.remove(previous)
            destination.append(copy.deepcopy(element))
    # These are preset placeholders, not a Kelvin reading. Camera Raw clamps
    # Temperature=0 to 2000 K even when WhiteBalance says As Shot.
    if _find_simple(destination, _CRS_NS, "WhiteBalance") == "As Shot":
        for local in ("Temperature", "Tint"):
            destination.attrib.pop("{" + _CRS_NS + "}" + local, None)
            for child in tuple(destination):
                if child.tag == "{" + _CRS_NS + "}" + local:
                    destination.remove(child)
    destination.set("{" + _IMPRINT_NS + "}RicohPresetId", preset_id)
    if basic_params is not None:
        for key, value in validate_basic_params(basic_params).items():
            destination.set("{" + _IMPRINT_NS + "}Basic" + key.title(), format(value, ".8g"))
    for key, (crs_name, low, high) in _BASIC_FIELDS.items():
        adjustment = _find_simple(destination, _IMPRINT_NS, "Basic" + key.title())
        if adjustment is not None:
            base = float(_find_simple(source, _CRS_NS, crs_name) or 0)
            absolute = max(low, min(high, base + float(adjustment)))
            destination.set("{" + _CRS_NS + "}" + crs_name, format(absolute, ".8g"))
    return _serialize_xmp(root)


def write_ricoh_preset(photo: str | Path, preset_id: str,
                       basic_params: dict[str, float] | None = None) -> str:
    """Merge a reviewed preset into the photo's current XMP packet."""
    if preset_id not in _PRESETS_BY_ID:
        raise KeyError(preset_id)
    photo_path = Path(photo)
    if not photo_path.is_file():
        raise FileNotFoundError
    with _WRITE_LOCK:
        sidecar = _sidecar_path(photo_path)
        existing_payload = sidecar.read_bytes() if sidecar is not None else None
        payload = _merge_preset_payload(existing_payload, preset_id, basic_params)
        _atomic_write_sidecar(photo_path, payload)
        return (sidecar or photo_path.with_suffix(".xmp")).name


def write_dehaze_settings(photo: str | Path, params: dict[str, float],
                          basic_params: dict[str, float] | None = None) -> str:
    """Merge the nine validated dehaze parameters into a photo's XMP packet."""
    if set(params) != set(_DEHAZE_FIELDS):
        raise ValueError("invalid dehaze settings")
    values = {key: float(value) for key, value in params.items()}
    if any(not (0.0 <= value <= 1.0) for value in values.values()):
        raise ValueError("invalid dehaze settings")
    photo_path = Path(photo)
    if not photo_path.is_file():
        raise FileNotFoundError
    with _WRITE_LOCK:
        sidecar = _sidecar_path(photo_path)
        existing_payload = sidecar.read_bytes() if sidecar is not None else None
        if existing_payload is None:
            root = ET.Element("{adobe:ns:meta/}xmpmeta")
            description = _description(root, create=True)
        else:
            root = _parse_xmp(existing_payload)
            description = _description(root, create=True)
        assert description is not None
        for field_name, value in values.items():
            local = "Dehaze" + "".join(part.title() for part in field_name.split("_"))
            description.set("{" + _IMPRINT_NS + "}" + local, format(value, ".8g"))
        if basic_params is not None:
            basic = validate_basic_params(basic_params)
            preset_id = _find_simple(description, _IMPRINT_NS, "RicohPresetId")
            baseline = _preset_controls(preset_id) if preset_id in _PRESETS_BY_ID else {}
            for key, (crs_name, low, high) in _BASIC_FIELDS.items():
                description.set("{" + _IMPRINT_NS + "}Basic" + key.title(), format(basic[key], ".8g"))
                absolute = max(low, min(high, float(baseline.get(key, 0)) + basic[key]))
                description.set("{" + _CRS_NS + "}" + crs_name, format(absolute, ".8g"))
        payload = _serialize_xmp(root)
        _atomic_write_sidecar(photo_path, payload)
        return (sidecar or photo_path.with_suffix(".xmp")).name


def write_photo_settings(
    photo: str | Path,
    dehaze_params: dict[str, float],
    basic_params: dict[str, float],
    ricoh_preset_id: str | None,
) -> dict[str, str]:
    """Atomically merge one complete Imprint photo-settings snapshot into XMP."""
    if set(dehaze_params) != set(_DEHAZE_FIELDS):
        raise ValueError("invalid dehaze settings")
    dehaze_values = {key: float(value) for key, value in dehaze_params.items()}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
           for value in dehaze_values.values()):
        raise ValueError("invalid dehaze settings")
    basic_values = validate_basic_params(basic_params)
    if ricoh_preset_id is not None and ricoh_preset_id not in _PRESETS_BY_ID:
        raise KeyError(ricoh_preset_id)

    photo_path = Path(photo)
    if not photo_path.is_file():
        raise FileNotFoundError
    with _WRITE_LOCK:
        sidecar = _sidecar_path(photo_path)
        existing_payload = sidecar.read_bytes() if sidecar is not None else None
        if ricoh_preset_id is not None:
            # Preset processing controls and basic adjustments are merged in
            # memory first. The full settings snapshot is committed once below.
            root = _parse_xmp(_merge_preset_payload(
                existing_payload, ricoh_preset_id, basic_values,
            ))
        else:
            root = (
                _parse_xmp(existing_payload)
                if existing_payload is not None
                else ET.Element("{adobe:ns:meta/}xmpmeta")
            )
            description = _description(root, create=True)
            assert description is not None
            existing_preset_id = _find_simple(description, _IMPRINT_NS, "RicohPresetId")
            baseline = (
                _preset_controls(existing_preset_id)
                if existing_preset_id in _PRESETS_BY_ID else {}
            )
            for key, (crs_name, low, high) in _BASIC_FIELDS.items():
                description.set("{" + _IMPRINT_NS + "}Basic" + key.title(),
                                format(basic_values[key], ".8g"))
                absolute = max(low, min(high, float(baseline.get(key, 0)) + basic_values[key]))
                description.set("{" + _CRS_NS + "}" + crs_name, format(absolute, ".8g"))

        description = _description(root, create=True)
        assert description is not None
        for field_name, value in dehaze_values.items():
            local = "Dehaze" + "".join(part.title() for part in field_name.split("_"))
            description.set("{" + _IMPRINT_NS + "}" + local, format(value, ".8g"))
        _atomic_write_sidecar(photo_path, _serialize_xmp(root))
        return {
            "name": (sidecar or photo_path.with_suffix(".xmp")).name,
            "status": "updated" if sidecar is not None else "written",
        }


def write_basic_settings(photo: str | Path, params: dict[str, float]) -> str:
    """Write ACR basic fields as preset-relative adjustments in an XMP sidecar."""
    values = validate_basic_params(params)
    photo_path = Path(photo)
    if not photo_path.is_file():
        raise FileNotFoundError
    with _WRITE_LOCK:
        sidecar = _sidecar_path(photo_path)
        root = _parse_xmp(sidecar.read_bytes()) if sidecar else ET.Element("{adobe:ns:meta/}xmpmeta")
        description = _description(root, create=True)
        assert description is not None
        preset_id = _find_simple(description, _IMPRINT_NS, "RicohPresetId")
        baseline = _preset_controls(preset_id) if preset_id in _PRESETS_BY_ID else {}
        for key, (crs_name, low, high) in _BASIC_FIELDS.items():
            description.set("{" + _IMPRINT_NS + "}Basic" + key.title(), format(values[key], ".8g"))
            absolute = max(low, min(high, float(baseline.get(key, 0)) + values[key]))
            description.set("{" + _CRS_NS + "}" + crs_name, format(absolute, ".8g"))
        _atomic_write_sidecar(photo_path, _serialize_xmp(root))
        return (sidecar or photo_path.with_suffix(".xmp")).name


def _summarize(files: list[dict[str, str]]) -> dict[str, object]:
    counts = {status: sum(item["status"] == status for item in files) for status in ("written", "updated", "skipped", "failed")}
    return {
        "total": len(files),
        "written": counts["written"] + counts["updated"],
        "created": counts["written"],
        "updated": counts["updated"],
        "skipped": counts["skipped"],
        "failed": counts["failed"],
        "files": files,
    }


def apply_ricoh_preset(paths: list[str], preset_id: str) -> dict[str, object]:
    """Merge a preset's Camera Raw settings into safe XMP sidecars."""
    if preset_id not in _PRESETS_BY_ID:
        raise KeyError(preset_id)

    photos, files = _collect_photos(paths)
    if len(photos) + len(files) > _MAX_PHOTOS:
        raise RicohBatchLimitError(f"最多处理 {_MAX_PHOTOS} 张照片")
    if not photos:
        return _summarize(files)
    try:
        _preset_payload(preset_id)
    except RuntimeError as exc:
        files.extend(
            {"name": photo.name, "status": "failed", "error": _safe_error(exc)}
            for photo in photos
        )
        return _summarize(files)

    for photo in photos:
        try:
            existing = _sidecar_path(photo)
            sidecar_name = write_ricoh_preset(photo, preset_id)
            files.append({"name": sidecar_name, "status": "updated" if existing else "written"})
        except FileExistsError as exc:
            files.append({"name": photo.stem + ".xmp", "status": "skipped", "error": _safe_error(exc)})
        except (OSError, RuntimeError, ValueError, ET.ParseError) as exc:
            files.append({"name": photo.stem + ".xmp", "status": "failed", "error": _safe_error(exc)})

    return _summarize(files)


def apply_ricoh_preset_to_session(
    photos: Iterable[tuple[str, str | Path]], preset_id: str,
    basic_params_by_photo: dict[str, dict[str, float]] | None = None,
    preset_ids_by_photo: dict[str, str | None] | None = None,
) -> dict[str, object]:
    """Apply one preset to session records, writing only one XMP per stem."""
    if preset_id not in _PRESETS_BY_ID:
        raise KeyError(preset_id)
    if any(value is not None and value not in _PRESETS_BY_ID
           for value in (preset_ids_by_photo or {}).values()):
        raise KeyError("unknown per-photo preset")
    records = list(photos)
    if len(records) > _MAX_PHOTOS:
        raise RicohBatchLimitError(f"最多处理 {_MAX_PHOTOS} 张照片")
    files: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for photo_id, raw_path in records:
        photo = Path(raw_path)
        key = (str(photo.parent).casefold(), photo.stem.casefold())
        name = photo.stem + ".xmp"
        selected_preset_id = (preset_ids_by_photo or {}).get(photo_id, preset_id)
        if selected_preset_id is None:
            files.append({"photo_id": photo_id, "name": name, "status": "skipped",
                          "error": "未选择理光预设"})
            continue
        if selected_preset_id not in _PRESETS_BY_ID:
            raise KeyError(selected_preset_id)
        if key in seen:
            files.append({"photo_id": photo_id, "name": name, "status": "skipped", "error": "同名伴生照片共用一个 XMP"})
            continue
        seen.add(key)
        try:
            existing = _sidecar_path(photo)
            name = write_ricoh_preset(
                photo, selected_preset_id,
                (basic_params_by_photo or {}).get(photo_id),
            )
            files.append({"photo_id": photo_id, "name": name, "status": "updated" if existing else "written"})
        except (OSError, RuntimeError, ValueError, ET.ParseError) as exc:
            files.append({"photo_id": photo_id, "name": name, "status": "failed", "error": _safe_error(exc)})
    return _summarize(files)


def write_dehaze_session_settings(
    photos: Iterable[tuple[str, str | Path, dict[str, float]]],
    basic_params_by_photo: dict[str, dict[str, float]] | None = None,
    preset_ids_by_photo: dict[str, str | None] | None = None,
) -> dict[str, object]:
    """Write per-photo dehaze values through active session records."""
    if any(value is not None and value not in _PRESETS_BY_ID
           for value in (preset_ids_by_photo or {}).values()):
        raise KeyError("unknown per-photo preset")
    records = list(photos)
    if len(records) > 5000:
        raise RicohBatchLimitError("最多处理 5000 张照片")
    files: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for photo_id, raw_path, params in records:
        photo = Path(raw_path)
        key = (str(photo.parent).casefold(), photo.stem.casefold())
        name = photo.stem + ".xmp"
        if key in seen:
            files.append({"photo_id": photo_id, "name": name, "status": "skipped", "error": "同名伴生照片共用一个 XMP"})
            continue
        seen.add(key)
        try:
            existing = _sidecar_path(photo)
            basic = (basic_params_by_photo or {}).get(photo_id)
            if preset_ids_by_photo is not None and photo_id in preset_ids_by_photo:
                if basic is None:
                    basic = read_photo_settings(photo)["basic_params"]
                name = write_photo_settings(
                    photo, params, basic, preset_ids_by_photo[photo_id],
                )["name"]
            else:
                name = write_dehaze_settings(photo, params, basic)
            files.append({"photo_id": photo_id, "name": name, "status": "updated" if existing else "written"})
        except (OSError, RuntimeError, ValueError, ET.ParseError) as exc:
            files.append({"photo_id": photo_id, "name": name, "status": "failed", "error": _safe_error(exc)})
    return _summarize(files)
