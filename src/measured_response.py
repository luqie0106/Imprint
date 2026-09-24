"""Preview helpers for measured ACR single-control output residuals.

The LUTs are empirical responses measured from synthetic TIFFs, not Adobe's
algorithms. HSL controls are combined by adding their independently measured
first-order RGB residuals at one shared input HSL coordinate. Interactions
between sliders were not measured, so multi-control results are approximations.
Color Grading wheel responses were measured on gray ramps and are only an
approximation when applied to colored photos or multiple active wheels.
"""

from __future__ import annotations

import sys
import io
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms


KINDS = ("hue", "saturation", "luminance")
BANDS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
GRADING_SETTINGS = ("Shadows", "Midtones", "Highlights", "Blending", "Balance")


def standard_preview_to_srgb(image: np.ndarray, path: str | Path) -> np.ndarray:
    """Honor an embedded ICC profile on a standard image's RGB preview.

    The image has already been oriented and resized by image_io. Untagged
    images follow the app's existing sRGB assumption.
    """
    with Image.open(path) as source:
        icc_bytes = source.info.get("icc_profile")
    if not icc_bytes:
        return image
    source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
    target_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    converted = ImageCms.profileToProfile(
        Image.fromarray(image, "RGB"), source_profile, target_profile,
        outputMode="RGB", renderingIntent=0,
    )
    return np.asarray(converted, dtype=np.uint8).copy()


def _lut_path() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    root = Path(frozen_root) if frozen_root else Path(__file__).resolve().parents[1]
    return root / "assets" / "acr_measured" / "hsl_preview_lut.npz"


@lru_cache(maxsize=1)
def _load_lut() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(_lut_path(), allow_pickle=False) as data:
        delta = np.asarray(data["delta_rgb"], dtype=np.float32)
        levels = np.asarray(data["levels"], dtype=np.float32)
        sats = np.asarray(data["saturations"], dtype=np.float32)
        lights = np.asarray(data["lightnesses"], dtype=np.float32)
    if delta.shape != (3, 8, len(levels), len(sats), len(lights), 360, 3):
        raise ValueError("ACR 实测响应数据尺寸不匹配")
    if not (np.all(np.diff(levels) > 0) and np.all(np.diff(sats) > 0)
            and np.all(np.diff(lights) > 0) and np.isfinite(delta).all()):
        raise ValueError("ACR 实测响应数据无效")
    return delta, levels, sats, lights


@lru_cache(maxsize=1)
def _load_grading_lut() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(_lut_path().with_name("grading_preview_lut.npz"), allow_pickle=False) as data:
        lstar = np.asarray(data["gray_cie_lstar"], dtype=np.float32)
        baseline = np.asarray(data["baseline_rgb"], dtype=np.float32)
        wheel = np.asarray(data["wheel_delta_rgb"], dtype=np.float32)
        blend = np.asarray(data["blending_delta_rgb"], dtype=np.float32)
        balance = np.asarray(data["balance_delta_rgb"], dtype=np.float32)
        blend_levels = np.asarray(data["blending_levels"], dtype=np.float32)
        balance_levels = np.asarray(data["balance_levels"], dtype=np.float32)
    if (lstar.shape != (4096,) or baseline.shape != (4096, 3) or wheel.shape != (3, 4096, 3)
            or blend.shape != (5, 4096, 3) or balance.shape != (5, 4096, 3)
            or not all(np.isfinite(x).all() for x in (lstar, baseline, wheel, blend, balance))
            or not np.all(np.diff(lstar) > 0)):
        raise ValueError("ACR 灰阶响应数据无效")
    return lstar, baseline, wheel, blend, balance, blend_levels, balance_levels


def apply_grading_gray_preview(image: np.ndarray, setting: str, value: float,
                               *, baseline_only: bool = False) -> np.ndarray:
    """Render a grayscale *input* through measured ACR gray-ramp responses.

    This intentionally converts the photo to gray before the lookup. It makes
    no claim about Color Grading on colored inputs.
    """
    if setting not in GRADING_SETTINGS or not np.isfinite(value):
        raise ValueError("未知 Color Grading 实测设置")
    if (setting == "Blending" and not 0 <= value <= 100) or (setting == "Balance" and not -100 <= value <= 100):
        raise ValueError("Color Grading 数值超出实测范围")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("灰阶响应预览需要 8-bit RGB")
    _, baseline, wheel, blend, balance, blend_levels, balance_levels = _load_grading_lut()
    encoded = image.astype(np.float32) / 255.0
    linear = np.where(encoded <= 0.04045, encoded / 12.92,
                      ((encoded + 0.055) / 1.055) ** 2.4)
    y = linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    gray = np.where(y <= 0.0031308, y * 12.92,
                    1.055 * np.power(y, 1.0 / 2.4) - 0.055)
    step = np.clip(np.rint(gray * 4095).astype(np.int16), 0, 4095)
    if baseline_only:
        return np.rint(np.clip(baseline[step], 0.0, 1.0) * 255).astype(np.uint8)
    if setting in ("Shadows", "Midtones", "Highlights"):
        residual = wheel[GRADING_SETTINGS.index(setting), step]
    else:
        levels, curves = (blend_levels, blend) if setting == "Blending" else (balance_levels, balance)
        upper = min(max(int(np.searchsorted(levels, value, side="right")), 1), len(levels) - 1)
        lower = upper - 1
        fraction = float((value - levels[lower]) / (levels[upper] - levels[lower]))
        residual = curves[lower, step] * (1.0 - fraction) + curves[upper, step] * fraction
    return np.rint(np.clip(baseline[step] + residual, 0.0, 1.0) * 255).astype(np.uint8)


def _measured_hsl_residual(rgb: np.ndarray, adjustments: dict[str, dict[str, float]]) -> np.ndarray:
    """Sample summed, single-control HSL response grids once per pixel."""
    has_active = False
    for kind in KINDS:
        mapping = adjustments.get(kind, {})
        if isinstance(mapping, dict):
            for value in mapping.values():
                try:
                    has_active |= np.isfinite(float(value)) and abs(float(value)) > 1e-8
                except (TypeError, ValueError):
                    continue
    if not has_active:
        return np.zeros_like(rgb)

    delta, levels, sats, lights = _load_lut()
    combined = np.zeros((len(sats), len(lights), 360, 3), dtype=np.float32)
    active = False
    for kind in KINDS:
        mapping = adjustments.get(kind, {})
        if not isinstance(mapping, dict):
            continue
        for band in BANDS:
            try:
                value = float(mapping.get(band, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if not np.isfinite(value) or abs(value) <= 1e-8:
                continue
            value = float(np.clip(value, levels[0], levels[-1]))
            upper = int(np.searchsorted(levels, value, side="right"))
            upper = min(max(upper, 1), len(levels) - 1)
            lower = upper - 1
            fraction = float((value - levels[lower]) / (levels[upper] - levels[lower]))
            model = delta[KINDS.index(kind), BANDS.index(band)]
            combined += model[lower] * (1.0 - fraction) + model[upper] * fraction
            active = True
    if not active:
        return np.zeros_like(rgb)

    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    hue = np.mod(hls[..., 0], 360.0)
    light = hls[..., 1]
    sat = hls[..., 2]
    # The measured domain is S=.10..95 and L=.05..95. Pixels outside it have
    # no measured response and therefore receive no residual.
    epsilon = 2e-5
    in_domain = ((sat >= sats[0] - epsilon) & (sat <= sats[-1] + epsilon)
                 & (light >= lights[0] - epsilon) & (light <= lights[-1] + epsilon))
    sat = np.clip(sat, sats[0], sats[-1])
    light = np.clip(light, lights[0], lights[-1])
    si = np.clip(np.searchsorted(sats, sat, side="right") - 1, 0, len(sats) - 2)
    li = np.clip(np.searchsorted(lights, light, side="right") - 1, 0, len(lights) - 2)
    sfrac = (sat - sats[si]) / (sats[si + 1] - sats[si])
    lfrac = (light - lights[li]) / (lights[li + 1] - lights[li])
    hue_floor = np.floor(hue)
    hi = hue_floor.astype(np.int16) % 360
    hfrac = (hue - hue_floor)[..., None]
    residual = np.zeros_like(rgb)
    for ds in (0, 1):
        sw = (1.0 - sfrac if ds == 0 else sfrac)[..., None]
        for dl in (0, 1):
            lw = (1.0 - lfrac if dl == 0 else lfrac)[..., None]
            at_hue = combined[si + ds, li + dl, hi]
            at_next = combined[si + ds, li + dl, (hi + 1) % 360]
            residual += sw * lw * (at_hue * (1.0 - hfrac) + at_next * hfrac)
    return np.where(in_domain[..., None], residual, 0.0)


def apply_hsl_controls_float(image: np.ndarray, adjustments: dict[str, dict[str, float]]) -> np.ndarray:
    """Apply measured HSL residuals to normalized float32 sRGB.

    HSL is calculated once from the original input. Every nonzero control's
    single-slider response grid is interpolated and summed before one lookup.
    Since slider interactions were not measured, combined settings are a
    first-order approximation. Out-of-domain pixels remain unchanged.
    """
    if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("实测 HSL 预览需要归一化 float32 RGB")
    rgb = np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    residual = _measured_hsl_residual(rgb, adjustments)
    return np.clip(rgb + residual, 0.0, 1.0).astype(np.float32, copy=False)


def apply_grading_wheels_float(
    image: np.ndarray,
    wheels: list[tuple[str, float, float, np.ndarray | None]],
) -> np.ndarray:
    """Apply measured gray-ramp wheel Lab residuals to normalized float RGB.

    Each wheel tuple is ``(name, hue, saturation, mask)`` for Shadows,
    Midtones, or Highlights. A list permits split toning and Color Grading to
    contribute cumulatively to the same wheel. Hue-0 measured A/B residuals
    are rotated to the requested hue and L residuals scale with saturation.
    Applying gray-ramp data to colored photos or multiple active wheels is an
    approximation, not measured parity.
    """
    if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("实测 Color Grading 预览需要归一化 float32 RGB")
    wheel_indexes = {"Shadows": 0, "Midtones": 1, "Highlights": 2}
    active = []
    for name, hue, saturation, mask in wheels:
        if name not in wheel_indexes:
            raise ValueError("未知 Color Grading 轮盘")
        hue = float(hue) if np.isfinite(hue) else 0.0
        saturation = float(saturation) if np.isfinite(saturation) else 0.0
        if abs(saturation) > 1e-8:
            active.append((name, hue, float(np.clip(saturation, -100.0, 100.0)), mask))
    if not active:
        return image.copy()

    lstar_axis, baseline_rgb, wheel_rgb, _, _, _, _ = _load_grading_lut()
    baseline_rgb = np.clip(baseline_rgb, 0.0, 1.0)
    baseline_lab = cv2.cvtColor(baseline_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    delta_lab = np.empty_like(wheel_rgb)
    for idx in range(3):
        measured_rgb = np.clip(baseline_rgb + wheel_rgb[idx], 0.0, 1.0)
        measured_lab = cv2.cvtColor(measured_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
        delta_lab[idx] = measured_lab - baseline_lab

    rgb = np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    upper = np.clip(np.searchsorted(lstar_axis, lab[..., 0], side="right"), 1, len(lstar_axis) - 1)
    lower = upper - 1
    fraction = ((lab[..., 0] - lstar_axis[lower])
                / (lstar_axis[upper] - lstar_axis[lower]))[..., None]
    delta_l = np.zeros_like(lab[..., 0])
    delta_a = np.zeros_like(lab[..., 0])
    delta_b = np.zeros_like(lab[..., 0])
    for name, hue, saturation, mask in active:
        response = delta_lab[wheel_indexes[name]][lower] * (1.0 - fraction)
        response += delta_lab[wheel_indexes[name]][upper] * fraction
        if mask is not None:
            if mask.shape != lab.shape[:2]:
                raise ValueError("Color Grading 色调遮罩尺寸不匹配")
            weight = np.clip(np.nan_to_num(mask, nan=0.0), 0.0, 1.0) * (saturation / 100.0)
        else:
            weight = saturation / 100.0
        angle = np.deg2rad(hue)
        delta_l += response[..., 0] * weight
        delta_a += (response[..., 1] * np.cos(angle) - response[..., 2] * np.sin(angle)) * weight
        delta_b += (response[..., 1] * np.sin(angle) + response[..., 2] * np.cos(angle)) * weight

    lab[..., 0] += delta_l
    lab[..., 1] += delta_a
    lab[..., 2] += delta_b
    graded = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.clip(np.nan_to_num(graded, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32, copy=False)


def apply_hsl_single_slider(image: np.ndarray, kind: str, band: str, value: float) -> np.ndarray:
    """Add measured output RGB residual for one slider to an sRGB preview.

    Input photos use their own display RGB as the starting point, so this does
    not reproduce the ACR all-zero rendering or untested control combinations.
    """
    if kind not in KINDS or band not in BANDS:
        raise ValueError("未知 HSL 实测滑杆")
    if not np.isfinite(value) or not -100 <= value <= 100:
        raise ValueError("HSL 实测滑杆须介于 -100 与 100")
    if image.dtype not in (np.uint8, np.uint16) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("预览图必须是 8/16-bit RGB")
    if value == 0:
        return image.copy()
    delta, levels, sats, lights = _load_lut()
    model = delta[KINDS.index(kind), BANDS.index(band)]
    upper = int(np.searchsorted(levels, value, side="right"))
    upper = min(max(upper, 1), len(levels) - 1)
    lower = upper - 1
    fraction = float((value - levels[lower]) / (levels[upper] - levels[lower]))
    grid = model[lower] * (1.0 - fraction) + model[upper] * fraction
    maximum = float(np.iinfo(image.dtype).max)
    output = np.empty_like(image)
    for start in range(0, image.shape[0], 64):
        end = min(start + 64, image.shape[0])
        rgb = image[start:end].astype(np.float32) / maximum
        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
        hue = np.mod(hls[..., 0], 360.0)
        light = hls[..., 1]
        sat = hls[..., 2]
        # Data were measured on S=.10..95 and L=.05..95. Outside this
        # envelope the response is unknown, so leave those pixels untouched.
        epsilon = 2e-5  # float32 HLS round-trip at a measured grid endpoint
        in_domain = ((sat >= sats[0] - epsilon) & (sat <= sats[-1] + epsilon)
                     & (light >= lights[0] - epsilon) & (light <= lights[-1] + epsilon))
        sat = np.clip(sat, sats[0], sats[-1])
        light = np.clip(light, lights[0], lights[-1])
        si = np.clip(np.searchsorted(sats, sat, side="right") - 1, 0, len(sats) - 2)
        li = np.clip(np.searchsorted(lights, light, side="right") - 1, 0, len(lights) - 2)
        sfrac = (sat - sats[si]) / (sats[si + 1] - sats[si])
        lfrac = (light - lights[li]) / (lights[li + 1] - lights[li])
        hi = np.floor(hue).astype(np.int16) % 360
        hfrac = (hue - np.floor(hue))[..., None]
        acc = np.zeros_like(rgb)
        for ds in (0, 1):
            sw = (1.0 - sfrac if ds == 0 else sfrac)[..., None]
            for dl in (0, 1):
                lw = (1.0 - lfrac if dl == 0 else lfrac)[..., None]
                at_hue = grid[si + ds, li + dl, hi]
                at_next = grid[si + ds, li + dl, (hi + 1) % 360]
                acc += sw * lw * (at_hue * (1.0 - hfrac) + at_next * hfrac)
        predicted = np.where(in_domain[..., None], np.clip(rgb + acc, 0.0, 1.0), rgb)
        output[start:end] = np.rint(predicted * maximum).astype(image.dtype)
    return output
