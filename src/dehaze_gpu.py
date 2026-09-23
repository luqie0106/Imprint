"""Optional GPU implementations for the dehaze transform.

This module intentionally has no eager torch or OpenCV imports.  ``dehaze``
imports it only after selecting an accelerator, which keeps a CPU-only sidecar
lightweight and startable.  Device tensors are local to each invocation; no
mutable image or tensor state is shared between preview and batch requests.
"""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dehaze import DehazeParams


_OPENCL_LOCK = RLock()
_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _smooth_chroma_gamut_t(image, torch):
    weights = torch.tensor(_WEIGHTS, dtype=torch.float32, device=image.device)
    luminance = torch.sum(image * weights, dim=2)
    chroma = image - luminance.unsqueeze(2)
    eps = torch.tensor(1e-7, dtype=image.dtype, device=image.device)
    positive_room = torch.where(
        chroma > eps,
        (1.0 - luminance.unsqueeze(2)) / chroma,
        torch.full_like(chroma, float("inf")),
    )
    negative_room = torch.where(
        chroma < -eps,
        luminance.unsqueeze(2) / (-chroma),
        torch.full_like(chroma, float("inf")),
    )
    limit = torch.amin(torch.minimum(positive_room, negative_room), dim=2)
    limit = torch.where(torch.isfinite(limit), limit, torch.ones_like(limit))
    scale = 0.5 * (1.0 + limit - torch.sqrt((1.0 - limit) ** 2 + 1e-10))
    scale = torch.clamp(scale, 0.0, 1.0)
    return luminance.unsqueeze(2) + chroma * scale.unsqueeze(2)


def _smooth_chroma_caps_t(image, caps, torch):
    weights = torch.tensor(_WEIGHTS, dtype=torch.float32, device=image.device)
    luminance = torch.sum(image * weights, dim=2)
    chroma = image - luminance.unsqueeze(2)
    eps = torch.tensor(1e-7, dtype=image.dtype, device=image.device)
    positive_room = torch.where(
        chroma > eps,
        (caps - luminance.unsqueeze(2)) / chroma,
        torch.full_like(chroma, float("inf")),
    )
    negative_room = torch.where(
        chroma < -eps,
        luminance.unsqueeze(2) / (-chroma),
        torch.full_like(chroma, float("inf")),
    )
    limit = torch.amin(torch.minimum(positive_room, negative_room), dim=2)
    limit = torch.where(torch.isfinite(limit), limit, torch.ones_like(limit))
    scale = 0.5 * (1.0 + limit - torch.sqrt((1.0 - limit) ** 2 + 1e-10))
    scale = torch.clamp(scale, 0.0, 1.0)
    return luminance.unsqueeze(2) + chroma * scale.unsqueeze(2)


def _apply_brightness_protection_t(source, image, strength, protection, torch):
    if protection <= 1e-6:
        return image
    weights = torch.tensor(_WEIGHTS, dtype=torch.float32, device=image.device)
    source_luma = torch.sum(source * weights, dim=2)
    image_luma = torch.sum(image * weights, dim=2)
    valid = (
        torch.isfinite(source_luma)
        & torch.isfinite(image_luma)
        & (source_luma > 0.08)
        & (source_luma < 0.88)
    )
    if int(torch.count_nonzero(valid).item()) < 8:
        return image
    source_median = torch.median(source_luma[valid])
    image_median = torch.median(image_luma[valid])
    if not bool(torch.isfinite(source_median) & torch.isfinite(image_median)):
        return image
    if float(source_median.item()) <= 1e-5 or float(image_median.item()) <= 1e-5:
        return image
    brightness_drop_ev = torch.log2(source_median / image_median)
    if not bool(torch.isfinite(brightness_drop_ev)):
        return image
    allowed_drop_ev = 0.08 + 0.22 * float(np.clip(strength, 0.0, 1.0))
    excess_drop_ev = torch.clamp(brightness_drop_ev - allowed_drop_ev, min=0.0)
    compensation_ev = torch.clamp(
        excess_drop_ev * float(np.clip(protection, 0.0, 1.0)), max=0.40
    )
    if float(compensation_ev.item()) <= 1e-6:
        return image
    gain = torch.pow(torch.tensor(2.0, device=image.device), compensation_ev)
    curve_luma = image_luma * gain / (1.0 + (gain - 1.0) * image_luma)
    scale = torch.where(image_luma > 1e-6, curve_luma / image_luma, torch.ones_like(image_luma))
    scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    protected = torch.clamp(torch.nan_to_num(image * scale.unsqueeze(2), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return _smooth_chroma_gamut_t(protected, torch)


def _torch_dehaze(image_rgb: np.ndarray, params: "DehazeParams", device: str, torch) -> np.ndarray:
    """Run the vectorized transform on a CUDA or MPS tensor."""
    peak = float(np.iinfo(image_rgb.dtype).max)
    source_np = image_rgb.astype(np.float32) / peak
    flat = source_np.reshape(-1, 3)
    dark = np.min(flat, axis=1)
    count = min(dark.size, max(16, int(dark.size * 0.001)))
    if count <= 0:
        atmosphere = np.ones(3, dtype=np.float32)
    else:
        indices = np.argpartition(dark, -count)[-count:]
        candidates = flat[indices]
        luminance = candidates @ np.asarray(_WEIGHTS, dtype=np.float32)
        brightest = candidates[np.argsort(luminance)[-max(1, count // 8):]]
        atmosphere = np.clip(np.median(brightest, axis=0), 0.35, 1.0).astype(np.float32)
    dark_reference = float(np.percentile(np.min(source_np, axis=2), 75))
    if not np.isfinite(dark_reference):
        dark_reference = 0.5
    haze_level = float(np.clip((dark_reference - 0.03) / 0.92, 0.12, 0.92))

    omega = (
        params.strength
        * (0.66 - 0.14 * params.fog_retention)
        * (0.90 + 0.10 * (1.0 - params.naturalness))
        * (0.92 + 0.08 * haze_level)
    )
    min_transmission = 0.27 + 0.21 * params.fog_retention + 0.11 * params.naturalness
    transmission = float(np.clip(1.0 - omega, min_transmission, 1.0))
    neutral_atmosphere = float(np.mean(atmosphere))
    neutral_mix = 0.45 * params.color_protection * (0.65 + 0.35 * params.naturalness)
    atmosphere = atmosphere * (1.0 - neutral_mix) + neutral_atmosphere * neutral_mix
    atmosphere = np.clip(atmosphere, 0.35, 1.0)

    with torch.no_grad():
        source = torch.from_numpy(np.ascontiguousarray(source_np)).to(device=device, dtype=torch.float32)
        weights = torch.tensor(_WEIGHTS, dtype=torch.float32, device=device)
        atmosphere_t = torch.tensor(atmosphere, dtype=torch.float32, device=device).reshape(1, 1, 3)
        recovered = (source - atmosphere_t) / transmission + atmosphere_t
        recovered = torch.clamp(recovered, 0.0, 1.0)
        luminance = torch.sum(source * weights, dim=2)
        global_blend = params.strength * (0.78 - 0.28 * params.fog_retention) * (0.90 + 0.10 * (1.0 - params.naturalness))
        blend = float(np.clip(global_blend, 0.0, 0.82))
        natural = source * (1.0 - blend) + recovered * blend

        enhanced_luma = torch.sum(natural * weights, dim=2)
        source_luma = torch.clamp(luminance, min=1e-4)
        luma_only = torch.clamp(source * (enhanced_luma / source_luma).unsqueeze(2), 0.0, 1.0)
        chroma_mix = params.color_protection * (0.72 + 0.28 * params.naturalness)
        natural = natural * (1.0 - chroma_mix) + luma_only * chroma_mix

        if params.color_recovery > 1e-6:
            source_chroma = source - luminance.unsqueeze(2)
            source_chroma_norm = torch.sqrt(torch.sum(source_chroma * source_chroma, dim=2))
            confidence = torch.clamp((source_chroma_norm - 0.006) / 0.084, 0.0, 1.0)
            confidence = confidence * confidence * (3.0 - 2.0 * confidence)
            source_direction = source_chroma / torch.clamp(source_chroma_norm.unsqueeze(2), min=1e-6)
            natural_luma = torch.sum(natural * weights, dim=2)
            natural_chroma = natural - natural_luma.unsqueeze(2)
            natural_chroma_norm = torch.sqrt(torch.sum(natural_chroma * natural_chroma, dim=2))
            aligned_chroma = torch.clamp(torch.sum(natural_chroma * source_direction, dim=2), min=0.0)
            recovery_amount = params.color_recovery * params.strength
            source_target_norm = source_chroma_norm * (1.0 + 1.80 * params.color_recovery * confidence)
            natural_target_norm = natural_chroma_norm * (1.0 + 0.30 * recovery_amount * confidence) * confidence
            target_chroma_norm = torch.maximum(torch.maximum(aligned_chroma * confidence, natural_target_norm), source_target_norm)
            target_chroma = source_direction * target_chroma_norm.unsqueeze(2)
            highlight_position = torch.clamp((luminance - 0.58) / 0.40, 0.0, 1.0)
            highlight_position = highlight_position * highlight_position * (3.0 - 2.0 * highlight_position)
            shadow_position = torch.clamp((0.26 - luminance) / 0.26, 0.0, 1.0)
            shadow_position = shadow_position * shadow_position * (3.0 - 2.0 * shadow_position)
            protection = (1.0 - highlight_position * params.highlight_protection) * (1.0 - shadow_position * params.shadow_protection)
            requested_recovery = torch.clamp(recovery_amount * protection * (0.95 + 0.35 * (1.0 - confidence)), 0.0, 0.95)
            neutral_guard = params.color_protection * (1.0 - confidence) * (1.08 + 0.12 * params.naturalness) * protection
            correction_strength = torch.clamp(torch.maximum(requested_recovery, neutral_guard), 0.0, 0.95)
            natural = natural_luma.unsqueeze(2) + (natural_chroma * (1.0 - correction_strength.unsqueeze(2)) + target_chroma * correction_strength.unsqueeze(2))
            natural = _smooth_chroma_gamut_t(natural, torch)

        if params.local_contrast > 1e-6:
            luma = torch.sum(natural * weights, dim=2)
            contrast_luma = luma + 0.55 * params.local_contrast * (2.0 * luma - 1.0) * luma * (1.0 - luma)
            natural = natural * (contrast_luma / torch.clamp(luma, min=1e-4)).unsqueeze(2)

        highlight_position = torch.clamp((luminance - 0.58) / 0.40, 0.0, 1.0)
        highlight_position = highlight_position * highlight_position * (3.0 - 2.0 * highlight_position)
        highlight_blend = torch.clamp(highlight_position * params.highlight_protection, 0.0, 1.0)
        natural = natural * (1.0 - highlight_blend.unsqueeze(2)) + source * highlight_blend.unsqueeze(2)
        shadow_position = torch.clamp((0.26 - luminance) / 0.26, 0.0, 1.0)
        shadow_position = shadow_position * shadow_position * (3.0 - 2.0 * shadow_position)
        shadow_blend = torch.clamp(shadow_position * params.shadow_protection, 0.0, 1.0)
        natural = natural * (1.0 - shadow_blend.unsqueeze(2)) + source * shadow_blend.unsqueeze(2)
        natural = _apply_brightness_protection_t(source, natural, params.strength, params.brightness_protection, torch)

        saturation_threshold = 0.97
        saturation_limit = max(0.0, saturation_threshold - 1.5 / peak)
        final_luma = torch.sum(natural * weights, dim=2)
        luma_cap = torch.where(luminance < saturation_threshold, torch.tensor(saturation_limit, device=device), torch.tensor(1.0, device=device))
        luma_scale = torch.minimum(torch.ones_like(final_luma), luma_cap / torch.clamp(final_luma, min=1e-4))
        natural = natural * luma_scale.unsqueeze(2)
        channel_cap = torch.where(source < saturation_threshold, torch.tensor(saturation_limit, device=device), torch.tensor(1.0, device=device))
        if params.color_recovery > 1e-6:
            natural = _smooth_chroma_caps_t(natural, channel_cap, torch)
        else:
            natural = torch.minimum(natural, channel_cap)
        result = torch.nan_to_num(natural, nan=0.0, posinf=1.0, neginf=0.0)
        result = torch.clamp(torch.round(result * peak), 0.0, peak).to(dtype=torch.int32)
        return result.cpu().numpy().astype(image_rgb.dtype, copy=False)


def _opencl_dehaze(image_rgb: np.ndarray, params: "DehazeParams") -> np.ndarray:
    """Run the dominant atmospheric-recovery pass through OpenCV UMat.

    OpenCV exposes few portable reductions for UMat, so atmospheric-light and
    percentile statistics are intentionally computed on the CPU.  The costly
    per-pixel subtraction/division/blend is nevertheless executed as UMat
    kernels before the remaining scalar protection curves are applied.
    """
    import cv2  # noqa: PLC0415 - optional backend

    peak = float(np.iinfo(image_rgb.dtype).max)
    source = image_rgb.astype(np.float32) / peak
    flat = source.reshape(-1, 3)
    dark = np.min(flat, axis=1)
    count = min(dark.size, max(16, int(dark.size * 0.001)))
    indices = np.argpartition(dark, -count)[-count:]
    candidates = flat[indices]
    candidate_luma = candidates @ np.asarray(_WEIGHTS, dtype=np.float32)
    atmosphere = np.clip(np.median(candidates[np.argsort(candidate_luma)[-max(1, count // 8):]], axis=0), 0.35, 1.0).astype(np.float32)
    dark_reference = float(np.percentile(np.min(source, axis=2), 75))
    haze_level = float(np.clip((dark_reference - 0.03) / 0.92, 0.12, 0.92)) if np.isfinite(dark_reference) else 0.5
    omega = params.strength * (0.66 - 0.14 * params.fog_retention) * (0.90 + 0.10 * (1.0 - params.naturalness)) * (0.92 + 0.08 * haze_level)
    transmission = float(np.clip(1.0 - omega, 0.27 + 0.21 * params.fog_retention + 0.11 * params.naturalness, 1.0))
    neutral_mix = 0.45 * params.color_protection * (0.65 + 0.35 * params.naturalness)
    atmosphere = atmosphere * (1.0 - neutral_mix) + float(np.mean(atmosphere)) * neutral_mix
    atmosphere = np.clip(atmosphere, 0.35, 1.0)
    with _OPENCL_LOCK:
        cv2.ocl.setUseOpenCL(True)
        source_u = cv2.UMat(source)
        atmosphere_scalar = tuple(float(value) for value in atmosphere)
        recovered_u = cv2.add(
            cv2.divide(cv2.subtract(source_u, atmosphere_scalar), transmission),
            atmosphere_scalar,
        )
        # A scalar passed to cv2.min/max affects only the first channel of a
        # multi-channel UMat.  Split/merge keeps the clamp on the OpenCL device
        # without allocating two extra full-resolution RGB constant images.
        recovered_u = cv2.merge(
            [cv2.min(cv2.max(channel, 0.0), 1.0) for channel in cv2.split(recovered_u)]
        )
        blend = float(np.clip(params.strength * (0.78 - 0.28 * params.fog_retention) * (0.90 + 0.10 * (1.0 - params.naturalness)), 0.0, 0.82))
        natural_u = cv2.add(cv2.multiply(source_u, 1.0 - blend), cv2.multiply(recovered_u, blend))
        natural = natural_u.get()
    # Scalar protection curves are deliberately the same equations as the
    # reference path.  The atmospheric recovery above is the expensive,
    # dominant per-pixel pass and has already run in OpenCL.  Importing these
    # helpers here avoids a module cycle while keeping the OpenCL output's
    # colour-recovery, brightness and gamut semantics identical to CPU.
    from dehaze import (  # noqa: PLC0415
        _apply_brightness_protection,
        _smooth_chroma_caps,
        _smooth_chroma_gamut,
    )
    weights = np.asarray(_WEIGHTS, dtype=np.float32)
    luminance = source @ weights
    enhanced_luma = natural @ weights
    luma_only = np.clip(source * (enhanced_luma / np.maximum(luminance, 1e-4))[..., None], 0.0, 1.0)
    chroma_mix = params.color_protection * (0.72 + 0.28 * params.naturalness)
    natural = natural * (1.0 - chroma_mix) + luma_only * chroma_mix
    if params.color_recovery > 1e-6:
        source_chroma = source - luminance[..., None]
        source_chroma_norm = np.sqrt(np.sum(source_chroma * source_chroma, axis=2))
        confidence = np.clip((source_chroma_norm - 0.006) / 0.084, 0.0, 1.0)
        confidence = confidence * confidence * (3.0 - 2.0 * confidence)
        source_direction = source_chroma / np.maximum(source_chroma_norm[..., None], 1e-6)
        natural_luma = natural @ weights
        natural_chroma = natural - natural_luma[..., None]
        natural_chroma_norm = np.sqrt(np.sum(natural_chroma * natural_chroma, axis=2))
        aligned_chroma = np.sum(natural_chroma * source_direction, axis=2)
        aligned_chroma = np.maximum(aligned_chroma, 0.0)
        recovery_amount = params.color_recovery * params.strength
        source_target_norm = source_chroma_norm * (1.0 + 1.80 * params.color_recovery * confidence)
        natural_target_norm = natural_chroma_norm * (1.0 + 0.30 * recovery_amount * confidence) * confidence
        target_chroma_norm = np.maximum(np.maximum(aligned_chroma * confidence, natural_target_norm), source_target_norm)
        target_chroma = source_direction * target_chroma_norm[..., None]
        highlight_position = np.clip((luminance - 0.58) / 0.40, 0.0, 1.0)
        highlight_position = highlight_position * highlight_position * (3.0 - 2.0 * highlight_position)
        shadow_position = np.clip((0.26 - luminance) / 0.26, 0.0, 1.0)
        shadow_position = shadow_position * shadow_position * (3.0 - 2.0 * shadow_position)
        protection = (1.0 - highlight_position * params.highlight_protection) * (1.0 - shadow_position * params.shadow_protection)
        requested_recovery = np.clip(recovery_amount * protection * (0.95 + 0.35 * (1.0 - confidence)), 0.0, 0.95)
        neutral_guard = params.color_protection * (1.0 - confidence) * (1.08 + 0.12 * params.naturalness) * protection
        correction_strength = np.clip(np.maximum(requested_recovery, neutral_guard), 0.0, 0.95)
        natural = natural_luma[..., None] + (natural_chroma * (1.0 - correction_strength[..., None]) + target_chroma * correction_strength[..., None])
        natural = _smooth_chroma_gamut(natural)
    if params.local_contrast > 1e-6:
        luma = natural @ weights
        contrast_luma = luma + 0.55 * params.local_contrast * (2.0 * luma - 1.0) * luma * (1.0 - luma)
        natural *= (contrast_luma / np.maximum(luma, 1e-4))[..., None]
    highlight_position = np.clip((luminance - 0.58) / 0.40, 0.0, 1.0)
    highlight_position = highlight_position * highlight_position * (3.0 - 2.0 * highlight_position)
    natural = natural * (1.0 - np.clip(highlight_position * params.highlight_protection, 0.0, 1.0)[..., None]) + source * np.clip(highlight_position * params.highlight_protection, 0.0, 1.0)[..., None]
    shadow_position = np.clip((0.26 - luminance) / 0.26, 0.0, 1.0)
    shadow_position = shadow_position * shadow_position * (3.0 - 2.0 * shadow_position)
    natural = natural * (1.0 - np.clip(shadow_position * params.shadow_protection, 0.0, 1.0)[..., None]) + source * np.clip(shadow_position * params.shadow_protection, 0.0, 1.0)[..., None]
    natural = _apply_brightness_protection(source, natural, params.strength, params.brightness_protection)
    saturation_limit = max(0.0, 0.97 - 1.5 / peak)
    final_luma = natural @ weights
    natural *= np.minimum(1.0, np.where(luminance < 0.97, saturation_limit, 1.0) / np.maximum(final_luma, 1e-4))[..., None]
    channel_cap = np.where(source < 0.97, saturation_limit, 1.0)
    if params.color_recovery > 1e-6:
        natural = _smooth_chroma_caps(natural, channel_cap)
    else:
        natural = np.minimum(natural, channel_cap)
    return np.clip(np.rint(np.nan_to_num(natural, nan=0.0, posinf=1.0, neginf=0.0) * peak), 0, peak).astype(image_rgb.dtype)


def apply_gpu(image_rgb: np.ndarray, params: "DehazeParams", backend: str) -> np.ndarray:
    """Dispatch to a concrete backend; exceptions are handled by the caller."""
    if backend in ("cuda", "mps"):
        import torch  # noqa: PLC0415 - only loaded for an explicit torch backend

        return _torch_dehaze(image_rgb, params, backend, torch)
    if backend == "opencl":
        return _opencl_dehaze(image_rgb, params)
    raise ValueError(f"unsupported GPU backend: {backend}")
