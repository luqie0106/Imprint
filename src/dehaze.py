"""Spatially consistent dehazing for preview and full-resolution RGB images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import threading

import numpy as np


ALGORITHM_VERSION = "natural-global-v8-source-hue-brightness-guard"


@dataclass(frozen=True)
class DehazeParams:
    strength: float = 0.0
    naturalness: float = 0.70
    fog_retention: float = 0.55
    local_contrast: float = 0.25
    color_recovery: float = 0.35
    color_protection: float = 0.80
    highlight_protection: float = 0.75
    shadow_protection: float = 0.75
    brightness_protection: float = 0.70

    def normalized(self) -> "DehazeParams":
        values = {}
        for key, value in asdict(self).items():
            value = float(value)
            if not np.isfinite(value):
                value = 1.0 if value > 0.0 else 0.0
            values[key] = float(np.clip(value, 0.0, 1.0))
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


def _smooth_chroma_gamut(image: np.ndarray) -> np.ndarray:
    """Keep a per-pixel chroma vector inside RGB gamut without channel cuts.

    The luminance is kept fixed while the complete chroma vector is scaled by
    one smooth factor.  This is intentionally a scalar operation on each
    pixel: it preserves the chroma direction and cannot create a colour seam
    at an image edge.
    """
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luminance = image @ weights
    chroma = image - luminance[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        positive_room = np.where(chroma > 1e-7, (1.0 - luminance[..., None]) / chroma, np.inf)
        negative_room = np.where(chroma < -1e-7, luminance[..., None] / (-chroma), np.inf)
    limit = np.min(np.minimum(positive_room, negative_room), axis=2)
    limit = np.where(np.isfinite(limit), limit, 1.0)
    # Smooth-min(1, limit), with a very small epsilon so the unconstrained
    # case remains numerically indistinguishable from an identity transform.
    scale = 0.5 * (1.0 + limit - np.sqrt((1.0 - limit) ** 2 + 1e-10))
    scale = np.clip(scale, 0.0, 1.0)
    return luminance[..., None] + chroma * scale[..., None]


def _smooth_chroma_caps(image: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Scale chroma smoothly to per-channel caps while retaining luminance."""
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luminance = image @ weights
    chroma = image - luminance[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        positive_room = np.where(chroma > 1e-7, (caps - luminance[..., None]) / chroma, np.inf)
        negative_room = np.where(chroma < -1e-7, luminance[..., None] / (-chroma), np.inf)
    limit = np.min(np.minimum(positive_room, negative_room), axis=2)
    limit = np.where(np.isfinite(limit), limit, 1.0)
    scale = 0.5 * (1.0 + limit - np.sqrt((1.0 - limit) ** 2 + 1e-10))
    scale = np.clip(scale, 0.0, 1.0)
    return luminance[..., None] + chroma * scale[..., None]


def _apply_brightness_protection(
    source: np.ndarray,
    image: np.ndarray,
    strength: float,
    protection: float,
) -> np.ndarray:
    """Compensate for an unusually large global midtone brightness loss.

    The comparison is deliberately made on one source-luminance-selected
    pixel set and uses medians rather than a whole-image mean.  This keeps a
    bright sky or a large dark foreground from controlling the guard.  A
    single compensation EV is then applied to every pixel, so equal RGB
    values remain spatially consistent and no local seam can be introduced.
    """
    if protection <= 1e-6:
        return image

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    source_luma = source @ weights
    image_luma = image @ weights
    valid = (
        np.isfinite(source_luma)
        & np.isfinite(image_luma)
        & (source_luma > 0.08)
        & (source_luma < 0.88)
    )
    if int(np.count_nonzero(valid)) < 8:
        return image

    source_median = float(np.median(source_luma[valid]))
    image_median = float(np.median(image_luma[valid]))
    if (
        not np.isfinite(source_median)
        or not np.isfinite(image_median)
        or source_median <= 1e-5
        or image_median <= 1e-5
    ):
        return image

    brightness_drop_ev = float(np.log2(source_median / image_median))
    if not np.isfinite(brightness_drop_ev):
        return image
    allowed_drop_ev = 0.08 + 0.22 * float(np.clip(strength, 0.0, 1.0))
    excess_drop_ev = max(0.0, brightness_drop_ev - allowed_drop_ev)
    compensation_ev = min(0.40, excess_drop_ev * float(np.clip(protection, 0.0, 1.0)))
    if not np.isfinite(compensation_ev) or compensation_ev <= 1e-6:
        return image

    gain = float(2.0 ** compensation_ev)
    if not np.isfinite(gain):
        return image

    # This fixed-endpoint curve is monotonic and raises midtones while
    # leaving exact black and white unchanged.  Convert the luma change into
    # one scalar per pixel and apply it to all RGB channels to preserve hue.
    curve_luma = image_luma * gain / (1.0 + (gain - 1.0) * image_luma)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scale = np.divide(
            curve_luma,
            image_luma,
            out=np.ones_like(image_luma),
            where=image_luma > 1e-6,
        )
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    protected = image * scale[..., None]
    protected = np.nan_to_num(protected, nan=0.0, posinf=1.0, neginf=0.0)
    protected = np.clip(protected, 0.0, 1.0)
    return _smooth_chroma_gamut(protected)


def _apply_dehaze_cpu(image_rgb: np.ndarray, params: DehazeParams | None = None) -> np.ndarray:
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

    if p.color_recovery > 1e-6:
        # Anchor only the added colour recovery to the source pixel's own
        # chroma direction.  A low-chroma source has no reliable hue, so its
        # recovery target smoothly approaches neutral instead of inheriting a
        # purple/green cast from the atmospheric-light estimate.
        source_chroma = source - luminance[..., None]
        source_chroma_norm = np.sqrt(np.sum(source_chroma * source_chroma, axis=2))
        # Low-saturation blue-grey skies and yellow-grey water still carry a
        # useful source hue.  Keep only a very small neutral dead-zone, then
        # ramp confidence over a narrow range so the recovery slider remains
        # visible before the source becomes strongly saturated.
        confidence = np.clip((source_chroma_norm - 0.006) / 0.084, 0.0, 1.0)
        confidence = confidence * confidence * (3.0 - 2.0 * confidence)
        source_direction = source_chroma / np.maximum(source_chroma_norm[..., None], 1e-6)

        natural_luma = natural @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        natural_chroma = natural - natural_luma[..., None]
        natural_chroma_norm = np.sqrt(np.sum(natural_chroma * natural_chroma, axis=2))

        # The target remains on the source hue line.  Retaining at least the
        # aligned current chroma avoids making a strongly coloured source look
        # flatter as recovery is increased; the small boost is the actual
        # conservative colour-recovery contribution.
        aligned_chroma = np.sum(natural_chroma * source_direction, axis=2)
        aligned_chroma = np.maximum(aligned_chroma, 0.0)
        recovery_amount = p.color_recovery * p.strength
        # The strength slider already gates how much of this target is mixed
        # below.  Do not multiply it into the target gain a second time: that
        # made 100% colour recovery nearly indistinguishable from 0% at the
        # default strength.  Confidence attenuates the natural/atmospheric
        # term, while the source chroma itself remains the stable hue anchor;
        # its magnitude is already tiny for a near-neutral pixel.
        source_target_norm = source_chroma_norm * (1.0 + 1.80 * p.color_recovery * confidence)
        natural_target_norm = natural_chroma_norm * (
            1.0 + 0.30 * recovery_amount * confidence
        ) * confidence
        target_chroma_norm = np.maximum(
            np.maximum(aligned_chroma * confidence, natural_target_norm),
            source_target_norm,
        )
        target_chroma = source_direction * target_chroma_norm[..., None]

        # High-light and shadow protection reduce chroma recovery smoothly;
        # they use only the source pixel's luminance and cannot form seams.
        highlight_position = np.clip((luminance - 0.58) / 0.40, 0.0, 1.0)
        highlight_position = highlight_position * highlight_position * (3.0 - 2.0 * highlight_position)
        shadow_position = np.clip((0.26 - luminance) / 0.26, 0.0, 1.0)
        shadow_position = shadow_position * shadow_position * (3.0 - 2.0 * shadow_position)
        protection = (
            (1.0 - highlight_position * p.highlight_protection)
            * (1.0 - shadow_position * p.shadow_protection)
        )
        requested_recovery = np.clip(
            recovery_amount * protection * (0.95 + 0.35 * (1.0 - confidence)),
            0.0,
            0.95,
        )
        # Neutral pixels need protection even at the conservative default
        # recovery setting.  Otherwise a chromatic atmosphere estimate can
        # leave a faint invented cast unless the user turns recovery to 100.
        # This guard only removes unsupported chroma; it never adds a hue.
        neutral_guard = (
            p.color_protection
            * (1.0 - confidence)
            * (1.08 + 0.12 * p.naturalness)
            * protection
        )
        correction_strength = np.maximum(requested_recovery, neutral_guard)
        correction_strength = np.clip(correction_strength, 0.0, 0.95)
        natural = natural_luma[..., None] + (
            natural_chroma * (1.0 - correction_strength[..., None])
            + target_chroma * correction_strength[..., None]
        )
        # Limit the recovery by scaling the complete chroma vector, rather
        # than clipping individual channels and rotating the source hue.
        natural = _smooth_chroma_gamut(natural)

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

    # Protect against a global, abnormal darkening introduced by the combined
    # recovery/tone operations.  This runs before the final near-saturation
    # guard and is intentionally disabled at zero for legacy compatibility.
    natural = _apply_brightness_protection(
        source,
        natural,
        p.strength,
        p.brightness_protection,
    )

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
    if p.color_recovery > 1e-6:
        natural = _smooth_chroma_caps(natural, channel_cap)
    else:
        # Keep the legacy zero-recovery path byte-for-byte compatible.
        natural = np.minimum(natural, channel_cap)

    result = np.nan_to_num(natural, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.rint(result * peak), 0, peak).astype(image_rgb.dtype)

# GPU backend selection is deliberately kept at this module boundary.  The
# CPU implementation above remains the compatibility reference and is also
# the unconditional safety net when an optional accelerator is unavailable or
# raises during execution.
_BACKEND_NAMES = frozenset(("auto", "cpu", "cuda", "mps", "opencl"))
_BACKEND_LOCK = threading.RLock()
_GPU_BACKENDS = ("cuda", "mps", "opencl")
_GPU_BACKEND_LABELS = {
    "cuda": "NVIDIA CUDA 可用",
    "mps": "Apple MPS 可用",
    "opencl": "OpenCL 可用",
}


def _requested_backend() -> str:
    value = os.environ.get("IMPRINT_DEHAZE_BACKEND", "auto").strip().lower()
    return value if value in _BACKEND_NAMES else "auto"


def _torch_backend_available(backend: str) -> bool:
    """Check a torch backend without importing torch on CPU/OpenCL hosts."""
    try:
        import torch  # noqa: PLC0415 - optional dependency, intentionally lazy
    except Exception:
        return False
    try:
        if backend == "cuda":
            return bool(torch.cuda.is_available())
        if backend == "mps":
            return bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            )
    except Exception:
        return False
    return False


def _opencl_backend_available() -> bool:
    """Return whether OpenCV can actually schedule UMat work on OpenCL."""
    try:
        import cv2  # noqa: PLC0415 - optional accelerator probe
    except Exception:
        return False
    try:
        if not bool(cv2.ocl.haveOpenCL()):
            return False
        # OpenCV's default is often disabled even when a platform is present.
        # This is a process-wide switch, so probing and execution are guarded by
        # _BACKEND_LOCK to avoid races with concurrent preview/batch requests.
        cv2.ocl.setUseOpenCL(True)
        return bool(cv2.ocl.useOpenCL())
    except Exception:
        return False


def _backend_available(backend: str) -> bool:
    if backend in ("cuda", "mps"):
        return _torch_backend_available(backend)
    if backend == "opencl":
        return _opencl_backend_available()
    return backend == "cpu"


def get_gpu_status() -> dict[str, object]:
    """Return a path-free snapshot of the available dehaze accelerators.

    Backend probing can touch process-wide OpenCL state, so all three probes
    are serialized with the same lock used by backend selection.  The result
    intentionally contains only stable backend identifiers and display text;
    device paths and driver details are never exposed through the API.
    """
    with _BACKEND_LOCK:
        available = [backend for backend in _GPU_BACKENDS if _backend_available(backend)]
    recommended = available[0] if available else None
    return {
        "available": bool(available),
        "backends": available,
        "recommended": recommended,
        "label": _GPU_BACKEND_LABELS[recommended] if recommended else "未检测到可用 GPU",
    }


# Keep a descriptive alias for callers that prefer an explicit query verb.
query_gpu_status = get_gpu_status


def _backend_candidates(requested: str) -> tuple[str, ...]:
    if requested == "cpu":
        return ("cpu",)
    if requested in ("cuda", "mps", "opencl"):
        return (requested, "cpu")
    # Keep this order stable: CUDA first, Apple MPS second, OpenCL third.
    return ("cuda", "mps", "opencl", "cpu")


def _run_gpu_backend(
    image_rgb: np.ndarray,
    params: DehazeParams,
    backend: str,
) -> np.ndarray:
    # Importing the adapter lazily is important: the ordinary CPU sidecar must
    # start even when torch is not installed, and OpenCL is optional too.
    from dehaze_gpu import apply_gpu  # noqa: PLC0415

    return apply_gpu(image_rgb, params, backend)


def apply_dehaze(
    image_rgb: np.ndarray,
    params: DehazeParams | None = None,
    *,
    backend: str | None = None,
) -> np.ndarray:
    """Dehaze an RGB image, preferring an available accelerator safely.

    ``IMPRINT_DEHAZE_BACKEND`` may be ``auto``, ``cpu``, ``cuda``, ``mps`` or
    ``opencl``.  An explicit ``backend`` takes precedence over the environment
    variable. Explicit accelerator requests are still safe: an unavailable
    device or a runtime failure falls back to the unchanged CPU algorithm.
    """
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a numpy array")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if image_rgb.dtype not in (np.uint8, np.uint16):
        raise TypeError("image_rgb must use uint8 or uint16 samples")

    normalized = (params or DehazeParams()).normalized()
    if normalized.strength <= 1e-6 or image_rgb.size == 0:
        return image_rgb.copy()

    if backend is None:
        requested = _requested_backend()
    elif isinstance(backend, str):
        normalized_backend = backend.strip().lower()
        requested = normalized_backend if normalized_backend in _BACKEND_NAMES else "auto"
    else:
        requested = "auto"
    # Do not cache this decision: tests and desktop sessions can change the
    # environment between calls, and device availability can change at runtime.
    for backend in _backend_candidates(requested):
        if backend == "cpu":
            return _apply_dehaze_cpu(image_rgb, normalized)
        try:
            with _BACKEND_LOCK:
                if not _backend_available(backend):
                    continue
                return _run_gpu_backend(image_rgb, normalized, backend)
        except Exception:
            # A driver/context failure must never make preview or export fail.
            # Try the next auto candidate, or the CPU reference for explicit
            # requests.  The exception is intentionally not logged here because
            # this function is called for every preview tile and paths may be
            # sensitive; callers can instrument their own backend adapter.
            continue

    return _apply_dehaze_cpu(image_rgb, normalized)
