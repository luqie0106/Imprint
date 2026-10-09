"""Small, bounded linear-RGB exposure adjustment for dehaze renders."""

from __future__ import annotations

import math

import cv2
import numpy as np


ALGORITHM_VERSION = "linear-median768-highlight095-ev2-v1"
TARGET_LUMA = 0.18
MAX_HIGHLIGHT = 0.95
MAX_EV = 2.0
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_REASONS = {"off", "target_reached", "highlight_limited", "applied", "black"}


def _validated_source_and_peak(image: np.ndarray) -> tuple[np.ndarray, float]:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("auto exposure requires an HxWx3 linear RGB array")
    if image.dtype != np.float32:
        raise TypeError("auto exposure requires float32 linear RGB")
    if image.size == 0:
        raise ValueError("auto exposure requires a non-empty image")
    from native_dense import rgb_peak
    from native_renderer import NativeRendererError
    try:
        return image, rgb_peak(image)
    except NativeRendererError:
        pass
    if not np.isfinite(image).all() or np.any(image < 0.0) or np.any(image > 1.0):
        raise ValueError("auto exposure input must be finite and in [0, 1]")
    return image, float(np.max(image))


def _validated_source(image: np.ndarray) -> np.ndarray:
    return _validated_source_and_peak(image)[0]


def estimate_auto_exposure(image: np.ndarray) -> tuple[float, str]:
    """Estimate one scalar EV from median linear luma and the actual RGB peak.

    Median luma is measured after area reduction to a longest edge of 768 px.
    The highlight bound uses the full input array so a small bright source is
    still protected when the median-analysis image is reduced.
    """
    source, peak = _validated_source_and_peak(image)
    return _estimate_auto_exposure_validated(source, peak)


def _estimate_auto_exposure_validated(source: np.ndarray, peak: float) -> tuple[float, str]:
    height, width = source.shape[:2]
    longest = max(height, width)
    if longest > 768:
        scale = 768.0 / longest
        analysis = cv2.resize(
            source,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        analysis = source
    luma = analysis @ _LUMA_WEIGHTS
    median_luma = float(np.median(luma))

    if median_luma <= 1e-8:
        return 0.0, "black"
    target_ev = math.log2(TARGET_LUMA / median_luma)
    if target_ev <= 0.0:
        return 0.0, "target_reached"

    if peak <= 0.0:
        highlight_ev = MAX_EV
    else:
        highlight_ev = math.log2(MAX_HIGHLIGHT / peak)
    if highlight_ev <= 1e-8:
        return 0.0, "highlight_limited"

    ev = max(0.0, min(target_ev, highlight_ev, MAX_EV))
    if ev <= 1e-8:
        return 0.0, "target_reached"
    if highlight_ev < min(target_ev, MAX_EV) - 1e-7:
        return float(ev), "highlight_limited"
    return float(ev), "applied"


def apply_auto_exposure(
    image: np.ndarray,
    *,
    enabled: bool = True,
    ev_override: float | None = None,
    reason_override: str | None = None,
    diagnostics: dict | None = None,
) -> np.ndarray:
    """Return a new array with bounded scalar exposure; never mutate ``image``."""
    source, peak = _validated_source_and_peak(image)
    if not enabled:
        ev, reason = 0.0, "off"
    elif ev_override is None:
        ev, reason = _estimate_auto_exposure_validated(source, peak)
    else:
        override = float(ev_override)
        if not math.isfinite(override) or not 0.0 <= override <= MAX_EV:
            raise ValueError("auto exposure EV override must be finite and in [0, 2]")
        if reason_override is not None and reason_override not in _REASONS - {"off"}:
            raise ValueError("invalid auto exposure reason override")
        ev, reason = override, reason_override or ("applied" if override > 0 else "target_reached")
        if peak >= MAX_HIGHLIGHT:
            safe_ev = 0.0
        elif peak > 0.0:
            safe_ev = max(0.0, math.log2(MAX_HIGHLIGHT / peak))
        else:
            safe_ev = MAX_EV
        if safe_ev < ev - 1e-7:
            ev, reason = safe_ev, "highlight_limited"

    # The standalone C++ gain kernel passed parity but was slower than NumPy
    # once its mandatory validation scan was included. Keep the faster path.
    result = source.copy()
    if ev > 0.0:
        np.multiply(result, np.float32(2.0 ** ev), out=result)
    if diagnostics is not None:
        diagnostics.update(auto_exposure_ev=float(ev), auto_exposure_reason=reason)
    return result
