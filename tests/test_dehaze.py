from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehaze import DehazeParams, apply_dehaze


def _synthetic_haze(dtype: np.dtype) -> np.ndarray:
    peak = np.iinfo(dtype).max
    base = np.zeros((120, 180, 3), dtype=np.float32)
    base[:, :90] = (0.18, 0.25, 0.20)
    base[:, 90:] = (0.62, 0.72, 0.58)
    base[35:85, 60:120] *= 0.55
    hazy = base * 0.46 + 0.54
    return np.clip(np.rint(hazy * peak), 0, peak).astype(dtype)


def test_shape_channels_dtype_range_and_no_mutation():
    source = _synthetic_haze(np.uint8)
    untouched = source.copy()
    result = apply_dehaze(source, DehazeParams())
    assert result.shape == source.shape
    assert result.dtype == np.uint8
    assert int(result.min()) >= 0 and int(result.max()) <= 255
    assert np.array_equal(source, untouched)
    assert np.isfinite(result).all()


def test_zero_strength_is_exact_copy():
    source = _synthetic_haze(np.uint8)
    result = apply_dehaze(source, DehazeParams(strength=0))
    assert np.array_equal(result, source)
    assert result is not source


def test_synthetic_haze_local_contrast_improves():
    source = _synthetic_haze(np.uint8)
    result = apply_dehaze(source, DehazeParams(strength=0.7, local_contrast=0.4))
    before = float(source[:, 90:].mean() - source[:, :90].mean())
    after = float(result[:, 90:].mean() - result[:, :90].mean())
    assert after > before


def test_uint16_supported_without_nan_or_inf():
    source = _synthetic_haze(np.uint16)
    result = apply_dehaze(source, DehazeParams())
    assert result.dtype == np.uint16
    assert result.shape == source.shape
    assert np.isfinite(result).all()


def _synthetic_skyline(dtype: np.dtype) -> np.ndarray:
    """A bright sky over a dark building with a hard, high-contrast edge."""
    peak = np.iinfo(dtype).max
    image = np.empty((160, 240, 3), dtype=np.float32)
    image[:80] = (0.78, 0.84, 0.92)
    image[80:] = (0.10, 0.12, 0.15)
    image[65:80] = (0.70, 0.76, 0.84)
    return np.rint(image * peak).astype(dtype)


def test_local_contrast_does_not_create_skyline_halo():
    source = _synthetic_skyline(np.uint8)
    without_local_contrast = apply_dehaze(
        source,
        DehazeParams(strength=0.45, local_contrast=0),
    )
    with_local_contrast = apply_dehaze(
        source,
        DehazeParams(strength=0.45, local_contrast=1.0),
    )

    # A global monotonic tone curve must not overshoot either plateau at the
    # skyline.  This catches the bright/dark halos made by local sharpening.
    before_profile = without_local_contrast.mean(axis=(1, 2)).astype(np.float32)
    after_profile = with_local_contrast.mean(axis=(1, 2)).astype(np.float32)
    edge_band = after_profile[76:84]
    plateaus = np.concatenate((after_profile[20:60], after_profile[100:140]))
    assert float(edge_band.max()) <= float(plateaus.max()) + 1.0
    assert float(edge_band.min()) >= float(plateaus.min()) - 1.0
    assert float(np.percentile(after_profile, 90) - np.percentile(after_profile, 10)) > \
        float(np.percentile(before_profile, 90) - np.percentile(before_profile, 10))


def _synthetic_night_skyline(dtype: np.dtype) -> np.ndarray:
    """Hazy blue night sky over a dark building, with isolated lights."""
    peak = np.iinfo(dtype).max
    scene = np.zeros((180, 280, 3), dtype=np.float32)
    scene[:100] = (0.22, 0.28, 0.38)
    scene[:100] += np.linspace(-0.03, 0.03, 280, dtype=np.float32)[None, :, None]
    scene[100:] = (0.055, 0.065, 0.085)
    scene[100:130] = (0.07, 0.08, 0.10)
    for y, x in ((118, 40), (124, 82), (112, 150), (135, 205), (156, 260)):
        scene[y:y + 5, x:x + 5] = (0.32, 0.28, 0.18)
    # Add a broad veil so the test exercises transmission recovery instead of
    # merely sharpening an already-contrasty synthetic edge.
    hazy = scene * 0.55 + 0.45
    return np.clip(np.rint(hazy * peak), 0, peak).astype(dtype)


def test_global_strength_has_no_skyline_halo_and_high_strength_is_stronger():
    source = _synthetic_night_skyline(np.uint8)
    low = apply_dehaze(source, DehazeParams(strength=0.35, local_contrast=0))
    high = apply_dehaze(source, DehazeParams(strength=1.0, local_contrast=0))

    source_profile = source.mean(axis=(1, 2)).astype(np.float32)
    low_profile = low.mean(axis=(1, 2)).astype(np.float32)
    high_profile = high.mean(axis=(1, 2)).astype(np.float32)
    low_change = np.abs(low_profile - source_profile)
    high_change = np.abs(high_profile - source_profile)

    # Global recovery is allowed to change the skyline itself, but may not
    # create a new value outside the transformed sky/building plateaus.
    skyline_band = high_profile[96:106]
    plateaus = np.concatenate((high_profile[20:80], high_profile[140:175]))
    assert float(skyline_band.max()) <= float(plateaus.max()) + 1.0
    assert float(skyline_band.min()) >= float(plateaus.min()) - 1.0
    assert float(high_change.mean()) > float(low_change.mean()) * 1.5


def test_equal_rgb_has_spatially_consistent_result():
    source = np.full((120, 220, 3), 0.38, dtype=np.float32)
    source[:60] = (0.78, 0.84, 0.92)
    source[60:] = (0.08, 0.10, 0.13)
    target = np.array((0.32, 0.35, 0.38), dtype=np.float32)
    # One target patch is surrounded by a flat field, the other touches a
    # high-contrast edge.  Their equal RGB samples must remain equal after the
    # global transform, independent of nearby structure.
    source[20:32, 25:37] = target
    source[54:66, 140:152] = target
    source = np.rint(source * 255).astype(np.uint8)

    result = apply_dehaze(source, DehazeParams(strength=1.0, local_contrast=0.7))
    flat_patch = result[24:28, 29:33].astype(np.int16)
    edge_patch = result[58:62, 144:148].astype(np.int16)
    assert int(np.abs(flat_patch - edge_patch).max()) <= 1


def test_high_strength_changes_entire_scene_more_than_low_strength():
    source = _synthetic_haze(np.uint8)
    low = apply_dehaze(source, DehazeParams(strength=0.2, local_contrast=0))
    high = apply_dehaze(source, DehazeParams(strength=1.0, local_contrast=0))
    source_luma = source.mean(axis=2).astype(np.float32)
    low_change = np.abs(low.mean(axis=2).astype(np.float32) - source_luma).mean()
    high_change = np.abs(high.mean(axis=2).astype(np.float32) - source_luma).mean()
    assert float(high_change) > float(low_change) * 2.0


def _synthetic_backlit_sun(dtype: np.dtype) -> np.ndarray:
    """Smooth bright haze around a saturated sun above a dark city."""
    peak = np.iinfo(dtype).max
    height, width = 220, 320
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    sky = np.empty((height, width, 3), dtype=np.float32)
    sky[:] = (0.66, 0.72, 0.82)
    sky += (yy / height)[..., None] * np.array((0.08, 0.06, 0.03), dtype=np.float32)
    distance = np.sqrt((xx - 160.0) ** 2 + (yy - 62.0) ** 2)
    haze = np.exp(-((distance / 54.0) ** 2))[..., None]
    sky += haze * np.array((0.17, 0.13, 0.08), dtype=np.float32)
    image = sky
    image[105:] = (0.055, 0.065, 0.085)
    image[105:135] = (0.09, 0.10, 0.12)
    for left, top, right in ((12, 82, 48), (58, 66, 96), (108, 92, 142), (172, 74, 211), (232, 86, 278), (286, 60, 317)):
        image[top:105, left:right] = (0.07, 0.08, 0.10)
    sun = distance <= 12.0
    image[sun] = (1.0, 0.995, 0.97)
    return np.clip(np.rint(image * peak), 0, peak).astype(dtype)


def test_backlit_sun_highlight_area_does_not_expand():
    source = _synthetic_backlit_sun(np.uint8)
    result = apply_dehaze(source, DehazeParams(strength=0.95, local_contrast=1.0))
    weights = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    source_luma = source.astype(np.float32) / 255.0 @ weights
    result_luma = result.astype(np.float32) / 255.0 @ weights

    # The smooth highlight rolloff must not turn surrounding veil pixels into
    # new near-saturated pixels; one quantisation level of slack is allowed.
    source_near_sat = int(np.count_nonzero(source_luma >= 0.97))
    result_near_sat = int(np.count_nonzero(result_luma >= 0.97))
    assert result_near_sat <= source_near_sat

    ring = (distance := np.sqrt((np.indices(source.shape[:2])[1] - 160.0) ** 2 +
                                (np.indices(source.shape[:2])[0] - 62.0) ** 2))
    ring = (ring >= 18.0) & (ring <= 52.0)
    assert int(np.count_nonzero(result_luma[ring] >= 0.97)) <= int(np.count_nonzero(source_luma[ring] >= 0.97))


def test_high_strength_remains_effective_with_large_dark_city():
    source = _synthetic_backlit_sun(np.uint8)
    low = apply_dehaze(source, DehazeParams(strength=0.25, local_contrast=0))
    high = apply_dehaze(source, DehazeParams(strength=0.90, local_contrast=0))
    source_luma = source.astype(np.float32).mean(axis=2) / 255.0
    low_change = np.abs(low.astype(np.float32).mean(axis=2) / 255.0 - source_luma)
    high_change = np.abs(high.astype(np.float32).mean(axis=2) / 255.0 - source_luma)
    midtones = (source_luma >= 0.10) & (source_luma <= 0.85)

    assert float(high_change[midtones].mean()) > float(low_change[midtones].mean()) * 1.5
    assert float(high_change[midtones].mean()) >= 0.025


def test_dark_majority_cannot_suppress_high_strength():
    source = np.full((180, 280, 3), (0.27, 0.29, 0.32), dtype=np.float32)
    source[:32] = (0.68, 0.72, 0.78)
    source[60:150, 35:95] = (0.20, 0.22, 0.25)
    source[75:160, 125:205] = (0.23, 0.25, 0.28)
    source = np.rint(source * 255).astype(np.uint8)

    low = apply_dehaze(source, DehazeParams(strength=0.2, local_contrast=0))
    high = apply_dehaze(source, DehazeParams(strength=0.9, local_contrast=0))
    source_luma = source.astype(np.float32).mean(axis=2) / 255.0
    low_change = np.abs(low.astype(np.float32).mean(axis=2) / 255.0 - source_luma)
    high_change = np.abs(high.astype(np.float32).mean(axis=2) / 255.0 - source_luma)
    midtones = (source_luma >= 0.18) & (source_luma <= 0.80)

    assert float(high_change[midtones].mean()) > float(low_change[midtones].mean()) * 2.0
    assert float(high_change[midtones].mean()) >= 0.04
