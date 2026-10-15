from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehaze import DehazeParams, _apply_brightness_protection, apply_dehaze
from dehaze_spatial import _transmission_map


def test_tiny_strength_converges_to_unmodified_image():
    for dtype in (np.uint8, np.uint16):
        source = _synthetic_haze(dtype)
        result = apply_dehaze(source, DehazeParams(strength=1e-5), backend="cpu")
        assert int(np.max(np.abs(result.astype(np.int32) - source))) <= 2


def test_spatial_bright_guard_does_not_carve_a_ring_into_solar_glare():
    from dehaze_spatial import _protect_spatial_details

    for dtype in (np.uint8, np.uint16):
        peak = np.iinfo(dtype).max
        # A smooth warm solar gradient crosses the estimated airlight. Its
        # recovered brightness is excessive, but correction must not drag
        # an annulus below the original radiance or reverse its gradient.
        ramp = np.linspace(0.06, 0.60, 512, dtype=np.float32)
        rgb = ramp[None, :, None] * np.array([1.0, 0.85, 0.60])
        source = np.rint(np.broadcast_to(rgb, (32, 512, 3)) * peak).astype(dtype)
        recovered = np.rint(np.clip(source.astype(np.float32) * 1.35, 0, peak)).astype(dtype)
        untouched = source.copy()
        result = _protect_spatial_details(source, recovered, 1.0, np.zeros((32, 512)))
        weights = np.array([0.2126, 0.7152, 0.0722])
        before_y = source.astype(np.float32) @ weights
        after_y = result.astype(np.float32) @ weights
        assert float(np.max(np.abs(after_y - before_y))) <= 1.0
        assert float(np.min(np.diff(after_y[16]))) >= -1.0
        assert np.array_equal(source, untouched)


def test_bright_structure_support_rejects_smooth_solar_glare():
    from dehaze_spatial import _bright_structure_support

    yy, xx = np.mgrid[:320, :480]
    radius2 = (xx - 180) ** 2 + (yy - 100) ** 2
    glare = 0.08 + 0.92 * np.exp(-radius2 / (2 * 45.0 ** 2))
    rgb = glare[..., None] * np.array([1.0, 0.86, 0.65])
    for dtype in (np.uint8, np.uint16):
        source = np.rint(rgb * np.iinfo(dtype).max).astype(dtype)
        support = _bright_structure_support(source, 2)
        assert float(support.max()) < 0.01
        # An actual narrow, bright structure must still receive the guard.
        source[180:290, 360:363] = np.iinfo(dtype).max // 3
        support = _bright_structure_support(source, 2)
        assert float(np.median(support[200:270, 361])) > 0.9


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


def test_default_strength_is_zero():
    assert DehazeParams().strength == 0


def test_brightness_protection_is_part_of_normalized_cache_parameters():
    assert DehazeParams().normalized().brightness_protection == 0.70
    assert DehazeParams(brightness_protection=0).cache_token() != (
        DehazeParams(brightness_protection=1).cache_token()
    )


def test_brightness_protection_allows_natural_mid_tone_dimming():
    source = np.full((20, 20, 3), 0.42, dtype=np.float32)
    # 0.12 EV is below the strength=0.45 allowance of about 0.18 EV.
    current = source * (2.0 ** -0.12)
    protected = _apply_brightness_protection(source, current, 0.45, 0.70)
    assert np.allclose(protected, current, atol=1e-7)


def test_brightness_protection_lifts_abnormal_dimming_without_restoring_all():
    source = np.full((32, 32, 3), 0.20, dtype=np.float32)
    # This deliberately exaggerated result is about 1 EV darker than source.
    unprotected = source * 0.5
    protected = _apply_brightness_protection(source, unprotected, 0.45, 0.70)
    source_luma = float(source[0, 0, 0])
    unprotected_luma = float(unprotected[0, 0, 0])
    protected_luma = float(protected[0, 0, 0])
    assert protected_luma > unprotected_luma + 0.01
    assert protected_luma < source_luma - 0.01


def test_brightness_protection_has_max_ev_and_preserves_endpoints():
    source = np.full((12, 12, 3), 0.40, dtype=np.float32)
    source[0, 0] = (0.0, 0.0, 0.0)
    source[0, 1] = (1.0, 1.0, 1.0)
    source[0, 2] = (0.98, 0.99, 1.0)
    unprotected = source * 0.5
    unprotected[0, 0] = (0.0, 0.0, 0.0)
    unprotected[0, 1] = (1.0, 1.0, 1.0)
    unprotected[0, 2] = (0.98, 0.99, 1.0)
    protected = _apply_brightness_protection(source, unprotected, 0.45, 1.0)
    gain_ev = float(np.log2(float(protected[4, 4, 0]) / float(unprotected[4, 4, 0])))

    assert gain_ev <= 0.40 + 1e-6
    assert np.array_equal(protected[0, 0], unprotected[0, 0])
    assert np.array_equal(protected[0, 1], unprotected[0, 1])
    assert np.isfinite(protected).all()
    assert float(protected.max()) <= 1.0


def test_zero_brightness_protection_matches_unprotected_path():
    source = _synthetic_haze(np.uint8)
    unprotected = apply_dehaze(source, DehazeParams(brightness_protection=0))
    protected_off = apply_dehaze(source, DehazeParams(brightness_protection=0.0))
    assert np.array_equal(unprotected, protected_off)


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


def _chroma_spread(rgb: np.ndarray) -> float:
    values = rgb.astype(np.float32)
    return float((values.max(axis=2) - values.min(axis=2)).mean())


def test_neutral_haze_is_not_given_a_purple_or_green_cast():
    # The blue sky makes the global atmospheric-light estimate chromatic.  A
    # neutral patch must use its own low-chroma source as the colour anchor.
    source = np.full((80, 120, 3), 0.52, dtype=np.float32)
    source[:, :60] = (0.72, 0.82, 0.92)
    source = np.rint(source * 255).astype(np.uint8)

    for recovery in (0.35, 1.0):
        result = apply_dehaze(
            source,
            DehazeParams(strength=1.0, color_recovery=recovery, local_contrast=0),
        )
        neutral = result[40, 90].astype(np.int16)
        assert int(neutral.max() - neutral.min()) <= 1


def test_color_recovery_does_not_rotate_source_hue():
    source_rgb = np.array((0.60, 0.30, 0.20), dtype=np.float32)
    source = np.tile(np.rint(source_rgb * 255).astype(np.uint8), (40, 40, 1))
    result = apply_dehaze(
        source,
        DehazeParams(strength=1.0, color_recovery=1.0, local_contrast=0),
    )[20, 20].astype(np.float32) / 255.0

    source_chroma = source_rgb - float(source_rgb @ np.array((0.2126, 0.7152, 0.0722)))
    result_luma = float(result @ np.array((0.2126, 0.7152, 0.0722)))
    result_chroma = result - result_luma
    cosine = float(
        (source_chroma @ result_chroma)
        / (np.linalg.norm(source_chroma) * np.linalg.norm(result_chroma))
    )
    # Staying in the same broad colour sector is not enough: a red-orange
    # source must remain on effectively the same hue ray after recovery.
    assert cosine >= 0.995
    assert result[0] > result[1] > result[2]


def test_color_recovery_increases_trusted_midtone_chroma_without_clipping():
    source = np.full((32, 32, 3), (0.55, 0.30, 0.22), dtype=np.float32)
    source = np.rint(source * 255).astype(np.uint8)
    low = apply_dehaze(
        source,
        DehazeParams(strength=1.0, color_recovery=0.0, local_contrast=0),
    )
    high = apply_dehaze(
        source,
        DehazeParams(strength=1.0, color_recovery=1.0, local_contrast=0),
    )

    assert _chroma_spread(high) >= _chroma_spread(low) - 1.0 / 255.0
    # This source is safely below the highlight guard.  Recovery must not
    # hard-clip one channel to the dtype maximum or the saturation cap.
    assert int(high.max()) < int(0.97 * 255)


def test_low_saturation_sky_and_sea_show_visible_recovery_without_hue_rotation():
    # Representative blue-grey sky/sea values are deliberately below strong
    # saturation.  They must still expose the recovery slider at the default
    # strength, while remaining on the source pixel's hue ray.
    height, width = 120, 200
    scene = np.empty((height, width, 3), dtype=np.float32)
    scene[:70] = (0.46, 0.52, 0.58)
    scene[70:] = (0.43, 0.50, 0.54)
    scene[:70] += np.linspace(-0.015, 0.015, width, dtype=np.float32)[None, :, None]
    scene[70:] += np.array((0.015, 0.010, -0.005), dtype=np.float32)
    scene[:15] = (0.72, 0.78, 0.84)
    source = np.clip(np.rint(scene * 255), 0, 255).astype(np.uint8)

    low = apply_dehaze(
        source,
        DehazeParams(strength=0.45, color_recovery=0.0, local_contrast=0),
    )
    high = apply_dehaze(
        source,
        DehazeParams(strength=0.45, color_recovery=1.0, local_contrast=0),
    )
    low_float = low.astype(np.float32) / 255.0
    high_float = high.astype(np.float32) / 255.0
    source_float = source.astype(np.float32) / 255.0
    assert float(np.mean(np.abs(high_float - low_float))) >= 0.02
    assert _chroma_spread(high) >= _chroma_spread(low) * 1.15
    assert int(high.max()) < int(0.97 * 255)

    weights = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    source_chroma = source_float - (source_float @ weights)[..., None]
    result_chroma = high_float - (high_float @ weights)[..., None]
    denominator = np.linalg.norm(source_chroma, axis=2) * np.linalg.norm(result_chroma, axis=2)
    cosine = np.sum(source_chroma * result_chroma, axis=2) / np.maximum(denominator, 1e-6)
    assert float(np.min(cosine)) >= 0.995


def test_color_recovery_uint16_finite_params_and_spatial_consistency():
    source = _synthetic_haze(np.uint16)
    untouched = source.copy()
    result = apply_dehaze(
        source,
        DehazeParams(
            strength=1.0,
            color_recovery=float("inf"),
            local_contrast=0.5,
        ),
    )
    assert result.dtype == np.uint16
    assert np.isfinite(result).all()
    assert np.array_equal(source, untouched)

    # Equal source samples receive the same output even when placed at
    # different coordinates and beside different neighbouring content.
    target = np.array((0.32, 0.35, 0.38), dtype=np.float32)
    scene = np.zeros((80, 120, 3), dtype=np.float32)
    scene[:40] = (0.75, 0.84, 0.92)
    scene[40:] = (0.08, 0.10, 0.13)
    scene[15:23, 20:28] = target
    scene[37:45, 80:88] = target
    scene = np.rint(scene * 255).astype(np.uint8)
    equal = apply_dehaze(
        scene,
        DehazeParams(strength=1.0, color_recovery=1.0, local_contrast=0.7),
    )
    assert np.array_equal(equal[17:21, 22:26], equal[39:43, 82:86])

    # NaN is normalized to a disabled recovery rather than leaking into the
    # output; this also exercises the finite-value guard in parameter parsing.
    nan_result = apply_dehaze(source, DehazeParams(color_recovery=float("nan")))
    assert np.isfinite(nan_result).all()


def _synthetic_spatial_skyline(dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth sky, dark roofline, and bright facade touching the sky."""
    height, width = 900, 1200
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    image = np.empty((height, width, 3), dtype=np.float32)
    normalized_x = (x[0] - width / 2) / (width / 2)
    vignette = 1.0 - 0.22 * normalized_x * normalized_x
    for channel, base in enumerate((0.72, 0.83, 0.97)):
        image[:, :, channel] = (
            base - 0.00004 * y + 0.002 * np.sin(x / 33.0 + y / 41.0)
        )

    roof = (350 + 10 * np.sin(x[0] / 95.0) + 5 * np.sin(x[0] / 31.0)).astype(int)
    for column in range(width):
        image[roof[column]:, column, :] = (0.18, 0.21, 0.23)

    facade = image[:, 360:780, :]
    facade_mask = np.arange(height)[:, None] >= roof[360:780][None, :]
    facade[facade_mask] = (0.70, 0.77, 0.82)
    # Subtle windows verify that local detail does not make the whole facade
    # behave like an extension of the sky component.
    image[430:780:45, 390:750, :] = (0.50, 0.57, 0.61)
    image *= vignette[None, :, None]

    peak = np.iinfo(dtype).max
    return np.rint(np.clip(image, 0.0, 1.0) * peak).astype(dtype), roof


def test_spatial_sky_protection_avoids_bright_building_halo_for_8_and_16_bit():
    for dtype in (np.uint8, np.uint16):
        source, roof = _synthetic_spatial_skyline(dtype)
        transmission = _transmission_map(source, DehazeParams(strength=0.8))

        column = 550
        roof_row = int(roof[column])
        sky_near_roof = float(np.median(
            transmission[roof_row - 28:roof_row - 15, column - 10:column + 10]
        ))
        sky_at_silhouette = float(np.median(
            transmission[roof_row - 8:roof_row - 2, column - 10:column + 10]
        ))
        bright_facade = float(np.median(
            transmission[roof_row + 40:roof_row + 180, column - 20:column + 20]
        ))

        assert sky_near_roof > 0.95
        assert sky_at_silhouette > 0.82
        assert bright_facade < 0.75
        assert float(transmission[roof_row + 20:roof_row + 220, 380:760].max()) < 0.72
        # Darkened corners remain open sky; a global brightness threshold
        # would carve a disk out of the bright center and fail this check.
        assert float(np.percentile(transmission[100], 5)) > 0.90
        assert np.isfinite(transmission).all()
        assert float(transmission.min()) >= 0.0
        assert float(transmission.max()) <= 1.0

        # The skyline handoff is gradual and has no local bright spike.
        profile = transmission[roof_row - 18:roof_row + 22, column]
        steps = np.diff(profile)
        assert float(np.max(np.abs(steps))) < 0.10
        assert float(np.max(steps)) < 0.01
