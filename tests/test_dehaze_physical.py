from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dehaze import DehazeParams
from dehaze_physical import (apply_physical_dehaze, _physical_pixels, _estimate_airlight,
                             _estimate_broad_airlight, physical_diagnostics)

LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)


def scene():
    rng = np.random.default_rng(20261002)
    image = np.empty((96, 144, 3), dtype=np.float32)
    image[:48] = [.62, .68, .74]
    image[48:] = [.25, .32, .29]
    image[65:85, 25:90] *= .7
    image += rng.uniform(-.002, .002, image.shape).astype(np.float32)
    return image


def broad_smoke_scene(smoke_rows=(100, 165)):
    """Dark upper field plus a wide, structured haze field at any height."""
    image = np.full((192, 160, 3), [.010, .011, .012], dtype=np.float32)
    start, stop = smoke_rows
    x = np.arange(24, 136)
    reflectance = np.where((x // 18) % 2, .035, .014).astype(np.float32)
    clear = np.broadcast_to(reflectance[None, :, None], (stop - start, x.size, 3))
    atmosphere = np.array([.14, .15, .16], dtype=np.float32)
    image[start:stop, 24:136] = .45 * clear + .55 * atmosphere
    return image


def relative_structure_contrast(image, rows=(100, 165)):
    start, stop = rows
    x = np.arange(24, 136)
    low = image[start:stop, 24:136][..., 0][((x // 18) % 2 == 0)[None, :].repeat(stop - start, axis=0)]
    high = image[start:stop, 24:136][..., 0][((x // 18) % 2 == 1)[None, :].repeat(stop - start, axis=0)]
    low_mean, high_mean = float(np.mean(low)), float(np.mean(high))
    return (high_mean - low_mean) / max((high_mean + low_mean) * .5, 1e-6)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("spatial", [False, True])
def test_dtype_shape_zero_range_and_immutable(dtype, spatial):
    source = scene()
    if dtype != np.float32:
        source = np.rint(source * np.iinfo(dtype).max).astype(dtype)
    original = source.copy()
    zero = apply_physical_dehaze(source, DehazeParams(strength=0), spatial=spatial)
    result = apply_physical_dehaze(source, DehazeParams(strength=.8), spatial=spatial)
    assert zero is not source and np.array_equal(zero, source)
    assert result.shape == source.shape and result.dtype == source.dtype
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= (1 if dtype == np.float32 else np.iinfo(dtype).max)
    assert np.array_equal(source, original)


@pytest.mark.parametrize("spatial", [False, True])
def test_strength_continuity_and_visibility_without_sharpening(spatial):
    source = scene()
    changes = []
    for strength in (0, 1e-5, .01, .1, .35, .7, 1):
        result = apply_physical_dehaze(source, DehazeParams(strength=strength, local_contrast=0, color_recovery=0), spatial=spatial)
        changes.append(float(np.mean(np.abs(result - source))))
    assert changes[0] == 0
    assert changes[1] < 5e-6
    assert np.min(np.diff(changes)) >= 0
    assert changes[-1] > .015
    # Genuine contrast across a veil improves without a local detail pass.
    result = apply_physical_dehaze(source, DehazeParams(strength=.7, local_contrast=0, color_recovery=0), spatial=spatial)
    before = float((source[:40] @ LUMA).mean() - (source[65:85, 25:90] @ LUMA).mean())
    after = float((result[:40] @ LUMA).mean() - (result[65:85, 25:90] @ LUMA).mean())
    assert after > before


def test_negative_channels_are_prevented_before_clipping():
    rng = np.random.default_rng(5)
    source = rng.uniform(0, 1, (80, 120, 3)).astype(np.float32)
    source[:5] *= .001
    params = DehazeParams(strength=1, local_contrast=0, color_recovery=0, color_protection=0)
    result = _physical_pixels(source, np.full(source.shape[:2], .01, dtype=np.float32),
                              np.array([.85, .68, .52], dtype=np.float32), params)
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1
    assert np.all(np.any(result > 0, axis=2))
    # True black-level shadows retain almost all of their source luminance;
    # hazy midtones are tested separately and may receive a stronger inverse.
    assert np.min((result[:5] @ LUMA) / (source[:5] @ LUMA)) >= .98


def test_noise_amplification_is_bounded_by_inverse_gain():
    x = np.linspace(.35, .45, 2048, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    params = DehazeParams(strength=1, local_contrast=0, color_recovery=0)
    result = _physical_pixels(source, np.full(source.shape[:2], .2, dtype=np.float32),
                              np.full(3, .6, dtype=np.float32), params)
    slopes = np.diff(result[0, :, 0]) / np.diff(x)
    assert np.max(slopes) <= 1.61
    assert np.min(slopes) >= 0


def test_solar_ramp_is_monotonic_and_does_not_expand_white_region():
    x = np.linspace(0, 1, 2048, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    source = np.repeat(source, 8, axis=0)
    for air in (.02, .1, .35, .65, .99):
        result = _physical_pixels(source, np.full(source.shape[:2], .4, dtype=np.float32),
                                  np.full(3, air, dtype=np.float32), DehazeParams(strength=1))
        assert np.min(np.diff(result[0, :, 0])) >= -1e-6
        assert np.max((result @ LUMA) - (source @ LUMA)) < 3e-7
        assert np.array_equal(result[:, -1], source[:, -1])
        assert np.count_nonzero(result @ LUMA >= .97) <= np.count_nonzero(source @ LUMA >= .97)


def test_neutral_remains_neutral_under_coloured_airlight():
    x = np.linspace(.03, .9, 256, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    result = _physical_pixels(source, np.full(source.shape[:2], .5, dtype=np.float32),
                              np.array([.3, .5, .7], dtype=np.float32), DehazeParams(strength=1))
    assert np.max(np.ptp(result, axis=2)) < 1e-6


def test_source_hue_is_stable_when_colour_protection_is_full():
    source = scene()
    result = apply_physical_dehaze(source, DehazeParams(strength=1, color_protection=1, color_recovery=1), spatial=True)
    before = source - (source @ LUMA)[..., None]
    after = result - (result @ LUMA)[..., None]
    cosine = np.sum(before * after, axis=2) / (np.linalg.norm(before, axis=2) * np.linalg.norm(after, axis=2))
    assert np.min(cosine) > .99999


def test_saturated_sun_is_not_selected_as_airlight():
    source = np.full((100, 160, 3), [.13, .15, .17], dtype=np.float32)
    source[10:30, 60:85] = 1
    air, confidence = _estimate_airlight(source, DehazeParams())
    assert np.max(air) < .2
    assert .15 <= confidence <= 1
    source[:] = [.002, .003, .004]
    source[10:15, 60:65] = 1
    _, uncertain = _estimate_airlight(source, DehazeParams())
    assert uncertain < confidence


def test_broad_airlight_finds_lower_haze_without_vertical_position_bias():
    lower = broad_smoke_scene((100, 165))
    upper = broad_smoke_scene((22, 87))
    lower_air, lower_confidence = _estimate_broad_airlight(lower, DehazeParams())
    upper_air, upper_confidence = _estimate_broad_airlight(upper, DehazeParams())
    assert lower_air is not None and upper_air is not None
    np.testing.assert_allclose(lower_air, upper_air, atol=2e-4, rtol=0)
    assert lower_confidence == pytest.approx(upper_confidence, abs=1e-5)

    # The existing upper-frame reference is dark here, so the full-frame pool
    # must carry the lower smoke into the fused estimate.
    air, confidence = _estimate_airlight(lower, DehazeParams())
    assert float(air @ LUMA) > .05
    assert confidence > .75


def test_broad_airlight_is_exposure_scale_consistent_without_clipping():
    source = broad_smoke_scene()
    base_air, base_confidence = _estimate_airlight(source, DehazeParams())
    for exposure in (.25, .5, 1, 2):
        scaled_air, confidence = _estimate_airlight(source * exposure, DehazeParams())
        np.testing.assert_allclose(scaled_air / exposure, base_air, atol=2e-5, rtol=2e-4)
        assert confidence == pytest.approx(base_confidence, abs=2e-5)


def test_broad_airlight_ignores_narrow_emission_and_preserves_the_lights():
    source = broad_smoke_scene()
    baseline, _ = _estimate_broad_airlight(source, DehazeParams())
    source[130, 30:130] = [.82, .82, .82]
    source[145, 48] = [1, 1, 1]
    air, confidence = _estimate_broad_airlight(source, DehazeParams())
    assert baseline is not None and air is not None
    np.testing.assert_allclose(air, baseline, atol=.006, rtol=.04)
    assert confidence > .75

    params = DehazeParams(strength=.8, local_contrast=0, color_recovery=0)
    result = apply_physical_dehaze(source, params, spatial=False)
    result_y = result @ LUMA
    assert result_y[130, 80] > 4 * result_y[129, 80]
    assert result_y[145, 48] > .90


def test_broad_haze_structure_contrast_grows_continuously_with_strength():
    source = broad_smoke_scene()
    strengths = (0, .2, .4, .6, .8, 1)
    contrasts = []
    changes = []
    for strength in strengths:
        params = DehazeParams(strength=strength, local_contrast=0, color_recovery=0)
        result = apply_physical_dehaze(source, params, spatial=False)
        contrasts.append(relative_structure_contrast(result))
        changes.append(float(np.mean(np.abs(result - source))))
    assert changes[0] == 0
    assert np.min(np.diff(changes)) >= 0
    assert np.min(np.diff(contrasts)) > 0
    assert contrasts[-1] > contrasts[0] * 1.25


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (8, 12), (12, 16)])
def test_dark_and_tiny_airlight_estimates_stay_finite_and_reject_an_isolated_light(shape):
    height, width = shape
    source = np.zeros((height, width, 3), dtype=np.float32)
    if height > 1 and width > 1:
        rng = np.random.default_rng(221)
        source[:] = rng.uniform(.0002, .003, source.shape).astype(np.float32)
        source[height // 2, width // 2] = 1
    air, confidence = _estimate_airlight(source, DehazeParams())
    assert np.isfinite(air).all() and np.isfinite(confidence)
    assert .15 <= confidence <= 1
    if height * width > 1:
        assert float(np.max(air)) < .02
    for dark in (np.zeros_like(source), np.full_like(source, .001)):
        estimated, certainty = _estimate_airlight(dark, DehazeParams())
        assert np.isfinite(estimated).all() and np.isfinite(certainty)
        result = apply_physical_dehaze(dark, DehazeParams(strength=.8), spatial=False)
        assert result.shape == dark.shape and np.isfinite(result).all()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -.001, 1.001])
def test_invalid_linear_data_is_rejected(value):
    source = scene()
    source[0, 0, 0] = value
    with pytest.raises(ValueError):
        apply_physical_dehaze(source, DehazeParams(strength=1))


@pytest.mark.parametrize("shape", [(12, 16), (12, 16, 1), (12, 16, 4)])
def test_non_rgb_channels_are_rejected(shape):
    with pytest.raises(ValueError):
        apply_physical_dehaze(np.zeros(shape, dtype=np.float32), DehazeParams(strength=1))


def test_uint16_keeps_sub_eight_bit_gradient_detail():
    ramp = np.arange(30000, 31024, dtype=np.uint16)
    source = np.repeat(ramp[None, :, None], 3, axis=2)
    result = apply_physical_dehaze(source, DehazeParams(strength=.7, local_contrast=0, color_recovery=0))
    assert np.unique(result[..., 0]).size > 512
    assert np.min(np.diff(result[0, :, 0].astype(np.int32))) >= 0


def test_nonfinite_parameters_normalize_and_cache_version_changes():
    source = scene()
    result = apply_physical_dehaze(source, DehazeParams(strength=np.inf, local_contrast=np.nan, shadow_protection=-np.inf))
    assert np.isfinite(result).all()
    report = physical_diagnostics(source, DehazeParams(strength=1))
    assert .15 <= report["airlight_confidence"] <= 1
    assert report["max_inverse_gain"] <= 1.8


def test_missing_float_gpu_uses_same_cpu_math(monkeypatch):
    import native_renderer
    def missing(*args):
        raise native_renderer.NativeRendererError("old library")
    monkeypatch.setattr(native_renderer, "native_physical_dehaze", missing, raising=False)
    source = scene()
    params = DehazeParams(strength=.7)
    assert np.array_equal(apply_physical_dehaze(source, params, backend="native", spatial=True),
                          apply_physical_dehaze(source, params, backend="cpu", spatial=True))


def test_known_haze_structure_contrast_keeps_growing_at_high_strength():
    # Two reflectances behind the same known veil; scaling the entire image
    # alone cannot improve this relative contrast.
    clear = np.full((32, 80, 3), .15, dtype=np.float32)
    clear[:, :40] = .03
    air = np.full(3, .30, dtype=np.float32)
    source = .5 * clear + .5 * air
    contrasts = []
    for strength in (0, .35, .7, 1):
        p = DehazeParams(strength=strength, local_contrast=0, color_recovery=0)
        result = _physical_pixels(source, np.full(source.shape[:2], .5, dtype=np.float32), air, p)
        low, high = np.mean(result[:, :40] @ LUMA), np.mean(result[:, 40:] @ LUMA)
        contrasts.append(float((high - low) / (high + low)))
    assert np.min(np.diff(contrasts)) > 0
    assert contrasts[-1] > 1.5 * contrasts[0]
    assert contrasts[-1] - contrasts[-2] > .03


def test_quadratic_shadow_budget_preserves_blacks_and_recovers_hazy_midtones():
    # Hold A and t fixed so these checks exercise the negative-side pixel
    # operator rather than scene estimation. The ramp covers the entire
    # negative side, including true shadows and veil-lifted midtones.
    air_value = .30
    x = np.unique(np.concatenate((
        np.linspace(0, air_value, 4097, dtype=np.float32),
        np.array((.0005, .15, .22), dtype=np.float32),
    )))
    source = np.broadcast_to(x[None, :, None], (1, x.size, 3)).copy()
    transmission = np.full(source.shape[:2], .5, dtype=np.float32)
    air = np.full(3, air_value, dtype=np.float32)
    strengths = (0, 1e-5, .01, .1, .35, .7, 1)
    settings = ((0, 0), (1, 0), (0, 1), (1, 1))
    outputs = {}

    for brightness, shadow in settings:
        by_strength = []
        for strength in strengths:
            params = DehazeParams(
                strength=strength, brightness_protection=brightness,
                shadow_protection=shadow, local_contrast=0, color_recovery=0,
            )
            result = _physical_pixels(source, transmission, air, params)[0, :, 0]
            by_strength.append(result)

            increments = np.diff(result)
            slopes = increments / np.diff(x)
            assert np.min(increments) >= -1e-7
            assert np.min(slopes) >= -1e-3
            assert np.max(slopes) <= 1.8 + 1e-3
            assert np.max(result - x) <= 1e-7

        by_strength = np.asarray(by_strength)
        outputs[(brightness, shadow)] = by_strength
        np.testing.assert_array_equal(by_strength[0], x)
        # The slider stays continuous from zero through small and full values,
        # and stronger inversion never brightens a pixel below A.
        strength_steps = np.diff(np.asarray(strengths, dtype=np.float32))
        strength_slopes = np.diff(by_strength, axis=0) / strength_steps[:, None]
        assert np.max(np.abs(strength_slopes)) <= 1.8 + 1e-3
        assert np.max(np.diff(by_strength, axis=0)) <= 1e-7

        full = by_strength[-1]
        shadow_loss = .0005 - full[np.searchsorted(x, .0005)]
        midtone_loss = .15 - full[np.searchsorted(x, .15)]
        assert shadow_loss >= -1e-7
        assert shadow_loss <= .01 * midtone_loss
        midtone_contrast = (full[np.searchsorted(x, .22)]
                            - full[np.searchsorted(x, .15)])
        assert midtone_contrast > .22 - .15

    # Both protection controls retain their authority over haze-lifted tones.
    strongest = {key: value[-1] for key, value in outputs.items()}
    midtone_index = np.searchsorted(x, .15)
    assert strongest[(0, 0)][midtone_index] < strongest[(1, 0)][midtone_index]
    assert strongest[(0, 0)][midtone_index] < strongest[(0, 1)][midtone_index]
    assert strongest[(1, 1)][midtone_index] > strongest[(1, 0)][midtone_index]
    assert strongest[(1, 1)][midtone_index] > strongest[(0, 1)][midtone_index]


def test_negative_toe_and_airlight_handoff_have_continuous_gray_gradient():
    x = np.linspace(.10, .45, 4097, dtype=np.float32)
    source = np.broadcast_to(x[None, :, None], (3, x.size, 3)).copy()
    result = _physical_pixels(source, np.full(source.shape[:2], .2, dtype=np.float32),
                              np.full(3, .30, dtype=np.float32),
                              DehazeParams(strength=1, local_contrast=0, color_recovery=0))
    increments = np.diff(result[1, :, 0])
    slopes = increments / np.diff(x)
    assert np.min(increments) > 0  # no new flat levels or gradient reversal
    # A hard active-constraint switch changes slope at one sample. Float32
    # roundoff plus the smooth 0.025*A shoulder permits a small finite change.
    assert np.max(np.abs(np.diff(slopes))) < .025


def test_gain_floor_has_no_hard_slope_switch():
    from dehaze_physical import _inverse_transmission
    p = DehazeParams(strength=1, highlight_protection=0)
    floor = 1 / (1 + .8 * p.strength * (1 - .35 * p.naturalness))
    x = np.linspace(floor - .02, floor + .02, 4097, dtype=np.float32)
    source = np.full((2, x.size, 3), .20, dtype=np.float32)
    t = _inverse_transmission(source, np.broadcast_to(x, source.shape[:2]), p)[0]
    slopes = np.diff(t) / np.diff(x)
    assert np.min(slopes) >= 0
    assert np.max(slopes) <= 1.01
    assert np.max(np.abs(np.diff(slopes))) < .04


def test_optical_thickness_and_pixel_tiles_have_no_256_row_seam(monkeypatch):
    import dehaze_physical as physical
    height, width = 600, 37
    y = np.linspace(.16, .28, height, dtype=np.float32)[:, None, None]
    source = np.broadcast_to(y, (height, width, 3)).copy()
    depth = np.broadcast_to(np.linspace(.1, .5, height, dtype=np.float32)[:, None],
                            (height, width)).copy()
    transmission = np.exp(-depth)
    regular = physical._regularize_transmission(source, transmission)
    regular_depth = -np.log(regular[:, width // 2])
    for boundary in (256, 512):
        differences = np.diff(regular_depth[boundary - 4:boundary + 4])
        assert np.max(differences) - np.min(differences) < 1e-5
    constant = physical._regularize_transmission(source, np.full((height, width), .72, dtype=np.float32))
    np.testing.assert_allclose(constant, .72, atol=1e-7, rtol=0)
    air = np.full(3, .32, dtype=np.float32)
    monkeypatch.setattr(physical, "_estimate_scene", lambda *_: (transmission, air, {}))
    p = DehazeParams(strength=.8)
    expected = _physical_pixels(source, regular, air, p)
    actual = apply_physical_dehaze(source, p, backend="cpu", spatial=True)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)


def test_soft_solar_glare_remains_monotonic_after_full_spatial_pipeline():
    yy, xx = np.mgrid[-128:129, -128:129]
    sun = np.minimum(1, .13 + .90 * np.exp(-(xx * xx + yy * yy) / 4000)).astype(np.float32)
    source = np.repeat(sun[..., None], 3, axis=2)
    result = apply_physical_dehaze(source, DehazeParams(strength=1, local_contrast=0,
                                                      color_recovery=0), spatial=True)
    result_y = result @ LUMA
    source_y = source @ LUMA
    assert np.max(result_y - source_y) < 3e-7
    assert np.count_nonzero(result_y >= .97) <= np.count_nonzero(source_y >= .97)
    for profile in (result_y[128], result_y[:, 128]):
        assert np.min(np.diff(profile[:129])) >= -1e-6
        assert np.max(np.diff(profile[128:])) <= 1e-6
    # The plateau in the original saturated core must remain intact.
    np.testing.assert_allclose(result[source_y >= .9999], source[source_y >= .9999], atol=2e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_coloured_solar_glare_cannot_expand_any_channel(dtype):
    yy, xx = np.mgrid[-128:129, -128:129]
    radius2 = xx * xx + yy * yy
    glow = np.exp(-radius2 / 3500).astype(np.float32)
    source = np.clip(np.array([.08, .10, .12], np.float32)
                     + glow[..., None] * np.array([1.12, .92, .53], np.float32), 0, 1)
    if dtype == np.uint16:
        source = np.rint(source * 65535).astype(dtype)
    original = source.copy()
    peak = 65535 if dtype == np.uint16 else 1
    result = apply_physical_dehaze(source, DehazeParams(strength=1, color_recovery=1), spatial=True)
    assert np.max(result.astype(np.float64) - source) <= (0 if dtype == np.uint16 else 2e-7)
    # The footprint is checked well outside the white core as well as at it.
    for threshold in (.2, .4, .6, .8, .97):
        for channel in range(3):
            assert np.count_nonzero(result[..., channel] >= threshold * peak) <= np.count_nonzero(source[..., channel] >= threshold * peak)
    # A coloured saturated core must not turn into a darker inner ring.
    tolerance = 1 if dtype == np.uint16 else 2e-7
    profile = result[128].astype(np.float64)
    assert np.min(np.diff(profile[:129], axis=0)) >= -tolerance
    assert np.max(np.diff(profile[128:], axis=0)) <= tolerance
    np.testing.assert_array_equal(source, original)


def test_channel_ceiling_does_not_crush_tiny_coloured_shadows():
    source = np.array([[[1e-9, .00006, .00021], [0, .00006, .00021]]], np.float32)
    result = _physical_pixels(source, np.full((1, 2), .2, np.float32),
                              np.array([.85, .68, .52], np.float32),
                              DehazeParams(strength=1, local_contrast=0, color_recovery=0, color_protection=0))
    assert np.min((result @ LUMA) / (source @ LUMA)) > .98
