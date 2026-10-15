"""Colour-footprint and dark exposure guards shared by CPU and Metal paths."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dehaze import DehazeParams
from dehaze_physical import (
    _dark_background_floor,
    _physical_pixels,
    _protect_dark_background,
    _luminance,
    apply_physical_dehaze,
)

LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)


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


def test_dark_sky_lift_leaves_recovered_smoke_and_lights_unchanged():
    source = np.full((64, 96, 3), .012, np.float32)
    smoke = np.zeros(source.shape[:2], dtype=bool)
    smoke[28:36, 8:28] = True
    source[smoke] = .040
    smoke[40:48, 28:48] = True
    source[40:48, 28:48] = .060
    lights = np.zeros(source.shape[:2], dtype=bool)
    lights[54:58, 60:70] = True
    source[lights] = np.array([.98, .82, .55], np.float32)

    # The inverse has already recovered the smoke separation and lights.
    # Only the dark sky remains below its source-guided background envelope.
    result = source * .25
    result[28:36, 8:28] = source[28:36, 8:28] * .70
    result[40:48, 28:48] = source[40:48, 28:48] * .78
    result[lights] = source[lights]
    params = DehazeParams(strength=1, brightness_protection=.70)
    atmosphere = np.full(3, .04, np.float32)
    floor_level = _dark_background_floor(source, atmosphere, params)
    protected = _protect_dark_background(source, result, floor_level)

    sky_source = _luminance(source[:18])
    sky_before = sky_source - _luminance(result[:18])
    sky_after = sky_source - _luminance(protected[:18])
    assert np.median(sky_after) < np.median(sky_before) * .5
    np.testing.assert_allclose(protected[smoke], result[smoke], rtol=0, atol=1e-7)
    np.testing.assert_allclose(protected[lights], result[lights], rtol=0, atol=1e-7)
    assert np.isfinite(protected).all()
    assert np.max(protected - source) <= 1e-7


def test_dark_background_floor_is_zero_for_bright_air_or_disabled_protection():
    source = np.full((40, 32, 3), .012, np.float32)
    source[29:33, 9:14] = np.array([.82, .46, .18], np.float32)
    params = DehazeParams(strength=1, brightness_protection=.70)
    assert _dark_background_floor(source, np.full(3, .11, np.float32), params) == 0
    assert _dark_background_floor(
        source, np.full(3, .04, np.float32),
        DehazeParams(strength=1, brightness_protection=0),
    ) == 0


def test_fixed_t_air_gray_ramp_has_monotone_c1_dark_background_handoff():
    ramp = np.linspace(0, .12, 4097, dtype=np.float32)
    source = np.repeat(ramp[:, None, None], 3, axis=2)
    transmission = np.full((ramp.size, 1), .40, np.float32)
    atmosphere = np.full(3, .06, np.float32)
    params = DehazeParams(strength=1, local_contrast=0, color_recovery=0)
    inverse = _physical_pixels(source, transmission, atmosphere, params)
    floor_level = .025
    protected = _protect_dark_background(source, inverse, floor_level)

    profile = protected[:, 0, 0]
    step = float(ramp[1] - ramp[0])
    slopes = np.diff(profile)
    assert np.min(slopes) >= -1e-8

    # Inspect the C1 handoff at the crossing of the inverse and monotone floor.
    floor = floor_level * (-np.expm1(-ramp / floor_level))
    delta = inverse[:, 0, 0] - floor
    crossings = np.flatnonzero((delta[:-1] < 0) & (delta[1:] >= 0))
    assert crossings.size == 1
    crossing = int(crossings[0])
    curvature = np.diff(slopes)
    near_handoff = curvature[max(0, crossing - 8):crossing + 10]
    assert np.max(np.abs(near_handoff)) < .05 * step


def test_dark_background_finisher_is_tile_invariant_and_preserves_tiny_black():
    rng = np.random.default_rng(1703)
    source = rng.uniform(0, .08, size=(513, 37, 3)).astype(np.float32)
    result = source * rng.uniform(.15, .75, size=(513, 37, 1)).astype(np.float32)
    floor_level = .02

    whole = _protect_dark_background(source, result, floor_level)
    tiled = np.empty_like(whole)
    for start in range(0, source.shape[0], 256):
        stop = min(source.shape[0], start + 256)
        tiled[start:stop] = _protect_dark_background(
            source[start:stop], result[start:stop], floor_level)
    np.testing.assert_allclose(whole, tiled, rtol=0, atol=1e-9)

    tiny_source = np.array([[[1e-9, 1e-9, 1e-9], [1e-9, .6e-9, .2e-9],
                             [0, 0, 0]]], np.float32)
    tiny_result = tiny_source * .1
    tiny_protected = _protect_dark_background(tiny_source, tiny_result, floor_level)
    assert np.isfinite(tiny_protected).all()
    before = _luminance(tiny_source)[0, :2]
    after = _luminance(tiny_protected)[0, :2]
    assert np.min(after / before) > .98
    np.testing.assert_array_equal(tiny_protected[0, 2], tiny_result[0, 2])


def test_zero_strength_remains_exact_identity():
    source = np.array([[[.012, .04, .08], [.4, .2, .1]]], np.float32)
    result = apply_physical_dehaze(source, DehazeParams(strength=0), spatial=True)
    np.testing.assert_array_equal(result, source)
