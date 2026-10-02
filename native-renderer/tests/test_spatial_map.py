"""Scene-level regressions for the shared automatic transmission map."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dehaze import DehazeParams
from dehaze_spatial import (
    _enhance_structure_details_in_tiles,
    _protect_dark_details_in_tiles,
    _structure_detail_support,
    _transmission_map,
    apply_spatial_dehaze,
)


def test_dim_cloud_band_does_not_create_a_false_skyline():
    height, width = 240, 360
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    # A broad dark cloud crosses open sky beside a narrow tower. Its edge
    # must not be treated as a roof with a different dehaze strength.
    sky = 0.08 - 0.025 * np.exp(-((y - 65.0) / 16.0) ** 2) + 0.003 * np.sin(x / 30.0)
    image = np.repeat(sky[:, :, None], 3, axis=2)
    image[120:] = 0.015
    image[100:120, 60:105] = 0.01
    source = np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    transmission = _transmission_map(source, DehazeParams(strength=1.0))
    open_sky = transmission[:95, 150:300]
    assert np.isfinite(transmission).all()
    assert float(np.ptp(open_sky)) < 0.015


def test_low_linear_daylight_keeps_spatial_dehazing():
    height, width = 240, 360
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    luma = np.where(y < 120, 0.14 - 0.01 * y / 120,
                    0.07 + 0.005 * np.sin(x / 14))
    rgb = np.repeat(np.broadcast_to(luma, (height, width))[:, :, None], 3, axis=2)
    source = np.rint(rgb * 65535).astype(np.uint16)

    transmission = _transmission_map(source, DehazeParams(strength=1.0))
    assert float(np.ptp(transmission)) > 0.08


def test_dark_land_is_protected_without_flattening_smooth_water():
    yy, xx = np.mgrid[:128, :256]
    luma = np.full((128, 256), 6000, dtype=np.uint16)
    luma[:, :128] = 6000 + ((xx[:, :128] + yy[:, :128]) % 12) * 300
    source = np.repeat(luma[:, :, None], 3, axis=2)
    darkened = np.rint(source * 0.3).astype(np.uint16)

    result = _protect_dark_details_in_tiles(source, darkened, 1.0)
    assert float(np.median(result[:, :100] / source[:, :100])) > 0.85
    assert float(np.median(result[:, 160:] / source[:, 160:])) < 0.35


def test_bright_turbine_edge_is_not_amplified_against_darker_sky():
    source = np.full((128, 256, 3), 9000, dtype=np.uint16)
    source[30:110, 125:128] = 18000
    enhanced = np.full_like(source, 7000)
    enhanced[30:110, 125:128] = 26000

    result = _protect_dark_details_in_tiles(
        source, enhanced, 1.0,
    )
    assert int(result[60, 126, 0]) <= int(source[60, 126, 0]) + 1
    assert int(result[60, 115, 0]) == 7000
    assert int(result[60, 126, 0] - result[60, 115, 0]) <= 9001


def test_distant_textured_detail_is_not_crushed():
    yy, xx = np.mgrid[:256, :256]
    luma = np.full((256, 256), 6000, dtype=np.uint16)
    luma[100:] = 6000 + ((xx[100:] // 4) % 2) * 2000
    luma[55:100, 40:200] = 6000 + ((xx[55:100, 40:200] // 4) % 2) * 2000
    source = np.repeat(luma[:, :, None], 3, axis=2)
    darkened = np.rint(source * 0.3).astype(np.uint16)

    result = _protect_dark_details_in_tiles(source, darkened, 1.0)
    source_range = int(source[75, 40:200, 0].max() - source[75, 40:200, 0].min())
    result_range = int(result[75, 40:200, 0].max() - result[75, 40:200, 0].min())
    assert result_range >= 0.8 * source_range


def _city_with_skyline_and_water(dtype):
    height, width = 200, 256
    yy, xx = np.mgrid[:height, :width]
    luma = np.full((height, width), 0.075, dtype=np.float32)
    sky = yy < 76
    luma[sky] = 0.16 + 0.008 * np.sin(xx[sky] / 40) * np.cos(yy[sky] / 30)
    luma[76:154] = 0.065
    building = (yy >= 88) & (yy < 150) & (xx >= 30) & (xx < 180)
    luma[building] += ((xx[building] // 4 + yy[building] // 3) % 2) * 0.020
    luma[154:] = 0.075 + 0.0002 * np.sin(xx[154:] / 7)
    rgb = np.repeat(luma[:, :, None], 3, axis=2)
    peak = np.iinfo(dtype).max
    return np.rint(rgb * peak).astype(dtype)


def test_local_detail_gain_is_bounded_and_preserves_sky_and_water():
    for dtype in (np.uint8, np.uint16):
        source = _city_with_skyline_and_water(dtype)
        support = _structure_detail_support(source)
        assert float(np.median(support[105:135, 55:155])) > 0.15
        assert float(np.max(support[:50])) < 0.01
        assert float(np.max(support[165:])) < 0.01

        weak = source.copy()
        strong = source.copy()
        _enhance_structure_details_in_tiles(
            weak, support, strength=0.45, local_contrast=0.8, naturalness=0.7,
        )
        _enhance_structure_details_in_tiles(
            strong, support, strength=0.9, local_contrast=0.8, naturalness=0.7,
        )
        assert weak.dtype == strong.dtype == dtype
        assert weak.shape == strong.shape == source.shape
        assert np.max(np.abs(strong[:50].astype(np.int32) - source[:50])) <= 1
        assert np.max(np.abs(strong[165:].astype(np.int32) - source[165:])) <= 1
        assert np.max(np.abs(strong.astype(np.int32) - source.astype(np.int32))) < 0.03 * np.iinfo(dtype).max

        before = np.diff(source[105:135, 55:155, 0].astype(np.int32), axis=1)
        weak_detail = np.diff(weak[105:135, 55:155, 0].astype(np.int32), axis=1)
        strong_detail = np.diff(strong[105:135, 55:155, 0].astype(np.int32), axis=1)
        assert float(np.std(strong_detail)) > float(np.std(weak_detail))
        assert float(np.std(weak_detail)) > float(np.std(before))


def test_auto_dehaze_strength_zero_and_integer_depths():
    params = DehazeParams(strength=0.0)
    for dtype in (np.uint8, np.uint16):
        source = _city_with_skyline_and_water(dtype)
        original = source.copy()
        result = apply_spatial_dehaze(source, params, backend="cpu")
        assert result.dtype == dtype
        assert result.shape == source.shape
        assert result is not source
        assert np.array_equal(result, original)
        assert np.array_equal(source, original)
