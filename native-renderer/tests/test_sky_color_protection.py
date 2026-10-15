"""Public-pipeline regressions for solar edges, even skies and shadow colour."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dehaze import DehazeParams
from dehaze_physical import apply_physical_dehaze

LUMA = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)


def _encode(source, dtype):
    if dtype == np.float32:
        return source.astype(np.float32, copy=True)
    return np.rint(source * np.iinfo(dtype).max).astype(dtype)


def _linear(image):
    if image.dtype == np.float32:
        return image
    return image.astype(np.float32) / np.iinfo(image.dtype).max


def _display(linear):
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055)


def _skyline_scene(sky):
    source = np.empty((320, 512, 3), dtype=np.float32)
    source[:] = sky
    source[280:] = (0.055, 0.065, 0.075)
    source[184:, 48:176] = (0.085, 0.105, 0.125)
    source[96:, 244:256] = (0.045, 0.055, 0.065)
    source[212:, 326:440] = (0.105, 0.125, 0.145)
    source[200:270:12, 60:164] *= 0.55
    source[225:275:10, 338:428] *= 0.6
    return source


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize("spatial", (False, True))
def test_public_pipeline_preserves_shape_depth_range_and_input(dtype, spatial):
    source = _encode(_skyline_scene((0.42, 0.52, 0.65)), dtype)
    original = source.copy()

    result = apply_physical_dehaze(
        source, DehazeParams(strength=0.85), backend="cpu", spatial=spatial,
    )

    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert not np.shares_memory(result, source)
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0
    assert float(_linear(result).max()) <= 1
    assert not np.array_equal(result, source)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize("spatial", (False, True))
def test_zero_strength_is_an_exact_independent_copy(dtype, spatial):
    source = _encode(_skyline_scene((0.42, 0.52, 0.65)), dtype)
    original = source.copy()
    params = DehazeParams(strength=0, local_contrast=1, color_recovery=1,
                          brightness_protection=1)

    result = apply_physical_dehaze(source, params, backend="cpu", spatial=spatial)

    assert result.dtype == source.dtype
    assert not np.shares_memory(result, source)
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, -0.01, 1.01))
@pytest.mark.parametrize("strength", (0, 1))
def test_public_pipeline_rejects_invalid_linear_samples(value, strength):
    source = np.full((8, 9, 3), 0.2, dtype=np.float32)
    source[2, 4, 1] = value
    with pytest.raises(ValueError, match="finite and within"):
        apply_physical_dehaze(source, DehazeParams(strength=strength), spatial=True)


@pytest.mark.parametrize("shape", ((8, 9), (8, 9, 1), (8, 9, 4)))
def test_public_pipeline_rejects_non_rgb_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        apply_physical_dehaze(np.zeros(shape, dtype=np.float32), DehazeParams(strength=1))


@pytest.mark.parametrize("shape", ((1, 1), (1, 7), (7, 1), (2, 3)))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
def test_automatic_pipeline_handles_small_images(shape, dtype):
    source = np.empty((*shape, 3), dtype=np.float32)
    source[:] = (0.12, 0.19, 0.27)
    source = _encode(source, dtype)
    original = source.copy()

    result = apply_physical_dehaze(source, DehazeParams(strength=1), spatial=True)

    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()
    assert float(_linear(result).min()) >= 0
    assert float(_linear(result).max()) <= 1
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("dtype", (np.float32, np.uint16))
@pytest.mark.parametrize("strength", (0.5, 1.0))
def test_automatic_sun_has_no_new_radial_ring_or_brightness_overshoot(dtype, strength):
    yy, xx = np.mgrid[:384, :512]
    center_y, center_x = 155, 256
    radius2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    glow = np.exp(-radius2 / (2 * 34.0 ** 2)).astype(np.float32)
    source = np.array((0.38, 0.48, 0.62), dtype=np.float32)
    source = source + glow[..., None] * np.array((0.72, 0.62, 0.48), dtype=np.float32)
    source = np.clip(source, 0, 1)
    source[310:] = (0.08, 0.105, 0.13)
    source[330::8] *= 0.5
    source = _encode(source, dtype)

    result = apply_physical_dehaze(
        source, DehazeParams(strength=strength), backend="cpu", spatial=True,
    )
    source_y = _linear(source) @ LUMA
    result_y = _linear(result) @ LUMA
    tolerance = 2e-5
    assert float(np.max(result_y - source_y)) <= tolerance
    # 沿多个方向检查完整太阳过渡，不能只比较中心与远处两个端点。
    radius = np.arange(145)
    for angle in np.arange(8) * np.pi / 4:
        rows = np.rint(center_y + radius * np.sin(angle)).astype(int)
        columns = np.rint(center_x + radius * np.cos(angle)).astype(int)
        assert float(np.max(np.diff(source_y[rows, columns]))) <= tolerance
        assert float(np.max(np.diff(result_y[rows, columns]))) <= tolerance
    assert float(result_y[center_y, center_x]) >= 0.99


@pytest.mark.parametrize("sky", ((0.42, 0.52, 0.65), (0.5, 0.5, 0.5),
                                 (0.10, 0.14, 0.19)))
@pytest.mark.parametrize("strength", (0.5, 1.0))
def test_uniform_sky_stays_even_next_to_roofs_and_tower_edges(sky, strength):
    source = _skyline_scene(sky)
    sky_mask = np.all(source == np.asarray(sky, dtype=np.float32), axis=2)
    result = apply_physical_dehaze(
        source, DehazeParams(strength=strength), backend="cpu", spatial=True,
    )

    # 相同天空 RGB 必须在建筑边缘、远处及处理块边界得到一致观感。
    sky_output = result[sky_mask]
    sky_luma = sky_output @ LUMA
    assert float(np.log2(sky_luma.max() / sky_luma.min())) <= 0.02
    assert float(np.max(np.ptp(_display(sky_output), axis=0))) <= 1 / 255


@pytest.mark.parametrize("sky", ((0.42, 0.52, 0.65), (0.65, 0.65, 0.65)))
@pytest.mark.parametrize("strength", (0.5, 1.0))
def test_vignetted_sky_matches_across_asymmetric_buildings(sky, strength):
    yy, xx = np.mgrid[:320, :512]
    illumination = 1 - 0.4 * ((xx - 255.5) / 255.5) ** 2 - 0.15 * yy / 319
    source = (np.asarray(sky, dtype=np.float32) * illumination[..., None]).astype(np.float32)
    sky_mask = np.ones(source.shape[:2], dtype=bool)
    for roof, left, right in ((70, 90, 104), (185, 40, 175), (280, 0, 512)):
        source[roof:, left:right] = (0.07, 0.09, 0.11)
        sky_mask[roof:, left:right] = False

    # 渐变和暗角本身对称；只有左侧建筑打破场景对称性。
    # 塔边天空比上部大气光候选暗，能显露被高光上限掩盖的边缘减光。
    paired_sky = sky_mask & sky_mask[:, ::-1]
    np.testing.assert_array_equal(source[paired_sky], source[:, ::-1][paired_sky])
    result = apply_physical_dehaze(
        source, DehazeParams(strength=strength), backend="cpu", spatial=True,
    )
    left = result[paired_sky]
    right = result[:, ::-1][paired_sky]
    difference_ev = np.abs(np.log2((left @ LUMA) / (right @ LUMA)))
    assert float(np.max(difference_ev)) <= 0.02
    assert float(np.max(np.abs(_display(left) - _display(right)))) <= 1 / 255


def test_automatic_pipeline_preserves_coloured_patches_and_neutral_greys():
    patches = np.array(((0.18, 0.07, 0.045), (0.035, 0.16, 0.07),
                        (0.055, 0.09, 0.22), (0.22, 0.16, 0.07),
                        (0.09, 0.09, 0.09), (0.4, 0.4, 0.4)), dtype=np.float32)
    source = np.empty((256, 384, 3), dtype=np.float32)
    source[:] = (0.5, 0.6, 0.7)
    for index, patch in enumerate(patches):
        source[160:, index * 64:(index + 1) * 64] = patch

    result = apply_physical_dehaze(source, DehazeParams(strength=1), spatial=True)
    output = result[208, np.arange(len(patches)) * 64 + 32]
    source_chroma = patches[:4] - np.mean(patches[:4], axis=1, keepdims=True)
    output_chroma = output[:4] - np.mean(output[:4], axis=1, keepdims=True)
    cosine = np.sum(source_chroma * output_chroma, axis=1) / (
        np.linalg.norm(source_chroma, axis=1) * np.linalg.norm(output_chroma, axis=1)
    )
    assert float(np.min(cosine)) >= np.cos(np.deg2rad(3))
    source_saturation = np.linalg.norm(source_chroma, axis=1) / (patches[:4] @ LUMA)
    output_saturation = np.linalg.norm(output_chroma, axis=1) / (output[:4] @ LUMA)
    assert float(np.min(output_saturation / source_saturation)) >= 0.9
    assert float(np.max(np.ptp(output[4:], axis=1))) <= 2e-6


def test_brightness_protection_recovers_dark_colour_without_flattening_detail():
    source = np.empty((320, 384, 3), dtype=np.float32)
    source[:] = (0.55, 0.64, 0.74)
    yy, xx = np.mgrid[:160, :384]
    light = ((xx // 8 + yy // 8) % 2) == 0
    facade = source[160:]
    facade[:] = (0.035, 0.045, 0.065)
    facade[light] = (0.08, 0.11, 0.14)
    common = dict(strength=1, naturalness=0.35, fog_retention=0,
                  shadow_protection=0, local_contrast=0,
                  color_protection=1, color_recovery=0)
    weak = apply_physical_dehaze(
        source, DehazeParams(**common, brightness_protection=0), spatial=True,
    )
    protected = apply_physical_dehaze(
        source, DehazeParams(**common, brightness_protection=1), spatial=True,
    )

    source_y = facade @ LUMA
    weak_y = weak[160:] @ LUMA
    protected_y = protected[160:] @ LUMA
    # 先确认样本确实触发明显减光，随后要求保护参数带来可见恢复。
    assert float(np.median(np.log2(source_y / weak_y))) >= 0.5
    assert float(np.median(np.log2(protected_y / weak_y))) >= 0.2
    assert float(np.max(protected_y - source_y)) <= 2e-6
    source_light = float(np.median(source_y[light]))
    source_dark = float(np.median(source_y[~light]))
    output_light = float(np.median(protected_y[light]))
    output_dark = float(np.median(protected_y[~light]))
    source_contrast = (source_light - source_dark) / (source_light + source_dark)
    output_contrast = (output_light - output_dark) / (output_light + output_dark)
    assert output_contrast >= 0.8 * source_contrast
    assert output_light - output_dark >= 0.65 * (source_light - source_dark)
    # 恢复后颜色仍沿原 RGB 比例变化，不得额外引入偏色。
    np.testing.assert_allclose(
        protected[160:] / protected_y[..., None],
        facade / source_y[..., None], atol=2e-5, rtol=0,
    )
