from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dehaze import DehazeParams
from dehaze_physical import (
    _backlit_optical_scale,
    _estimate_scene,
    _physical_pixels,
    apply_physical_dehaze,
)

LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)


def backlit_city_scene(*, sun=True):
    """A hazy city with a bright sky and dense two-axis facade detail."""
    height, width = 640, 480
    yy, xx = np.mgrid[:height, :width]
    sky = np.array([.52, .60, .69], dtype=np.float32)
    source = np.empty((height, width, 3), dtype=np.float32)
    source[:] = sky
    if sun:
        radius2 = (xx - 72) ** 2 + (yy - 58) ** 2
        halo = (.42 * np.exp(-radius2 / (2 * 14.0 ** 2))).astype(np.float32)
        source = np.clip(source + halo[..., None], 0, 1)
        source[radius2 <= 31 ** 2] = 1

    source[192:] = [.055, .065, .075]
    buildings = ((0, 78, 208), (91, 174, 192), (188, 279, 224),
                 (294, 387, 198), (399, 480, 216))
    for left, right, roof in buildings:
        x = xx[roof:, left:right] - left
        y = yy[roof:, left:right] - roof
        facade = np.empty((height - roof, right - left, 3), dtype=np.float32)
        facade[:] = [.105, .125, .145]
        windows = ((x % 16) < 7) | ((y % 16) < 3)
        facade[windows] = [.035, .045, .055]
        facade[:, -5:] = [.085, .095, .105]
        source[roof:, left:right] = facade
    return source.astype(np.float32, copy=False)


def synthetic_solar_backlight(*, lower=(.025, .025, .025), sun=True,
                              overexposed_sky=False):
    """Create a flat sky over dark ground with an optional compact solar core."""
    height, width = 640, 480
    yy, xx = np.mgrid[:height, :width]
    source = np.empty((height, width, 3), dtype=np.float32)
    source[:] = [.52, .60, .69]
    source[320:] = lower
    if overexposed_sky:
        source[:288] = 1
    elif sun:
        radius2 = (xx - 72) ** 2 + (yy - 58) ** 2
        source[radius2 <= 31 ** 2] = 1
    return source


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_triggered_backlit_city_strength_zero_is_an_exact_copy(dtype):
    source = backlit_city_scene()
    if dtype == np.uint16:
        source = np.rint(source * np.iinfo(np.uint16).max).astype(np.uint16)
    original = source.copy()

    result = apply_physical_dehaze(
        source, DehazeParams(strength=0, local_contrast=.7), spatial=True,
    )
    assert result is not source
    assert result.dtype == source.dtype
    assert np.array_equal(result, source)
    assert np.array_equal(source, original)


def test_spatial_backlit_city_matches_single_physical_operator_and_native_mock(monkeypatch):
    import dehaze_physical as physical
    import native_renderer

    source = backlit_city_scene()
    # Keep one facade densely windowed and make another broad, smooth wall
    # with a single narrow dark seam. Both contain the same wall RGB values.
    source[220:, 294:387] = [.105, .125, .145]
    source[280:600, 338:342] = [.035, .045, .055]
    original = source.copy()
    dense_point = (397, 39)
    sparse_point = (397, 320)
    np.testing.assert_array_equal(source[dense_point], source[sparse_point])

    params = DehazeParams(strength=.9, brightness_protection=.7,
                          local_contrast=.6, color_recovery=0)
    transmission = np.full(source.shape[:2], .62, dtype=np.float32)
    atmosphere = np.array([.214, .221, .228], dtype=np.float32)

    def fixed_scene(_source, _params, _spatial):
        return transmission, atmosphere, {}

    def mock_native(_source, p, t, air):
        # This stands in for the native operator with its Python float math.
        # It validates shared pipeline behavior, not a real GPU.
        return physical._physical_pixels(_source, t, air, p)

    monkeypatch.setattr(physical, "_estimate_scene", fixed_scene)
    monkeypatch.setattr(native_renderer, "native_physical_dehaze", mock_native, raising=False)
    monkeypatch.setattr(native_renderer, "get_last_native_physical_backend",
                        lambda: "test double", raising=False)

    regular_transmission = physical._regularize_transmission(source, transmission)
    expected = physical._physical_pixels(source, regular_transmission, atmosphere,
                                         params.normalized())
    cpu = apply_physical_dehaze(source, params, backend="cpu", spatial=True)
    native_mock = apply_physical_dehaze(source, params, backend="native", spatial=True)

    # The spatial path must add no separate neighborhood or density-based gain.
    np.testing.assert_array_equal(cpu, expected)
    np.testing.assert_array_equal(native_mock, expected)
    np.testing.assert_array_equal(cpu[dense_point], cpu[sparse_point])
    sky = np.indices(source.shape[:2])[0] < 180
    sun = np.max(source, axis=2) >= .9999
    np.testing.assert_array_equal(cpu[sky], expected[sky])
    np.testing.assert_array_equal(cpu[sun], expected[sun])
    assert np.isfinite(cpu).all()
    assert float(cpu.min()) >= 0 and float(cpu.max()) <= 1
    assert np.array_equal(source, original)


@pytest.mark.parametrize(
    "transmission,airlight,strength,naturalness,brightness,highlight,shadow,local",
    [
        (.18, .24, .85, .15, 0.0, 0.0, 0.0, .0),
        (.38, .42, 1.0, .60, .7, .35, .85, .6),
        (.72, .78, .55, .95, 1.0, 1.0, 1.0, .9),
        (.51, .31, .70, .35, .25, .8, .2, .35),
        (.625, .214, 1.0, .70, .7, .75, .75, .25),
        (.78, .221, 1.0, .70, .7, .75, .75, .25),
    ],
)
def test_physical_operator_gray_ramp_is_monotonic_through_solar_handoffs(
    transmission, airlight, strength, naturalness, brightness, highlight,
    shadow, local,
):
    import dehaze_physical as physical

    x = np.linspace(0, 1, 8193, dtype=np.float32)[None, :]
    source = np.repeat(x[..., None], 3, axis=2)
    t = np.full(x.shape, transmission, dtype=np.float32)
    air = np.full(3, airlight, dtype=np.float32)
    params = DehazeParams(
        strength=strength, naturalness=naturalness,
        brightness_protection=brightness, highlight_protection=highlight,
        shadow_protection=shadow, local_contrast=local, color_recovery=0,
    )

    result = physical._physical_pixels(source, t, air, params)
    profile = result[0, :, 0]
    assert np.isfinite(result).all()
    assert float(np.min(np.diff(profile))) >= -2e-7
    assert float(result.min()) >= 0 and float(result.max()) <= 1
    assert profile[0] == 0
    # The full ramp covers both the atmospheric-light crossing and the solar
    # highlight-protection handoff; the source-luma ceiling preserves white.
    assert float(np.max(result @ LUMA - source @ LUMA)) <= 3e-7
    np.testing.assert_array_equal(result[0, -1], source[0, -1])


def test_backlit_optical_scale_requires_joint_solar_and_dark_ground_evidence():
    source = synthetic_solar_backlight()
    params = DehazeParams(strength=.9)
    scale = _backlit_optical_scale(source, .2, params)
    assert 0 < scale < .5

    cases = [
        ("brightness protection disabled", source, .2,
         DehazeParams(strength=.9, brightness_protection=0)),
        ("strength disabled", source, .2, DehazeParams(strength=0)),
        ("no solar core", synthetic_solar_backlight(sun=False), .2, params),
        ("low light", source * np.float32(.08), .2, params),
        ("broadly clipped sky", synthetic_solar_backlight(overexposed_sky=True), .2, params),
        ("bright ground", synthetic_solar_backlight(lower=(.45, .50, .55)), .2, params),
        ("reliable airlight", source, .95, params),
    ]
    for label, image, confidence, candidate_params in cases:
        assert _backlit_optical_scale(image, confidence, candidate_params) == pytest.approx(
            1.0, abs=1e-7
        ), label


def test_backlit_optical_scale_changes_continuously_across_scene_ratio_gate():
    params = DehazeParams(strength=.9)
    sky_luma = float(np.array([.52, .60, .69], dtype=np.float32) @ LUMA)
    ratios = np.linspace(7.8, 14.2, 129)
    scales = []
    for ratio in ratios:
        source = synthetic_solar_backlight(lower=(sky_luma / ratio,) * 3)
        scales.append(_backlit_optical_scale(source, .2, params))

    scales = np.asarray(scales)
    assert np.all(np.isfinite(scales))
    assert np.max(np.diff(scales)) <= 1e-7
    assert np.max(np.abs(np.diff(scales))) < .01
    assert scales[0] == pytest.approx(1.0)
    assert scales[-1] < .5


def test_fixed_transmission_weak_inverse_retains_surface_luminance_and_solar_core():
    import dehaze_physical as physical

    source = synthetic_solar_backlight()
    source[360:600, :144] = [.25, .28, .31]
    source[360:600, 160:304] = [.40, .43, .46]
    original = source.copy()
    params = DehazeParams(
        strength=.85, local_contrast=0, color_recovery=0,
        brightness_protection=.7,
    ).normalized()
    atmosphere = np.array([.62, .68, .74], dtype=np.float32)
    base_t = .5
    scale = _backlit_optical_scale(source, .2, params)
    weak_t = float(np.exp(np.log(base_t) * scale))

    stronger = _physical_pixels(
        source, np.full(source.shape[:2], base_t, dtype=np.float32), atmosphere, params,
    )
    weaker = _physical_pixels(
        source, np.full(source.shape[:2], weak_t, dtype=np.float32), atmosphere, params,
    )
    dark_region = (slice(390, 560), slice(20, 120))
    bright_region = (slice(390, 560), slice(180, 280))
    stronger_y = stronger @ LUMA
    weaker_y = weaker @ LUMA
    assert float(np.mean(weaker_y[dark_region])) > float(np.mean(stronger_y[dark_region]))
    assert float(np.mean(weaker_y[bright_region])) > float(np.mean(stronger_y[bright_region]))
    assert float(np.mean(weaker_y[bright_region])) > float(np.mean(weaker_y[dark_region]))
    assert float(np.mean(stronger_y[bright_region])) > float(np.mean(stronger_y[dark_region]))
    for result in (stronger, weaker):
        assert np.isfinite(result).all()
        assert float(result.min()) >= 0 and float(result.max()) <= 1
    sun = np.max(source, axis=2) >= .9999
    np.testing.assert_array_equal(weaker[sun], source[sun])
    np.testing.assert_array_equal(source, original)


def test_estimate_scene_scales_only_automatic_spatial_transmission(monkeypatch):
    import dehaze_physical as physical

    source = synthetic_solar_backlight()
    params = DehazeParams(strength=.9)
    requested = .5
    confidence = .2
    atmosphere = np.array([.58, .64, .70], dtype=np.float32)
    monkeypatch.setattr(physical, "_estimate_airlight",
                        lambda _rgb, _params: (atmosphere, confidence))
    monkeypatch.setattr(
        physical, "_transmission_map",
        lambda image, _params: np.full(image.shape[:2], requested, dtype=np.float32),
    )
    monkeypatch.setattr(physical, "_global_transmission",
                        lambda _rgb, _params: (requested, 0.0))

    automatic_t, _, automatic_stats = _estimate_scene(source, params, spatial=True)
    manual_t, _, manual_stats = _estimate_scene(source, params, spatial=False)
    base_t = 1.0 - (1.0 - requested) * confidence
    expected_automatic_t = np.float32(
        np.exp(np.log(base_t) * automatic_stats["backlit_optical_scale"])
    )
    np.testing.assert_array_equal(automatic_t, np.full(source.shape[:2], expected_automatic_t))
    np.testing.assert_array_equal(manual_t, np.full(source.shape[:2], np.float32(base_t)))
    assert automatic_stats["backlit_optical_scale"] < 1.0
    assert manual_stats["backlit_optical_scale"] == 1.0
    assert float(np.mean(automatic_t)) > float(np.mean(manual_t))

    unprotected_t, _, unprotected_stats = _estimate_scene(
        source, DehazeParams(strength=.9, brightness_protection=0), spatial=True,
    )
    np.testing.assert_array_equal(unprotected_t, manual_t)
    assert unprotected_stats["backlit_optical_scale"] == 1.0
