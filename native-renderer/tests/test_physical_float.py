"""Metal physical float dehaze parity and Python ABI validation."""

from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import native_renderer  # noqa: E402
from dehaze import DehazeParams, _global_transmission  # noqa: E402
import dehaze_physical as physical  # noqa: E402
from dehaze_physical import _physical_pixels  # noqa: E402
from native_renderer import NativeRendererError, native_physical_dehaze  # noqa: E402


def _inputs(case: str):
    rng = np.random.default_rng(4821)
    if case == "random":
        source = rng.uniform(0.0, 1.0, (31, 53, 3)).astype(np.float32)
        transmission = rng.uniform(0.22, 1.0, (31, 53)).astype(np.float32)
        airlight = np.array((0.66, 0.71, 0.75), dtype=np.float32)
    elif case == "gray-ramp":
        ramp = np.linspace(0.0, 1.0, 257, dtype=np.float32)
        source = np.broadcast_to(ramp[None, :, None], (7, 257, 3)).copy()
        source[2:5, 116:142] = np.array((0.24, 0.30, 0.37), dtype=np.float32)
        source[1:6, 206:245] = np.array((0.985, 0.965, 0.91), dtype=np.float32)
        transmission = np.broadcast_to(
            np.linspace(0.24, 0.98, 257, dtype=np.float32)[None, :], (7, 257)
        ).copy()
        airlight = np.array((0.72, 0.70, 0.69), dtype=np.float32)
    else:
        source = rng.uniform(0.08, 0.93, (19, 23, 3)).astype(np.float32)
        source[:, :8] *= np.array((1.0, 0.82, 0.64), dtype=np.float32)
        source[3:13, 11:20] = np.array((0.996, 0.963, 0.91), dtype=np.float32)
        y, x = np.mgrid[:19, :23]
        transmission = (0.24 + 0.72 * (x + y) / (22 + 18)).astype(np.float32)
        airlight = np.array((0.59, 0.68, 0.77), dtype=np.float32)
    return source, transmission, airlight


def _params():
    return DehazeParams(
        strength=0.84, naturalness=0.43, fog_retention=0.52,
        local_contrast=0.61, color_recovery=0.73, color_protection=0.66,
        highlight_protection=0.81, shadow_protection=0.72,
        brightness_protection=0.76,
    )


def _night_skyline_scene():
    height, width = 180, 256
    yy, xx = np.mgrid[:height, :width]
    source = np.empty((height, width, 3), dtype=np.float32)
    source[:] = (0.032, 0.037, 0.043)
    source[104:] = (0.022, 0.025, 0.029)
    building = (yy >= 104) & (yy < 166) & (xx >= 50) & (xx < 190)
    source[building] = (0.010, 0.012, 0.014)
    # A few compact warm windows sit below the skyline, away from the upper
    # scene samples used to identify a broad sun or daylight.
    source[120:126, 69:74] = (0.42, 0.29, 0.15)
    source[138:143, 113:119] = (0.34, 0.25, 0.14)
    source[151:156, 164:169] = (0.38, 0.27, 0.13)
    return source


def test_spatial_estimator_does_not_apply_a_second_local_confidence_gate(monkeypatch):
    source = _night_skyline_scene()
    params = DehazeParams(strength=0.9, local_contrast=0, color_recovery=0)
    monkeypatch.setattr(
        physical, "_transmission_map",
        lambda image, _params: np.full(image.shape[:2], 0.5, dtype=np.float32),
    )

    transmission, _airlight, stats = physical._estimate_scene(source, params, spatial=True)
    np.testing.assert_allclose(
        transmission, transmission.flat[0], atol=1e-7, rtol=0,
    )
    expected = 1.0 - (1.0 - 0.5) * stats["airlight_confidence"]
    np.testing.assert_allclose(transmission, expected, atol=1e-7, rtol=0)

    result = physical.apply_physical_dehaze(source, params, backend="cpu", spatial=True)
    # These source pixels have exactly the same RGB, but one sits beside the
    # building and the other is open sky. A local gate followed by smoothing
    # used to make the near-building sky brighter.
    np.testing.assert_array_equal(source[100, 49], source[100, 220])
    np.testing.assert_allclose(result[100, 49], result[100, 220], atol=2e-7, rtol=0)


@pytest.mark.parametrize("strength", (0.35, 0.7, 1.0))
def test_night_skyline_full_pipeline_keeps_sky_even_and_lights_intact(strength):
    source = _night_skyline_scene()
    guide = source @ physical._LUMA
    scene_median = float(np.median(guide))
    top_sky = float(np.median(guide[: source.shape[0] // 5]))
    upper_peak = float(np.percentile(
        np.max(source[: source.shape[0] // 2], axis=2), 99,
    ))
    assert scene_median < 0.06
    assert top_sky < 0.09
    assert upper_peak < 0.75  # below the broad-sun classification threshold

    params = DehazeParams(strength=strength, local_contrast=0, color_recovery=0)
    global_t, _ = _global_transmission(source, params)
    spatial_t = physical._transmission_map(source, params)
    np.testing.assert_allclose(spatial_t, global_t, atol=1e-7, rtol=0)

    result = physical.apply_physical_dehaze(source, params, backend="cpu", spatial=True)
    # The same dark-sky RGB directly beside the skyline and in open sky must
    # receive the same output; this catches a bright rim from leaked depth.
    np.testing.assert_array_equal(source[100, 49], source[100, 220])
    np.testing.assert_allclose(result[100, 49], result[100, 220], atol=2e-7, rtol=0)
    # Dehazing may retain a real window light, but must not wash it into the
    # neighboring dark facade or create a local sky brightness bump.
    lamp_y = float(result[122, 71] @ physical._LUMA)
    source_lamp_y = float(source[122, 71] @ physical._LUMA)
    facade_y = float(result[122, 80] @ physical._LUMA)
    assert lamp_y >= 0.95 * source_lamp_y
    assert lamp_y > 6.0 * facade_y
    np.testing.assert_array_equal(source[122, 75], source[122, 105])
    np.testing.assert_allclose(result[122, 75], result[122, 105], atol=2e-7, rtol=0)


@pytest.mark.parametrize("case", ("random", "gray-ramp", "cast-nearwhite"))
def test_metal_physical_float_matches_numpy_reference(case):
    requested = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND")
    if requested and requested.lower() != "metal":
        pytest.skip(f"Physical float parity is implemented for Metal, not {requested}")
    source, transmission, airlight = _inputs(case)
    original = source.copy()
    params = _params()
    expected = _physical_pixels(source, transmission, airlight, params)
    try:
        actual = native_physical_dehaze(source, params, transmission, airlight)
    except NativeRendererError as exc:
        if requested and requested.lower() == "metal":
            pytest.fail(f"Requested Metal physical float operator failed: {exc}")
        pytest.skip(f"Metal physical float renderer is unavailable: {exc}")

    assert actual.shape == source.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual - expected)) <= 3e-6
    actual_u16 = np.rint(actual * 65535.0).astype(np.int64)
    expected_u16 = np.rint(expected * 65535.0).astype(np.int64)
    assert np.max(np.abs(actual_u16 - expected_u16)) <= 2
    np.testing.assert_array_equal(source, original)
    assert native_renderer.get_last_native_physical_backend() == "Metal"


@pytest.mark.parametrize(
    "params", (DehazeParams(strength=0.7), _params()),
    ids=("default-parameters-strength-0.7", "nontrivial-parameters"),
)
def test_metal_physical_float_nearwhite_clipped_channels_match_reference(params):
    requested = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND")
    if requested and requested.lower() != "metal":
        pytest.skip(f"Physical float parity is implemented for Metal, not {requested}")
    nearwhite = np.array(
        ((1.0, 1.0, 0.90624857), (1.0, 1.0, 0.98956281), (1.0, 1.0, 0.999)),
        dtype=np.float32,
    )
    yy, xx = np.mgrid[:18, :33]
    source = nearwhite[(xx + 2 * yy) % len(nearwhite)].copy()
    transmission = (0.85 + 0.15 * (xx + yy) / (32 + 17)).astype(np.float32)
    airlight = np.array((0.2342, 0.21455, 0.1864), dtype=np.float32)
    original = source.copy()
    expected = _physical_pixels(source, transmission, airlight, params)
    try:
        actual = native_physical_dehaze(source, params, transmission, airlight)
    except NativeRendererError as exc:
        if requested and requested.lower() == "metal":
            pytest.fail(f"Requested Metal physical float operator failed: {exc}")
        pytest.skip(f"Metal physical float renderer is unavailable: {exc}")

    assert actual.shape == source.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual - expected)) <= 3e-6
    actual_u16 = np.rint(actual * 65535.0).astype(np.int64)
    expected_u16 = np.rint(expected * 65535.0).astype(np.int64)
    assert np.max(np.abs(actual_u16 - expected_u16)) <= 2
    np.testing.assert_array_equal(source, original)
    assert native_renderer.get_last_native_physical_backend() == "Metal"


@pytest.mark.parametrize(
    "strength,naturalness",
    ((0.86, 0.34), (0.12, 0.88), (1e-5, 0.45)),
    ids=("strong-gain-floor", "soft-gain-floor", "near-identity-strength"),
)
def test_metal_physical_float_negative_toe_and_solar_color_match_reference(
    strength, naturalness
):
    requested = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND")
    if requested and requested.lower() != "metal":
        pytest.skip(f"Physical float parity is implemented for Metal, not {requested}")
    params = DehazeParams(
        strength=strength, naturalness=naturalness, local_contrast=0.29,
        color_recovery=0.57, color_protection=0.31, highlight_protection=0.87,
        shadow_protection=0.13, brightness_protection=0.27,
    )
    airlight = np.array((0.63, 0.69, 0.76), dtype=np.float32)
    source = np.array((
        (0.0005, 0.0005, 0.0005),  # true black-level shadow
        (0.004, 0.008, 0.013),       # deep negative side, toe shoulder
        (0.080, 0.120, 0.180),
        (0.150, 0.170, 0.190),       # hazy negative-side midtone
        (0.220, 0.240, 0.260),
        (0.330, 0.360, 0.400),
        (0.625, 0.685, 0.755),       # close below A
        (0.630, 0.690, 0.760),       # exactly at A
        (0.635, 0.695, 0.765),       # close above A
        (0.800, 0.760, 0.730),       # mixed positive and negative channels
        (0.996, 0.986, 0.963),       # colored solar highlight
    ), dtype=np.float32)[None, :, :]
    gain = 1.0 + 0.8 * params.strength * (1.0 - 0.35 * params.naturalness)
    floor = 1.0 / gain
    width = 0.015 * (1.0 - floor)
    transmission = np.array((
        0.22,
        0.22,
        max(0.0, floor - 0.5 * width),
        0.22,
        0.22,
        floor,
        min(1.0, floor + 0.5 * width),
        min(1.0, floor + 3.0 * width),
        0.75,
        0.42,
        0.93,
    ), dtype=np.float32)[None, :]
    original = source.copy()
    expected = _physical_pixels(source, transmission, airlight, params)
    try:
        actual = native_physical_dehaze(source, params, transmission, airlight)
    except NativeRendererError as exc:
        if requested and requested.lower() == "metal":
            pytest.fail(f"Requested Metal physical float operator failed: {exc}")
        pytest.skip(f"Metal physical float renderer is unavailable: {exc}")

    assert actual.shape == source.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual - expected)) <= 3e-6
    actual_u16 = np.rint(actual * 65535.0).astype(np.int64)
    expected_u16 = np.rint(expected * 65535.0).astype(np.int64)
    assert np.max(np.abs(actual_u16 - expected_u16)) <= 2
    np.testing.assert_array_equal(source, original)
    assert native_renderer.get_last_native_physical_backend() == "Metal"


def _solar_ramp_reference(brightness_protection, airlight_values):
    params = DehazeParams(
        strength=0.91, naturalness=0.24, fog_retention=0.30,
        local_contrast=0.74, color_recovery=0.96, color_protection=0.08,
        highlight_protection=0.94, shadow_protection=0.12,
        brightness_protection=brightness_protection,
    )
    height = width = 97
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    radius = np.sqrt((xx - 48.0) ** 2 + (yy - 48.0) ** 2)
    halo = np.exp(-((radius / 22.0) ** 2))
    source = np.empty((height, width, 3), dtype=np.float32)
    source[:] = (0.52, 0.60, 0.68)
    source += halo[..., None] * np.array((0.43, 0.32, 0.20), dtype=np.float32)
    transmission = np.full((height, width), 0.46, dtype=np.float32)
    airlight = np.array(airlight_values, dtype=np.float32)
    original = source.copy()
    expected = _physical_pixels(source, transmission, airlight, params)
    return params, source, transmission, airlight, original, expected, radius


def _assert_solar_radial_continuity(source, radius, expected):
    radial_edges = np.arange(0.0, 32.0, 2.0, dtype=np.float32)
    rings = [(radius >= edge) & (radius < edge + 2.0) for edge in radial_edges]
    output_luma = expected @ physical._LUMA
    luma_profile = np.array([output_luma[ring].mean() for ring in rings])
    red_blue_profile = np.array([
        (expected[..., 0] - expected[..., 2])[ring].mean() for ring in rings
    ])
    assert np.isfinite(expected).all()
    assert float(np.max(output_luma - source @ physical._LUMA)) <= 3e-7
    assert float(np.max(np.diff(luma_profile))) <= 2e-6
    assert float(np.max(np.diff(red_blue_profile))) <= 2e-6


@pytest.mark.parametrize("brightness_protection", (0.0, 0.7, 1.0))
@pytest.mark.parametrize(
    "airlight_values",
    (
        pytest.param((0.63, 0.69, 0.76), id="bright-airlight"),
        pytest.param((0.07, 0.09, 0.12), id="dim-airlight"),
    ),
)
def test_physical_float_colored_solar_ramp_cpu_radial_gradient(
    brightness_protection, airlight_values,
):
    _, source, _, _, original, expected, radius = _solar_ramp_reference(
        brightness_protection, airlight_values,
    )
    _assert_solar_radial_continuity(source, radius, expected)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("brightness_protection", (0.0, 0.7, 1.0))
@pytest.mark.parametrize(
    "airlight_values",
    (
        pytest.param((0.63, 0.69, 0.76), id="bright-airlight"),
        pytest.param((0.07, 0.09, 0.12), id="dim-airlight"),
    ),
)
def test_metal_physical_float_colored_solar_ramp_matches_reference(
    brightness_protection, airlight_values,
):
    requested = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND")
    if requested and requested.lower() != "metal":
        pytest.skip(f"Physical float parity is implemented for Metal, not {requested}")

    params, source, transmission, airlight, original, expected, radius = (
        _solar_ramp_reference(brightness_protection, airlight_values)
    )
    # This also runs before a possible Metal-unavailable skip below.
    _assert_solar_radial_continuity(source, radius, expected)
    try:
        actual = native_physical_dehaze(source, params, transmission, airlight)
    except NativeRendererError as exc:
        if requested and requested.lower() == "metal":
            pytest.fail(f"Requested Metal physical float operator failed: {exc}")
        pytest.skip(f"Metal physical float renderer is unavailable: {exc}")

    assert actual.shape == source.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual - expected)) <= 3e-6
    actual_u16 = np.rint(actual * 65535.0).astype(np.int64)
    expected_u16 = np.rint(expected * 65535.0).astype(np.int64)
    assert np.max(np.abs(actual_u16 - expected_u16)) <= 2
    np.testing.assert_array_equal(source, original)
    assert native_renderer.get_last_native_physical_backend() == "Metal"


def test_old_renderer_without_optional_float_abi_raises(monkeypatch):
    monkeypatch.setattr(native_renderer, "_get_library",
                        lambda: (SimpleNamespace(), "old-renderer"))
    source, transmission, airlight = _inputs("random")
    with pytest.raises(NativeRendererError, match="ABI is unavailable"):
        native_physical_dehaze(source, _params(), transmission, airlight)


def test_physical_status_marks_old_renderer_unavailable(monkeypatch):
    monkeypatch.setattr(native_renderer, "_get_library",
                        lambda: (SimpleNamespace(), "/private/local/old-renderer.dylib"))
    monkeypatch.setattr(native_renderer, "_create_renderer",
                        lambda _library: pytest.fail("Old ABI must not create a renderer"))
    status = native_renderer.get_native_physical_status()
    assert status == {"available": False, "backend": None,
                      "gpu_available": False, "cpu_available": False}


@pytest.mark.parametrize("backend", ("D3D12", "CUDA", "Unknown"))
def test_physical_status_rejects_non_metal_backend_and_destroys_renderer(backend, monkeypatch):
    destroyed = []
    library = SimpleNamespace(
        im_renderer_render_physical_float=object(),
        im_renderer_destroy=lambda renderer: destroyed.append(renderer),
    )
    monkeypatch.setattr(native_renderer, "_get_library", lambda: (library, "/private/local/lib.dylib"))
    monkeypatch.setattr(native_renderer, "_create_renderer",
                        lambda _library: ("fake-renderer", backend))
    status = native_renderer.get_native_physical_status()
    assert status == {"available": False, "backend": None,
                      "gpu_available": False, "cpu_available": False}
    assert destroyed == ["fake-renderer"]


def test_physical_status_reports_actual_metal_backend_and_never_reports_cpu(monkeypatch):
    destroyed = []
    library = SimpleNamespace(
        im_renderer_render_physical_float=object(),
        im_renderer_destroy=lambda renderer: destroyed.append(renderer),
    )
    monkeypatch.setattr(native_renderer, "_get_library", lambda: (library, "/private/local/lib.dylib"))
    monkeypatch.setattr(native_renderer, "_create_renderer",
                        lambda _library: ("metal-renderer", "Metal"))
    status = native_renderer.get_native_physical_status()
    assert status == {"available": True, "backend": "Metal",
                      "gpu_available": True, "cpu_available": False}
    assert destroyed == ["metal-renderer"]


def test_physical_status_fails_safe_without_leaking_library_path(monkeypatch):
    monkeypatch.setattr(
        native_renderer, "_get_library",
        lambda: (_ for _ in ()).throw(NativeRendererError("missing /private/user/native.dylib")),
    )
    status = native_renderer.get_native_physical_status()
    assert status == {"available": False, "backend": None,
                      "gpu_available": False, "cpu_available": False}
    assert "/private" not in repr(status)


@pytest.mark.parametrize("which", ("source", "transmission", "airlight", "params"))
def test_physical_float_bridge_rejects_invalid_inputs_before_loading_native(which, monkeypatch):
    def unexpected_load():
        pytest.fail("Invalid input must be rejected before attempting to load a native library")

    monkeypatch.setattr(native_renderer, "_get_library", unexpected_load)
    source, transmission, airlight = _inputs("random")
    params = _params()
    if which == "source":
        source[0, 0, 0] = np.nan
        error = ValueError
    elif which == "transmission":
        transmission[0, 0] = 1.01
        error = ValueError
    elif which == "airlight":
        airlight[0] = -0.01
        error = ValueError
    else:
        params = {"strength": 0.5}
        error = ValueError
    with pytest.raises(error):
        native_physical_dehaze(source, params, transmission, airlight)


def test_zero_strength_metal_physical_float_is_an_exact_copy():
    requested = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND")
    if requested and requested.lower() != "metal":
        pytest.skip(f"Physical float parity is implemented for Metal, not {requested}")
    source, transmission, airlight = _inputs("cast-nearwhite")
    params = DehazeParams(strength=0.0)
    try:
        actual = native_physical_dehaze(source, params, transmission, airlight)
    except NativeRendererError as exc:
        if requested and requested.lower() == "metal":
            pytest.fail(f"Requested Metal physical float operator failed: {exc}")
        pytest.skip(f"Metal physical float renderer is unavailable: {exc}")
    np.testing.assert_array_equal(actual, source)
