from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dehaze import DehazeParams, _atmospheric_light, apply_dehaze


class NativeParams(ctypes.Structure):
    _fields_ = [
        ("strength", ctypes.c_float),
        ("naturalness", ctypes.c_float),
        ("fog_retention", ctypes.c_float),
        ("local_contrast", ctypes.c_float),
        ("color_recovery", ctypes.c_float),
        ("color_protection", ctypes.c_float),
        ("highlight_protection", ctypes.c_float),
        ("shadow_protection", ctypes.c_float),
        ("brightness_protection", ctypes.c_float),
    ]


def _native_library():
    library_path = os.environ.get("IMPRINT_NATIVE_REFERENCE_LIB")
    if not library_path:
        pytest.fail("Set IMPRINT_NATIVE_REFERENCE_LIB to the CMake-built reference test library")
    library = ctypes.CDLL(library_path)
    function = library.im_native_dehaze_reference_run
    function.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(NativeParams),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = ctypes.c_int
    return function


def _rgb16(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-0.00045, 0.00045, values.shape).astype(np.float32)
    return np.clip(np.rint((values + noise) * 65535.0), 0, 65535).astype(np.uint16)


def _scenes() -> dict[str, np.ndarray]:
    h, w = 96, 128
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    x = xx / np.float32(w - 1)
    y = yy / np.float32(h - 1)

    haze = np.empty((h, w, 3), dtype=np.float32)
    haze[:] = (0.17, 0.23, 0.19)
    haze += x[..., None] * np.array((0.16, 0.12, 0.19), dtype=np.float32)
    haze += y[..., None] * np.array((0.12, 0.09, 0.06), dtype=np.float32)
    haze = haze * np.float32(0.56) + np.float32(0.40)

    sky_sea = np.empty((h, w, 3), dtype=np.float32)
    sky_sea[:58] = (0.47, 0.53, 0.59)
    sky_sea[58:] = (0.42, 0.49, 0.54)
    sky_sea += x[..., None] * np.array((0.011, 0.007, 0.004), dtype=np.float32)
    sky_sea[58:] += np.array((0.012, 0.009, -0.003), dtype=np.float32)
    sky_sea[:13] += np.array((0.19, 0.18, 0.17), dtype=np.float32)

    backlit = np.empty((h, w, 3), dtype=np.float32)
    backlit[:] = (0.69, 0.75, 0.83)
    backlit += y[..., None] * np.array((0.06, 0.05, 0.035), dtype=np.float32)
    backlit[58:] = (0.09, 0.11, 0.14)
    backlit[55:60] = (0.63, 0.69, 0.77)
    backlit[62:85, 22:50] = (0.14, 0.16, 0.18)

    distance = np.sqrt((xx - np.float32(84.0)) ** 2 + (yy - np.float32(28.0)) ** 2)
    solar = np.empty((h, w, 3), dtype=np.float32)
    solar[:] = (0.76, 0.80, 0.84)
    halo = np.exp(-((distance / np.float32(18.0)) ** 2))[..., None]
    solar += halo * np.array((0.20, 0.17, 0.12), dtype=np.float32)
    solar[distance < 5.0] = (1.0, 0.996, 0.985)
    solar[62:] = (0.11, 0.13, 0.16)

    dark_midtones = np.empty((h, w, 3), dtype=np.float32)
    dark_midtones[:] = (0.198, 0.204, 0.211)
    dark_midtones += (x[..., None] - 0.5) * np.array((0.018, 0.015, 0.012), dtype=np.float32)
    dark_midtones += y[..., None] * np.array((0.004, 0.003, 0.002), dtype=np.float32)

    return {
        "typical_haze": _rgb16(haze, 301),
        "low_saturation_sky_sea": _rgb16(sky_sea, 302),
        "dark_foreground_bright_background": _rgb16(backlit, 303),
        "near_saturated_sun_halo": _rgb16(solar, 304),
        "brightness_guard_dark_midtones": _rgb16(dark_midtones, 305),
    }


def _run_reference(function, image: np.ndarray, params: DehazeParams):
    output = np.empty_like(image)
    native_params = NativeParams(*vars(params).values())
    stats = (ctypes.c_float * 5)()
    status = function(
        image.shape[1],
        image.shape[0],
        image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.byref(native_params),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        stats,
    )
    assert status == 0
    return output, np.asarray(stats, dtype=np.float32)


def _expected_stats(image: np.ndarray) -> tuple[np.ndarray, float]:
    source = image.astype(np.float32) / np.float32(65535.0)
    atmosphere = _atmospheric_light(source)
    dark_reference = float(np.percentile(np.min(source, axis=2), 75))
    haze_level = float(np.clip((dark_reference - 0.03) / 0.92, 0.12, 0.92))
    return atmosphere, haze_level


@pytest.mark.parametrize(
    ("scene_name", "params"),
    [
        ("typical_haze", DehazeParams(strength=0.82, local_contrast=0.35, color_recovery=0.55)),
        ("low_saturation_sky_sea", DehazeParams(strength=0.45, local_contrast=0.15, color_recovery=1.0)),
        ("dark_foreground_bright_background", DehazeParams(strength=0.90, local_contrast=0.4)),
        ("near_saturated_sun_halo", DehazeParams(strength=0.95, local_contrast=1.0)),
        ("typical_haze", DehazeParams(strength=0.0, local_contrast=1.0)),
        ("typical_haze", DehazeParams(strength=0.82, color_recovery=0.0, local_contrast=0.35)),
        ("typical_haze", DehazeParams(strength=0.82, local_contrast=1.0)),
        (
            "brightness_guard_dark_midtones",
            DehazeParams(
                strength=1.0,
                naturalness=0.0,
                fog_retention=0.0,
                local_contrast=0.0,
                color_recovery=0.0,
                color_protection=0.0,
                highlight_protection=0.0,
                shadow_protection=0.0,
                brightness_protection=1.0,
            ),
        ),
        (
            "brightness_guard_dark_midtones",
            DehazeParams(
                strength=1.0,
                naturalness=0.0,
                fog_retention=0.0,
                local_contrast=0.0,
                color_recovery=0.0,
                color_protection=0.0,
                highlight_protection=0.0,
                shadow_protection=0.0,
                brightness_protection=0.0,
            ),
        ),
    ],
)
def test_cpp_reference_matches_python_rgb16(scene_name: str, params: DehazeParams):
    function = _native_library()
    image = _scenes()[scene_name]
    untouched = image.copy()
    cpp, stats = _run_reference(function, image, params)
    python = apply_dehaze(image, params, backend="cpu")

    absolute_error = np.abs(cpp.astype(np.int32) - python.astype(np.int32))
    mae = float(absolute_error.mean())
    p99 = float(np.percentile(absolute_error, 99))
    maximum = int(absolute_error.max())
    print(f"{scene_name} {params}: MAE={mae:.4f}, P99={p99:.2f}, max={maximum}, gain={stats[4]:.7f}")

    expected_air, expected_haze = _expected_stats(image)
    assert float(np.max(np.abs(stats[:3] - expected_air))) <= 3.0 / 65535.0
    assert abs(float(stats[3]) - expected_haze) <= 3e-6
    assert mae <= 1.0
    assert p99 <= 4.0
    assert maximum <= 64
    if scene_name == "brightness_guard_dark_midtones" and params.brightness_protection > 0:
        assert stats[4] > 1.0
    assert np.array_equal(image, untouched)
