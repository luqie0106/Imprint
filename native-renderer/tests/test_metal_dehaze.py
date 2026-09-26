from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dehaze import DehazeParams, apply_dehaze  # noqa: E402
from test_dehaze_reference import NativeParams, _scenes  # noqa: E402


class NativeBasicParams(ctypes.Structure):
    _fields_ = [
        ("exposure", ctypes.c_float),
        ("contrast", ctypes.c_float),
        ("highlights", ctypes.c_float),
        ("shadows", ctypes.c_float),
        ("whites", ctypes.c_float),
        ("blacks", ctypes.c_float),
        ("vibrance", ctypes.c_float),
        ("saturation", ctypes.c_float),
    ]


class NativeFilterParams(ctypes.Structure):
    _fields_ = [
        ("curve_mix", ctypes.c_float),
        ("lut_mix", ctypes.c_float),
        ("exposure_ev", ctypes.c_float),
        ("contrast", ctypes.c_float),
        ("saturation", ctypes.c_float),
        ("warmth", ctypes.c_float),
        ("tint", ctypes.c_float),
        ("fade", ctypes.c_float),
    ]


def _configure_library(library: ctypes.CDLL) -> None:
    renderer = ctypes.c_void_p
    uint16_pointer = ctypes.POINTER(ctypes.c_uint16)
    library.im_renderer_create.argtypes = [ctypes.c_int, ctypes.POINTER(renderer)]
    library.im_renderer_create.restype = ctypes.c_int
    library.im_renderer_destroy.argtypes = [renderer]
    library.im_renderer_destroy.restype = None
    library.im_renderer_last_error.argtypes = [renderer]
    library.im_renderer_last_error.restype = ctypes.c_char_p
    library.im_renderer_backend_name.argtypes = [renderer]
    library.im_renderer_backend_name.restype = ctypes.c_char_p
    library.im_renderer_upload_preview_image.argtypes = [
        renderer, ctypes.c_uint32, ctypes.c_uint32, uint16_pointer, ctypes.c_size_t
    ]
    library.im_renderer_upload_preview_image.restype = ctypes.c_int
    library.im_renderer_upload_filter.argtypes = [
        renderer,
        ctypes.POINTER(NativeFilterParams),
        uint16_pointer,
        ctypes.c_size_t,
        uint16_pointer,
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    library.im_renderer_upload_filter.restype = ctypes.c_int
    library.im_renderer_render.argtypes = [
        renderer, ctypes.c_int, ctypes.POINTER(NativeParams), ctypes.POINTER(NativeBasicParams)
    ]
    library.im_renderer_render.restype = ctypes.c_int
    library.im_renderer_get_output_size.argtypes = [
        renderer,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.im_renderer_get_output_size.restype = ctypes.c_int
    library.im_renderer_copy_output.argtypes = [renderer, uint16_pointer, ctypes.c_size_t]
    library.im_renderer_copy_output.restype = ctypes.c_int


def _check_status(library: ctypes.CDLL, renderer: ctypes.c_void_p, status: int, operation: str) -> None:
    if status != 0:
        detail = library.im_renderer_last_error(renderer)
        message = detail.decode("utf-8", errors="replace") if detail else "unknown native error"
        pytest.fail(f"{operation} failed with status {status}: {message}")


@pytest.fixture(scope="module")
def native_renderer():
    library_path = os.environ.get("IMPRINT_NATIVE_RENDERER_LIB")
    if not library_path:
        pytest.fail("Set IMPRINT_NATIVE_RENDERER_LIB to the CMake-built native renderer library")

    requested_backend = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND", "Metal").strip().lower()
    backend_kinds = {"metal": (1, "Metal"), "cuda": (2, "CUDA"), "d3d12": (3, "D3D12")}
    if requested_backend not in backend_kinds:
        pytest.fail("IMPRINT_NATIVE_RENDERER_BACKEND must be Metal, CUDA, or D3D12")
    backend_kind, expected_backend = backend_kinds[requested_backend]

    library = ctypes.CDLL(library_path)
    _configure_library(library)
    renderer = ctypes.c_void_p()
    # Explicit backend selection prevents an accidental fallback from passing.
    status = library.im_renderer_create(backend_kind, ctypes.byref(renderer))
    if status != 0:
        detail = library.im_renderer_last_error(None)
        message = detail.decode("utf-8", errors="replace") if detail else "unknown native error"
        pytest.fail(f"Could not create the {expected_backend} renderer (status {status}): {message}")

    backend = library.im_renderer_backend_name(renderer)
    if backend != expected_backend.encode("ascii"):
        library.im_renderer_destroy(renderer)
        pytest.fail(f"Expected the {expected_backend} backend, got {backend!r}")

    neutral_filter = NativeFilterParams(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    _check_status(
        library,
        renderer,
        library.im_renderer_upload_filter(renderer, ctypes.byref(neutral_filter), None, 0, None, 0, 0),
        "upload neutral filter",
    )
    try:
        yield library, renderer, expected_backend
    finally:
        library.im_renderer_destroy(renderer)


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
def test_native_rgb16_color_difference(scene_name: str, params: DehazeParams, native_renderer):
    library, renderer, backend = native_renderer
    image = np.ascontiguousarray(_scenes()[scene_name])
    original = image.copy()
    native_params = NativeParams(*(getattr(params, field) for field in DehazeParams.__dataclass_fields__))
    basic = NativeBasicParams()
    uint16_pointer = ctypes.POINTER(ctypes.c_uint16)

    _check_status(
        library,
        renderer,
        library.im_renderer_upload_preview_image(
            renderer,
            image.shape[1],
            image.shape[0],
            image.ctypes.data_as(uint16_pointer),
            image.size,
        ),
        "upload RGB16 test image",
    )
    _check_status(
        library,
        renderer,
        library.im_renderer_render(renderer, 0, ctypes.byref(native_params), ctypes.byref(basic)),
        f"render with {backend}",
    )

    width = ctypes.c_uint32()
    height = ctypes.c_uint32()
    value_count = ctypes.c_size_t()
    _check_status(
        library,
        renderer,
        library.im_renderer_get_output_size(
            renderer, ctypes.byref(width), ctypes.byref(height), ctypes.byref(value_count)
        ),
        f"get {backend} output size",
    )
    assert (width.value, height.value) == (image.shape[1], image.shape[0])
    assert value_count.value == image.size
    native = np.empty_like(image)
    _check_status(
        library,
        renderer,
        library.im_renderer_copy_output(
            renderer, native.ctypes.data_as(uint16_pointer), native.size
        ),
        f"copy {backend} output",
    )
    python = apply_dehaze(image, params, backend="cpu")
    assert np.array_equal(image, original), f"{backend}/Python comparison must not modify the source"

    signed = native.astype(np.int32) - python.astype(np.int32)
    absolute = np.abs(signed)
    channel_mae = absolute.mean(axis=(0, 1))
    channel_bias = signed.mean(axis=(0, 1))
    channel_max = absolute.max(axis=(0, 1))
    mae = float(absolute.mean())
    p99 = float(np.percentile(absolute, 99))
    maximum = int(absolute.max())
    exact_pixels = float(np.all(absolute == 0, axis=2).mean() * 100.0)

    print(
        f"{scene_name} params={params}: MAE={mae:.4f}, P99={p99:.2f}, max={maximum}, "
        f"exact_pixels={exact_pixels:.2f}%, "
        f"channel_MAE_RGB={np.round(channel_mae, 4).tolist()}, "
        f"channel_bias_RGB={np.round(channel_bias, 4).tolist()}, "
        f"channel_max_RGB={channel_max.tolist()}"
    )

    # Match the tolerances already used for the C++ reference vs Python check.
    assert mae <= 1.0
    assert p99 <= 4.0
    assert maximum <= 64
