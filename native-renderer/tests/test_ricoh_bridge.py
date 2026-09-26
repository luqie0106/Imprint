from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import native_renderer  # noqa: E402
from ricoh_filter import apply_ricoh_preview_effect  # noqa: E402


@pytest.fixture
def native_renderer_handle():
    try:
        library, _ = native_renderer._get_library()
        handle, backend = native_renderer._create_renderer(library)
    except native_renderer.NativeRendererError as exc:
        pytest.skip(f"Native renderer is unavailable: {exc}")
    requested = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND")
    if requested and backend.lower() != requested.lower():
        library.im_renderer_destroy(handle)
        pytest.fail(f"Requested native backend {requested}, but created {backend}")
    try:
        yield library, handle, backend
    finally:
        library.im_renderer_destroy(handle)


def _copy_output(library, handle, width: int, height: int) -> np.ndarray:
    out_width = native_renderer.ctypes.c_uint32()
    out_height = native_renderer.ctypes.c_uint32()
    count = native_renderer.ctypes.c_size_t()
    assert library.im_renderer_get_output_size(
        handle, native_renderer.ctypes.byref(out_width),
        native_renderer.ctypes.byref(out_height), native_renderer.ctypes.byref(count)
    ) == 0
    assert (out_width.value, out_height.value, count.value) == (width, height, width * height * 3)
    output = np.empty((height, width, 3), dtype=np.uint16)
    pointer = output.ctypes.data_as(native_renderer.ctypes.POINTER(native_renderer.ctypes.c_uint16))
    assert library.im_renderer_copy_output(handle, pointer, output.size) == 0
    return output


def test_c_abi_ricoh_validates_lut_and_applies_it(native_renderer_handle):
    library, handle, _ = native_renderer_handle
    image = np.array(
        [[[12000, 30000, 52000], [42000, 18000, 9000]],
         [[25000, 48000, 16000], [60000, 22000, 39000]]],
        dtype=np.uint16,
    )
    image_pointer = image.ctypes.data_as(native_renderer.ctypes.POINTER(native_renderer.ctypes.c_uint16))
    edge = 2
    lut = np.empty(edge**3 * 3, dtype=np.uint16)
    for red in range(edge):
        for green in range(edge):
            for blue in range(edge):
                at = ((red * edge + green) * edge + blue) * 3
                lut[at:at + 3] = ((1 - red) * 65535, green * 65535, blue * 65535)
    lut_pointer = lut.ctypes.data_as(native_renderer.ctypes.POINTER(native_renderer.ctypes.c_uint16))

    assert library.im_renderer_render_ricoh_full(
        handle, 2, 2, image_pointer, image.size, lut_pointer, lut.size - 1, edge
    ) == 1
    assert library.im_renderer_render_ricoh_full(
        handle, 2, 2, image_pointer, image.size, lut_pointer, lut.size, 1
    ) == 1
    assert library.im_renderer_render_ricoh_full(
        handle, 2, 2, image_pointer, image.size - 1, lut_pointer, lut.size, edge
    ) == 1

    assert library.im_renderer_render_ricoh_full(
        handle, 2, 2, image_pointer, image.size, lut_pointer, lut.size, edge
    ) == 0
    output = _copy_output(library, handle, 2, 2)
    assert np.any(output != image)
    np.testing.assert_array_equal(image, np.array(
        [[[12000, 30000, 52000], [42000, 18000, 9000]],
         [[25000, 48000, 16000], [60000, 22000, 39000]]], dtype=np.uint16
    ))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_native_ricoh_preserves_dtype_shape_and_input(dtype, native_renderer_handle):
    _, _, backend = native_renderer_handle
    del backend
    rng = np.random.default_rng(728)
    image = rng.integers(0, np.iinfo(dtype).max + 1, size=(7, 9, 3), dtype=dtype)
    original = image.copy()
    output = native_renderer.native_ricoh(image, "gr3_positive_film")
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    np.testing.assert_array_equal(image, original)


def test_ricoh_lut_cache_separates_parameters_and_measured_color(monkeypatch):
    native_renderer._ricoh_lut_cache.clear()
    monkeypatch.setattr(native_renderer, "_RICOH_LUT_EDGE", 3)
    basic = dict.fromkeys(native_renderer._BASIC_FIELDS, 0.0)
    first = native_renderer._ricoh_lut("gr3_standard", basic, False)
    first_key = next(iter(native_renderer._ricoh_lut_cache))
    again = native_renderer._ricoh_lut("gr3_standard", basic, False)
    adjusted = {**basic, "exposure": 0.5}
    native_renderer._ricoh_lut("gr3_standard", adjusted, False)
    native_renderer._ricoh_lut("gr3_standard", basic, True)
    for exposure in (0.25, 0.75, 1.0):
        native_renderer._ricoh_lut("gr3_standard", {**basic, "exposure": exposure}, False)

    assert first is again
    assert len(native_renderer._ricoh_lut_cache) == native_renderer._RICOH_LUT_CACHE_LIMIT
    assert first_key not in native_renderer._ricoh_lut_cache
    assert first.flags.writeable is False


def test_native_ricoh_unknown_preset_keeps_key_error():
    with pytest.raises(KeyError):
        native_renderer.native_ricoh(np.zeros((1, 1, 3), dtype=np.uint8), "unknown-preset")


def test_native_ricoh_smoke_and_python_lut_error_metrics(native_renderer_handle):
    _, _, backend = native_renderer_handle
    if backend not in {"Metal", "D3D12"}:
        pytest.skip(f"Ricoh LUT comparison is supported for Metal/D3D12, selected {backend}")

    rng = np.random.default_rng(991)
    basic = dict.fromkeys(native_renderer._BASIC_FIELDS, 0.0)
    basic["exposure"] = 0.25
    metrics = {}
    for dtype in (np.uint8, np.uint16):
        image = rng.integers(0, np.iinfo(dtype).max + 1, size=(20, 24, 3), dtype=dtype)
        measured = dtype is np.uint8
        native = native_renderer.native_ricoh(
            image, "gr3_positive_film", basic, use_measured_color=measured
        )
        python = apply_ricoh_preview_effect(
            image, "gr3_positive_film", basic, use_measured_color=measured
        )
        absolute_error = np.abs(native.astype(np.int32) - python.astype(np.int32))
        mae = float(absolute_error.mean())
        p99 = float(np.percentile(absolute_error, 99))
        maximum = int(absolute_error.max())
        metrics[f"{np.dtype(dtype).name} (measured={measured})"] = (mae, p99, maximum)
        # LUT interpolation approximates nonlinear per-pixel Python processing;
        # these broad limits catch channel/order failures without requiring parity.
        assert mae < (10.0 if dtype is np.uint8 else 1800.0)
        assert p99 < (25.0 if dtype is np.uint8 else 9000.0)
    print(f"{backend} Ricoh vs Python MAE/P99/Max: {metrics}")
