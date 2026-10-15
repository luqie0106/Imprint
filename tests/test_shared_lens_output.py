from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lens_correction as lens
import native_dense
from native_renderer import NativeRendererError


class _Lens:
    model = "test lens"
    calib_distortion = [object()]
    calib_tca = [object()]
    calib_vignetting = []


class _Modifier:
    def apply_subpixel_geometry_distortion(self, xu, yu, width, height):
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        maps = np.empty((height, width, 3, 2), dtype=np.float32)
        for channel in range(3):
            maps[:, :, channel, 0] = x
            maps[:, :, channel, 1] = y + yu
        return maps


def _install_plan(monkeypatch):
    modifier = _Modifier()
    plan = types.SimpleNamespace(
        camera=types.SimpleNamespace(model="test camera"),
        lens=_Lens(),
        modifier=modifier,
        flags={"TCA": 1, "DISTORTION": 2, "SCALE": 4, "VIGNETTING": 8},
        enabled=3,
        scale=1.0,
        lock=threading.RLock(),
    )
    monkeypatch.setattr(lens, "_lens_plan", lambda *args: plan)
    monkeypatch.setitem(sys.modules, "lensfunpy", types.SimpleNamespace())
    return plan


def _metadata():
    return {
        "Make": "Test",
        "Model": "Test camera",
        "LensMake": "Test",
        "LensModel": "test lens",
        "FocalLength": 35.0,
        "FNumber": 4.0,
    }


def _shared_bridge(monkeypatch, source, *, enabled=True, allocation_error=False):
    owner = object()
    allocations = []
    bridge = types.ModuleType("native_shared")
    bridge.shared_memory_enabled = lambda: enabled
    bridge.shared_memory_eligible = lambda shape, dtype=None: True
    bridge.shared_buffer_owner = lambda value: owner if value is source else None

    def allocate(shape, dtype=np.uint16):
        if allocation_error:
            raise NativeRendererError("shared allocation unavailable")
        output = np.empty(shape, dtype=dtype)
        allocations.append(output)
        return output

    bridge.allocate_shared_rgb = allocate
    monkeypatch.setitem(sys.modules, "native_shared", bridge)
    return allocations


def _install_destination_remap(monkeypatch, destinations, sources=None):
    def remap(source, maps, *, destination=None):
        assert destination is not None
        assert destination.dtype == np.uint16
        assert destination.flags.c_contiguous
        destinations.append(destination)
        if sources is not None:
            sources.append(source)
        start_y = int(maps[0, 0, 0, 1])
        destination[...] = start_y + 7

    monkeypatch.setattr(native_dense, "lens_remap", remap)


def test_shared_input_copies_into_distinct_output_and_remaps_directly_to_strips(monkeypatch):
    _install_plan(monkeypatch)
    image = np.full((270, 5, 3), 1234, dtype=np.uint16)
    original = image.copy()
    allocations = _shared_bridge(monkeypatch, image)
    destinations = []
    sources = []
    _install_destination_remap(monkeypatch, destinations, sources)

    corrected, result = lens.apply_lens_correction(image, _metadata())

    assert result.applied
    assert np.array_equal(image, original)
    assert len(allocations) == 1
    assert corrected is allocations[0]
    assert corrected is not image
    assert not np.shares_memory(corrected, image)
    assert corrected.dtype == np.uint16
    assert corrected.flags.c_contiguous
    assert len(sources) == 3
    assert all(source is not image and not np.shares_memory(source, corrected) for source in sources)
    assert all(np.array_equal(source, original) for source in sources)
    assert [destination.shape[0] for destination in destinations] == [128, 128, 14]
    assert all(destination.flags.c_contiguous for destination in destinations)
    assert [destination.ctypes.data for destination in destinations] == [
        corrected[0:128].ctypes.data,
        corrected[128:256].ctypes.data,
        corrected[256:270].ctypes.data,
    ]
    assert [int(corrected[y, 0, 0]) for y in (0, 128, 256)] == [7, 135, 263]


@pytest.mark.parametrize("mode", ["ordinary", "disabled", "allocation_error"])
def test_non_shared_or_disabled_path_uses_regular_independent_copy(monkeypatch, mode):
    _install_plan(monkeypatch)
    image = np.full((12, 4, 3), 999, dtype=np.uint16)
    original = image.copy()
    allocations = []
    if mode == "ordinary":
        allocations = _shared_bridge(monkeypatch, np.zeros_like(image))
    elif mode == "disabled":
        allocations = _shared_bridge(monkeypatch, image, enabled=False)
    else:
        allocations = _shared_bridge(monkeypatch, image, allocation_error=True)
    destinations = []
    _install_destination_remap(monkeypatch, destinations)

    corrected, result = lens.apply_lens_correction(image, _metadata())

    assert result.applied
    assert np.array_equal(image, original)
    assert corrected is not image
    assert not np.shares_memory(corrected, image)
    assert corrected.flags.c_contiguous
    assert allocations == []
    assert len(destinations) == 1


def test_opencv_fallback_still_writes_the_expected_geometry(monkeypatch):
    source = np.arange(6 * 7 * 3, dtype=np.uint16).reshape(6, 7, 3) * 200
    original = source.copy()
    corrected = source.copy()

    class _Map:
        def apply_subpixel_geometry_distortion(self, xu, yu, width, height):
            return None

        def apply_geometry_distortion(self, xu, yu, width, height):
            y, x = np.mgrid[0:height, 0:width].astype(np.float32)
            return np.stack((x + np.float32(0.25), y + np.float32(yu)), axis=-1)

    def unavailable(*args, **kwargs):
        raise NativeRendererError("test native kernel unavailable")

    monkeypatch.setattr(native_dense, "lens_remap", unavailable)
    geometry = _Map()

    applied = lens._apply_geometry(
        corrected,
        geometry,
        width=source.shape[1],
        height=source.shape[0],
        geometry_enabled=True,
        tca_enabled=False,
        distortion_enabled=True,
    )

    maps = geometry.apply_geometry_distortion(0.0, 0.0, source.shape[1], source.shape[0])
    expected = np.stack(
        [
            cv2.remap(
                source[..., channel],
                maps[..., 0],
                maps[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    assert applied == (True, False, True)
    assert np.array_equal(corrected, expected)
    assert np.array_equal(source, original)
