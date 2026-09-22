from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lens_correction import (
    LensCorrectionNotAppliedError,
    LensMatchError,
    LensfunUnavailableError,
    apply_lens_correction,
)


class _Camera:
    maker = "Nikon"
    model = "Z6_3"
    crop_factor = 1.0


class _Lens:
    maker = "NIKON"
    model = "NIKKOR Z 70-180mm f/2.8"
    calib_distortion = [object()]
    calib_tca = [object()]


class _Modifier:
    instances = []

    def __init__(self, lens, crop, width, height):
        self.width = width
        self.height = height
        self.initialize_calls = []
        self.map_calls = []
        self.__class__.instances.append(self)

    def initialize(self, focal_length, aperture, distance, *, pixel_format, flags, scale):
        self.initialize_calls.append((focal_length, aperture, distance, pixel_format, flags, scale))

    def get_auto_scale(self, reverse=False):
        return 1.15

    def apply_subpixel_geometry_distortion(self, xu, yu, width, height):
        self.map_calls.append((xu, yu, width, height))
        # Identity maps plus one pixel in x make the baked correction visible.
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        x += 1.0
        result = np.empty((height, width, 3, 2), dtype=np.float32)
        for channel in range(3):
            result[:, :, channel, 0] = x
            result[:, :, channel, 1] = y + yu
        return result


class _Database:
    instances = []

    def __init__(self, *, paths, load_bundled):
        self.paths = paths
        self.load_bundled = load_bundled
        self.camera = _Camera()
        self.lens = _Lens()
        self.modifier = None
        self.__class__.instances.append(self)

    def find_cameras(self, maker, model, loose_search=True):
        return [self.camera]

    def find_lenses(self, camera, maker, model, loose_search=True):
        return [self.lens]


def _fake_lensfun(monkeypatch, *, geometry=True):
    flags = types.SimpleNamespace(TCA=1, DISTORTION=2, SCALE=4, VIGNETTING=8)
    modifier = _Modifier if geometry else type(
        "VignettingOnlyModifier",
        (_Modifier,),
        {"apply_subpixel_geometry_distortion": lambda self, *args: None,
         "apply_geometry_distortion": lambda self, *args: None},
    )
    fake = types.SimpleNamespace(Database=_Database, Modifier=modifier, ModifyFlags=flags)
    monkeypatch.setitem(sys.modules, "lensfunpy", fake)
    return fake


def _metadata():
    return {
        "Make": "Nikon",
        "Model": "Z6_3",
        "LensMake": "NIKON",
        "LensModel": "NIKKOR Z 70-180mm f/2.8",
        "FocalLength": 70.0,
        "FNumber": 4.5,
        "SubjectDistance": 4.0,
    }


def test_applies_uint16_in_strips_and_does_not_mutate_input(monkeypatch):
    _fake_lensfun(monkeypatch)
    image = np.arange(3 * 300 * 3, dtype=np.uint16).reshape(300, 3, 3)
    original = image.copy()

    corrected, result = apply_lens_correction(image, _metadata())

    assert corrected.shape == image.shape
    assert corrected.dtype == np.uint16
    assert np.array_equal(image, original)
    assert not np.array_equal(corrected, image)
    assert result.applied
    assert result.distortion_applied
    assert result.tca_applied
    assert not result.vignetting_applied
    # 300 rows must be split into no more than the configured strip height.
    calls = _Modifier.instances[-1].map_calls
    assert [call[3] for call in calls] == [128, 128, 44]
    assert all(call[2] == 3 for call in calls)
    assert _Modifier.instances[-1].initialize_calls[0][-1] == 0.0


def test_missing_match_raises_in_strict_mode(monkeypatch):
    fake = _fake_lensfun(monkeypatch)

    class EmptyDatabase(_Database):
        def find_cameras(self, maker, model, loose_search=True):
            return []

    fake.Database = EmptyDatabase
    with pytest.raises(LensMatchError):
        apply_lens_correction(np.zeros((4, 4, 3), dtype=np.uint16), _metadata())


def test_missing_dependency_raises(monkeypatch):
    monkeypatch.delitem(sys.modules, "lensfunpy", raising=False)
    # Import machinery may still find a real installation; force the import
    # path to fail for this isolated test.
    import builtins

    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "lensfunpy":
            raise ImportError("test dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(LensfunUnavailableError):
        apply_lens_correction(np.zeros((4, 4, 3), dtype=np.uint16), _metadata())


def test_invalid_optics_raise_before_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "lensfunpy", raising=False)
    metadata = _metadata()
    metadata["FocalLength"] = 0
    with pytest.raises(Exception, match="FocalLength"):
        apply_lens_correction(np.zeros((4, 4, 3), dtype=np.uint16), metadata)


def test_only_vignetting_or_no_geometry_is_not_sufficient(monkeypatch):
    _fake_lensfun(monkeypatch, geometry=False)
    with pytest.raises(LensCorrectionNotAppliedError):
        apply_lens_correction(np.zeros((4, 4, 3), dtype=np.uint16), _metadata())
