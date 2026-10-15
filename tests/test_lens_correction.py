from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lens_correction import (
    LensCorrectionNotAppliedError,
    LensMetadataUnavailableError,
    LensMatchError,
    LensfunUnavailableError,
    _apply_geometry,
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
    calib_vignetting = [object()]


class _Modifier:
    instances = []
    color_calls = []

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

    def apply_color_modification(self, image):
        self.__class__.color_calls.append((image.dtype, image.shape))
        image *= 2.0
        return True


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


class _L2D20cCamera:
    maker = "Hasselblad"
    model = "L2D-20c"
    crop_factor = 1.953


class _L2D20cLens:
    maker = "Hasselblad"
    model = "L2D-20c & compatibles"
    calib_distortion = [object()]
    calib_tca = []
    calib_vignetting = []


class _L2D20cDatabase:
    instances = []

    def __init__(self, *, paths, load_bundled):
        self.paths = paths
        self.load_bundled = load_bundled
        self.camera = _L2D20cCamera()
        self.lens = _L2D20cLens()
        self.lens_queries = []
        self.__class__.instances.append(self)

    def find_cameras(self, maker, model, loose_search=True):
        return [self.camera]

    def find_lenses(self, camera, maker, model, loose_search=True):
        self.lens_queries.append((maker, model))
        if model == self.lens.model and maker == self.lens.maker:
            return [self.lens]
        return []


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


def _l2d20c_metadata(lens_model):
    return {
        "Make": "Hasselblad",
        "Model": "L2D-20c",
        "LensMake": "Hasselblad",
        "LensModel": lens_model,
        "FocalLength": 12.29,
        "FNumber": 2.8,
        "SubjectDistance": 4.0,
    }


def test_applies_vignetting_and_uint16_geometry_without_mutating_input(monkeypatch):
    _fake_lensfun(monkeypatch)
    image = np.full((300, 3, 3), 10000, dtype=np.uint16)
    original = image.copy()

    corrected, result = apply_lens_correction(image, _metadata())

    assert corrected.shape == image.shape
    assert corrected.dtype == np.uint16
    assert np.array_equal(image, original)
    assert np.all(corrected == 20000)
    assert result.applied
    assert result.distortion_applied
    assert result.tca_applied
    assert result.vignetting_applied
    assert _Modifier.color_calls[-1] == (np.dtype(np.float32), image.shape)
    # 300 rows must be split into no more than the configured strip height.
    calls = _Modifier.instances[-1].map_calls
    assert [call[3] for call in calls] == [128, 128, 44]
    assert all(call[2] == 3 for call in calls)
    initialize_call = _Modifier.instances[-1].initialize_calls[0]
    assert initialize_call[3] is np.float32
    assert initialize_call[4] & 8  # VIGNETTING
    assert _Modifier.instances[-1].initialize_calls[0][-1] == 0.0


@pytest.mark.parametrize("map_path", ["combined", "geometry_fallback"])
def test_geometry_remap_avoids_negative_ringing_around_point_lights(map_path):
    class SubpixelMapModifier:
        def apply_subpixel_geometry_distortion(self, xu, yu, width, height):
            if map_path != "combined":
                return None
            y, x = np.mgrid[0:height, 0:width].astype(np.float32)
            x += 0.25
            y += yu
            maps = np.empty((height, width, 3, 2), dtype=np.float32)
            for channel in range(3):
                maps[:, :, channel, 0] = x
                maps[:, :, channel, 1] = y
            return maps

        def apply_geometry_distortion(self, xu, yu, width, height):
            if map_path != "geometry_fallback":
                return None
            y, x = np.mgrid[0:height, 0:width].astype(np.float32)
            x += 0.25
            y += yu
            return np.stack((x, y), axis=-1)

    source = np.full((8, 8, 3), 3047, dtype=np.uint16)
    source[4, 4, 1] = 60000
    original = source.copy()
    corrected = source.copy()

    distortion_applied, tca_applied, geometry_applied = _apply_geometry(
        corrected,
        SubpixelMapModifier(),
        width=8,
        height=8,
        geometry_enabled=True,
        tca_enabled=True,
        distortion_enabled=True,
    )

    assert geometry_applied
    assert distortion_applied
    assert tca_applied is (map_path == "combined")
    assert corrected.shape == source.shape
    assert corrected.dtype == source.dtype
    assert np.array_equal(source, original)
    assert corrected.min() >= source.min()
    assert corrected.max() <= source.max()
    assert not np.array_equal(corrected, source)


def test_profile_without_vignetting_calibration_skips_color_modification(monkeypatch):
    fake = _fake_lensfun(monkeypatch)

    class NoVignettingDatabase(_Database):
        def __init__(self, *, paths, load_bundled):
            super().__init__(paths=paths, load_bundled=load_bundled)
            self.lens.calib_vignetting = ()

    fake.Database = NoVignettingDatabase
    _Modifier.color_calls.clear()
    image = np.full((4, 4, 3), 10000, dtype=np.uint16)
    original = image.copy()

    corrected, result = apply_lens_correction(image, _metadata())

    assert np.array_equal(image, original)
    assert np.array_equal(corrected, original)
    assert _Modifier.color_calls == []
    assert result.applied
    assert not result.vignetting_applied


@pytest.mark.parametrize("lens_model", ["24.0 mm f/2.8", ""])
def test_l2d20c_fixed_lens_maps_to_lensfun_profile(monkeypatch, lens_model):
    fake = _fake_lensfun(monkeypatch)
    fake.Database = _L2D20cDatabase
    image = (10000 + np.arange(8 * 8 * 3, dtype=np.uint16)).reshape(8, 8, 3)
    original = image.copy()

    corrected, result = apply_lens_correction(image, _l2d20c_metadata(lens_model))

    assert result.applied
    assert result.distortion_applied
    assert result.lens_name == "L2D-20c & compatibles"
    assert np.array_equal(image, original)
    queries = _L2D20cDatabase.instances[-1].lens_queries
    assert queries[0] == ("Hasselblad", "L2D-20c & compatibles")
    assert not np.array_equal(corrected, image)


@pytest.mark.parametrize(
    "metadata_changes",
    [
        {"Model": "L2D-20c Plus"},
        {"FocalLength": 24.0},
        {"Make": "DJI"},
    ],
)
def test_l2d20c_mapping_does_not_apply_to_other_camera_metadata(
    monkeypatch, metadata_changes
):
    fake = _fake_lensfun(monkeypatch)
    fake.Database = _L2D20cDatabase
    metadata = _l2d20c_metadata("24.0 mm f/2.8")
    metadata.update(metadata_changes)

    with pytest.raises(LensMatchError):
        apply_lens_correction(np.full((8, 8, 3), 10000, dtype=np.uint16), metadata)

    queries = _L2D20cDatabase.instances[-1].lens_queries
    assert all(model != "L2D-20c & compatibles" for _, model in queries)


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("FocalLength", 0, "FocalLength"), ("FNumber", None, "FNumber")],
)
def test_invalid_optics_raise_typed_missing_metadata_before_dependency(
    monkeypatch, field, value, message,
):
    monkeypatch.delitem(sys.modules, "lensfunpy", raising=False)
    metadata = _metadata()
    metadata[field] = value
    with pytest.raises(LensMetadataUnavailableError, match=message):
        apply_lens_correction(np.zeros((4, 4, 3), dtype=np.uint16), metadata)


def test_only_vignetting_or_no_geometry_is_not_sufficient(monkeypatch):
    _fake_lensfun(monkeypatch, geometry=False)
    with pytest.raises(LensCorrectionNotAppliedError):
        apply_lens_correction(np.zeros((4, 4, 3), dtype=np.uint16), _metadata())


def test_lens_plan_reuses_modifier_and_invalidates_optics_and_dimensions(monkeypatch):
    import lens_correction as impl
    fake = _fake_lensfun(monkeypatch)
    impl._PLAN_CACHE.clear()
    impl._DATABASE_CACHE.clear()
    image = np.full((9, 7, 3), 1000, dtype=np.uint16)
    metadata = _metadata()
    first, _ = apply_lens_correction(image, metadata)
    modifier = _Modifier.instances[-1]
    second, _ = apply_lens_correction(image, metadata)
    assert _Modifier.instances[-1] is modifier
    assert len(modifier.initialize_calls) == 1
    np.testing.assert_array_equal(first, second)
    apply_lens_correction(image, {**metadata, "FocalLength": 80.0})
    assert _Modifier.instances[-1] is not modifier
    optical = _Modifier.instances[-1]
    apply_lens_correction(image[:8], metadata)
    assert _Modifier.instances[-1] is not optical
    for width in range(10, 17):
        apply_lens_correction(np.zeros((9, width, 3), dtype=np.uint16), metadata)
    assert len(impl._PLAN_CACHE) == impl._MAX_LENS_PLANS
    assert len(impl._DATABASE_CACHE) == 1


def test_lens_plan_invalidates_changed_calibration(monkeypatch):
    import lens_correction as impl
    _fake_lensfun(monkeypatch)
    impl._PLAN_CACHE.clear()
    impl._DATABASE_CACHE.clear()
    signature = [1]
    monkeypatch.setattr(impl, "_calibration_signature", lambda: tuple(signature))
    image = np.zeros((9, 7, 3), dtype=np.uint16)
    apply_lens_correction(image, _metadata())
    modifier = _Modifier.instances[-1]
    signature[0] = 2
    apply_lens_correction(image, _metadata())
    assert _Modifier.instances[-1] is not modifier
