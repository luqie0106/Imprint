"""Formal GainMap API, normalized-u16 adapter and Python integration."""
import ctypes as C
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import native_renderer as native

spec = importlib.util.spec_from_file_location('gainmap_api_reference', ROOT/'tests/acr_support/gainmap_oracle.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
SCALE = np.asarray([0x37800080], np.uint32).view(np.float32)[0]


@pytest.fixture(scope='module')
def library():
    path = ROOT/'native-renderer/build'/('libimprint_renderer.0.1.0.dylib' if sys.platform == 'darwin' else 'libimprint_renderer.so')
    assert path.is_file()
    for name in ('acr_gainmap.cpp', 'acr_gainmap.hpp', 'acr_gainmap_api.cpp'):
        assert path.stat().st_mtime_ns >= (ROOT/'native-renderer/src'/name).stat().st_mtime_ns, 'rebuild stale library'
    result = C.CDLL(str(path))
    native._configure_library(result)
    assert hasattr(result, 'im_native_acr_gainmap_prepared_f32')
    assert hasattr(result, 'im_native_acr_gainmap_prepared_u16')
    return result, path


def metadata():
    gains = np.linspace(.9, 1.6, 3*4*3, dtype=np.float32).reshape(3, 4, 3)
    return SimpleNamespace(top=-3, left=5, bottom=4, right=12, plane=0,
        planes=3, row_pitch=1, col_pitch=1, map_points_v=3, map_points_h=4,
        map_planes=3, map_spacing_v=.5, map_spacing_h=1/3,
        map_origin_v=0., map_origin_h=0., gains=gains)


def expected(image, m, floor, stage=3, tile=(3, 2)):
    h, w, p = image.shape
    normalized = image if image.dtype == np.float32 else np.multiply(image.astype(np.float32), SCALE, dtype=np.float32)
    rects = (-3, 5, -3+h, 5+w, m.top, m.left, m.bottom, m.right)
    controls = (p, m.plane, m.planes, m.row_pitch, m.col_pitch,
                m.map_points_v, m.map_points_h, m.map_planes, stage)
    geometry = (m.map_spacing_v, m.map_spacing_h, m.map_origin_v, m.map_origin_h)
    out = oracle.reference(normalized.reshape(h, w*p), w*p, rects, controls,
        geometry, m.gains.reshape(-1), black_floor=floor, tile_w=tile[1], tile_h=tile[0])
    if image.dtype == np.uint16:
        assert np.isfinite(out).all() and np.all(out >= 0) and np.all(out <= 1)
        return np.rint(np.multiply(out, np.float32(65535), dtype=np.float32)).astype(np.uint16)
    return out


@pytest.mark.parametrize('dtype', [np.float32, np.uint16])
@pytest.mark.parametrize('stage,floor', [(3, 0), (3, 4100), (1, 65535)])
def test_python_adapter_matches_independent_pixels(library, monkeypatch, dtype, stage, floor):
    monkeypatch.setattr(native, '_get_library', lambda: library)
    # Noncontiguous, odd sample width exercises native row padding/format conversion.
    backing = (np.arange(7*14*3, dtype=np.uint16).reshape(7, 14, 3)*100+6000).astype(dtype)
    if dtype == np.float32:
        backing *= np.float32(1/65535)
        backing[0, 0, 0] = -.25
    image = backing[:, ::2]
    assert not image.flags.c_contiguous
    before = image.copy()
    m = metadata()
    actual = native.native_acr_gainmap_prepared(image, m, black_floor=floor,
        list_stage=stage, image_origin=(-3, 5), tile_shape=(3, 2))
    want = expected(image, m, floor, stage)
    np.testing.assert_array_equal(actual.view(np.uint32) if dtype == np.float32 else actual,
                                  want.view(np.uint32) if dtype == np.float32 else want)
    np.testing.assert_array_equal(image, before)
    assert actual.dtype == dtype and actual.shape == image.shape


def test_u16_adapter_rejects_negative_conversion_domain_atomically(library):
    lib, _ = library
    source = np.zeros((2, 2, 3), np.uint16)
    destination = np.full(18, 0xA55A, np.uint16)
    rects = np.asarray([0, 0, 2, 2]*2, np.int32)
    controls = np.asarray([3, 0, 3, 1, 1, 1, 1, 1, 3], np.uint32)
    geometry = np.asarray([1., 1., 0., 0.], np.float64)
    gain = np.asarray([2.], np.float32)
    status = lib.im_native_acr_gainmap_prepared_u16(
        source.ctypes.data_as(C.POINTER(C.c_uint16)), source.size, 6,
        rects.ctypes.data_as(C.POINTER(C.c_int32)), controls.ctypes.data_as(C.POINTER(C.c_uint32)),
        geometry.ctypes.data_as(C.POINTER(C.c_double)), gain.ctypes.data_as(C.POINTER(C.c_float)),
        gain.size, 4100, 8, 8, destination.ctypes.data_as(C.POINTER(C.c_uint16)), destination.size)
    assert status == 1
    assert np.all(destination == 0xA55A)


def test_u16_row_adapter_preserves_absolute_plane_fallback(library, monkeypatch):
    monkeypatch.setattr(native, '_get_library', lambda: library)
    image = np.full((1, 1, 17), 12000, np.uint16)
    m = metadata()
    m.planes = 17
    actual = native.native_acr_gainmap_prepared(image, m, black_floor=4100,
        image_origin=(-3, 5), tile_shape=(1, 1))
    np.testing.assert_array_equal(actual, expected(image, m, 4100, tile=(1, 1)))


@pytest.mark.parametrize('bad', ['short_source', 'short_output', 'source_count_overflow',
    'output_count_overflow', 'stride_overflow', 'unaligned_source',
    'unaligned_output', 'source_alias', 'rect_alias', 'controls_alias', 'map_alias'])
def test_u16_adapter_invalid_spans_leave_output_unchanged(library, bad):
    lib, _ = library
    source = np.full(24, 12000, np.uint16)
    destination = np.full(24, 0xA55A, np.uint16)
    rects = np.asarray([0, 0, 2, 2]*2, np.int32)
    controls = np.asarray([3, 0, 3, 1, 1, 1, 1, 1, 3], np.uint32)
    geometry = np.asarray([1., 1., 0., 0.], np.float64)
    gain = np.ones(8, np.float32)
    sp, dp = source.ctypes.data, destination.ctypes.data
    sc, dc, stride = 12, 12, 6
    if bad == 'short_source': sc = 11
    elif bad == 'short_output': dc = 11
    elif bad == 'source_count_overflow': sc = C.c_size_t(-1).value
    elif bad == 'output_count_overflow': dc = C.c_size_t(-1).value
    elif bad == 'stride_overflow': stride = C.c_size_t(-1).value
    elif bad == 'unaligned_source': sp += 1
    elif bad == 'unaligned_output': dp += 1
    elif bad == 'source_alias': dp = sp+2
    elif bad == 'rect_alias': dp = rects.ctypes.data
    elif bad == 'controls_alias': dp = controls.ctypes.data
    elif bad == 'map_alias': dp = gain.ctypes.data
    arrays = [source, destination, rects, controls, geometry, gain]
    before = [a.tobytes() for a in arrays]
    status = lib.im_native_acr_gainmap_prepared_u16(
        C.cast(sp, C.POINTER(C.c_uint16)), sc, stride,
        rects.ctypes.data_as(C.POINTER(C.c_int32)), controls.ctypes.data_as(C.POINTER(C.c_uint32)),
        geometry.ctypes.data_as(C.POINTER(C.c_double)), gain.ctypes.data_as(C.POINTER(C.c_float)),
        gain.size, 4100, 8, 8, C.cast(dp, C.POINTER(C.c_uint16)), dc)
    assert status != 0
    assert [a.tobytes() for a in arrays] == before


@pytest.mark.parametrize('change', [dict(black_floor=True), dict(black_floor=65535),
    dict(tile_shape=(0, 4)), dict(image_origin=(False, 0))])
def test_invalid_metadata_rejects_before_library(monkeypatch, change):
    def unexpected():
        raise AssertionError('invalid metadata should reject before loading native')
    monkeypatch.setattr(native, '_get_library', unexpected)
    args = dict(black_floor=4100)
    args.update(change)
    with pytest.raises(ValueError):
        native.native_acr_gainmap_prepared(np.zeros((2, 2, 3), np.uint16), metadata(), **args)
