"""Formal prepared Warp adapter; source normalization also runs in C++."""
import ctypes as C
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import native_renderer as native

F32 = C.POINTER(C.c_float)
U16 = C.POINTER(C.c_uint16)
I32 = C.POINTER(C.c_int32)
U32 = C.POINTER(C.c_uint32)
F64 = C.POINTER(C.c_double)
SCALE = np.asarray([0x37800080], np.uint32).view(np.float32)[0]


@pytest.fixture(scope='module')
def library():
    path = ROOT/'native-renderer/build'/('libimprint_renderer.0.1.0.dylib'
                                         if sys.platform == 'darwin' else 'libimprint_renderer.0.1.0.so')
    assert path.is_file()
    for name in ('acr_warp.cpp', 'acr_warp.hpp', 'acr_warp_api.cpp'):
        assert path.stat().st_mtime_ns >= (ROOT/'native-renderer/src'/name).stat().st_mtime_ns
    result = C.CDLL(str(path))
    native._configure_library(result)
    return result, str(path)


def coeffs():
    return np.asarray([[1.001807569, -.06643970903718387,
                        .03824060737334283, -.05319209398095417, 0, 0]], np.float64)


@pytest.mark.parametrize('planes', [1, 2, 3, 4])
@pytest.mark.parametrize('dtype', [np.float32, np.uint16])
def test_native_normalization_and_noncontiguous_input(library, monkeypatch, planes, dtype):
    monkeypatch.setattr(native, '_get_library', lambda: library)
    image = np.random.default_rng(28).integers(0, 65536, (17, 38, planes), dtype=np.uint16)[:, ::2]
    image = image.astype(dtype)[:, ::-1]
    before = image.copy()
    if dtype == np.float32:
        image *= SCALE
        before = image.copy()
    normalized = image if dtype == np.float32 else np.multiply(image.astype(np.float32), SCALE, dtype=np.float32)
    options = dict(aspect=1.125, center=(.4, .7), image_origin=(-7, 9), tile_shape=(9, 11))
    actual = native.native_acr_warp_prepared(image, coeffs(), **options)
    expected = native.native_acr_warp_prepared(normalized, coeffs(), **options)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(image, before)
    assert actual.dtype == np.float32 and actual.shape == image.shape


@pytest.mark.parametrize('dtype', [np.float32, np.uint16])
@pytest.mark.parametrize('aspect', [1., 1e-300, 1e300])
def test_nop_small_image_and_sample_tail(library, monkeypatch, dtype, aspect):
    monkeypatch.setattr(native, '_get_library', lambda: library)
    image = np.asarray([[[0, 32768, 65535]]], dtype=dtype)
    if dtype == np.float32:
        image = np.asarray([[[-.5, -0., 1.25]]], np.float32)
    output = native.native_acr_warp_prepared(image, np.asarray([[1., 0, 0, 0, 0, 0]]), aspect=aspect)
    expected = image if dtype == np.float32 else np.multiply(image.astype(np.float32), SCALE, dtype=np.float32)
    np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize('change', ['float_coefs', 'tangent', 'center', 'aspect', 'origin', 'tiles', 'dtype', 'nan'])
def test_python_rejects_unsupported_input_before_native(library, monkeypatch, change):
    monkeypatch.setattr(native, '_get_library', lambda: pytest.fail('invalid inputs reached library loading'))
    image = np.ones((10, 10, 3), np.float32)
    parameters = coeffs()
    options = dict(aspect=1.)
    if change == 'float_coefs': parameters = parameters.astype(np.float32)
    elif change == 'tangent': parameters[0, 4] = .01
    elif change == 'center': options['center'] = (False, .5)
    elif change == 'aspect': options['aspect'] = complex(1., 1.)
    elif change == 'origin': options['image_origin'] = (2147483640, 0)
    elif change == 'tiles': options['tile_shape'] = (0, 256)
    elif change == 'dtype': image = image.astype(np.float64)
    elif change == 'nan': parameters[0, 0] = np.nan
    with pytest.raises(ValueError): native.native_acr_warp_prepared(image, parameters, **options)


@pytest.mark.parametrize('bad', ['stride', 'src_count', 'dst_count', 'alias', 'unaligned', 'overflow', 'bad_geometry'])
def test_u16_formal_failures_are_atomic(library, bad):
    lib, _ = library
    source = np.ones((9, 11, 3), np.uint16)
    output = np.full(source.size+4, -753.25, np.float32)
    rectangles = np.asarray([0, 0, 9, 11], np.int32)
    controls = np.asarray([3, 1], np.uint32)
    geometry = np.asarray([.5, .5, 1.], np.float64)
    parameters = coeffs()
    src_count, dst_count, stride = source.size, output.size, 33
    pointer = source.ctypes.data_as(U16)
    if bad == 'stride': stride = 32
    elif bad == 'src_count': src_count -= 1
    elif bad == 'dst_count': dst_count = source.size-1
    elif bad == 'alias': pointer = C.cast(output.ctypes.data, U16)
    elif bad == 'unaligned': pointer = C.cast(source.ctypes.data+1, U16)
    elif bad == 'overflow': stride = C.c_size_t(-1).value
    elif bad == 'bad_geometry': geometry[2] = np.inf
    before = output.copy()
    status = lib.im_native_acr_warp_prepared_u16_f32(
        pointer, src_count, stride, rectangles.ctypes.data_as(I32), controls.ctypes.data_as(U32),
        geometry.ctypes.data_as(F64), parameters.ctypes.data_as(F64), parameters.size,
        256, 256, output.ctypes.data_as(F32), dst_count)
    assert status != 0
    np.testing.assert_array_equal(output.view(np.uint32), before.view(np.uint32))
