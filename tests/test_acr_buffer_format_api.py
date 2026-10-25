"""Python orchestration of native CopyArea scalar storage conversions."""
from pathlib import Path
import ctypes as C
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import native_renderer as native


@pytest.fixture(scope='module')
def library():
    path = ROOT/'native-renderer/build'/('libimprint_renderer.0.1.0.dylib'
                                         if sys.platform == 'darwin' else 'libimprint_renderer.0.1.0.so')
    assert path.is_file()
    for name in ('acr_buffer_format.cpp', 'acr_buffer_format.hpp', 'acr_buffer_format_api.cpp'):
        assert path.stat().st_mtime_ns >= (ROOT/'native-renderer/src'/name).stat().st_mtime_ns
    result = C.CDLL(str(path))
    native._configure_library(result)
    return result, str(path)


@pytest.mark.parametrize('shape', [(7,13), (5,17,3)])
def test_native_roundtrip_noncontiguous_and_odd_rows(library, monkeypatch, shape):
    monkeypatch.setattr(native, '_get_library', lambda: library)
    backing = np.random.default_rng(327).integers(0,65536,shape,dtype=np.uint16)
    image = backing[:,::-1]
    before = backing.copy()
    normalized = native.native_acr_buffer_to_float(image)
    actual = native.native_acr_buffer_to_u16(normalized)
    scale = np.asarray([0x37800080],np.uint32).view(np.float32)[0]
    reference = image.astype(np.float32)*scale
    np.testing.assert_array_equal(normalized.view(np.uint32), reference.view(np.uint32))
    np.testing.assert_array_equal(actual, image)
    np.testing.assert_array_equal(backing, before)
    assert actual.dtype==np.uint16 and normalized.dtype==np.float32


def test_ieee_clamp_and_signed_zero(library, monkeypatch):
    monkeypatch.setattr(native, '_get_library', lambda: library)
    image=np.asarray([[np.nan,np.inf,-np.inf,-0.,0.,-.25,.5,1.,1.25]],np.float32)
    before=image.view(np.uint32).copy()
    actual=native.native_acr_buffer_to_u16(image)
    np.testing.assert_array_equal(actual,np.asarray([[0,65535,0,0,0,0,32768,65535,65535]],np.uint16))
    np.testing.assert_array_equal(image.view(np.uint32),before)


@pytest.mark.parametrize('image,to_float', [
    (np.zeros((2,3),np.float64),False),
    (np.zeros((2,3),np.int16),True),
    (np.zeros((0,3),np.float32),False),
    (np.zeros((3,),np.uint16),True),
    (np.zeros((1,1,1,1),np.float32),False),
])
def test_invalid_python_contract_does_not_load_library(monkeypatch,image,to_float):
    monkeypatch.setattr(native, '_get_library',lambda:pytest.fail('invalid input loaded native library'))
    function=native.native_acr_buffer_to_float if to_float else native.native_acr_buffer_to_u16
    with pytest.raises(ValueError):function(image)
