"""Formal-library boundary for the local profile scalar quantizer."""
import ctypes
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import native_renderer as native


@pytest.fixture(scope='module')
def library():
    build = ROOT / 'native-renderer/build'
    names = (['libimprint_renderer.dylib'] if sys.platform == 'darwin' else
             ['Release/imprint_renderer.dll', 'imprint_renderer.dll']
             if sys.platform == 'win32' else ['libimprint_renderer.so'])
    path = next((build / n for n in names if (build / n).is_file()), None)
    assert path is not None
    lib = ctypes.CDLL(str(path))
    fn = lib.im_native_acr_profile_gain_scalar_quantize
    fn.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    fn.restype = ctypes.c_int
    return lib, str(path)


@pytest.mark.parametrize('value,expected', [
    (-0., 0.), (-1., 0.), (.005, .01), (1.234, 1.23), (3., 2.),
    (float('inf'), 2.), (-float('inf'), 0.), (float('nan'), 2.),
])
def test_formal_wrapper(library, monkeypatch, value, expected):
    monkeypatch.setattr(native, '_get_library', lambda: library)
    result = native.native_acr_quantize_profile_gain_scalar(value)
    assert struct.pack('=d', result) == struct.pack('=d', expected)


@pytest.mark.parametrize('bits', [0x7ff0000000000001, 0xfff0000000000001])
def test_signaling_nan_unchanged(library, monkeypatch, bits):
    value = struct.unpack('=d', struct.pack('=Q', bits))[0]
    output = ctypes.c_double(123.25)
    before = bytes(output)
    assert library[0].im_native_acr_profile_gain_scalar_quantize(value, ctypes.byref(output)) == 1
    assert bytes(output) == before
    monkeypatch.setattr(native, '_get_library', lambda: library)
    with pytest.raises(native.NativeRendererError):
        native.native_acr_quantize_profile_gain_scalar(value)


def test_invalid_pointer(library):
    fn = library[0].im_native_acr_profile_gain_scalar_quantize
    assert fn(1., None) == 1
    storage = (ctypes.c_ubyte * 24)(*([0xa5] * 24))
    before = bytes(storage)
    ptr = ctypes.cast(ctypes.addressof(storage) + 1, ctypes.POINTER(ctypes.c_double))
    assert fn(1., ptr) == 1
    assert bytes(storage) == before


@pytest.mark.parametrize('value', [True, '1', None, 10 ** 1000])
def test_invalid_wrapper_input(monkeypatch, value):
    monkeypatch.setattr(native, '_get_library', lambda: pytest.fail('unexpected library load'))
    with pytest.raises(ValueError):
        native.native_acr_quantize_profile_gain_scalar(value)
