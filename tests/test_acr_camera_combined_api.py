"""HWC Python adaptation of the prepared modern Camera CPU entry."""
from pathlib import Path
import ctypes as C
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import native_renderer as native


@pytest.fixture(scope='module')
def library():
    path = ROOT / 'native-renderer/build' / ('libimprint_renderer.0.1.0.dylib'
        if sys.platform == 'darwin' else 'libimprint_renderer.0.1.0.so')
    assert path.is_file()
    for name in ('acr_camera_combined.cpp', 'acr_camera_combined.hpp', 'acr_camera_combined_api.cpp'):
        assert path.stat().st_mtime_ns >= (ROOT / 'native-renderer/src' / name).stat().st_mtime_ns
    lib = C.CDLL(str(path))
    native._configure_library(lib)
    return lib, str(path)


def test_hwc_permutation_and_noncontiguous(library, monkeypatch):
    monkeypatch.setattr(native, '_get_library', lambda: library)
    backing = np.arange(63, dtype=np.float32).reshape(3, 7, 3) / np.float32(32)
    source = backing[:, ::-1]
    before = backing.copy()
    controls = [1, 1, 1, 0, *np.eye(3).ravel()]
    actual = native.native_acr_camera_combined_prepared(source, controls, (2, 0, 1), 1)
    np.testing.assert_array_equal(actual, source[..., [2, 0, 1]])
    np.testing.assert_array_equal(backing, before)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous


@pytest.mark.parametrize('field,value', [('order3', [0, 0, 2]), ('order3', [0., 1., 2.]),
    ('flags', -1), ('flags', 2**32), ('flags', True), ('controls', [1]*12)])
def test_invalid_metadata_before_library(monkeypatch, field, value):
    monkeypatch.setattr(native, '_get_library', lambda: pytest.fail('invalid metadata loaded library'))
    args = dict(controls=[1, 1, 1, 0, *np.eye(3).ravel()], order3=(0, 1, 2), flags=0)
    args[field] = value
    with pytest.raises(ValueError):
        native.native_acr_camera_combined_prepared(np.zeros((1, 1, 3), np.float32), **args)


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_native_rejects_nonfinite_pixels(library, monkeypatch, bad):
    monkeypatch.setattr(native, '_get_library', lambda: library)
    source = np.full((1, 1, 3), bad, np.float32)
    with pytest.raises(ValueError):
        native.native_acr_camera_combined_prepared(source, [1, 1, 1, 0, *np.eye(3).ravel()])
