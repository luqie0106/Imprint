"""Optional native profile symbols must retain exact reference behavior."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import camera_profile as cp
from native_renderer import NativeRendererError
from test_camera_profile import _profile


@pytest.mark.parametrize('failure', ['missing', 'disabled', 'status'])
def test_render_native_failure_retains_python_pixels(monkeypatch, failure):
    source = np.random.default_rng(35).integers(0, 65536, (40, 31, 3), dtype=np.uint16)
    saved = source.copy()
    monkeypatch.setenv('IMPRINT_NATIVE_DENSE', '0')
    expected = cp.render_profile(source, (1, 1, 1), _profile(), 1.3)
    if failure == 'disabled':
        pass
    elif failure == 'missing':
        monkeypatch.delenv('IMPRINT_NATIVE_DENSE')
        def missing(*a):
            raise NativeRendererError('missing symbol')
        monkeypatch.setattr(cp.native_dense, '_function', missing)
    else:
        monkeypatch.delenv('IMPRINT_NATIVE_DENSE')
        monkeypatch.setattr(cp.native_dense, '_function', lambda *a: lambda *args: 2)
    actual = cp.render_profile(source, (1, 1, 1), _profile(), 1.3)
    assert np.array_equal(actual, expected)
    assert cp.get_last_profile_backend() == 'python'
    assert np.array_equal(source, saved)


def test_transfer_failure_and_identity_keep_reference(monkeypatch):
    camera = np.random.default_rng(9).integers(4000, 15000, (192, 192, 3), dtype=np.uint16)
    reference = camera.copy()
    processed = np.clip(reference.astype(float) * 1.1 + [80, -50, 70], 0, 65535).astype(np.uint16)
    saved = [a.copy() for a in (camera, reference, processed)]
    monkeypatch.setenv('IMPRINT_NATIVE_DENSE', '0')
    expected = cp.transfer_enhancement(camera, reference, processed)
    monkeypatch.delenv('IMPRINT_NATIVE_DENSE')
    monkeypatch.setattr(cp.native_dense, '_function', lambda *a: lambda *args: 2)
    actual = cp.transfer_enhancement(camera, reference, processed)
    assert np.array_equal(actual, expected)
    assert cp.get_last_transfer_backend() == 'python'
    identity = cp.transfer_enhancement(camera, reference, reference)
    assert np.array_equal(identity, camera)
    assert cp.get_last_transfer_backend() == 'identity'
    for a, b in zip((camera, reference, processed), saved):
        assert np.array_equal(a, b)
