"""Aligned color-space images reuse optical maps without changing pixels."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import lens_correction as lens
from test_lens_correction import _fake_lensfun, _metadata, _Modifier


def test_batch_matches_individual_and_generates_each_strip_once(monkeypatch):
    _fake_lensfun(monkeypatch)
    lens._PLAN_CACHE.clear()
    images = [np.random.default_rng(seed).integers(0, 65536, (300, 7, 3), dtype=np.uint16)
              for seed in range(3)]
    originals = [image.copy() for image in images]
    expected = [lens.apply_lens_correction(image, _metadata(), backend='cpu') for image in images]
    modifier = _Modifier.instances[-1]
    modifier.map_calls.clear()
    actual = lens.apply_lens_correction_many(images, _metadata(), backend='cpu')
    assert len(modifier.map_calls) == 3  # Three output strips, not three maps per strip.
    for (output, result), (baseline, baseline_result), original, image in zip(actual, expected, originals, images):
        np.testing.assert_array_equal(output, baseline)
        np.testing.assert_array_equal(image, original)
        assert result == baseline_result
        assert not np.shares_memory(output, image)


def test_batch_rejects_misaligned_images_before_processing():
    with pytest.raises(ValueError, match='aligned'):
        lens.apply_lens_correction_many([np.zeros((2, 3, 3), np.uint16), np.zeros((3, 3, 3), np.uint16)], {})


def test_gpu_partial_failure_restarts_entire_batch_and_closes(monkeypatch):
    import native_renderer
    class Geometry:
        def apply_geometry_distortion(self, x, y, width, height):
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            return np.stack((xx + .25, yy + y + .5), axis=-1)
    class Broken:
        instances = []
        def __init__(self, source, mode):
            self.closed = False
            self.instances.append(self)
        def remap(self, maps, y):
            if y: raise lens.NativeRendererError('injected failure after first strip')
        def finish(self, destination):
            raise AssertionError('failed GPU must not be read')
        def close(self): self.closed = True
    monkeypatch.setattr(native_renderer, 'NativeLensRemapper', Broken, raising=False)
    monkeypatch.setattr(lens.sys, 'platform', 'darwin')
    monkeypatch.delenv('IMPRINT_NATIVE_LENS_REMAP', raising=False)
    sources = [np.random.default_rng(i).integers(0, 65536, (1000, 1000, 3), dtype=np.uint16) for i in range(2)]
    expected = [source.copy() for source in sources]
    actual = [source.copy() for source in sources]
    options = dict(width=1000, height=1000, geometry_enabled=True, tca_enabled=False, distortion_enabled=True)
    lens._apply_geometry_many(expected, Geometry(), _allow_gpu=False, **options)
    lens._apply_geometry_many(actual, Geometry(), **options)
    for a, b in zip(actual, expected): np.testing.assert_array_equal(a, b)
    assert len(Broken.instances) == 2 and all(instance.closed for instance in Broken.instances)
