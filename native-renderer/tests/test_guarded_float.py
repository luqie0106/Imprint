"""Guarded GPU math is accepted only against the unchanged Python formulas."""
import os
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import native_renderer as native
import dehaze_physical as physical
from dehaze import DehazeParams

@pytest.mark.parametrize("floor", [-.01, 2.1, float("nan"), float("inf")])
def test_invalid_floor_rejected_before_gpu_loading(floor):
    source=np.zeros((3,7,3),np.float32)
    with pytest.raises(ValueError,match="floor"):
        native.native_physical_dehaze(source,DehazeParams(strength=.8),np.ones((3,7),np.float32),np.ones(3,np.float32),dark_floor=floor)

@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setenv("IMPRINT_NATIVE_GUARDED_FLOAT","1")
    try:
        lib,_=native._get_library()
        if not hasattr(lib,"im_renderer_render_physical_guarded_float"):
            pytest.fail("Requested new GPU guard ABI is missing")
        handle,backend=native._create_renderer(lib)
        try:
            if not lib.im_renderer_supports_physical_guarded_float(handle):
                raise native.NativeRendererError("Guarded GPU capability unavailable")
        finally:
            lib.im_renderer_destroy(handle)
    except native.NativeRendererError as exc:
        if os.environ.get("IMPRINT_REQUIRE_GPU")=="1":pytest.fail(str(exc))
        pytest.skip(str(exc))

@pytest.mark.parametrize("floor", [0., .000001, .003, .03, .12])
def test_fused_gpu_matches_python_and_separate_guard(gpu, floor):
    rng=np.random.default_rng(8491)
    source=rng.uniform(0,.25,(61,97,3)).astype(np.float32)
    source[0,:4]=[[0,0,0],[1e-10]*3,[.98,.95,.89],[.06,.01,.03]]
    saved=source.copy()
    t=rng.uniform(.3,1,source.shape[:2]).astype(np.float32)
    air=np.array([.07,.06,.055],np.float32)
    p=DehazeParams(strength=.85,naturalness=.6,local_contrast=.35)
    actual=native.native_physical_dehaze(source,p,t,air,dark_floor=floor)
    expected=physical._protect_dark_background(source,physical._physical_pixels(source,t,air,p),floor)
    assert np.max(np.abs(actual-expected))<=3e-6
    assert np.max(np.abs(np.rint(actual*65535)-np.rint(expected*65535)))<=1
    assert native.get_last_native_physical_guarded()==(floor>1e-8)
    np.testing.assert_array_equal(source,saved)


def test_zero_strength_guard_preserves_identity(gpu):
    source=np.random.default_rng(3).uniform(0,1,(13,19,3)).astype(np.float32)
    actual=native.native_physical_dehaze(source,DehazeParams(strength=0),np.ones(source.shape[:2],np.float32),np.ones(3,np.float32)*.08,dark_floor=.12)
    np.testing.assert_array_equal(actual,source)
