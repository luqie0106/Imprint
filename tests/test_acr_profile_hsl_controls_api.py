"""Public CPU ABI and Python boundary for prepared Profile/HSL24 controls."""
import ctypes as C
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import native_renderer as native

P = C.POINTER(C.c_int32)
Result = native._AcrProfileHslControlsResult


@pytest.fixture(scope="module")
def library():
    library, path = native._get_library()
    function = library.im_native_acr_profile_hsl_controls
    function.argtypes = [P, C.c_size_t, P, C.c_size_t, C.c_double,
                         C.c_uint32, C.c_uint32, C.POINTER(Result)]
    function.restype = C.c_int
    return library, path


def test_control_merge_then_existing_hsl_pixels(library, monkeypatch):
    monkeypatch.setattr(native, "_get_library", lambda: library)
    current = np.zeros(24, np.int32)
    defaults = np.zeros(24, np.int32)
    current[0], defaults[0] = 50, 50
    current[8], defaults[8] = -50, -50
    saved = current.copy(), defaults.copy()
    values, mask, blended = native.native_acr_profile_hsl_controls(current, defaults, 1.)
    assert mask == 0xffffff
    assert values[0] == 83 and values[8] == -83
    assert blended[0] == pytest.approx(83.33333333333331)
    assert blended[8] == pytest.approx(-83.33333333333333)
    np.testing.assert_array_equal(current, saved[0])
    np.testing.assert_array_equal(defaults, saved[1])
    rgb = np.array([[[.5, .2, .1], [.1, .3, .6], [.7, .4, .2]]], np.float32)
    actual = native.native_acr_hsl_sdr(rgb, np.r_[values, 0.].astype(np.float64))
    expected_controls = np.zeros(25, np.float64)
    expected_controls[0], expected_controls[8] = 83., -83.
    expected = native.native_acr_hsl_sdr(rgb, expected_controls)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert not np.array_equal(actual, rgb)


@pytest.mark.parametrize("mono,gray", [(True, 0), (False, 1), (True, 255)])
def test_gated_lanes_retain_full_current(library, monkeypatch, mono, gray):
    monkeypatch.setattr(native, "_get_library", lambda: library)
    current = np.arange(24, dtype=np.int32) + 1000
    values, mask, blended = native.native_acr_profile_hsl_controls(
        current, [75]*24, 2., monochrome_profile=mono, convert_to_gray=gray)
    np.testing.assert_array_equal(values, current)
    np.testing.assert_array_equal(blended, current.astype(np.float64))
    assert mask == 0


def test_layout_failure_preservation_and_alias_checks(library):
    assert Result.values.offset == 0 and Result.blended.offset == 96 and Result.wrote_mask.offset == 288
    function = library[0].im_native_acr_profile_hsl_controls
    current = (C.c_int32*24)(*range(24))
    defaults = (C.c_int32*24)(*([50]*24))
    output = Result()
    C.memset(C.addressof(output), 0xa5, C.sizeof(output))
    before = bytes(output)
    for scalar in (float("nan"), float("inf"), -float("inf")):
        assert function(current, 24, defaults, 24, scalar, 0, 0, C.byref(output)) == 1
        assert bytes(output) == before
    for count, dcount, mono, gray in ((23,24,0,0),(24,25,0,0),(24,24,2,0),(24,24,0,256)):
        assert function(current,count,defaults,dcount,1.,mono,gray,C.byref(output)) == 1
        assert bytes(output) == before
    assert function(current,24,defaults,24,1.,0,0,None) == 1
    for pointer in (None, C.cast(C.addressof(current)+1,P),
                    C.cast(C.addressof(output),P), C.cast(C.c_size_t(-4).value,P)):
        assert function(pointer,24,defaults,24,1.,0,0,C.byref(output)) == 1
        assert bytes(output) == before
        assert function(current,24,pointer,24,1.,0,0,C.byref(output)) == 1
        assert bytes(output) == before
    storage=(C.c_ubyte*(C.sizeof(Result)+8))(*([0xa5]*(C.sizeof(Result)+8)))
    misaligned=C.cast(C.addressof(storage)+1,C.POINTER(Result))
    saved=bytes(storage)
    assert function(current,24,defaults,24,1.,0,0,misaligned) == 1
    assert bytes(storage)==saved


@pytest.mark.parametrize("field,value", [
    ("current", [0]*23), ("current", [False]+[0]*23),
    ("current", [0.]+[0]*23), ("current", [1<<31]+[0]*23),
    ("profile_defaults", [-(1<<31)-1]+[0]*23),
    ("profile_scalar", True), ("profile_scalar", float("nan")),
    ("profile_scalar", 10**1000), ("monochrome_profile", 1),
    ("convert_to_gray", True), ("convert_to_gray", -1), ("convert_to_gray", 256),
])
def test_invalid_wrapper_input_before_loading(monkeypatch, field, value):
    def fail_load():
        pytest.fail("invalid inputs must be checked before loading")
    monkeypatch.setattr(native, "_get_library", fail_load)
    args=dict(current=[0]*24,profile_defaults=[0]*24,profile_scalar=1.)
    args[field]=value
    with pytest.raises(ValueError):native.native_acr_profile_hsl_controls(**args)
