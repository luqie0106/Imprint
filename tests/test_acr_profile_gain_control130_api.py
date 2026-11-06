"""The formal-library boundary for the explicit profile ID130 suboperation."""
import ctypes
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import native_renderer as native


@pytest.fixture(scope="module")
def library():
    build = ROOT / "native-renderer" / "build"
    names = (["libimprint_renderer.dylib"] if sys.platform == "darwin" else
             ["Release/imprint_renderer.dll", "imprint_renderer.dll"]
             if sys.platform == "win32" else ["libimprint_renderer.so"])
    path = next((build / name for name in names if (build / name).is_file()), None)
    assert path is not None, "build imprint_renderer before checking the public ABI"
    lib = ctypes.CDLL(str(path))
    fn = lib.im_native_acr_profile_gain_control130
    fn.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32,
                   ctypes.c_double, ctypes.POINTER(native._AcrProfileGainControl130Result)]
    fn.restype = ctypes.c_int
    return lib, str(path)


@pytest.mark.parametrize("current,default,version,scalar,expected", [
    (100, 100, 0x05070001, 1., (167, True, 166.66666666666666)),
    (100, 100, 0x05070001, 2., (180, True, 180.)),
    (0, 100, 0xffffffff, 1., (100, True, 100.)),
    (-999999, 100, 0x05070000, 1., (-999999, False, -999999.)),
    (30, -999999, 0x05070001, 1., (30, False, 30.)),
    (30, 201, 0x05070001, 1., (30, False, 30.)),
    (-(1 << 31), 0, 0x05070001, 1., (0, True, float(-(1 << 31)))),
    ((1 << 31) - 1, 0, 0x05070001, 1., (200, True, float((1 << 31) - 1))),
])
def test_wrapper_uses_formal_library(library, monkeypatch, current, default,
                                     version, scalar, expected):
    monkeypatch.setattr(native, "_get_library", lambda: library)
    assert native.native_acr_profile_gain_control130(
        current, default, version, scalar) == expected


@pytest.mark.parametrize("scalar", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_output_is_unchanged(library, scalar):
    result = native._AcrProfileGainControl130Result(-123, 17, 42.25)
    before = bytes(result)
    fn = library[0].im_native_acr_profile_gain_control130
    assert fn(100, 100, 0x05070001, scalar, ctypes.byref(result)) == 1
    assert bytes(result) == before


def test_output_layout_and_pointer_validation(library):
    result_type = native._AcrProfileGainControl130Result
    assert ctypes.sizeof(result_type) == 16
    assert (result_type.value.offset, result_type.wrote.offset,
            result_type.blended.offset) == (0, 4, 8)
    fn = library[0].im_native_acr_profile_gain_control130
    assert fn(100, 100, 0x05070001, 1., None) == 1
    storage = (ctypes.c_ubyte * 32)(*([0xa5] * 32))
    pointer = ctypes.cast(ctypes.addressof(storage) + 1, ctypes.POINTER(result_type))
    before = bytes(storage)
    assert fn(100, 100, 0x05070001, 1., pointer) == 1
    assert bytes(storage) == before


@pytest.mark.parametrize("values", [
    (True, 100, 0x05070001, 1.),
    (1 << 31, 100, 0x05070001, 1.),
    (0, -(1 << 31) - 1, 0x05070001, 1.),
    (0, 100, -1, 1.),
    (0, 100, 1 << 32, 1.),
    (0, 100, 0x05070001, True),
    (0, 100, 0x05070001, float("nan")),
    (0, 100, 0x05070001, 10 ** 1000),
])
def test_wrapper_rejects_values_before_loading(monkeypatch, values):
    def should_not_load():
        pytest.fail("invalid inputs must be checked before loading a library")
    monkeypatch.setattr(native, "_get_library", should_not_load)
    with pytest.raises(ValueError):
        native.native_acr_profile_gain_control130(*values)
