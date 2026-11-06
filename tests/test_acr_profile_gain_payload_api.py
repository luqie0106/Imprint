"""Public C++ parser boundary used by selected-profile metadata transport."""
from pathlib import Path
import ctypes
import struct
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
    assert path is not None, "build imprint_renderer before testing its public parser"
    assert path.stat().st_mtime_ns >= (ROOT / "native-renderer/src/acr_profile_gain_image_api.cpp").stat().st_mtime_ns
    return ctypes.CDLL(str(path)), str(path)


def payload(tag=52544, order=0, data_type=3):
    endian = ">" if order else "<"
    header = struct.pack(endian + "II4dI5f", 1, 1, 1., 1., 0., 0., 2,
                         .1, .2, .3, .4, .5)
    if tag == 52544:
        header += struct.pack(endian + "I3f", data_type, 1., .25, 2.)
    body = (bytes([0, 255]) if data_type == 0 else
            struct.pack(endian + "2H", 0, 65535) if data_type == 1 else
            struct.pack(endian + "2e", .5, 1.5) if data_type == 2 else
            struct.pack(endian + "2f", .5, 1.5))
    return header + body


@pytest.mark.parametrize("tag,data_type", [(52525, 3)] + [(52544, t) for t in range(4)])
@pytest.mark.parametrize("order", [0, 1])
def test_public_parser_accepts_formats_and_preserves_bytes(library, monkeypatch, tag, data_type, order):
    monkeypatch.setattr(native, "_get_library", lambda: library)
    original = payload(tag, order, data_type)
    data = bytearray(original)
    assert native.native_acr_validate_profile_gain_payload(data, tag=tag, byte_order=order) is None
    assert bytes(data) == original


@pytest.mark.parametrize("mutate", [
    lambda p: b"", lambda p: p[:-1], lambda p: p + b"\0",
    lambda p: bytes(4) + p[4:],
    lambda p: p[:64] + struct.pack("<I", 4) + p[68:],
    lambda p: p[:68] + struct.pack("<f", 0.) + p[72:],
    lambda p: p[:80] + struct.pack("<f", float("nan")) + p[84:],
])
def test_public_parser_rejects_malformed_payload(library, monkeypatch, mutate):
    monkeypatch.setattr(native, "_get_library", lambda: library)
    with pytest.raises(ValueError, match="parser rejected"):
        native.native_acr_validate_profile_gain_payload(mutate(payload()))


@pytest.mark.parametrize("kwargs", [{"tag": 9}, {"tag": 52544.0}, {"tag": True},
                                    {"byte_order": 2}, {"byte_order": False}])
def test_invalid_transport_controls_do_not_load_library(monkeypatch, kwargs):
    monkeypatch.setattr(native, "_get_library", lambda: pytest.fail("invalid controls loaded library"))
    with pytest.raises(ValueError):
        native.native_acr_validate_profile_gain_payload(payload(), **kwargs)


def test_missing_native_symbol_is_explicit(monkeypatch):
    monkeypatch.setattr(native, "_get_library", lambda: (object(), "old library"))
    with pytest.raises(native.NativeRendererError, match="does not provide"):
        native.native_acr_validate_profile_gain_payload(payload())


def test_c_boundary_rejects_null_and_invalid_controls(library):
    function = library[0].im_native_acr_profile_gain_payload_validate
    function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                         ctypes.c_uint32, ctypes.c_uint32]
    function.restype = ctypes.c_int
    assert function(None, 0, 52544, 0) == 1
    encoded = payload()
    data = (ctypes.c_uint8 * len(encoded)).from_buffer_copy(encoded)
    assert function(data, len(data), 0, 0) == 1
    assert function(data, len(data), 52544, 2) == 1
