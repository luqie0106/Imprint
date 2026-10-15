from __future__ import annotations

import ctypes

import pytest
import rawpy
import threading

from src import raw_decoder


class _FakeFunction:
    def __init__(self, function):
        self._function = function
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._function(*args)


class _FakeLibrary:
    def __init__(self, *, open_result=1, open_message=b"not a Nikon HE file", probe_is_he=0):
        def probe(_path, output, error, _capacity):
            output._obj.value = probe_is_he
            error.value = b""
            return 0

        def fail_open(_path, _handle, _info, error, _capacity):
            error.value = open_message
            return open_result

        self.imprint_raw_probe_nikon_he = _FakeFunction(probe)
        self.imprint_raw_open = _FakeFunction(fail_open)
        self.imprint_raw_close = _FakeFunction(lambda _handle: None)
        self.imprint_raw_copy_visible = _FakeFunction(lambda *_args: 1)
        self.imprint_raw_postprocess = _FakeFunction(lambda *_args: 1)
        self.imprint_raw_extract_thumb = _FakeFunction(lambda *_args: 1)
        self.imprint_raw_decoder_backend = _FakeFunction(lambda _handle: b"c++/cpu")


def _unsupported(message="unsupported RAW"):
    return rawpy.LibRawFileUnsupportedError(message)


def test_regular_raw_uses_rawpy_without_loading_native(monkeypatch, tmp_path):
    source = tmp_path / "ordinary.arw"
    expected = object()
    monkeypatch.setattr(raw_decoder.rawpy, "imread", lambda path: expected)
    monkeypatch.setattr(
        raw_decoder, "_load_native_library",
        lambda: pytest.fail("ordinary RAW must not load the native fallback"),
    )

    assert raw_decoder.open_raw(source) is expected


def test_nef_metadata_probe_routes_he_to_native_without_calling_rawpy(monkeypatch, tmp_path):
    source = tmp_path / "nikon_he.nef"
    library_path = tmp_path / "libimprint_raw.so"
    library_path.touch()
    fake_library = _FakeLibrary(probe_is_he=1)
    expected = object()
    monkeypatch.setattr(raw_decoder, "_candidate_libraries", lambda: [library_path])
    monkeypatch.setattr(raw_decoder.ctypes, "CDLL", lambda _path: fake_library)
    monkeypatch.setattr(
        raw_decoder.rawpy, "imread",
        lambda _path: pytest.fail("metadata probe identified HE, so rawpy must not unpack it"),
    )
    monkeypatch.setattr(
        raw_decoder, "NativeRaw",
        lambda path, *, _library: expected if path == source and _library is fake_library else None,
    )

    assert raw_decoder.open_raw(source) is expected


def test_non_he_nef_uses_original_rawpy_object_after_probe(monkeypatch, tmp_path):
    source = tmp_path / "ordinary.nef"
    library_path = tmp_path / "libimprint_raw.so"
    library_path.touch()
    fake_library = _FakeLibrary(probe_is_he=0)
    expected = object()
    monkeypatch.setattr(raw_decoder, "_candidate_libraries", lambda: [library_path])
    monkeypatch.setattr(raw_decoder.ctypes, "CDLL", lambda _path: fake_library)
    monkeypatch.setattr(raw_decoder.rawpy, "imread", lambda _path: expected)

    assert raw_decoder.open_raw(source) is expected


def test_only_unsupported_nef_uses_native_fallback(monkeypatch, tmp_path):
    source = tmp_path / "unsupported.arw"
    original = _unsupported()
    monkeypatch.setattr(raw_decoder.rawpy, "imread", lambda _path: (_ for _ in ()).throw(original))
    monkeypatch.setattr(
        raw_decoder, "_load_native_library",
        lambda: pytest.fail("non-NEF RAW must not use the native fallback"),
    )

    with pytest.raises(rawpy.LibRawFileUnsupportedError) as caught:
        raw_decoder.open_raw(source)
    assert caught.value is original


def test_nef_fallback_reports_missing_library_and_preserves_rawpy_cause(monkeypatch, tmp_path):
    source = tmp_path / "unsupported.nef"
    original = _unsupported()
    monkeypatch.setattr(raw_decoder.rawpy, "imread", lambda _path: (_ for _ in ()).throw(original))
    monkeypatch.setattr(raw_decoder, "_candidate_libraries", lambda: [])
    monkeypatch.delenv("IMPRINT_NATIVE_RAW_LIB", raising=False)

    with pytest.raises(raw_decoder.NativeRawLibraryUnavailableError) as caught:
        raw_decoder.open_raw(source)

    assert caught.value.__cause__ is original
    assert "needs the Imprint native RAW library" in str(caught.value)


def test_deferred_rawpy_unsupported_nef_explains_missing_library(monkeypatch, tmp_path):
    source = tmp_path / "deferred.nef"
    original = _unsupported("unsupported during postprocess")

    class DeferredRaw:
        def postprocess(self, **_kwargs):
            raise original

        def close(self):
            pass

    monkeypatch.setattr(raw_decoder.rawpy, "imread", lambda _path: DeferredRaw())
    monkeypatch.setattr(raw_decoder.rawpy, "RawPy", DeferredRaw)
    monkeypatch.setattr(raw_decoder, "_candidate_libraries", lambda: [])
    monkeypatch.delenv("IMPRINT_NATIVE_RAW_LIB", raising=False)

    with raw_decoder.open_raw(source) as raw:
        with pytest.raises(raw_decoder.NativeRawLibraryUnavailableError) as caught:
            raw.postprocess()

    assert caught.value.__cause__ is original
    assert "deferred an unsupported RAW error" in str(caught.value)


def test_native_c_abi_rejection_is_explicit_and_preserves_rawpy_cause(
    monkeypatch, tmp_path
):
    source = tmp_path / "not_he.nef"
    library_path = tmp_path / "libimprint_raw.so"
    library_path.touch()
    original = _unsupported()
    fake_library = _FakeLibrary(open_message=b"NEF is not Nikon HE compression 13 or 14")
    monkeypatch.setattr(raw_decoder.rawpy, "imread", lambda _path: (_ for _ in ()).throw(original))
    monkeypatch.setattr(raw_decoder, "_candidate_libraries", lambda: [library_path])
    monkeypatch.setattr(raw_decoder.ctypes, "CDLL", lambda _path: fake_library)

    with pytest.raises(raw_decoder.NativeRawError) as caught:
        raw_decoder.open_raw(source)

    assert caught.value.__cause__ is original
    assert "compression 13 or 14" in str(caught.value)
    assert fake_library.imprint_raw_open.argtypes is not None


def test_native_library_override_is_authoritative(monkeypatch, tmp_path):
    override = tmp_path / "custom" / "imprint-raw.dylib"
    monkeypatch.setenv("IMPRINT_NATIVE_RAW_LIB", str(override))

    assert raw_decoder._candidate_libraries() == [override]


def test_frozen_library_search_includes_meipass(monkeypatch, tmp_path):
    frozen_root = tmp_path / "bundle"
    monkeypatch.delenv("IMPRINT_NATIVE_RAW_LIB", raising=False)
    monkeypatch.setattr(raw_decoder.sys, "_MEIPASS", str(frozen_root), raising=False)

    candidates = raw_decoder._candidate_libraries()

    assert candidates[0] == frozen_root / "native-raw/build/libimprint_raw.dylib"
    assert frozen_root / "native-raw/build/libimprint_raw.so" in candidates
    assert frozen_root / "native-raw/build/Release/imprint_raw.dll" in candidates
    assert frozen_root / "native-raw/build/Release/libimprint_raw.dll" in candidates
    assert frozen_root / "imprint_raw.dll" in candidates


def test_native_raw_rejects_non_nef_before_loading_library(monkeypatch, tmp_path):
    monkeypatch.setattr(
        raw_decoder, "_load_native_library",
        lambda: pytest.fail("extension check must happen before loading native code"),
    )

    with pytest.raises(raw_decoder.NativeRawError, match="NEF files only"):
        raw_decoder.NativeRaw(tmp_path / "photo.dng")


def test_postprocess_accepts_rawpy_color_space_enums_and_copies_output():
    returned_pixels = (ctypes.c_uint8 * 3)(17, 23, 41)
    seen_colors = []

    def postprocess(_handle, params_pointer, data_pointer, size_pointer,
                    width_pointer, height_pointer, channels_pointer, bits_pointer,
                    error, _capacity):
        params = ctypes.cast(
            params_pointer, ctypes.POINTER(raw_decoder._CPostprocessParams)
        ).contents
        seen_colors.append(params.output_color)
        ctypes.cast(data_pointer, ctypes.POINTER(raw_decoder._UINT8_POINTER))[0] = (
            ctypes.cast(returned_pixels, raw_decoder._UINT8_POINTER)
        )
        ctypes.cast(size_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = 3
        ctypes.cast(width_pointer, ctypes.POINTER(ctypes.c_uint32))[0] = 1
        ctypes.cast(height_pointer, ctypes.POINTER(ctypes.c_uint32))[0] = 1
        ctypes.cast(channels_pointer, ctypes.POINTER(ctypes.c_uint32))[0] = 3
        ctypes.cast(bits_pointer, ctypes.POINTER(ctypes.c_uint32))[0] = 8
        error.value = b""
        return 0

    library = _FakeLibrary()
    library.imprint_raw_postprocess = _FakeFunction(postprocess)
    native = raw_decoder.NativeRaw.__new__(raw_decoder.NativeRaw)
    native._lock = threading.RLock()
    native._handle = ctypes.c_void_p(1234)
    native._library = raw_decoder._bind_library(library)

    srgb = native.postprocess(output_color=rawpy.ColorSpace.sRGB)
    camera_rgb = native.postprocess(output_color=rawpy.ColorSpace.raw)

    assert seen_colors == [rawpy.ColorSpace.sRGB.value, rawpy.ColorSpace.raw.value]
    assert srgb.dtype.name == "uint8"
    assert srgb.tolist() == [[[17, 23, 41]]]
    assert camera_rgb.tolist() == [[[17, 23, 41]]]
    returned_pixels[0] = 0
    assert srgb[0, 0, 0] == 17
