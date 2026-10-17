from __future__ import annotations

import ctypes
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import native_renderer as nr


_TEMPERATURES = {
    1: 5500, 4: 5500, 9: 5500, 18: 5500, 20: 5500,
    2: 4150, 14: 4150,
    3: 2850, 17: 2850,
    10: 6500, 19: 6500, 21: 6500,
    11: 7500, 22: 7500,
    12: 6400,
    13: 5050,
    15: 3525,
    16: 2925,
    23: 5000,
    24: 3200,
}


def _double_bits(value: float) -> bytes:
    return struct.pack("=d", value)


@pytest.fixture(scope="module")
def native_api():
    try:
        library, _ = nr._get_library()
    except nr.NativeRendererError as exc:
        pytest.skip(f"native renderer library is unavailable: {exc}")
    function = getattr(library, "im_native_acr_illuminant_temperatures", None)
    assert function is not None, "native renderer library is missing the illuminant ABI"
    function.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


def _temperature(code: int) -> float:
    return float(_TEMPERATURES.get(code, 0))


def test_native_lookup_matches_recovered_double_bits_for_all_uint16_codes(native_api):
    for first in range(0, 65536, 2):
        codes = (ctypes.c_uint32 * 2)(first, first + 1)
        output = (ctypes.c_double * 2)(-123.0, -456.0)
        status = native_api(codes, 2, output, 2)
        if first == 254:
            assert status == 1
            assert tuple(output) == (-123.0, -456.0)
            continue
        assert status == 0
        assert _double_bits(output[0]) == _double_bits(_temperature(first))
        assert _double_bits(output[1]) == _double_bits(_temperature(first + 1))


def test_native_lookup_rejects_bad_counts_nulls_and_misalignment_without_writes(native_api):
    codes = (ctypes.c_uint32 * 2)(1, 2)
    output = (ctypes.c_double * 2)(-123.0, -456.0)
    for bad_count in (1, 3):
        assert native_api(codes, bad_count, output, 2) == 1
        assert tuple(output) == (-123.0, -456.0)
    for bad_output_count in (1, 3):
        assert native_api(codes, 2, output, bad_output_count) == 1
        assert tuple(output) == (-123.0, -456.0)
    assert native_api(None, 2, output, 2) == 1
    assert tuple(output) == (-123.0, -456.0)
    assert native_api(codes, 2, None, 2) == 1
    assert tuple(output) == (-123.0, -456.0)

    storage = ctypes.create_string_buffer(32)
    base = ctypes.addressof(storage)
    misaligned_address = base + 1
    if misaligned_address % ctypes.alignment(ctypes.c_uint32) == 0:
        misaligned_address += 1
    assert misaligned_address % ctypes.alignment(ctypes.c_uint32) != 0
    misaligned_codes = ctypes.cast(misaligned_address, ctypes.POINTER(ctypes.c_uint32))
    assert native_api(misaligned_codes, 2, output, 2) == 1
    assert tuple(output) == (-123.0, -456.0)

    misaligned_output_address = base + 1
    if misaligned_output_address % ctypes.alignment(ctypes.c_double) == 0:
        misaligned_output_address += 1
    assert misaligned_output_address % ctypes.alignment(ctypes.c_double) != 0
    misaligned_output = ctypes.cast(misaligned_output_address, ctypes.POINTER(ctypes.c_double))
    assert native_api(codes, 2, misaligned_output, 2) == 1
    assert tuple(output) == (-123.0, -456.0)


def test_native_lookup_rejects_overflowing_ranges_and_alias_without_writes(native_api):
    codes = (ctypes.c_uint32 * 2)(1, 2)
    output = (ctypes.c_double * 2)(-123.0, -456.0)
    pointer_bits = ctypes.sizeof(ctypes.c_void_p) * 8
    address_max = (1 << pointer_bits) - 1
    invalid_codes = ctypes.cast(address_max - 3, ctypes.POINTER(ctypes.c_uint32))
    assert native_api(invalid_codes, 2, output, 2) == 1
    assert tuple(output) == (-123.0, -456.0)
    invalid_output = ctypes.cast(address_max - 7, ctypes.POINTER(ctypes.c_double))
    assert native_api(codes, 2, invalid_output, 2) == 1
    assert tuple(output) == (-123.0, -456.0)

    aliased = (ctypes.c_double * 4)(-123.0, -456.0, 1.0, 2.0)
    before = ctypes.string_at(ctypes.addressof(aliased), ctypes.sizeof(aliased))
    aliased_codes = ctypes.cast(aliased, ctypes.POINTER(ctypes.c_uint32))
    assert native_api(aliased_codes, 2, aliased, 2) == 1
    after = ctypes.string_at(ctypes.addressof(aliased), ctypes.sizeof(aliased))
    assert after == before


@pytest.mark.parametrize("invalid_code", [65536, 0xFFFFFFFF])
@pytest.mark.parametrize("lane", [0, 1])
def test_native_lookup_rejects_values_outside_uint16_without_writes(
        native_api, invalid_code, lane):
    values = [1, 2]
    values[lane] = invalid_code
    codes = (ctypes.c_uint32 * 2)(*values)
    output = (ctypes.c_double * 2)(-123.0, -456.0)
    assert native_api(codes, 2, output, 2) == 1
    assert tuple(output) == (-123.0, -456.0)


@pytest.mark.parametrize("lane", [0, 1])
def test_native_lookup_rejects_descriptor_code_255_without_writes(native_api, lane):
    values = [1, 2]
    values[lane] = 255
    codes = (ctypes.c_uint32 * 2)(*values)
    output = (ctypes.c_double * 2)(-123.0, -456.0)
    assert native_api(codes, 2, output, 2) == 1
    assert tuple(output) == (-123.0, -456.0)


@pytest.mark.parametrize("codes", [
    (), (1,), (1, 2, 3),
    (True, 1), (np.bool_(False), 1),
    (1.0, 2), (np.float64(1.0), 2),
    (-1, 2), (65536, 2), (255, 1),
])
def test_python_adapter_rejects_invalid_values_before_loading_native(codes, monkeypatch):
    monkeypatch.setattr(nr, "_get_library", lambda: pytest.fail("native library should not load"))
    with pytest.raises(ValueError):
        nr.native_acr_illuminant_temperatures(codes)


def test_python_adapter_uses_native_result_and_accepts_numpy_integers(monkeypatch):
    calls = []

    def lookup(codes, count, output, output_count):
        calls.append((tuple(codes), count, output_count))
        output[0] = 321.25
        output[1] = 654.5
        return 0

    library = SimpleNamespace(im_native_acr_illuminant_temperatures=lookup)
    monkeypatch.setattr(nr, "_get_library", lambda: (library, "test-native"))
    assert nr.native_acr_illuminant_temperatures((np.uint16(7), np.int64(8))) == (321.25, 654.5)
    assert calls == [((7, 8), 2, 2)]


def test_python_adapter_reports_missing_native_abi(monkeypatch):
    monkeypatch.setattr(nr, "_get_library", lambda: (SimpleNamespace(), "test-native"))
    with pytest.raises(nr.NativeRendererError, match="illuminant lookup"):
        nr.native_acr_illuminant_temperatures((1, 2))
