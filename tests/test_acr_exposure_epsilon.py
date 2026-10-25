"""Bit-exact tests for the recovered double-to-float exposure epsilon producer."""

from __future__ import annotations

import ctypes as C
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_exposure_epsilon.cpp"
HEADER = NATIVE / "src/acr_exposure_epsilon.hpp"
API_SOURCE = NATIVE / "src/acr_exposure_epsilon_api.cpp"
BRIDGE_SOURCE = NATIVE / "tools/acr_exposure_epsilon_bridge.cpp"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_exposure_epsilon.cpp.o"
API_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_exposure_epsilon_api.cpp.o"
PUBLIC_HEADER = NATIVE / "include/imprint_renderer.h"
sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
sys.path.insert(0, str(ROOT / "src"))
import exposure_epsilon_oracle as oracle  # noqa: E402
import native_renderer as native  # noqa: E402


U32 = C.c_uint32
FLOAT = C.c_float
FLOAT_PTR = C.POINTER(FLOAT)
SIZE = C.c_size_t
OK = 0
INVALID = 1
SHORT = 4


def _bits(value):
    return int(np.asarray([value], dtype=np.float32).view(np.uint32)[0])


def _float_ptr(address):
    return C.cast(C.c_void_p(address), FLOAT_PTR)


def _compile_bridge(kind: str, output: Path) -> None:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the epsilon bridge"
    assert BRIDGE_SOURCE.is_file(), f"missing epsilon C ABI bridge: {BRIDGE_SOURCE}"
    assert SOURCE.is_file() and HEADER.is_file(), "missing epsilon source/header"
    if kind == "source":
        inputs = [SOURCE]
    else:
        assert OBJECT.is_file(), f"rebuild renderer: missing fresh official object {OBJECT.name}"
        assert OBJECT.stat().st_mtime_ns >= max(SOURCE.stat().st_mtime_ns,
                                                HEADER.stat().st_mtime_ns), (
            f"rebuild renderer: stale official object {OBJECT.name}"
        )
        inputs = [OBJECT]
    command = [
        compiler, "-std=c++17", "-O2", "-fPIC", "-fno-fast-math", "-ffp-contract=off",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", str(NATIVE / "src"), "-I", str(NATIVE / "include"), str(BRIDGE_SOURCE),
        *(str(path) for path in inputs), "-o", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.prepare_fn = self.library.acr_test_exposure_pixel_epsilon
        self.prepare_fn.argtypes = [U32, C.c_double, C.c_double, FLOAT_PTR]
        self.prepare_fn.restype = C.c_int

    def raw(self, flag, flare, ev90, output_pointer):
        return self.prepare_fn(flag, flare, ev90, output_pointer)

    def apply(self, flag, flare, ev90):
        output = np.asarray([np.float32(-1234.5)], np.float32)
        status = self.raw(flag, flare, ev90, output.ctypes.data_as(FLOAT_PTR))
        return status, output[0]


@pytest.fixture(scope="module", params=("source", "official_object"),
                ids=("source", "official"))
def bridge(request, tmp_path_factory):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_exposure_epsilon_" + request.param) / (
        "lexposure_epsilon" + suffix
    )
    _compile_bridge(request.param, output)
    return Bridge(output)


@pytest.fixture(scope="module")
def formal_library():
    path = NATIVE / "build" / (
        "libimprint_renderer.0.1.0.dylib" if sys.platform == "darwin"
        else "libimprint_renderer.0.1.0.so"
    )
    assert path.is_file(), f"missing formal native renderer artifact: {path}"
    for artifact in (SOURCE, HEADER, API_SOURCE, PUBLIC_HEADER):
        assert artifact.is_file(), f"missing formal epsilon input: {artifact}"
        assert path.stat().st_mtime_ns >= artifact.stat().st_mtime_ns, (
            f"rebuild formal native renderer: stale relative to {artifact.name}"
        )
    for obj, inputs in ((OBJECT, (SOURCE, HEADER)),
                        (API_OBJECT, (API_SOURCE, PUBLIC_HEADER))):
        assert obj.is_file(), f"missing formal epsilon object: {obj}"
        assert obj.stat().st_mtime_ns >= max(item.stat().st_mtime_ns for item in inputs), (
            f"rebuild formal native renderer: stale object {obj.name}"
        )
        assert path.stat().st_mtime_ns >= obj.stat().st_mtime_ns, (
            f"relink formal native renderer: stale relative to {obj.name}"
        )
    library = C.CDLL(str(path))
    native._configure_library(library)
    function = library.im_native_acr_prepare_exposure_pixel_epsilon
    function.argtypes = [U32, C.c_double, C.c_double, FLOAT_PTR, SIZE]
    function.restype = C.c_int
    return library, str(path)


def _formal_apply(library, flag, flare, ev90, output, count=1):
    function = library.im_native_acr_prepare_exposure_pixel_epsilon
    return function(flag, flare, ev90, output, count)


@pytest.mark.parametrize("flag,flare,ev90,expected_bits", [
    (1, 0.125, 0.0, 0x36000000),
    (0, 0.0, 0.0, 0x36000000),
    (0, -0.125, 0.0, 0x36000000),
    (0, math.nextafter(2.0 ** -12, 0.0), 0.0, 0x36000000),
    (0, 2.0 ** -12, 0.0, 0x36000000),
    (0, math.nextafter(2.0 ** -12, math.inf), 0.0, 0x36000000),
    (0, math.nextafter(2.0 ** -113, 0.0), 0.0, 0x03800000),
    (0, 2.0 ** -113, 0.0, 0x03800000),
    (0, math.nextafter(2.0 ** -113, math.inf), 0.0, 0x03800000),
    (0, 2.0 ** -112, 0.0, 0x04000000),
    (0, 1.0e300, 0.0, 0x36000000),
    (1, 1.0e300, -3.0, 0x37800000),
    (1, 0.0, math.nextafter(-3.0, -math.inf), 0x37800000),
    (1, 0.0, -3.0, 0x37800000),
    (1, 0.0, -3.0 + 2.0 ** -20, 0x377ffff5),
    (1, 0.0, 101.0 - 2.0 ** -20, 0x03800006),
    (1, 0.0, 101.0, 0x03800000),
    (1, 0.0, 101.0 + 2.0 ** -20, 0x03800000),
    (0, 1.0e300, -1.0e300, 0x37800000),
    (0, 1.0e300, 1.0e300, 0x03800000),
], ids=(
    "nonzero-flag-default", "zero-flare-default", "negative-flare-default",
    "flare-cap-prev", "flare-cap-exact", "flare-cap-next",
    "flare-min-prev", "flare-min-exact", "flare-min-next", "flare-above-min",
    "finite-large-flare-capped", "negative-three-ev-upper-bound",
    "max-bound-prev-ev", "max-bound-exact-ev", "max-bound-inside-ev",
    "min-bound-inside-ev", "min-bound-exact-ev", "min-bound-next-ev",
    "finite-extreme-ev-overflow", "finite-extreme-ev-underflow",
))
def test_independent_oracle_literals_match_source_and_official_objects(
    bridge, flag, flare, ev90, expected_bits
):
    assert oracle.REFERENCE_ID == "acr-009e02b0-exposure-pixel-epsilon-f64-to-f32-v1"
    assert oracle.PRODUCER_ENTRY == 0x009E02B0
    assert oracle.SOURCE_BINARY_SHA256 == (
        "09978d95dfe76b18495fea9550b051a89f4598ab0fd4734d69d22d330496f40a"
    )
    expected = oracle.prepare(flag, flare, ev90)
    assert oracle.float32_bits(expected) == expected_bits
    status, actual = bridge.apply(flag, flare, ev90)
    assert status == OK
    assert _bits(actual) == expected_bits


@pytest.mark.parametrize("flag,flare,ev90", [
    (0, math.nan, 0.0),
    (0, math.inf, 0.0),
    (0, -math.inf, 0.0),
    (0, 1.0, math.nan),
    (0, 1.0, math.inf),
    (0, 1.0, -math.inf),
])
def test_nonfinite_inputs_are_rejected_without_writing(bridge, flag, flare, ev90):
    output = np.asarray([0x42F6E979], np.uint32).view(np.float32)
    before = output.view(np.uint8).copy()
    status = bridge.raw(flag, flare, ev90, output.ctypes.data_as(FLOAT_PTR))
    assert status == INVALID
    np.testing.assert_array_equal(output.view(np.uint8), before)
    with pytest.raises(ValueError):
        oracle.prepare(flag, flare, ev90)


def test_null_and_misaligned_outputs_are_rejected_without_writing(bridge):
    assert bridge.raw(0, 0.0, 0.0, C.cast(C.c_void_p(0), FLOAT_PTR)) == INVALID
    storage = (C.c_ubyte * 8)(*range(8))
    address = C.addressof(storage)
    before = bytes(storage)
    status = bridge.raw(0, 0.0, 0.0, _float_ptr(address + 1))
    assert status == INVALID
    assert bytes(storage) == before


def test_formal_api_and_python_adapter_match_independent_bits(formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    for flag, flare, ev90, expected_bits in (
        (1, 0.0, 0.0, 0x36000000),
        (0, 2.0 ** -112, 0.0, 0x04000000),
        (1, 0.0, -3.0 + 2.0 ** -20, 0x377FFFF5),
        (0, 1.0e300, 1.0e300, 0x03800000),
    ):
        expected = oracle.prepare(flag, flare, ev90)
        assert oracle.float32_bits(expected) == expected_bits
        output = np.asarray([np.float32(-1234.5)], np.float32)
        status = _formal_apply(library, flag, flare, ev90,
                               output.ctypes.data_as(FLOAT_PTR))
        assert status == OK
        assert _bits(output[0]) == expected_bits
        adapted = native.native_acr_prepare_exposure_pixel_epsilon(flag, flare, ev90)
        assert _bits(adapted) == expected_bits


def test_formal_short_invalid_and_alignment_statuses_preserve_output(formal_library):
    library, _ = formal_library
    output = np.asarray([0x42F6E979], np.uint32).view(np.float32)
    before = output.view(np.uint8).copy()
    status = _formal_apply(library, 0, 0.0, 0.0,
                           output.ctypes.data_as(FLOAT_PTR), count=0)
    assert status == SHORT
    np.testing.assert_array_equal(output.view(np.uint8), before)

    status = _formal_apply(library, 0, math.nan, 0.0,
                           output.ctypes.data_as(FLOAT_PTR))
    assert status == INVALID
    np.testing.assert_array_equal(output.view(np.uint8), before)

    storage = (C.c_ubyte * 8)(*range(8))
    storage_before = bytes(storage)
    status = _formal_apply(library, 0, 0.0, 0.0,
                           _float_ptr(C.addressof(storage) + 1))
    assert status == INVALID
    assert bytes(storage) == storage_before
