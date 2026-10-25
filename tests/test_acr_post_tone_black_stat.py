"""Exact prepared-log C+658 statistics; independent raw-derived oracle."""
from __future__ import annotations

import ctypes as C
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_highlight_stat.cpp"
HEADER = NATIVE / "src/acr_highlight_stat.hpp"
API = NATIVE / "src/acr_post_tone_stats_api.cpp"
PUBLIC = NATIVE / "include/imprint_renderer.h"
BRIDGE = NATIVE / "tools/acr_post_tone_stat_bridge.cpp"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_highlight_stat.cpp.o"
API_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_post_tone_stats_api.cpp.o"
sys.path.insert(0, str(ROOT / "tests/acr_support"))
sys.path.insert(0, str(ROOT / "src"))
import post_tone_black_stat_oracle as oracle  # noqa: E402
import native_renderer as native  # noqa: E402

FP = C.POINTER(C.c_float)
DP = C.POINTER(C.c_double)
EPS = 2.0 ** -12


def _ptr(address, kind=DP):
    return C.cast(C.c_void_p(address), kind)


def _exact(actual, expected):
    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def _fresh(obj, inputs):
    assert obj.is_file(), f"rebuild renderer: missing official object {obj}"
    assert obj.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in inputs), (
        f"rebuild renderer: stale official object {obj}"
    )


class Bridge:
    def __init__(self, path):
        self.library = C.CDLL(str(path))
        self.fn = self.library.acr_test_compute_post_tone_stats_with_black
        self.fn.argtypes = [FP, C.c_uint64, C.c_double, DP, C.c_size_t]
        self.fn.restype = C.c_int
        self.old = self.library.acr_test_compute_post_tone_stats
        self.old.argtypes = [FP, C.c_uint64, C.c_double, DP]
        self.old.restype = C.c_int
        self.formal = self.library.im_native_acr_post_tone_stats_prepared_f32
        self.formal.argtypes = [FP, C.c_size_t, C.c_double, DP, C.c_size_t]
        self.formal.restype = C.c_int

    def apply(self, source, epsilon=EPS, formal=False):
        output = np.full(8, -1234.5, np.float64)
        fn = self.formal if formal else self.fn
        status = fn(source.ctypes.data_as(FP), source.size, epsilon,
                    output.ctypes.data_as(DP), output.size)
        assert status == 0
        np.testing.assert_array_equal(output[5:], np.full(3, -1234.5))
        return output[:5].copy()


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "a C++17 compiler is required"
    if request.param == "source":
        inputs = [SOURCE, API]
    else:
        _fresh(OBJECT, [SOURCE, HEADER])
        _fresh(API_OBJECT, [API, HEADER, PUBLIC])
        inputs = [OBJECT, API_OBJECT]
    out = tmp_path_factory.mktemp("post_tone_" + request.param) / (
        "stats.dylib" if sys.platform == "darwin" else "stats.so")
    subprocess.run([compiler, "-std=c++17", "-O2", "-fPIC", "-fno-fast-math",
                    "-ffp-contract=off", "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    "-I", str(NATIVE / "include"), "-I", str(NATIVE / "src"),
                    str(BRIDGE), *(str(p) for p in inputs), "-o", str(out)],
                   check=True, capture_output=True, text=True)
    return Bridge(out)


def _unique(count):
    return np.asarray(-10.0 + np.arange(count, dtype=np.float64) / 16384.0, np.float32)


CASES = [(f"M-{n}", _unique(n), EPS) for n in (1, 9999, 10000, 10001, 19999, 20000, 20001)]
CASES += [
    ("low-equality", np.array([-12, -11, -11, -10], np.float32), EPS),
    ("high-equality-and-repetition", np.array([-11, -11, -10, -10, -9.5, -9.5, -9, -9], np.float32), EPS),
    ("gate-p1-n-minus-one", np.array([-11] + [-10] * 60 + [-9.5] * 38 + [-9], np.float32), EPS),
    ("gate-p99-n-minus-one", np.array([-10] * 40 + [-9.75] * 40 + [-9.5] * 19 + [-9], np.float32), EPS),
    ("floor-n-minus-one", np.arange(-10, -8.4375, .015625, dtype=np.float32), EPS),
    ("all-discarded", np.array([-20, -19, -18], np.float32), EPS),
    ("fallback-nonpower", np.array([-120, -119], np.float32), .0002),
    ("zero-positive", np.zeros(37, np.float32), EPS),
    ("zero-negative", np.full(37, -0.0, np.float32), EPS),
    ("filtered-not-original", np.concatenate((np.full(20000, -20, np.float32), _unique(10000))), EPS),
    ("duplicate-high", np.concatenate((_unique(10000), np.full(20000, 5, np.float32))), EPS),
    ("epsilon-min-subnormal", np.array([-1000, -999, -998], np.float32), math.ulp(0.0)),
]
_rng = np.random.default_rng(20261022)
for _n in (2, 3, 99, 100, 101, 1024, 9999, 10000, 20001):
    CASES.append((f"random-{_n}", np.asarray(_rng.normal(-10, 4, _n), np.float32), EPS))
    CASES.append((f"repeated-{_n}", np.asarray(_rng.integers(-24, 9, _n) * .5, np.float32), .0002))


@pytest.mark.parametrize("name,pixels,epsilon", CASES, ids=[c[0] for c in CASES])
def test_all_five_fields_match_independent_reference_and_legacy_four(bridge, name, pixels, epsilon):
    before = pixels.view(np.uint32).copy()
    expected = oracle.compute(pixels, epsilon)
    _exact(bridge.apply(pixels, epsilon), expected)
    _exact(bridge.apply(pixels, epsilon, formal=True), expected)
    legacy = np.full(7, -4321.25, np.float64)
    assert bridge.old(pixels.ctypes.data_as(FP), pixels.size, epsilon,
                      legacy.ctypes.data_as(DP)) == 0
    _exact(legacy[:4], expected[1:])
    np.testing.assert_array_equal(legacy[4:], np.full(3, -4321.25))
    np.testing.assert_array_equal(pixels.view(np.uint32), before)


@pytest.mark.parametrize("count", (1, 9999, 10000, 10001, 19999, 20000, 20001))
def test_literal_black_quantile_population_boundaries(count):
    pixels = _unique(count)
    result = oracle.compute(pixels, EPS)
    index = min(math.floor(count * .0001), count - 1)
    assert result[0] == float(pixels[index])
    assert result[4] == count
    if count in (10000, 20000):
        assert index != math.floor((count - 1) * .0001)
    # The old filtered 1-percent rank is M*.01, independently from gate ranks.
    assert result[1] == float(pixels[min(math.floor(count * .01), count - 1)])


def test_explicit_filter_witnesses():
    _exact(oracle.compute(np.array([-12, -11, -11, -10], np.float32), EPS),
           np.array([-11, -11, -10, -10, 2], np.float64))
    _exact(oracle.compute(np.array([-11, -11, -10, -10, -9.5, -9.5, -9, -9], np.float32), EPS),
           np.array([-11, -11, -9, -9, 6], np.float64))
    assert oracle.compute(np.array([-11] + [-10] * 60 + [-9.5] * 38 + [-9], np.float32), EPS)[4] == 4
    assert oracle.compute(np.array([-10] * 40 + [-9.75] * 40 + [-9.5] * 19 + [-9], np.float32), EPS)[4] == 43
    # N=30000 but M=10000: black index must be 1, not original-count index 3.
    pixels = np.concatenate((np.full(20000, -20, np.float32), _unique(10000)))
    assert oracle.compute(pixels, EPS)[0] == float(_unique(10000)[1])
    # Every original sample is below the low gate: population is one fallback.
    _exact(oracle.compute(np.array([-20], np.float32), EPS),
           np.array([-12, -12, -12, -12, 1], np.float64))


@pytest.mark.parametrize("epsilon", (0, -1, math.nan, math.inf, -math.inf, math.nextafter(EPS, math.inf)))
def test_epsilon_rejection_is_atomic(bridge, epsilon):
    pixels = _unique(4)
    output = np.arange(8, dtype=np.float64) + .125
    before = output.view(np.uint64).copy()
    assert bridge.fn(pixels.ctypes.data_as(FP), 4, epsilon, output.ctypes.data_as(DP), 8) == 102
    assert bridge.formal(pixels.ctypes.data_as(FP), 4, epsilon, output.ctypes.data_as(DP), 8) == 1
    np.testing.assert_array_equal(output.view(np.uint64), before)


@pytest.mark.parametrize("bad", (math.nan, math.inf, -math.inf))
def test_nonfinite_rejection_is_atomic(bridge, bad):
    pixels = np.array([-10, bad, -9], np.float32)
    output = np.arange(8, dtype=np.float64)
    before = output.tobytes()
    assert bridge.fn(pixels.ctypes.data_as(FP), 3, EPS, output.ctypes.data_as(DP), 8) == 103
    assert bridge.formal(pixels.ctypes.data_as(FP), 3, EPS, output.ctypes.data_as(DP), 8) == 1
    assert output.tobytes() == before


def test_pointer_count_short_span_and_alias_rejections_are_atomic(bridge):
    pixels = _unique(8)
    output = np.arange(16, dtype=np.float64)
    source = pixels.ctypes.data_as(FP)
    dest = output.ctypes.data_as(DP)
    before = output.tobytes()
    cases = [
        (None, 8, dest, 8, 100, 1),
        (source, 8, None, 8, 100, 1),
        (source, 0, dest, 8, 101, 1),
        (source, (1 << 26) + 1, dest, 8, 101, 1),
        (_ptr(pixels.ctypes.data + 1, FP), 8, dest, 8, 100, 1),
        (source, 8, _ptr(output.ctypes.data + 1), 8, 100, 1),
        *((source, 8, dest, n, 100, 4) for n in (1, 2, 3, 4)),
        (source, 8, dest, 0, 100, 4),
        (_ptr((1 << (C.sizeof(C.c_void_p) * 8)) - 4, FP), 8, dest, 8, 106, 1),
        (source, 8, _ptr((1 << (C.sizeof(C.c_void_p) * 8)) - 8), 8, 106, 1),
        (source, 8, dest, (1 << (C.sizeof(C.c_size_t) * 8)) - 1, 106, 1),
    ]
    for src, count, out, out_count, core_status, formal_status in cases:
        assert bridge.fn(src, count, EPS, out, out_count) == core_status
        assert bridge.formal(src, count, EPS, out, out_count) == formal_status
        assert output.tobytes() == before
    storage = np.arange(64, dtype=np.float64)
    storage_before = storage.tobytes()
    address = storage.ctypes.data
    # Includes overlap only in the declared output padding, outside five writes.
    for source_offset, output_offset, out_count in ((0, 0, 8), (16, 0, 8), (48, 0, 8), (0, 16, 8)):
        src = _ptr(address + source_offset, FP)
        out = _ptr(address + output_offset)
        assert bridge.fn(src, 8, EPS, out, out_count) == 104
        assert bridge.formal(src, 8, EPS, out, out_count) == 1
        assert storage.tobytes() == storage_before


@pytest.fixture(scope="module")
def formal_library():
    path = NATIVE / "build" / ("libimprint_renderer.0.1.0.dylib" if sys.platform == "darwin"
                                else "libimprint_renderer.0.1.0.so")
    _fresh(OBJECT, [SOURCE, HEADER])
    _fresh(API_OBJECT, [API, HEADER, PUBLIC])
    _fresh(path, [OBJECT, API_OBJECT])
    library = C.CDLL(str(path))
    native._configure_library(library)
    return library, str(path)


def test_formal_library_and_adapter_exact_results(formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    fn = library.im_native_acr_post_tone_stats_prepared_f32
    for name, pixels, epsilon in CASES:
        output = np.full(8, 1234.5, np.float64)
        assert fn(pixels.ctypes.data_as(FP), pixels.size, epsilon, output.ctypes.data_as(DP), 8) == 0
        _exact(output[:5], oracle.compute(pixels, epsilon))
        np.testing.assert_array_equal(output[5:], np.full(3, 1234.5))
        adapted = native.native_acr_post_tone_stats_prepared(pixels, epsilon)
        _exact(adapted, oracle.compute(pixels, epsilon))
    for pixels in (_unique(100).reshape(10, 10), _unique(100).reshape(10, 10).T,
                   _unique(100)[::2]):
        _exact(native.native_acr_post_tone_stats_prepared(pixels, EPS),
               oracle.compute(np.ascontiguousarray(pixels), EPS))


def test_adapter_invalid_inputs_and_missing_symbol(formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    for pixels in ([1, 2], np.array([], np.float32), np.ones(2, np.float64),
                   np.ones((1, 1, 1), np.float32), np.array(1, np.float32),
                   np.array([math.nan], np.float32), np.array([math.inf], np.float32),
                   np.broadcast_to(np.float32(1), ((1 << 26) + 1,))):
        with pytest.raises(ValueError):
            native.native_acr_post_tone_stats_prepared(pixels, EPS)
    for epsilon in (0, -1, math.nan, math.inf, EPS * 2, object()):
        with pytest.raises(ValueError):
            native.native_acr_post_tone_stats_prepared(_unique(4), epsilon)
    monkeypatch.setattr(native, "_get_library", lambda: (object(), "missing"))
    with pytest.raises(native.NativeRendererError):
        native.native_acr_post_tone_stats_prepared(_unique(4), EPS)


def test_adapter_preserves_runtime_error_status(monkeypatch):
    class FailedLibrary:
        @staticmethod
        def im_native_acr_post_tone_stats_prepared_f32(*args):
            return 5
    monkeypatch.setattr(native, "_get_library", lambda: (FailedLibrary(), "runtime-failure"))
    with pytest.raises(native.NativeRendererError, match="status 5"):
        native.native_acr_post_tone_stats_prepared(_unique(4), EPS)
