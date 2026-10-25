"""Exact prepared basic-slot request loop; no RAW/default preparation claim."""
from __future__ import annotations

import ctypes as C
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
MODULES = ("acr_post_tone_basic", "acr_post_tone_basic_api")
sys.path.insert(0, str(ROOT / "tests/acr_support"))
import post_tone_basic_oracle as oracle  # noqa: E402

IP = C.POINTER(C.c_int32)
N = 132
MARK = np.int32(-0x1234567)
PVS = (0, 1, 0x0506fffe, 0x0506ffff, 0x05070000, 0x05070001,
       0x7fffffff, 0xfffffffe, 0xffffffff)


def pointer(array=None, offset=0):
    address = 0 if array is None else array.ctypes.data + offset
    return C.cast(C.c_void_p(address), IP)


def exact(actual, expected):
    assert actual.dtype == expected.dtype == np.int32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def _fresh(path, dependencies):
    assert path.is_file(), f"missing official artifact: {path}"
    assert path.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in dependencies), (
        f"rebuild stale official artifact: {path}")


class ABI:
    def __init__(self, path, formal=False):
        self.library = C.CDLL(str(path))
        self.fn = getattr(self.library, "im_native_acr_normalize_post_tone_basic_i32" if formal
                          else "acr_test_normalize_post_tone_basic_i32")
        self.fn.argtypes = [C.c_uint32, IP, C.c_size_t, IP, C.c_size_t, IP, C.c_size_t]
        self.fn.restype = C.c_int

    def apply(self, values, defaults, pv):
        before, default_before = values.copy(), defaults.copy()
        out = np.full(N + 7, MARK, np.int32)
        assert self.fn(pv, pointer(values), N, pointer(defaults), N, pointer(out), len(out)) == 0
        exact(values, before)
        exact(defaults, default_before)
        exact(out[N:], np.full(7, MARK, np.int32))
        return out[:N].copy()


@pytest.fixture(scope="module", params=("source", "official_object", "formal_api"))
def abi(request, tmp_path_factory):
    sources = [NATIVE / f"src/{module}.cpp" for module in MODULES]
    header = NATIVE / "src/acr_post_tone_basic.hpp"
    public = NATIVE / "include/imprint_renderer.h"
    if request.param == "formal_api":
        path = NATIVE / "build" / ("libimprint_renderer.0.1.0.dylib" if sys.platform == "darwin"
                                    else "libimprint_renderer.0.1.0.so")
        _fresh(path, [*sources, header, public])
        return ABI(path, formal=True)
    inputs = sources
    if request.param == "official_object":
        inputs = [NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/{module}.cpp.o"
                  for module in MODULES]
        for obj, source in zip(inputs, sources):
            dependencies = [source, header]
            if source.stem.endswith("_api"):
                dependencies.append(public)
            _fresh(obj, dependencies)
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "C++17 compiler required"
    path = tmp_path_factory.mktemp("post_tone_basic_" + request.param) / (
        "basic.dylib" if sys.platform == "darwin" else "basic.so")
    bridge = NATIVE / "tools/acr_post_tone_basic_bridge.cpp"
    subprocess.run([compiler, "-std=c++17", "-O2", "-fPIC",
                    "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    "-I", str(NATIVE / "include"), "-I", str(NATIVE / "src"),
                    str(bridge), *(str(p) for p in inputs), "-o", str(path)],
                   check=True, capture_output=True, text=True)
    return ABI(path)


def prepared():
    values = (np.arange(N, dtype=np.int32) * 7919 - 800003).astype(np.int32)
    defaults = (np.arange(N, dtype=np.int32) * -65537 + 300001).astype(np.int32)
    return values, defaults


def test_oracle_literal_and_manual_witnesses():
    assert len(oracle.SWITCH_BYTES) == 122
    assert hashlib.sha256(oracle.SWITCH_BYTES).hexdigest() == oracle.SWITCH_SHA256
    values, defaults = prepared()
    values[14] = np.iinfo(np.int32).min
    out = oracle.normalize(values, defaults, 0xffffffff)
    assert out[14] == 0 and out[100] == 100
    for index in (0, 7, 11, 12, 5, 8, 13, 0x83):
        assert out[index] == 0
    for index in (9, 10, 15, 112, 113, 114, 115, 116, 117, 118, 129, 130):
        assert out[index] == values[index]
    exact(out[25:35], defaults[25:35])
    exact(oracle.normalize(values, defaults, 0x05070000)[25:35], np.zeros(10, np.int32))


@pytest.mark.parametrize("pv", PVS)
def test_all_indices_and_process_version_boundary(abi, pv):
    values, defaults = prepared()
    exact(abi.apply(values, defaults, pv), oracle.normalize(values, defaults, pv))


@pytest.mark.parametrize("sentinel", (np.iinfo(np.int32).min, -1, 0, 1, np.iinfo(np.int32).max))
def test_each_index_i32_extremes_and_clarity_clamp(abi, sentinel):
    for pv in (0x0506ffff, 0x05070000, 0xffffffff):
        for index in range(N):
            values = np.full(N, -153, np.int32)
            defaults = np.full(N, 719, np.int32)
            values[index] = sentinel
            defaults[index] = np.int32(~int(sentinel))
            exact(abi.apply(values, defaults, pv), oracle.normalize(values, defaults, pv))


def test_random_raw_int32_bits_and_repeated_default_switches(abi):
    rng = np.random.default_rng(0x9dcb00)
    for _ in range(100):
        values = rng.integers(0, 1 << 32, N, dtype=np.uint32).view(np.int32)
        defaults = rng.integers(0, 1 << 32, N, dtype=np.uint32).view(np.int32)
        for pv in (0xffffffff, 0x05070000, 0x0506ffff, int(rng.integers(0, 1 << 32))):
            exact(abi.apply(values, defaults, pv), oracle.normalize(values, defaults, pv))


def test_inputs_may_overlap_read_only(abi):
    storage = np.arange(N + 1, dtype=np.int32) * -919
    for values, defaults in ((storage[:N], storage[1:]), (storage[1:], storage[:N]),
                             (storage[:N], storage[:N])):
        exact(abi.apply(values, defaults, 0xffffffff), oracle.normalize(values, defaults, 0xffffffff))


@pytest.mark.parametrize("which", ("values", "defaults", "both"))
def test_exact_inplace_alias_allowed_only_exact_output_capacity(abi, which):
    values, defaults = prepared()
    if which == "both":
        defaults = values
    output = defaults if which == "defaults" else values
    expected = oracle.normalize(values, defaults, 0xffffffff)
    untouched = (values if output is defaults else defaults).copy()
    assert abi.fn(0xffffffff, pointer(values), N, pointer(defaults), N, pointer(output), N) == 0
    exact(output, expected)
    if which != "both":
        exact(values if output is defaults else defaults, untouched)


def test_invalid_counts_pointers_alignment_and_span_are_atomic(abi):
    values, defaults = prepared()
    out = np.full(N + 9, MARK, np.int32)
    v, d, o = pointer(values), pointer(defaults), pointer(out)
    size_max = C.c_size_t(-1).value
    uintptr_max = (1 << (C.sizeof(C.c_void_p) * 8)) - 1
    overflow = C.cast(C.c_void_p(uintptr_max - 3), IP)
    cases = [(None, N, d, N, o, N), (v, N, None, N, o, N), (v, N, d, N, None, N),
             (pointer(values, 1), N, d, N, o, N), (v, N, pointer(defaults, 1), N, o, N),
             (v, N, d, N, pointer(out, 1), N),
             (overflow, N, d, N, o, N), (v, N, overflow, N, o, N),
             (v, N, d, N, overflow, N), (v, N, d, N, o, size_max),
             (v, N, d, N, o, size_max // 2), (v, N, d, N, o, 0)]
    for count in (0, 1, N - 1, N + 1, size_max):
        cases.extend(((v, count, d, N, o, N), (v, N, d, count, o, N)))
    cases.append((v, N, d, N, o, N - 1))
    before, vb, db = out.copy(), values.copy(), defaults.copy()
    for args in cases:
        assert abi.fn(0, *args) != 0
        exact(out, before)
        exact(values, vb)
        exact(defaults, db)


def test_partial_alias_and_padding_only_alias_are_rejected_atomically(abi):
    _, defaults = prepared()
    # All ranges actually fit their backing allocation, including output padding.
    for input_start, out_start, out_count in ((1, 0, N), (0, 1, N),
                                               (0, 0, N + 1), (N, 0, N + 1)):
        for alias_defaults in (False, True):
            storage = np.arange(3 * N + 8, dtype=np.int32) * 17
            before = storage.copy()
            aliased = pointer(storage, input_start * 4)
            v, d = (pointer(defaults), aliased) if alias_defaults else (aliased, pointer(defaults))
            assert abi.fn(0, v, N, d, N, pointer(storage, out_start * 4), out_count) != 0
            exact(storage, before)


def test_exact_alias_with_one_input_does_not_exempt_partial_other_input(abi):
    for reverse in (False, True):
        storage = np.arange(N + 1, dtype=np.int32) * -317
        before = storage.copy()
        base, shifted = pointer(storage), pointer(storage, 4)
        v, d = (shifted, base) if reverse else (base, shifted)
        assert abi.fn(0xffffffff, v, N, d, N, base, N) != 0
        exact(storage, before)


@pytest.fixture(scope="module")
def formal_adapter():
    sys.path.insert(0, str(ROOT / "src"))
    import native_renderer
    path = NATIVE / "build" / ("libimprint_renderer.0.1.0.dylib" if sys.platform == "darwin"
                                else "libimprint_renderer.0.1.0.so")
    _fresh(path, [NATIVE / f"src/{m}.cpp" for m in MODULES])
    api = ABI(path, formal=True)
    return native_renderer, api.library, str(path)


def test_typed_adapter_exact_noncontiguous_and_unaligned_inputs(formal_adapter, monkeypatch):
    native, library, path = formal_adapter
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    values, defaults = prepared()
    strided = np.repeat(values, 2)[::2]
    unaligned = np.ndarray((N,), dtype=np.int32, buffer=bytearray(N * 4 + 1), offset=1)
    unaligned[:] = defaults
    for v, d in ((values, defaults), (strided, unaligned), (values[::-1], defaults[::-1])):
        vb, db = v.copy(), d.copy()
        for pv in (*PVS, np.uint32(0xffffffff), np.int64(0)):
            result = native.native_acr_normalize_post_tone_basic(v, d, pv)
            assert result.shape == (N,) and result.dtype == np.int32
            exact(result, oracle.normalize(v, d, pv))
        exact(v, vb)
        exact(d, db)


def test_adapter_rejects_unprepared_types_shape_and_pv(formal_adapter, monkeypatch):
    native, library, path = formal_adapter
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    values, defaults = prepared()
    bad = [values.tolist(), values.astype(np.int64), values.astype(np.float32),
           values.astype(np.float64), values.astype(np.uint32), values.astype(">i4"),
           values.reshape(12, 11), values[:N - 1], np.zeros(N + 1, np.int32),
           np.array(1, np.int32), None]
    for array in bad:
        for v, d in ((array, defaults), (values, array)):
            with pytest.raises(ValueError):
                native.native_acr_normalize_post_tone_basic(v, d, 0)
    for pv in (-1, 1 << 32, 0.0, 1.5, True, False, np.bool_(True), "0", object()):
        with pytest.raises(ValueError):
            native.native_acr_normalize_post_tone_basic(values, defaults, pv)


def test_adapter_missing_symbol_and_status_translation(formal_adapter, monkeypatch):
    native, _, _ = formal_adapter
    values, defaults = prepared()
    monkeypatch.setattr(native, "_get_library", lambda: (object(), "missing"))
    with pytest.raises(native.NativeRendererError):
        native.native_acr_normalize_post_tone_basic(values, defaults, 0)
    for status in (1, 4, 2, 5):
        class Failed:
            @staticmethod
            def im_native_acr_normalize_post_tone_basic_i32(*args):
                return status
        monkeypatch.setattr(native, "_get_library", lambda: (Failed(), "failure"))
        with pytest.raises(ValueError if status in (1, 4) else native.NativeRendererError):
            native.native_acr_normalize_post_tone_basic(values, defaults, 0)
