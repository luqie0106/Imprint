import ctypes as C
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_bayer_gain.cpp"
HEADER = SOURCE.with_suffix(".hpp")
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_bayer_gain.cpp.o"
BRIDGE = NATIVE / "tools/acr_bayer_gain_bridge.cpp"
spec = importlib.util.spec_from_file_location("gain_oracle", ROOT / "tests/acr_support/bayer_gain_oracle.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
P = C.POINTER(C.c_float)
D = C.POINTER(C.c_double)
S = C.c_size_t


def ptr(a):
    return a.ctypes.data_as(P)


class Bridge:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        self.prep = self.lib.acr_test_gain_prepare
        self.prep.argtypes = [D, S, C.c_double, C.c_int, P, C.POINTER(C.c_int)]
        self.prep.restype = C.c_int
        self.rgb = self.lib.acr_test_gain_rgb
        self.rgb.argtypes = [P, S, P, S, P, S, S, P, C.c_int]
        self.rgb.restype = C.c_int
        self.cfa = self.lib.acr_test_gain_cfa
        self.cfa.argtypes = [P, S, S, C.c_uint, C.c_uint, P, C.c_int]
        self.cfa.restype = C.c_int

    def prepare(self, gains, pivot, inverse=False):
        gains = np.asarray(gains, dtype=np.float64)
        result = np.full(6, -312.5, dtype=np.float32)
        clip = C.c_int(777)
        status = self.prep(gains.ctypes.data_as(D), gains.size, pivot, inverse, ptr(result), C.byref(clip))
        return status, result, clip.value


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    inputs = [SOURCE]
    if request.param == "official_object":
        assert OBJECT.exists(), "rebuild renderer: missing official gain object"
        assert OBJECT.stat().st_mtime_ns >= max(SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns), "rebuild renderer: stale official gain object"
        inputs = [OBJECT]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("bayer_gain_" + request.param) / ("libgain" + suffix)
    compiler = shutil.which("clang++") or shutil.which("c++")
    command = [compiler, "-std=c++17", "-O2", "-fPIC", "-fno-fast-math", "-ffp-contract=off",
               "-dynamiclib" if sys.platform == "darwin" else "-shared", "-I", str(NATIVE / "src"),
               str(BRIDGE), *(str(p) for p in inputs), "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return Bridge(output)


def equal_bits(a, b):
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("inverse", (False, True))
def test_prepare_matches_instruction_model(bridge, inverse):
    rng = np.random.default_rng(2026101901)
    examples = [[1, 1, 1], [2, 1, .5], [1 + 2**-26, .3, 7], [.9, 1.00000002, 1.2]]
    examples += rng.uniform(.01, 127, (80, 3)).tolist()
    for i, gains in enumerate(examples):
        pivot = [0., .01, .15, .8, float(np.nextafter(np.float32(1), np.float32(0)))][i % 5]
        status, actual, clip = bridge.prepare(gains, pivot, inverse)
        assert status == 0
        expected, expected_clip = oracle.prepare(gains, pivot, inverse)
        equal_bits(actual, expected)
        assert bool(clip) == expected_clip


@pytest.mark.parametrize("gains", ([1, 1, 1], [1, 2, .5], [2, 1, .5], [2, .5, 1], [2, 3, 4], [.8, .7, .6], [1 + 2**-26, 1.4, .8]))
def test_rgb_two_selected_planes_and_clipping(bridge, gains):
    status, values, clip = bridge.prepare(gains, .12)
    assert status == 0
    rng = np.random.default_rng(320)
    for width in (1, 7, 8, 19):
        rows = rng.uniform(-1, 2, (3, width + 5)).astype(np.float32)
        before = rows.copy()
        expected = oracle.rgb(rows, width, values, clip)
        assert bridge.rgb(ptr(rows[0]), rows.shape[1], ptr(rows[1]), rows.shape[1], ptr(rows[2]), rows.shape[1], width, ptr(values), clip) == 0
        equal_bits(rows, expected)
        equal_bits(rows[:, width:], before[:, width:])
        if gains == [2, 3, 4]:
            equal_bits(rows[2], before[2])


@pytest.mark.parametrize("inverse", (False, True))
def test_cfa_four_origin_parities(bridge, inverse):
    status, values, clip = bridge.prepare([2, 1, .4], .13, inverse)
    assert status == 0
    rng = np.random.default_rng(400)
    for row_parity in (0, 1):
        for col_parity in (0, 1):
            for width in (2, 14, 16, 34):
                row = rng.uniform(-2, 3, width + 4).astype(np.float32)
                expected = oracle.cfa(row, width, values, clip, row_parity, col_parity)
                assert bridge.cfa(ptr(row), row.size, width, row_parity, col_parity, ptr(values), clip) == 0
                equal_bits(row, expected)


def test_explicit_fma_and_signed_zero(bridge):
    # Each parameter is a literal binary32 value. FMA produces 2**-46;
    # separate multiplication/addition would lose the residual.
    values = np.asarray([1 + 2**-23, 1, 1, -(1 + 2**-22), 0, 0], dtype=np.float32)
    rows = np.asarray([[1 + 2**-23, -0.], [0., -0.], [0., -0.]], dtype=np.float32)
    expected = oracle.rgb(rows, 2, values, False)
    assert bridge.rgb(ptr(rows[0]), 2, ptr(rows[1]), 2, ptr(rows[2]), 2, 2, ptr(values), 0) == 0
    equal_bits(rows, expected)
    assert rows[0, 0] == np.float32(2**-46)
    # Strictly-negative clamp preserves -0, while a max(value,+0) would not.
    state = np.asarray([1, 1, 1, -0., -0., -0.], dtype=np.float32)
    row = np.asarray([-0., 0.], dtype=np.float32)
    assert bridge.cfa(ptr(row), 2, 2, 0, 0, ptr(state), 1) == 0
    assert row.view(np.uint32).tolist() == [0x80000000, 0]


def test_inverse_precedes_binary32_conversion(bridge):
    gain = 40.36190848642672
    status, values, clip = bridge.prepare([gain, 1, 1], .13, True)
    assert status == 0
    assert values.view(np.uint32)[0] == 0x3CCAF6B1
    wrong = np.float32(np.float32(1) / np.float32(gain))
    assert wrong.view(np.uint32) == 0x3CCAF6B2


def test_pivot_bias_keeps_original_instruction_sequence(bridge):
    gain, pivot = 8.828765454849897, 0.06170278114606977
    status, values, clip = bridge.prepare([gain, 1, 1], pivot)
    assert status == 0 and clip == 1
    assert values.view(np.uint32)[3] == 0xBEF75334
    simplified = np.float32(np.float32(pivot) * np.float32(1 - np.float32(gain)))
    assert simplified.view(np.uint32) == 0xBEF75333


def test_prepare_failure_is_atomic(bridge):
    for gains, pivot in [([0, 1, 1], 0), ([128, 1, 1], .2), ([np.nan, 1, 1], .2), ([1, 2], .2), ([1, 2, 3], 1.), ([1, 2, 3], np.nan), ([1, 2, 3], -.1)]:
        status, values, clip = bridge.prepare(gains, pivot)
        assert status != 0
        assert np.all(values == np.float32(-312.5)) and clip == 777
    status, values, clip = bridge.prepare([1e-300, 1, 1], .13, True)
    assert status != 0 and clip == 777
    assert np.all(values == np.float32(-312.5))


def test_row_preflight_and_overlap_are_atomic(bridge):
    status, values, clip = bridge.prepare([2, 1, .4], .13)
    assert status == 0
    rows = np.full((3, 20), .2, dtype=np.float32)
    for width, count, rp, cp in [(3, 20, 0, 0), (20, 19, 0, 0), (4, 20, 2, 0), (4, 20, 0, 2), (2**63, 2**63, 0, 0)]:
        before = rows.copy()
        assert bridge.cfa(ptr(rows[0]), count, width, rp, cp, ptr(values), clip) != 0
        equal_bits(rows, before)
    before = rows.copy()
    assert bridge.rgb(ptr(rows[0]), 20, ptr(rows[0]), 20, ptr(rows[2]), 20, 8, ptr(values), clip) != 0
    equal_bits(rows, before)
    rows[2, 7] = np.nan
    before = rows.copy()
    assert bridge.rgb(ptr(rows[0]), 20, ptr(rows[1]), 20, ptr(rows[2]), 20, 8, ptr(values), clip) != 0
    equal_bits(rows, before)
