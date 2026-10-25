"""Scalar P2 and table tests; original host flags remain explicit inputs."""
import ctypes as C
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'native-renderer'
SOURCES = [NATIVE / ('src/' + name + '.cpp') for name in
           ('acr_stage3_curve', 'acr_float_lut')]
OBJECTS = [NATIVE / ('build/CMakeFiles/imprint_renderer.dir/src/' + p.name + '.o')
           for p in SOURCES]
BRIDGE = NATIVE / 'tools/acr_stage3_curve_bridge.cpp'
spec = importlib.util.spec_from_file_location('stage3_curve_oracle', ROOT / 'tests/acr_support/stage3_curve_oracle.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
D = C.POINTER(C.c_double)
F = C.POINTER(C.c_float)


def ptr(v):
    return v.ctypes.data_as(D if v.dtype == np.float64 else F)


class Bridge:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        self.lib.acr_test_stage3_curve_prepare.argtypes = [C.c_double, C.c_int, D]
        self.lib.acr_test_stage3_curve_eval.argtypes = [D, C.c_double, C.c_int, D]
        self.lib.acr_test_stage3_curve_lut.argtypes = [D, C.c_int, F, C.c_size_t]
        for name in ('prepare', 'eval', 'lut'):
            getattr(self.lib, 'acr_test_stage3_curve_' + name).restype = C.c_int

    def prepare(self, pivot, cubic):
        out = np.full(3, -17., dtype=np.float64)
        status = self.lib.acr_test_stage3_curve_prepare(pivot, cubic, ptr(out))
        return status, out


@pytest.fixture(scope='module', params=['source', 'official_object'])
def bridge(request, tmp_path_factory):
    inputs = SOURCES
    if request.param == 'official_object':
        for obj, src in zip(OBJECTS, SOURCES):
            assert obj.is_file(), 'rebuild missing official object'
            assert obj.stat().st_mtime_ns >= max(src.stat().st_mtime_ns, src.with_suffix('.hpp').stat().st_mtime_ns), 'rebuild stale object'
        inputs = OBJECTS
    path = tmp_path_factory.mktemp('stage3_curve_' + request.param) / ('libcurve' + ('.dylib' if sys.platform == 'darwin' else '.so'))
    subprocess.run([shutil.which('clang++') or shutil.which('c++'), '-std=c++17', '-O2',
                    '-fno-fast-math', '-ffp-contract=off', '-fPIC',
                    '-dynamiclib' if sys.platform == 'darwin' else '-shared', '-I', str(NATIVE / 'src'),
                    str(BRIDGE), *(str(p) for p in inputs), '-o', str(path)], check=True, capture_output=True)
    return Bridge(path)


@pytest.mark.parametrize('cubic', [0, 1])
def test_scalar_composition_instruction_bits(bridge, cubic):
    rng = np.random.default_rng(197712)
    for pivot in (0., 4100 * float.fromhex('0x1.0001000100010p-16'), .5):
        status, native = bridge.prepare(pivot, cubic)
        assert status == 0
        prepared = oracle.prepare(pivot, cubic)
        np.testing.assert_array_equal(native.view(np.uint64), np.asarray(prepared).view(np.uint64))
        values = [-.1, -0., 0., pivot, .25, .5, 1., 1.1]
        values += rng.uniform(0, 1, 100).tolist()
        values += [np.nextafter(pivot, 0.), np.nextafter(pivot, 1.)]
        for inverse in (0, 1):
            for sample in values:
                out = np.asarray([-99.], dtype=np.float64)
                assert bridge.lib.acr_test_stage3_curve_eval(ptr(native), sample, inverse, ptr(out)) == 0
                expected = np.asarray([oracle.evaluate(prepared, sample, inverse)], dtype=np.float64)
                np.testing.assert_array_equal(out.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('cubic', [0, 1])
def test_complete_table_matches_scalar_reference(bridge, cubic):
    status, prepared = bridge.prepare(4100 * float.fromhex('0x1.0001000100010p-16'), cubic)
    assert status == 0
    for inverse in (0, 1):
        out = np.full(4100, -17., dtype=np.float32)
        assert bridge.lib.acr_test_stage3_curve_lut(ptr(prepared), inverse, ptr(out), 4100) == 0
        expected = np.asarray([oracle.evaluate(prepared, i * (1./4096.), inverse) for i in range(4097)], dtype=np.float32)
        np.testing.assert_array_equal(out[:4097].view(np.uint32), expected.view(np.uint32))
        assert out.view(np.uint32)[4097] == out.view(np.uint32)[4096]
        assert np.all(out[4098:] == np.float32(-17))


def test_rejections_publish_nothing(bridge):
    for pivot, cubic in ((np.nan, 0), (np.inf, 0), (-1., 0), (1., 0), (.1, 2)):
        status, out = bridge.prepare(pivot, cubic)
        assert status != 0 and np.all(out == -17.)
    status, prepared = bridge.prepare(.1, 1)
    assert status == 0
    for sample, inverse in ((np.nan, 0), (np.inf, 1), (.3, 2)):
        out = np.asarray([-17.], dtype=np.float64)
        assert bridge.lib.acr_test_stage3_curve_eval(ptr(prepared), sample, inverse, ptr(out)) != 0
        assert out[0] == -17.
    before = prepared.copy()
    assert bridge.lib.acr_test_stage3_curve_eval(ptr(prepared), .5, 0, ptr(prepared)) != 0
    np.testing.assert_array_equal(prepared.view(np.uint64), before.view(np.uint64))
    out = np.full(4098, -17., dtype=np.float32)
    assert bridge.lib.acr_test_stage3_curve_lut(ptr(prepared), 0, ptr(out), 4097) != 0
    assert np.all(out == -17.)
