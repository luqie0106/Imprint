"""Native/source-object integer LUT parity and memory contract."""
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
SOURCE = NATIVE / 'src/acr_u16_lut.cpp'
OBJECT = NATIVE / 'build/CMakeFiles/imprint_renderer.dir/src/acr_u16_lut.cpp.o'
BRIDGE = NATIVE / 'tools/acr_u16_lut_bridge.cpp'
spec = importlib.util.spec_from_file_location('u16_lut_oracle', ROOT / 'tests/acr_support/u16_lut_oracle.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
U, F, S = C.POINTER(C.c_uint16), C.POINTER(C.c_float), C.c_size_t


def ptr(a):
    return a.ctypes.data_as(U if a.dtype == np.uint16 else F)


class Bridge:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        self.lib.acr_test_u16_lut_build.argtypes = [F, S, U, S]
        self.lib.acr_test_u16_lut_build.restype = C.c_int
        self.lib.acr_test_u16_lut_apply.argtypes = [U, S, U, S, U, S, S]
        self.lib.acr_test_u16_lut_apply.restype = C.c_int


@pytest.fixture(scope='module', params=['source', 'official_object'])
def bridge(request, tmp_path_factory):
    source = SOURCE
    if request.param == 'official_object':
        assert OBJECT.is_file(), 'rebuild missing u16 LUT object'
        assert OBJECT.stat().st_mtime_ns >= max(SOURCE.stat().st_mtime_ns, SOURCE.with_suffix('.hpp').stat().st_mtime_ns), 'rebuild stale u16 LUT object'
        source = OBJECT
    path = tmp_path_factory.mktemp('u16_lut_' + request.param) / ('libu16' + ('.dylib' if sys.platform == 'darwin' else '.so'))
    subprocess.run([shutil.which('clang++') or shutil.which('c++'), '-std=c++17', '-O2',
                    '-fno-fast-math', '-ffp-contract=off', '-fPIC',
                    '-dynamiclib' if sys.platform == 'darwin' else '-shared', '-I', str(NATIVE/'src'),
                    str(BRIDGE), str(source), '-o', str(path)], check=True, capture_output=True)
    return Bridge(path)


def tables():
    x = np.arange(4097, dtype=np.float64)/4096.
    rng = np.random.default_rng(197713)
    for a in [x.astype(np.float32), (.2+.37*x*x).astype(np.float32),
              rng.uniform(-.5,1.5,4097).astype(np.float32), np.full(4097,-.1,dtype=np.float32)]:
        yield np.concatenate((a,a[-1:]))


def test_full_domain_tables_match_instruction_order(bridge):
    for table in tables():
        out = np.full(65540, 12345, dtype=np.uint16)
        assert bridge.lib.acr_test_u16_lut_build(ptr(table),4098,ptr(out),65540) == 0
        np.testing.assert_array_equal(out[:65536], oracle.build(table))
        assert np.all(out[65536:] == 12345)


def test_lookup_full_domain_in_place_and_padding(bridge):
    table = ((np.arange(65536,dtype=np.uint32)*7123+7)&65535).astype(np.uint16)
    src = np.arange(65536,dtype=np.uint16)
    out = np.full(65540,12345,dtype=np.uint16)
    assert bridge.lib.acr_test_u16_lut_apply(ptr(table),65536,ptr(src),65536,ptr(out),65540,65536)==0
    np.testing.assert_array_equal(out[:65536],table)
    assert np.all(out[65536:]==12345)
    a = np.concatenate((src,np.asarray([5,6,7],dtype=np.uint16)))
    assert bridge.lib.acr_test_u16_lut_apply(ptr(table),65536,ptr(a),a.size,ptr(a),a.size,65536)==0
    np.testing.assert_array_equal(a[:65536],table)
    np.testing.assert_array_equal(a[-3:],[5,6,7])


def test_rejections_are_atomic(bridge):
    good = next(tables())
    for value in (np.nan,np.inf,40000.):
        table = good.copy(); table[100]=value
        out=np.full(65536,12345,dtype=np.uint16)
        assert bridge.lib.acr_test_u16_lut_build(ptr(table),4098,ptr(out),65536)!=0
        assert np.all(out==12345)
    table=good.copy();table[-1]=0
    out=np.full(65536,12345,dtype=np.uint16)
    assert bridge.lib.acr_test_u16_lut_build(ptr(table),4098,ptr(out),65536)!=0
    assert np.all(out==12345)
    table=np.arange(65536,dtype=np.uint16)
    src=np.arange(16,dtype=np.uint16); saved=src.copy()
    assert bridge.lib.acr_test_u16_lut_apply(ptr(table),65536,ptr(src),16,
        C.cast(C.c_void_p(src.ctypes.data+2),U),15,15)!=0
    np.testing.assert_array_equal(src,saved)
    size_max=S(-1).value
    assert bridge.lib.acr_test_u16_lut_apply(ptr(table),65536,ptr(src),size_max,
        ptr(out),size_max,size_max//2+1)!=0
    assert np.all(out==12345)
