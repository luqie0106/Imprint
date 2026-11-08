"""Exact selected BuildStats fields against saved-instruction scalar oracle."""
import ctypes as C
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT/'native-renderer'
sys.path.insert(0,str(ROOT/'tests/acr_support'))
import log_scene_stats_oracle as oracle

F = C.POINTER(C.c_float)
D = C.POINTER(C.c_double)


@pytest.fixture(scope='module',params=('source','official_object'))
def native(request,tmp_path_factory):
    source = NATIVE/'src/acr_tone_kernels.cpp'
    header = NATIVE/'src/acr_tone_kernels.hpp'
    core = source
    if request.param=='official_object':
        core = NATIVE/'build/CMakeFiles/imprint_renderer.dir/src/acr_tone_kernels.cpp.o'
        assert core.is_file() and core.stat().st_mtime_ns>=max(source.stat().st_mtime_ns,header.stat().st_mtime_ns)
    path = tmp_path_factory.mktemp('log_stats_'+request.param)/('libstats.dylib' if sys.platform=='darwin' else 'libstats.so')
    compiler = shutil.which('clang++') or shutil.which('c++')
    assert compiler
    subprocess.run([compiler,'-std=c++17','-O2','-fPIC','-fno-fast-math','-ffp-contract=off',
        '-dynamiclib' if sys.platform=='darwin' else '-shared',str(core),
        str(NATIVE/'tools/acr_log_scene_stats_bridge.cpp'),'-o',str(path)],check=True,capture_output=True,text=True)
    lib = C.CDLL(str(path))
    lib.acr_test_log_scene_stats.argtypes = [F,C.c_size_t,C.c_double,D]
    lib.acr_test_log_scene_stats.restype = C.c_int
    return lib


def call(native,source,epsilon):
    source = np.ascontiguousarray(source,np.float32)
    before = source.copy()
    out = np.full(6,-1234.5,np.float64)
    status = native.acr_test_log_scene_stats(source.ctypes.data_as(F),source.size,epsilon,out.ctypes.data_as(D))
    np.testing.assert_array_equal(source.view(np.uint32),before.view(np.uint32))
    return status,out


def test_quantile_repeat_boundary(native):
    # floor(.99*(N-1)) selects -4; saved ceil selects 0. Threshold shifts from
    # -4.5 to -.5, so all 99 dark duplicates must survive, not just one.
    source = np.array([-4.]*99+[0.],np.float32)
    status,out = call(native,source,2**-12)
    expected,meta = oracle.selected_stats(source,2**-12)
    assert status==0 and meta['raw_high']==0 and out[4]==100
    np.testing.assert_array_equal(out.view(np.uint64),np.asarray(expected).view(np.uint64))
    # Saved center body on the independently filtered groups, epsilon=2^-12.
    assert out.view(np.uint64)[0]==13839524155527130545


@pytest.mark.parametrize('counts',[(20000,1),(20000,20000,1),(20000,20000,20000),
                                   (10000,30000,20,1),(4,100,600,1),(1,)])
def test_group_endpoint_scan_and_weighted_accumulation(native,counts):
    values = np.linspace(-8.,0.,len(counts),dtype=np.float32)
    source = np.repeat(values,counts)
    for epsilon in (2**-12,3.214e-6):
        status,out = call(native,source,epsilon)
        expected,_ = oracle.selected_stats(source,epsilon)
        assert status==0
        np.testing.assert_array_equal(out.view(np.uint64),np.asarray(expected).view(np.uint64))
        if counts==(20000,20000,20000):
            assert out[0]==-8. and out[4]==40001


def test_random_and_degenerate_stats(native):
    rng = np.random.default_rng(0x9d6fac)
    cases = [np.full(50,-30.),np.full(50,-4.),np.array([-11.,-11.,-4.,0.]),
             np.array([-120.,120.]),np.array([-0.,+0.])]
    for n in (2,3,7,99,100,101,1001,10003):
        cases += [rng.uniform(-20.,4.,n),rng.integers(-80,17,n)/4.]
    for source in cases:
        for epsilon in (2**-12,2**-20):
            status,out = call(native,source,epsilon)
            expected,_ = oracle.selected_stats(source,epsilon)
            assert status==0
            np.testing.assert_array_equal(out.view(np.uint64),np.asarray(expected).view(np.uint64))


def test_invalid_stats_preserve_destination(native):
    for source,epsilon in (([],2**-12),([0.,np.nan],2**-12),([np.inf],2**-12),
                           ([0.],0.),([0.],float('inf')),([0.],float('nan'))):
        status,out = call(native,source,epsilon)
        assert status!=0
        np.testing.assert_array_equal(out,np.full(6,-1234.5))
