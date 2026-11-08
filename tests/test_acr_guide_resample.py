"""Gaussian guide resampling against an independent saved-baseline oracle."""
import ctypes as C
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'native-renderer'
sys.path.insert(0,str(ROOT/'tests/acr_support'))
import guide_resample_oracle as oracle

F=C.POINTER(C.c_float)


@pytest.fixture(scope='module',params=('source','official_object'))
def native(request,tmp_path_factory):
    source=NATIVE/'src/acr_tone_kernels.cpp'
    header=NATIVE/'src/acr_tone_kernels.hpp'
    core=source
    if request.param=='official_object':
        core=NATIVE/'build/CMakeFiles/imprint_renderer.dir/src/acr_tone_kernels.cpp.o'
        assert core.is_file() and core.stat().st_mtime_ns>=max(source.stat().st_mtime_ns,header.stat().st_mtime_ns)
    path=tmp_path_factory.mktemp('guide_resample_'+request.param)/('libresize.dylib' if sys.platform=='darwin' else 'libresize.so')
    compiler=shutil.which('clang++') or shutil.which('c++')
    assert compiler
    subprocess.run([compiler,'-std=c++17','-O2','-fPIC','-fno-fast-math','-ffp-contract=off',
        '-dynamiclib' if sys.platform=='darwin' else '-shared',str(core),
        str(NATIVE/'tools/acr_guide_resample_bridge.cpp'),'-o',str(path)],check=True,capture_output=True,text=True)
    lib=C.CDLL(str(path))
    lib.acr_test_guide_resize.argtypes=[F,C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,F]
    lib.acr_test_guide_resize.restype=C.c_int
    return lib


def call(native,source,shape):
    source=np.ascontiguousarray(source,np.float32)
    before=source.copy()
    output=np.full(shape,-9123.25,np.float32)
    status=native.acr_test_guide_resize(source.ctypes.data_as(F),source.shape[1],source.shape[0],
                                      shape[1],shape[0],output.ctypes.data_as(F))
    np.testing.assert_array_equal(source.view(np.uint32),before.view(np.uint32))
    return status,output


@pytest.mark.parametrize('shape,destination',[
    ((1,1),(3,5)),((1,17),(1,4)),((17,1),(4,1)),((5,7),(3,4)),((3,4),(5,7)),
    ((8,8),(4,4)),((7,7),(4,4)),((4,4),(8,8)),((4,4),(7,7)),((19,23),(19,23)),
    ((33,41),(3,2)),((3,2),(33,41)),((3,257),(2,129)),((2,256),(1,128)),
])
def test_baseline_resize_and_borders(native,shape,destination):
    rng=np.random.default_rng(shape[0]*1000+shape[1])
    for kind in ('random','constant','edge'):
        source=rng.uniform(-20,4,shape).astype(np.float32)
        if kind=='constant':source.fill(-4.25)
        if kind=='edge':source[:,:shape[1]//2]=-12;source[:,shape[1]//2:]=2
        status,actual=call(native,source,destination)
        assert status==0
        expected=oracle.resize(source,destination)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def test_pyramid_reduce_expand(native):
    rng=np.random.default_rng(0x1341a30)
    source=rng.uniform(-12,4,(33,47)).astype(np.float32)
    levels=oracle.pyramid(source)
    for large,small in zip(levels,levels[1:]):
        status,actual=call(native,large,small.shape)
        assert status==0
        np.testing.assert_array_equal(actual.view(np.uint32),small.view(np.uint32))
        status,actual=call(native,small,large.shape)
        assert status==0
        expected=oracle.resize(small,large.shape)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def test_identity_preserves_signed_zero_and_extreme_bits(native):
    source=np.array([0.,-0.,np.finfo(np.float32).max,-np.finfo(np.float32).max,
                     np.nextafter(np.float32(0),np.float32(1)),1.],np.float32).reshape(2,3)
    status,actual=call(native,source,source.shape)
    assert status==0
    np.testing.assert_array_equal(actual.view(np.uint32),source.view(np.uint32))


def test_invalid_resize_preserves_output(native):
    source=np.ones((3,5),np.float32)
    for value in (np.nan,np.inf,-np.inf):
        source[-1,-1]=value
        status,actual=call(native,source,(7,9))
        assert status!=0 and np.all(actual==np.float32(-9123.25))
    source[-1,-1]=1.
    output=np.full(9,-9123.25,np.float32)
    for w,h,ow,oh in ((0,3,3,3),(5,3,0,3),(0xffffffff,2,3,3),(5,3,65536,3)):
        assert native.acr_test_guide_resize(source.ctypes.data_as(F),w,h,ow,oh,output.ctypes.data_as(F))!=0
        assert np.all(output==np.float32(-9123.25))
