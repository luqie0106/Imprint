"""Original instruction FMA acceptance for the3-plane prepared camera factory."""
from __future__ import annotations
import ctypes as C
import math
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'native-renderer'
sys.path.insert(0,str(ROOT/'tests/acr_support'))
import camera_prepare_instruction_oracle as oracle
PTR=C.POINTER(C.c_double)
SENTINEL=-1234.5
MODULES=('acr_camera_prepare','acr_camera_prepare_api')
T=(.713,.139,.051,.223,.647,.117,.034,.071,.793)
P=(1.31,-.27,-.04,-.36,1.47,-.11,.018,-.031,1.013)
CANARY=(.3,1.,.7,*T,*P,1.1,1.3,.9,37.,-23.,18.,45.,-42.,11.,57.)


def ptr(x):return x.ctypes.data_as(PTR)
def bits(x):return np.asarray(x,np.float64).view(np.uint64)
def equal(a,b):np.testing.assert_array_equal(bits(a),bits(b))


class API:
    def __init__(self,path):
        self.lib=C.CDLL(str(path))
        self.fn=self.lib.im_native_acr_prepare_camera_stage
        self.fn.argtypes=[PTR,C.c_size_t,PTR,C.c_size_t]
        self.fn.restype=C.c_int

    def prepare(self,x):
        x=np.ascontiguousarray(x,np.float64)
        out=np.full(30,SENTINEL,np.float64)
        before=x.copy()
        status=self.fn(ptr(x),x.size,ptr(out),out.size)
        equal(x,before)
        return status,out


@pytest.fixture(scope='module',params=('source','official_object','formal_api'))
def api(request,tmp_path_factory):
    if request.param=='formal_api':
        path=NATIVE/'build'/('libimprint_renderer.0.1.0.dylib' if sys.platform=='darwin' else 'libimprint_renderer.0.1.0.so')
        assert path.is_file()
        for name in MODULES:
            assert path.stat().st_mtime_ns >= (NATIVE/f'src/{name}.cpp').stat().st_mtime_ns
        return API(path)
    inputs=[]
    for name in MODULES:
        source=NATIVE/f'src/{name}.cpp'
        if request.param=='source':inputs.append(source)
        else:
            obj=NATIVE/f'build/CMakeFiles/imprint_renderer.dir/src/{name}.cpp.o'
            assert obj.is_file() and obj.stat().st_mtime_ns >= source.stat().st_mtime_ns
            inputs.append(obj)
    path=tmp_path_factory.mktemp('camera_instructions_'+request.param)/('libcamera.dylib' if sys.platform=='darwin' else 'libcamera.so')
    compiler=shutil.which('clang++') or shutil.which('c++')
    assert compiler
    subprocess.run([compiler,'-std=c++17','-O2','-fPIC','-fno-fast-math','-ffp-contract=off',
        '-dynamiclib' if sys.platform=='darwin' else '-shared','-I',str(NATIVE/'include'),
        *map(str,inputs),'-o',str(path)],check=True,capture_output=True,text=True)
    return API(path)


def _assert_fields(out,expected):
    equal(out[:9],expected['matrix'])
    equal(out[24:27],expected['bases'])


def test_fixed_original_instruction_canary(api):
    expected=oracle.prepare(CANARY)
    # Filled from the independent original-instruction oracle before comparing
    # the implementation; pinned below by a separate frozen-value test.
    status,out=api.prepare(CANARY)
    assert status==0
    _assert_fields(out,expected)


def test_seeded_prepared_inputs_exact(api):
    rng=np.random.default_rng(0x877fa8)
    for _ in range(160):
        white=[rng.uniform(.1,.99),1.,rng.uniform(.1,.99)]
        t=rng.uniform(-.3,1.4,9)
        p=rng.uniform(-.3,1.4,9)
        source=rng.uniform(.1,2.,3)
        cal=rng.integers(-100,101,6).astype(float)
        tint=float(rng.integers(-100,101))
        controls=(*white,*t,*p,*source,*cal,tint)
        expected=oracle.prepare(controls)
        status,out=api.prepare(controls)
        assert status==0
        _assert_fields(out,expected)


def test_zero_tint_is_zero_context_bases(api):
    controls=(*CANARY[:-1],0.)
    expected=oracle.prepare(controls)
    equal(expected['bases'],np.zeros(3))
    status,out=api.prepare(controls)
    assert status==0
    _assert_fields(out,expected)


@pytest.mark.parametrize('index,value',[(0,.0009),(0,math.nan),(21,0.),(24,.5),(30,101.)])
def test_existing_supported_input_rejection_is_atomic(api,index,value):
    controls=list(CANARY)
    controls[index]=value
    status,out=api.prepare(controls)
    assert status!=0
    equal(out,[SENTINEL]*30)


def test_frozen_canary_distinguishes_previous_separate_arithmetic():
    expected=oracle.prepare(CANARY)
    np.testing.assert_array_equal(bits(expected['matrix']),np.array([
        0x3fd7d40f60000000,0xbfc8a8f540000000,0xbfc2192dc0000000,
        0x3fb35be440000000,0x3fe40be920000000,0x3fb2d0f640000000,
        0x3f388a6e20000000,0xbf762b84c0000000,0x3fe3f20920000000],np.uint64))
    np.testing.assert_array_equal(bits(expected['bases']),np.array([
        0x3f62015aed1ef24c,0,0x3f6bc2a4a6296505],np.uint64))
    # Historical separate multiply/add yields different black compensation.
    historical=np.array([0x3f62015aed1ef254,0,0x3f6bc2a4a629650b],np.uint64)
    assert np.any(bits(expected['bases'])!=historical)


def test_python_adapter_fields():
    sys.path.insert(0,str(ROOT/'src'))
    import native_renderer
    out=native_renderer.native_acr_prepare_camera_stage(CANARY)
    assert out.shape==(30,) and out.dtype==np.float64
    _assert_fields(out,oracle.prepare(CANARY))


def test_context_bases_can_be_negative_when_minimum_exceeds_one(api):
    identity=(1.,0.,0.,0.,1.,0.,0.,0.,1.)
    large=(100.,0.,0.,0.,100.,0.,0.,0.,100.)
    controls=(.3,1.,.7,*large,*identity,1.1,1.3,.9,*([0.]*6),-100.)
    expected=oracle.prepare(controls)
    assert expected['minimum']>1. and expected['denominator']<0.
    assert np.any(expected['bases']<0.)
    status,out=api.prepare(controls)
    assert status==0
    _assert_fields(out,expected)


def test_minimum_one_is_declared_singular_domain_and_atomic(api):
    identity=(1.,0.,0.,0.,1.,0.,0.,0.,1.)
    scaled=(50.,0.,0.,0.,50.,0.,0.,0.,50.)
    controls=(.3,1.,.7,*scaled,*identity,1.,1.,1.,*([0.]*6),-100.)
    with pytest.raises(ValueError,match='denominator'):
        oracle.prepare(controls)
    status,out=api.prepare(controls)
    assert status!=0
    equal(out,[SENTINEL]*30)
