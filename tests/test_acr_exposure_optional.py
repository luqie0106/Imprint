"""Independent original-instruction checks for prepared optional assembly."""
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
import exposure_optional_oracle as oracle

PTR=C.POINTER(C.c_double)
SENTINEL=-1234.5
MODULES=('acr_exposure_optional_prepare','acr_exposure_optional_prepare_api',
         'acr_exposure_qcap','acr_exposure_black_prepare','acr_negative_curve','acr_whites')
DEFAULT=(-10.,40.,1.,1.,.005,0.,0.,0.,0.,-8.,-1.,-2.,0.,0.,.1,.2,.3,0.)
NAMES=('blacks','pg','producer','mode','k','e98','a0','b0','white80','lo','hi',
       'a','high','a8','red','green','blue','caller')


def context(**changes):
    return tuple(changes.get(n,v) for n,v in zip(NAMES,DEFAULT))


def ptr(x):
    return x.ctypes.data_as(PTR) if x is not None else C.cast(C.c_void_p(),PTR)


def bits(x):
    return np.asarray(x,dtype=np.float64).view(np.uint64)


def equal_bits(a,b):
    np.testing.assert_array_equal(bits(a),bits(b))


def compile_bridge(kind,path):
    compiler=shutil.which('clang++') or shutil.which('c++')
    assert compiler
    bridge=NATIVE/'tools/acr_exposure_optional_prepare_bridge.cpp'
    assert bridge.is_file()
    inputs=[]
    for name in MODULES:
        source=NATIVE/f'src/{name}.cpp'
        assert source.is_file()
        if kind=='source':
            inputs.append(source)
        else:
            obj=NATIVE/f'build/CMakeFiles/imprint_renderer.dir/src/{name}.cpp.o'
            assert obj.is_file(),f'missing formal object {obj.name}'
            assert obj.stat().st_mtime_ns>=source.stat().st_mtime_ns,f'stale {obj.name}'
            inputs.append(obj)
    subprocess.run([compiler,'-std=c++17','-O2','-fPIC','-fno-fast-math','-ffp-contract=off',
                    '-dynamiclib' if sys.platform=='darwin' else '-shared',
                    '-I',str(NATIVE/'src'),'-I',str(NATIVE/'include'),str(bridge),
                    *map(str,inputs),'-o',str(path)],check=True,capture_output=True,text=True)


class Bridge:
    def __init__(self,path,formal=False):
        self.lib=C.CDLL(str(path))
        self.fn=getattr(self.lib,'im_native_acr_prepare_exposure_optional' if formal
                        else 'acr_test_prepare_exposure_optional')
        self.fn.argtypes=[PTR,C.c_size_t,PTR,C.c_size_t]
        self.fn.restype=C.c_int

    def prepare(self,controls,count=None,capacity=18):
        controls=np.ascontiguousarray(controls,dtype=np.float64)
        before=controls.copy()
        out=np.full(18,SENTINEL,np.float64)
        status=self.fn(ptr(controls),controls.size if count is None else count,ptr(out),capacity)
        equal_bits(controls,before)
        return status,out


@pytest.fixture(scope='module',params=('source','official_object'))
def bridge(request,tmp_path_factory):
    path=tmp_path_factory.mktemp('optional_'+request.param)/('liboptional.dylib' if sys.platform=='darwin' else 'liboptional.so')
    compile_bridge(request.param,path)
    return Bridge(path)


@pytest.fixture(scope='module')
def formal():
    path=NATIVE/'build'/('libimprint_renderer.0.1.0.dylib' if sys.platform=='darwin' else 'libimprint_renderer.0.1.0.so')
    assert path.is_file()
    for name in MODULES:
        assert path.stat().st_mtime_ns >= (NATIVE/f'src/{name}.cpp').stat().st_mtime_ns
    return Bridge(path,True)


CASES=[context(producer=p,caller=c,blacks=b,mode=m)
       for p,c in ((0.,0.),(1.,1.),(1.,0.),(0.,1.))
       for b in (-100.,-10.,0.,25.,100.) for m in (0.,1.)]
CASES += [context(a0=a0,b0=b0,white80=.7) for a0 in (-120.,-8.,-.5,0.,.5,8.,9.,120.)
          for b0 in (-1.,0.,1.)]
CASES += [context(red=0.,green=0.,blue=0.,blacks=0.,producer=0.),
          context(red=0.,green=0.,blue=0.,blacks=100.,producer=0.),
          context(red=0.,green=.3,blue=0.),context(a8=-120.),
          context(a8=120.,red=.1,green=.2,blue=.3),context(red=100.,green=200.,blue=300.)]


@pytest.mark.parametrize('controls',CASES)
def test_source_official_exact_prepared_assembly(bridge,controls):
    status,out=bridge.prepare(controls)
    assert status==0
    equal_bits(out,oracle.prepare(controls))


def test_fixed_separate_predicates_and_distinct_cap(bridge):
    # Literal expected bits fixed from raw-instruction oracle before native run.
    expected=np.array([0xbff0000000000000,0x3fdfffffffffffff,0x3fdfffffffffffff,
        0x3f847ae147ae1478,0,0x3f2a36e2eb1c4329,0x3fc6a09e667f3bcb,
        0x3ff0000000000000,0,0x3fb9a5657fb69985,0,0x3fc6a09e667f3bcb,
        0x3f97f1cb03e4b548,0x3fc6a09e667f3bcb,0x3fbf94bccf3a38b2,
        0x3fc999999999999a,0,0x3fb999999999999a],np.uint64)
    np.testing.assert_array_equal(bits(oracle.prepare(DEFAULT)),expected)
    status,out=bridge.prepare(DEFAULT)
    assert status==0
    np.testing.assert_array_equal(bits(out),expected)
    assert out[6]!=out[2]  # cap is not q
    reverse=context(producer=0.,caller=1.)
    status,out=bridge.prepare(reverse)
    assert status==0
    equal_bits(out,oracle.prepare(reverse))
    assert bits(out[15])==np.uint64(0x3f947ae147ae1478)
    assert bits(out[17])==np.uint64(0x3f847ae147ae1478)


def test_inactive_channels_skip_cap_and_second_predicate(bridge):
    for caller in (0.,1.):
        controls=context(red=0.,green=0.,blue=0.,blacks=100.,producer=0.,caller=caller)
        status,out=bridge.prepare(controls)
        assert status==0 and out[7]==0 and out[8]==1
        equal_bits(out[[6,9,10,11,12,13,14,15,17]],np.zeros(9))
        equal_bits(out,oracle.prepare(controls))


def test_one_positive_channel_constructs_all_three(bridge):
    controls=context(red=0.,green=.3,blue=0.,blacks=0.,producer=0.)
    status,out=bridge.prepare(controls)
    assert status==0 and out[7]==1
    equal_bits(out,oracle.prepare(controls))
    equal_bits(out[[9,10,13,14]],np.zeros(4))


def test_upper_literal_is_point99(bridge):
    controls=context(red=100.,green=200.,blue=300.)
    status,out=bridge.prepare(controls)
    assert status==0
    equal_bits(out,oracle.prepare(controls))
    for a,b in zip(out[[9,11,13]],out[[10,12,14]]):
        assert a+b==oracle.UPPER
    assert oracle.UPPER==.99 and oracle.UPPER!=.98


def test_random_uint64(bridge):
    rng=np.random.default_rng(0x922f20)
    for _ in range(220):
        controls=(int(rng.integers(-100,101)),int(rng.integers(-100,101)),
                  int(rng.integers(0,2)),int(rng.integers(0,2)),.005,
                  rng.uniform(-3,2),rng.uniform(-12,12),rng.uniform(-2,2),rng.uniform(-2,2),
                  rng.uniform(-12,-6),rng.uniform(-3,0),rng.uniform(-6,-1),rng.uniform(-2,2),
                  rng.uniform(-2,2),*rng.uniform(0,.9,3),int(rng.integers(0,2)))
        status,out=bridge.prepare(controls)
        assert status==0
        equal_bits(out,oracle.prepare(controls))


@pytest.mark.parametrize('k,base', [(1e307, 1e308), (-1e307, 1e308),
                                   (1e307, np.finfo(float).max)])
def test_unclamped_nonfinite_values_follow_original_min_and_trigger(bridge,k,base):
    controls=context(k=k,blacks=-10.,producer=0.,a8=0.,red=base,green=base,blue=base)
    expected=oracle.prepare(controls)
    status,out=bridge.prepare(controls)
    assert status==0
    equal_bits(out,expected)
    # The first case has three negative infinities and skips the entire
    # channel branch; the other two recover to the numeric .99 upper bound.
    assert out[7] == (0. if k>0 and base==1e308 else 1.)


INVALID=[context(**{name:v}) for name in NAMES for v in (math.nan,math.inf,-math.inf)]
INVALID += [context(blacks=.5),context(pg=2**31),context(producer=.5),context(mode=2.),
            context(caller=-1.),context(a8=120.1),context(red=-.01),context(a0=120.1),
            context(a=121.,high=121.),context(a8=120.,red=np.finfo(float).max)]


@pytest.mark.parametrize('controls',INVALID)
def test_failure_is_atomic(bridge,controls):
    status,out=bridge.prepare(controls)
    assert status!=0
    equal_bits(out,[SENTINEL]*18)


def _buffer_contract(fn):
    controls=np.array(DEFAULT,np.float64)
    out=np.full(24,SENTINEL,np.float64)
    before=out.copy()
    for cp,count,op,capacity in ((None,18,ptr(out),24),(ptr(controls),17,ptr(out),24),
        (ptr(controls),19,ptr(out),24),(ptr(controls),18,None,24),(ptr(controls),18,ptr(out),17),
        (C.cast(controls.ctypes.data+1,PTR),18,ptr(out),24),
        (ptr(controls),18,C.cast(out.ctypes.data+1,PTR),24),
        (ptr(controls),18,ptr(out),C.c_size_t(-1).value)):
        assert fn(cp,count,op,capacity)!=0
        equal_bits(out,before)
    for offset in (0,18):
        storage=np.full(40,SENTINEL,np.float64)
        storage[offset:offset+18]=controls
        before_alias=storage.copy()
        assert fn(C.cast(storage.ctypes.data+offset*8,PTR),18,ptr(storage),40)!=0
        equal_bits(storage,before_alias)
    assert fn(ptr(controls),18,ptr(out),24)==0
    equal_bits(out[:18],oracle.prepare(DEFAULT))
    equal_bits(out[18:],before[18:])


def test_bridge_buffers(bridge):
    _buffer_contract(bridge.fn)


def test_formal_api(formal):
    _buffer_contract(formal.fn)
    for controls in CASES:
        status,out=formal.prepare(controls)
        assert status==0
        equal_bits(out,oracle.prepare(controls))
    for controls in INVALID:
        status,out=formal.prepare(controls)
        assert status!=0
        equal_bits(out,[SENTINEL]*18)


def test_python_adapter():
    sys.path.insert(0,str(ROOT/'src'))
    import native_renderer
    for controls in CASES:
        out=native_renderer.native_acr_prepare_exposure_optional(controls)
        assert out.dtype==np.float64 and out.shape==(18,)
        equal_bits(out,oracle.prepare(controls))
    for controls in INVALID:
        with pytest.raises((ValueError,RuntimeError)):
            native_renderer.native_acr_prepare_exposure_optional(controls)


@pytest.mark.parametrize('controls', [DEFAULT,
    context(red=0.,green=.3,blue=0.,blacks=0.,producer=0.),
    context(red=0.,green=0.,blue=0.,blacks=100.,producer=0.),
    context(red=0.,green=0.,blue=0.,blacks=0.,producer=0.),
    context(blacks=100.,producer=0.),
])
def test_prepared_plan_builds_native_tables_without_identity_shortcuts(controls):
    sys.path.insert(0,str(ROOT/'src'))
    import native_renderer
    import exposure_context_lut_oracle as tables
    plan,luts=native_renderer.native_acr_prepare_exposure_optional_luts(controls)
    expected=oracle.prepare(controls)
    equal_bits(plan,expected)
    assert len(luts)==4
    for channel in range(3):
        if expected[7]:
            reference=tables.channel_lut(expected[9+2*channel],expected[10+2*channel],expected[15])
            assert luts[channel].dtype==np.float32 and luts[channel].shape==(4098,)
            np.testing.assert_array_equal(luts[channel].view(np.uint32),reference.view(np.uint32))
        else:
            assert luts[channel] is None
    if expected[8]:
        reference=tables.post_lut(expected[16])
        np.testing.assert_array_equal(luts[3].view(np.uint32),reference.view(np.uint32))
    else:
        assert luts[3] is None
