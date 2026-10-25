"""Independent offline response/producer validation, not Adobe runtime parity."""
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
NATIVE = ROOT / 'native-renderer'
SOURCE = NATIVE / 'src/acr_exposure_black_prepare.cpp'
HEADER = SOURCE.with_suffix('.hpp')
API = NATIVE / 'src/acr_exposure_black_prepare_api.cpp'
BRIDGE = NATIVE / 'tools/acr_exposure_black_prepare_bridge.cpp'
OBJECT = NATIVE / 'build/CMakeFiles/imprint_renderer.dir/src/acr_exposure_black_prepare.cpp.o'
API_OBJECT = OBJECT.with_name('acr_exposure_black_prepare_api.cpp.o')
sys.path.insert(0, str(ROOT / 'tests/acr_support'))
sys.path.insert(0, str(ROOT / 'src'))
import exposure_black_prepare_oracle as oracle
import native_renderer as native
DP = C.POINTER(C.c_double)

def ptr(a):
    return a.ctypes.data_as(DP)

def assert_bits(actual, expected):
    # Both sides use this host's libm. Compare every binary64 bit, including
    # signed zero; tolerance would hide an operation-order regression.
    a = np.asarray(actual, dtype=np.float64).view(np.uint64)
    e = np.asarray(expected, dtype=np.float64).view(np.uint64)
    np.testing.assert_array_equal(a, e)

@pytest.fixture(scope='module', params=['source', 'object'])
def bridge(request, tmp_path_factory):
    compiler = shutil.which('clang++') or shutil.which('c++')
    assert compiler
    inputs = [SOURCE]
    if request.param == 'object':
        assert OBJECT.exists(), 'rebuild renderer: missing black response object'
        assert OBJECT.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in (SOURCE, HEADER))
        inputs = [OBJECT]
    out = tmp_path_factory.mktemp('black_prepare') / 'bridge.dylib'
    subprocess.run([compiler, '-std=c++17', '-O2', '-fPIC', '-fno-fast-math',
                    '-ffp-contract=off', '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                    '-I', str(NATIVE / 'src'), str(BRIDGE), *map(str, inputs), '-o', str(out)],
                   check=True, capture_output=True, text=True)
    lib = C.CDLL(str(out))
    lib.acr_test_black_response.argtypes = [C.c_double] * 5 + [DP]
    lib.acr_test_black_response.restype = C.c_int
    lib.acr_test_prepare_black.argtypes = [C.c_int32, C.c_int32, C.c_uint32, C.c_uint32,
                                          C.c_double, C.c_double, DP, C.c_size_t]
    lib.acr_test_prepare_black.restype = C.c_int
    return lib

# Literal hex canaries frozen from the disassembly-derived oracle before reading
# the C++ implementation. Include ascending, descending (nonzero t), equality,
# linear and literal near-equal constructor behavior.
@pytest.mark.parametrize('args,expected', [
    ((.6, 0, .25, .005, .5), '0x1.089c13573b5d4p-4'),
    ((.6, 0, .25, .2, .3), '0x1.84d818940a712p-2'),
    ((.5, -.25, 0, .2, .3), '0x1.27d27d27d27d2p-2'),
    ((.8, 0, .25, .75, 3), '0x1.8000000000000p+1'),
    ((.1, 0, .25, .005, .5), '0x1.0624dd2f1a9fcp-9'),
])
def test_response_scalar_canaries(bridge, args, expected):
    target = float.fromhex(expected)
    assert_bits(oracle.response(*args), target)
    out = np.array([-1234.5])
    assert bridge.acr_test_black_response(*args, ptr(out)) == 0
    assert_bits(out[0], target)

@pytest.mark.parametrize('t0,t1,k,q', [(0,.25,.005,.5),(-.25,0,.005,.9),
                                      (0,.25,.2,.3),(0,.25,.75,3)])
def test_response_thresholds_and_signed_zero(bridge,t0,t1,k,q):
    for z in [t0,t1,math.nextafter(t1,-math.inf),math.nextafter(t1,math.inf),
              math.nextafter(1,-math.inf),1,math.nextafter(1,math.inf),-0.0,0.0]:
        out = np.array([-1234.5])
        assert bridge.acr_test_black_response(z,t0,t1,k,q,ptr(out)) == 0
        assert_bits(out[0],oracle.response(z,t0,t1,k,q))
    for k in [0.0,-0.0]:
        out = np.array([123.0])
        assert bridge.acr_test_black_response(-0.0,0,.25,k,.5,ptr(out)) == 0
        assert_bits(out[0],oracle.response(-0.0,0,.25,k,.5))

@pytest.mark.parametrize('args,status', [
    ((.6,0,.25,.25,1),2),  # equal slopes, literal Q invalid
    ((.6,0,.25,1e-10,-.1),2), # descending then near-equal to zero
    ((.6,0,.25,0,-.1),2),
    ((.6,0,0,.005,.5),1),
    ((.6,0,1,.005,.5),1),
    ((math.nan,0,.25,.005,.5),1),
    ((.6,0,.25,math.inf,.5),1),
    ((.6,0,math.nextafter(0,1),1e308,.5),3),
    ((-1e308,0,.25,1e308,.5),3),
])
def test_response_errors_do_not_write(bridge,args,status):
    out=np.array([-1234.5])
    before=out.tobytes()
    assert bridge.acr_test_black_response(*args,ptr(out)) == status
    assert out.tobytes() == before

def test_default_prepare_full_control_sweep(bridge):
    for mode in (0,1):
        for active in (0,1):
            for pg in (-100,0,75):
                for q in (.0625,.1,.5,.9):
                    for blacks in range(-100,101):
                        out=np.full(5,-1234.5)
                        assert bridge.acr_test_prepare_black(blacks,pg,active,mode,.005,q,ptr(out),5)==0
                        assert_bits(out[:3],oracle.prepare(blacks,pg,active,mode,.005,q))
                        np.testing.assert_array_equal(out[3:],[-1234.5]*2)

def test_prepare_errors_and_extreme_int32(bridge):
    for blacks,pg in [(-2**31,2**31-1),(2**31-1,-2**31)]:
        for active in (0,1):
            for mode in (0,1):
                out=np.full(3,-1234.5)
                assert bridge.acr_test_prepare_black(blacks,pg,active,mode,.005,.5,ptr(out),3)==0
                assert_bits(out,oracle.prepare(blacks,pg,active,mode,.005,.5))
    for active,mode,k,q in [(2,0,.005,.5),(0,2,.005,.5),(0,0,math.nan,.5),
                            (0,0,.005,.01),(0,0,.005,math.inf),(0,0,.005,.91)]:
        out=np.full(3,-1234.5); before=out.tobytes()
        assert bridge.acr_test_prepare_black(-50,0,active,mode,k,q,ptr(out),3)==1
        assert out.tobytes()==before
    out=np.full(3,-1234.5); before=out.tobytes()
    assert bridge.acr_test_prepare_black(0,0,0,0,.005,.5,ptr(out),2)==1
    assert out.tobytes()==before
    assert bridge.acr_test_prepare_black(0,0,0,0,.005,.5,None,3)==1

@pytest.fixture(scope='module')
def formal():
    path=NATIVE/'build'/('libimprint_renderer.0.1.0.dylib' if sys.platform=='darwin' else 'libimprint_renderer.0.1.0.so')
    for obj,inputs in [(OBJECT,(SOURCE,HEADER)),(API_OBJECT,(API,NATIVE/'include/imprint_renderer.h'))]:
        assert obj.is_file() and obj.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in inputs)
        assert path.stat().st_mtime_ns >= obj.stat().st_mtime_ns
    lib=C.CDLL(str(path)); native._configure_library(lib)
    fn=lib.im_native_acr_prepare_black_response
    fn.argtypes=[DP,C.c_size_t,DP,C.c_size_t]; fn.restype=C.c_int
    return lib,str(path)

def test_formal_and_adapter_sweep(formal,monkeypatch):
    lib,path=formal
    monkeypatch.setattr(native,'_get_library',lambda:(lib,path))
    for mode in (0,1):
        for active in (0,1):
            for q in (.0625,.5,.9):
                for blacks in range(-100,101):
                    controls=np.array([blacks,37,active,mode,.005,q],dtype=np.float64)
                    original=controls.tobytes(); out=np.full(5,-1234.5)
                    assert lib.im_native_acr_prepare_black_response(ptr(controls),6,ptr(out),5)==0
                    expected=oracle.prepare(blacks,37,active,mode,.005,q)
                    assert_bits(out[:3],expected)
                    assert controls.tobytes()==original
                    np.testing.assert_array_equal(out[3:],[-1234.5]*2)
                    adapted=native.native_acr_prepare_black_response(blacks,37,gain_adjust_active=active,mode=mode,k=.005,q=q)
                    assert adapted.dtype==np.float64 and adapted.shape==(3,)
                    assert_bits(adapted,expected)

def test_formal_errors_alias_alignment_padding(formal):
    lib,_=formal; fn=lib.im_native_acr_prepare_black_response
    controls=np.array([0,0,0,0,.005,.5]); out=np.full(4,-1234.5)
    before=out.tobytes()
    for count in (0,5,7):
        assert fn(ptr(controls),count,ptr(out),4)==1
        assert out.tobytes()==before
    assert fn(ptr(controls),6,ptr(out),2)==4
    assert fn(None,6,ptr(out),4)==1
    assert fn(ptr(controls),6,None,3)==1
    for i,value in [(0,.1),(1,2**31),(2,2),(3,-1),(4,math.nan),(5,.91),(5,math.inf)]:
        bad=controls.copy(); bad[i]=value
        assert fn(ptr(bad),6,ptr(out),4)==1
        assert out.tobytes()==before
    arena=np.full(20,-1234.5); arena[6:12]=controls; snapshot=arena.tobytes()
    # The declared output padding overlaps controls even though its first three do not.
    assert fn(ptr(arena[6:]),6,ptr(arena),7)==1
    assert arena.tobytes()==snapshot
    assert fn(ptr(arena[6:]),6,ptr(arena[7:]),3)==1
    assert arena.tobytes()==snapshot
    storage=(C.c_ubyte*64)(*range(64)); snapshot=bytes(storage)
    misaligned=C.cast(C.addressof(storage)+1,DP)
    assert fn(ptr(controls),6,misaligned,3)==1
    assert bytes(storage)==snapshot
    assert fn(misaligned,6,ptr(out),4)==1
    assert out.tobytes()==before

@pytest.mark.parametrize('kwargs', [
    dict(blacks=True),dict(blacks=1.2),dict(profile_gain_param=1.2),
    dict(blacks=2**31),dict(profile_gain_param=-2**31-1),
    dict(gain_adjust_active=2),dict(mode=.5),dict(k=math.nan),dict(q=math.inf),dict(q=.01),
])
def test_adapter_rejects_invalid(formal,monkeypatch,kwargs):
    lib,path=formal; monkeypatch.setattr(native,'_get_library',lambda:(lib,path))
    params=dict(blacks=0,profile_gain_param=0,gain_adjust_active=False,mode=True,k=.005,q=.5)
    params.update(kwargs)
    with pytest.raises((ValueError,TypeError,RuntimeError)):
        native.native_acr_prepare_black_response(**params)


def test_additional_response_domains_and_nonfinite_inputs(bridge):
    # Equality can succeed inside q's producer range when caller thresholds differ.
    for args in [(0.3,-13,-1,.75,.875),(.6,0,.25,-.2,.3),
                 (.6,0,.25,.2,-.5),(.6,-.25,0,.4,.1)]:
        out=np.array([-1234.5]); before=out.tobytes()
        try:
            expected=oracle.response(*args)
        except oracle.InvalidCurve:
            assert bridge.acr_test_black_response(*args,ptr(out))==2
            assert out.tobytes()==before
        else:
            assert bridge.acr_test_black_response(*args,ptr(out))==0
            assert_bits(out[0],expected)
    for index in range(5):
        for invalid in (math.nan,math.inf,-math.inf):
            args=[.6,0,.25,.005,.5]; args[index]=invalid
            out=np.array([-1234.5]); before=out.tobytes()
            assert bridge.acr_test_black_response(*args,ptr(out))==1
            assert out.tobytes()==before
    assert bridge.acr_test_black_response(.6,0,.25,.005,.5,None)==1


def test_producer_constructor_error_is_atomic(bridge):
    # m0=m1=.5 in mode1, producing the literal invalid equality constructor.
    with pytest.raises(oracle.InvalidCurve):
        oracle.prepare(-60,0,0,1,.125,.5)
    out=np.full(5,-1234.5); before=out.tobytes()
    assert bridge.acr_test_prepare_black(-60,0,0,1,.125,.5,ptr(out),5)==2
    assert out.tobytes()==before


def test_factory_epsilon_neighbors_and_endpoint_extension(bridge):
    for delta in (math.nextafter(oracle.EPS,0),oracle.EPS,
                  math.nextafter(oracle.EPS,math.inf)):
        # Descending branch passes m1=0 to F. Keep its exact epsilon decision.
        k=delta*.25
        for z in (.250000001,.6,1.0,2.0):
            args=(z,0,.25,k,-.1)
            out=np.array([-1234.5]); before=out.tobytes()
            try:
                expected=oracle.response(*args)
            except oracle.InvalidCurve:
                status=2
            except oracle.NonfiniteResult:
                status=3
            else:
                status=0
            assert bridge.acr_test_black_response(*args,ptr(out))==status
            if status==0:
                assert_bits(out[0],expected)
            else:
                assert out.tobytes()==before


def test_formal_span_size_overflow(formal):
    lib,_=formal; fn=lib.im_native_acr_prepare_black_response
    controls=np.array([0,0,0,0,.005,.5]); out=np.full(3,-1234.5)
    before=out.tobytes()
    assert fn(ptr(controls),6,ptr(out),C.c_size_t(-1).value)==1
    assert out.tobytes()==before


def test_nonfinite_intermediate_recovers_through_original_fminnm(bridge):
    # dq=q-k overflows, u becomes Inf, but FMINNM restores the finite upper
    # coordinate. Q construction and endpoint evaluation remain finite.
    for z in (1.,2.):
        args=(z,-1e308,0.,-1e308,1e308)
        assert oracle.response(*args)==1e308
        out=np.array([-1234.5])
        assert bridge.acr_test_black_response(*args,ptr(out))==0
        assert out[0]==1e308


def test_negative_slope_descending_factory_keeps_signed_divisor(bridge):
    # m0<0 and m1<m0: the raw signed divisor gives u>1, so original Q
    # validation rejects rather than repairing/reordering the points.
    for args in [(.6,0,.25,-.2,-1),(.5,-.25,0,-.2,-1),
                 (.6,0,.25,-oracle.EPS*.25,-.1)]:
        with pytest.raises(oracle.InvalidCurve):
            oracle.response(*args)
        out=np.array([-1234.5]); before=out.tobytes()
        assert bridge.acr_test_black_response(*args,ptr(out))==2
        assert out.tobytes()==before


def test_formal_constructor_error_is_atomic(formal):
    lib,_=formal; controls=np.array([-60,0,0,1,.125,.5])
    out=np.full(5,-1234.5); before=out.tobytes()
    assert lib.im_native_acr_prepare_black_response(ptr(controls),6,ptr(out),5)==1
    assert out.tobytes()==before
