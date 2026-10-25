from __future__ import annotations
import ctypes
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
F=ctypes.POINTER(ctypes.c_float);U=ctypes.POINTER(ctypes.c_uint16)

@pytest.fixture(scope='module',params=['source','official_object'])
def bridge(request,tmp_path_factory):
    source=ROOT/'native-renderer/src/acr_stage3_core3_format.cpp'
    obj=ROOT/'native-renderer/build/CMakeFiles/imprint_renderer.dir/src/acr_stage3_core3_format.cpp.o'
    if request.param=='official_object':
        assert obj.exists(),'Build formal format object first'
        assert obj.stat().st_mtime_ns>=source.stat().st_mtime_ns,'Format object is stale'
    target=tmp_path_factory.mktemp('format_bridge')/('format.dylib' if sys.platform=='darwin' else 'format.so')
    subprocess.run(['c++','-std=c++17','-O2','-fPIC','-shared','-fno-fast-math','-ffp-contract=off',
       '-I',str(ROOT/'native-renderer/src'),str(ROOT/'native-renderer/tools/acr_stage3_core3_format_bridge.cpp'),
       str(source if request.param=='source' else obj),'-o',str(target)],check=True,capture_output=True)
    lib=ctypes.CDLL(str(target))
    lib.core3_format_u16_f32.argtypes=[U,ctypes.c_size_t,F,ctypes.c_size_t,ctypes.c_size_t]
    lib.core3_format_f32_u16.argtypes=[F,ctypes.c_size_t,U,ctypes.c_size_t,ctypes.c_size_t]
    lib.core3_format_u16_f32.restype=lib.core3_format_f32_u16.restype=ctypes.c_int
    return lib

def p(a,t):return a.ctypes.data_as(t)

def test_entire_uint16_domain(bridge):
    source=np.arange(65536,dtype=np.uint16);before=source.copy()
    output=np.full(65540,-7,np.float32)
    assert bridge.core3_format_u16_f32(p(source,U),source.size,p(output,F),output.size,source.size)==0
    scale=np.array([0x37800080],np.uint32).view(np.float32)[0]
    expected=np.multiply(source.astype(np.float32),scale,dtype=np.float32)
    np.testing.assert_array_equal(output[:65536].view(np.uint32),expected.view(np.uint32))
    assert output[257].view(np.uint32)==0x3b808080
    wrong=(source.astype(np.float64)/65535).astype(np.float32)
    assert np.count_nonzero(wrong.view(np.uint32)!=expected.view(np.uint32))==512
    assert np.all(output[65536:]==-7)
    back=np.full(65540,12345,np.uint16)
    assert bridge.core3_format_f32_u16(p(output,F),65536,p(back,U),back.size,65536)==0
    np.testing.assert_array_equal(back[:65536],source)
    np.testing.assert_array_equal(source,before)
    assert np.all(back[65536:]==12345)


def test_half_integer_neighbors_and_random(bridge):
    n=np.arange(65532,dtype=np.float64)
    centers=((n+0.5)/65535).astype(np.float32)
    source=np.concatenate([np.nextafter(centers,np.float32(0)),centers,
       np.nextafter(centers,np.float32(1)),np.random.default_rng(19010).random(32768,dtype=np.float32)])
    before=source.view(np.uint32).copy();output=np.full(source.size+4,777,np.uint16)
    product=np.multiply(source,np.float32(65535),dtype=np.float32)
    expected=np.rint(product).astype(np.uint16)
    assert bridge.core3_format_f32_u16(p(source,F),source.size,p(output,U),output.size,source.size)==0
    np.testing.assert_array_equal(output[:source.size],expected)
    np.testing.assert_array_equal(source.view(np.uint32),before)
    assert np.all(output[source.size:]==777)
    probes=np.array([0,0.5,1,np.float32(2.5/65535)],np.float32);out=np.zeros(4,np.uint16)
    assert bridge.core3_format_f32_u16(p(probes,F),4,p(out,U),4,4)==0
    np.testing.assert_array_equal(out,[0,32768,65535,2])

@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf,-0.01,1.01])
def test_reject_domain_before_first_write(bridge,bad):
    source=np.full(16,0.5,np.float32);source[-1]=bad
    output=np.full(20,1234,np.uint16)
    assert bridge.core3_format_f32_u16(p(source,F),source.size,p(output,U),output.size,source.size)!=0
    assert np.all(output==1234)


def test_rejections_and_alias_leave_bytes(bridge):
    source=np.arange(16,dtype=np.uint16);output=np.full(20,-7,np.float32)
    for width,capacity in [(3,20),(16,15),(2**63,20)]:
        assert bridge.core3_format_u16_f32(p(source,U),source.size,p(output,F),capacity,width)!=0
        assert np.all(output==-7)
    raw=np.zeros(128,np.uint8);before=raw.copy()
    assert bridge.core3_format_u16_f32(ctypes.cast(raw.ctypes.data,U),16,ctypes.cast(raw.ctypes.data,F),16,16)!=0
    np.testing.assert_array_equal(raw,before)
    assert bridge.core3_format_u16_f32(None,16,p(output,F),20,16)!=0
    assert np.all(output==-7)
