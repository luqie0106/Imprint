from __future__ import annotations
import ctypes
import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('core3_green_reference',ROOT/'tests/acr_support/core3_green_oracle.py')
oracle=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(oracle)
PTR=ctypes.POINTER(ctypes.c_float)
def ptr(a,offset=0): return ctypes.cast(a.ctypes.data+4*offset,PTR)

@pytest.fixture(scope='module',params=['source','official_object'])
def bridge(request,tmp_path_factory):
    target=tmp_path_factory.mktemp('green_bridge')/('green.dylib' if sys.platform=='darwin' else 'green.so')
    source=ROOT/'native-renderer/src/acr_stage3_core3_green.cpp'
    obj=ROOT/'native-renderer/build/CMakeFiles/imprint_renderer.dir/src/acr_stage3_core3_green.cpp.o'
    if request.param=='official_object':
        assert obj.exists(), 'Build formal renderer object before running official-object tests'
        assert obj.stat().st_mtime>=source.stat().st_mtime, 'Formal object is stale'
    subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',
      '-I',str(ROOT/'native-renderer/src'),str(ROOT/'native-renderer/tools/acr_stage3_core3_green_bridge.cpp'),
      str(source if request.param=='source' else obj),'-o',str(target)],check=True,capture_output=True)
    lib=ctypes.CDLL(str(target));fn=lib.core3_green_bridge
    fn.argtypes=[PTR]*7+[ctypes.c_size_t]*3+[ctypes.c_uint32]+[PTR]*3
    fn.restype=ctypes.c_int
    return fn

@pytest.mark.parametrize('phase',[0,1])
@pytest.mark.parametrize('controls',[(0,0.02,0,0),(10,0.03,0.7,0.2),(25,0.1,1,0.5),(40,0.1,0.8,1),(0,1,0.9,0.0001)])
def test_green_random_instruction_reference(bridge,phase,controls):
    for seed in range(5):
        rng=np.random.default_rng(910+seed);width=16;stride=32;y=4;x=4
        source=rng.uniform(-0.25,1.25,(9,stride)).astype(np.float32)
        planes={n:rng.uniform(0,4,(9,stride)).astype(np.float32) for n in (0,1,2,5,7,8)}
        if seed==0:
            for p in planes.values():p.fill(0)
        if seed==1:
            for n in (0,1,2,5):planes[n].fill(0)
        if seed==3: planes[7]*=np.float32(100)
        if seed==4: planes[5]*=np.float32(100)
        saved=[a.copy() for a in [source,*planes.values()]]
        green=np.full(width+4,-77,np.float32);aux=np.full(width+4,-88,np.float32)
        args=[ptr(source,y*stride+x)]+[ptr(planes[n],y*stride+x) for n in (0,1,2,5,7,8)]
        c=np.array(controls,np.float32)
        status=bridge(*args,stride,stride,width,phase,ptr(c),ptr(green),ptr(aux))
        assert status==0
        eg,ea=oracle.reference(source,planes,y,x,width,phase,controls)
        np.testing.assert_array_equal(green[:width].view(np.uint32),eg.view(np.uint32))
        if controls[2]:np.testing.assert_array_equal(aux[:width].view(np.uint32),ea.view(np.uint32))
        else:assert np.all(aux==-88)
        assert np.all(green[width:]==-77) and np.all(aux[width:]==-88)
        for a,b in zip([source,*planes.values()],saved):np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))

def test_green_rejects_before_writes(bridge):
    data=np.ones((9,32),np.float32);p=ptr(data,4*32+4)
    g=np.full(8,-9,np.float32);a=np.full(8,-8,np.float32)
    good=np.array([10,0.02,1,0.2],np.float32)
    for stride,width,phase,c in [(2**63,4,0,good),(32,3,0,good),(32,4,2,good),(32,4,0,np.array([0,1,1,-1],np.float32))]:
        assert bridge(*([p]*7),stride,stride,width,phase,ptr(c),ptr(g),ptr(a))!=0
        assert np.all(g==-9) and np.all(a==-8)

@pytest.mark.parametrize('vertical',[False,True])
def test_green_hardcoded_direction_probe(bridge,vertical):
    stride=16;y=4;x=4;width=4
    source=np.full((9,stride),0.25,np.float32)
    source[y,x-2]=0.25;source[y,x-1]=0.125
    source[y,x+1]=0.375;source[y,x+2]=0.75
    source[y-2,x]=0.25;source[y-1,x]=0.5
    source[y+1,x]=0.75;source[y+2,x]=0.5
    planes={n:np.ones((9,stride),np.float32) for n in (0,1,2,5,7,8)}
    planes[8 if vertical else 7].fill(0)
    controls=np.array([10,0.03,0.7,0.2],np.float32)
    green=np.full(8,-9,np.float32);aux=np.full(8,-8,np.float32)
    args=[ptr(source,y*stride+x)]+[ptr(planes[n],y*stride+x) for n in (0,1,2,5,7,8)]
    assert bridge(*args,stride,stride,width,0,ptr(controls),ptr(green),ptr(aux))==0
    # Exact binary fractions: H=3/16, V=19/32. E angle is 0 or 90;
    # second angle is 45 so it cannot override the selected direction.
    assert green[0]==np.float32(19/32 if vertical else 3/16)
    assert green[1]==source[y,x+1]
    assert aux[0]==0
