"""Mask sampling controls: ceil interval count, fixed FMA ordering, prepared remap."""
from __future__ import annotations
import ctypes as C
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'native-renderer'
SENTINEL=-1234.5
F=C.POINTER(C.c_float)
D=C.POINTER(C.c_double)
LIBM=C.CDLL(None)
for name,argc in (('fma',3),('log',1),('exp2',1),('log2',1)):
    fn=getattr(LIBM,name)
    fn.argtypes=[C.c_double]*argc
    fn.restype=C.c_double


@pytest.fixture(scope='module',params=('source','official_object'))
def native(request,tmp_path_factory):
    source=NATIVE/'src/acr_tone_kernels.cpp'
    header=NATIVE/'src/acr_tone_kernels.hpp'
    core=source
    if request.param=='official_object':
        core=NATIVE/'build/CMakeFiles/imprint_renderer.dir/src/acr_tone_kernels.cpp.o'
        assert core.is_file() and core.stat().st_mtime_ns>=max(source.stat().st_mtime_ns,header.stat().st_mtime_ns)
    bridge=NATIVE/'tools/acr_mask_controls_bridge.cpp'
    path=tmp_path_factory.mktemp('mask_controls_'+request.param)/('libmask.dylib' if sys.platform=='darwin' else 'libmask.so')
    compiler=shutil.which('clang++') or shutil.which('c++')
    assert compiler
    subprocess.run([compiler,'-std=c++17','-O2','-fPIC','-fno-fast-math','-ffp-contract=off',
        '-dynamiclib' if sys.platform=='darwin' else '-shared',str(core),str(bridge),'-o',str(path)],
        check=True,capture_output=True,text=True)
    lib=C.CDLL(str(path))
    lib.acr_test_mask_sampling.argtypes=[C.c_double,C.c_double,D]
    lib.acr_test_mask_sigma.argtypes=[C.c_double,C.c_double,C.c_double,C.c_int,D]
    lib.acr_test_mask_remap.argtypes=[F,C.c_uint32,C.c_uint32,C.c_double,C.c_double,C.c_double,F]
    for name in ('acr_test_mask_sampling','acr_test_mask_sigma','acr_test_mask_remap'):
        getattr(lib,name).restype=C.c_int
    return lib


def sampling(native,low,high):
    out=np.full(5,SENTINEL,np.float64)
    status=native.acr_test_mask_sampling(low,high,out.ctypes.data_as(D))
    return status,out


def sigma_reference(low,high,epsilon,mode):
    xs=(2.0,3.5,5.0,6.5,12.0,14.0)
    ys=(1.4,1.6,2.3,2.5,2.5,2.0)
    width=high-low
    if width<=xs[0]: linear=ys[0]
    elif width>=xs[-1]: linear=ys[-1]
    else:
        i=next(i for i,x in enumerate(xs) if x>=width)
        linear=LIBM.fma((width-xs[i-1])/(xs[i]-xs[i-1]),ys[i]-ys[i-1],ys[i-1])
    if mode:
        logarithm=LIBM.log(linear)/float.fromhex('0x1.62e42fefa39efp-1')
        linear=LIBM.exp2(LIBM.fma(mode*.5,float.fromhex('0x1.f113baed305dbp-2')-logarithm,logarithm))
    return LIBM.log2(epsilon+linear)


def test_sampling_ceil_float_boundaries(native):
    ranges=[0.,.01,.5,1.,1.5,2.5,39.5,40.]
    for integer in range(1,41):
        x=np.float32(integer)
        ranges.extend([float(np.nextafter(x,np.float32(0))),float(x),
                       float(np.nextafter(x,np.float32(np.inf)))])
    for value in ranges:
        status,out=sampling(native,0.0,value)
        assert status==0
        width=np.float32(value)
        intervals=max(1,math.ceil(float(width)))
        step=np.float32(width/np.float32(intervals))
        inv=np.float32(1.0)/step if step else np.float32(0)
        np.testing.assert_array_equal(out,[0.,float(width),intervals,float(step),float(inv)])
    assert sampling(native,-3.0,-.5)[1][2]==3
    # Inputs first round individually to float, rather than ceil(double span).
    assert sampling(native,0.,math.nextafter(2.,math.inf))[1][2]==2


def test_mask_sigma_knots_and_adjacent_values(native):
    rng=np.random.default_rng(0xdafa20)
    widths=[0.,1.,20.,*rng.uniform(0,20,200)]
    for v in (2.,3.5,5.,6.5,12.,14.):
        widths.extend([math.nextafter(v,0.),v,math.nextafter(v,math.inf)])
    for width in widths:
        for mode in range(3):
            out=C.c_double(SENTINEL)
            assert native.acr_test_mask_sigma(0.,width,2**-12,mode,C.byref(out))==0
            expected=sigma_reference(0.,width,2**-12,mode)
            assert np.float64(out.value).view(np.uint64)==np.float64(expected).view(np.uint64)


def test_invalid_controls_preserve_outputs(native):
    for low,high in [(1,0),(0,41),(float('nan'),1),(0,float('inf')),
                     (1e300,1e300),(1.,math.nextafter(1.,math.inf)),
                     (0.,float(np.nextafter(np.float32(0),np.float32(1))))]:
        status,out=sampling(native,low,high)
        assert status!=0
        np.testing.assert_array_equal(out,np.full(5,SENTINEL))
    for low,high,e,mode in [(1,0,2**-12,0),(0,1,0,0),(0,1,2**-11,0),
                           (0,1,2**-12,-1),(0,1,2**-12,3),(0,float('nan'),2**-12,0)]:
        out=C.c_double(SENTINEL)
        assert native.acr_test_mask_sigma(low,high,e,mode,C.byref(out))!=0
        assert out.value==SENTINEL


def test_fractional_span_remap_saved_instruction_witness(native):
    # Synthetic input retained from 2026-11-08-guide-mask-controls. Expected
    # updated by guide-resample/mask-comparison.json: byte-checked Gaussian,
    # V/H baseline, anchor and residual bodies under a full-bank schedule.
    # Geometry/borders are statically transcribed; no complete Adobe runtime.
    # Keep bits here so no ignored research/binary is needed. Span=2.5 -> 3
    # intervals; count-only truncation produces max error 0.07847214.
    source=np.array([
        [0xc08a5e43,0x3f1c85ac,0xbf275a2c,0xbe7316d8,0xc06b0e68],
        [0xc0869066,0xc0003855,0xc07c395b,0xc007bbde,0xc017e903],
        [0x3f1334c4,0xc0649e0c,0xc08a981c,0xbffe365c,0xc053db1f],
    ],dtype=np.uint32).view(np.float32)
    expected=np.array([
        [0xc0549b1d,0xbf1aedbe,0xbf8478cb,0xbf0eb4e8,0xc04374f1],
        [0xc04ce904,0xc005c21f,0xc0402bb7,0xc001c713,0xc00a1870],
        [0xbf18a66e,0xc033de7e,0xc062a756,0xbff29b30,0xc03b90e0],
    ],dtype=np.uint32).view(np.float32)
    before=source.copy()
    output=np.full_like(source,SENTINEL)
    assert native.acr_test_mask_remap(source.ctypes.data_as(F),5,3,-4.,-1.5,1.4,
                                    output.ctypes.data_as(F))==0
    np.testing.assert_array_equal(output.view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(source.view(np.uint32),before.view(np.uint32))


def test_remap_rejects_before_committing_output(native):
    source=np.linspace(-5,1,15,dtype=np.float32)
    for low,high,sigma in ((-4.,-1.5,0.),(-4.,-1.5,float('nan')),(0.,41.,1.4)):
        output=np.full_like(source,SENTINEL)
        assert native.acr_test_mask_remap(source.ctypes.data_as(F),5,3,low,high,sigma,
                                        output.ctypes.data_as(F))!=0
        np.testing.assert_array_equal(output,np.full_like(output,SENTINEL))
    source[-1]=np.nan
    output=np.full_like(source,SENTINEL)
    assert native.acr_test_mask_remap(source.ctypes.data_as(F),5,3,-4.,-1.5,1.4,
                                    output.ctypes.data_as(F))!=0
    np.testing.assert_array_equal(output,np.full_like(output,SENTINEL))
