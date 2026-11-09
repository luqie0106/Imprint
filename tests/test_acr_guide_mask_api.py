"""Shipped prepared-mask ABI: both boundary modes, diagnostics and atomicity."""
import ctypes as C
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/acr_support")]
import native_renderer as native
from acr_support import guide_mask_oracle as oracle

FP, DP = C.POINTER(C.c_float), C.POINTER(C.c_double)


@pytest.fixture(scope="module")
def function():
    lib, _ = native._get_library()
    fn = lib.im_native_acr_guide_mask_prepared_f32
    fn.argtypes = [C.c_uint32, C.c_uint32, FP, C.c_size_t, C.c_double,
                   C.c_uint32, C.c_uint32, FP, C.c_size_t, DP, C.c_size_t]
    fn.restype = C.c_int
    return fn


@pytest.mark.parametrize("shape", [(1,3),(1,17),(3,5),(3,6),(3,257)])
@pytest.mark.parametrize("boundary", [2,4])
def test_formal_api_matches_full_bank_and_second_pass_stats(shape, boundary):
    log = np.random.default_rng(sum(shape)).uniform(-12,-.1,shape).astype(np.float32)
    before = log.copy()
    for mode, epsilon in ((0,2**-12),(1,3.214e-6),(2,2**-12)):
        expected, stats, sigma, level = oracle.reference(log,epsilon,mode,boundary)
        delta, diagnostics = native.native_acr_guide_mask_prepared(
            log, epsilon, sigma_mode=mode, boundary_mode=boundary,
        )
        np.testing.assert_array_equal(delta.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(diagnostics[:18].view(np.uint64), stats.view(np.uint64))
        assert diagnostics[18] == sigma and diagnostics[19] == level
        assert delta.shape == log.shape and delta.dtype == np.float32
    np.testing.assert_array_equal(log.view(np.uint32), before.view(np.uint32))


def test_raw_abi_failure_keeps_both_outputs(function):
    source = np.linspace(-8,-1,15,dtype=np.float32)
    output = np.full(15,-9123.25,np.float32)
    diagnostics = np.full(20,-1234.5,np.float64)
    p, q = source.ctypes.data_as(FP), output.ctypes.data_as(FP)
    d = diagnostics.ctypes.data_as(DP)
    original = source.tobytes(), output.tobytes(), diagnostics.tobytes()
    defaults = dict(w=5,h=3,p=p,n=15,e=2**-12,s=0,b=4,q=q,m=15,d=d,k=20)

    def call(**patch):
        v = dict(defaults,**patch)
        return function(v['w'],v['h'],v['p'],v['n'],v['e'],v['s'],v['b'],v['q'],v['m'],v['d'],v['k'])

    for patch in ({'w':0},{'h':0},{'w':65536},{'w':1,'h':1,'n':1,'m':1},
                  {'n':14},{'m':14},{'k':19},{'s':3},{'b':3},
                  {'e':0.},{'e':float('nan')},{'e':float('inf')},
                  {'e':1e-300},{'e':2**-11},{'p':None},{'q':None},{'d':None},
                  {'p':C.cast(source.ctypes.data+1,FP)},
                  {'q':C.cast(output.ctypes.data+1,FP)},
                  {'d':C.cast(diagnostics.ctypes.data+1,DP)},
                  {'p':C.cast(C.c_size_t(-4).value,FP)},
                  {'q':p},{'d':C.cast(p,DP)},{'d':C.cast(q,DP)}):
        assert call(**patch) == 1, patch
        assert (source.tobytes(),output.tobytes(),diagnostics.tobytes()) == original
    source[-1] = np.nan
    assert call() == 1
    assert output.tobytes() == original[1] and diagnostics.tobytes() == original[2]


def test_missing_symbol_has_no_python_approximation(monkeypatch):
    monkeypatch.setattr(native,"_get_library",lambda:(object(),"missing"))
    with pytest.raises(native.NativeRendererError):
        native.native_acr_guide_mask_prepared(np.zeros((2,3),np.float32),2**-12,
                                              sigma_mode=0,boundary_mode=4)


@pytest.mark.parametrize("patch", [
    {'log_guide':np.zeros((2,3),np.float64)}, {'log_guide':np.zeros((2,3,1),np.float32)},
    {'log_guide':np.zeros((1,1),np.float32)}, {'log_guide':np.zeros((0,3),np.float32)},
    {'log_guide':np.full((2,3),np.nan,np.float32)},
    {'epsilon':True},{'epsilon':0.},{'epsilon':float('nan')},{'epsilon':float('inf')},
    {'epsilon':10**1000},{'epsilon':1e-300},{'epsilon':2**-11},
    {'sigma_mode':True},{'sigma_mode':1.0},{'sigma_mode':3},
    {'boundary_mode':False},{'boundary_mode':4.0},{'boundary_mode':0},{'boundary_mode':5},
])
def test_invalid_wrapper_input_before_library_load(monkeypatch, patch):
    monkeypatch.setattr(native,"_get_library",lambda:pytest.fail("invalid input loaded library"))
    kwargs = dict(log_guide=np.zeros((2,3),np.float32),epsilon=2**-12,sigma_mode=0,boundary_mode=4)
    kwargs.update(patch)
    with pytest.raises(ValueError):
        native.native_acr_guide_mask_prepared(**kwargs)
