"""Prepared q/cap parity from independent ARM64 scalar instructions."""
from __future__ import annotations

import ctypes as C
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'native-renderer'
sys.path.insert(0, str(ROOT / 'tests/acr_support'))
import exposure_qcap_oracle as oracle

PTR = C.POINTER(C.c_double)
SENTINEL = -1234.5
DEFAULT = (0., 0., 0., 0., -8., -1., -2., 0.)
MODULES = ('acr_exposure_qcap', 'acr_negative_curve', 'acr_whites', 'acr_exposure_qcap_api')


def ptr(x):
    return x.ctypes.data_as(PTR) if x is not None else C.cast(C.c_void_p(), PTR)


def bits(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


def equal_bits(actual, expected):
    np.testing.assert_array_equal(bits(actual), bits(expected))


def context(**changes):
    names = ('e98', 'a0', 'b0', 'white80', 'lo', 'hi', 'a', 'high')
    return tuple(changes.get(name, value) for name, value in zip(names, DEFAULT))


def compile_bridge(kind, output):
    compiler = shutil.which('clang++') or shutil.which('c++')
    assert compiler, 'C++ compiler required'
    bridge = NATIVE / 'tools/acr_exposure_qcap_bridge.cpp'
    assert bridge.is_file(), 'q/cap bridge has not been provided'
    inputs = []
    for name in MODULES:
        source = NATIVE / f'src/{name}.cpp'
        assert source.is_file(), f'missing source {source.name}'
        if kind == 'source':
            inputs.append(source)
        else:
            obj = NATIVE / f'build/CMakeFiles/imprint_renderer.dir/src/{name}.cpp.o'
            assert obj.is_file(), f'missing formal object {obj.name}'
            assert obj.stat().st_mtime_ns >= source.stat().st_mtime_ns, f'stale {obj.name}'
            inputs.append(obj)
    command = [compiler, '-std=c++17', '-O2', '-fPIC', '-fno-fast-math', '-ffp-contract=off',
               '-dynamiclib' if sys.platform == 'darwin' else '-shared',
               '-I', str(NATIVE/'src'), '-I', str(NATIVE/'include'), str(bridge),
               *map(str, inputs), '-o', str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)


class Bridge:
    def __init__(self, path, formal=False):
        self.library = C.CDLL(str(path))
        self.prepare_fn = getattr(self.library, 'im_native_acr_prepare_exposure_qcap' if formal
                                  else 'acr_test_prepare_exposure_qcap')
        self.prepare_fn.argtypes = [PTR, C.c_size_t, PTR, C.c_size_t]
        self.prepare_fn.restype = C.c_int
        if not formal:
            self.response_fn = self.library.acr_test_exposure_context_response
            self.response_fn.argtypes = [PTR, C.c_size_t, C.c_double, PTR, C.c_size_t]
            self.response_fn.restype = C.c_int

    def prepare(self, controls, count=None, capacity=3):
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        before = controls.copy()
        out = np.full(3, SENTINEL, dtype=np.float64)
        status = self.prepare_fn(ptr(controls), controls.size if count is None else count, ptr(out), capacity)
        equal_bits(controls, before)
        return status, out

    def response(self, controls, value):
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        out = np.array([SENTINEL], dtype=np.float64)
        status = self.response_fn(ptr(controls), 8, value, ptr(out), 1)
        return status, out[0]


@pytest.fixture(scope='module', params=('source', 'official_object'))
def bridge(request, tmp_path_factory):
    path = tmp_path_factory.mktemp('qcap_'+request.param)/('libqcap.dylib' if sys.platform == 'darwin' else 'libqcap.so')
    compile_bridge(request.param, path)
    return Bridge(path)


@pytest.fixture(scope='module')
def formal():
    path = NATIVE/'build'/('libimprint_renderer.0.1.0.dylib' if sys.platform == 'darwin' else 'libimprint_renderer.0.1.0.so')
    assert path.is_file(), 'formal native renderer has not been built'
    for name in MODULES:
        assert path.stat().st_mtime_ns >= (NATIVE/f'src/{name}.cpp').stat().st_mtime_ns
    return Bridge(path, formal=True)


CASES = [context(a0=ev, b0=b, white80=w)
         for ev in (0., -0., .5, 2., 8., 9., 120., -.5, -2., -8., -120.)
         for b, w in ((0., 0.), (1.25, 0.), (-1., .7))]
CASES += [context(a=a, high=hi, e98=e) for a, hi, e in
          ((-2., 8., 0.), (-2., 2., 0.), (-2., math.nextafter(2., -math.inf), 0.),
           (-2., math.nextafter(2., math.inf), 0.), (-120., -120., 0.),
           (118., 130., -120.), (-20., -20., 0.), (0., 0., 0.))]


@pytest.mark.parametrize('ctx', CASES)
def test_source_and_object_match_original_scalar(bridge, ctx):
    status, out = bridge.prepare(ctx)
    assert status == 0
    equal_bits(out, oracle.prepare(ctx))


@pytest.mark.parametrize('ev', (0., .25, 2., 8., 9., 120., -.25, -2., -8., -120.))
@pytest.mark.parametrize('value', (0., -0., 2**-20, .18, .5, 1., 2.))
def test_direct_response(bridge, ev, value):
    ctx = context(a0=ev)
    status, out = bridge.response(ctx, value)
    assert status == 0
    equal_bits(out, oracle.response(ctx, value))


def test_four_segments_naturally_saturate(bridge):
    for ev in (8., 8.000000000000002, 9., 120.):
        status, out = bridge.prepare(context(a0=ev))
        assert status == 0
        equal_bits(out, oracle.prepare(context(a0=8.)))


def test_q_limits_and_v_switch(bridge):
    for ctx, q in ((context(a=-20., high=-20.), .0625), (context(a=0., high=0.), oracle.QMAX)):
        status, out = bridge.prepare(ctx)
        assert status == 0
        assert out[2] == q
    for hi in (math.nextafter(2., -math.inf), 2., math.nextafter(2., math.inf)):
        status, out = bridge.prepare(context(a=-2., high=hi))
        assert status == 0
        equal_bits(out, oracle.prepare(context(a=-2., high=hi)))


def test_negative_whites_double_square_counterexample(bridge):
    ctx = context(b0=-1., white80=.7, a=-3., high=-3.)
    c = oracle.negative_clip(ctx)
    double_square = c*c
    float_square = float(np.float32(np.float32(c)*np.float32(c)))
    assert bits(double_square) != bits(float_square)
    y = oracle.wrapper(math.exp2(-3.), 0.)
    a = y/(y+1.)
    wrong = min(a*(y/float_square+1.), 1.)
    correct = oracle.response(ctx, math.exp2(-3.))
    assert bits(correct) == np.uint64(0x3FC18DE06193D9A9)
    assert bits(correct) != bits(wrong), 'fixture must detect using the float clip square'
    status, out = bridge.prepare(ctx)
    assert status == 0
    equal_bits(out[1], correct)


def test_positive_clip_squared_underflow_keeps_original_fminnm(bridge):
    ctx = context(b0=-1., white80=2., lo=-600., hi=-600., a=-3., high=-3.)
    c = oracle.negative_clip(ctx)
    assert c > 0 and c*c == 0
    for value in (0., .125):
        expected = oracle.response(ctx, value)
        assert expected == 1.
        status, out = bridge.response(ctx, value)
        assert status == 0
        equal_bits(out, expected)


def test_direct_response_failure_is_atomic(bridge):
    for value in (-1., math.nan, math.inf, -math.inf):
        status, out = bridge.response(DEFAULT, value)
        assert status != 0
        equal_bits(out, SENTINEL)


def test_random_finite_exact_uint64(bridge):
    rng = np.random.default_rng(9226)
    for _ in range(300):
        ctx = (rng.uniform(-5, 3), rng.uniform(-12, 12), rng.uniform(-2, 2),
               rng.uniform(-2, 2), rng.uniform(-12, -6), rng.uniform(-3, 0),
               rng.uniform(-8, 0), rng.uniform(-4, 4))
        status, out = bridge.prepare(ctx)
        assert status == 0
        equal_bits(out, oracle.prepare(ctx))


INVALID = [context(**{key: value}) for key in ('e98', 'a0', 'b0', 'white80', 'lo', 'hi', 'a', 'high')
           for value in (math.nan, math.inf, -math.inf)]
INVALID += [context(e98=120.01), context(a0=-120.01), context(b0=2.01), context(white80=-2.01),
            context(a=121., high=121.), context(a=-121., high=-121.),
            context(lo=-2000., hi=-2000., b0=-1., white80=1.)]


@pytest.mark.parametrize('ctx', INVALID)
def test_failure_leaves_output_unchanged(bridge, ctx):
    status, out = bridge.prepare(ctx)
    assert status != 0
    equal_bits(out, [SENTINEL]*3)


def test_buffer_failures_unchanged(bridge):
    for count, capacity in ((7, 3), (9, 3), (8, 2), (8, 0)):
        status, out = bridge.prepare(DEFAULT, count=count, capacity=capacity)
        assert status != 0
        equal_bits(out, [SENTINEL]*3)
    out = np.full(3, SENTINEL)
    assert bridge.prepare_fn(ptr(None), 8, ptr(out), 3) != 0
    equal_bits(out, [SENTINEL]*3)
    controls = np.array(DEFAULT)
    assert bridge.prepare_fn(ptr(controls), 8, ptr(None), 3) != 0


def test_formal_public_api(formal):
    for ctx in CASES:
        status, out = formal.prepare(ctx)
        assert status == 0
        equal_bits(out, oracle.prepare(ctx))
    for ctx in INVALID:
        status, out = formal.prepare(ctx)
        assert status != 0
        equal_bits(out, [SENTINEL]*3)


def test_python_adapter_matches_oracle():
    sys.path.insert(0, str(ROOT/'src'))
    import native_renderer
    for ctx in CASES:
        out = native_renderer.native_acr_prepare_exposure_qcap(ctx)
        assert out.dtype == np.float64 and out.shape == (3,)
        equal_bits(out, oracle.prepare(ctx))
    for ctx in INVALID:
        with pytest.raises((ValueError, RuntimeError)):
            native_renderer.native_acr_prepare_exposure_qcap(ctx)


def _check_public_buffer_contract(fn):
    controls = np.asarray(DEFAULT, np.float64)
    output = np.full(12, SENTINEL, np.float64)
    before = output.copy()
    for cp, count, op, capacity in (
        (C.cast(controls.ctypes.data+1, PTR), 8, ptr(output), 12),
        (ptr(controls), 8, C.cast(output.ctypes.data+1, PTR), 12),
        (ptr(controls), 8, ptr(output), C.c_size_t(-1).value),
        (ptr(controls), 8, ptr(output), (1 << (8*C.sizeof(C.c_size_t)-1))),
    ):
        assert fn(cp, count, op, capacity) != 0
        equal_bits(output, before)
    # First three outputs do not reach the controls, but declared padding does.
    for offset in (0, 3):
        storage = np.full(20, SENTINEL, np.float64)
        storage[offset:offset+8] = controls
        snapshot = storage.copy()
        cp = C.cast(storage.ctypes.data+offset*8, PTR)
        assert fn(cp, 8, ptr(storage), 12) != 0
        equal_bits(storage, snapshot)
    assert fn(ptr(controls), 8, ptr(output), output.size) == 0
    equal_bits(output[:3], oracle.prepare(DEFAULT))
    equal_bits(output[3:], before[3:])


def test_bridge_full_span_alignment_overflow_padding(bridge):
    _check_public_buffer_contract(bridge.prepare_fn)


def test_formal_full_span_alignment_overflow_padding(formal):
    _check_public_buffer_contract(formal.prepare_fn)


def test_direct_response_overflow_is_recovered_by_clip(bridge):
    ctx = context(e98=120.)
    status, output = bridge.response(ctx, np.finfo(np.float64).max)
    assert status == 0
    equal_bits(output, oracle.response(ctx, np.finfo(np.float64).max))


def test_v_preserves_negative_zero_statistic(bridge):
    ctx = context(a=-0., high=-math.nextafter(0., 1.))
    status, out = bridge.prepare(ctx)
    assert status == 0
    equal_bits(out, oracle.prepare(ctx))
    assert bits(out[0]) == np.uint64(0x8000000000000000)
