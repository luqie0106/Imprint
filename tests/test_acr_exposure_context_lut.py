"""Independent raw-transcription tests for explicitly prepared Exposure curves."""
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
SOURCE = NATIVE / 'src/acr_exposure_context_lut.cpp'
HEADER = SOURCE.with_suffix('.hpp')
API = NATIVE / 'src/acr_exposure_context_lut_api.cpp'
BRIDGE = NATIVE / 'tools/acr_exposure_context_lut_bridge.cpp'
OBJECT = NATIVE / 'build/CMakeFiles/imprint_renderer.dir/src/acr_exposure_context_lut.cpp.o'
API_OBJECT = OBJECT.with_name('acr_exposure_context_lut_api.cpp.o')
sys.path.insert(0, str(ROOT / 'tests/acr_support'))
sys.path.insert(0, str(ROOT / 'src'))
import exposure_context_lut_oracle as oracle
import native_renderer as native
F = C.POINTER(C.c_float)
D = C.POINTER(C.c_double)
S = C.c_size_t
SENTINEL = np.float32(-1234.5)


def ptr(array, kind=F):
    return None if array is None else array.ctypes.data_as(kind)


def assert_bits(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual, np.float32).view(np.uint32),
                                  np.asarray(expected, np.float32).view(np.uint32))


@pytest.fixture(scope='module', params=['source', 'object'])
def bridge(request, tmp_path_factory):
    compiler = shutil.which('clang++') or shutil.which('c++')
    assert compiler
    inputs = [SOURCE, API]
    if request.param == 'object':
        for obj, source in ((OBJECT, SOURCE), (API_OBJECT, API)):
            assert obj.is_file(), f'fresh official build required: {obj}'
            assert obj.stat().st_mtime_ns >= max(source.stat().st_mtime_ns, HEADER.stat().st_mtime_ns)
        inputs = [OBJECT, API_OBJECT]
    output = tmp_path_factory.mktemp('exposure-context') / 'bridge.dylib'
    subprocess.run([compiler, '-std=c++17', '-O2', '-fPIC', '-fno-fast-math',
                    '-ffp-contract=off', '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                    '-I', str(NATIVE / 'src'), '-I', str(NATIVE / 'include'), str(BRIDGE),
                    *map(str, inputs), '-o', str(output)], check=True, capture_output=True, text=True)
    lib = C.CDLL(str(output))
    signatures = {
        'acr_test_evaluate_exposure_channel': [C.c_double] * 4 + [D],
        'acr_test_evaluate_exposure_post': [C.c_double] * 2 + [D],
        'acr_test_build_exposure_channel': [C.c_double] * 3 + [F, S],
        'acr_test_build_exposure_post': [C.c_double, F, S],
        'im_native_acr_prepare_exposure_context_lut': [C.c_uint32, D, S, F, S],
    }
    for name, args in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, C.c_int
    return lib


# Frozen before the native implementation existed. Values follow the saved raw
# operations and host libm; these are scalar double bits, not float LUT tolerance.
CHANNEL_PROBES = [
    (.1, (0, 0, 0), 0x3fb999999999999b),
    (.3, (0, 0, 0), 0x3fd3333333333332),
    (.1, (0, .2, .25), 0x3f8687c0789c3ee6),
    (.3, (0, .2, .25), 0x3fc4843e83b495e4),
    (.65, (.1, .7, .5), 0x3fb710dcdaffdd1c),
    (.3, (.25, .75, 0), 0x3f50cf713e30374a),
    (.65, (.1, .1, .5), 0x3fe22c3f35ba781a),
]
POST_PROBES = [
    (-0., 0., 0x8000000000000000),
    (-0., 1., 0x8000000000000000),
    (.0031308, .25, 0x3f779c6c06a4a1d4),
    (.18, .5, 0x3fcd26ba512da8d4),
    (.75, 1, 0x3fe85e0a72f05397),
    (math.nextafter(0, 1), 1, 0xd),
    (math.nextafter(0, 1), .5, 0x4),
    (.18, 0, 0x3fc70a3d70a3d70a),
    (.18, math.nextafter(0, 1), 0x3fc70a3d70a3d70a),
]


@pytest.mark.parametrize('x,controls,expected', CHANNEL_PROBES)
def test_original_channel_scalar_probes(bridge, x, controls, expected):
    output = C.c_double(-123)
    assert oracle.bits(oracle.channel(x, *controls)) == expected
    assert bridge.acr_test_evaluate_exposure_channel(x, *controls, C.byref(output)) == 0
    assert oracle.bits(output.value) == expected


@pytest.mark.parametrize('x,p,expected', POST_PROBES)
def test_original_post_scalar_probes(bridge, x, p, expected):
    output = C.c_double(-123)
    assert oracle.bits(oracle.post(x, p)) == expected
    assert bridge.acr_test_evaluate_exposure_post(x, p, C.byref(output)) == 0
    assert oracle.bits(output.value) == expected


def test_stable_root_order_and_degeneracy():
    assert oracle.quadratic(0, 0, 1) == []
    assert oracle.quadratic(0, 2, -1) == [.5]
    assert oracle.quadratic(1, 0, 1) == []
    assert oracle.quadratic(1, -1, 0) == [1, 0]
    assert oracle.quadratic(1, 1, 0) == [-1, -0.0]
    # Positive and negative B preserve q/A then C/q, including two valid roots.
    assert oracle.quadratic(1, -1.5, .5) == [1, .5]
    assert oracle.quadratic(-1, 1.5, -.5) == [1, .5]


def test_channel_inclusive_knee_bounds_and_random_scalars(bridge):
    rng = np.random.default_rng(22819)
    cases = [(0., .5, .5), (0., .5, 0.), (0., 0., .5)]
    cases += [tuple(row) for row in rng.random((18, 3)) * [1, 1, .5]]
    for a, b, h in cases:
        state = oracle.channel_state(a, b, h)
        # a=0 cases make transformed thresholds exactly accessible in x.
        xs = list(rng.random(12)) + [0., 1.]
        if a == 0:
            for boundary in [state[4], state[7]]:
                xs += [max(0, math.nextafter(boundary, -math.inf)), boundary,
                       min(1, math.nextafter(boundary, math.inf))]
        for x in xs:
            out = C.c_double()
            assert bridge.acr_test_evaluate_exposure_channel(x, a, b, h, C.byref(out)) == 0
            assert oracle.bits(out.value) == oracle.bits(oracle.channel(x, a, b, h))


def test_post_encode_decode_and_control_boundaries(bridge):
    rng = np.random.default_rng(22820)
    threshold = oracle.constant(0x3f69a5c37387b719)
    xs = [0., math.nextafter(0, 1), math.nextafter(threshold, 0), threshold,
          math.nextafter(threshold, 1), .25, 1.] + list(rng.random(40))
    for p in [0., math.nextafter(0, 1), .25, .5, math.nextafter(.5, 1), 1.]:
        for x in xs:
            out = C.c_double()
            assert bridge.acr_test_evaluate_exposure_post(x, p, C.byref(out)) == 0
            assert oracle.bits(out.value) == oracle.bits(oracle.post(x, p))


@pytest.mark.parametrize('kind,controls', [
    (1, (0., 0., 0.)), (1, (.1, .7, .5)), (1, (.25, .75, 0.)),
    (1, (.9999999999999999, .9999999999999999, .5)),
    (1, (.031, .073, .126)), (1, (.61, .22, .42)),
    (2, (0.,)), (2, (math.nextafter(0, 1),)), (2, (.25,)), (2, (.5,)), (2, (1.,)),
])
def test_all_4098_samples_guard_and_padding(bridge, kind, controls):
    expected = oracle.channel_lut(*controls) if kind == 1 else oracle.post_lut(*controls)
    output = np.full(4103, SENTINEL, np.float32)
    fn = bridge.acr_test_build_exposure_channel if kind == 1 else bridge.acr_test_build_exposure_post
    assert fn(*controls, ptr(output), output.size) == 0
    assert_bits(output[:4098], expected)
    assert_bits(output[4098:], np.full(5, SENTINEL, np.float32))
    assert output.view(np.uint32)[4096] == output.view(np.uint32)[4097]
    raw = np.asarray(controls, np.float64)
    output.fill(SENTINEL)
    assert bridge.im_native_acr_prepare_exposure_context_lut(kind, ptr(raw, D), raw.size,
                                                           ptr(output), output.size) == 0
    assert_bits(output[:4098], expected)
    assert_bits(output[4098:], np.full(5, SENTINEL, np.float32))


@pytest.mark.parametrize('kind,valid', [(1, (.1, .2, .25)), (2, (.5,))])
def test_invalid_inputs_are_atomic(bridge, kind, valid):
    build = bridge.acr_test_build_exposure_channel if kind == 1 else bridge.acr_test_build_exposure_post
    evaluate = bridge.acr_test_evaluate_exposure_channel if kind == 1 else bridge.acr_test_evaluate_exposure_post
    output = np.full(4100, SENTINEL, np.float32)
    before = output.copy()
    bad_cases = []
    for index in range(len(valid)):
        limit = .5 if kind == 1 and index == 2 else 1.
        upper = limit if kind == 1 and index < 2 else math.nextafter(limit, math.inf)
        for bad in [-math.nextafter(0, 1), upper, math.nan, math.inf, -math.inf]:
            args = list(valid)
            args[index] = bad
            bad_cases.append(args)
    for args in bad_cases:
        assert build(*args, ptr(output), output.size) == 1
        assert_bits(output, before)
        raw = np.asarray(args, np.float64)
        assert bridge.im_native_acr_prepare_exposure_context_lut(kind, ptr(raw, D), len(args),
                                                               ptr(output), output.size) == 1
        assert_bits(output, before)
        scalar = C.c_double(-123.)
        assert evaluate(.5, *args, C.byref(scalar)) == 1
        assert scalar.value == -123.
    for x in [-math.nextafter(0, 1), math.nextafter(1, math.inf), math.nan, math.inf]:
        scalar = C.c_double(-123.)
        assert evaluate(x, *valid, C.byref(scalar)) == 1
        assert scalar.value == -123.
    assert evaluate(.5, *valid, None) == 1
    assert build(*valid, None, 4098) == 1
    assert build(*valid, ptr(output), 4097) == 1
    assert_bits(output, before)


def test_formal_capacity_null_alignment_overlap_and_atomicity(bridge):
    fn = bridge.im_native_acr_prepare_exposure_context_lut
    output = np.full(4104, SENTINEL, np.float32)
    before = output.copy()
    controls = np.array([.1, .2, .25], np.float64)
    for kind, cp, count, op, capacity, expected in [
        (0, ptr(controls, D), 3, ptr(output), 4098, 1),
        (3, ptr(controls, D), 3, ptr(output), 4098, 1),
        (1, None, 3, ptr(output), 4098, 1),
        (1, ptr(controls, D), 2, ptr(output), 4098, 1),
        (1, ptr(controls, D), 4, ptr(output), 4098, 1),
        (2, ptr(controls, D), 3, ptr(output), 4098, 1),
        (1, ptr(controls, D), 3, None, 4098, 1),
        (1, ptr(controls, D), 3, ptr(output), 4097, 4),
        (1, C.cast(controls.ctypes.data + 1, D), 3, ptr(output), 4098, 1),
        (1, ptr(controls, D), 3, C.cast(output.ctypes.data + 1, F), 4098, 1),
        (1, ptr(controls, D), 3, ptr(output), S(-1).value, 1),
    ]:
        assert fn(kind, cp, count, op, capacity) == expected
        assert_bits(output, before)
    # Controls alias even output padding: the entire declared span is reserved.
    storage = np.full(4110, SENTINEL, np.float32)
    for offset in [0, 4098 * 4]:
        aliased = C.cast(storage.ctypes.data + offset, D)
        for i, value in enumerate(controls):
            aliased[i] = value
        snapshot = storage.copy()
        assert fn(1, aliased, 3, ptr(storage), storage.size) == 1
        assert_bits(storage, snapshot)


@pytest.mark.parametrize('controls', [(0., 0., 0.), (.1, .7, .5), (.31, .07, .126)])
def test_python_channel_adapter(controls):
    result = native.native_acr_prepare_exposure_channel_lut(*controls)
    assert result.dtype == np.float32 and result.shape == (4098,) and result.flags.c_contiguous
    assert_bits(result, oracle.channel_lut(*controls))


@pytest.mark.parametrize('p', [0., math.nextafter(0, 1), .25, .5, 1.])
def test_python_post_adapter(p):
    result = native.native_acr_prepare_exposure_post_lut(p)
    assert result.dtype == np.float32 and result.shape == (4098,) and result.flags.c_contiguous
    assert_bits(result, oracle.post_lut(p))


def test_python_adapters_reject_before_loading(monkeypatch):
    monkeypatch.setattr(native, '_get_library', lambda: pytest.fail('invalid input loaded library'))
    for controls in [(math.nan, 0, 0), (0, math.inf, 0), ([0], 0, 0)]:
        with pytest.raises((ValueError, TypeError)):
            native.native_acr_prepare_exposure_channel_lut(*controls)
    for p in [math.nan, math.inf, [.5]]:
        with pytest.raises((ValueError, TypeError)):
            native.native_acr_prepare_exposure_post_lut(p)


def test_python_adapters_reject_invalid_prepared_domains():
    for controls in [(1, 0, 0), (0, 1, 0), (0, 0, .50001), (-1, 0, 0)]:
        with pytest.raises(ValueError):
            native.native_acr_prepare_exposure_channel_lut(*controls)
    for p in [-.01, 1.01]:
        with pytest.raises(ValueError):
            native.native_acr_prepare_exposure_post_lut(p)
