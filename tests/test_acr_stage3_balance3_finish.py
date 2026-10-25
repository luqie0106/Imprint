import ctypes as C
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'native-renderer'
P = C.POINTER(C.c_float)
S = C.c_size_t


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests/acr_support' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finish_ref = load('balance3_finish_oracle')
control_ref = load('balance3_controls_oracle')


def ptr(a, index=0):
    return C.cast(int(a.ctypes.data + index * 4), P)


@pytest.fixture(scope='module', params=['source', 'official_object'])
def lib(request, tmp_path_factory):
    names = ['acr_stage3_balance3_finish', 'acr_stage3_balance3_controls']
    sources = [NATIVE / 'src' / (n + '.cpp') for n in names]
    objects = [NATIVE / 'build/CMakeFiles/imprint_renderer.dir/src' / (n + '.cpp.o') for n in names]
    linked = sources
    if request.param == 'official_object':
        for obj, source in zip(objects, sources):
            assert obj.is_file(), 'rebuild renderer: missing official object'
            assert obj.stat().st_mtime_ns >= max(source.stat().st_mtime_ns, source.with_suffix('.hpp').stat().st_mtime_ns), 'rebuild renderer: stale official object'
        linked = objects
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    out = tmp_path_factory.mktemp('balance_finish_' + request.param) / ('libfinish' + suffix)
    command = [shutil.which('clang++') or shutil.which('c++'), '-std=c++17', '-O2', '-fPIC', '-fno-fast-math', '-ffp-contract=off',
               '-dynamiclib' if sys.platform == 'darwin' else '-shared', '-I', str(NATIVE / 'src'),
               *(str(NATIVE / 'tools' / (n + '_bridge.cpp')) for n in names), *(str(p) for p in linked), '-o', str(out)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = C.CDLL(str(out))
    result.acr_test_balance3_finish.argtypes = [C.c_int, P, S, P, S, S, C.c_int, C.c_int, P, P, S]
    result.acr_test_balance3_finish.restype = C.c_int
    result.acr_test_balance3_controls.argtypes = [C.c_double, C.c_double, C.c_int, P]
    result.acr_test_balance3_controls.restype = C.c_int
    return result


def call(lib, source, stride, diff, width, phase, even, limits, dst, branch=1, center=None, **counts):
    center = stride + 4 if center is None else center
    return lib.acr_test_balance3_finish(branch, ptr(source, center), stride, ptr(diff),
        counts.get('diff_count', diff.size), width, phase, even, ptr(limits), ptr(dst), counts.get('dst_count', dst.size))


def equal(a, b):
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize('branch', [0, 1])
@pytest.mark.parametrize('phase,even', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_hardcoded_lane_selection_and_preservation(lib, branch, phase, even):
    src = np.full(36, .25, dtype=np.float32)
    diff = np.full(8, .5, dtype=np.float32)
    dst = np.full(8, -99.25, dtype=np.float32)
    limits = np.asarray([-.5, .5, -.5, .5], dtype=np.float32)
    assert call(lib, src, 12, diff, 4, phase, even, limits, dst, branch) == 0
    corrected = 0. if even else .5
    if branch == 1:
        expected = [corrected, .25, corrected, .25] if phase == 0 else [.25, corrected, .25, corrected]
    else:
        expected = [-99.25, corrected, -99.25, corrected] if phase == 0 else [corrected, -99.25, corrected, -99.25]
    equal(dst[:4], np.asarray(expected, dtype=np.float32))
    assert np.all(dst[4:] == np.float32(-99.25))


@pytest.mark.parametrize('branch', [0, 1])
def test_random_rows_match_independent_instruction_model(lib, branch):
    rng = np.random.default_rng(2026101908)
    for width in (4, 12, 32):
        stride = width + 12
        for phase in (0, 1):
            for even in (0, 1):
                source = rng.uniform(-.7, 1.5, 3 * stride).astype(np.float32)
                diff = rng.uniform(-1, 1, width + 4).astype(np.float32)
                dst = rng.uniform(-3, 2, width + 8).astype(np.float32)
                limits = np.asarray([-.13, .19, -.07, .17], dtype=np.float32)
                original_source, original_diff = source.copy(), diff.copy()
                expected = finish_ref.reference_row(source, stride + 4, stride, diff, width, phase, even, limits,
                    branch='optimized' if branch else 'default', destination=dst)
                assert call(lib, source, stride, diff, width, phase, even, limits, dst, branch) == 0
                equal(dst, expected)
                equal(source, original_source)
                equal(diff, original_diff)


def test_zero_signs_and_both_explicit_fmas(lib):
    src = np.full(36, -0., dtype=np.float32)
    diff = np.zeros(8, dtype=np.float32)
    limits = np.full(4, -0., dtype=np.float32)
    for branch, phase, selected_bits in [(0, 1, 0x80000000), (1, 0, 0)]:
        dst = np.full(8, 7., dtype=np.float32)
        assert call(lib, src, 12, diff, 4, phase, 0, limits, dst, branch) == 0
        assert dst.view(np.uint32)[0] == selected_bits
    tiny = np.nextafter(np.float32(0), np.float32(1))
    limits[:] = [-1, 1, -1, 1]
    for second in (False, True):
        src[:] = 0
        diff[:] = 0
        src[16 + 12 + 1] = 8 * tiny
        if second:
            diff[0] = tiny
        else:
            src[16] = tiny
        dst = np.full(8, 7., dtype=np.float32)
        assert call(lib, src, 12, diff, 4, 1 if second else 0, 0, limits, dst, 0 if second else 1) == 0
        assert dst.view(np.uint32)[0] == 2


def test_capacity_nonfinite_and_last_guard_rejection_is_atomic(lib):
    src = np.full(36, .25, dtype=np.float32)
    diff = np.zeros(8, dtype=np.float32)
    limits = np.asarray([-.5, .5, -.5, .5], dtype=np.float32)
    for branch in (0, 1):
        dst = np.full(8, 31., dtype=np.float32)
        before = dst.copy()
        assert call(lib, src, 12, diff, 4, 0, 0, limits, dst, branch, diff_count=3) == 6
        equal(dst, before)
        src[16 + 12 + 4] = np.nan  # last south-right read, even if its lane is copied
        assert call(lib, src, 12, diff, 4, 0, 0, limits, dst, branch) == 9
        equal(dst, before)
        src[32] = .25
        assert call(lib, src, 12, diff, 4, 0, 0, limits, src[32:], branch) == 8


@pytest.mark.parametrize('direct', [False, True])
def test_controls_match_independent_double_sequence(lib, direct):
    rng = np.random.default_rng(5120)
    cases = [(1, 1), (1 / 1.07, 1.07), (-0., 1), (0., 0.), (-1., 1.), (2., 1.), (1e300, 1e300)]
    cases += [tuple(sorted(v)) for v in rng.uniform(.02, 5, (120, 2))]
    for lower, upper in cases:
        expected_status, expected = control_ref.reference(lower, upper, direct)
        actual = np.full(4, -99.25, dtype=np.float32)
        assert lib.acr_test_balance3_controls(lower, upper, direct, ptr(actual)) == expected_status
        if expected_status == 0:
            equal(actual, expected)
        else:
            assert np.all(actual == np.float32(-99.25))


def test_controls_keep_near_one_rounding_residual(lib):
    actual = np.full(4, 3., dtype=np.float32)
    assert lib.acr_test_balance3_controls(1, 1, 0, ptr(actual)) == 0
    assert actual.view(np.uint32).tolist() == [0, 0x24800000, 0xA4800000, 0]
