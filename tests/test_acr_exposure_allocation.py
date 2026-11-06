"""Stateful finite modern allocation; independent hard sequence assertions."""
import ctypes as C
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARRAY = C.c_double * 13


@pytest.fixture(scope='module')
def bridge(tmp_path_factory):
    compiler = shutil.which('clang++') or shutil.which('c++')
    assert compiler
    suffix = '.dylib' if sys.platform == 'darwin' else '.dll' if sys.platform == 'win32' else '.so'
    flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
    output = tmp_path_factory.mktemp('allocation') / ('bridge' + suffix)
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
        '-fno-fast-math', '-ffp-contract=off', *flags,
        str(ROOT/'native-renderer/tools/acr_exposure_allocation_bridge.cpp'),
        str(ROOT/'native-renderer/src/acr_exposure_allocation.cpp'),
        '-o', str(output)], check=True, capture_output=True)
    fn = C.CDLL(str(output)).acr_exposure_allocation_bridge
    fn.argtypes = [C.POINTER(C.c_double), C.c_double, C.c_double, C.c_uint, C.POINTER(C.c_double)]
    fn.restype = C.c_int
    return fn


def state(amount=0.):
    return ARRAY(-3., 0., 0., -.5, amount, 1., 0., 0., 0., 0., 0., 0., 0.)


def test_repeated_calls_consume_source_ev_and_accumulate_curve(bridge):
    value = state()
    assert bridge(value, 0., 0., 0, value) == 0
    assert value[1] == -1. and value[3] == 0.
    assert value[9] == -2.5 and value[11] == -2.5
    assert bridge(value, 2., 2., 0, value) == 0
    assert value[1] == -2. and value[3] == 0. and value[5] == 4.
    assert list(value)[8:13] == [2., 0., 0., -2., 0.]
    # Starting over from the old source EV produces different lanes. The
    # operator must carry the actual previous state between invocations.
    fresh = state()
    assert bridge(fresh, 2., 2., 0, fresh) == 0
    assert fresh[1] == -1. and fresh[9] == -.5


@pytest.mark.parametrize('amount', [-2., 1., 3.])
def test_nonzero_profile_gate_preserves_existing_curve(bridge, amount):
    value = state(amount)
    value[1] = -.75
    assert bridge(value, 0., 0., 0, value) == 0
    assert value[1] == -.75 and value[9] == -3.5


def test_hdr_routes_negative_ev_without_curve_contribution(bridge):
    value = state()
    value[1] = -.75
    assert bridge(value, 0., 0., 1, value) == 0
    assert value[1] == -.75 and value[9] == -3.5


def test_first_argument_is_independent_of_allocation_second_argument(bridge):
    value = ARRAY(*([0.] * 13)); value[5] = 1.
    result = ARRAY()
    assert bridge(value, -0., 2., 0, result) == 0
    assert math.copysign(1., result[2]) == -1.
    assert result[8] == 2. and result[5] == 4.
    assert result[3] == 0. and math.copysign(1., result[3]) == 1.


@pytest.mark.parametrize('index', range(13))
def test_nonfinite_snapshot_rejects_atomically(bridge, index):
    value = state(); value[index] = math.nan
    before = bytes(value)
    assert bridge(value, 0., 0., 0, value) == 1
    assert bytes(value) == before


@pytest.mark.parametrize('first,second', [(math.inf, 0.), (0., -math.inf), (0., math.nan)])
def test_nonfinite_arguments_reject_atomically(bridge, first, second):
    value = state(); before = bytes(value)
    assert bridge(value, first, second, 0, value) == 1
    assert bytes(value) == before


def test_result_overflow_preserves_output(bridge):
    value = state(); value[3] = 1024.
    result = ARRAY(*([123.25] * 13)); before = bytes(result)
    assert bridge(value, 0., 0., 0, result) == 2
    assert bytes(result) == before


def test_bridge_rejects_partial_alias_and_misalignment_without_writes(bridge):
    storage = C.create_string_buffer(120)
    C.memmove(storage, state(), 104)
    before = bytes(storage)
    a = C.cast(storage, C.POINTER(C.c_double))
    for offset in (1, 8):
        b = C.cast(C.addressof(storage) + offset, C.POINTER(C.c_double))
        assert bridge(a, 0., 0., 0, b) == 3
        assert bytes(storage) == before
    assert bridge(None, 0., 0., 0, a) == 3
    assert bridge(a, 0., 0., 2, a) == 3
    assert bytes(storage) == before
