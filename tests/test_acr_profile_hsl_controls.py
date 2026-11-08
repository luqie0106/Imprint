"""Profile -> HSL24 indexed writes: independent double-bit/integer oracle.

Both strict-FP source compilation and the current official CMake object are
tested. These are prepared control-loop tests, not complete Adobe Prepare or
pixel HSL tests. Run -k source until the official object has been built.
"""

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
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_profile_hsl_controls.cpp"
HEADER = SOURCE.with_suffix(".hpp")
BRIDGE = NATIVE / "tools/acr_profile_hsl_controls_bridge.cpp"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_profile_hsl_controls.cpp.o"
sys.path.insert(0, str(ROOT / "tests/acr_support"))
import profile_hsl_controls_oracle as oracle  # noqa: E402

I32_PTR = C.POINTER(C.c_int32)
DOUBLE_PTR = C.POINTER(C.c_double)
U32_PTR = C.POINTER(C.c_uint32)
ALL_LANES = (1 << 24) - 1
INT_MIN, INT_MAX = -(2**31), 2**31 - 1
INITIAL_VALUES = np.arange(24, dtype=np.int32) * 7919 - 1234567
INITIAL_BLENDED = np.array([(-1.0 if i % 2 else 1.0) * (9123.25 + i)
                            for i in range(24)], dtype=np.float64)
INITIAL_BLENDED[0] = -0.0
INITIAL_MASK = 0xA5C39E71


@pytest.fixture(scope="module", params=("source", "official_object"))
def native(request, tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the focused bridge"
    core = SOURCE
    if request.param == "official_object":
        assert OBJECT.is_file(), f"Build the official CMake object first: {OBJECT}"
        assert OBJECT.stat().st_mtime_ns >= max(SOURCE.stat().st_mtime_ns,
                                               HEADER.stat().st_mtime_ns), "CMake object is stale"
        core = OBJECT
    if sys.platform == "darwin":
        suffix, shared = ".dylib", ["-dynamiclib", "-fPIC"]
    elif sys.platform == "win32":
        suffix, shared = ".dll", ["-shared"]
    else:
        suffix, shared = ".so", ["-shared", "-fPIC"]
    output = tmp_path_factory.mktemp("profile_hsl_" + request.param) / ("libprofile_hsl" + suffix)
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", "-ffp-contract=off", *shared,
        str(core), str(BRIDGE), "-o", str(output),
    ], check=True, capture_output=True, text=True)
    library = C.CDLL(str(output))
    function = library.acr_test_profile_hsl_controls
    function.argtypes = [I32_PTR, I32_PTR, C.c_double, C.c_uint32, C.c_uint32,
                         I32_PTR, DOUBLE_PTR, U32_PTR]
    function.restype = C.c_int
    return function


def call(native, current, defaults, scalar=1.0, monochrome=0, gray=0, *, null=None):
    current = np.asarray(current, dtype=np.int32)
    defaults = np.asarray(defaults, dtype=np.int32)
    assert current.shape == defaults.shape == (24,)
    assert current.flags.c_contiguous and defaults.flags.c_contiguous
    before_current, before_defaults = current.tobytes(), defaults.tobytes()
    values, blended = INITIAL_VALUES.copy(), INITIAL_BLENDED.copy()
    wrote = C.c_uint32(INITIAL_MASK)
    args = [current.ctypes.data_as(I32_PTR), defaults.ctypes.data_as(I32_PTR),
            scalar, monochrome, gray, values.ctypes.data_as(I32_PTR),
            blended.ctypes.data_as(DOUBLE_PTR), C.byref(wrote)]
    if null is not None:
        args[null] = None
    status = native(*args)
    assert current.tobytes() == before_current
    assert defaults.tobytes() == before_defaults
    return status, values, blended, wrote.value


def check(native, current, defaults, scalar=1.0, monochrome=0, gray=0):
    result = call(native, current, defaults, scalar, monochrome, gray)
    status, values, blended, wrote = result
    expected = oracle.evaluate(current, defaults, scalar, monochrome, gray)
    assert status == 0
    assert wrote == expected.wrote_mask
    np.testing.assert_array_equal(values, expected.values)
    np.testing.assert_array_equal(blended.view(np.uint64),
                                  np.asarray(expected.blended_bits, dtype=np.uint64))
    return result


def assert_preserved(result, status):
    actual_status, values, blended, wrote = result
    assert actual_status == status
    assert values.tobytes() == INITIAL_VALUES.tobytes()
    assert blended.tobytes() == INITIAL_BLENDED.tobytes()
    assert wrote == INITIAL_MASK


@pytest.mark.parametrize("scalar", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_complete_201_by_201_control_grid(native, scalar):
    # Each Cartesian pair is tested, batched into 24-lane calls. The last
    # batch's padding uses invalid defaults so mask boundaries are checked.
    pairs = [(c, d) for c in range(-100, 101) for d in range(-100, 101)]
    for start in range(0, len(pairs), 24):
        batch = pairs[start:start + 24]
        current = [c for c, _ in batch] + [INT_MIN] * (24 - len(batch))
        defaults = [d for _, d in batch] + [-999999] * (24 - len(batch))
        check(native, current, defaults, scalar)


@pytest.mark.parametrize("scalar", [
    -sys.float_info.max, -2.0, -0.0,
    math.nextafter(0.0, 1.0), sys.float_info.min, 1.0e-300,
    math.nextafter(1.0, 0.0), math.nextafter(1.0, math.inf),
    math.nextafter(2.0, 0.0), math.nextafter(2.0, math.inf),
    100.0, sys.float_info.max,
])
def test_scalar_boundaries_and_extremes(native, scalar):
    current = [INT_MIN, INT_MAX, -999999, -101, -100, -99, -1, 0, 1, 99, 100, 101] * 2
    for default in (-100, -99, -1, 0, 1, 99, 100):
        check(native, current, [default] * 24, scalar)


def test_zero_default_preserves_unclamped_double_but_caller_clamps_int(native):
    current = [INT_MIN, INT_MAX, -999999, -101, -100, -99, -1, 0, 1, 99, 100, 101] * 2
    for scalar in (-2.0, -0.0, 0.5, 1.0, 2.0, sys.float_info.max):
        _, values, blended, mask = check(native, current, [0] * 24, scalar)
        assert mask == ALL_LANES
        np.testing.assert_array_equal(blended, np.asarray(current, dtype=np.float64))
        np.testing.assert_array_equal(values, np.clip(current, -100, 100))


def test_invalid_defaults_skip_without_clamping_current(native):
    current = [INT_MIN, INT_MAX, -999999, -101, 101, 42] * 4
    defaults = [-999999, INT_MIN, INT_MAX, -101, 101, 999999] * 4
    _, values, blended, mask = check(native, current, defaults, 2.0)
    assert mask == 0
    np.testing.assert_array_equal(values, current)
    np.testing.assert_array_equal(blended, np.asarray(current, dtype=np.float64))


def test_all_lanes_have_independent_gate_and_mask_bits(native):
    current = [INT_MIN, INT_MAX, -999999, -101, 101, 42] * 4
    for lane in range(24):
        defaults = [-999999] * 24
        defaults[lane] = [-100, -1, 0, 1, 100][lane % 5]
        assert check(native, current, defaults, 1.5)[3] == 1 << lane
    defaults = [100 if lane % 2 else -999999 for lane in range(24)]
    assert check(native, current, defaults)[3] == 0xAAAAAA
    defaults = [-999999 if lane % 2 else -100 for lane in range(24)]
    assert check(native, current, defaults)[3] == 0x555555


def test_full_byte_convert_to_gray_and_monochrome_gates(native):
    current = [INT_MIN, INT_MAX, -999999, -101, 101, 42] * 4
    defaults = [-100, -1, 0, 1, 100, 42] * 4
    for gray in range(256):
        for monochrome in (0, 1):
            _, values, blended, mask = check(native, current, defaults, 1.5, monochrome, gray)
            assert mask == (0 if monochrome or gray == 1 else ALL_LANES)
            if mask == 0:
                np.testing.assert_array_equal(values, current)
                np.testing.assert_array_equal(blended, np.asarray(current, dtype=np.float64))


@pytest.mark.parametrize("scalar", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejects_before_gates_and_preserves_entire_result(native, scalar):
    # Bridge verifies byte-for-byte core result preservation, including padding,
    # in addition to the external arrays/mask and unchanged input checked here.
    for defaults in ([0] * 24, [42] * 24, [-999999] * 24):
        for monochrome, gray in ((0, 0), (1, 0), (0, 1), (0, 2), (1, 255)):
            result = call(native, [INT_MIN, INT_MAX, -999999] * 8,
                          defaults, scalar, monochrome, gray)
            assert_preserved(result, 1)


@pytest.mark.parametrize("null", [0, 1, 5, 6, 7])
def test_bridge_null_arguments_preserve_outputs(native, null):
    assert_preserved(call(native, [42] * 24, [50] * 24, null=null), 2)


@pytest.mark.parametrize("monochrome, gray", [
    (2, 0), (255, 0), (0xFFFFFFFF, 0), (0, 256), (0, 0xFFFFFFFF), (2, 256),
])
def test_bridge_rejects_noncanonical_argument_widths(native, monochrome, gray):
    assert_preserved(call(native, [42] * 24, [50] * 24,
                          monochrome=monochrome, gray=gray), 2)


def test_instruction_order_and_half_away_rounding_witnesses(native):
    # Exact halves obtained from the saved operation sequence. D=-25/+25,
    # S=.5 and C=0 produce -12.5/+12.5, which must round away from zero.
    current = [0] * 24
    defaults = [-25, 25] * 12
    _, values, blended, _ = check(native, current, defaults, 0.5)
    np.testing.assert_array_equal(blended, [-12.5, 12.5] * 12)
    np.testing.assert_array_equal(values, [-13, 13] * 12)

    # D!=0, S=0 still normalizes and denormalizes C: the double is not
    # generally equal to C, though the rounded integer remains unchanged.
    current = list(range(-12, 12))
    _, values, blended, _ = check(native, current, [1] * 24, 0.0)
    np.testing.assert_array_equal(values, current)
    assert any(oracle.double_bits(actual) != oracle.double_bits(float(c))
               for actual, c in zip(blended, current))
