from __future__ import annotations

import ctypes as C
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BUILD = NATIVE / "build"
ORACLE_PATH = ROOT / "tests/acr_support/balance3_basic_oracle.py"
ORACLE_SPEC = importlib.util.spec_from_file_location(
    "balance3_basic_oracle", ORACLE_PATH)
oracle = importlib.util.module_from_spec(ORACLE_SPEC)
assert ORACLE_SPEC.loader is not None
ORACLE_SPEC.loader.exec_module(oracle)

BRIDGE_CPP = NATIVE / "tools/acr_stage3_balance3_finish_bridge.cpp"
HEADER = NATIVE / "src/acr_stage3_balance3_finish.hpp"
KERNEL_CPP = NATIVE / "src/acr_stage3_balance3_finish.cpp"
OFFICIAL_OBJECT = BUILD / (
    "CMakeFiles/imprint_renderer.dir/src/acr_stage3_balance3_finish.cpp.o")
SHARED_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"

OK = 0
INVALID_STRIDE = 3
BUFFER_TOO_SMALL = 6
OVERLAPPING_BUFFERS = 8
UNSUPPORTED_DOMAIN = 9
SIZE = C.c_size_t
F32 = C.POINTER(C.c_float)
SENTINEL = np.float32(-1234.5)
SENTINEL_BITS = int(np.asarray([SENTINEL], dtype=np.float32).view(np.uint32)[0])
GUARD_COLUMNS = 4


def _aligned_f32(count: int, fill: float | None = None) -> np.ndarray:
    """Allocate a contiguous float32 vector with a 16-byte aligned start."""
    backing = np.empty(count + 4, dtype=np.float32)
    shift = ((-backing.ctypes.data) % 16) // np.dtype(np.float32).itemsize
    result = backing[shift:shift + count]
    assert result.ctypes.data % 16 == 0
    if fill is not None:
        result.fill(np.float32(fill))
    return result


def _ptr(array: np.ndarray, index: int = 0) -> F32:
    address = int(array.ctypes.data + index * np.dtype(np.float32).itemsize)
    return C.cast(address, F32)


def _compile_bridge(target: Path, linked_file: Path) -> Path:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "a C++ compiler is required to build the balance3 basic bridge"
    command = [
        compiler, "-std=c++17", "-O2", "-fPIC", "-fno-fast-math",
        "-ffp-contract=off", "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", str(NATIVE / "src"), str(BRIDGE_CPP), str(linked_file),
        "-o", str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, (
        "balance3 basic bridge link failed:\n" + completed.stdout + completed.stderr)
    return target


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    required = (BRIDGE_CPP, HEADER, KERNEL_CPP)
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, "balance3 basic source inputs are missing: " + ", ".join(missing)

    if request.param == "source":
        linked_file = KERNEL_CPP
    else:
        assert OFFICIAL_OBJECT.is_file(), (
            "formal imprint_renderer balance3 finish object is missing; rebuild the renderer")
        stale = [str(path) for path in (KERNEL_CPP, HEADER)
                 if OFFICIAL_OBJECT.stat().st_mtime_ns < path.stat().st_mtime_ns]
        assert not stale, (
            "formal imprint_renderer balance3 finish object is stale; rebuild the renderer: "
            + ", ".join(stale))
        linked_file = OFFICIAL_OBJECT

    target = tmp_path_factory.mktemp("balance3_basic_" + request.param) / (
        "libbalance3_basic" + SHARED_SUFFIX)
    result = C.CDLL(str(_compile_bridge(target, linked_file)))
    result.acr_test_balance3_basic.argtypes = [
        F32, SIZE, F32, SIZE, SIZE, C.c_int, C.c_int, F32, SIZE]
    result.acr_test_balance3_basic.restype = C.c_int
    return result


def _case(width: int, *, stride: int | None = None, seed: int = 20261019):
    stride = width + 4 if stride is None else stride
    assert stride % 4 == 0 and stride >= width + 2
    source = _aligned_f32(3 * stride + 8, 0.0)
    center = stride + GUARD_COLUMNS
    diff = _aligned_f32(width + 8, 0.0)
    dst = _aligned_f32(width + 8, SENTINEL)
    rng = np.random.default_rng(seed)
    source[:] = rng.uniform(-1.0, 1.5, size=source.size).astype(np.float32)
    diff[:] = rng.uniform(-1.0, 1.0, size=diff.size).astype(np.float32)
    return source, center, stride, diff, dst


def _call(bridge, source, center, stride, diff, width, phase, row_even,
          dst, *, diff_count=None, dst_count=None):
    return bridge.acr_test_balance3_basic(
        _ptr(source, center), stride, _ptr(diff),
        diff.size if diff_count is None else diff_count,
        width, phase, row_even, _ptr(dst),
        dst.size if dst_count is None else dst_count)


def _assert_untouched(destination: np.ndarray, before: np.ndarray) -> None:
    oracle.assert_float32_bits(destination, before,
                               message="rejected call modified destination")


@pytest.mark.parametrize("phase", [0, 1])
@pytest.mark.parametrize("row_even", [0, 1])
def test_hardcoded_lane_selection_direct_values_and_preservation(
        bridge, phase, row_even):
    width = 4
    source, center, stride, diff, dst = _case(width)
    source.fill(np.float32(0.25))
    diff.fill(np.float32(0.5))
    before = dst.copy()

    assert _call(bridge, source, center, stride, diff, width, phase,
                 row_even, dst) == OK

    selected_value = np.float32(0.0 if row_even else 0.5)
    expected = before.copy()
    for lane in range(width):
        if (lane & 1) != phase:
            expected[lane] = selected_value
    oracle.assert_float32_bits(dst, expected)
    assert np.all(dst[width:].view(np.uint32) == SENTINEL_BITS)


@pytest.mark.parametrize("width", [4, 12, 32, 64])
@pytest.mark.parametrize("phase,row_even", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_random_rows_match_independent_instruction_model(
        bridge, width, phase, row_even):
    source, center, stride, diff, dst = _case(
        width, seed=9000 + width * 19 + phase * 3 + row_even)
    source_before, diff_before, destination_before = (
        source.copy(), diff.copy(), dst.copy())
    expected = oracle.reference_row(source, center, stride, diff, width,
                                    phase, row_even, destination_before)

    assert _call(bridge, source, center, stride, diff, width, phase,
                 row_even, dst) == OK

    oracle.assert_float32_bits(dst, expected)
    oracle.assert_float32_bits(source, source_before,
                               message="kernel modified source or guard samples")
    oracle.assert_float32_bits(diff, diff_before,
                               message="kernel modified diff")
    assert np.all(dst[width:].view(np.uint32) == SENTINEL_BITS)


def test_diagonal_addition_association_is_nw_ne_then_sw_then_se(bridge):
    width = 4
    source, center, stride, diff, dst = _case(width)
    source.fill(np.float32(0.0))
    diff.fill(np.float32(0.0))
    # At lane zero the required sequence is ((NW + NE) + SW) + SE = 7.
    source[center - stride - 1] = np.float32(1.0e20)
    source[center - stride + 1] = np.float32(-1.0e20)
    source[center + stride - 1] = np.float32(3.0)
    source[center + stride + 1] = np.float32(4.0)

    # phase=1 selects lane zero; the direct accumulator is 7 * .125 = .875.
    assert _call(bridge, source, center, stride, diff, width, 1, 0, dst) == OK
    assert int(dst.view(np.uint32)[0]) == 0x3F600000


def test_signed_zero_is_preserved_and_negative_values_become_positive_zero(bridge):
    width = 4
    source, center, stride, diff, dst = _case(width)
    source.fill(np.float32(-0.0))
    diff.fill(np.float32(-0.0))
    dst.fill(np.float32(7.0))

    # lane zero is selected with phase one; strict comparison must retain -0.
    assert _call(bridge, source, center, stride, diff, width, 1, 0, dst) == OK
    assert int(dst.view(np.uint32)[0]) == 0x80000000

    source.fill(np.float32(-0.25))
    diff.fill(np.float32(0.0))
    dst.fill(np.float32(7.0))
    assert _call(bridge, source, center, stride, diff, width, 1, 0, dst) == OK
    assert int(dst.view(np.uint32)[0]) == 0x00000000


def test_extreme_finite_accumulators_clamp_infinities_and_oracle_nan_semantics(bridge):
    width = 4
    source, center, stride, diff, dst = _case(width)
    source.fill(np.float32(0.0))
    diff.fill(np.float32(0.0))
    largest = np.finfo(np.float32).max

    for diagonal, expected_bits in ((largest, 0x3F800000),
                                    (-largest, 0x00000000)):
        source.fill(np.float32(0.0))
        source[center - stride - 1] = diagonal
        source[center - stride + 1] = diagonal
        source[center + stride - 1] = diagonal
        source[center + stride + 1] = diagonal
        dst.fill(np.float32(7.0))
        # Four finite inputs overflow the sequential sum to +Inf or -Inf.
        assert _call(bridge, source, center, stride, diff, width, 1, 0,
                     dst) == OK
        assert int(dst.view(np.uint32)[0]) == expected_bits

    # Finite API inputs cannot produce +Inf followed by -Inf in this
    # left-associated sum, but the recovered FMINNM rule still treats a NaN
    # accumulator as numeric-min(1, NaN) == 1.
    assert oracle._fminnm32(np.float32(np.nan), np.float32(1.0)) == np.float32(1.0)


def test_direct_accumulator_differs_from_subtract_then_readd(bridge):
    width = 4
    source, center, stride, diff, dst = _case(width)
    source.fill(np.float32(0.0))
    diff.fill(np.float32(0.0))
    # Exact float bit patterns for the lane-one instruction-order counterexample.
    values = np.asarray([
        0x3FB365FE, 0x3EA6D5C5, 0x3F8195AB, 0x3FF61EBA,
        0xBEC15352, 0x3F64B248,
    ], dtype=np.uint32).view(np.float32)
    source[center - stride] = values[0]       # NW
    source[center - stride + 2] = values[1]   # NE
    source[center + stride] = values[2]       # SW
    source[center + stride + 2] = values[3]   # SE
    source[center + 1] = values[4]
    diff[1] = values[5]

    raw = oracle.accumulator(source, center, stride, diff, 1, 0)
    direct = oracle._fminnm32(raw, np.float32(1.0))
    if direct < np.float32(0.0):
        direct = np.float32(0.0)
    readded = oracle.add32(values[4], oracle.sub32(raw, values[4]))
    wrong = oracle._fminnm32(readded, np.float32(1.0))
    if wrong < np.float32(0.0):
        wrong = np.float32(0.0)
    assert int(oracle.uint32_bits(np.asarray([direct]))[0]) == 0x3F573844
    assert int(oracle.uint32_bits(np.asarray([wrong]))[0]) == 0x3F573843

    # phase zero selects lane one. The native result must use the direct value.
    assert _call(bridge, source, center, stride, diff, width, 0, 0, dst) == OK
    assert int(dst.view(np.uint32)[1]) == 0x3F573844


def test_last_guard_nonfinite_inputs_reject_atomically(bridge):
    width = 4
    source, center, stride, diff, dst = _case(width)
    before = dst.copy()
    # Last south-east guard read, on lane three (which is unselected here).
    source[center + stride + width] = np.float32(np.inf)
    assert _call(bridge, source, center, stride, diff, width, 1, 0,
                 dst) == UNSUPPORTED_DOMAIN
    _assert_untouched(dst, before)

    source[center + stride + width] = np.float32(0.0)
    diff[width - 1] = np.float32(np.nan)
    assert _call(bridge, source, center, stride, diff, width, 1, 0,
                 dst) == UNSUPPORTED_DOMAIN
    _assert_untouched(dst, before)


def test_alias_short_capacity_and_invalid_stride_reject_atomically(bridge):
    width = 4
    source, center, stride, diff, dst = _case(width)
    before = dst.copy()
    assert _call(bridge, source, center, stride, diff, width, 0, 0, dst,
                 diff_count=width - 1) == BUFFER_TOO_SMALL
    _assert_untouched(dst, before)
    assert _call(bridge, source, center, stride, diff, width, 0, 0, dst,
                 dst_count=width - 1) == BUFFER_TOO_SMALL
    _assert_untouched(dst, before)

    source_before = source.copy()
    assert bridge.acr_test_balance3_basic(
        _ptr(source, center), stride, _ptr(diff), diff.size, width, 0, 0,
        _ptr(source), source.size) == OVERLAPPING_BUFFERS
    oracle.assert_float32_bits(source, source_before,
                               message="source/destination alias rejection wrote data")

    diff_before = diff.copy()
    assert bridge.acr_test_balance3_basic(
        _ptr(source, center), stride, _ptr(diff), diff.size, width, 0, 0,
        _ptr(diff), diff.size) == OVERLAPPING_BUFFERS
    oracle.assert_float32_bits(diff, diff_before,
                               message="diff/destination alias rejection wrote data")

    assert _call(bridge, source, center, width, diff, width, 0, 0, dst) == INVALID_STRIDE
    _assert_untouched(dst, before)
