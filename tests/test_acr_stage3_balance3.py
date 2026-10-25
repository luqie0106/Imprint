from __future__ import annotations

import ctypes
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
ORACLE_PATH = ROOT / "tests/acr_support/balance3_oracle.py"
SPEC = importlib.util.spec_from_file_location("balance3_oracle", ORACLE_PATH)
oracle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(oracle)

BRIDGE_CPP = NATIVE / "tools/acr_stage3_balance3_bridge.cpp"
HEADER = NATIVE / "src/acr_stage3_balance3.hpp"
KERNEL_CPP = NATIVE / "src/acr_stage3_balance3.cpp"
OFFICIAL_OBJECT = BUILD / "CMakeFiles/imprint_renderer.dir/src/acr_stage3_balance3.cpp.o"
SHARED_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"

OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
INVALID_STRIDE = 3
UNSUPPORTED_ALIGNMENT = 4
INVALID_PARAMETERS = 5
ARITHMETIC_OVERFLOW = 6
OVERLAPPING_BUFFERS = 7
SENTINEL = np.float32(-1234.5)
SENTINEL_BITS = int(np.asarray([SENTINEL], dtype=np.float32).view(np.uint32)[0])
SIZE = ctypes.c_size_t
F32 = ctypes.POINTER(ctypes.c_float)


def _compile_bridge(target: Path, linked_file: Path | None) -> Path:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "a C++ compiler is required to build the balance3 bridge"
    shared_flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    command = [
        compiler, "-std=c++17", "-O2", "-fPIC", "-fno-fast-math",
        "-ffp-contract=off", shared_flag, "-I", str(NATIVE / "src"),
        str(BRIDGE_CPP),
    ]
    if linked_file is not None:
        command.append(str(linked_file))
    command.extend(["-o", str(target)])
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, (
        "balance3 bridge link failed:\n" + completed.stdout + completed.stderr)
    return target


class Bridge:
    def __init__(self, path: Path):
        self.library = ctypes.CDLL(str(path))
        self.difference = self.library.acr_test_balance3_difference
        self.difference.argtypes = [F32, SIZE, SIZE, ctypes.c_int, F32]
        self.difference.restype = ctypes.c_int
        self.filter = self.library.acr_test_balance3_filter
        self.filter.argtypes = [F32, SIZE, SIZE, F32, SIZE, F32]
        self.filter.restype = ctypes.c_int
        self.subtract = self.library.acr_test_balance3_subtract
        self.subtract.argtypes = [F32, SIZE, SIZE, F32, SIZE, F32]
        self.subtract.restype = ctypes.c_int


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    if request.param == "source":
        missing = [str(path) for path in (BRIDGE_CPP, HEADER, KERNEL_CPP)
                   if not path.exists()]
        assert not missing, "balance3 source bridge inputs are missing: " + ", ".join(missing)
        linked_file = KERNEL_CPP
    else:
        missing = [str(path) for path in (BRIDGE_CPP, HEADER, KERNEL_CPP, OFFICIAL_OBJECT)
                   if not path.exists()]
        assert not missing, (
            "formal imprint_renderer balance3 object is missing; rebuild the renderer "
            "before running official_object cases: " + ", ".join(missing))
        stale = [str(path) for path in (KERNEL_CPP, HEADER)
                 if OFFICIAL_OBJECT.stat().st_mtime_ns < path.stat().st_mtime_ns]
        assert not stale, (
            "formal imprint_renderer balance3 object is stale; rebuild the renderer "
            "before running official_object cases: " + ", ".join(stale))
        linked_file = OFFICIAL_OBJECT

    target = tmp_path_factory.mktemp("balance3_" + request.param) / (
        "libbalance3" + SHARED_SUFFIX)
    return Bridge(_compile_bridge(target, linked_file))


def _aligned_f32(count: int, fill: float | None = None) -> np.ndarray:
    """Allocate a contiguous float32 vector whose first sample is 16-byte aligned."""
    backing = np.empty(count + 4, dtype=np.float32)
    shift = ((-backing.ctypes.data) % 16) // np.dtype(np.float32).itemsize
    result = backing[shift:shift + count]
    assert result.ctypes.data % 16 == 0
    if fill is not None:
        result.fill(np.float32(fill))
    return result


def _random_row_case(width: int, radius: int, seed: int):
    # Keep the center pointer 16-byte aligned while supplying radius samples
    # on both horizontal sides of every output.
    guards = ((radius + 3) // 4) * 4
    row_samples = width + 2 * guards
    stride = ((row_samples + 7) // 4) * 4
    stride = max(stride, 4)
    rows = max(1, 2 * radius + 1)
    rng = np.random.default_rng(seed)
    source = _aligned_f32(rows * stride)
    source[:] = rng.uniform(-1.25, 1.25, size=source.size).astype(np.float32)
    center_offset = radius * stride + guards
    coeff = _aligned_f32(radius + 1)
    coeff[:] = rng.uniform(-0.7, 0.7, size=coeff.size).astype(np.float32)
    return source, center_offset, stride, coeff


def _ptr(array: np.ndarray, index: int = 0) -> F32:
    address = int(array.ctypes.data + index * np.dtype(np.float32).itemsize)
    return ctypes.cast(address, F32)


def _call_difference(bridge, source, center_offset, stride, width, reverse, dst):
    return bridge.difference(_ptr(source, center_offset), stride, width, reverse,
                             _ptr(dst))


def _call_diagonal(bridge, operation, source, center_offset, stride, width,
                   coeff, radius, dst):
    function = bridge.filter if operation == "filter" else bridge.subtract
    return function(_ptr(source, center_offset), stride, width, _ptr(coeff), radius,
                    _ptr(dst))


def test_reference_has_a_hardcoded_single_rounding_fma_counterexample():
    one_up = np.asarray([0x3F800001], dtype=np.uint32).view(np.float32)[0]
    one_below = np.asarray([0x3F7FFFFE], dtype=np.uint32).view(np.float32)[0]
    fused = oracle.fma32(one_up, one_below, np.float32(-1.0))
    separate = oracle.add32(oracle.mul32(one_up, one_below), np.float32(-1.0))
    assert int(oracle.uint32_bits(np.asarray([fused]))[0]) == 0xA8800000
    assert int(oracle.uint32_bits(np.asarray([separate]))[0]) == 0x00000000


@pytest.mark.parametrize("reverse", [0, 1])
def test_difference_row_matches_instruction_order_and_preserves_source(bridge, reverse):
    width, radius, seed = 20, 1, 13870 + reverse
    source, center, stride, _ = _random_row_case(width, radius, seed)
    # Make the four directional samples different so row/column swaps are visible.
    source[center - stride - 1] = np.float32(-0.75)
    source[center - stride + 1] = np.float32(0.125)
    source[center + stride - 1] = np.float32(0.5)
    source[center + stride + 1] = np.float32(1.0)
    before = source.copy()
    destination = _aligned_f32(width + 5, SENTINEL)
    expected = oracle.difference_row(source, center, stride, width, reverse)

    status = _call_difference(bridge, source, center, stride, width, reverse, destination)

    assert status == OK
    assert np.isfinite(destination[:width]).all()
    oracle.assert_float32_bits(destination[:width], expected)
    assert np.all(destination[width:].view(np.uint32) == SENTINEL_BITS)
    oracle.assert_float32_bits(source, before, message="difference modified source/padding")


@pytest.mark.parametrize("width,radius", [(8, 0), (12, 1), (20, 3), (8, 32)])
# The recovered APIs specialize direction: 008139cc is plus/filter and
# 00813cf8 is minus/subtract.
@pytest.mark.parametrize(("direction", "operation"),
                         [("plus", "filter"), ("minus", "subtract")])
def test_diagonal_random_rows_match_independent_float32_reference(
        bridge, width, radius, direction, operation):
    source, center, stride, coeff = _random_row_case(
        width, radius, 9000 + width * 13 + radius * 7 + (direction == "minus"))
    # Exercise exact signed-zero propagation along with finite random samples.
    if radius == 0:
        source[center] = np.float32(-0.0)
        coeff[0] = np.float32(1.0)
    destination = _aligned_f32(width + 5, SENTINEL)
    if operation == "subtract":
        destination[:width] = np.linspace(-0.5, 0.5, width, dtype=np.float32)
    source_before = source.copy()
    destination_before = destination.copy()
    expected = oracle.diagonal_row(
        source, center, stride, width, coeff, radius, direction=direction,
        destination=destination_before, subtract=(operation == "subtract"))

    status = _call_diagonal(
        bridge, operation, source, center, stride, width, coeff, radius, destination)

    assert status == OK
    assert np.isfinite(destination[:width]).all()
    oracle.assert_float32_bits(destination[:width], expected)
    assert np.all(destination[width:].view(np.uint32) == SENTINEL_BITS)
    oracle.assert_float32_bits(source, source_before,
                               message="diagonal kernel modified source/padding")


@pytest.mark.parametrize(("direction", "operation"),
                         [("plus", "filter"), ("minus", "subtract")])
def test_diagonal_kernel_uses_fma_not_separate_multiply_add(
        bridge, operation, direction):
    width, radius = 8, 1
    source, center, stride, coeff = _random_row_case(width, radius, 813908)
    # At output 0 the pair is (1 - 2^-23), with c0*center exactly -1.
    source[center] = np.float32(-1.0)
    coeff[0] = np.float32(1.0)
    coeff[1] = np.asarray([0x3F800001], dtype=np.uint32).view(np.float32)[0]
    neighbor = np.asarray([0x3F7FFFFE], dtype=np.uint32).view(np.float32)[0]
    diagonal = stride + 1 if direction == "plus" else stride - 1
    source[center - diagonal] = neighbor
    source[center + diagonal] = np.float32(0.0)
    destination = _aligned_f32(width + 3, 0.0)
    expected = oracle.diagonal_row(
        source, center, stride, width, coeff, radius, direction=direction,
        destination=destination.copy(), subtract=(operation == "subtract"))
    separate_pair = oracle.add32(source[center - diagonal],
                                 source[center + diagonal])
    separate = oracle.add32(
        oracle.mul32(coeff[1], separate_pair), oracle.mul32(coeff[0], source[center]))
    if operation == "subtract":
        separate = oracle.add32(destination[0], -separate)

    status = _call_diagonal(
        bridge, operation, source, center, stride, width, coeff, radius, destination)

    assert status == OK
    assert int(oracle.uint32_bits(destination[:1])[0]) == int(
        oracle.uint32_bits(expected[:1])[0])
    assert int(oracle.uint32_bits(expected[:1])[0]) != int(
        oracle.uint32_bits(np.asarray([separate], dtype=np.float32))[0])


def test_difference_preserves_negative_zero_from_reverse_subtraction(bridge):
    width, stride = 8, 16
    source = _aligned_f32(3 * stride, 0.0)
    center = stride + 4
    source[center] = np.float32(0.0)
    source[center - stride - 1:center - stride + width + 1] = np.float32(-0.0)
    source[center + stride - 1:center + stride + width + 1] = np.float32(-0.0)
    destination = _aligned_f32(width, SENTINEL)

    status = _call_difference(bridge, source, center, stride, width, 1, destination)

    assert status == OK
    assert np.all(destination.view(np.uint32) == np.uint32(0x80000000))


def test_asymmetric_impulses_distinguish_diagonal_directions_and_difference(bridge):
    width, radius = 8, 3
    source, center, stride, coeff = _random_row_case(width, radius, 813908)
    source.fill(np.float32(0.0))
    coeff.fill(np.float32(0.0))
    coeff[0] = np.float32(1.0)
    coeff[1] = np.float32(0.5)
    coeff[2] = np.float32(-0.25)
    coeff[3] = np.float32(0.125)
    plus_d, minus_d = stride + 1, stride - 1
    source[center - plus_d] = np.float32(2.0)
    source[center + plus_d] = np.float32(1.0)
    source[center - minus_d] = np.float32(7.0)
    source[center + 2 * minus_d] = np.float32(-3.0)
    plus = _aligned_f32(width, SENTINEL)
    minus = _aligned_f32(width, 0.0)
    status_plus = _call_diagonal(
        bridge, "filter", source, center, stride, width, coeff, radius, plus)
    status_minus = _call_diagonal(
        bridge, "subtract", source, center, stride, width, coeff, radius, minus)
    assert status_plus == status_minus == OK
    oracle.assert_float32_bits(
        plus, oracle.diagonal_row(source, center, stride, width, coeff, radius,
                                  direction="plus"))
    oracle.assert_float32_bits(
        minus, oracle.diagonal_row(source, center, stride, width, coeff, radius,
                                  direction="minus", destination=np.zeros(width,
                                                                           dtype=np.float32),
                                  subtract=True))
    assert not np.array_equal(plus.view(np.uint32), minus.view(np.uint32))

    source[center - stride - 1] = np.float32(-0.25)
    source[center - stride + 1] = np.float32(0.75)
    source[center + stride - 1] = np.float32(1.25)
    source[center + stride + 1] = np.float32(2.0)
    difference = _aligned_f32(width, SENTINEL)
    status = _call_difference(bridge, source, center, stride, width, 0, difference)
    assert status == OK
    np.testing.assert_array_equal(
        oracle.uint32_bits(difference),
        oracle.uint32_bits(oracle.difference_row(source, center, stride, width, 0)))


@pytest.mark.parametrize("operation", ["filter", "subtract"])
def test_radius_zero_is_a_separate_multiply_and_optional_subtract(
        bridge, operation):
    width, stride = 12, 16
    source = _aligned_f32(stride, 0.0)
    center = 0
    source[:width] = np.linspace(-2.0, 2.0, width, dtype=np.float32)
    coeff = _aligned_f32(1)
    coeff[0] = np.float32(-0.75)
    destination = _aligned_f32(width + 2, SENTINEL)
    destination[:width] = np.float32(0.25)
    before = destination.copy()
    expected = oracle.diagonal_row(
        source, center, stride, width, coeff, 0, direction="minus",
        destination=before, subtract=(operation == "subtract"))

    status = _call_diagonal(
        bridge, operation, source, center, stride, width, coeff, 0, destination)

    assert status == OK
    oracle.assert_float32_bits(destination[:width], expected)
    assert np.all(destination[width:].view(np.uint32) == SENTINEL_BITS)


def _failure_call(bridge, expected_status, source, center, stride, width,
                  destination, *, operation="difference", reverse=0,
                  coeff=None, radius=0, source_pointer=None, destination_pointer=None):
    before = destination.view(np.uint32).copy()
    center_pointer = (_ptr(source, center) if source_pointer is None else source_pointer)
    destination_pointer = (_ptr(destination) if destination_pointer is None
                           else destination_pointer)
    if operation == "difference":
        status = bridge.difference(center_pointer, stride, width, reverse,
                                   destination_pointer)
    else:
        function = bridge.filter if operation == "filter" else bridge.subtract
        coeff_pointer = (_ptr(coeff) if coeff is not None
                         else ctypes.cast(ctypes.c_void_p(0), F32))
        status = function(center_pointer, stride, width, coeff_pointer, radius,
                          destination_pointer)
    assert status == expected_status
    np.testing.assert_array_equal(destination.view(np.uint32), before,
                                  err_msg="rejected call partially wrote destination")


def test_rejected_dimensions_stride_and_reverse_leave_output_untouched(bridge):
    source, center, stride, coeff = _random_row_case(8, 1, 2718)
    destination = _aligned_f32(12, SENTINEL)
    _failure_call(bridge, INVALID_DIMENSIONS, source, center, stride, 0, destination)
    _failure_call(bridge, INVALID_DIMENSIONS, source, center, stride, 6, destination)
    _failure_call(bridge, INVALID_STRIDE, source, center, 8, 8, destination)
    _failure_call(bridge, INVALID_STRIDE, source, center, 17, 8, destination)
    _failure_call(bridge, INVALID_PARAMETERS, source, center, stride, 8, destination,
                  reverse=2)
    _failure_call(bridge, INVALID_PARAMETERS, source, center, stride, 8, destination,
                  operation="filter", coeff=coeff, radius=33)


@pytest.mark.parametrize("bad_value", [np.float32(np.nan), np.float32(np.inf)])
def test_nonfinite_coefficient_rejected_before_writing(bridge, bad_value):
    source, center, stride, coeff = _random_row_case(8, 1, 449)
    coeff[1] = bad_value
    destination = _aligned_f32(11, SENTINEL)
    _failure_call(bridge, INVALID_PARAMETERS, source, center, stride, 8, destination,
                  operation="filter", coeff=coeff, radius=1)


def test_null_arguments_unaligned_pointers_and_overlap_are_rejected_atomically(bridge):
    source, center, stride, coeff = _random_row_case(8, 1, 814)
    destination = _aligned_f32(12, SENTINEL)
    _failure_call(bridge, INVALID_ARGUMENT, source, center, stride, 8, destination,
                  source_pointer=ctypes.cast(ctypes.c_void_p(0), F32))

    unaligned_source = ctypes.cast(
        ctypes.c_void_p(int(source.ctypes.data + (center + 1) * 4)), F32)
    _failure_call(bridge, UNSUPPORTED_ALIGNMENT, source, center, stride, 8, destination,
                  source_pointer=unaligned_source)
    unaligned_destination = ctypes.cast(
        ctypes.c_void_p(int(destination.ctypes.data + 4)), F32)
    _failure_call(bridge, UNSUPPORTED_ALIGNMENT, source, center, stride, 8, destination,
                  destination_pointer=unaligned_destination)

    source_before = source.view(np.uint32).copy()
    aliased_destination = _ptr(source, center)
    status = bridge.difference(_ptr(source, center), stride, 8, 0, aliased_destination)
    assert status == OVERLAPPING_BUFFERS
    np.testing.assert_array_equal(source.view(np.uint32), source_before,
                                  err_msg="overlap rejection modified source")

    _failure_call(bridge, INVALID_ARGUMENT, source, center, stride, 8, destination,
                  operation="filter", coeff=None, radius=1)


def test_pointer_span_overflow_is_rejected_before_any_read_or_write(bridge):
    width = 4
    source = _aligned_f32(64, 0.25)
    destination = _aligned_f32(8, SENTINEL)
    huge_stride = ((1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1) & ~3
    _failure_call(bridge, ARITHMETIC_OVERFLOW, source, 16, huge_stride, width,
                  destination)
