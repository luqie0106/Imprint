from __future__ import annotations

import ctypes
import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BUILD = NATIVE / "build"
SPEC = importlib.util.spec_from_file_location(
    "core3_colour_oracle", ROOT / "tests/acr_support/core3_colour_oracle.py")
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)

SHARED_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
SOURCE_BRIDGE = BUILD / f"libimprint_stage3_core3_colour_test_bridge{SHARED_SUFFIX}"
OFFICIAL_OBJECT = (
    BUILD / "CMakeFiles/imprint_renderer.dir/src/acr_stage3_core3_colour.cpp.o")
BRIDGE_CPP = NATIVE / "tools/acr_stage3_core3_colour_bridge.cpp"


def _compile_official_object_bridge(path: Path) -> Path:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ is required to link the official renderer object")
    if sys.platform == "darwin":
        shared_flag = "-dynamiclib"
    else:
        shared_flag = "-shared"
    command = [
        compiler, "-std=c++17", "-fPIC", "-fno-fast-math",
        "-ffp-contract=off", shared_flag, str(BRIDGE_CPP),
        str(OFFICIAL_OBJECT), "-o", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, (
        "official object bridge link failed:\n" + completed.stdout +
        completed.stderr)
    return path


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    if request.param == "source":
        if not SOURCE_BRIDGE.exists():
            pytest.skip("build imprint_stage3_core3_colour_test_bridge first")
        return oracle.Bridge(SOURCE_BRIDGE)

    if not OFFICIAL_OBJECT.exists():
        pytest.skip("build imprint_renderer before testing its official object")
    path = tmp_path_factory.mktemp("core3_colour_official") / (
        "libcore3_colour_official_bridge" + SHARED_SUFFIX)
    return oracle.Bridge(_compile_official_object_bridge(path))


def _assert_same(got: np.ndarray, expected: np.ndarray, message: str):
    assert oracle.same_nonfinite_semantics(got, expected), message


@pytest.mark.parametrize("width", [4, 12, 32])
def test_random_finite_rows_match_float32_oracle(bridge, width):
    source_stride = width + 8
    green_stride = width + 12
    source = oracle.make_rows(421 + width, width, source_stride)
    green = oracle.make_rows(918 + width, width, green_stride)
    source_before = source.copy()
    green_before = green.copy()

    for phase in (0, 1):
        status, out_a, out_b = bridge.execute(source, green, width, phase)
        assert status == oracle.OK
        expected_a, expected_b = oracle.reference(source, green, width, phase)
        _assert_same(out_a[:width], expected_a,
                     f"A differs from independent oracle at width={width}, phase={phase}")
        _assert_same(out_b[:width], expected_b,
                     f"B differs from independent oracle at width={width}, phase={phase}")
        assert np.all(out_a[width:] == np.float32(-101.0))
        assert np.all(out_b[width:] == np.float32(-202.0))

    assert np.array_equal(source.view(np.uint32), source_before.view(np.uint32))
    assert np.array_equal(green.view(np.uint32), green_before.view(np.uint32))


def test_fma_subnormal_tie_is_single_rounded(bridge):
    width = 4
    source = oracle.aligned_rows(3, 16)
    green = oracle.aligned_rows(3, 20)
    source.fill(np.float32(0.0))
    green.fill(np.float32(0.0))
    smallest = np.nextafter(np.float32(0.0), np.float32(1.0))
    green[1, oracle.GUARD + 2] = smallest
    source[0, oracle.GUARD + 1] = np.float32(2.0) * smallest

    status, out_a, out_b = bridge.execute(source, green, width, phase=0)
    assert status == oracle.OK
    expected_a, expected_b = oracle.reference(source, green, width, phase=0)
    _assert_same(out_a[:width], expected_a, "FMA probe A differs from oracle")
    _assert_same(out_b[:width], expected_b, "FMA probe B differs from oracle")

    # D = fmaf(0.25f, 2 subnormal units, 1 subnormal unit):
    # exact 1.5 units rounds to even 2. Separate multiply/add rounds the
    # product to zero first and would leave only 1 unit.
    fused = oracle._fma(0.25, np.float32(2.0) * smallest, smallest)
    separate_product = np.float32(np.float32(0.25) *
                                  (np.float32(2.0) * smallest))
    separate = oracle._add(separate_product, smallest)
    assert fused.view(np.uint32) != separate.view(np.uint32)
    assert out_b[2].view(np.uint32) == fused.view(np.uint32)


def test_diagonal_sum_uses_recovered_right_association(bridge):
    width = 4
    source = oracle.aligned_rows(3, 16)
    green = oracle.aligned_rows(3, 20)
    source.fill(np.float32(0.0))
    green.fill(np.float32(0.0))
    green[1, oracle.GUARD + 2] = np.float32(0.125)
    source[0, oracle.GUARD + 1] = np.float32(1.0e20)
    source[0, oracle.GUARD + 3] = np.float32(-1.0e20)
    source[2, oracle.GUARD + 1] = np.float32(3.0)
    source[2, oracle.GUARD + 3] = np.float32(4.0)

    status, _, out_b = bridge.execute(source, green, width, phase=0)
    assert status == oracle.OK
    expected_a, expected_b = oracle.reference(source, green, width, phase=0)
    _assert_same(out_b[:width], expected_b,
                 "diagonal association differs from independent oracle")
    assert out_b[2] == np.float32(0.125)

    # The rejected left-associated sum would be 7, and its clamped candidate
    # would be 1, making this probe distinguish the recovered grouping.
    left_associated = oracle._add(
        oracle._add(oracle._add(np.float32(1.0e20), np.float32(-1.0e20)),
                    np.float32(3.0)), np.float32(4.0))
    wrong = oracle._clamp01(oracle._fma(0.25, left_associated, 0.125))
    assert wrong == np.float32(1.0)
    assert out_b[2] != wrong


def test_nan_and_infinity_classification_matches_oracle(bridge):
    width = 8
    source = oracle.make_rows(992, width, width + 8)
    green = oracle.make_rows(993, width, width + 12)
    source[1, oracle.GUARD + 1] = np.float32(np.nan)
    source[1, oracle.GUARD + 3] = np.float32(np.inf)
    source[0, oracle.GUARD + 4] = np.float32(-np.inf)
    green[0, oracle.GUARD + 4] = np.float32(np.inf)
    for phase in (0, 1):
        status, out_a, out_b = bridge.execute(source, green, width, phase)
        assert status == oracle.OK
        expected_a, expected_b = oracle.reference(source, green, width, phase)
        _assert_same(out_a[:width], expected_a, "NaN/Inf A classification differs")
        _assert_same(out_b[:width], expected_b, "NaN/Inf B classification differs")


def test_rejections_leave_output_sentinels_unchanged(bridge):
    width = 8
    source = oracle.make_rows(61, width, 16)
    green = oracle.make_rows(62, width, 20)
    out_a = oracle.aligned_float_array(width + 4)
    out_b = oracle.aligned_float_array(width + 4)
    source_center = source[1, oracle.GUARD:]
    green_center = green[1, oracle.GUARD:]
    pointer = ctypes.POINTER(ctypes.c_float)

    def rejected(expected, source_ptr, green_ptr, source_stride, green_stride,
                 call_width, phase, out_a_ptr=None, out_b_ptr=None):
        out_a.fill(np.float32(123.0))
        out_b.fill(np.float32(456.0))
        status = bridge.run(
            source_ptr, green_ptr, source_stride, green_stride, call_width,
            phase,
            out_a.ctypes.data_as(pointer) if out_a_ptr is None else out_a_ptr,
            out_b.ctypes.data_as(pointer) if out_b_ptr is None else out_b_ptr,
        )
        assert status == expected
        assert np.all(out_a == np.float32(123.0))
        assert np.all(out_b == np.float32(456.0))

    rejected(oracle.INVALID_ARGUMENT, None, oracle._pointer(green_center), 16,
             20, width, 0)
    rejected(oracle.INVALID_DIMENSIONS, oracle._pointer(source_center),
             oracle._pointer(green_center), 16, 20, 3, 0)
    rejected(oracle.INVALID_PHASE, oracle._pointer(source_center),
             oracle._pointer(green_center), 16, 20, width, 2)

    unaligned_source = source[1, oracle.GUARD + 1:]
    rejected(oracle.UNSUPPORTED_ALIGNMENT, oracle._pointer(unaligned_source, False),
             oracle._pointer(green_center), 16, 20, width, 0)
    unaligned_out = out_a[1:]
    rejected(oracle.UNSUPPORTED_ALIGNMENT, oracle._pointer(source_center),
             oracle._pointer(green_center), 16, 20, width, 0,
             out_a_ptr=oracle._pointer(unaligned_out, False))

    max_ptrdiff_samples = int(np.iinfo(np.intp).max) // np.dtype(np.float32).itemsize
    unrepresentable_stride = max_ptrdiff_samples + 1
    rejected(oracle.INVALID_DIMENSIONS, oracle._pointer(source_center),
             oracle._pointer(green_center), unrepresentable_stride, 20,
             width, 0)
