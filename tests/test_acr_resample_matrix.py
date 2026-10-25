"""Source and optional formal-object checks for ordinary matrix decomposition."""

from __future__ import annotations

import ctypes as C
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools/acr_resample_matrix_bridge.cpp"
MATRIX_SOURCE = NATIVE / "src/acr_resample_matrix.cpp"
MATRIX_HEADER = NATIVE / "src/acr_resample_matrix.hpp"
MATRIX_OBJECT = (NATIVE / "build/CMakeFiles/imprint_renderer.dir/"
                 "src/acr_resample_matrix.cpp.o")

sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import resample_matrix_oracle as oracle  # noqa: E402


SENTINEL_BITS = (
    0x3FF0000000000001, 0x8000000000000000,
    0x7FF8000000000042, 0x0000000000000000,
    0xFFFFFFFFFFFFFFFF, 0x0123456789ABCDEF,
)
DOUBLE_PTR = C.POINTER(C.c_double)
UINT64_PTR = C.POINTER(C.c_uint64)


def _compile_bridge(kind: str, output: Path) -> tuple[list[str], list[Path]]:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the matrix bridge"
    assert BRIDGE_SOURCE.is_file(), f"missing test bridge: {BRIDGE_SOURCE}"
    assert MATRIX_HEADER.is_file(), f"missing production header: {MATRIX_HEADER}"
    command = [
        compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", "-ffp-contract=off", "-fPIC",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", str(NATIVE / "src"), "-I", str(NATIVE / "include"),
        str(BRIDGE_SOURCE),
    ]
    if kind == "source":
        inputs = [MATRIX_SOURCE]
    elif kind == "formal_object":
        inputs = [MATRIX_OBJECT]
        assert MATRIX_OBJECT.is_file(), f"missing formal renderer object: {MATRIX_OBJECT}"
        assert MATRIX_OBJECT.stat().st_mtime_ns >= max(
            MATRIX_SOURCE.stat().st_mtime_ns, MATRIX_HEADER.stat().st_mtime_ns
        ), f"rebuild formal renderer: stale object {MATRIX_OBJECT.name}"
    else:
        raise AssertionError("ACR_RESAMPLE_MATRIX_BRIDGE must be source or formal_object")
    command.extend(str(path) for path in inputs)
    command.extend(["-o", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    return command, inputs


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.function = self.library.acr_test_decompose_ordinary_resample_matrix
        self.function.argtypes = [DOUBLE_PTR, UINT64_PTR]
        self.function.restype = C.c_int

    def call(self, matrix: tuple[float, ...]) -> tuple[int, tuple[int, ...]]:
        input_values = (C.c_double * 9)(*matrix)
        output_bits = (C.c_uint64 * 6)(*SENTINEL_BITS)
        status = self.function(input_values, output_bits)
        assert status != 99, "native output tuple changed on failure"
        return int(status), tuple(output_bits)


@pytest.fixture(scope="module")
def bridge(tmp_path_factory) -> Bridge:
    kind = os.environ.get("ACR_RESAMPLE_MATRIX_BRIDGE", "source")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_resample_matrix_" + kind) / ("bridge" + suffix)
    _compile_bridge(kind, output)
    return Bridge(output)


def _tuple_bits(values: tuple[float, ...]) -> tuple[int, ...]:
    return struct.unpack("=6Q", struct.pack("=6d", *values))


def _bits_as_doubles(bits: tuple[int, ...]) -> tuple[float, ...]:
    return struct.unpack("=6d", struct.pack("=6Q", *bits))


PROBES = [
    pytest.param((1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0),
                 (0.0, 1.0, 0.0, 0.0, 1.0, 0.0), id="identity"),
    pytest.param((1.0, 0.0, 7.0,
                  0.0, 1.0, -3.0,
                  0.0, 0.0, 1.0),
                 (-7.0, 1.0, 0.0, 3.0, 1.0, 0.0), id="translation"),
    pytest.param((2.0, 0.0, 0.0,
                  0.0, 4.0, 0.0,
                  0.0, 0.0, 1.0),
                 (0.0, 0.5, 0.0, 0.0, 0.25, 0.0), id="scale"),
    pytest.param((1.0, 0.0, 0.0,
                  0.25, 1.0, 0.0,
                  0.0, 0.0, 1.0),
                 (0.0, 1.0, 0.0, 0.0, 1.0, -0.25), id="shear"),
    # This dense, non-affine matrix is the inverse of an integer unimodular N.
    # Its expected tuple catches row/field swaps and H*inverse(H) in place of
    # the recovered N*inverse(H) product.
    pytest.param((156.0, -37.0, 5.0,
                  -121.0, 29.0, -4.0,
                  29.0, -7.0, 1.0),
                 (-5.0 / 11.0, 1.0 / 11.0, 2.0 / 11.0,
                  19.0, 11.0, 5.0), id="dense-non-affine-binding"),
]


@pytest.mark.parametrize(("matrix", "known"), PROBES)
def test_fixed_probes_match_hard_coded_contract(bridge: Bridge,
                                                matrix: tuple[float, ...],
                                                known: tuple[float, ...]) -> None:
    oracle_status, expected = oracle.decompose_ordinary_resample_matrix(matrix)
    assert oracle_status == oracle.ResampleMatrixStatus.OK
    assert expected is not None
    status, native_bits = bridge.call(matrix)
    assert status == int(oracle.ResampleMatrixStatus.OK)
    assert native_bits == _tuple_bits(expected)
    assert _bits_as_doubles(native_bits) == pytest.approx(known, rel=2e-15, abs=2e-15)


def test_seeded_stable_general_matrices_match_bitwise(bridge: Bridge) -> None:
    rng = random.Random(20261023)
    for sample in range(64):
        matrix = [rng.uniform(-0.35, 0.35) for _ in range(9)]
        for diagonal in range(3):
            matrix[diagonal * 3 + diagonal] += 4.0 + rng.random()
        values = tuple(matrix)
        oracle_status, expected = oracle.decompose_ordinary_resample_matrix(values)
        assert oracle_status == oracle.ResampleMatrixStatus.OK, sample
        assert expected is not None
        status, native_bits = bridge.call(values)
        assert status == int(oracle_status), sample
        assert native_bits == _tuple_bits(expected), sample


@pytest.mark.parametrize("determinant", [
    math.nextafter(oracle.DETERMINANT_THRESHOLD, 0.0),
    oracle.DETERMINANT_THRESHOLD,
    math.nextafter(oracle.DETERMINANT_THRESHOLD, math.inf),
])
def test_determinant_threshold_is_strict(bridge: Bridge, determinant: float) -> None:
    matrix = (1.0, 0.0, 0.0,
              0.0, determinant, 0.0,
              0.0, 0.0, 1.0)
    oracle_status, expected = oracle.decompose_ordinary_resample_matrix(matrix)
    status, native_bits = bridge.call(matrix)
    assert status == int(oracle_status)
    if expected is None:
        assert oracle_status == oracle.ResampleMatrixStatus.SINGULAR_MATRIX
        assert native_bits == SENTINEL_BITS
    else:
        assert oracle_status == oracle.ResampleMatrixStatus.OK
        assert native_bits == _tuple_bits(expected)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf],
                         ids=["nan", "positive-infinity", "negative-infinity"])
def test_non_finite_input_is_rejected_atomically(bridge: Bridge,
                                                 bad_value: float) -> None:
    matrix = (bad_value, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0)
    oracle_status, expected = oracle.decompose_ordinary_resample_matrix(matrix)
    status, native_bits = bridge.call(matrix)
    assert oracle_status == oracle.ResampleMatrixStatus.NON_FINITE_INPUT
    assert expected is None
    assert status == int(oracle_status)
    assert native_bits == SENTINEL_BITS


@pytest.mark.parametrize(("matrix", "expected_status"), [
    ((1.0, 2.0, 3.0,
      2.0, 4.0, 6.0,
      0.0, 0.0, 1.0), oracle.ResampleMatrixStatus.SINGULAR_MATRIX),
    # M is invertible, but its inverse has N11=0, so the second inverse fails.
    ((1.0, 0.0, 0.0,
      0.0, 0.0, 1.0,
      0.0, 1.0, 0.0), oracle.ResampleMatrixStatus.SINGULAR_MATRIX),
    # The determinant is finite and accepted; the later C22 fused cofactor
    # overflows for these finite inputs and must produce non_finite_result.
    ((1.0e200, 0.0, 0.0,
      0.0, 1.0e200, 0.0,
      0.0, 0.0, 1.0e-200), oracle.ResampleMatrixStatus.NON_FINITE_RESULT),
], ids=["singular-first-inverse", "singular-second-inverse",
       "finite-input-cofactor-overflow"])
def test_failures_leave_native_tuple_bytes_unchanged(
        bridge: Bridge, matrix: tuple[float, ...],
        expected_status: oracle.ResampleMatrixStatus) -> None:
    oracle_status, expected = oracle.decompose_ordinary_resample_matrix(matrix)
    status, native_bits = bridge.call(matrix)
    assert oracle_status == expected_status
    assert expected is None
    assert status == int(expected_status)
    assert status != 99, "native output tuple changed on failure"
    assert native_bits == SENTINEL_BITS


def test_signed_zero_bits_match_the_independent_oracle(bridge: Bridge) -> None:
    matrix = (-1.0, -0.0, 0.0,
              -0.0, 1.0, -0.0,
              0.0, -0.0, 1.0)
    oracle_status, expected = oracle.decompose_ordinary_resample_matrix(matrix)
    status, native_bits = bridge.call(matrix)
    assert oracle_status == oracle.ResampleMatrixStatus.OK
    assert expected is not None
    assert status == int(oracle_status)
    assert native_bits == _tuple_bits(expected)
    expected_zero_signs = [bits >> 63 for bits, value in zip(
        native_bits, _bits_as_doubles(native_bits)) if value == 0.0]
    oracle_zero_signs = [math.copysign(1.0, value) < 0.0 for value in expected
                         if value == 0.0]
    assert expected_zero_signs
    assert expected_zero_signs == [int(negative) for negative in oracle_zero_signs]
    assert any(expected_zero_signs), "probe should exercise negative-zero results"
