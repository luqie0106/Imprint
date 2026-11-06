"""Focused tests for the finite-scalar ID 130 indexed-loop subdomain."""

from __future__ import annotations

import ctypes as C
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_control130_bridge.cpp"
CONTROL_SOURCE = NATIVE / "src" / "acr_profile_gain_control130.cpp"

OK = 0
NON_FINITE_SCALAR = 1
MIN_PROFILE_VERSION = 0x05070001


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.function = self.library.acr_profile_gain_control130_bridge
        self.function.argtypes = [
            C.c_int32,
            C.c_int32,
            C.c_uint32,
            C.c_double,
            C.POINTER(C.c_int32),
            C.POINTER(C.c_uint32),
            C.POINTER(C.c_double),
        ]
        self.function.restype = C.c_int

    def call(self, current: int, profile_default: int, version: int,
             scalar: float, *, initial=(0x13579, 0x2468, 17.25)):
        value = C.c_int32(initial[0])
        wrote = C.c_uint32(initial[1])
        blended = C.c_double(initial[2])
        status = self.function(
            current, profile_default, version, scalar,
            C.byref(value), C.byref(wrote), C.byref(blended),
        )
        return status, value.value, wrote.value, blended.value


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the focused bridge"
    assert BRIDGE_SOURCE.is_file()
    assert CONTROL_SOURCE.is_file()

    if sys.platform == "darwin":
        suffix, shared_flags = ".dylib", ["-dynamiclib"]
    elif sys.platform == "win32":
        suffix, shared_flags = ".dll", ["-shared"]
    else:
        suffix, shared_flags = ".so", ["-shared", "-fPIC"]

    tmp_path = tmp_path_factory.mktemp("profile_gain_control130")
    output = tmp_path / f"libprofile_gain_control130{suffix}"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fno-fast-math",
        "-ffp-contract=off",
        *shared_flags,
        "-I",
        str(NATIVE / "src"),
        str(BRIDGE_SOURCE),
        str(CONTROL_SOURCE),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return Bridge(output)


def test_instruction_witness_values_and_caller_rounding(bridge):
    status, value, wrote, blended = bridge.call(
        100, 100, MIN_PROFILE_VERSION, 1.0)
    assert status == OK
    assert wrote == 1
    assert blended == 166.66666666666666
    assert value == 167

    status, value, wrote, blended = bridge.call(
        100, 100, MIN_PROFILE_VERSION, 2.0)
    assert (status, value, wrote, blended) == (OK, 180, 1, 180.0)


def test_version_and_profile_default_gates(bridge):
    for version in (0, MIN_PROFILE_VERSION - 1, 0x05070000):
        status, value, wrote, blended = bridge.call(
            137, 100, version, 1.0)
        assert (status, value, wrote, blended) == (OK, 137, 0, 137.0)

    # The comparison is unsigned; the all-ones version passes.
    status, value, wrote, blended = bridge.call(
        100, 100, 0xFFFFFFFF, 1.0)
    assert (status, value, wrote, blended) == (
        OK, 167, 1, 166.66666666666666)

    for profile_default in (-999999, -1, 201, 2**31 - 1):
        status, value, wrote, blended = bridge.call(
            -37, profile_default, MIN_PROFILE_VERSION, 1.0)
        assert (status, value, wrote, blended) == (OK, -37, 0, -37.0)


def test_zero_default_early_return_and_int32_current_edges(bridge):
    status, value, wrote, blended = bridge.call(
        20, 0, MIN_PROFILE_VERSION, 1.0)
    assert (status, value, wrote, blended) == (OK, 20, 1, 20.0)

    int32_min = -(2**31)
    int32_max = 2**31 - 1
    status, value, wrote, blended = bridge.call(
        int32_min, 0, MIN_PROFILE_VERSION, 1.0)
    assert (status, value, wrote, blended) == (
        OK, 0, 1, float(int32_min))
    status, value, wrote, blended = bridge.call(
        int32_max, 0, MIN_PROFILE_VERSION, 1.0)
    assert (status, value, wrote, blended) == (
        OK, 200, 1, float(int32_max))

    # Current C may equal the sentinel; the direct loop treats it as a value.
    status, value, wrote, blended = bridge.call(
        -999999, 100, MIN_PROFILE_VERSION, 1.0)
    assert (status, value, wrote, blended) == (OK, 100, 1, 100.0)

    status, value, wrote, blended = bridge.call(
        int32_min, 100, MIN_PROFILE_VERSION, 1.0)
    assert (status, value, wrote, blended) == (OK, 100, 1, 100.0)
    status, value, wrote, blended = bridge.call(
        int32_max, 100, MIN_PROFILE_VERSION, 1.0)
    assert (status, value, wrote, blended) == (OK, 200, 1, 200.0)


@pytest.mark.parametrize(
    ("scalar", "expected"),
    [
        (-2.0, 100.0),
        (-0.0, 100.0),
        (0.0, 100.0),
        (0.5, 140.0),
        (1.0e-300, 100.0),
        (1.0, 166.66666666666666),
        (2.0, 180.0),
        (sys.float_info.max, 200.0),
    ],
)
def test_finite_scalar_piecewise_branches(bridge, scalar, expected):
    status, value, wrote, blended = bridge.call(
        100, 100, MIN_PROFILE_VERSION, scalar)
    assert status == OK
    assert wrote == 1
    assert blended == expected
    assert value == math.trunc(blended + (0.5 if blended > 0 else -0.5))


def test_zero_and_full_current_normalization_boundaries(bridge):
    status, value, wrote, blended = bridge.call(
        0, 100, MIN_PROFILE_VERSION, 1.0)
    assert (status, value, wrote, blended) == (OK, 100, 1, 100.0)

    status, value, wrote, blended = bridge.call(
        200, 100, MIN_PROFILE_VERSION, 0.25)
    assert (status, value, wrote, blended) == (OK, 200, 1, 200.0)

    status, value, wrote, blended = bridge.call(
        100, 200, MIN_PROFILE_VERSION, 1.0)
    assert (status, value, wrote, blended) == (OK, 200, 1, 200.0)


@pytest.mark.parametrize("scalar", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_scalar_rejects_without_changing_result(bridge, scalar):
    initial = (-1234567, 0xDEADBEEF, -91.25)
    status, value, wrote, blended = bridge.call(
        100, 100, MIN_PROFILE_VERSION, scalar, initial=initial)
    assert status == NON_FINITE_SCALAR
    assert (value, wrote, blended) == initial


def test_non_finite_scalar_is_rejected_before_no_write_gates(bridge):
    initial = (41, 0xABCD, 9.5)
    status, value, wrote, blended = bridge.call(
        -4, 201, 0, float("nan"), initial=initial)
    assert status == NON_FINITE_SCALAR
    assert (value, wrote, blended) == initial


