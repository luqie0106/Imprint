"""Tests for the local finite and IEEE-special b83674 scalar quantizer."""

from __future__ import annotations

import ctypes as C
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_scalar_bridge.cpp"
SCALAR_SOURCE = NATIVE / "src" / "acr_profile_gain_scalar.cpp"

OK = 0
SIGNALING_NAN = 1
BRIDGE_INVALID_OUTPUT = 2


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.function = self.library.acr_profile_gain_scalar_bridge
        self.function.argtypes = [C.c_double, C.POINTER(C.c_double)]
        self.function.restype = C.c_int

    @staticmethod
    def double_from_bits(bits: int) -> C.c_double:
        return C.c_double.from_buffer_copy(
            bits.to_bytes(C.sizeof(C.c_double), byteorder=sys.byteorder))

    @staticmethod
    def bits(value: C.c_double) -> int:
        return int.from_bytes(C.string_at(C.byref(value), C.sizeof(value)),
                              byteorder=sys.byteorder)

    def call(self, input_value: C.c_double, initial_bits: int):
        output = self.double_from_bits(initial_bits)
        status = self.function(input_value, C.byref(output))
        return status, output


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the focused bridge"
    assert BRIDGE_SOURCE.is_file()
    assert SCALAR_SOURCE.is_file()

    if sys.platform == "darwin":
        suffix, shared_flags = ".dylib", ["-dynamiclib"]
    elif sys.platform == "win32":
        suffix, shared_flags = ".dll", ["-shared"]
    else:
        suffix, shared_flags = ".so", ["-shared", "-fPIC"]

    temp = tmp_path_factory.mktemp("profile_gain_scalar")
    output = temp / f"libprofile_gain_scalar{suffix}"
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
        str(SCALAR_SOURCE),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return Bridge(output)


def _oracle_from_bits(bits: int):
    exponent = bits & 0x7FF0000000000000
    fraction = bits & 0x000FFFFFFFFFFFFF
    negative = (bits & 0x8000000000000000) != 0

    if exponent == 0x7FF0000000000000:
        if fraction == 0:
            return 0.0 if negative else 2.0
        quiet = (fraction & 0x0008000000000000) != 0
        return 2.0 if quiet else None

    scalar = struct.unpack(
        ">d", bits.to_bytes(8, byteorder="big"))[0]
    capped = scalar if scalar < 2.0 else 2.0
    scaled = 0.0 if capped < 0.0 else capped * 100.0
    shifted = scaled + (0.5 if scaled > 0.0 else -0.5)
    quantized = math.trunc(shifted)
    return float(quantized) * 0.01


def _bits_from_float(value: float) -> int:
    return int.from_bytes(struct.pack(">d", value), byteorder="big")


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (0.0, 0.0),
        (-0.0, 0.0),
        (-0.25, 0.0),
        (0.005, 0.01),
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 2.0),
        (float.fromhex("0x0.0000000000001p-1022"), 0.0),
        (sys.float_info.max, 2.0),
        (-sys.float_info.max, 0.0),
        (float("inf"), 2.0),
        (-float("inf"), 0.0),
    ],
)
def test_finite_and_infinity_witnesses(bridge, input_value, expected):
    initial_bits = 0x405ED00000000000  # 123.25 sentinel
    input_bits = _bits_from_float(input_value)
    expected_bits = _bits_from_float(expected)
    result = Bridge.double_from_bits(initial_bits)
    status = bridge.function(Bridge.double_from_bits(input_bits), C.byref(result))
    assert status == OK
    assert Bridge.bits(result) == expected_bits


def test_half_step_neighbors_match_independent_oracle(bridge):
    center = 0.005
    values = [
        math.nextafter(center, -math.inf),
        center,
        math.nextafter(center, math.inf),
        math.nextafter(0.015, -math.inf),
        0.015,
        math.nextafter(0.015, math.inf),
    ]
    for value in values:
        bits = _bits_from_float(value)
        expected = _oracle_from_bits(bits)
        result = Bridge.double_from_bits(0x405ED00000000000)
        status = bridge.function(Bridge.double_from_bits(bits), C.byref(result))
        assert status == OK
        assert Bridge.bits(result) == _bits_from_float(expected)


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        (0x7FF8000000000001, 2.0),
        (0xFFF8000000001234, 2.0),
    ],
)
def test_quiet_nan_payloads_follow_numeric_fminnm_branch(bridge, bits, expected):
    result = Bridge.double_from_bits(0x405ED00000000000)
    status = bridge.function(Bridge.double_from_bits(bits), C.byref(result))
    assert status == OK
    assert Bridge.bits(result) == _bits_from_float(expected)


@pytest.mark.parametrize(
    "bits",
    [0x7FF0000000000001, 0xFFF0000000000001],
)
def test_signaling_nan_payloads_fail_without_changing_output(bridge, bits):
    sentinel_bits = 0x405ED00000000000
    result = Bridge.double_from_bits(sentinel_bits)
    status = bridge.function(Bridge.double_from_bits(bits), C.byref(result))
    assert status == SIGNALING_NAN
    assert Bridge.bits(result) == sentinel_bits


def test_null_output_is_rejected_without_a_write(bridge):
    status = bridge.function(Bridge.double_from_bits(_bits_from_float(1.0)), None)
    assert status == BRIDGE_INVALID_OUTPUT
