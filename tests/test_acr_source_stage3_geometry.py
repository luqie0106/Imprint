"""Focused tests for the finite Stage3 source geometry subdomain."""

from __future__ import annotations

import ctypes as C
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools" / "acr_source_stage3_geometry_bridge.cpp"
GEOMETRY_SOURCE = NATIVE / "src" / "acr_source_stage3_geometry.cpp"
RATIONAL_SOURCE = NATIVE / "src" / "acr_source_rational.cpp"

OK = 0
NON_FINITE_SCALE = 1
RATIONAL_OVERFLOW = 2
INVALID_BRIDGE_ARGUMENT = 3


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.function = self.library.acr_source_stage3_geometry_bridge
        self.function.argtypes = [
            C.POINTER(C.c_uint32),
            C.POINTER(C.c_double),
            C.POINTER(C.c_uint32),
            C.POINTER(C.c_double),
        ]
        self.function.restype = C.c_int


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the focused bridge"
    assert BRIDGE_SOURCE.is_file()
    assert GEOMETRY_SOURCE.is_file()
    assert RATIONAL_SOURCE.is_file()

    if sys.platform == "darwin":
        suffix, shared_flags = ".dylib", ["-dynamiclib"]
    elif sys.platform == "win32":
        suffix, shared_flags = ".dll", ["-shared"]
    else:
        suffix, shared_flags = ".so", ["-shared", "-fPIC"]

    temp = tmp_path_factory.mktemp("acr_source_stage3_geometry")
    output = temp / f"libacr_source_stage3_geometry{suffix}"
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
        str(GEOMETRY_SOURCE),
        str(RATIONAL_SOURCE),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return Bridge(output)


def words(*pairs):
    assert len(pairs) == 6
    return [word for pair in pairs for word in pair]


def invoke(bridge, rational_words, scales):
    rational_input = (C.c_uint32 * 12)(*rational_words)
    scale_input = (C.c_double * 2)(*scales)
    rational_output = (C.c_uint32 * 12)(*[0xA5A5A5A5] * 12)
    scale_output = (C.c_double * 2)(123.25, -77.5)
    status = bridge.function(
        rational_input, scale_input, rational_output, scale_output)
    return status, list(rational_output), list(scale_output), rational_output, scale_output


def raw_bytes(array):
    return C.string_at(C.addressof(array), C.sizeof(array))


def test_x_only_rescales_only_x_slots_and_resets_x(bridge):
    input_words = words((1, 1), (7, 11), (10, 3), (13, 17), (1, 1), (19, 23))
    status, output_words, output_scales, _, _ = invoke(
        bridge, input_words, (1.2, 0.75))

    assert status == OK
    assert output_words == words(
        (19661, 16384), (7, 11), (4, 1), (13, 17),
        (894784853, 1073741824), (19, 23))
    assert output_scales == [1.0, 0.75]


def test_y_only_rescales_only_y_slots_and_resets_y(bridge):
    input_words = words((5, 7), (14, 21), (11, 13), (1, 1), (17, 19), (12, 4))
    status, output_words, output_scales, _, _ = invoke(
        bridge, input_words, (0.5, 3.0))

    assert status == OK
    assert output_words == words(
        (5, 7), (14, 7), (11, 13), (3, 1), (17, 19), (1, 1))
    assert output_scales == [0.5, 1.0]


def test_both_axes_and_recovered_numeric_witnesses(bridge):
    # Independent hard-coded witnesses from the recovered scale helper.
    status, output_words, output_scales, _, _ = invoke(
        bridge,
        words((1, 1), (14, 21), (10, 3), (1, 1), (1, 1), (12, 4)),
        (1.2, 3.0),
    )

    assert status == OK
    assert output_words == words(
        (19661, 16384), (14, 7), (4, 1), (3, 1),
        (894784853, 1073741824), (1, 1))
    assert output_scales == [1.0, 1.0]

    status, output_words, output_scales, _, _ = invoke(
        bridge, words((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),
        (32768.25, 1.0),
    )
    assert status == OK
    assert output_words[0:2] == [32768, 1]
    assert output_scales == [1.0, 1.0]


def test_scales_at_or_below_one_preserve_rational_words_and_signed_zero(bridge):
    input_words = words((0xFFFFFFFF, 0), (3, 7), (13, 17), (19, 23), (29, 31), (37, 41))
    negative_zero = -0.0
    status, output_words, output_scales, _, scale_output = invoke(
        bridge, input_words, (1.0, negative_zero))

    assert status == OK
    assert output_words == input_words
    assert output_scales[0] == 1.0
    assert raw_bytes(scale_output)[8:] == struct.pack("=d", negative_zero)

    status, output_words, output_scales, _, _ = invoke(
        bridge, input_words, (-2.0, 0.25))
    assert status == OK
    assert output_words == input_words
    assert output_scales == [-2.0, 0.25]


def test_zero_denominator_positive_scale_preserves_rational_pairs(bridge):
    input_words = words((123, 0), (456, 0), (7, 1), (8, 1), (9, 0), (10, 0))
    status, output_words, output_scales, _, _ = invoke(
        bridge, input_words, (2.0, 3.0))

    assert status == OK
    assert output_words == words((123, 0), (456, 0), (14, 1), (24, 1), (9, 0), (10, 0))
    assert output_scales == [1.0, 1.0]


def test_input_and_output_spans_may_alias(bridge):
    rational_words = words((1, 1), (14, 21), (10, 3), (1, 1), (1, 1), (12, 4))
    rational_values = (C.c_uint32 * 12)(*rational_words)
    scale_values = (C.c_double * 2)(1.2, 3.0)
    status = bridge.function(
        rational_values, scale_values, rational_values, scale_values)

    assert status == OK
    assert list(rational_values) == words(
        (19661, 16384), (14, 7), (4, 1), (3, 1),
        (894784853, 1073741824), (1, 1))
    assert list(scale_values) == [1.0, 1.0]


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_scale_rejects_without_output_writes(bridge, axis, bad_scale):
    scales = [2.0, 2.0]
    scales[axis] = bad_scale
    status, _, _, rational_output, scale_output = invoke(
        bridge, words((1, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)),
        scales)

    assert status == NON_FINITE_SCALE
    assert list(rational_output) == [0xA5A5A5A5] * 12
    assert raw_bytes(scale_output) == struct.pack("=dd", 123.25, -77.5)


def test_finite_scale_rational_overflow_rejects_atomically(bridge):
    status, _, _, rational_output, scale_output = invoke(
        bridge, words((1, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)),
        (sys.float_info.max, 1.0))

    assert status == RATIONAL_OVERFLOW
    assert list(rational_output) == [0xA5A5A5A5] * 12
    assert raw_bytes(scale_output) == struct.pack("=dd", 123.25, -77.5)

    # rational_30 succeeds first; rational_40 then overflows, so a partially
    # normalized candidate must still never reach either output span.
    late_failure_words = words(
        (1, 1), (2, 3), (0xFFFFFFFF, 1), (6, 7), (8, 9), (10, 11))
    status, _, _, rational_output, scale_output = invoke(
        bridge, late_failure_words, (1.2, 1.0))
    assert status == RATIONAL_OVERFLOW
    assert list(rational_output) == [0xA5A5A5A5] * 12
    assert raw_bytes(scale_output) == struct.pack("=dd", 123.25, -77.5)


def test_null_bridge_arguments_are_rejected_before_writes(bridge):
    rational_input = (C.c_uint32 * 12)(*([1] * 12))
    scale_input = (C.c_double * 2)(1.0, 1.0)
    rational_output = (C.c_uint32 * 12)(*([0xA5A5A5A5] * 12))
    scale_output = (C.c_double * 2)(123.25, -77.5)
    rational_before = raw_bytes(rational_output)
    scale_before = raw_bytes(scale_output)

    calls = [
        (None, scale_input, rational_output, scale_output),
        (rational_input, None, rational_output, scale_output),
        (rational_input, scale_input, None, scale_output),
        (rational_input, scale_input, rational_output, None),
    ]
    for arguments in calls:
        assert bridge.function(*arguments) == INVALID_BRIDGE_ARGUMENT
        assert raw_bytes(rational_output) == rational_before
        assert raw_bytes(scale_output) == scale_before


@pytest.mark.parametrize("bad_argument", ["rational_input", "scale_input", "rational_output", "scale_output"])
def test_misaligned_bridge_pointer_is_rejected_without_writes(bridge, bad_argument):
    rational_input = (C.c_uint32 * 12)(*([1] * 12))
    scale_input = (C.c_double * 2)(1.0, 1.0)
    rational_output = (C.c_uint32 * 12)(*([0xA5A5A5A5] * 12))
    scale_output = (C.c_double * 2)(123.25, -77.5)
    byte_storage = C.create_string_buffer(96)
    misaligned_address = C.addressof(byte_storage) + 1
    misaligned_u32 = C.cast(misaligned_address, C.POINTER(C.c_uint32))
    misaligned_double = C.cast(misaligned_address, C.POINTER(C.c_double))
    args = [rational_input, scale_input, rational_output, scale_output]
    if bad_argument.endswith("input") and bad_argument.startswith("rational"):
        args[0] = misaligned_u32
    elif bad_argument.endswith("input"):
        args[1] = misaligned_double
    elif bad_argument.endswith("output") and bad_argument.startswith("rational"):
        args[2] = misaligned_u32
    else:
        args[3] = misaligned_double
    rational_before = raw_bytes(rational_output)
    scale_before = raw_bytes(scale_output)

    assert bridge.function(*args) == INVALID_BRIDGE_ARGUMENT
    assert raw_bytes(rational_output) == rational_before
    assert raw_bytes(scale_output) == scale_before


def test_overlapping_output_spans_are_rejected_without_writes(bridge):
    input_words = (C.c_uint32 * 12)(*([1] * 12))
    input_scales = (C.c_double * 2)(1.0, 1.0)
    output_storage = C.create_string_buffer(64)
    base = C.addressof(output_storage)
    assert base % C.alignment(C.c_double) == 0
    rational_output = C.cast(base, C.POINTER(C.c_uint32))
    scale_output = C.cast(base + 40, C.POINTER(C.c_double))
    before = raw_bytes(output_storage)

    assert bridge.function(
        input_words, input_scales, rational_output, scale_output
    ) == INVALID_BRIDGE_ARGUMENT
    assert raw_bytes(output_storage) == before


def test_output_address_plus_span_overflow_is_rejected(bridge):
    input_words = (C.c_uint32 * 12)(*([1] * 12))
    input_scales = (C.c_double * 2)(1.0, 1.0)
    rational_output = (C.c_uint32 * 12)(*([0xA5A5A5A5] * 12))
    max_address = (1 << (C.sizeof(C.c_void_p) * 8)) - 1
    near_end = max_address - 7
    assert near_end % C.alignment(C.c_double) == 0
    invalid_scale_output = C.cast(near_end, C.POINTER(C.c_double))
    before = raw_bytes(rational_output)

    assert bridge.function(
        input_words, input_scales, rational_output, invalid_scale_output
    ) == INVALID_BRIDGE_ARGUMENT
    assert raw_bytes(rational_output) == before
