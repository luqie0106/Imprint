"""Instruction-derived tests for the ordinary ACR resample-axis helper."""

from __future__ import annotations

import ctypes as C
import math
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_resample_axis.cpp"
HEADER = NATIVE / "src/acr_resample_axis.hpp"
BRIDGE_SOURCE = NATIVE / "tools/acr_resample_axis_bridge.cpp"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_resample_axis.cpp.o"

sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import resample_axis_instruction_oracle as oracle  # noqa: E402


I32 = C.c_int32
U64 = C.c_uint64
U64_PTR = C.POINTER(U64)
OUT_WORDS = 6
OK = 0
INVALID = 1
UNSUPPORTED_SKEW = 2


def _compile_bridge(kind: str, output: Path) -> None:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the resample-axis bridge"
    assert BRIDGE_SOURCE.is_file(), f"missing C ABI bridge: {BRIDGE_SOURCE}"
    assert SOURCE.is_file() and HEADER.is_file(), "missing axis source/header"
    if kind == "source":
        inputs = [SOURCE]
    else:
        assert OBJECT.is_file(), f"build renderer first: missing official object {OBJECT.name}"
        assert OBJECT.stat().st_mtime_ns >= max(SOURCE.stat().st_mtime_ns,
                                                HEADER.stat().st_mtime_ns), (
            f"rebuild renderer: stale official object {OBJECT.name}"
        )
        inputs = [OBJECT]
    command = [
        compiler, "-std=c++17", "-O2", "-fPIC", "-fno-fast-math", "-ffp-contract=off",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", str(NATIVE / "src"), "-I", str(NATIVE / "include"),
        str(BRIDGE_SOURCE), *(str(path) for path in inputs), "-o", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.prepare_fn = self.library.acr_test_prepare_resample_axis
        self.prepare_fn.argtypes = [C.c_double, C.c_double, C.c_double,
                                    I32, I32, I32, U64_PTR]
        self.prepare_fn.restype = C.c_int

    def raw(self, translation, along_scale, cross_scale,
            bound_start, bound_end, half_taps, output):
        return self.prepare_fn(translation, along_scale, cross_scale,
                               bound_start, bound_end, half_taps, output)

    def prepare(self, translation, along_scale, cross_scale,
                bound_start, bound_end, half_taps):
        output = (U64 * OUT_WORDS)(*[0xC0DEC0DEC0DEC0DE] * OUT_WORDS)
        status = self.raw(translation, along_scale, cross_scale,
                          bound_start, bound_end, half_taps, output)
        return status, tuple(int(word) for word in output)


@pytest.fixture(scope="module", params=("source", "official_object"),
                ids=("source", "official"))
def bridge(request, tmp_path_factory):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_resample_axis_" + request.param) / (
        "resample_axis" + suffix
    )
    _compile_bridge(request.param, output)
    return Bridge(output)


def _assert_matches_literal_and_oracle(bridge, args, literal):
    status, expected = oracle.prepare(*args)
    assert status == OK
    assert expected == literal
    actual_status, actual = bridge.prepare(*args)
    assert actual_status == OK
    assert actual == literal


def test_oracle_is_pinned_to_the_three_raw_instruction_captures():
    assert oracle.REFERENCE_ID == "acr-00d36a50-ordinary-resample-axis-arm64-q32-v1"
    assert oracle.SOURCE_SLICE_SHA256 == (
        "09978d95dfe76b18495fea9550b051a89f4598ab0fd4734d69d22d330496f40a"
    )
    assert oracle.RAW_CAPTURE_SHA256 == (
        "df698c3d92fd69f6b0f59c8004b66f3c1f0d6415ba6c1b2659a4529a3621dd4c",
        "a62c29c2be74bf94bc41ea8b758282ce0c13073b22caf887b8aabc6dbd3fae21",
        "e164ba12b22e1e7ce3f7490e4f488c4c9959fb93e67659029ba8e565f694087e",
    )


@pytest.mark.parametrize(("value", "expected"), [
    (2.0 ** -33, 1),
    (-(2.0 ** -33), -1),
    (3.0 * (2.0 ** -33), 2),
    (-3.0 * (2.0 ** -33), -2),
    (math.nextafter(2.0 ** -33, 0.0), 1),
    (-math.nextafter(2.0 ** -33, 0.0), -1),
    (0.0, 0),
    (-0.0, 0),
    (math.ulp(0.0), 0),
    (-math.ulp(0.0), 0),
    (math.nan, 0),
    (math.inf, (1 << 63) - 1),
    (-math.inf, -(1 << 63)),
    (float(1 << 31), (1 << 63) - 1),
    (-float(1 << 31), -(1 << 63)),
    (math.nextafter(float(1 << 31), 0.0), (1 << 63) - 1024),
    (math.nextafter(-float(1 << 31), math.inf), -(1 << 63) + 1024),
    (1.0e300, (1 << 63) - 1),
    (-1.0e300, -(1 << 63)),
    (sys.float_info.max, (1 << 63) - 1),
    (-sys.float_info.max, -(1 << 63)),
])
def test_q32_matches_explicit_instruction_witnesses(value, expected):
    assert oracle.quantize_q32(value) == expected


@pytest.mark.parametrize("half_taps", [-(1 << 31), -1, 0, (1 << 31) - 1])
def test_quantized_identity_ignores_half_taps_and_bounds(bridge, half_taps):
    args = (-0.0, 1.0, -0.0, (1 << 31) - 1, -(1 << 31), half_taps)
    _assert_matches_literal_and_oracle(
        bridge, args, (oracle.KIND_IDENTITY, 0, 0, 0, 0, 0)
    )


@pytest.mark.parametrize(("translation", "half_taps", "expected_offset"), [
    (1.0, 0, 0xFFFFFFFF),
    (-1.0, -(1 << 31), 1),
    (float((1 << 31) - 1), 0, 0x80000001),
    (-float(1 << 31), (1 << 31) - 1, 0x80000000),
])
def test_exact_integer_translation_uses_wrapping_w_negation(
    bridge, translation, half_taps, expected_offset
):
    args = (translation, 1.0, 0.0, 123, -456, half_taps)
    _assert_matches_literal_and_oracle(
        bridge, args,
        (oracle.KIND_INTEGER_TRANSLATION, 0, 0, 0, 0, expected_offset),
    )


@pytest.mark.parametrize(("translation", "expected_q"), [
    (2.0 ** -33, 1),
    (-(2.0 ** -33), -1),
    (3.0 * (2.0 ** -33), 2),
    (-3.0 * (2.0 ** -33), -2),
])
def test_fractional_translation_half_ties_enter_general_sampler(
    bridge, translation, expected_q
):
    args = (translation, 0.75, 0.0, -7, 11, 2)
    expected_origin = oracle.wrap64(
        expected_q + (oracle.quantize_q32(0.75) >> 1)
        + (oracle.wrap32(1 - 2) << 32) - 0x7F000000
    )
    expected = (oracle.KIND_RESAMPLE, expected_origin,
                oracle.quantize_q32(0.75), oracle.wrap32(-7 - 1) << 32,
                oracle.wrap32(11 - 2) << 32, 0)
    _assert_matches_literal_and_oracle(bridge, args, expected)


def test_negative_odd_step_and_wrapped_geometry_have_literal_words(bridge):
    args = (-(2.0 ** -32), -3.0 * (2.0 ** -32), 0.0,
            (1 << 31) - 1, -(1 << 31), (1 << 31) - 1)
    literal = (
        oracle.KIND_RESAMPLE,
        0x8000000180FFFFFD,
        0xFFFFFFFFFFFFFFFD,
        0x0000000100000000,
        0x0000000100000000,
        0,
    )
    _assert_matches_literal_and_oracle(bridge, args, literal)


def test_positive_extreme_origin_additions_wrap_as_x_registers(bridge):
    args = (float((1 << 31) - 1), float((1 << 31) - 1), 0.0,
            -13, 17, 1)
    literal = (
        oracle.KIND_RESAMPLE,
        0xBFFFFFFE01000000,
        0x7FFFFFFF00000000,
        0xFFFFFFF300000000,
        0x0000001000000000,
        0,
    )
    _assert_matches_literal_and_oracle(bridge, args, literal)


@pytest.mark.parametrize(("cross_scale", "expected_status"), [
    (2.0 ** -34, OK),
    (-(2.0 ** -34), OK),
    (2.0 ** -33, UNSUPPORTED_SKEW),
    (-(2.0 ** -33), UNSUPPORTED_SKEW),
    (math.nan, OK),
    (math.inf, UNSUPPORTED_SKEW),
    (-math.inf, UNSUPPORTED_SKEW),
])
def test_skew_is_decided_after_q32_quantization(bridge, cross_scale, expected_status):
    args = (0.125, 0.5, cross_scale, 3, 19, 2)
    expected_status_from_oracle, expected_words = oracle.prepare(*args)
    assert expected_status_from_oracle == expected_status
    sentinel = (U64 * OUT_WORDS)(*[0xA5A55A5AA5A55A5A] * OUT_WORDS)
    before = bytes(sentinel)
    status = bridge.raw(*args, sentinel)
    assert status == expected_status
    if status == OK:
        assert tuple(int(word) for word in sentinel) == expected_words
    else:
        assert bytes(sentinel) == before


@pytest.mark.parametrize(("translation", "along_scale", "expected_kind", "expected_step"), [
    (math.nan, 1.0, oracle.KIND_IDENTITY, 0),
    (math.inf, 1.0, oracle.KIND_RESAMPLE, 1 << 32),
    (-math.inf, 1.0, oracle.KIND_INTEGER_TRANSLATION, 0),
    (sys.float_info.max, 0.75, oracle.KIND_RESAMPLE, 3 << 30),
    (-sys.float_info.max, 0.75, oracle.KIND_RESAMPLE, 3 << 30),
    (0.125, sys.float_info.max, oracle.KIND_RESAMPLE, (1 << 63) - 1),
    (0.125, -sys.float_info.max, oracle.KIND_RESAMPLE, 1 << 63),
    (0.125, math.nan, oracle.KIND_RESAMPLE, 0),
    (0.125, math.inf, oracle.KIND_RESAMPLE, (1 << 63) - 1),
    (0.125, -math.inf, oracle.KIND_RESAMPLE, 1 << 63),
])
def test_nonfinite_and_subnormal_binary64_inputs_follow_fcvtzs(
    bridge, translation, along_scale, expected_kind, expected_step
):
    args = (translation, along_scale, 0.0, -13, 5, 1)
    status, words = oracle.prepare(*args)
    assert status == OK
    assert words[0] == expected_kind
    if expected_kind == oracle.KIND_RESAMPLE:
        assert words[2] == expected_step
    if translation == math.inf:
        assert words[5] == 0
    elif translation == -math.inf:
        assert words[5] == 0x80000000
    actual_status, actual = bridge.prepare(*args)
    assert actual_status == OK
    assert actual == words

    for subnormal in (math.ulp(0.0), -math.ulp(0.0)):
        assert oracle.quantize_q32(subnormal) == 0
        subnormal_args = (0.125, subnormal, 0.0, -13, 5, 1)
        actual_status, actual = bridge.prepare(*subnormal_args)
        assert actual_status == OK
        assert actual == oracle.prepare(*subnormal_args)[1]


@pytest.mark.parametrize("half_taps", [-(1 << 31), -1, 0])
def test_nonpositive_half_taps_reject_general_sampler_without_mutation(bridge, half_taps):
    args = (0.125, 0.5, 0.0, -3, 9, half_taps)
    status, words = oracle.prepare(*args)
    assert status == INVALID and words is None
    sentinel = (U64 * OUT_WORDS)(*[0x3141592653589793] * OUT_WORDS)
    before = bytes(sentinel)
    assert bridge.raw(*args, sentinel) == INVALID
    assert bytes(sentinel) == before


def test_nonzero_quantized_skew_rejects_without_mutation(bridge):
    args = (0.125, 0.5, 2.0 ** -33, 0, 10, 2)
    sentinel = (U64 * OUT_WORDS)(*[0xFEEDFACECAFEBEEF] * OUT_WORDS)
    before = bytes(sentinel)
    assert bridge.raw(*args, sentinel) == UNSUPPORTED_SKEW
    assert bytes(sentinel) == before


def test_null_output_and_invalid_tap_count_return_invalid(bridge):
    assert bridge.raw(0.0, 0.5, 0.0, 0, 10, 1, None) == INVALID
    assert bridge.raw(0.125, 0.5, 0.0, 0, 10, 0, None) == INVALID


def _double_from_u64(bits: int) -> float:
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


def _signed_from_u32(bits: int) -> int:
    return bits - (1 << 32) if bits & (1 << 31) else bits


def _cross_term(rng: random.Random) -> float:
    choice = rng.randrange(10)
    if choice < 4:
        return 0.0
    if choice < 6:
        return math.copysign(2.0 ** -34, -1.0 if rng.getrandbits(1) else 1.0)
    if choice < 8:
        return math.copysign(2.0 ** -33, -1.0 if rng.getrandbits(1) else 1.0)
    if choice == 8:
        return math.nan
    return _double_from_u64(rng.getrandbits(64))


def test_seeded_raw_binary64_fuzz_matches_oracle_for_2000_cases_per_bridge(bridge):
    seed = 0xAC00D36A50
    rng = random.Random(seed)
    status_counts = {OK: 0, INVALID: 0, UNSUPPORTED_SKEW: 0}

    for index in range(2000):
        translation_bits = rng.getrandbits(64)
        scale_bits = rng.getrandbits(64)
        translation = _double_from_u64(translation_bits)
        along_scale = _double_from_u64(scale_bits)
        cross_scale = _cross_term(rng)
        bound_start = _signed_from_u32(rng.getrandbits(32))
        bound_end = _signed_from_u32(rng.getrandbits(32))
        tap_class = index % 3
        if tap_class == 0:
            half_taps = rng.randrange(1, 1 << 31)
        elif tap_class == 1:
            half_taps = 0
        else:
            half_taps = -rng.randrange(1, 1 << 31)

        args = (translation, along_scale, cross_scale,
                bound_start, bound_end, half_taps)
        expected_status, expected_words = oracle.prepare(*args)
        sentinel_seed = (0x9E3779B97F4A7C15 ^ index) & ((1 << 64) - 1)
        sentinel_words = tuple(
            (sentinel_seed + word_index * 0x100000001B3) & ((1 << 64) - 1)
            for word_index in range(OUT_WORDS)
        )
        output = (U64 * OUT_WORDS)(*sentinel_words)
        before = bytes(output)
        status = bridge.raw(*args, output)
        detail = (
            f"seed={seed} case={index} T=0x{translation_bits:016x} "
            f"S=0x{scale_bits:016x} B={bound_start} E={bound_end} N={half_taps}"
        )

        assert status == expected_status, detail
        status_counts[status] += 1
        if status == OK:
            assert tuple(int(word) for word in output) == expected_words, detail
        else:
            assert bytes(output) == before, detail

    assert all(status_counts[status] > 0 for status in (OK, INVALID, UNSUPPORTED_SKEW))
