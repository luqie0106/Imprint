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
SOURCE = NATIVE / "src/acr_bayer_gain.cpp"
HEADER = SOURCE.with_suffix(".hpp")
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_bayer_gain.cpp.o"
BRIDGE = NATIVE / "tools/acr_bayer_gain_bridge.cpp"
spec = importlib.util.spec_from_file_location(
    "neutral_to_gains_oracle", ROOT / "tests/acr_support/neutral_to_gains_oracle.py"
)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
D = C.POINTER(C.c_double)
S = C.c_size_t
F = C.POINTER(C.c_float)


def ptr(array):
    return array.ctypes.data_as(D)


def pointer_at(address):
    return C.cast(C.c_void_p(address), D)


class Bridge:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        self.convert = self.lib.acr_test_gain_from_neutral
        self.convert.argtypes = [D, S, D, S]
        self.convert.restype = C.c_int
        self.prepare_source_floor = self.lib.acr_test_gain_prepare_source_floor
        self.prepare_source_floor.argtypes = [D, S, C.c_uint16, C.c_int, F, C.POINTER(C.c_int)]
        self.prepare_source_floor.restype = C.c_int

    def run(self, neutral, count, out=None, out_count=3):
        if out is None:
            out = np.full(3, -913.25, dtype=np.float64)
        neutral_ptr = None if neutral is None else ptr(neutral)
        return self.convert(neutral_ptr, count, ptr(out), out_count), out

    def run_source_floor(self, gains, source_floor, inverse=0, count=3, out=None, clip=None):
        if out is None:
            out = np.full(6, -913.25, dtype=np.float32)
        if clip is None:
            clip = C.c_int(-31)
        gains_ptr = None if gains is None else ptr(gains)
        status = self.prepare_source_floor(
            gains_ptr, count, source_floor, inverse, out.ctypes.data_as(F), C.byref(clip)
        )
        return status, out, clip


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    inputs = [SOURCE]
    if request.param == "official_object":
        assert SOURCE.exists() and HEADER.exists(), "missing neutral-to-gain source/header"
        assert OBJECT.exists(), "rebuild renderer: missing official neutral-to-gain object"
        assert OBJECT.stat().st_mtime_ns >= max(
            SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns
        ), "rebuild renderer: stale official neutral-to-gain object"
        inputs = [OBJECT]

    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "C++ compiler required for source/object bridge fixture"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("neutral_to_gains_" + request.param) / (
        "libneutral" + suffix
    )
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I",
        str(NATIVE / "src"),
        str(BRIDGE),
        *(str(path) for path in inputs),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return Bridge(output)


def same_bits(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float64).view(np.uint64),
        np.asarray(expected, dtype=np.float64).view(np.uint64),
    )


def same_float_bits(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32).view(np.uint32),
        np.asarray(expected, dtype=np.float32).view(np.uint32),
    )


def assert_converts(bridge, neutral, count=3):
    original = None if neutral is None else np.asarray(neutral, dtype=np.float64).copy()
    status, actual = bridge.run(neutral, count)
    assert status == 0
    same_bits(actual, oracle.reference(neutral, count))
    if original is not None:
        same_bits(neutral, original)
    return actual


def test_default_neutral_bits_and_count_fallbacks(bridge):
    assert oracle.DEFAULT_NEUTRAL.view(np.uint64).tolist() == [
        0x3FD999999999999A,
        0x3FF0000000000000,
        0x3FE3333333333333,
    ]
    assert_converts(bridge, None, 0)
    assert_converts(bridge, None, 4)
    # A finite-count mismatch ignores the supplied values as well.
    ignored = np.asarray([np.nan, -17.0, np.inf, 3.0], dtype=np.float64)
    assert_converts(bridge, ignored, 4)


@pytest.mark.parametrize(
    "neutral",
    ([0.0, 0.8, 0.5], [-3.0, 0.8, 0.5], [0.7, 0.0, 0.5]),
)
def test_finite_nonpositive_neutral_uses_default(bridge, neutral):
    assert_converts(bridge, np.asarray(neutral, dtype=np.float64))


def test_normalized_threshold_and_adjacent_float64_values(bridge):
    threshold = np.float64(0.125)
    below = np.nextafter(threshold, np.float64(-np.inf))
    above = np.nextafter(threshold, np.float64(np.inf))

    below_result = assert_converts(bridge, np.asarray([below, 1.0, 0.5]))
    same_bits(below_result, oracle.reference())
    at_result = assert_converts(bridge, np.asarray([threshold, 1.0, 0.5]))
    same_bits(at_result, np.asarray([1.0, threshold, 0.25]))
    above_result = assert_converts(bridge, np.asarray([above, 1.0, 0.5]))
    same_bits(above_result, np.asarray([1.0, above, np.float64(above / 0.5)]))


def test_random_neutrals_across_scales_match_binary64_sequence(bridge):
    rng = np.random.default_rng(2026101907)
    examples = []
    for exponent in (-250, -100, 0, 100, 250):
        for _ in range(30):
            scale = np.float64(10.0**exponent) * np.float64(rng.uniform(0.1, 9.0))
            ratios = rng.uniform(0.125, 1.0, size=3).astype(np.float64)
            ratios[int(rng.integers(0, 3))] = 1.0
            examples.append(scale * ratios)
    for _ in range(120):
        exponents = rng.uniform(-250.0, 250.0, size=3)
        examples.append(np.power(np.float64(10.0), exponents))

    for neutral in examples:
        assert_converts(bridge, np.asarray(neutral, dtype=np.float64))


@pytest.mark.parametrize(
    "neutral",
    (
        (0.446625, 1.0, 0.5235839999999999),
        (0.43835599999999997, 1.0, 0.5203249999999999),
    ),
)
def test_prepared_dji_native_neutrals(bridge, neutral):
    assert_converts(bridge, np.asarray(neutral, dtype=np.float64))


def test_preserves_reciprocal_product_divide_rounding(bridge):
    neutral = np.asarray(
        [
            float.fromhex("0x1.e14a462d31271p-265"),
            float.fromhex("0x1.13e9c837a7cccp-262"),
            float.fromhex("0x1.8b13d5fd48f12p-263"),
        ],
        dtype=np.float64,
    )
    actual = assert_converts(bridge, neutral)
    expected_bits = [
        0x3FF0000000000000,
        0x3FCBE8DF0DC9F2B3,
        0x3FD37DD12D94E6B9,
    ]
    assert actual.view(np.uint64).tolist() == expected_bits

    simplified = np.asarray(
        [np.min(neutral) / value for value in neutral], dtype=np.float64
    )
    assert simplified.view(np.uint64).tolist() == [
        0x3FF0000000000000,
        0x3FCBE8DF0DC9F2B2,
        0x3FD37DD12D94E6B8,
    ]
    assert np.all(np.abs(actual.view(np.int64) - simplified.view(np.int64)) <= 1)


def test_invalid_calls_are_atomic(bridge):
    sentinel = np.asarray([-91.5, 123.25, -0.0], dtype=np.float64)
    before = sentinel.copy()

    assert bridge.convert(None, 3, ptr(sentinel), 3) == 1
    same_bits(sentinel, before)
    assert bridge.convert(ptr(np.asarray([0.4, 0.8, 0.6])), 3, ptr(sentinel), 2) == 2
    same_bits(sentinel, before)

    for bad in (
        [np.nan, 1.0, 0.5],
        [np.inf, 1.0, 0.5],
        [-np.inf, 1.0, 0.5],
        [
            np.nextafter(np.float64(0.0), np.float64(1.0)),
            np.nextafter(np.float64(0.0), np.float64(1.0)) * 2.0,
            np.nextafter(np.float64(0.0), np.float64(1.0)) * 3.0,
        ],
    ):
        values = np.asarray(bad, dtype=np.float64)
        saved = values.copy()
        assert bridge.convert(ptr(values), 3, ptr(sentinel), 3) == 3
        same_bits(sentinel, before)
        same_bits(values, saved)


def test_null_output_misalignment_overlap_and_address_wrap(bridge):
    valid = np.asarray([0.4, 0.8, 0.6], dtype=np.float64)
    output = np.full(3, -77.0, dtype=np.float64)
    saved = output.copy()
    assert bridge.convert(ptr(valid), 3, None, 3) == 1
    same_bits(output, saved)

    raw = C.create_string_buffer(32)
    misaligned = pointer_at(C.addressof(raw) + 1)
    assert bridge.convert(misaligned, 3, ptr(output), 3) == 8
    same_bits(output, saved)

    aliased = np.asarray([0.4, 0.8, 0.6], dtype=np.float64)
    alias_before = aliased.copy()
    assert bridge.convert(ptr(aliased), 3, ptr(aliased), 3) == 11
    same_bits(aliased, alias_before)

    max_address = (1 << (C.sizeof(C.c_void_p) * 8)) - 1
    wrapping_input = pointer_at(max_address - 7)
    assert bridge.convert(wrapping_input, 3, ptr(output), 3) == 10
    same_bits(output, saved)

    wrapping_output = pointer_at(max_address - 7)
    assert bridge.convert(ptr(valid), 3, wrapping_output, 3) == 10
    same_bits(output, saved)


@pytest.mark.parametrize("source_floor", (0, 1040, 3992))
@pytest.mark.parametrize("inverse", (0, 1))
def test_source_floor_adapter_matches_independent_float_sequence(
    bridge, source_floor, inverse
):
    gains = np.asarray([0.72, 1.0, 1.38], dtype=np.float64)
    saved_gains = gains.copy()
    status, actual, clip = bridge.run_source_floor(gains, source_floor, inverse)
    expected, expected_clip = oracle.source_floor_reference(
        gains, source_floor, inverse=bool(inverse)
    )
    assert status == 0
    same_float_bits(actual, expected)
    assert clip.value == expected_clip
    same_bits(gains, saved_gains)


def test_source_floor_adapter_random_reasonable_gains(bridge):
    rng = np.random.default_rng(2026101911)
    floors = (0, 1, 1040, 3992, 16383, 32768, 65534)
    for _ in range(160):
        gains = rng.uniform(0.15, 6.0, size=3).astype(np.float64)
        source_floor = int(rng.choice(floors))
        inverse = int(rng.integers(0, 2))
        status, actual, clip = bridge.run_source_floor(gains, source_floor, inverse)
        expected, expected_clip = oracle.source_floor_reference(
            gains, source_floor, inverse=bool(inverse)
        )
        assert status == 0
        same_float_bits(actual, expected)
        assert clip.value == expected_clip


def test_source_floor_65535_rejects_without_publishing_partial_output(bridge):
    gains = np.asarray([0.72, 1.0, 1.38], dtype=np.float64)
    out = np.asarray([-17.5, 2.25, -0.0, 41.0, -9.0, 3.5], dtype=np.float32)
    before = out.copy()
    clip = C.c_int(23)

    status, actual, actual_clip = bridge.run_source_floor(
        gains, 65535, inverse=1, out=out, clip=clip
    )
    assert status == 4  # source floor maps to an unsupported pivot of 1.0f
    same_float_bits(actual, before)
    assert actual_clip.value == 23


def test_source_floor_invalid_gain_and_count_reject_atomically(bridge):
    out = np.asarray([-17.5, 2.25, -0.0, 41.0, -9.0, 3.5], dtype=np.float32)
    before = out.copy()
    clip = C.c_int(-19)

    invalid = np.asarray([0.72, np.nan, 1.38], dtype=np.float64)
    status, actual, actual_clip = bridge.run_source_floor(
        invalid, 1040, out=out, clip=clip
    )
    assert status == 3
    same_float_bits(actual, before)
    assert actual_clip.value == -19

    valid = np.asarray([0.72, 1.0, 1.38], dtype=np.float64)
    status, actual, actual_clip = bridge.run_source_floor(
        valid, 1040, count=2, out=out, clip=clip
    )
    assert status == 2
    same_float_bits(actual, before)
    assert actual_clip.value == -19
