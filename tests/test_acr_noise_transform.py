"""Bit tests for the scalar transform entrypoints; these do not model the LUT image path."""

import ctypes as C
import importlib.util
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_noise_transform.cpp"
HEADER = SOURCE.with_suffix(".hpp")
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_noise_transform.cpp.o"
BRIDGE = NATIVE / "tools/acr_noise_transform_bridge.cpp"
ORACLE_SPEC = importlib.util.spec_from_file_location(
    "noise_transform_oracle", ROOT / "tests/acr_support/noise_transform_oracle.py"
)
oracle = importlib.util.module_from_spec(ORACLE_SPEC)
ORACLE_SPEC.loader.exec_module(oracle)

DOUBLE = C.c_double
DOUBLE_PTR = C.POINTER(DOUBLE)
SENTINEL = -1234.5


def double_ptr(array):
    return array.ctypes.data_as(DOUBLE_PTR)


def bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


class Bridge:
    def __init__(self, path):
        self.library = C.CDLL(str(path))
        self.prepare_fn = self.library.acr_test_noise_prepare
        self.prepare_fn.argtypes = [DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE_PTR]
        self.prepare_fn.restype = C.c_int
        self.apply_fn = self.library.acr_test_noise_apply
        self.apply_fn.argtypes = [DOUBLE_PTR, DOUBLE, C.c_int, DOUBLE_PTR]
        self.apply_fn.restype = C.c_int

    def prepare(self, slope, offset, input_black, output_black):
        result = np.full(10, SENTINEL, dtype=np.float64)
        status = self.prepare_fn(
            slope, offset, input_black, output_black, double_ptr(result)
        )
        return status, result

    def apply_raw(self, prepared_pointer, sample, inverse, output_pointer):
        return self.apply_fn(prepared_pointer, sample, int(inverse), output_pointer)

    def apply(self, prepared, sample, inverse=False):
        result = np.asarray([SENTINEL], dtype=np.float64)
        status = self.apply_raw(
            double_ptr(prepared), sample, inverse, double_ptr(result)
        )
        return status, result[0]


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    inputs = [SOURCE]
    if request.param == "official_object":
        assert OBJECT.is_file(), "rebuild renderer: missing official noise-transform object"
        assert OBJECT.stat().st_mtime_ns >= max(
            SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns
        ), "rebuild renderer: stale official noise-transform object"
        inputs = [OBJECT]

    assert BRIDGE.is_file(), f"missing noise-transform C ABI bridge: {BRIDGE}"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_noise_" + request.param) / (
        "libnoise_transform" + suffix
    )
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the scalar bridge"
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


def assert_same_bits(actual, expected):
    np.testing.assert_array_equal(bits(actual), bits(expected))


def test_dji_fixture_prepared_values_have_locked_binary64_bits(bridge):
    parameters = (5.88172456e-05, 3.241287029e-08, 4100 / 65535, 0.25)
    expected = oracle.prepare(*parameters)
    status, actual = bridge.prepare(*parameters)
    assert status == 0
    assert_same_bits(actual, expected)
    assert int(bits(actual)[2]) == 0x40187CCB5341D427
    assert int(bits(actual)[3]) == 0x3F70146392CEF4BD


def test_prepare_matches_independent_binary64_oracle(bridge):
    rng = np.random.default_rng(2026101911)
    examples = [
        (1.0, 0.0, 0.0, 0.0),
        (5.88172456e-05, 3.241287029e-08, 4100 / 65535, 0.25),
        (0.25, 1e-5, 0.0, 0.5),
        (1.25, 0.04, np.nextafter(1.0, 0.0), 0.125),
    ]
    for _ in range(40):
        slope = float(10.0 ** rng.uniform(-4.0, 0.15))
        offset = float(10.0 ** rng.uniform(-12.0, -0.65))
        input_black, output_black = rng.uniform(0.0, 1.0, size=2)
        examples.append((slope, offset, float(input_black), float(output_black)))

    for parameters in examples:
        expected = oracle.prepare(*parameters)
        status, actual = bridge.prepare(*parameters)
        assert status == 0, parameters
        assert_same_bits(actual, expected)


def test_forward_and_inverse_scalar_methods_match_bit_for_bit(bridge):
    rng = np.random.default_rng(2026101912)
    examples = [
        (5.88172456e-05, 3.241287029e-08, 4100 / 65535, 0.25),
        (1.0, 0.0, 0.0, 0.0),
        (0.003, 0.0002, 0.2, 0.7),
    ]
    for _ in range(18):
        examples.append((
            float(10.0 ** rng.uniform(-4.0, 0.15)),
            float(10.0 ** rng.uniform(-12.0, -0.65)),
            float(rng.uniform(0.0, 0.9)),
            float(rng.uniform(0.0, 0.9)),
        ))

    for parameters in examples:
        status, prepared = bridge.prepare(*parameters)
        assert status == 0, parameters
        input_black, output_black = parameters[2], parameters[3]
        forward_samples = [
            -0.25,
            -0.0,
            0.0,
            np.nextafter(input_black, -np.inf),
            input_black,
            np.nextafter(input_black, np.inf),
            1.0,
            1.25,
            2.0,
        ]
        inverse_samples = [
            -0.25,
            -0.0,
            0.0,
            np.nextafter(output_black, -np.inf),
            output_black,
            np.nextafter(output_black, np.inf),
            1.0,
            1.25,
            2.0,
        ]
        for inverse, samples in ((False, forward_samples), (True, inverse_samples)):
            for sample in samples:
                expected = oracle.apply(prepared.tolist(), float(sample), inverse)
                result_status, actual = bridge.apply(prepared, float(sample), inverse)
                assert result_status == 0, (parameters, inverse, sample)
                assert_same_bits([actual], [expected])


def test_prepare_invalid_parameters_leave_output_untouched(bridge):
    null = C.cast(C.c_void_p(0), DOUBLE_PTR)
    assert bridge.prepare_fn(1.0, 1e-8, 0.1, 0.2, null) != 0

    invalid = [
        (0.0, 1e-8, 0.1, 0.2),
        (-1.0, 1e-8, 0.1, 0.2),
        (float("nan"), 1e-8, 0.1, 0.2),
        (float("inf"), 1e-8, 0.1, 0.2),
        (1.0, -1e-8, 0.1, 0.2),
        (1.0, float("nan"), 0.1, 0.2),
        (1.0, float("inf"), 0.1, 0.2),
        (1.0, 1e-8, -1e-9, 0.2),
        (1.0, 1e-8, 1.0, 0.2),
        (1.0, 1e-8, float("nan"), 0.2),
        (1.0, 1e-8, 0.1, -1e-9),
        (1.0, 1e-8, 0.1, 1.0),
        (1.0, 1e-8, 0.1, float("inf")),
        (1e308, 1e308, 0.1, 0.2),
    ]
    for parameters in invalid:
        status, output = bridge.prepare(*parameters)
        assert status != 0, parameters
        assert_same_bits(output, np.full(10, SENTINEL, dtype=np.float64))


def test_apply_rejects_nonfinite_samples_without_writing(bridge):
    status, prepared = bridge.prepare(5.88172456e-05, 3.241287029e-08, 0.1, 0.25)
    assert status == 0
    for inverse in (False, True):
        for sample in (float("nan"), float("inf"), float("-inf")):
            output = np.asarray([SENTINEL], dtype=np.float64)
            result = bridge.apply_raw(
                double_ptr(prepared), sample, inverse, double_ptr(output)
            )
            assert result != 0
            assert_same_bits(output, [SENTINEL])


def test_null_misaligned_and_overlapping_pointers_are_rejected_atomically(bridge):
    status, prepared = bridge.prepare(5.88172456e-05, 3.241287029e-08, 0.1, 0.25)
    assert status == 0

    null = C.cast(C.c_void_p(0), DOUBLE_PTR)
    output = np.asarray([SENTINEL], dtype=np.float64)
    assert bridge.apply_raw(null, 0.5, False, double_ptr(output)) != 0
    assert_same_bits(output, [SENTINEL])
    assert bridge.apply_raw(double_ptr(prepared), 0.5, False, null) != 0

    storage = np.zeros(11, dtype=np.float64)
    storage[1:] = prepared
    unaligned_prepared = C.cast(C.c_void_p(storage.ctypes.data + 1), DOUBLE_PTR)
    assert bridge.apply_raw(unaligned_prepared, 0.5, False, double_ptr(output)) != 0
    assert_same_bits(output, [SENTINEL])

    byte_storage = np.zeros(16, dtype=np.uint8)
    misaligned_output = C.cast(C.c_void_p(byte_storage.ctypes.data + 1), DOUBLE_PTR)
    assert bridge.apply_raw(double_ptr(prepared), 0.5, False, misaligned_output) != 0

    for output_index in (0, 9):
        aliased = prepared.copy()
        before = aliased.copy()
        status = bridge.apply_raw(
            double_ptr(aliased), 0.5, False,
            C.cast(C.c_void_p(aliased.ctypes.data + output_index * 8), DOUBLE_PTR),
        )
        assert status != 0
        assert_same_bits(aliased, before)


def test_apply_rejects_inconsistent_prepared_state_atomically(bridge):
    status, prepared = bridge.prepare(5.88172456e-05, 3.241287029e-08, 0.1, 0.25)
    assert status == 0
    tampered = prepared.copy()
    tampered[2] = np.nextafter(tampered[2], np.inf)
    output = np.asarray([SENTINEL], dtype=np.float64)
    status = bridge.apply_raw(double_ptr(tampered), 0.5, False, double_ptr(output))
    assert status != 0
    assert_same_bits(output, [SENTINEL])


def test_oracle_has_stable_dji_bit_pattern():
    prepared = oracle.prepare(5.88172456e-05, 3.241287029e-08, 4100 / 65535, 0.25)
    assert int(bits(prepared)[2]) == 0x40187CCB5341D427
    assert int(bits(prepared)[3]) == 0x3F70146392CEF4BD
    assert struct.unpack("<Q", struct.pack("<d", prepared[2]))[0] == 0x40187CCB5341D427
