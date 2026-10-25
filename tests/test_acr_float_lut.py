"""Bit-exact tests for the 4096-segment ACR float LUT bridge and objects."""

from __future__ import annotations

import ctypes as C
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_float_lut.cpp"
HEADER = SOURCE.with_suffix(".hpp")
NOISE_SOURCE = NATIVE / "src/acr_noise_transform.cpp"
NOISE_HEADER = NOISE_SOURCE.with_suffix(".hpp")
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_float_lut.cpp.o"
NOISE_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_noise_transform.cpp.o"
BRIDGE = NATIVE / "tools/acr_float_lut_bridge.cpp"
sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import float_lut_oracle as oracle  # noqa: E402
import noise_transform_oracle as noise_oracle  # noqa: E402


FLOAT = C.c_float
FLOAT_PTR = C.POINTER(FLOAT)
SIZE = C.c_size_t
SIZE_MAX = C.c_size_t(-1).value
TABLE_SIZE = oracle.TABLE_SIZE
SENTINEL = np.float32(-1234.5)

OK = 0
INVALID_ARGUMENT = 1
INVALID_PARAMETERS = 2
OVERLAP = 3
EVALUATION_FAILURE = 4


def float_ptr(array):
    return array.ctypes.data_as(FLOAT_PTR)


def null_float_ptr():
    return C.cast(C.c_void_p(0), FLOAT_PTR)


def bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def assert_same_bits(actual, expected):
    np.testing.assert_array_equal(bits(actual), bits(expected))


def _compiled_inputs(request_kind):
    if request_kind == "source":
        return [SOURCE, NOISE_SOURCE]
    for obj, source, header in (
        (OBJECT, SOURCE, HEADER),
        (NOISE_OBJECT, NOISE_SOURCE, NOISE_HEADER),
    ):
        assert obj.is_file(), f"rebuild renderer: missing official object {obj.name}"
        assert obj.stat().st_mtime_ns >= max(source.stat().st_mtime_ns, header.stat().st_mtime_ns), (
            f"rebuild renderer: stale official object {obj.name}"
        )
    return [OBJECT, NOISE_OBJECT]


class Bridge:
    def __init__(self, path):
        self.library = C.CDLL(str(path))
        self.build_fn = self.library.acr_test_lut_build
        self.build_fn.argtypes = [C.c_int, C.POINTER(C.c_double), SIZE, FLOAT_PTR, SIZE]
        self.build_fn.restype = C.c_int
        self.apply_fn = self.library.acr_test_lut_apply
        self.apply_fn.argtypes = [
            FLOAT_PTR, SIZE, FLOAT_PTR, SIZE, FLOAT_PTR, SIZE, SIZE, C.c_int
        ]
        self.apply_fn.restype = C.c_int

    def build_raw(self, kind, params, out, capacity=None):
        params = np.asarray(params, dtype=np.float64)
        params_ptr = (
            params.ctypes.data_as(C.POINTER(C.c_double)) if params.size else
            C.cast(C.c_void_p(0), C.POINTER(C.c_double))
        )
        if capacity is None:
            capacity = 0 if out is None else out.size
        output_ptr = float_ptr(out) if out is not None else null_float_ptr()
        return self.build_fn(int(kind), params_ptr, params.size, output_ptr, capacity)

    def build(self, kind, params=()):
        output = np.full(TABLE_SIZE, SENTINEL, dtype=np.float32)
        status = self.build_raw(kind, params, output)
        return status, output

    def apply_raw(self, table, table_count, src, src_count, dst, dst_count, width, signed):
        return self.apply_fn(
            float_ptr(table) if table is not None else null_float_ptr(),
            table_count,
            float_ptr(src) if src is not None else null_float_ptr(),
            src_count,
            float_ptr(dst) if dst is not None else null_float_ptr(),
            dst_count,
            width,
            int(signed),
        )

    def apply(self, table, samples, signed=False):
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        output = np.full(samples.size, SENTINEL, dtype=np.float32)
        status = self.apply_raw(
            table, TABLE_SIZE, samples, samples.size, output, output.size,
            samples.size, signed,
        )
        return status, output


@pytest.fixture(scope="module", params=("source", "official_object"), ids=("source", "official"))
def bridge(request, tmp_path_factory):
    inputs = _compiled_inputs(request.param)
    assert BRIDGE.is_file(), f"missing float-LUT C ABI bridge: {BRIDGE}"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_float_lut_" + request.param) / ("libfloat_lut" + suffix)
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the float-LUT bridge"
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


def _prepared_noise_params():
    return noise_oracle.prepare(5.88172456e-05, 3.241287029e-08, 4100 / 65535, 0.25)


def _expected_table(kind, params):
    return oracle.build(kind, params if kind in (2, 3) else None)


@pytest.mark.parametrize("kind", (0, 1, 2, 3))
def test_build_matches_independent_oracle_at_every_node_and_guard(bridge, kind):
    params = _prepared_noise_params() if kind in (2, 3) else ()
    expected = _expected_table(kind, params)
    status, actual = bridge.build(kind, params)
    assert status == OK
    assert_same_bits(actual, expected)
    assert_same_bits(actual[-2:], expected[-2:])


@pytest.mark.parametrize("kind", (0, 1, 2, 3))
def test_unsigned_grid_and_random_samples_match_oracle_bit_for_bit(bridge, kind):
    params = _prepared_noise_params() if kind in (2, 3) else ()
    status, table = bridge.build(kind, params)
    assert status == OK

    rng = np.random.default_rng(2026101901 + kind)
    nodes = np.arange(oracle.SEGMENTS + 1, dtype=np.float32) / np.float32(oracle.SEGMENTS)
    neighbors = np.concatenate((
        np.nextafter(nodes, np.float32(-np.inf)),
        nodes,
        np.nextafter(nodes, np.float32(np.inf)),
    ))
    neighbors = neighbors[(neighbors >= np.float32(0.0)) & (neighbors <= np.float32(1.0))]
    lower_limit = np.float32(-1.0 / oracle.SEGMENTS)
    upper_limit = np.float32((oracle.SEGMENTS + 1) / oracle.SEGMENTS)
    lower_inside = np.nextafter(lower_limit, np.float32(np.inf))
    upper_inside = np.nextafter(upper_limit, np.float32(-np.inf))
    samples = np.concatenate((
        np.asarray([lower_inside, 0.0, -0.0, 1.0, upper_inside], dtype=np.float32),
        neighbors,
        rng.random(1024, dtype=np.float32),
    ))
    status, actual = bridge.apply(table, samples)
    assert status == OK
    expected = np.asarray([oracle.interpolate(table, float(x)) for x in samples], dtype=np.float32)
    assert_same_bits(actual, expected)

    for boundary_probe in (
        lower_limit,
        np.nextafter(lower_limit, np.float32(-np.inf)),
        upper_limit,
        np.nextafter(upper_limit, np.float32(np.inf)),
    ):
        rejected = np.asarray([boundary_probe], dtype=np.float32)
        untouched = np.asarray([SENTINEL], dtype=np.float32)
        result = bridge.apply_raw(table, TABLE_SIZE, rejected, 1, untouched, 1, 1, False)
        assert result == INVALID_PARAMETERS, (boundary_probe, oracle.f32(float(boundary_probe) * oracle.SEGMENTS))
        assert_same_bits(untouched, [SENTINEL])


@pytest.mark.parametrize("kind", (0, 1, 2, 3))
def test_signed_grid_and_nextafter_nodes_match_oracle(bridge, kind):
    params = _prepared_noise_params() if kind in (2, 3) else ()
    status, table = bridge.build(kind, params)
    assert status == OK

    integer_grid = np.arange(-3 * oracle.SEGMENTS, 3 * oracle.SEGMENTS + 1, dtype=np.int32)
    grid = integer_grid.astype(np.float32) / np.float32(oracle.SEGMENTS)
    node_values = np.arange(oracle.SEGMENTS + 1, dtype=np.float32) / np.float32(oracle.SEGMENTS)
    around_nodes = np.concatenate((
        np.nextafter(node_values, np.float32(-np.inf)),
        node_values,
        np.nextafter(node_values, np.float32(np.inf)),
    ))
    signed_samples = np.concatenate((
        grid,
        -around_nodes,
        around_nodes,
        np.asarray([-3.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    ))
    status, actual = bridge.apply(table, signed_samples, signed=True)
    assert status == OK
    expected = np.asarray(
        [oracle.apply_signed(table, float(x)) for x in signed_samples], dtype=np.float32
    )
    assert_same_bits(actual, expected)


@pytest.mark.parametrize("kind", (0, 2, 3))
def test_random_finite_tables_match_scalar_interpolation(bridge, kind):
    params = _prepared_noise_params() if kind in (2, 3) else ()
    status, base = bridge.build(kind, params)
    assert status == OK
    rng = np.random.default_rng(2026101907 + kind)
    samples = rng.random(256, dtype=np.float32)
    for _ in range(8):
        table = rng.uniform(-20.0, 20.0, size=TABLE_SIZE).astype(np.float32)
        table[-1] = table[-2]
        before = table.copy()
        status, actual = bridge.apply(table, samples)
        assert status == OK
        expected = np.asarray([oracle.interpolate(table, float(x)) for x in samples], dtype=np.float32)
        assert_same_bits(actual, expected)
        assert_same_bits(table, before)
    assert np.isfinite(base).all()


def test_interpolation_fma_counterexample_is_locked(bridge):
    table = np.zeros(TABLE_SIZE, dtype=np.float32)
    table[1] = np.float32(7.817676067352295)
    table[2] = np.float32(-6.387670516967773)
    table[-1] = table[-2] = np.float32(0.0)
    sample = np.float32(1.125 / oracle.SEGMENTS)
    expected = oracle.interpolate(table, float(sample))
    assert int(bits([expected])[0]) == 0x40C15821
    # Difference-form interpolation produces the neighboring binary32 value.
    t = oracle.f32(float(sample) * oracle.SEGMENTS)
    fraction = oracle.f32(float(t) - 1.0)
    difference_form = oracle.fma32(fraction, oracle.f32(table[2] - table[1]), table[1])
    assert int(bits([difference_form])[0]) == 0x40C15820
    status, actual = bridge.apply(table, np.asarray([sample], dtype=np.float32))
    assert status == OK
    assert int(bits(actual)[0]) == 0x40C15821


@pytest.mark.parametrize("signed", (False, True))
def test_in_place_and_padding_are_supported_without_touching_padding_or_table(bridge, signed):
    params = _prepared_noise_params()
    status, table = bridge.build(2, params)
    assert status == OK
    table_before = table.copy()
    width = 6
    source = np.asarray(
        ([-1.5, -0.0, 0.0, 0.125, 1.0, 2.5, np.nan, np.inf] if signed else
         [0.25, -0.0, 0.0, 0.125, 1.0, 0.75, np.nan, np.inf]),
        dtype=np.float32,
    )
    expected = np.asarray(
        [oracle.apply_signed(table, float(x)) if signed else oracle.interpolate(table, float(x))
         for x in source[:width]],
        dtype=np.float32,
    )
    in_place = source.copy()
    status = bridge.apply_raw(table, TABLE_SIZE, in_place, in_place.size,
                              in_place, in_place.size, width, signed)
    assert status == OK
    assert_same_bits(in_place[:width], expected)
    assert np.isnan(in_place[width]) and np.isposinf(in_place[width + 1])
    assert_same_bits(table, table_before)

    padded_src = source.copy()
    padded_dst = np.full(source.size, SENTINEL, dtype=np.float32)
    status = bridge.apply_raw(table, TABLE_SIZE, padded_src, padded_src.size,
                              padded_dst, padded_dst.size, width, signed)
    assert status == OK
    assert_same_bits(padded_dst[:width], expected)
    assert_same_bits(padded_dst[width:], np.full(source.size - width, SENTINEL, dtype=np.float32))
    assert_same_bits(table, table_before)


@pytest.mark.parametrize("kind", (0, 1, 2, 3))
def test_build_capacity_and_invalid_requests_fail_atomically(bridge, kind):
    params = _prepared_noise_params() if kind in (2, 3) else ()
    output = np.full(TABLE_SIZE + 1, SENTINEL, dtype=np.float32)
    status = bridge.build_raw(kind, params, output, TABLE_SIZE - 1)
    assert status == INVALID_ARGUMENT
    assert_same_bits(output, np.full(TABLE_SIZE + 1, SENTINEL, dtype=np.float32))

    status = bridge.build_raw(99, (), output)
    assert status == INVALID_ARGUMENT
    assert_same_bits(output, np.full(TABLE_SIZE + 1, SENTINEL, dtype=np.float32))
    status = bridge.build_raw(2, (), output)
    assert status == INVALID_ARGUMENT
    assert_same_bits(output, np.full(TABLE_SIZE + 1, SENTINEL, dtype=np.float32))
    status = bridge.build_raw(0, (), None)
    assert status == INVALID_ARGUMENT

    params_with_nan = np.asarray([0.0] * 9 + [np.nan], dtype=np.float64)
    status = bridge.build_raw(2, params_with_nan, output)
    assert status == EVALUATION_FAILURE
    assert_same_bits(output, np.full(TABLE_SIZE + 1, SENTINEL, dtype=np.float32))


def test_build_writes_only_the_declared_table_and_preserves_extra_capacity(bridge):
    output = np.full(TABLE_SIZE + 2, SENTINEL, dtype=np.float32)
    status = bridge.build_raw(0, (), output, output.size)
    assert status == OK
    assert_same_bits(output[:TABLE_SIZE], oracle.build(0))
    assert_same_bits(output[TABLE_SIZE:], np.full(2, SENTINEL, dtype=np.float32))


def test_apply_rejects_nonfinite_samples_tables_and_bad_guard_atomically(bridge):
    status, table = bridge.build(1)
    assert status == OK
    lower_limit = np.float32(-1.0 / oracle.SEGMENTS)
    upper_limit = np.float32((oracle.SEGMENTS + 1) / oracle.SEGMENTS)
    unsigned_outside = (
        lower_limit,
        np.nextafter(lower_limit, np.float32(-np.inf)),
        upper_limit,
        np.nextafter(upper_limit, np.float32(np.inf)),
        np.finfo(np.float32).max,
    )
    invalid_samples = [
        (False, sample) for sample in (np.nan, np.inf, -np.inf, *unsigned_outside)
    ] + [(True, sample) for sample in (np.nan, np.inf, -np.inf)]
    for signed, sample in invalid_samples:
        src = np.asarray([sample], dtype=np.float32)
        dst = np.asarray([SENTINEL], dtype=np.float32)
        status = bridge.apply_raw(table, TABLE_SIZE, src, 1, dst, 1, 1, signed)
        assert status != OK, (signed, sample)
        assert_same_bits(dst, [SENTINEL])

    for bad_value in (np.nan, np.inf, -np.inf):
        bad_table = table.copy()
        bad_table[31] = np.float32(bad_value)
        dst = np.asarray([SENTINEL], dtype=np.float32)
        src = np.asarray([0.25], dtype=np.float32)
        status = bridge.apply_raw(bad_table, TABLE_SIZE, src, 1, dst, 1, 1, False)
        assert status != OK
        assert_same_bits(dst, [SENTINEL])
        assert np.isnan(bad_table[31]) if np.isnan(bad_value) else bad_table[31] == bad_value

    bad_guard = table.copy()
    bad_guard[-1] = np.nextafter(bad_guard[-1], np.float32(np.inf))
    dst = np.asarray([SENTINEL], dtype=np.float32)
    src = np.asarray([0.25], dtype=np.float32)
    status = bridge.apply_raw(bad_guard, TABLE_SIZE, src, 1, dst, 1, 1, False)
    assert status == INVALID_PARAMETERS
    assert_same_bits(dst, [SENTINEL])


def test_apply_rejects_bad_flags_counts_and_size_overflow_atomically(bridge):
    status, table = bridge.build(0)
    assert status == OK
    src = np.asarray([0.25, 0.75], dtype=np.float32)
    dst = np.full(2, SENTINEL, dtype=np.float32)
    for signed, src_count, dst_count, width in (
        (2, 2, 2, 2),
        (-1, 2, 2, 2),
        (0, 2, 2, 0),
        (0, 1, 2, 2),
        (0, 2, 1, 2),
        (0, 2, 2, 3),
    ):
        status = bridge.apply_raw(table, TABLE_SIZE, src, src_count, dst, dst_count, width, signed)
        assert status == INVALID_ARGUMENT
        assert_same_bits(dst, np.full(2, SENTINEL, dtype=np.float32))

    status = bridge.apply_raw(table, TABLE_SIZE - 1, src, 2, dst, 2, 2, False)
    assert status == INVALID_ARGUMENT
    assert_same_bits(dst, np.full(2, SENTINEL, dtype=np.float32))
    overflow_width = SIZE_MAX // C.sizeof(FLOAT) + 1
    status = bridge.apply_fn(
        float_ptr(table), TABLE_SIZE, float_ptr(src), SIZE_MAX, float_ptr(dst), SIZE_MAX,
        overflow_width, 0,
    )
    assert status == INVALID_ARGUMENT
    assert_same_bits(dst, np.full(2, SENTINEL, dtype=np.float32))

    for table_pointer, src_pointer, dst_pointer in (
        (None, src, dst),
        (table, None, dst),
        (table, src, None),
    ):
        status = bridge.apply_raw(table_pointer, TABLE_SIZE, src_pointer, 2,
                                  dst_pointer, 2, 2, False)
        assert status == INVALID_ARGUMENT
        assert_same_bits(dst, np.full(2, SENTINEL, dtype=np.float32))


def test_apply_rejects_partial_aliases_and_table_destination_overlap_atomically(bridge):
    status, table = bridge.build(0)
    assert status == OK
    original_table = table.copy()

    storage = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    before = storage.copy()
    status = bridge.apply_fn(
        float_ptr(table), TABLE_SIZE,
        float_ptr(storage[:4]), 4,
        float_ptr(storage[1:]), 4,
        4, 0,
    )
    assert status == OVERLAP
    assert_same_bits(storage, before)

    table_dst = table.copy()
    table_before = table_dst.copy()
    source = np.asarray([0.25, 0.75], dtype=np.float32)
    status = bridge.apply_fn(
        float_ptr(table_dst), TABLE_SIZE,
        float_ptr(source), source.size,
        float_ptr(table_dst[10:]), TABLE_SIZE - 10,
        2, 0,
    )
    assert status == OVERLAP
    assert_same_bits(table_dst, table_before)
    assert_same_bits(table, original_table)


def test_signed_overflow_result_fails_without_modifying_destination(bridge):
    table = np.full(TABLE_SIZE, np.finfo(np.float32).max, dtype=np.float32)
    table[-1] = table[-2]
    src = np.asarray([np.finfo(np.float32).max], dtype=np.float32)
    dst = np.asarray([SENTINEL], dtype=np.float32)
    status = bridge.apply_raw(table, TABLE_SIZE, src, 1, dst, 1, 1, True)
    assert status == INVALID_PARAMETERS
    assert_same_bits(dst, [SENTINEL])

