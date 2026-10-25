"""Bitwise CPU validation of the bounded ARM64 profile-gain transcription."""

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
SUPPORT = ROOT / "tests" / "acr_support"
CORE_SOURCE = NATIVE / "src" / "acr_profile_gain.cpp"
BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_bridge.cpp"
OBJECT_SUFFIX = ".obj" if sys.platform == "win32" else ".o"
OFFICIAL_OBJECT = NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/acr_profile_gain.cpp{OBJECT_SUFFIX}"

sys.path.insert(0, str(SUPPORT))
import profile_gain_oracle as oracle  # noqa: E402


F32_PTR = C.POINTER(C.c_float)
U32_PTR = C.POINTER(C.c_uint32)
SIZE = C.c_size_t
SIZE_MAX = SIZE(-1).value

OK = 0
INVALID_ARGUMENT = 1
DIMENSIONS = 2
STRIDE = 3
SHORT_BUFFER = 4
ALIGNMENT = 5
OVERLAP = 6
OVERFLOW = 7
RESOURCE_LIMIT = 8
INVALID_PREPARED = 9
NONFINITE_INPUT = 10
NONFINITE_INTERMEDIATE = 11

SENTINEL = np.float32(-731.625)


def _float_ptr(array: np.ndarray | None):
    if array is None:
        return C.cast(C.c_void_p(0), F32_PTR)
    return array.ctypes.data_as(F32_PTR)


def _word_ptr(array: np.ndarray | None):
    if array is None:
        return C.cast(C.c_void_p(0), U32_PTR)
    return array.ctypes.data_as(U32_PTR)


def _compile_bridge(tmp_path: Path, kind: str) -> Path:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ is required to compile the profile-gain bridge")
    assert BRIDGE_SOURCE.is_file(), f"missing profile-gain bridge: {BRIDGE_SOURCE}"

    if kind == "source":
        assert CORE_SOURCE.is_file(), f"missing profile-gain source: {CORE_SOURCE}"
        implementation = CORE_SOURCE
    else:
        assert kind == "official_object"
        if not OFFICIAL_OBJECT.is_file():
            pytest.skip(
                "official renderer object is unavailable at the platform CMake path "
                f"({OFFICIAL_OBJECT}); official-object validation was not run"
            )
        assert CORE_SOURCE.is_file(), f"missing profile-gain source: {CORE_SOURCE}"
        if OFFICIAL_OBJECT.stat().st_mtime_ns < CORE_SOURCE.stat().st_mtime_ns:
            pytest.skip("profile-gain official object is stale; rebuild imprint_renderer")
        implementation = OFFICIAL_OBJECT

    if sys.platform == "darwin":
        suffix = ".dylib"
        shared_flags = ["-dynamiclib"]
    elif sys.platform == "win32":
        suffix = ".dll"
        shared_flags = ["-shared", "-fPIC"]
    else:
        suffix = ".so"
        shared_flags = ["-shared", "-fPIC"]
    output = tmp_path / f"libprofile_gain_{kind}{suffix}"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        *shared_flags,
        "-I",
        str(NATIVE / "src"),
        str(BRIDGE_SOURCE),
        str(implementation),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.apply_fn = self.library.acr_profile_gain_bridge
        self.apply_fn.argtypes = [
            F32_PTR, SIZE,
            U32_PTR, SIZE,
            F32_PTR, SIZE,
            C.POINTER(F32_PTR), C.POINTER(SIZE),
            F32_PTR, SIZE,
            SIZE, SIZE, SIZE, SIZE,
        ]
        self.apply_fn.restype = C.c_int

    def apply_raw(
        self,
        fields: np.ndarray | None,
        geometry: np.ndarray | None,
        table: np.ndarray | None,
        inputs: list[np.ndarray | None] | None,
        destination: np.ndarray | None,
        *,
        outer: int,
        inner: int,
        value_stride: int,
        coord_stride: int,
        fields_count: int | None = None,
        geometry_count: int | None = None,
        table_count: int | None = None,
        input_counts: list[int] | None = None,
        dest_count: int | None = None,
        null_input_vector: bool = False,
        null_input_counts: bool = False,
    ) -> int:
        field_ptr = _float_ptr(fields)
        geometry_ptr = _word_ptr(geometry)
        table_ptr = _float_ptr(table)
        dest_ptr = _float_ptr(destination)
        if inputs is None:
            input_vector = C.cast(C.c_void_p(0), C.POINTER(F32_PTR))
            count_vector = C.cast(C.c_void_p(0), C.POINTER(SIZE))
        else:
            assert len(inputs) == 5
            pointer_array = (F32_PTR * 5)(*(_float_ptr(value) for value in inputs))
            lengths = input_counts or [0 if value is None else value.size for value in inputs]
            count_array = (SIZE * 5)(*(int(x) for x in lengths))
            input_vector = C.cast(C.c_void_p(0), C.POINTER(F32_PTR)) if null_input_vector else pointer_array
            count_vector = C.cast(C.c_void_p(0), C.POINTER(SIZE)) if null_input_counts else count_array
        return int(self.apply_fn(
            field_ptr,
            len(fields) if fields_count is None and fields is not None else (fields_count or 0),
            geometry_ptr,
            len(geometry) if geometry_count is None and geometry is not None else (geometry_count or 0),
            table_ptr,
            len(table) if table_count is None and table is not None else (table_count or 0),
            input_vector,
            count_vector,
            dest_ptr,
            len(destination) if dest_count is None and destination is not None else (dest_count or 0),
            outer,
            inner,
            value_stride,
            coord_stride,
        ))


@pytest.fixture(scope="module", params=("source", "official_object"), ids=("source", "official_object"))
def bridge(request, tmp_path_factory):
    path = _compile_bridge(tmp_path_factory.mktemp("acr_profile_gain_" + request.param), request.param)
    return Bridge(path)


def _prepared():
    # Different u/v affine transforms and distinct bounds make accidental
    # coordinate reuse visible.  The last field is deliberately non-unit so
    # the powf path is part of ordinary coverage.
    fields = np.asarray([
        0.04, 0.11, 0.07, 0.03, 0.02,
        1.4, -0.1,
        0.55, 0.1,
        0.125, 2.75,
        0.25, 1.5,
        1.1, 4.0, 0.75,
    ], dtype=np.float32)
    stride_u = 9
    stride_v = 41
    geometry = np.asarray([stride_v, stride_u, 3, 2, 4], dtype=np.uint32)
    table = np.empty(stride_v * 3 + 8, dtype=np.float32)
    # Include non-table padding with distinct values.  The address formula
    # must honor both explicit row strides rather than infer a dense layout.
    for index in range(table.size):
        table[index] = np.float32(-0.8 + index * 0.0137)
    return fields, geometry, table


def _bitwise_equal(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
    actual_bits = np.ascontiguousarray(actual, dtype=np.float32).view(np.uint32)
    expected_bits = np.ascontiguousarray(expected, dtype=np.float32).view(np.uint32)
    assert np.array_equal(actual_bits, expected_bits), (
        f"{message}: first mismatch "
        f"{np.flatnonzero(actual_bits.reshape(-1) != expected_bits.reshape(-1))[:8].tolist()}"
    )


def _strided_case(seed: int = 0xAC2024):
    fields, geometry, table = _prepared()
    outer, inner = 4, 32  # 128 randomized active samples, across four rows.
    value_stride, coord_stride = 39, 37
    inputs = [np.full(outer * value_stride, np.float32(-13.25), dtype=np.float32)
              for _ in range(3)]
    inputs += [np.full(outer * coord_stride, np.float32(-19.5), dtype=np.float32)
               for _ in range(2)]
    rng = np.random.default_rng(seed)
    for row in range(outer):
        for col in range(inner):
            offset = row * inner + col
            # Four adversarial classes plus broad finite random coverage.
            if offset == 0:  # constant sample
                values = (0.4, 0.3, 0.5)
                cv, cu = (0.2, 0.7)
            elif offset < 8:  # ramp
                t = np.float32((offset - 1) / 7)
                values = (float(t), float(np.float32(0.15 + 0.6 * t)), float(np.float32(0.8 - 0.4 * t)))
                cv, cu = (float(np.float32(0.1 + t)), float(np.float32(-0.2 + 2.8 * t)))
            elif offset == 8:  # impulse
                values = (-0.75, 1.9, 0.25)
                cv, cu = (0.8, 1.6)
            elif offset == 9:  # signed zeros reach the ordered min/max selects
                values = (-0.0, +0.0, -0.0)
                cv, cu = (0.5, 0.5)
            elif offset == 10:  # finite overshoot values are legal inputs
                values = (-4.0, 7.0, 2.0)
                cv, cu = (3.5, -1.0)
            else:
                values = tuple(float(x) for x in rng.uniform(-2.25, 3.5, size=3))
                cv = float(rng.uniform(-3.0, 4.0))
                cu = float(rng.uniform(-3.0, 5.0))
            for channel in range(3):
                inputs[channel][row * value_stride + col] = np.float32(values[channel])
            inputs[3][row * coord_stride + col] = np.float32(cv)
            inputs[4][row * coord_stride + col] = np.float32(cu)
    return fields, geometry, table, inputs, outer, inner, value_stride, coord_stride


def _randomized_prepared_case(rng: np.random.Generator, state_index: int):
    """Build one legal, numerically bounded prepared-state/table geometry."""
    max_u = 0 if state_index % 9 == 0 else int(rng.integers(1, 6))
    max_v = 0 if state_index % 11 == 0 else int(rng.integers(1, 5))
    if state_index % 10 == 3:
        max_z = int(rng.integers(2, 7))  # reserve deterministic q-base-cap states
    else:
        max_z = 0 if state_index % 13 == 0 else int(rng.integers(1, 7))

    exponent_values = (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.7, 2.0, 3.0)
    exponent = np.float32(exponent_values[state_index % len(exponent_values)])
    if exponent < 0 and max_z < 2:
        max_z = int(rng.integers(2, 7))

    stride_u = max_z + 1 + int(rng.integers(1, 5))
    stride_v = (max_u + 1) * stride_u + int(rng.integers(1, 7))
    geometry = np.asarray([stride_v, stride_u, max_u, max_v, max_z], dtype=np.uint32)

    def axis_bounds(maximum: int) -> tuple[np.float32, np.float32]:
        if maximum == 0:
            upper = np.float32(rng.uniform(0.08, 0.92))
        elif state_index % 5 == 0:
            upper = np.float32(maximum)
        else:
            # Fractional upper in the last legal cell: trunc(upper)==maximum.
            upper = np.float32(maximum + rng.uniform(0.08, 0.92))
        lower = np.float32(rng.uniform(0.0, float(upper)))
        return lower, upper

    lower_u, upper_u = axis_bounds(max_u)
    lower_v, upper_v = axis_bounds(max_v)

    if state_index % 10 == 3:
        # With p1=1 and c1=1, the prelookup coordinate is exactly one.  The
        # larger index scale makes trunc(q) exceed max_z without approaching
        # the int32 conversion boundary.
        coeffs = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        prelookup_scale = np.float32(1.0)
        index_scale = np.float32(max_z + 2.0)
    elif exponent < 0:
        # For negative powf exponents keep z strictly positive and q small;
        # the separate negative-exponent/zero-z error probe covers infinity.
        coeffs = np.asarray([0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        prelookup_scale = np.float32(1.0)
        index_scale = np.float32(0.125)
    elif state_index % 4 == 0:
        coeffs = np.zeros(5, dtype=np.float32)
        prelookup_scale = np.float32(rng.choice((-2.0, -1.0, -0.25, 0.0, 0.25, 0.75, 1.0, 1.5, 2.0)))
        index_scale = np.float32(rng.choice((0.0, 0.25, 1.0, max_z + 0.5, 2.0 * max_z + 1.0)))
    elif state_index % 4 == 1:
        coeffs = np.asarray([0.25, 0.125, 0.5, 0.0625, 0.1], dtype=np.float32)
        prelookup_scale = np.float32(rng.choice((-2.0, -1.0, -0.25, 0.0, 0.25, 0.75, 1.0, 1.5, 2.0)))
        index_scale = np.float32(rng.choice((0.0, 0.25, 1.0, max_z + 0.5, 2.0 * max_z + 1.0)))
    elif state_index % 4 == 2:
        coeffs = np.asarray([-0.25, -0.125, -0.5, -0.0625, -0.1], dtype=np.float32)
        prelookup_scale = np.float32(rng.choice((-2.0, -1.0, -0.25, 0.0, 0.25, 0.75, 1.0, 1.5, 2.0)))
        index_scale = np.float32(rng.choice((0.0, 0.25, 1.0, max_z + 0.5, 2.0 * max_z + 1.0)))
    else:
        coeffs = rng.uniform(-0.5, 0.5, size=5).astype(np.float32)
        prelookup_scale = np.float32(rng.uniform(-2.0, 2.0))
        index_scale = np.float32(rng.uniform(0.0, 2.0 * max_z + 2.0))
    if exponent >= 0 and state_index % 10 != 3:
        if state_index % 8 == 5:
            prelookup_scale = np.float32(-1.25)
        elif state_index % 8 == 6:
            prelookup_scale = np.float32(0.0)
        elif state_index % 8 == 7:
            prelookup_scale = np.float32(1.25)

    affine_scales = (-1.5, -0.5, 0.25, 1.25, 2.0)
    fields = np.asarray([
        *coeffs,
        affine_scales[state_index % len(affine_scales)],
        rng.uniform(-2.0, 2.0),
        affine_scales[(state_index * 3 + 1) % len(affine_scales)],
        rng.uniform(-2.0, 2.0),
        lower_u, upper_u, lower_v, upper_v,
        prelookup_scale, index_scale, exponent,
    ], dtype=np.float32)

    # Add both inter-cell and tail padding; all addresses selected by the
    # prepared geometry remain well inside the backed table allocation.
    maximum_index = max_z + max_u * stride_u + max_v * stride_v
    table_count = maximum_index + 1 + int(rng.integers(1, 9))
    table = rng.uniform(-2.0, 2.0, size=table_count).astype(np.float32)

    outer = int(rng.integers(1, 4))
    inner = int(rng.integers(2, 7))
    value_stride = inner + int(rng.integers(0, 5))
    coord_stride = inner + int(rng.integers(0, 6))
    inputs = [
        np.full(outer * value_stride, np.float32(-17.25), dtype=np.float32)
        for _ in range(3)
    ]
    inputs.extend(
        np.full(outer * coord_stride, np.float32(-23.5), dtype=np.float32)
        for _ in range(2)
    )

    sample_rng = np.random.default_rng(int(rng.integers(0, 2**32, dtype=np.uint32)))
    for row in range(outer):
        for col in range(inner):
            inputs[0][row * value_stride + col] = np.float32(sample_rng.uniform(-1.5, 1.5))
            if exponent < 0:
                inputs[1][row * value_stride + col] = np.float32(sample_rng.uniform(0.25, 1.5))
            else:
                inputs[1][row * value_stride + col] = np.float32(sample_rng.uniform(-1.5, 1.5))
            inputs[2][row * value_stride + col] = np.float32(sample_rng.uniform(-1.5, 1.5))
            inputs[3][row * coord_stride + col] = np.float32(sample_rng.uniform(-5.0, 5.0))
            inputs[4][row * coord_stride + col] = np.float32(sample_rng.uniform(-5.0, 5.0))

    # Exercise both clamps on both independently scaled axes.  Scales are
    # selected away from zero, so +/-32 reliably reach opposite bounds.
    inputs[3][0] = np.float32(32.0 if fields[7] > 0 else -32.0)
    inputs[3][1] = np.float32(-32.0 if fields[7] > 0 else 32.0)
    inputs[4][0] = np.float32(32.0 if fields[5] > 0 else -32.0)
    inputs[4][1] = np.float32(-32.0 if fields[5] > 0 else 32.0)
    if state_index % 10 == 3:
        inputs[0][0] = np.float32(0.0)
        inputs[1][0] = np.float32(1.0)
        inputs[2][0] = np.float32(0.0)

    return fields, geometry, table, inputs, outer, inner, value_stride, coord_stride


def test_bitwise_hard_probes_random_and_padded_rows(bridge):
    fields, geometry, table, inputs, outer, inner, value_stride, coord_stride = _strided_case()
    output = np.full(outer * coord_stride, SENTINEL, dtype=np.float32)
    table_before = table.copy()
    input_before = [values.copy() for values in inputs]
    expected_active = oracle.reference_rows(fields, geometry, table, inputs, outer, inner, value_stride, coord_stride)

    status = bridge.apply_raw(
        fields, geometry, table, inputs, output,
        outer=outer, inner=inner, value_stride=value_stride, coord_stride=coord_stride,
    )

    assert status == OK
    expected = np.full_like(output, SENTINEL)
    for row in range(outer):
        expected[row * coord_stride:row * coord_stride + inner] = expected_active[row]
    _bitwise_equal(output, expected, "active outputs and untouched destination padding")
    _bitwise_equal(table, table_before, "table remains read-only")
    for index, (actual, before) in enumerate(zip(inputs, input_before)):
        _bitwise_equal(actual, before, f"input {index} remains read-only")


def test_x4_x5_coordinates_distinct_bounds_and_q_base_cap(bridge):
    fields, geometry, table = _prepared()
    one = [
        np.asarray([0.4], dtype=np.float32),
        np.asarray([0.3], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
        np.asarray([9.0], dtype=np.float32),  # x4 / coord_v reaches v upper 1.5
        np.asarray([-8.0], dtype=np.float32),  # x5 / coord_u reaches u lower 0.125
    ]
    out = np.asarray([SENTINEL], dtype=np.float32)
    status = bridge.apply_raw(fields, geometry, table, one, out, outer=1, inner=1,
                              value_stride=1, coord_stride=1)
    assert status == OK
    expected = oracle.reference_sample(fields, geometry, table, 0.4, 0.3, 0.5, 9.0, -8.0)
    _bitwise_equal(out, np.asarray([expected], dtype=np.float32), "separate x4/v and x5/u bounds")
    swapped_expected = oracle.reference_sample(fields, geometry, table, 0.4, 0.3, 0.5, -8.0, 9.0)
    assert oracle.bits32([expected])[0] != oracle.bits32([swapped_expected])[0], (
        "fixture must distinguish x4/v from x5/u"
    )

    # A high but int32-safe q must cap both z indices at max_z.  Padding after
    # max_z contains distinct values so a missing cap cannot pass by chance.
    capped_fields = fields.copy()
    capped_fields[14] = np.float32(20.0)
    capped_fields[15] = np.float32(0.63)
    cap_out = np.asarray([SENTINEL], dtype=np.float32)
    status = bridge.apply_raw(capped_fields, geometry, table, one, cap_out,
                              outer=1, inner=1, value_stride=1, coord_stride=1)
    assert status == OK
    cap_expected = oracle.reference_sample(capped_fields, geometry, table, 0.4, 0.3, 0.5, 9.0, -8.0)
    _bitwise_equal(cap_out, np.asarray([cap_expected], dtype=np.float32), "q integer base cap")


def test_singleton_and_fractional_upper_endpoint_neighbor_caps(bridge):
    fields, geometry, table = _prepared()
    # Extend backing storage so an incorrect uncapped neighbor remains inside
    # allocated memory and produces a deterministic bitwise mismatch.
    table = np.concatenate((table, np.linspace(1.25, 3.0, 64, dtype=np.float32)))
    p0, p1, p2 = (np.asarray([value], dtype=np.float32) for value in (0.4, 0.3, 0.5))

    cases = []
    # A single-cell axis may use a fractional constant coordinate: trunc(0.5)
    # is zero and min(base+1,maxIndex) must keep both corners at zero.
    singleton_u_fields = fields.copy()
    singleton_u_fields[9:11] = np.asarray([0.5, 0.5], dtype=np.float32)
    singleton_u_geometry = geometry.copy()
    singleton_u_geometry[2] = 0
    cases.append(("singleton_u", singleton_u_fields, singleton_u_geometry, 0.2, 0.7))

    singleton_v_fields = fields.copy()
    singleton_v_fields[11:13] = np.asarray([0.5, 0.5], dtype=np.float32)
    singleton_v_geometry = geometry.copy()
    singleton_v_geometry[3] = 0
    cases.append(("singleton_v", singleton_v_fields, singleton_v_geometry, 0.2, 0.7))

    # upper=2.75 is permitted with max index 2 because trunc(upper)==2;
    # each fractional endpoint's neighbor is capped back to the same cell.
    fractional_u_fields = fields.copy()
    fractional_u_fields[10] = np.float32(2.75)
    fractional_u_geometry = geometry.copy()
    fractional_u_geometry[2] = 2
    cases.append(("fractional_upper_u", fractional_u_fields, fractional_u_geometry, 0.2, 9.0))

    fractional_v_fields = fields.copy()
    fractional_v_fields[12] = np.float32(2.75)
    fractional_v_geometry = geometry.copy()
    fractional_v_geometry[3] = 2
    cases.append(("fractional_upper_v", fractional_v_fields, fractional_v_geometry, 9.0, 0.7))

    for label, case_fields, case_geometry, coord_v, coord_u in cases:
        inputs = [p0, p1, p2,
                  np.asarray([coord_v], dtype=np.float32),
                  np.asarray([coord_u], dtype=np.float32)]
        output = np.asarray([SENTINEL], dtype=np.float32)
        status = bridge.apply_raw(case_fields, case_geometry, table, inputs, output,
                                  outer=1, inner=1, value_stride=1, coord_stride=1)
        assert status == OK, label
        expected = oracle.reference_sample(case_fields, case_geometry, table,
                                           0.4, 0.3, 0.5, coord_v, coord_u)
        _bitwise_equal(output, np.asarray([expected], dtype=np.float32), label)


def test_finite_zero_negative_exponents_and_negative_prescale_are_legal(bridge):
    base_fields, geometry, table = _prepared()
    probes = []

    # powf(-0, 0) is evaluated and its finite result is selected because the
    # exponent is not one.  Zero transfer scale is therefore a valid prepared
    # value, as is a finite negative prelookup scale.
    zero_exponent_fields = base_fields.copy()
    zero_exponent_fields[:5] = np.float32(0.0)
    zero_exponent_fields[13] = np.float32(-2.0)
    zero_exponent_fields[14] = np.float32(0.25)
    zero_exponent_fields[15] = np.float32(0.0)
    probes.append(("zero_exponent_negative_scale", zero_exponent_fields, (0.4, 0.3, 0.5)))

    # Keep z positive and index_scale small so a negative exponent has a
    # finite powf result and q remains below the final z index.
    negative_exponent_fields = base_fields.copy()
    negative_exponent_fields[:5] = np.float32(0.0)
    negative_exponent_fields[1] = np.float32(0.5)
    negative_exponent_fields[13] = np.float32(1.0)
    negative_exponent_fields[14] = np.float32(0.125)
    negative_exponent_fields[15] = np.float32(-1.0)
    probes.append(("negative_exponent_positive_z", negative_exponent_fields, (0.0, 1.0, 0.0)))

    # Both the weighted value and prelookup scale are negative, producing
    # positive z before powf; no clamp-to-zero ambiguity is present.
    negative_product_fields = base_fields.copy()
    negative_product_fields[:5] = np.float32(0.0)
    negative_product_fields[1] = np.float32(-0.5)
    negative_product_fields[13] = np.float32(-2.0)
    negative_product_fields[14] = np.float32(0.5)
    negative_product_fields[15] = np.float32(0.5)
    probes.append(("negative_weighted_value_times_negative_scale", negative_product_fields, (0.0, 1.0, 0.0)))

    for label, fields, (p0, p1, p2) in probes:
        inputs = [np.asarray([p0], dtype=np.float32),
                  np.asarray([p1], dtype=np.float32),
                  np.asarray([p2], dtype=np.float32),
                  np.asarray([0.2], dtype=np.float32),
                  np.asarray([0.7], dtype=np.float32)]
        output = np.asarray([SENTINEL], dtype=np.float32)
        status = bridge.apply_raw(fields, geometry, table, inputs, output,
                                  outer=1, inner=1, value_stride=1, coord_stride=1)
        assert status == OK, f"{label}: status {status}"
        expected = oracle.reference_sample(fields, geometry, table, p0, p1, p2, 0.2, 0.7)
        _bitwise_equal(output, np.asarray([expected], dtype=np.float32), label)


def test_one_hundred_randomized_prepared_states_against_oracle(bridge):
    rng = np.random.default_rng(0x5047544D)
    fingerprints: set[tuple] = set()
    exponents: set[float] = set()
    prelookup_scales: set[float] = set()
    max_u_values: set[int] = set()
    max_v_values: set[int] = set()
    max_z_values: set[int] = set()
    q_cap_state_count = 0

    for state_index in range(100):
        fields, geometry, table, inputs, outer, inner, value_stride, coord_stride = (
            _randomized_prepared_case(rng, state_index)
        )
        fingerprint = (
            fields.tobytes(), geometry.tobytes(), table.tobytes(),
            outer, inner, value_stride, coord_stride,
        )
        fingerprints.add(fingerprint)
        exponents.add(float(fields[15]))
        prelookup_scales.add(float(fields[13]))
        max_u_values.add(int(geometry[2]))
        max_v_values.add(int(geometry[3]))
        max_z_values.add(int(geometry[4]))
        if state_index % 10 == 3:
            q_cap_state_count += 1

        output = np.full(outer * coord_stride, SENTINEL, dtype=np.float32)
        table_before = table.tobytes()
        inputs_before = [item.tobytes() for item in inputs]
        expected_active = oracle.reference_rows(
            fields, geometry, table, inputs, outer, inner, value_stride, coord_stride,
        )
        status = bridge.apply_raw(
            fields, geometry, table, inputs, output,
            outer=outer, inner=inner,
            value_stride=value_stride, coord_stride=coord_stride,
        )
        assert status == OK, f"random prepared state {state_index}: status {status}"

        expected = np.full_like(output, SENTINEL)
        for row in range(outer):
            expected[row * coord_stride:row * coord_stride + inner] = expected_active[row]
        _bitwise_equal(output, expected, f"random prepared state {state_index}")
        assert table.tobytes() == table_before, f"table changed in state {state_index}"
        for input_index, (actual, before) in enumerate(zip(inputs, inputs_before)):
            assert actual.tobytes() == before, f"input {input_index} changed in state {state_index}"

    assert len(fingerprints) == 100, "prepared/table states must be distinct"
    assert len(exponents) == 9, "zero, negative, and positive powf exponents must be covered"
    assert 0.0 in exponents and any(value < 0.0 for value in exponents)
    assert any(value < 0.0 for value in prelookup_scales)
    assert 0.0 in prelookup_scales and any(value > 0.0 for value in prelookup_scales)
    assert 0 in max_u_values and max(max_u_values) > 0
    assert 0 in max_v_values and max(max_v_values) > 0
    assert 0 in max_z_values and max(max_z_values) > 0
    assert q_cap_state_count == 10


def _basic_args():
    fields, geometry, table = _prepared()
    inputs = [np.asarray([0.4], dtype=np.float32),
              np.asarray([0.3], dtype=np.float32),
              np.asarray([0.5], dtype=np.float32),
              np.asarray([0.2], dtype=np.float32),
              np.asarray([0.7], dtype=np.float32)]
    output = np.asarray([SENTINEL], dtype=np.float32)
    return fields, geometry, table, inputs, output


@pytest.mark.parametrize("bad_field", [
    "nan_field", "inf_field", "nan_table", "inf_table",
    "bad_bounds", "nan_prelookup", "inf_prelookup", "nan_exponent", "inf_exponent",
])
def test_invalid_prepared_state_fails_without_writing(bridge, bad_field):
    fields, geometry, table, inputs, output = _basic_args()
    if bad_field == "nan_field":
        fields[0] = np.nan
    elif bad_field == "inf_field":
        fields[0] = np.inf
    elif bad_field == "nan_table":
        table[0] = np.nan
    elif bad_field == "inf_table":
        table[0] = np.inf
    elif bad_field == "bad_bounds":
        fields[9], fields[10] = np.float32(2.0), np.float32(1.0)
    elif bad_field == "nan_prelookup":
        fields[13] = np.nan
    elif bad_field == "inf_prelookup":
        fields[13] = np.inf
    elif bad_field == "nan_exponent":
        fields[15] = np.nan
    elif bad_field == "inf_exponent":
        fields[15] = np.inf
    before = output.tobytes()
    status = bridge.apply_raw(fields, geometry, table, inputs, output,
                              outer=1, inner=1, value_stride=1, coord_stride=1)
    assert status == INVALID_PREPARED
    assert output.tobytes() == before


@pytest.mark.parametrize("bad_input", [
    "nan_sample", "inf_sample", "nan_coord", "inf_coord",
    "q_int32_domain", "u_intermediate_overflow", "v_intermediate_overflow",
    "negative_exponent_zero_z",
])
def test_nonfinite_or_unsafe_numeric_paths_fail_without_writing(bridge, bad_input):
    fields, geometry, table, inputs, output = _basic_args()
    expected_status = NONFINITE_INPUT if bad_input in (
        "nan_sample", "inf_sample", "nan_coord", "inf_coord",
    ) else NONFINITE_INTERMEDIATE
    if bad_input == "nan_sample":
        inputs[1][0] = np.nan
    elif bad_input == "inf_sample":
        inputs[2][0] = np.inf
    elif bad_input == "nan_coord":
        inputs[4][0] = np.nan
    elif bad_input == "inf_coord":
        inputs[3][0] = np.inf
    elif bad_input == "q_int32_domain":
        fields[14] = np.float32(4.0e10)
        fields[15] = np.float32(1.0)
    elif bad_input == "u_intermediate_overflow":
        fields[5] = np.float32(np.finfo(np.float32).max)
        inputs[4][0] = np.float32(np.finfo(np.float32).max)
    elif bad_input == "v_intermediate_overflow":
        fields[7] = np.float32(np.finfo(np.float32).max)
        inputs[3][0] = np.float32(np.finfo(np.float32).max)
    else:
        fields[:5] = np.float32(0.0)
        fields[13] = np.float32(1.0)
        fields[14] = np.float32(0.5)
        fields[15] = np.float32(-0.5)
        inputs[0][0] = inputs[1][0] = inputs[2][0] = np.float32(0.0)
    before = output.tobytes()
    status = bridge.apply_raw(fields, geometry, table, inputs, output,
                              outer=1, inner=1, value_stride=1, coord_stride=1)
    assert status == expected_status
    assert output.tobytes() == before


@pytest.mark.parametrize(
    "case,expected_status",
    [
        ("short_fields", INVALID_ARGUMENT),
        ("short_geometry", INVALID_ARGUMENT),
        ("null_inputs_vector", INVALID_ARGUMENT),
        ("null_input_counts", INVALID_ARGUMENT),
        ("null_input_pointer", INVALID_ARGUMENT),
        ("short_input", SHORT_BUFFER),
        ("short_table", INVALID_PREPARED),
        ("short_destination", SHORT_BUFFER),
        ("bad_stride", STRIDE),
        ("bad_alignment", ALIGNMENT),
        ("destination_aliases_input", OVERLAP),
        ("destination_aliases_table", OVERLAP),
        ("destination_aliases_fields", OVERLAP),
        ("destination_aliases_geometry", OVERLAP),
        ("integer_span_overflow", OVERFLOW),
        ("active_resource_limit", RESOURCE_LIMIT),
    ],
)
def test_contract_failures_leave_destination_byte_identical(bridge, case, expected_status):
    fields, geometry, table, inputs, output = _basic_args()
    fields_arg = fields
    geometry_arg = geometry
    table_arg = table
    inputs_arg = inputs
    output_arg = output
    kwargs = dict(outer=1, inner=1, value_stride=1, coord_stride=1)
    call_kwargs = {}
    if case == "short_fields":
        call_kwargs["fields_count"] = 15
    elif case == "short_geometry":
        call_kwargs["geometry_count"] = 4
    elif case == "null_inputs_vector":
        call_kwargs["null_input_vector"] = True
    elif case == "null_input_counts":
        call_kwargs["null_input_counts"] = True
    elif case == "null_input_pointer":
        inputs_arg = [None, *inputs[1:]]
    elif case == "short_input":
        call_kwargs["input_counts"] = [0, 1, 1, 1, 1]
    elif case == "short_table":
        call_kwargs["table_count"] = 3
    elif case == "short_destination":
        call_kwargs["dest_count"] = 0
    elif case == "bad_stride":
        kwargs["value_stride"] = 0
    elif case == "bad_alignment":
        raw = np.zeros(16 * 4 + 1, dtype=np.uint8)
        fields_arg = np.ndarray((16,), dtype=np.float32, buffer=raw, offset=1)
        fields_arg[:] = fields
    elif case == "destination_aliases_input":
        output_arg = inputs[0]
    elif case == "destination_aliases_table":
        output_arg = table
    elif case == "destination_aliases_fields":
        output_arg = fields
    elif case == "destination_aliases_geometry":
        output_arg = geometry.view(np.float32)
    elif case == "integer_span_overflow":
        call_kwargs["dest_count"] = SIZE_MAX
    elif case == "active_resource_limit":
        kwargs.update(outer=1, inner=(1 << 26) + 1,
                      value_stride=(1 << 26) + 1, coord_stride=(1 << 26) + 1)
    before = output_arg.tobytes()
    status = bridge.apply_raw(fields_arg, geometry_arg, table_arg, inputs_arg, output_arg,
                              **kwargs, **call_kwargs)
    assert status == expected_status
    assert output_arg.tobytes() == before


def test_dimension_and_geometry_guards_fail_without_writing(bridge):
    fields, geometry, table, inputs, output = _basic_args()
    before = output.tobytes()
    status = bridge.apply_raw(fields, geometry, table, inputs, output,
                              outer=SIZE_MAX, inner=2, value_stride=2, coord_stride=2)
    assert status == OVERFLOW
    assert output.tobytes() == before

    status = bridge.apply_raw(fields, geometry, table, inputs, output,
                              outer=0, inner=1, value_stride=1, coord_stride=1)
    assert status == DIMENSIONS
    assert output.tobytes() == before

    # Use a coordinate bound that exceeds max_u to exercise invalid prepared
    # geometry without any destination write.
    invalid_fields = fields.copy()
    invalid_fields[10] = np.float32(4.0)
    status = bridge.apply_raw(invalid_fields, geometry, table, inputs, output,
                              outer=1, inner=1, value_stride=1, coord_stride=1)
    assert status == INVALID_PREPARED
    assert output.tobytes() == before


def test_missing_and_null_prepared_arguments_are_invalid_argument(bridge):
    fields, geometry, table, inputs, output = _basic_args()
    before = output.tobytes()
    for args in (
        (None, geometry, table, inputs, output),
        (fields, None, table, inputs, output),
        (fields, geometry, table, inputs, None),
    ):
        status = bridge.apply_raw(*args, outer=1, inner=1, value_stride=1, coord_stride=1)
        assert status == INVALID_ARGUMENT
        assert output.tobytes() == before

    # A missing prepared table is a malformed prepared state under the fixed
    # CPU-subset status contract; missing streaming buffers remain invalid args.
    status = bridge.apply_raw(fields, geometry, None, inputs, output,
                              outer=1, inner=1, value_stride=1, coord_stride=1)
    assert status == INVALID_PREPARED
    assert output.tobytes() == before
