"""Independent matrix -> owning plan -> pixel execution checks."""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools/acr_resample_plan_bridge.cpp"
PLAN_SOURCE = NATIVE / "src/acr_resample_plan.cpp"
MATRIX_SOURCE = NATIVE / "src/acr_resample_matrix.cpp"
AXIS_SOURCE = NATIVE / "src/acr_resample_axis.cpp"
COEFFICIENT_SOURCE = NATIVE / "src/acr_resample_coefficients.cpp"
SEPARABLE_SOURCE = NATIVE / "src/acr_resample_separable.cpp"
ROW_SOURCE = NATIVE / "src/acr_resample_row.cpp"
VERTICAL_SOURCE = NATIVE / "src/acr_resample_vertical.cpp"
OBJECT_ROOT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src"
OBJECTS = {
    "plan": OBJECT_ROOT / "acr_resample_plan.cpp.o",
    "matrix": OBJECT_ROOT / "acr_resample_matrix.cpp.o",
    "axis": OBJECT_ROOT / "acr_resample_axis.cpp.o",
    "coefficients": OBJECT_ROOT / "acr_resample_coefficients.cpp.o",
    "separable": OBJECT_ROOT / "acr_resample_separable.cpp.o",
    "row": OBJECT_ROOT / "acr_resample_row.cpp.o",
    "vertical": OBJECT_ROOT / "acr_resample_vertical.cpp.o",
}
SENTINEL = np.float32(-71.25)
STATUS_OK = 0
STATUS_PLAN_INVALID_BOUNDS = 1
STATUS_PLAN_MATRIX_FAILURE = 2
STATUS_PLAN_UNSUPPORTED_SKEW = 3
STATUS_PLAN_UNSUPPORTED_SCALE = 4
STATUS_PLAN_COEFFICIENT_FAILURE = 5
STATUS_PLAN_AXIS_FAILURE = 6
STATUS_EXEC_BASE = 100
STATUS_SOURCE_WINDOW = 5
KIND_RESAMPLE = 0
KIND_IDENTITY = 1
KIND_INTEGER_TRANSLATION = 2
METADATA_WORDS = 26
COEFFICIENT_CAPACITY = 8192
F32_PTR = C.POINTER(C.c_float)
U32_PTR = C.POINTER(C.c_uint32)
U64_PTR = C.POINTER(C.c_uint64)

sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import ordinary_resample_oracle as pixel_oracle  # noqa: E402
import resample_axis_instruction_oracle as axis_oracle  # noqa: E402
import resample_coefficients_oracle as coefficient_oracle  # noqa: E402
import resample_matrix_oracle as matrix_oracle  # noqa: E402


class PlanArgs(C.Structure):
    _fields_ = [
        ("matrix", C.POINTER(C.c_double)),
        ("bounds", C.POINTER(C.c_int32)),
        ("source", F32_PTR),
        ("source_count", C.c_size_t),
        ("source_top", C.c_int32),
        ("source_left", C.c_int32),
        ("source_height", C.c_size_t),
        ("source_width", C.c_size_t),
        ("channels", C.c_size_t),
        ("destination_top", C.c_int32),
        ("destination_left", C.c_int32),
        ("destination_height", C.c_size_t),
        ("destination_width", C.c_size_t),
        ("output", F32_PTR),
        ("output_count", C.c_size_t),
        ("metadata", U64_PTR),
        ("metadata_count", C.c_size_t),
        ("vertical_coefficient_bits", U32_PTR),
        ("vertical_coefficient_capacity", C.c_size_t),
        ("horizontal_coefficient_bits", U32_PTR),
        ("horizontal_coefficient_capacity", C.c_size_t),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiler() -> str:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the ordinary plan bridge"
    return compiler


def _compile_bridge(kind: str, output: Path) -> tuple[list[str], list[Path]]:
    compiler = _compiler()
    command = [
        compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", "-ffp-contract=off", "-fPIC",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", str(NATIVE / "src"), "-I", str(NATIVE / "include"),
        str(BRIDGE_SOURCE),
    ]
    if kind == "source":
        inputs = [PLAN_SOURCE, MATRIX_SOURCE, AXIS_SOURCE,
                  COEFFICIENT_SOURCE, SEPARABLE_SOURCE, ROW_SOURCE,
                  VERTICAL_SOURCE]
    elif kind == "formal_object":
        inputs = [OBJECTS[name] for name in
                  ("plan", "matrix", "axis", "coefficients", "separable",
                   "row", "vertical")]
        dependencies = {
            "plan": [PLAN_SOURCE, NATIVE / "src/acr_resample_plan.hpp",
                     NATIVE / "src/acr_resample_matrix.hpp",
                     NATIVE / "src/acr_resample_axis.hpp",
                     NATIVE / "src/acr_resample_coefficients.hpp",
                     NATIVE / "src/acr_resample_separable.hpp"],
            "matrix": [MATRIX_SOURCE, NATIVE / "src/acr_resample_matrix.hpp"],
            "axis": [AXIS_SOURCE, NATIVE / "src/acr_resample_axis.hpp"],
            "coefficients": [COEFFICIENT_SOURCE,
                             NATIVE / "src/acr_resample_coefficients.hpp"],
            "separable": [SEPARABLE_SOURCE,
                          NATIVE / "src/acr_resample_separable.hpp",
                          NATIVE / "src/acr_resample_row.hpp",
                          NATIVE / "src/acr_resample_vertical.hpp",
                          NATIVE / "src/acr_resample_axis.hpp"],
            "row": [ROW_SOURCE, NATIVE / "src/acr_resample_row.hpp"],
            "vertical": [VERTICAL_SOURCE,
                         NATIVE / "src/acr_resample_vertical.hpp"],
        }
        for name, obj in zip(
                ("plan", "matrix", "axis", "coefficients", "separable",
                 "row", "vertical"), inputs):
            assert obj.is_file(), f"missing formal renderer object: {obj}"
            newest_input = max(path.stat().st_mtime_ns
                               for path in dependencies[name])
            assert obj.stat().st_mtime_ns >= newest_input, (
                f"rebuild formal renderer: stale object {obj.name}"
            )
    else:
        raise AssertionError(
            "ACR_RESAMPLE_PLAN_BRIDGE must be source or formal_object"
        )
    command.extend(str(path) for path in inputs)
    command.extend(["-o", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True,
                   cwd=ROOT)
    return command, inputs


class Bridge:
    def __init__(self, path: Path):
        self.path = path
        self.sha256 = _sha256(path)
        self.library = C.CDLL(str(path))
        self.plan_function = self.library.acr_test_ordinary_resample_plan
        self.plan_function.argtypes = [C.POINTER(PlanArgs)]
        self.plan_function.restype = C.c_int
        self.alias_function = self.library.acr_test_ordinary_resample_plan_alias
        self.alias_function.argtypes = [F32_PTR, C.c_size_t]
        self.alias_function.restype = C.c_int
        self.alias_failure_function = (
            self.library.acr_test_ordinary_resample_plan_alias_failure
        )
        self.alias_failure_function.argtypes = [F32_PTR, C.c_size_t]
        self.alias_failure_function.restype = C.c_int

    def call(self, matrix, bounds, source: np.ndarray, source_top: int,
             source_left: int, destination: tuple[int, int, int, int],
             *, sentinel: float = float(SENTINEL)):
        dt, dl, dh, dw = map(int, destination)
        image = np.ascontiguousarray(source, dtype=np.float32)
        assert image.ndim == 3
        matrix_array = (C.c_double * 9)(*map(float, matrix))
        bounds_array = (C.c_int32 * 4)(*map(int, bounds))
        output = np.full(dh * dw * image.shape[2], sentinel,
                         dtype=np.float32)
        metadata = np.full(METADATA_WORDS, 0xA5A5A5A5A5A5A5A5,
                           dtype=np.uint64)
        vertical_bits = np.full(COEFFICIENT_CAPACITY, 0xDEADBEEF,
                                dtype=np.uint32)
        horizontal_bits = np.full(COEFFICIENT_CAPACITY, 0xDEADBEEF,
                                  dtype=np.uint32)
        source_before = image.view(np.uint32).tobytes()
        request = PlanArgs(
            matrix_array, bounds_array, image.ctypes.data_as(F32_PTR),
            image.size, source_top, source_left, image.shape[0], image.shape[1],
            image.shape[2], dt, dl, dh, dw,
            output.ctypes.data_as(F32_PTR) if output.size else F32_PTR(),
            output.size, metadata.ctypes.data_as(U64_PTR), metadata.size,
            vertical_bits.ctypes.data_as(U32_PTR), vertical_bits.size,
            horizontal_bits.ctypes.data_as(U32_PTR), horizontal_bits.size,
        )
        status = int(self.plan_function(C.byref(request)))
        assert status != 99, "C++ bridge detected failed-call mutation"
        assert image.view(np.uint32).tobytes() == source_before, (
            "plan execution modified its input pixels"
        )
        return (status, output.reshape(dh, dw, image.shape[2]),
                tuple(int(word) for word in metadata),
                tuple(int(word) for word in vertical_bits),
                tuple(int(word) for word in horizontal_bits))

    def alias(self) -> tuple[int, np.ndarray]:
        output = np.full(4, SENTINEL, dtype=np.float32)
        status = int(self.alias_function(output.ctypes.data_as(F32_PTR),
                                         output.size))
        assert status != 99, "C++ bridge detected input/result alias corruption"
        return status, output

    def alias_failure(self) -> tuple[int, np.ndarray]:
        output = np.full(12, SENTINEL, dtype=np.float32)
        status = int(self.alias_failure_function(
            output.ctypes.data_as(F32_PTR), output.size))
        assert status != 99, "C++ bridge detected failed alias mutation"
        return status, output


@pytest.fixture(scope="module")
def bridge(request, tmp_path_factory) -> Bridge:
    kind = os.environ.get("ACR_RESAMPLE_PLAN_BRIDGE", "source")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    build_dir = tmp_path_factory.mktemp("acr_resample_plan_" + kind)
    output = build_dir / ("plan_bridge" + suffix)
    command, inputs = _compile_bridge(kind, output)
    result = Bridge(output)
    validation_dir_setting = os.environ.get(
        "ACR_RESAMPLE_PLAN_VALIDATION_DIR")

    def record_validation() -> None:
        if validation_dir_setting is None:
            return
        if not validation_dir_setting.strip():
            raise AssertionError(
                "ACR_RESAMPLE_PLAN_VALIDATION_DIR must be nonempty"
            )
        validation_dir = Path(validation_dir_setting)
        validation_dir.mkdir(parents=True, exist_ok=True)
        record_path = validation_dir / f"{kind}-plan-validation.json"
        assert not record_path.exists(), (
            f"refusing to overwrite existing evidence: {record_path}"
        )
        record = {
            "oracle_version": coefficient_oracle.ORACLE_VERSION,
            "matrix_oracle_version": matrix_oracle.ORACLE_VERSION,
            "axis_oracle_version": axis_oracle.REFERENCE_ID,
            "pixel_oracle_version": pixel_oracle.ORACLE_VERSION,
            "bridge_kind": kind,
            "bridge_sha256": result.sha256,
            "bridge_path": str(result.path),
            "bridge_source_sha256": _sha256(BRIDGE_SOURCE),
            "coefficient_oracle_sha256": _sha256(
                ROOT / "tests/acr_support/resample_coefficients_oracle.py"),
            "test_sha256": _sha256(ROOT / "tests/test_acr_resample_plan.py"),
            "linked_input_hashes": {
                str(path.relative_to(ROOT)): _sha256(path) for path in inputs
            },
            "compiler_command": command,
            "pytest_failed_count_at_fixture_teardown": request.session.testsfailed,
            "pytest_test_nodes": [
                item.nodeid for item in request.session.items
                if item.path == ROOT / "tests/test_acr_resample_plan.py"
            ],
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n",
                               encoding="utf-8")

    request.addfinalizer(record_validation)
    return result


def _signed64(value: int) -> int:
    word = int(value) & ((1 << 64) - 1)
    return word - (1 << 64) if word & (1 << 63) else word


def _signed32(value: int) -> int:
    word = int(value) & ((1 << 32) - 1)
    return word - (1 << 32) if word & (1 << 31) else word


def _word(value: int) -> int:
    return int(value) & ((1 << 64) - 1)


def _table_metadata(table: dict | None) -> tuple[int, ...]:
    if table is None:
        return (0, 0, 0, 0, 0, 0, 0)
    values = table["values"]
    return (
        coefficient_oracle.f64_bits(float(table["q"])),
        coefficient_oracle.f64_bits(float(table["scale"])),
        coefficient_oracle.f64_bits(float(table["parameter_a"])),
        coefficient_oracle.f64_bits(float(table["parameter_b"])),
        int(table["tap_count"]), int(table["stride"]), len(values),
    )


def _axis_expectation(translation: float, scale: float, cross: float,
                      start: int, end: int) -> tuple[int, dict | None,
                                                     tuple[int, ...],
                                                     dict | None,
                                                     tuple[int, ...]]:
    status, first = axis_oracle.prepare(
        translation, scale, cross, start, end, 1)
    if status == axis_oracle.STATUS_UNSUPPORTED_SKEW:
        return STATUS_PLAN_UNSUPPORTED_SKEW, None, (), None, ()
    if status != axis_oracle.STATUS_OK or first is None:
        return STATUS_PLAN_AXIS_FAILURE, None, (), None, ()

    kind = int(first[0])
    table = None
    if kind == axis_oracle.KIND_RESAMPLE:
        step = _signed64(first[2])
        if step <= 0:
            return STATUS_PLAN_UNSUPPORTED_SCALE, None, (), None, ()
        quantized_scale = float(step) * float.fromhex("0x1p-32")
        q = 1.0 / quantized_scale
        try:
            table = coefficient_oracle.reference_table(q)
        except (ValueError, ArithmeticError):
            return STATUS_PLAN_COEFFICIENT_FAILURE, None, (), None, ()
        half_taps = int(table["tap_count"]) // 2
        status, words = axis_oracle.prepare(
            translation, scale, cross, start, end, half_taps)
        if status == axis_oracle.STATUS_UNSUPPORTED_SKEW:
            return STATUS_PLAN_UNSUPPORTED_SKEW, None, (), None, ()
        if status != axis_oracle.STATUS_OK or words is None:
            return STATUS_PLAN_AXIS_FAILURE, None, (), None, ()
        coeff = np.asarray(table["values"], dtype=np.float32).reshape(
            128, int(table["stride"]))
        axis = {
            "kind": KIND_RESAMPLE,
            "origin": _signed64(words[1]),
            "step": _signed64(words[2]),
            "lower": _signed64(words[3]),
            "upper": _signed64(words[4]),
            "taps": int(table["tap_count"]),
            "coefficients": coeff,
        }
        coefficient_bits = tuple(
            coefficient_oracle.f32_bits(value)
            for value in table["values"]
        )
    else:
        words = first
        offset = _signed32(words[5])
        axis = {
            "kind": kind,
            "integer_offset": offset,
        }
        coefficient_bits = ()

    metadata = (
        kind, _word(_signed32(words[5])), _word(words[1]), _word(words[2]),
        _word(words[3]), _word(words[4]),
    )
    return STATUS_OK, axis, metadata, table, coefficient_bits


def expected_plan(matrix, bounds) -> dict:
    """Return independent metadata and pixel-oracle axes for one plan input."""
    top, left, bottom, right = map(int, bounds)
    vertical_span = bottom - top
    horizontal_span = right - left
    if (vertical_span <= 0 or horizontal_span <= 0
            or vertical_span > (1 << 31) - 1
            or horizontal_span > (1 << 31) - 1):
        return {"status": STATUS_PLAN_INVALID_BOUNDS}

    matrix_status, tuples = matrix_oracle.decompose_ordinary_resample_matrix(
        matrix)
    if matrix_status != matrix_oracle.ResampleMatrixStatus.OK:
        return {"status": STATUS_PLAN_MATRIX_FAILURE}
    assert tuples is not None
    v_status, vertical, v_meta, v_table, v_bits = _axis_expectation(
        tuples[0], tuples[1], tuples[2], top, bottom)
    if v_status != STATUS_OK:
        return {"status": v_status}
    h_status, horizontal, h_meta, h_table, h_bits = _axis_expectation(
        tuples[3], tuples[4], tuples[5], left, right)
    if h_status != STATUS_OK:
        return {"status": h_status}
    metadata = v_meta + h_meta + _table_metadata(v_table) + _table_metadata(h_table)
    return {
        "status": STATUS_OK,
        "metadata": metadata,
        "vertical": vertical,
        "horizontal": horizontal,
        "vertical_table": v_table,
        "horizontal_table": h_table,
        "vertical_coeff_bits": v_bits,
        "horizontal_coeff_bits": h_bits,
        "tuples": tuples,
    }


def _assert_success_matches_oracles(bridge: Bridge, matrix, bounds,
                                    image: np.ndarray, source_top: int,
                                    source_left: int,
                                    destination: tuple[int, int, int, int]):
    expected = expected_plan(matrix, bounds)
    assert expected["status"] == STATUS_OK
    status, output, metadata, vertical_bits, horizontal_bits = bridge.call(
        matrix, bounds, image, source_top, source_left, destination)
    assert status == STATUS_OK
    assert metadata == expected["metadata"]
    assert vertical_bits[:len(expected["vertical_coeff_bits"])] == \
        expected["vertical_coeff_bits"]
    assert horizontal_bits[:len(expected["horizontal_coeff_bits"])] == \
        expected["horizontal_coeff_bits"]
    for actual, required in (
            (vertical_bits, expected["vertical_coeff_bits"]),
            (horizontal_bits, expected["horizontal_coeff_bits"])):
        assert all(value == 0xDEADBEEF for value in actual[len(required):]), (
            "bridge wrote beyond the reported coefficient table"
        )
    reference = pixel_oracle.reference(
        np.ascontiguousarray(image, dtype=np.float32), source_top, source_left,
        destination, expected["vertical"], expected["horizontal"])
    np.testing.assert_array_equal(output.view(np.uint32),
                                  reference.view(np.uint32))
    return expected, output


def test_identity_and_integer_translation_combinations_need_no_coefficients(bridge):
    source_top, source_left = -7, 9
    image = np.arange(12 * 14 * 3, dtype=np.float32).reshape(12, 14, 3)
    destinations = (-1, 0, 2)
    for vertical_offset in destinations:
        for horizontal_offset in destinations:
            matrix = (1.0, 0.0, float(vertical_offset),
                      0.0, 1.0, float(horizontal_offset),
                      0.0, 0.0, 1.0)
            destination = (source_top + 4, source_left + 5, 3, 4)
            bounds = (destination[0], destination[1],
                      destination[0] + destination[2],
                      destination[1] + destination[3])
            expected, _ = _assert_success_matches_oracles(
                bridge, matrix, bounds, image, source_top, source_left,
                destination)
            assert expected["vertical_coeff_bits"] == ()
            assert expected["horizontal_coeff_bits"] == ()
            assert expected["metadata"][18:26] == (0,) * 8


def test_literal_translation_sign_and_tuple_witness(bridge):
    matrix = (1.0, 0.0, 3.0,
              0.0, 1.0, -2.0,
              0.0, 0.0, 1.0)
    expected = expected_plan(matrix, (-3, 6, 1, 10))
    assert expected["status"] == STATUS_OK
    assert expected["tuples"] == (-3.0, 1.0, 0.0, 2.0, 1.0, 0.0)
    assert expected["metadata"][:6] == (
        KIND_INTEGER_TRANSLATION, 3, 0, 0, 0, 0)
    assert expected["metadata"][6:12] == (
        KIND_INTEGER_TRANSLATION, (1 << 64) - 2, 0, 0, 0, 0)

    image = np.arange(12 * 12, dtype=np.float32).reshape(12, 12, 1)
    destination = (-3, 6, 3, 4)
    bounds = (-3, 6, 0, 10)
    actual, _ = _assert_success_matches_oracles(
        bridge, matrix, bounds, image, -6, 3, destination)
    assert actual["metadata"][:6] == expected["metadata"][:6]


def test_mixed_general_scale_fractional_translation_matches_all_oracles(bridge):
    source_top, source_left = -13, 17
    image = np.random.default_rng(4012).normal(size=(64, 68, 3)).astype(np.float32)
    matrix = (2.0, 0.0, 0.25,
              0.0, 1.5, -0.5,
              0.0, 0.0, 1.0)
    destination = (source_top + 20, source_left + 19, 8, 9)
    bounds = (destination[0], destination[1],
              destination[0] + destination[2],
              destination[1] + destination[3])
    expected, _ = _assert_success_matches_oracles(
        bridge, matrix, bounds, image, source_top, source_left, destination)
    assert expected["vertical"]["kind"] == KIND_RESAMPLE
    assert expected["horizontal"]["kind"] == KIND_RESAMPLE
    assert expected["vertical_table"]["tap_count"] > 0
    assert expected["horizontal_table"]["tap_count"] > 0


def test_q_uses_reciprocal_of_quantized_scale_with_hard_probe(bridge):
    scale = 7.3
    matrix = (scale, 0.0, 0.125,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0)
    source_top, source_left = 4, -8
    destination = (source_top + 24, source_left + 20, 5, 6)
    bounds = (destination[0], destination[1],
              destination[0] + destination[2],
              destination[1] + destination[3])
    expected = expected_plan(matrix, bounds)
    assert expected["status"] == STATUS_OK
    quantized_q_bits = expected["metadata"][12]
    raw_scale = float(expected["tuples"][1])
    wrong_raw_reciprocal_bits = coefficient_oracle.f64_bits(1.0 / raw_scale)
    assert quantized_q_bits != wrong_raw_reciprocal_bits
    quantized_table = expected["vertical_table"]
    wrong_table = coefficient_oracle.reference_table(1.0 / raw_scale)
    assert any(
        coefficient_oracle.f32_bits(left) != coefficient_oracle.f32_bits(right)
        for left, right in zip(quantized_table["values"], wrong_table["values"])
    )

    image = np.random.default_rng(73).normal(size=(64, 64, 1)).astype(np.float32)
    _assert_success_matches_oracles(
        bridge, matrix, bounds, image, source_top, source_left, destination)


@pytest.mark.parametrize("seed", range(64))
def test_sixty_four_supported_affine_matrices_match_pixel_oracle(bridge, seed):
    rng = np.random.default_rng(seed + 20261023)
    vertical_matrix_scale = float(rng.uniform(0.55, 2.4))
    horizontal_matrix_scale = float(rng.uniform(0.55, 2.4))
    vertical_translation = float(rng.uniform(-1.5, 1.5))
    horizontal_translation = float(rng.uniform(-1.5, 1.5))
    matrix = (vertical_matrix_scale, 0.0, vertical_translation,
              0.0, horizontal_matrix_scale, horizontal_translation,
              0.0, 0.0, 1.0)
    source_top, source_left = -21, 11
    image = rng.normal(size=(64, 64, 3)).astype(np.float32)
    destination = (source_top + 23, source_left + 24, 5, 6)
    bounds = (destination[0], destination[1],
              destination[0] + destination[2],
              destination[1] + destination[3])
    _assert_success_matches_oracles(
        bridge, matrix, bounds, image, source_top, source_left, destination)


@pytest.mark.parametrize(("name", "matrix", "bounds", "expected_status"), [
    ("singular-matrix", (1.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 1.0),
     (2, 3, 8, 9), STATUS_PLAN_MATRIX_FAILURE),
    ("skew", (1.0, 0.0, 0.0,
              0.25, 1.0, 0.0,
              0.0, 0.0, 1.0),
     (2, 3, 8, 9), STATUS_PLAN_UNSUPPORTED_SKEW),
    ("negative-general-scale", (-2.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0),
     (2, 3, 8, 9), STATUS_PLAN_UNSUPPORTED_SCALE),
    ("zero-quantized-scale", (float(1 << 34), 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0),
     (2, 3, 8, 9), STATUS_PLAN_UNSUPPORTED_SCALE),
    ("horizontal-coefficient-cap", (2.0, 0.0, 0.0,
                                     0.0, 1.0e-5, 0.0,
                                     0.0, 0.0, 1.0),
     (2, 3, 8, 9), STATUS_PLAN_COEFFICIENT_FAILURE),
    ("empty-bounds", (1.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       0.0, 0.0, 1.0),
     (2, 3, 2, 9), STATUS_PLAN_INVALID_BOUNDS),
    ("oversized-bounds", (1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0),
     (-(1 << 31), 0, (1 << 31) - 1, 4), STATUS_PLAN_INVALID_BOUNDS),
])
def test_prepare_failures_leave_plan_metadata_and_result_unchanged(
        bridge, name, matrix, bounds, expected_status):
    del name
    image = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    destination = (5, 7, 2, 2)
    status, output, metadata, vertical_bits, horizontal_bits = bridge.call(
        matrix, bounds, image, 0, 0, destination)
    assert status == expected_status
    assert np.all(output.view(np.uint32) ==
                  np.asarray([SENTINEL], dtype=np.float32).view(np.uint32)[0])
    assert metadata == (0xA5A5A5A5A5A5A5A5,) * METADATA_WORDS
    assert vertical_bits == (0xDEADBEEF,) * COEFFICIENT_CAPACITY
    assert horizontal_bits == (0xDEADBEEF,) * COEFFICIENT_CAPACITY


def test_source_window_failure_preserves_prepared_plan_and_result(bridge):
    matrix = (1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0)
    destination = (-4, 4, 1, 2)
    bounds = (-4, 4, -3, 6)
    image = np.arange(4 * 4, dtype=np.float32).reshape(4, 4, 1)
    status, output, metadata, vertical_bits, horizontal_bits = bridge.call(
        matrix, bounds, image, -2, 3, destination)
    assert status == STATUS_EXEC_BASE + STATUS_SOURCE_WINDOW
    assert np.all(output.view(np.uint32) ==
                  np.asarray([SENTINEL], dtype=np.float32).view(np.uint32)[0])
    assert metadata == (0xA5A5A5A5A5A5A5A5,) * METADATA_WORDS
    assert vertical_bits == (0xDEADBEEF,) * COEFFICIENT_CAPACITY
    assert horizontal_bits == (0xDEADBEEF,) * COEFFICIENT_CAPACITY


def test_plan_copy_and_move_execute_and_result_can_alias_input(bridge):
    status, output = bridge.alias()
    assert status == STATUS_OK
    np.testing.assert_array_equal(
        output.view(np.uint32),
        np.asarray([5.0, 6.0, 9.0, 10.0], dtype=np.float32).view(np.uint32))

    failed_status, unchanged = bridge.alias_failure()
    assert failed_status == STATUS_SOURCE_WINDOW
    np.testing.assert_array_equal(
        unchanged.view(np.uint32),
        np.arange(12, dtype=np.float32).view(np.uint32))


def test_coefficient_oracle_has_stable_fma_and_float_normalization_witnesses():
    assert coefficient_oracle.REFERENCE_ID == \
        "acr-012e3a20-float-coefficient-rows-arm64-fma32-v1"
    table = coefficient_oracle.reference_table(7.3)
    assert (table["tap_count"], table["stride"]) == (4, 8)
    assert coefficient_oracle.f64_bits(table["parameter_a"]) == \
        0x3FDC444444444444
    assert coefficient_oracle.f64_bits(table["parameter_b"]) == \
        0x3FD1DDDDDDDDDDDE
    assert coefficient_oracle.f32_bits(table["values"][0]) == 0x3D96C16C
    assert coefficient_oracle.f32_bits(table["values"][1]) == 0x3F5A4FA5
    assert all(
        coefficient_oracle.f32_bits(table["values"][phase * 8 + tap]) == 0
        for phase in range(128) for tap in range(4, 8)
    )
