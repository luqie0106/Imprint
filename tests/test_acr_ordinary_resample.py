"""Independent source/formal bridge checks for ordinary ACR resampling."""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools/acr_ordinary_resample_bridge.cpp"
SEPARABLE_SOURCE = NATIVE / "src/acr_resample_separable.cpp"
ROW_SOURCE = NATIVE / "src/acr_resample_row.cpp"
VERTICAL_SOURCE = NATIVE / "src/acr_resample_vertical.cpp"
AXIS_SOURCE = NATIVE / "src/acr_resample_axis.cpp"
SEPARABLE_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_resample_separable.cpp.o"
ROW_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_resample_row.cpp.o"
VERTICAL_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_resample_vertical.cpp.o"
AXIS_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_resample_axis.cpp.o"
SENTINEL = np.float32(-71.25)
F32_PTR = C.POINTER(C.c_float)

sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import ordinary_resample_oracle as oracle  # noqa: E402


class AxisArgs(C.Structure):
    _fields_ = [
        ("kind", C.c_int),
        ("integer_offset", C.c_int32),
        ("factory_translation", C.c_double),
        ("origin", C.c_int64),
        ("step", C.c_int64),
        ("lower", C.c_int64),
        ("upper", C.c_int64),
        ("coefficients", F32_PTR),
        ("coefficient_count", C.c_size_t),
        ("tap_count", C.c_size_t),
        ("stride", C.c_size_t),
    ]


class Request(C.Structure):
    _fields_ = [
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
        ("vertical", AxisArgs),
        ("horizontal", AxisArgs),
        ("output", F32_PTR),
        ("output_count", C.c_size_t),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compile_bridge(kind: str, output: Path) -> tuple[list[str], list[Path]]:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the ordinary-resample bridge"
    assert BRIDGE_SOURCE.is_file(), f"missing test bridge: {BRIDGE_SOURCE}"
    command = [
        compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", "-ffp-contract=off",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", str(NATIVE / "src"), "-I", str(NATIVE / "include"),
        str(BRIDGE_SOURCE),
    ]
    if kind == "source":
        inputs = [SEPARABLE_SOURCE, ROW_SOURCE, VERTICAL_SOURCE, AXIS_SOURCE]
    elif kind == "formal_object":
        inputs = [SEPARABLE_OBJECT, ROW_OBJECT, VERTICAL_OBJECT, AXIS_OBJECT]
        dependencies = [
            (SEPARABLE_OBJECT, [SEPARABLE_SOURCE,
                                NATIVE / "src/acr_resample_separable.hpp",
                                NATIVE / "src/acr_resample_axis.hpp",
                                NATIVE / "src/acr_resample_row.hpp",
                                NATIVE / "src/acr_resample_vertical.hpp"]),
            (ROW_OBJECT, [ROW_SOURCE, NATIVE / "src/acr_resample_row.hpp"]),
            (VERTICAL_OBJECT, [VERTICAL_SOURCE,
                               NATIVE / "src/acr_resample_vertical.hpp"]),
            (AXIS_OBJECT, [AXIS_SOURCE, NATIVE / "src/acr_resample_axis.hpp"]),
        ]
        for obj, required in dependencies:
            assert obj.is_file(), f"missing formal renderer object: {obj.name}"
            newest_input = max(path.stat().st_mtime_ns for path in required)
            assert obj.stat().st_mtime_ns >= newest_input, (
                f"rebuild formal renderer: stale object {obj.name}"
            )
    else:
        raise AssertionError("ACR_ORDINARY_RESAMPLE_BRIDGE must be source or formal_object")
    command.extend(str(path) for path in inputs)
    command.extend(["-o", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    return command, inputs


class Bridge:
    def __init__(self, path: Path, kind: str, inputs: list[Path]):
        self.path = path
        self.kind = kind
        self.inputs = inputs
        self.sha256 = _sha256(path)
        self.library = C.CDLL(str(path))
        self.function = self.library.acr_test_ordinary_resample
        self.function.argtypes = [C.POINTER(Request)]
        self.function.restype = C.c_int
        self.factory_function = self.library.acr_test_ordinary_resample_factory_translation
        self.factory_function.argtypes = [C.POINTER(Request)]
        self.factory_function.restype = C.c_int
        self.legacy_function = self.library.acr_test_legacy_resample_separable
        self.legacy_function.argtypes = [C.POINTER(Request)]
        self.legacy_function.restype = C.c_int
        self.alias_function = self.library.acr_test_ordinary_resample_result_alias
        self.alias_function.argtypes = [F32_PTR, C.c_size_t]
        self.alias_function.restype = C.c_int
        self.alias_failure_function = self.library.acr_test_ordinary_resample_result_alias_failure
        self.alias_failure_function.argtypes = [F32_PTR, C.c_size_t]
        self.alias_failure_function.restype = C.c_int
        self.mutation_guard_function = self.library.acr_test_ordinary_failure_mutation_guard
        self.mutation_guard_function.argtypes = []
        self.mutation_guard_function.restype = C.c_int

    def call(self, source: np.ndarray, source_top: int, source_left: int,
             destination: tuple[int, int, int, int], vertical: dict,
             horizontal: dict, *, source_count: int | None = None,
             output_count: int | None = None,
             output_storage_count: int | None = None,
             source_pointer: F32_PTR | None = None,
             output_pointer: F32_PTR | None = None,
             factory: bool = False,
             legacy: bool = False,
             sentinel: float = float(SENTINEL)) -> tuple[int, np.ndarray]:
        dt, dl, dh, dw = destination
        image = np.ascontiguousarray(source, dtype=np.float32)
        required_output_count = int(dh) * int(dw) * image.shape[2]
        count = required_output_count if output_count is None else output_count
        storage_count = count if output_storage_count is None else output_storage_count
        output = np.full(max(storage_count, 0), sentinel, dtype=np.float32)
        pinned_coefficients = []

        def as_axis(axis: dict) -> AxisArgs:
            coefficients = axis.get("coefficients")
            coefficient_array = (None if coefficients is None else
                                 np.ascontiguousarray(coefficients, dtype=np.float32))
            if coefficient_array is not None:
                pinned_coefficients.append(coefficient_array)
            ptr = (F32_PTR() if coefficient_array is None else
                   coefficient_array.ctypes.data_as(F32_PTR))
            if "coefficient_pointer" in axis:
                ptr = axis["coefficient_pointer"]
            shape = (0, 0) if coefficient_array is None else coefficient_array.shape
            return AxisArgs(
                int(axis.get("kind", oracle.KIND_RESAMPLE)),
                int(axis.get("integer_offset", 0)),
                float(axis.get("factory_translation", 0.0)),
                int(axis.get("origin", 0)), int(axis.get("step", 0)),
                int(axis.get("lower", 0)), int(axis.get("upper", 0)),
                ptr,
                int(axis.get("coefficient_count", 0 if coefficient_array is None
                            else coefficient_array.size)),
                int(axis.get("taps", 0 if coefficient_array is None
                            else axis.get("taps", coefficient_array.shape[1]))),
                int(axis.get("stride", 0 if coefficient_array is None
                            else coefficient_array.shape[1])),
            )

        v_axis = as_axis(vertical)
        h_axis = as_axis(horizontal)
        source_ptr = image.ctypes.data_as(F32_PTR)
        if source_pointer is not None:
            source_ptr = source_pointer
        output_ptr = output.ctypes.data_as(F32_PTR) if output.size else F32_PTR()
        if output_pointer is not None:
            output_ptr = output_pointer
        request = Request(
            source_ptr,
            image.size if source_count is None else source_count,
            source_top, source_left, image.shape[0], image.shape[1], image.shape[2],
            dt, dl, dh, dw, v_axis, h_axis, output_ptr, count,
        )
        function = (self.legacy_function if legacy else
                    self.factory_function if factory else self.function)
        status = function(C.byref(request))
        assert status != 99, (
            "bridge detected that a failed C++ executor call changed the result "
            "vector bitwise"
        )
        if count == required_output_count and output.size == required_output_count:
            output = output.reshape(int(dh), int(dw), image.shape[2])
        return status, output


@pytest.fixture(scope="module")
def bridge(request, tmp_path_factory) -> Bridge:
    kind = os.environ.get("ACR_ORDINARY_RESAMPLE_BRIDGE", "source")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_ordinary_" + kind) / ("bridge" + suffix)
    validation_dir_setting = os.environ.get("ACR_ORDINARY_RESAMPLE_VALIDATION_DIR")
    if validation_dir_setting is None:
        validation_dir = tmp_path_factory.mktemp("acr_ordinary_validation_" + kind)
    else:
        if not validation_dir_setting.strip():
            raise AssertionError("ACR_ORDINARY_RESAMPLE_VALIDATION_DIR must be nonempty")
        validation_dir = Path(validation_dir_setting)
    command, inputs = _compile_bridge(kind, output)
    result = Bridge(output, kind, inputs)

    def record_validation() -> None:
        validation_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "oracle_version": oracle.ORACLE_VERSION,
            "bridge_kind": kind,
            "artifact_sha256": result.sha256,
            "artifact_path": str(result.path),
            "oracle_sha256": _sha256(ROOT / "tests/acr_support/ordinary_resample_oracle.py"),
            "test_sha256": _sha256(ROOT / "tests/test_acr_ordinary_resample.py"),
            "linked_input_hashes": {str(path.relative_to(ROOT)): _sha256(path)
                                    for path in inputs},
            "bridge_source_sha256": _sha256(BRIDGE_SOURCE),
            "compiler_command": command,
            "pytest_failed_count_at_fixture_teardown": request.session.testsfailed,
            "pytest_test_node_count": len(request.session.items),
            "pytest_test_nodes": [item.nodeid for item in request.session.items
                                  if item.path == ROOT / "tests/test_acr_ordinary_resample.py"],
        }
        record_path = validation_dir / f"{kind}-bridge-validation.json"
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    request.addfinalizer(record_validation)
    return result


def _rows(taps: int, phase_bias: int = 0) -> np.ndarray:
    stride = ((taps + 7) // 8) * 8
    bases = {
        1: [0.75],
        2: [0.25, 0.5],
        3: [0.125, -0.25, 0.75],
        4: [0.125, -0.25, 0.5, 0.75],
        6: [0.25, 0.125, -0.5, 0.375, 0.625, 0.25],
        8: [0.125, 0.25, 0.125, -0.25, 0.375, 0.125, 0.25, 0.125],
        10: [0.125, -0.0625, 0.25, 0.03125, -0.125,
             0.375, 0.0625, -0.03125, 0.25, 0.125],
    }
    base = np.asarray(bases[taps], dtype=np.float32)
    rows = np.zeros((128, stride), dtype=np.float32)
    for phase in range(128):
        rows[phase, :taps] = base
        rows[phase, phase_bias % taps] = np.float32(
            rows[phase, phase_bias % taps]
            + np.float32((phase + 1) * (2.0 ** -13)))
    return rows


def _axis(kind: int, *, taps: int = 4, origin: int = 0,
          step: int = 0, lower: int = -(1 << 62),
          upper: int = (1 << 62), coefficients: np.ndarray | None = None,
          integer_offset: int = 0, **extra) -> dict:
    result = {"kind": kind, "origin": origin, "step": step,
              "lower": lower, "upper": upper, "integer_offset": integer_offset}
    if kind == oracle.KIND_RESAMPLE:
        result.update(taps=taps,
                      coefficients=_rows(taps) if coefficients is None else coefficients)
    result.update(extra)
    return result


def _assert_success(bridge, source, top, left, destination, vertical, horizontal,
                    *, factory: bool = False, legacy: bool = False):
    image = np.ascontiguousarray(source, dtype=np.float32)
    before = image.view(np.uint32).copy()
    expected = oracle.reference(image, top, left, destination, vertical, horizontal)
    status, actual = bridge.call(image, top, left, destination, vertical,
                                 horizontal, factory=factory, legacy=legacy)
    assert status == 0
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert np.array_equal(image.view(np.uint32), before)
    return actual


def _identity_poison(kind: int = oracle.KIND_IDENTITY, **extra) -> dict:
    axis = _axis(kind, origin=-(1 << 63), step=(1 << 63) - 1,
                 lower=(1 << 62), upper=-(1 << 62), coefficients=None,
                 taps=0, coefficient_pointer=C.cast(C.c_void_p(0x1000), F32_PTR),
                 coefficient_count=(1 << 64) - 1, stride=(1 << 64) - 1)
    axis.update(extra)
    return axis


def _translation_axis(offset: int, **extra) -> dict:
    axis = _axis(
        oracle.KIND_INTEGER_TRANSLATION, origin=-(1 << 63),
        step=(1 << 63) - 1, lower=(1 << 62), upper=-(1 << 62),
        integer_offset=offset,
        coefficient_pointer=C.cast(C.c_void_p(0x1000), F32_PTR),
        coefficient_count=(1 << 64) - 1, stride=(1 << 64) - 1)
    axis.update(extra)
    return axis


def test_oracle_version_and_identity_literal_probes_are_pinned(bridge):
    assert oracle.ORACLE_VERSION == "acr-ordinary-resample-vh-fma32-2026-10-23-v1"
    literal_cases = [
        ((0x3F800000, 0x40000000, 0x40400000, 0x40800000), (0x40400000, 0x40800000)),
        ((0xBF800000, 0xC0000000, 0xC0400000, 0xC0800000), (0xC0400000, 0xC0800000)),
        ((0x00000001, 0x00000002, 0x00000003, 0x00000004), (0x00000003, 0x00000004)),
        ((0x80000000, 0x3F000000, 0x80000000, 0x3E800000), (0x80000000, 0x3E800000)),
        ((0x3EAAAAAB, 0x3F2AAAAB, 0x3F800000, 0x3F99999A), (0x3F800000, 0x3F99999A)),
        ((0xBF000000, 0x3F000000, 0xBF800000, 0x40000000), (0xBF800000, 0x40000000)),
        ((0x3F000001, 0x3F000002, 0x3F000003, 0x3F000004), (0x3F000003, 0x3F000004)),
        ((0x00800000, 0x80800000, 0x7F000000, 0xFF000000), (0x7F000000, 0xFF000000)),
        ((0x3F7FFFFF, 0x3F800001, 0xBF7FFFFF, 0xBF800001), (0xBF7FFFFF, 0xBF800001)),
        ((0x41200000, 0xC1200000, 0x41A00000, 0xC1A00000), (0x41A00000, 0xC1A00000)),
        ((0x3DCCCCCD, 0x3E4CCCCD, 0x3E99999A, 0x3ECCCCCD), (0x3E99999A, 0x3ECCCCCD)),
        ((0xBF19999A, 0x3F19999A, 0xBF4CCCCD, 0x3F4CCCCD), (0xBF4CCCCD, 0x3F4CCCCD)),
        ((0x3F3504F3, 0x3F9E377A, 0x3FC90FDB, 0x402DF854), (0x3FC90FDB, 0x402DF854)),
        ((0x3F666666, 0x3F8CCCCD, 0x3FA66666, 0x3FCCCCCD), (0x3FA66666, 0x3FCCCCCD)),
        ((0xBEAAAAAB, 0x3EAAAAAB, 0xBFAAAAAB, 0x3FAAAAAB), (0xBFAAAAAB, 0x3FAAAAAB)),
        ((0x3F000000, 0xBF000000, 0x3F800000, 0xBF800000), (0x3F800000, 0xBF800000)),
    ]
    assert len(literal_cases) >= 16
    # Convert the pinned words to values only after preserving the independent
    # literal expected bits; each case exercises a cropped two-dimensional copy.
    for words, expected_bits in literal_cases:
        source = np.asarray(words, dtype=np.uint32).view(np.float32).reshape(2, 2, 1)
        identity = _identity_poison()
        actual = _assert_success(bridge, source, -5, -9, (-4, -9, 1, 2),
                                 identity, identity)
        assert tuple(int(word) for word in actual.view(np.uint32).ravel()) == expected_bits


def test_identity_paths_copy_crops_and_preserve_signed_zero_subnormals(bridge):
    words = np.asarray([
        0x80000000, 0x00000001, 0x3F800000,
        0x00000002, 0x80000000, 0xBF800000,
        0x3F000000, 0x00000003, 0x40000000,
        0x00000004, 0x80000000, 0x40400000,
    ], dtype=np.uint32)
    source = words.view(np.float32).reshape(3, 4, 1)
    identity = _identity_poison()
    actual = _assert_success(bridge, source, -7, -11, (-6, -10, 2, 3),
                             identity, identity)
    assert np.array_equal(actual.view(np.uint32).ravel(),
                          np.asarray([0xBF800000, 0x3F000000, 0x00000003,
                                      0x00000004, 0x80000000, 0x40400000],
                                     dtype=np.uint32))


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("orientation", ["vertical_identity", "horizontal_identity"])
def test_identity_in_each_axis_with_general_other_axis(bridge, channels, orientation):
    rng = np.random.default_rng(20261023 + channels)
    source = rng.uniform(-0.8, 1.7, (14, 19, channels)).astype(np.float32)
    if orientation == "vertical_identity":
        vertical = _identity_poison()
        horizontal = _axis(oracle.KIND_RESAMPLE, taps=4,
                           origin=(-8 * oracle.FIXED_ONE) + (127 << 25),
                           step=oracle.FIXED_ONE // 2,
                           lower=-50 * oracle.FIXED_ONE,
                           upper=50 * oracle.FIXED_ONE)
    else:
        vertical = _axis(oracle.KIND_RESAMPLE, taps=6,
                         origin=(-9 * oracle.FIXED_ONE) + (127 << 25),
                         step=oracle.FIXED_ONE // 3,
                         lower=-50 * oracle.FIXED_ONE,
                         upper=50 * oracle.FIXED_ONE)
        horizontal = _identity_poison()
    _assert_success(bridge, source, -9, -13, (-8, -12, 8, 9),
                    vertical, horizontal)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("vertical_kind", [0, 1, 2])
@pytest.mark.parametrize("horizontal_kind", [0, 1, 2])
def test_all_nine_ordinary_axis_kind_combinations(
    bridge, channels, vertical_kind, horizontal_kind
):
    rng = np.random.default_rng(20261023 + channels * 100 + vertical_kind * 10
                                + horizontal_kind)
    source = rng.uniform(-0.75, 1.25, (15, 18, channels)).astype(np.float32)

    def make(kind, start, offset):
        if kind == oracle.KIND_IDENTITY:
            return _identity_poison(integer_offset=offset)
        if kind == oracle.KIND_INTEGER_TRANSLATION:
            return _translation_axis(offset)
        return _axis(oracle.KIND_RESAMPLE, taps=6,
                     origin=(start + 3) * oracle.FIXED_ONE + (127 << 25),
                     step=oracle.FIXED_ONE // 2,
                     lower=(start - 10) * oracle.FIXED_ONE,
                     upper=(start + 30) * oracle.FIXED_ONE)

    vertical = make(vertical_kind, -12, 2)
    horizontal = make(horizontal_kind, -17, -2)
    _assert_success(bridge, source, -12, -17, (-8, -13, 3, 4),
                    vertical, horizontal)


def test_integer_translation_copies_signed_zero_and_subnormal_bits(bridge):
    words = np.asarray([
        0x00000001, 0x80000000, 0x3F800000, 0x00000002, 0xBF800000,
        0x00000003, 0x80000000, 0x40000000, 0x00000004, 0x40400000,
        0x00000005, 0x80000000, 0x40800000, 0x00000006, 0x40A00000,
        0x00000007, 0x80000000, 0x40C00000, 0x00000008, 0x40E00000,
    ], dtype=np.uint32)
    source = words.view(np.float32).reshape(4, 5, 1)
    vertical = _translation_axis(-1)  # source row = destination row + 1
    horizontal = _translation_axis(1)  # source column = destination column - 1
    actual = _assert_success(bridge, source, -5, -9, (-4, -6, 2, 3),
                             vertical, horizontal)
    expected_source = source[2:4, 2:5]
    assert np.array_equal(actual.view(np.uint32), expected_source.view(np.uint32))


@pytest.mark.parametrize(("factory_translation", "axis_name", "destination",
                          "expected"), [
    (2, "vertical", (-1, -4, 2, 3), [76, 77, 78, 88, 89, 90]),
    (-2, "horizontal", (-2, -1, 2, 3), [41, 42, 43, 53, 54, 55]),
])
def test_axis_factory_translation_sign_reaches_executor_literal_probe(
    bridge, factory_translation, axis_name, destination, expected
):
    source = np.arange(10 * 12, dtype=np.float32).reshape(10, 12, 1)
    top, left = -5, -8
    if axis_name == "vertical":
        vertical = _translation_axis(-factory_translation,
                                     factory_translation=factory_translation)
        horizontal = _identity_poison()
    else:
        vertical = _identity_poison()
        horizontal = _translation_axis(-factory_translation,
                                       factory_translation=factory_translation)
    actual = _assert_success(bridge, source, top, left, destination,
                             vertical, horizontal, factory=True)
    assert np.array_equal(actual.ravel(), np.asarray(expected, dtype=np.float32))


def test_integer_translation_identity_offset_is_ignored(bridge):
    source = np.arange(6 * 8, dtype=np.float32).reshape(6, 8, 1)
    identity = _identity_poison(integer_offset=-(1 << 31))
    actual = _assert_success(bridge, source, -7, -12, (-6, -11, 2, 3),
                             identity, identity)
    assert np.array_equal(actual.ravel(), np.asarray([9, 10, 11, 17, 18, 19],
                                                    dtype=np.float32))


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("taps", [4, 6, 8, 10])
def test_general_axes_match_independent_oracle_with_negative_origin_and_crop(
    bridge, channels, taps
):
    rng = np.random.default_rng(20261023 + channels * 100 + taps)
    source = rng.uniform(-1.5, 2.0, (18, 23, channels)).astype(np.float32)
    vertical = _axis(
        oracle.KIND_RESAMPLE, taps=taps,
        origin=-11 * oracle.FIXED_ONE + (127 << 25),
        step=oracle.FIXED_ONE // 2,
        lower=-100 * oracle.FIXED_ONE, upper=100 * oracle.FIXED_ONE)
    horizontal = _axis(
        oracle.KIND_RESAMPLE, taps=taps,
        origin=-17 * oracle.FIXED_ONE + (127 << 25),
        step=oracle.FIXED_ONE // 3,
        lower=-100 * oracle.FIXED_ONE, upper=100 * oracle.FIXED_ONE)
    _assert_success(bridge, source, -11, -17, (-8, -13, 7, 8),
                    vertical, horizontal)


def test_legacy_two_general_axis_entry_matches_the_versioned_oracle(bridge):
    rng = np.random.default_rng(2026102304)
    source = rng.uniform(-1.25, 1.75, (17, 21, 3)).astype(np.float32)
    vertical = _axis(oracle.KIND_RESAMPLE, taps=6,
                     origin=-10 * oracle.FIXED_ONE + (127 << 25),
                     step=oracle.FIXED_ONE // 2,
                     lower=-40 * oracle.FIXED_ONE,
                     upper=40 * oracle.FIXED_ONE)
    horizontal = _axis(oracle.KIND_RESAMPLE, taps=8,
                       origin=-15 * oracle.FIXED_ONE + (126 << 25),
                       step=oracle.FIXED_ONE // 3,
                       lower=-40 * oracle.FIXED_ONE,
                       upper=40 * oracle.FIXED_ONE)
    _assert_success(bridge, source, -10, -15, (-7, -11, 6, 7),
                    vertical, horizontal, legacy=True)


def test_legacy_two_general_axis_failure_checks_vector_atomicity(bridge):
    source = np.ones((10, 12, 1), dtype=np.float32)
    source[-1, -1, 0] = np.nan
    vertical = _axis(oracle.KIND_RESAMPLE, taps=4)
    horizontal = _axis(oracle.KIND_RESAMPLE, taps=6)
    _assert_failure_unchanged(bridge, source, 0, 0, (1, 1, 2, 3),
                              vertical, horizontal, 7, legacy=True)


def test_noncontiguous_manual_coefficient_rows_are_pinned_and_copied(bridge):
    source = np.arange(15 * 20, dtype=np.float32).reshape(15, 20, 1) / 11
    backing = _rows(8)
    strided_rows = np.empty((128, 16), dtype=np.float32)
    strided_rows[:, 0::2] = backing
    strided_rows[:, 1::2] = -17.0
    rows = strided_rows[:, 0::2]
    assert not rows.flags.c_contiguous
    vertical = _axis(oracle.KIND_RESAMPLE, taps=8,
                     origin=-7 * oracle.FIXED_ONE + (127 << 25),
                     step=oracle.FIXED_ONE // 2,
                     lower=-20 * oracle.FIXED_ONE,
                     upper=20 * oracle.FIXED_ONE,
                     coefficients=rows)
    horizontal = _translation_axis(-1)
    _assert_success(bridge, source, -7, -10, (-4, -6, 3, 5),
                    vertical, horizontal)


def test_identity_ignores_null_poisoned_coefficients_and_unused_geometry(bridge):
    source = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    vertical = _identity_poison(coefficient_pointer=F32_PTR())
    horizontal = _identity_poison(coefficient_pointer=C.cast(C.c_void_p(0x4), F32_PTR))
    _assert_success(bridge, source, -4, -8, (-3, -7, 1, 4), vertical, horizontal)


def test_identity_copy_path_rejects_nonfinite_source_atomically(bridge):
    source = np.ones((3, 5, 1), dtype=np.float32)
    source[2, 4, 0] = np.nan
    identity = _identity_poison()
    _assert_failure_unchanged(bridge, source, -2, -4, (-1, -3, 1, 2),
                              identity, identity, 7)


def test_vertical_accumulation_starts_at_positive_zero_and_skips_zero_weights(bridge):
    source = np.zeros((8, 7, 1), dtype=np.float32)
    source.view(np.uint32)[2, :, 0] = 0x80000000
    rows = np.zeros((128, 8), dtype=np.float32)
    rows[:, 0] = 1.0
    rows[:, 1] = -0.0
    vertical = _axis(oracle.KIND_RESAMPLE, taps=4,
                     origin=2 * oracle.FIXED_ONE, step=0, lower=0,
                     upper=3 * oracle.FIXED_ONE + 0xFE000000,
                     coefficients=rows)
    horizontal = _identity_poison()
    actual = _assert_success(bridge, source, 0, -3, (2, -3, 1, 4),
                             vertical, horizontal)
    assert np.all(actual.view(np.uint32) == 0)


def test_phase_127_upper_clamp_and_general_h_identity_mixed_paths(bridge):
    source = np.arange(13 * 17, dtype=np.float32).reshape(13, 17, 1) / 7
    rows = np.zeros((128, 8), dtype=np.float32)
    rows[:, 0] = 1.0
    rows[127, 0] = 2.0
    vertical = _axis(oracle.KIND_RESAMPLE, taps=4,
                     origin=-oracle.FIXED_ONE + (127 << 25),
                     step=oracle.FIXED_ONE,
                     lower=-20 * oracle.FIXED_ONE,
                     upper=20 * oracle.FIXED_ONE,
                     coefficients=rows)
    horizontal = _identity_poison()
    _assert_success(bridge, source, -6, -8, (-5, -7, 3, 6),
                    vertical, horizontal)


def test_phase_127_advances_to_phase_zero_at_the_next_integer(bridge):
    source = np.ones((12, 9, 1), dtype=np.float32)
    rows = np.zeros((128, 8), dtype=np.float32)
    rows[127, 0] = 2.0
    rows[0, 0] = 3.0
    vertical = _axis(oracle.KIND_RESAMPLE, taps=4,
                     origin=-6 * oracle.FIXED_ONE + (127 << 25),
                     step=1 << 25, lower=-20 * oracle.FIXED_ONE,
                     upper=20 * oracle.FIXED_ONE, coefficients=rows)
    horizontal = _identity_poison()
    actual = _assert_success(bridge, source, -6, -4, (0, -4, 2, 3),
                             vertical, horizontal)
    assert np.array_equal(actual.ravel(), np.asarray([2, 2, 2, 3, 3, 3],
                                                     dtype=np.float32))


def test_result_vector_alias_reads_before_replacement(bridge):
    output = np.full(2, SENTINEL, dtype=np.float32)
    status = bridge.alias_function(output.ctypes.data_as(F32_PTR), output.size)
    assert status == 0
    assert np.array_equal(output.view(np.uint32),
                          np.asarray([0x40800000, 0x40A00000], dtype=np.uint32))


def test_result_vector_alias_failure_keeps_original_contents(bridge):
    output = np.full(6, SENTINEL, dtype=np.float32)
    status = bridge.alias_failure_function(output.ctypes.data_as(F32_PTR), output.size)
    assert status == 5
    assert np.array_equal(output, np.arange(6, dtype=np.float32))


def test_bridge_atomicity_guard_detects_a_private_signed_zero_mutation(bridge):
    # The C++ bridge's test-only self-probe changes its own -0 to +0. It must
    # return 99; the production executor and its source remain untouched.
    assert bridge.mutation_guard_function() == 99


def test_seeded_bounded_randomized_80_cases_per_selected_bridge(bridge):
    rng = np.random.default_rng(0xAC20261023)
    for case_index in range(80):
        channels = (1, 3)[case_index % 2]
        height = int(rng.integers(9, 19))
        width = int(rng.integers(12, 25))
        source_top = int(rng.integers(-23, -3))
        source_left = int(rng.integers(-31, -5))
        source = rng.uniform(-1.0, 1.5, (height, width, channels)).astype(np.float32)

        def choose_axis(axis_start: int, extent: int, identity_allowed: bool,
                        phase_bias: int) -> dict:
            kind = int(rng.integers(0, 3))
            if identity_allowed and kind == oracle.KIND_IDENTITY:
                return _identity_poison()
            if kind == oracle.KIND_INTEGER_TRANSLATION:
                offset = int(rng.choice([-2, -1, 1, 2]))
                return _translation_axis(offset)
            taps = int(rng.choice([4, 6, 8]))
            rows = _rows(taps, phase_bias)
            step = int(rng.choice([-1, 0, 1])) * oracle.FIXED_ONE
            phase = int(rng.integers(0, 128))
            origin = ((axis_start + int(rng.integers(0, extent - taps + 1)))
                      * oracle.FIXED_ONE + (phase << 25))
            lower = (axis_start - 5) * oracle.FIXED_ONE
            upper = (axis_start + extent + 5) * oracle.FIXED_ONE
            return _axis(oracle.KIND_RESAMPLE, taps=taps, origin=origin,
                         step=step, lower=lower, upper=upper,
                         coefficients=rows)

        vertical = choose_axis(source_top, height, True, case_index % 8)
        horizontal = choose_axis(source_left, width, True, (case_index + 3) % 8)
        destination_top = source_top + 4
        destination_left = source_left + 4
        destination_height = int(rng.integers(1, min(height - 6, 7) + 1))
        destination_width = int(rng.integers(1, min(width - 6, 8) + 1))
        _assert_success(
            bridge, source, source_top, source_left,
            (destination_top, destination_left,
             destination_height, destination_width), vertical, horizontal)


def _assert_failure_unchanged(bridge, source, top, left, destination,
                              vertical, horizontal, expected_status, **options):
    status, output = bridge.call(source, top, left, destination, vertical,
                                 horizontal, **options)
    assert status == expected_status
    assert np.all(output.view(np.uint32) == SENTINEL.view(np.uint32))


def test_invalid_kind_and_missing_general_coefficients_are_atomic(bridge):
    source = np.ones((8, 9, 1), dtype=np.float32)
    valid = _axis(oracle.KIND_RESAMPLE, taps=4)
    identity = _identity_poison()
    _assert_failure_unchanged(bridge, source, 0, 0, (0, 0, 2, 2),
                              _axis(99), identity, 12)
    _assert_failure_unchanged(bridge, source, 0, 0, (0, 0, 2, 2),
                              _axis(oracle.KIND_RESAMPLE, coefficients=None,
                                    coefficient_count=0), identity, 3)
    _assert_failure_unchanged(bridge, source, 0, 0, (0, 0, 2, 2),
                              valid, _axis(oracle.KIND_RESAMPLE,
                                           coefficients=None,
                                           coefficient_count=0), 4)


@pytest.mark.parametrize("axis_name", ["vertical", "horizontal"])
def test_integer_translation_out_of_source_bounds_fails_atomically(bridge, axis_name):
    source = np.ones((6, 7, 1), dtype=np.float32)
    translation = _translation_axis(1)
    identity = _identity_poison()
    vertical, horizontal = ((translation, identity) if axis_name == "vertical"
                            else (identity, translation))
    destination = (-4, -8, 1, 2) if axis_name == "vertical" else (-4, -8, 2, 1)
    _assert_failure_unchanged(bridge, source, -4, -8, destination,
                              vertical, horizontal, 5)


@pytest.mark.parametrize("axis_name", ["vertical", "horizontal"])
def test_integer_translation_relocated_rectangle_overflow_is_atomic(
    bridge, axis_name
):
    source = np.ones((4, 4, 1), dtype=np.float32)
    maximum = (1 << 31) - 1
    identity = _identity_poison()
    translation = _translation_axis(5)  # relocated source rectangle overflows int32
    if axis_name == "vertical":
        top, left, destination = maximum - 4, 0, (maximum - 2, 0, 1, 1)
        vertical, horizontal = translation, identity
    else:
        top, left, destination = 0, maximum - 4, (0, maximum - 2, 1, 1)
        vertical, horizontal = identity, translation
    _assert_failure_unchanged(bridge, source, top, left, destination,
                              vertical, horizontal, 6)


def test_integer_translation_checks_whole_source_view_relocation(bridge):
    source = np.ones((4, 4, 1), dtype=np.float32)
    minimum = -(1 << 31)
    vertical = _translation_axis(-5)
    horizontal = _identity_poison()
    # Requested source starts at INT32_MIN+5 and is in range, while shifting
    # the complete source view by stored_offset=-5 would underflow INT32.
    _assert_failure_unchanged(bridge, source, minimum + 4, 0,
                              (minimum, 0, 1, 1), vertical, horizontal, 6)


@pytest.mark.parametrize("failure", ["short_source", "zero_extent", "rect_overflow",
                                      "destination_extent_overflow",
                                      "identity_outside", "source_window_too_short",
                                      "geometry_no_overlap", "coordinate_overflow",
                                      "short_output"])
def test_invalid_requests_and_ranges_preserve_output(bridge, failure):
    source = np.ones((8, 9, 1), dtype=np.float32)
    identity = _identity_poison()
    general = _axis(oracle.KIND_RESAMPLE, taps=4, origin=0, step=0,
                    lower=-2 * oracle.FIXED_ONE,
                    upper=20 * oracle.FIXED_ONE)
    opts = {}
    destination = (0, 0, 2, 2)
    vertical, horizontal = general, general
    expected = 2
    if failure == "short_source":
        opts["source_count"] = source.size - 1
    elif failure == "zero_extent":
        destination = (0, 0, 0, 2)
    elif failure == "rect_overflow":
        source_top = (1 << 31) - 1
        _assert_failure_unchanged(bridge, source, source_top, 0,
                                  (0, 0, 2, 2), general, general, 2)
        return
    elif failure == "destination_extent_overflow":
        destination = ((1 << 31) - 1, 0, 1, 1)
    elif failure == "identity_outside":
        vertical, horizontal = identity, identity
        destination = (9, 0, 1, 2)
        expected = 5
    elif failure == "source_window_too_short":
        vertical = _axis(oracle.KIND_RESAMPLE, taps=10)
        expected = 5
    elif failure == "geometry_no_overlap":
        vertical = _axis(oracle.KIND_RESAMPLE, taps=4,
                         lower=20 * oracle.FIXED_ONE,
                         upper=21 * oracle.FIXED_ONE)
        expected = 5
    elif failure == "coordinate_overflow":
        vertical = _axis(oracle.KIND_RESAMPLE, taps=4,
                         origin=(1 << 63) - 1, step=1,
                         lower=-4 * oracle.FIXED_ONE,
                         upper=10 * oracle.FIXED_ONE)
        destination = (1, 0, 2, 2)
        expected = 6
    elif failure == "short_output":
        opts["output_count"] = 3
    _assert_failure_unchanged(bridge, source, 0, 0, destination,
                              vertical, horizontal, expected, **opts)


def test_bridge_rejects_huge_and_unaligned_output_spans_without_writing(bridge):
    source = np.ones((8, 9, 1), dtype=np.float32)
    identity = _identity_poison()
    status, output = bridge.call(
        source, 0, 0, (0, 0, 2, 2), identity, identity,
        output_count=(1 << 64) - 1, output_storage_count=4)
    assert status == 1
    assert np.all(output == SENTINEL)

    raw = np.full(4 * 4 + 1, 0xA5, dtype=np.uint8)
    before = raw.copy()
    unaligned = C.cast(C.c_void_p(raw.ctypes.data + 1), F32_PTR)
    status, output = bridge.call(
        source, 0, 0, (0, 0, 2, 2), identity, identity,
        output_pointer=unaligned)
    assert status == 1
    assert np.all(output == SENTINEL)
    assert np.array_equal(raw, before)


@pytest.mark.parametrize("failure", ["nan_source", "inf_source", "nan_weight",
                                      "nan_padding", "infinite_weight"])
def test_nonfinite_source_or_general_coefficients_fail_atomically(bridge, failure):
    source = np.ones((9, 11, 1), dtype=np.float32)
    vertical = _axis(oracle.KIND_RESAMPLE, taps=4)
    horizontal = _axis(oracle.KIND_RESAMPLE, taps=4)
    expected = 7 if failure in ("nan_source", "inf_source") else 8
    if failure == "nan_source":
        source[0, 0, 0] = np.nan
    elif failure == "inf_source":
        source[-1, -1, 0] = np.inf
    else:
        field = "nan_padding" if failure == "nan_padding" else "nan_weight"
        rows = _rows(4)
        rows[127, 7 if field == "nan_padding" else 2] = (
            np.inf if failure == "infinite_weight" else np.nan)
        vertical["coefficients"] = rows
    _assert_failure_unchanged(bridge, source, 0, 0, (0, 0, 2, 2),
                              vertical, horizontal, expected)


def test_finite_inputs_that_overflow_intermediate_fail_atomically(bridge):
    source = np.full((8, 8, 1), np.finfo(np.float32).max, dtype=np.float32)
    rows = np.zeros((128, 8), dtype=np.float32)
    rows[:, :4] = np.finfo(np.float32).max
    vertical = _axis(oracle.KIND_RESAMPLE, taps=4,
                     lower=0, upper=4 * oracle.FIXED_ONE + 0xFE000000,
                     coefficients=rows)
    horizontal = _identity_poison()
    _assert_failure_unchanged(bridge, source, 0, 0, (0, 0, 1, 3),
                              vertical, horizontal, 9)

