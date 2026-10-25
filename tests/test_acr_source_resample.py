"""Independent oracle checks for source mode 2 plus ordinary resampling."""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools/acr_source_resample_bridge.cpp"
SOURCE_RESAMPLE_SOURCE = NATIVE / "src/acr_source_resample.cpp"
REQUEST_SOURCE = NATIVE / "src/acr_resample_request.cpp"
PLAN_SOURCE = NATIVE / "src/acr_resample_plan.cpp"
MATRIX_SOURCE = NATIVE / "src/acr_resample_matrix.cpp"
AXIS_SOURCE = NATIVE / "src/acr_resample_axis.cpp"
COEFFICIENT_SOURCE = NATIVE / "src/acr_resample_coefficients.cpp"
SEPARABLE_SOURCE = NATIVE / "src/acr_resample_separable.cpp"
ROW_SOURCE = NATIVE / "src/acr_resample_row.cpp"
VERTICAL_SOURCE = NATIVE / "src/acr_resample_vertical.cpp"
OBJECT_ROOT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src"
OBJECTS = {
    "source": OBJECT_ROOT / "acr_source_resample.cpp.o",
    "request": OBJECT_ROOT / "acr_resample_request.cpp.o",
    "plan": OBJECT_ROOT / "acr_resample_plan.cpp.o",
    "matrix": OBJECT_ROOT / "acr_resample_matrix.cpp.o",
    "axis": OBJECT_ROOT / "acr_resample_axis.cpp.o",
    "coefficients": OBJECT_ROOT / "acr_resample_coefficients.cpp.o",
    "separable": OBJECT_ROOT / "acr_resample_separable.cpp.o",
    "row": OBJECT_ROOT / "acr_resample_row.cpp.o",
    "vertical": OBJECT_ROOT / "acr_resample_vertical.cpp.o",
}
SENTINEL = np.float32(-71.25)
F32_PTR = C.POINTER(C.c_float)

sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
from test_acr_resample_plan import expected_plan  # noqa: E402
import ordinary_resample_oracle as pixel_oracle  # noqa: E402
import resample_request_oracle as request_oracle  # noqa: E402


class SourceMode2Args(C.Structure):
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
        ("matrix", C.POINTER(C.c_double)),
        ("plan_bounds", C.POINTER(C.c_int32)),
        ("output", F32_PTR),
        ("output_count", C.c_size_t),
    ]


def _compile_bridge(kind: str, output: Path) -> tuple[list[str], list[Path]]:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the source-resample bridge"
    command = [
        compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", "-ffp-contract=off", "-fPIC",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", str(NATIVE / "src"), "-I", str(NATIVE / "include"),
        str(BRIDGE_SOURCE),
    ]
    if kind == "source":
        inputs = [
            SOURCE_RESAMPLE_SOURCE, REQUEST_SOURCE, PLAN_SOURCE,
            MATRIX_SOURCE, AXIS_SOURCE, COEFFICIENT_SOURCE,
            SEPARABLE_SOURCE, ROW_SOURCE, VERTICAL_SOURCE,
        ]
    elif kind == "formal_object":
        inputs = [OBJECTS[name] for name in (
            "source", "request", "plan", "matrix", "axis",
            "coefficients", "separable", "row", "vertical")]
        dependencies = {
            "source": [SOURCE_RESAMPLE_SOURCE,
                       NATIVE / "src/acr_source_resample.hpp",
                       NATIVE / "src/acr_resample_request.hpp",
                       NATIVE / "src/acr_resample_plan.hpp"],
            "request": [REQUEST_SOURCE,
                        NATIVE / "src/acr_resample_request.hpp",
                        NATIVE / "src/acr_resample_plan.hpp"],
            "plan": [PLAN_SOURCE, NATIVE / "src/acr_resample_plan.hpp",
                     NATIVE / "src/acr_resample_matrix.hpp",
                     NATIVE / "src/acr_resample_axis.hpp",
                     NATIVE / "src/acr_resample_coefficients.hpp",
                     NATIVE / "src/acr_resample_separable.hpp"],
            "matrix": [MATRIX_SOURCE,
                       NATIVE / "src/acr_resample_matrix.hpp"],
            "axis": [AXIS_SOURCE, NATIVE / "src/acr_resample_axis.hpp"],
            "coefficients": [COEFFICIENT_SOURCE,
                             NATIVE / "src/acr_resample_coefficients.hpp"],
            "separable": [SEPARABLE_SOURCE,
                          NATIVE / "src/acr_resample_separable.hpp",
                          NATIVE / "src/acr_resample_axis.hpp",
                          NATIVE / "src/acr_resample_row.hpp",
                          NATIVE / "src/acr_resample_vertical.hpp"],
            "row": [ROW_SOURCE, NATIVE / "src/acr_resample_row.hpp"],
            "vertical": [VERTICAL_SOURCE,
                         NATIVE / "src/acr_resample_vertical.hpp"],
        }
        for name, obj in zip(
                ("source", "request", "plan", "matrix", "axis",
                 "coefficients", "separable", "row", "vertical"), inputs):
            assert obj.is_file(), f"missing formal renderer object: {obj}"
            newest_input = max(path.stat().st_mtime_ns
                               for path in dependencies[name])
            assert obj.stat().st_mtime_ns >= newest_input, (
                f"rebuild formal renderer: stale object {obj.name}"
            )
    else:
        raise AssertionError(
            "ACR_SOURCE_RESAMPLE_BRIDGE must be source or formal_object"
        )
    command.extend(str(path) for path in inputs)
    command.extend(["-o", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True,
                   cwd=ROOT)
    return command, inputs


class Bridge:
    def __init__(self, path: Path):
        self.path = path
        self.library = C.CDLL(str(path))
        self.function = self.library.acr_test_ordinary_source_mode2
        self.function.argtypes = [C.POINTER(SourceMode2Args)]
        self.function.restype = C.c_int
        self.alias_function = self.library.acr_test_ordinary_source_mode2_alias
        self.alias_function.argtypes = [C.POINTER(SourceMode2Args)]
        self.alias_function.restype = C.c_int
        self.physical_window_function = (
            self.library.acr_test_ordinary_source_physical_window_baseline
        )
        self.physical_window_function.argtypes = [C.POINTER(SourceMode2Args)]
        self.physical_window_function.restype = C.c_int

    def call(self, source: np.ndarray, source_top: int, source_left: int,
             destination: tuple[int, int, int, int], matrix,
             plan_bounds: tuple[int, int, int, int] | None = None, *,
             source_count: int | None = None, alias: bool = False,
             physical_window: bool = False):
        image = np.ascontiguousarray(source, dtype=np.float32)
        dt, dl, dh, dw = map(int, destination)
        if plan_bounds is None:
            plan_bounds = (dt, dl, dt + dh, dl + dw)
        matrix_array = (C.c_double * 9)(*map(float, matrix))
        bounds_array = (C.c_int32 * 4)(*map(int, plan_bounds))
        output = np.full(dh * dw * image.shape[2], SENTINEL,
                         dtype=np.float32)
        request = SourceMode2Args(
            image.ctypes.data_as(F32_PTR),
            image.size if source_count is None else source_count,
            source_top, source_left, image.shape[0], image.shape[1],
            image.shape[2], dt, dl, dh, dw, matrix_array, bounds_array,
            output.ctypes.data_as(F32_PTR) if output.size else F32_PTR(),
            output.size,
        )
        if alias:
            function = self.alias_function
        elif physical_window:
            function = self.physical_window_function
        else:
            function = self.function
        status = int(function(C.byref(request)))
        return status, output.reshape(dh, dw, image.shape[2])


@pytest.fixture(scope="module")
def bridge(tmp_path_factory) -> Bridge:
    kind = os.environ.get("ACR_SOURCE_RESAMPLE_BRIDGE", "source")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_source_resample_" + kind) / (
        "bridge" + suffix)
    _compile_bridge(kind, output)
    return Bridge(output)


def _axis_for_request(axis: dict) -> dict:
    return {
        "kind": int(axis["kind"]),
        "integer_offset": int(axis.get("integer_offset", 0)),
        "origin": int(axis.get("origin", 0)),
        "step": int(axis.get("step", 0)),
        "lower": int(axis.get("lower", 0)),
        "upper": int(axis.get("upper", 0)),
        "tap_count": int(axis.get("taps", 0)),
    }


def _mode2_reference(source: np.ndarray, source_top: int, source_left: int,
                     destination: tuple[int, int, int, int], matrix,
                     plan_bounds: tuple[int, int, int, int] | None = None):
    dt, dl, dh, dw = map(int, destination)
    if plan_bounds is None:
        plan_bounds = (dt, dl, dt + dh, dl + dw)
    plan = expected_plan(matrix, plan_bounds)
    assert plan["status"] == 0
    status, request, intermediate = request_oracle.prepare_source_request(
        request_oracle.Rect(*destination),
        _axis_for_request(plan["vertical"]),
        _axis_for_request(plan["horizontal"]),
    )
    assert status == request_oracle.STATUS_OK
    assert request is not None and intermediate is not None

    image = np.ascontiguousarray(source, dtype=np.float32)
    height, width, _channels = image.shape
    source_bottom = source_top + height
    source_right = source_left + width
    assert max(source_top, request.top) < min(source_bottom, request.bottom)
    assert max(source_left, request.left) < min(source_right, request.right)

    # The analyzed mode-2 ordinary float path copies words from the nearest
    # valid source row/column to materialize each requested out-of-domain pixel.
    rows = np.clip(np.arange(request.top, request.bottom, dtype=np.int64)
                   - source_top, 0, height - 1)
    columns = np.clip(np.arange(request.left, request.right, dtype=np.int64)
                      - source_left, 0, width - 1)
    materialized = image[rows[:, None], columns[None, :], :]
    output = pixel_oracle.reference(
        materialized, request.top, request.left, destination,
        plan["vertical"], plan["horizontal"])
    return output, request, intermediate, plan


def _image(height: int, width: int, channels: int) -> np.ndarray:
    values = np.arange(height * width * channels, dtype=np.float32)
    values = (values - np.float32(17.0)) / np.float32(7.0)
    return values.reshape(height, width, channels)


def _assert_failed_atomic(status: int, actual: np.ndarray) -> None:
    assert status != 0
    assert status != 99, "bridge detected native result mutation on failure"
    np.testing.assert_array_equal(
        actual.view(np.uint32),
        np.full(actual.shape, SENTINEL, dtype=np.float32).view(np.uint32),
    )


@pytest.mark.parametrize(("name", "matrix", "source_rect", "destination"), [
    ("top-left-extension",
     (1.4, 0.0, 0.3, 0.0, 0.75, -0.25, 0.0, 0.0, 1.0),
     (-4, 5, 12, 13), (-4, 5, 8, 9)),
    ("bottom-right-extension",
     (0.75, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
     (0, 0, 10, 11), (6, 7, 4, 4)),
])
def test_mode2_edges_and_corners_match_independent_oracles(
        bridge, name, matrix, source_rect, destination):
    source_top, source_left, height, width = source_rect
    source = _image(height, width, 1)
    expected, request, _intermediate, _plan = _mode2_reference(
        source, source_top, source_left, destination, matrix)
    status, actual = bridge.call(source, source_top, source_left,
                                 destination, matrix)
    assert status == 0, name
    np.testing.assert_array_equal(actual.view(np.uint32),
                                  expected.view(np.uint32))
    if name == "top-left-extension":
        assert request.top < source_top
    else:
        assert request.bottom > source_top + height
        assert request.right > source_left + width


def test_mode2_matches_hard_probe_and_differs_from_old_physical_window_clamp(
        bridge):
    matrix = (1.4, 0.0, 0.3, 0.0, 0.75, -0.25, 0.0, 0.0, 1.0)
    source_top, source_left = -4, 5
    destination = (-4, 5, 8, 9)
    rows = (np.arange(12, dtype=np.float32)[:, None] * np.float32(100.0))
    columns = np.arange(13, dtype=np.float32)[None, :]
    source = (rows + columns).astype(np.float32)[:, :, None]
    expected, request, _intermediate, plan = _mode2_reference(
        source, source_top, source_left, destination, matrix)
    old_physical_window = pixel_oracle.reference(
        source, source_top, source_left, destination,
        plan["vertical"], plan["horizontal"])
    assert request.top == -5
    assert int(expected.view(np.uint32)[0, 0, 0]) == 0x429EA877
    assert int(old_physical_window.view(np.uint32)[0, 0, 0]) == 0x42CC5B42
    assert int(expected.view(np.uint32)[0, 0, 0]) != int(
        old_physical_window.view(np.uint32)[0, 0, 0])

    status, actual = bridge.call(source, source_top, source_left,
                                 destination, matrix)
    assert status == 0
    np.testing.assert_array_equal(actual.view(np.uint32),
                                  expected.view(np.uint32))
    assert int(actual.view(np.uint32)[0, 0, 0]) == 0x429EA877

    old_status, old_actual = bridge.call(
        source, source_top, source_left, destination, matrix,
        physical_window=True)
    assert old_status == 0
    np.testing.assert_array_equal(
        old_actual.view(np.uint32), old_physical_window.view(np.uint32))
    assert int(old_actual.view(np.uint32)[0, 0, 0]) == 0x42CC5B42
    assert not np.array_equal(actual.view(np.uint32),
                              old_actual.view(np.uint32))


def test_mode2_all_inside_fractional_scales_match_pixel_oracle(bridge):
    source_top, source_left = -20, 10
    source = _image(50, 55, 3)
    destination = (-2, 24, 8, 9)
    matrix = (2.0, 0.0, 0.25, 0.0, 1.5, -0.5, 0.0, 0.0, 1.0)
    expected, request, _intermediate, _plan = _mode2_reference(
        source, source_top, source_left, destination, matrix)
    assert source_top <= request.top and request.bottom <= source_top + 50
    assert source_left <= request.left and request.right <= source_left + 55
    status, actual = bridge.call(source, source_top, source_left,
                                 destination, matrix)
    assert status == 0
    np.testing.assert_array_equal(actual.view(np.uint32),
                                  expected.view(np.uint32))


def test_mode2_materializes_a_one_by_one_source_for_fractional_sampling(bridge):
    source = np.asarray([[[1.25, -2.5, 7.0]]], dtype=np.float32)
    destination = (3, -2, 1, 1)
    matrix = (1.4, 0.0, 0.1, 0.0, 1.3, -0.1, 0.0, 0.0, 1.0)
    expected, request, _intermediate, _plan = _mode2_reference(
        source, 3, -2, destination, matrix)
    assert (request.top, request.left, request.height, request.width) == (
        2, -3, 4, 4)
    status, actual = bridge.call(source, 3, -2, destination, matrix)
    assert status == 0
    np.testing.assert_array_equal(actual.view(np.uint32),
                                  expected.view(np.uint32))


@pytest.mark.parametrize(("matrix", "source_rect", "destination", "bounds"), [
    ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
     (10, -7, 4, 5), (9, -8, 3, 4), (9, -8, 12, -4)),
    ((1.0, 0.0, 3.0, 0.0, 1.0, -2.0, 0.0, 0.0, 1.0),
     (-6, 3, 12, 12), (-3, 6, 3, 4), (-3, 6, 0, 10)),
])
def test_mode2_identity_and_integer_translation_match_oracle(
        bridge, matrix, source_rect, destination, bounds):
    source_top, source_left, height, width = source_rect
    source = _image(height, width, 3)
    expected, _request, _intermediate, plan = _mode2_reference(
        source, source_top, source_left, destination, matrix, bounds)
    if matrix[2] == 0.0 and matrix[5] == 0.0:
        assert plan["vertical"]["kind"] == pixel_oracle.KIND_IDENTITY
        assert plan["horizontal"]["kind"] == pixel_oracle.KIND_IDENTITY
    else:
        assert plan["vertical"]["kind"] == pixel_oracle.KIND_INTEGER_TRANSLATION
        assert plan["horizontal"]["kind"] == pixel_oracle.KIND_INTEGER_TRANSLATION
    status, actual = bridge.call(source, source_top, source_left,
                                 destination, matrix, bounds)
    assert status == 0
    np.testing.assert_array_equal(actual.view(np.uint32),
                                  expected.view(np.uint32))


def test_mode2_ignores_nan_outside_materialized_source_request(bridge):
    source_top, source_left = -4, 5
    source = _image(12, 13, 1)
    source[-1, -1, 0] = np.float32(np.nan)
    matrix = (1.4, 0.0, 0.3, 0.0, 0.75, -0.25, 0.0, 0.0, 1.0)
    destination = (-4, 5, 8, 9)
    expected, request, _intermediate, _plan = _mode2_reference(
        source, source_top, source_left, destination, matrix)
    assert request.bottom <= source_top + source.shape[0]
    assert request.right <= source_left + source.shape[1] - 1
    status, actual = bridge.call(source, source_top, source_left,
                                 destination, matrix)
    assert status == 0
    np.testing.assert_array_equal(actual.view(np.uint32),
                                  expected.view(np.uint32))


def test_mode2_rejects_nan_copied_into_materialized_request_atomically(bridge):
    source = np.asarray([[[np.float32(np.nan)]]], dtype=np.float32)
    status, actual = bridge.call(
        source, 3, -2, (3, -2, 1, 1),
        (1.4, 0.0, 0.1, 0.0, 1.3, -0.1, 0.0, 0.0, 1.0),
    )
    _assert_failed_atomic(status, actual)


@pytest.mark.parametrize("bad_count", ["short", "size_max"])
def test_mode2_invalid_declared_source_span_fails_atomically(bridge, bad_count):
    source = _image(3, 4, 1)
    destination = (1, 1, 1, 1)
    matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    declared = source.size - 1 if bad_count == "short" else (1 << (8 * C.sizeof(C.c_size_t))) - 1
    status, actual = bridge.call(source, 0, 0, destination, matrix,
                                 source_count=declared)
    _assert_failed_atomic(status, actual)


def test_mode2_rejects_disjoint_request_and_rectangle_overflow_atomically(bridge):
    matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    source = _image(3, 4, 1)
    status, disjoint = bridge.call(source, 100, -100, (0, 0, 2, 2), matrix)
    assert status == 3  # OrdinarySourceMode2Status::unsupported_disjoint_request
    _assert_failed_atomic(status, disjoint)

    status, overflow = bridge.call(
        source, 0, 0, ((1 << 31) - 1, 0, 1, 1), matrix,
        plan_bounds=(0, 0, 2, 2),
    )
    _assert_failed_atomic(status, overflow)


def test_mode2_aliases_old_result_storage_without_losing_source_pixels(bridge):
    source = _image(4, 5, 1)
    destination = (1, 2, 2, 2)
    matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    expected, _request, _intermediate, _plan = _mode2_reference(
        source, 0, 0, destination, matrix)
    status, actual = bridge.call(source, 0, 0, destination, matrix,
                                 alias=True)
    assert status == 0
    np.testing.assert_array_equal(actual.view(np.uint32),
                                  expected.view(np.uint32))


def test_mode2_alias_failure_preserves_old_result_storage(bridge):
    source = _image(1, 1, 1)
    source[0, 0, 0] = np.float32(np.nan)
    destination = (3, -2, 1, 1)
    matrix = (1.4, 0.0, 0.1, 0.0, 1.3, -0.1, 0.0, 0.0, 1.0)
    status, actual = bridge.call(source, 3, -2, destination, matrix,
                                 alias=True)
    _assert_failed_atomic(status, actual)
