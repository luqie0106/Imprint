"""Independent checks for the ordinary sampler logical source rectangle."""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools/acr_resample_request_bridge.cpp"
REQUEST_SOURCE = NATIVE / "src/acr_resample_request.cpp"
REQUEST_HEADER = NATIVE / "src/acr_resample_request.hpp"
REQUEST_OBJECT = (NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/"
                  "acr_resample_request.cpp.o")

sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import resample_request_oracle as oracle  # noqa: E402


class RectArgs(C.Structure):
    _fields_ = [
        ("top", C.c_int32),
        ("left", C.c_int32),
        ("height", C.c_size_t),
        ("width", C.c_size_t),
    ]


F32_PTR = C.POINTER(C.c_float)


class AxisArgs(C.Structure):
    _fields_ = [
        ("kind", C.c_int),
        ("integer_offset", C.c_int32),
        ("origin", C.c_int64),
        ("step", C.c_int64),
        ("lower", C.c_int64),
        ("upper", C.c_int64),
        ("tap_count", C.c_size_t),
        ("coefficients", F32_PTR),
        ("coefficient_count", C.c_size_t),
        ("stride", C.c_size_t),
    ]


class RequestArgs(C.Structure):
    _fields_ = [
        ("destination", RectArgs),
        ("vertical", AxisArgs),
        ("horizontal", AxisArgs),
    ]


class ResultArgs(C.Structure):
    _fields_ = [
        ("source", RectArgs),
        ("intermediate", RectArgs),
    ]


class Bridge:
    def __init__(self, path: Path, kind: str):
        self.path = path
        self.kind = kind
        self.library = C.CDLL(str(path))
        self.function = self.library.acr_test_prepare_ordinary_source_request
        self.function.argtypes = [C.POINTER(RequestArgs), C.POINTER(ResultArgs)]
        self.function.restype = C.c_int

    def call(self, destination: oracle.Rect, vertical: dict,
             horizontal: dict) -> tuple[int, ResultArgs, bytes]:
        coefficient_arrays: list[object] = []

        def make_axis(axis: dict) -> AxisArgs:
            values = axis.get("coefficients", ())
            array = None
            if values:
                array = (C.c_float * len(values))(*values)
                coefficient_arrays.append(array)
            pointer = F32_PTR() if array is None else C.cast(array, F32_PTR)
            return AxisArgs(
                int(axis.get("kind", oracle.KIND_RESAMPLE)),
                int(axis.get("integer_offset", 0)),
                int(axis.get("origin", 0)),
                int(axis.get("step", 1 << 32)),
                int(axis.get("lower", oracle.INT64_MIN)),
                int(axis.get("upper", oracle.INT64_MAX)),
                int(axis.get("tap_count", 4)),
                pointer,
                int(axis.get("coefficient_count", 0 if array is None else len(values))),
                int(axis.get("stride", 0 if array is None else len(values))),
            )

        request = RequestArgs(
            RectArgs(destination.top, destination.left,
                     destination.height, destination.width),
            make_axis(vertical), make_axis(horizontal),
        )
        output = ResultArgs()
        C.memset(C.byref(output), 0xA5, C.sizeof(output))
        output.source = RectArgs(71, -72, 73, 74)
        output.intermediate = RectArgs(-75, 76, 77, 78)
        before = C.string_at(C.byref(output), C.sizeof(output))
        status = self.function(C.byref(request), C.byref(output))
        # Keep coefficient arrays alive through the native call.
        assert coefficient_arrays is not None
        return status, output, before


def _compile_bridge(kind: str, output: Path) -> list[str]:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the source-request bridge"
    assert BRIDGE_SOURCE.is_file(), f"missing test bridge: {BRIDGE_SOURCE}"
    command = [
        compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-I", str(NATIVE / "src"), "-I", str(NATIVE / "include"),
        str(BRIDGE_SOURCE),
    ]
    if kind == "source":
        assert REQUEST_SOURCE.is_file(), f"missing implementation source: {REQUEST_SOURCE}"
        inputs = [REQUEST_SOURCE]
    elif kind == "formal_object":
        dependencies = [
            REQUEST_SOURCE,
            REQUEST_HEADER,
            NATIVE / "src/acr_resample_plan.hpp",
            NATIVE / "src/acr_resample_axis.hpp",
            NATIVE / "src/acr_resample_separable.hpp",
            NATIVE / "src/acr_resample_coefficients.hpp",
        ]
        assert REQUEST_OBJECT.is_file(), f"missing formal renderer object: {REQUEST_OBJECT}"
        newest_input = max(path.stat().st_mtime_ns for path in dependencies)
        assert REQUEST_OBJECT.stat().st_mtime_ns >= newest_input, (
            f"rebuild formal renderer: stale object {REQUEST_OBJECT.name}"
        )
        inputs = [REQUEST_OBJECT]
    else:
        raise AssertionError("ACR_RESAMPLE_REQUEST_BRIDGE must be source or formal_object")
    command.extend(str(path) for path in inputs)
    command.extend([
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-o", str(output),
    ])
    subprocess.run(command, check=True, capture_output=True, text=True)
    return command


@pytest.fixture(scope="module")
def bridge(tmp_path_factory) -> Bridge:
    kind = os.environ.get("ACR_RESAMPLE_REQUEST_BRIDGE", "source")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_source_request_" + kind) / ("bridge" + suffix)
    _compile_bridge(kind, output)
    return Bridge(output, kind)


def axis(kind: int = oracle.KIND_RESAMPLE, **values) -> dict:
    return {
        "kind": kind,
        "integer_offset": 0,
        "origin": 0,
        "step": 1 << 32,
        "lower": oracle.INT64_MIN,
        "upper": oracle.INT64_MAX,
        "tap_count": 4,
        **values,
    }


def _actual_rect(args: RectArgs) -> oracle.Rect:
    return oracle.Rect(args.top, args.left, args.height, args.width)


def assert_matches_oracle(bridge: Bridge, destination: oracle.Rect,
                          vertical: dict, horizontal: dict) -> int:
    expected_status, expected_source, expected_intermediate = (
        oracle.prepare_source_request(destination, vertical, horizontal)
    )
    status, output, before = bridge.call(destination, vertical, horizontal)
    assert status == expected_status
    if status == oracle.STATUS_OK:
        assert _actual_rect(output.source) == expected_source
        assert _actual_rect(output.intermediate) == expected_intermediate
    else:
        assert C.string_at(C.byref(output), C.sizeof(output)) == before
    return status


def test_four_corner_mixed_axes_fractional_negative_q32_and_poison_coefficients(bridge):
    destination = oracle.Rect(-3, -5, 8, 11)
    vertical = axis(
        origin=0,
        step=1 << 31,
        tap_count=4,
        coefficients=[float("nan"), float("inf"), -float("inf"), 19.0],
        stride=0,
    )
    horizontal = axis(
        origin=1 << 30,
        step=3 << 30,
        tap_count=6,
        coefficients=[-float("nan"), 0.0, float("inf")],
        stride=1,
    )
    assert_matches_oracle(bridge, destination, vertical, horizontal)


def test_identity_and_integer_translation_do_not_consume_coefficient_tables(bridge):
    destination = oracle.Rect(-6, 8, 5, 7)
    vertical = axis(
        oracle.KIND_IDENTITY,
        origin=oracle.INT64_MAX,
        step=0,
        lower=1,
        upper=-1,
        tap_count=(1 << 63) - 1,
        coefficients=[float("nan"), float("inf")],
        stride=0,
    )
    horizontal = axis(
        oracle.KIND_INTEGER_TRANSLATION,
        integer_offset=-3,
        origin=oracle.INT64_MIN,
        step=-1,
        lower=9,
        upper=-9,
        tap_count=0,
        coefficients=[-float("inf"), 123.0],
        stride=(1 << 63) - 1,
    )
    assert_matches_oracle(bridge, destination, vertical, horizontal)


def test_identity_axes_preserve_both_half_open_lanes(bridge):
    destination = oracle.Rect(-13, 17, 9, 12)
    identity = axis(oracle.KIND_IDENTITY)
    assert_matches_oracle(bridge, destination, identity, identity)


def test_signed64_origin_step_wrap_and_w32_upper_endpoint_wrap(bridge):
    destination = oracle.Rect(0, 0, 2, 2)
    vertical = axis(oracle.KIND_IDENTITY)
    horizontal = axis(
        origin=oracle.INT64_MAX,
        step=1 << 32,
        tap_count=4,
    )
    assert_matches_oracle(bridge, destination, vertical, horizontal)


def test_integer_translation_at_int32_edges_is_exact(bridge):
    destination = oracle.Rect(oracle.INT32_MIN, oracle.INT32_MAX - 1, 1, 1)
    identity = axis(oracle.KIND_IDENTITY)
    translated = axis(oracle.KIND_INTEGER_TRANSLATION, integer_offset=0)
    assert_matches_oracle(bridge, destination, translated, identity)


@pytest.mark.parametrize(
    "destination",
    [
        oracle.Rect(0, 0, 0, 1),
        oracle.Rect(0, 0, 1, 0),
        oracle.Rect(0, 0, oracle.INT32_MAX + 1, 1),
        oracle.Rect(oracle.INT32_MAX, 0, 1, 1),
        oracle.Rect(0, oracle.INT32_MAX, 1, 1),
    ],
)
def test_bad_destination_rectangles_fail_atomically(bridge, destination):
    valid = axis(oracle.KIND_IDENTITY)
    assert_matches_oracle(bridge, destination, valid, valid)


@pytest.mark.parametrize(
    "bad_axis",
    [
        axis(kind=99),
        axis(step=0),
        axis(step=-1),
        axis(lower=2, upper=1),
        axis(tap_count=0),
        axis(tap_count=3),
        axis(tap_count=1 << 31),
    ],
)
def test_bad_resample_axis_kind_step_bounds_and_taps_fail_atomically(bridge, bad_axis):
    destination = oracle.Rect(-2, 3, 4, 5)
    assert_matches_oracle(
        bridge, destination, axis(oracle.KIND_IDENTITY), bad_axis
    )


@pytest.mark.parametrize(
    ("destination", "horizontal", "expected_status"),
    [
        (
            oracle.Rect(0, oracle.INT32_MIN, 1, 1),
            axis(oracle.KIND_INTEGER_TRANSLATION, integer_offset=1),
            oracle.STATUS_COORDINATE_OVERFLOW,
        ),
        (
            oracle.Rect(0, 0, 1, 2),
            axis(origin=oracle.INT64_MIN, step=oracle.INT64_MAX, tap_count=2),
            oracle.STATUS_INVALID_RECTANGLE,
        ),
    ],
)
def test_horizontal_failure_precedes_vertical_axis_validation(
    bridge, destination, horizontal, expected_status
):
    vertical = axis(kind=99)
    assert assert_matches_oracle(
        bridge, destination, vertical, horizontal
    ) == expected_status


def test_integer_translation_coordinate_overflow_fails_atomically(bridge):
    destination = oracle.Rect(oracle.INT32_MIN, 4, 1, 2)
    vertical = axis(oracle.KIND_INTEGER_TRANSLATION, integer_offset=1)
    horizontal = axis(oracle.KIND_IDENTITY)
    assert_matches_oracle(bridge, destination, vertical, horizontal)


def test_resample_span_overflow_fails_atomically(bridge):
    destination = oracle.Rect(0, 0, 1, 2)
    vertical = axis(oracle.KIND_IDENTITY)
    horizontal = axis(origin=oracle.INT64_MIN, step=oracle.INT64_MAX, tap_count=2)
    assert_matches_oracle(bridge, destination, vertical, horizontal)


def test_randomized_four_corner_requests(bridge):
    rng = random.Random(20261024)
    q32 = 1 << 32
    for _ in range(160):
        destination = oracle.Rect(
            rng.randint(-200, 200), rng.randint(-200, 200),
            rng.randint(1, 20), rng.randint(1, 20),
        )

        def random_axis() -> dict:
            kind = rng.randrange(3)
            if kind == oracle.KIND_RESAMPLE:
                lower = rng.randint(-40, 0) * q32 - rng.randrange(q32)
                upper = rng.randint(1, 40) * q32 + rng.randrange(q32)
                return axis(
                    kind,
                    origin=rng.randint(-24, 24) * q32 + rng.randint(-(q32 - 1), q32 - 1),
                    step=rng.randint(1, 5) * q32 + rng.randint(0, q32 - 1),
                    lower=lower,
                    upper=upper,
                    tap_count=2 * rng.randint(1, 8),
                    coefficients=[float("nan"), float("inf")],
                    stride=0,
                )
            if kind == oracle.KIND_INTEGER_TRANSLATION:
                return axis(kind, integer_offset=rng.randint(-30, 30))
            return axis(kind)

        assert_matches_oracle(bridge, destination, random_axis(), random_axis())

