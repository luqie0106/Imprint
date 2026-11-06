"""Checks for the independently gated, prepared level-zero bounds helper."""

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
BRIDGE_SOURCE = NATIVE / "tools/acr_anchor_level0_bridge.cpp"
GEOMETRY_SOURCE = NATIVE / "src/acr_anchor_geometry.cpp"
HEADER = NATIVE / "src/acr_anchor_geometry.hpp"
FORMAL_OBJECT = (NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/"
                 "acr_anchor_geometry.cpp.o")

sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import anchor_level0_oracle as oracle  # noqa: E402


class _FilterInput(C.Structure):
    _fields_ = [("non_null", C.c_int32),
                ("rects", C.c_int32 * 16)]


_I32x4 = C.c_int32 * 4
_I32x8 = C.c_int32 * 8
OUTPUT_SENTINEL = bytes([0xA5]) * C.sizeof(_I32x8)


def _compile_bridge(kind: str, output: Path) -> list[str]:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the anchor bridge"
    assert BRIDGE_SOURCE.is_file(), f"missing test bridge: {BRIDGE_SOURCE}"
    assert HEADER.is_file(), f"missing production header: {HEADER}"
    command = [
        compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", "-ffp-contract=off", "-fPIC",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", str(NATIVE / "src"), "-I", str(NATIVE / "include"),
        str(BRIDGE_SOURCE),
    ]
    if kind == "source":
        assert GEOMETRY_SOURCE.is_file(), f"missing source: {GEOMETRY_SOURCE}"
        inputs = [GEOMETRY_SOURCE]
    elif kind == "formal_object":
        assert FORMAL_OBJECT.is_file(), (
            "formal renderer object is unavailable: " + str(FORMAL_OBJECT))
        source_mtime = max(GEOMETRY_SOURCE.stat().st_mtime_ns,
                           HEADER.stat().st_mtime_ns)
        object_mtime = FORMAL_OBJECT.stat().st_mtime_ns
        assert object_mtime >= source_mtime, (
            "stale formal object: " + str(FORMAL_OBJECT) +
            f" (object mtime_ns={object_mtime}, source/header mtime_ns="
            f"{source_mtime}); rebuild the formal renderer before running "
            "ACR_ANCHOR_LEVEL0_BRIDGE=formal_object")
        inputs = [FORMAL_OBJECT]
    else:
        raise AssertionError(
            "ACR_ANCHOR_LEVEL0_BRIDGE must be source or formal_object")
    command.extend(str(path) for path in inputs)
    command.extend(["-o", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    return command


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.function = self.library.acr_test_compute_anchor_level0_bounds
        self.function.argtypes = [
            C.c_int32, C.c_int32,
            C.POINTER(C.c_int32), C.c_size_t,
            C.POINTER(C.c_int32), C.c_size_t,
            C.POINTER(_FilterInput), C.c_size_t, C.c_size_t,
            C.POINTER(C.c_int32), C.c_size_t,
        ]
        self.function.restype = C.c_int

    @staticmethod
    def _filters(filters):
        if not filters:
            return None
        encoded = (_FilterInput * len(filters))()
        for index, item in enumerate(filters):
            non_null = item.non_null
            encoded[index].non_null = int(non_null)
            encoded[index].rects[:] = (
                *item.offset_58, *item.offset_78,
                *item.offset_68, *item.offset_88,
            )
        return encoded

    def call_raw(self, transform_actual: int, transform_reference: int,
                 actual, reference, filters=None, *,
                 actual_span_bytes: int | None = None,
                 reference_span_bytes: int | None = None,
                 filter_count: int | None = None,
                 filter_span_bytes: int | None = None,
                 output_span_bytes: int | None = None):
        actual_buffer = _I32x4(*actual)
        reference_buffer = _I32x4(*reference)
        encoded_filters = self._filters(filters)
        actual_bytes = (C.sizeof(actual_buffer) if actual_span_bytes is None
                        else actual_span_bytes)
        reference_bytes = (C.sizeof(reference_buffer)
                           if reference_span_bytes is None
                           else reference_span_bytes)
        count = (len(filters) if filter_count is None and filters is not None
                 else 0 if filter_count is None else filter_count)
        filter_bytes = (
            C.sizeof(_FilterInput) * len(filters)
            if filter_span_bytes is None and filters is not None
            else 0 if filter_span_bytes is None else filter_span_bytes)
        output_buffer = (_I32x8).from_buffer_copy(OUTPUT_SENTINEL)
        out_bytes = (C.sizeof(output_buffer)
                     if output_span_bytes is None else output_span_bytes)
        status = int(self.function(
            transform_actual, transform_reference,
            actual_buffer, actual_bytes, reference_buffer, reference_bytes,
            encoded_filters, count, filter_bytes,
            output_buffer, out_bytes,
        ))
        return status, tuple(output_buffer), bytes(output_buffer)

    def call(self, case: oracle.Input):
        return self.call_raw(
            int(case.transform_actual), int(case.transform_reference),
            case.actual_level0_bounds, case.reference_level0_bounds,
            case.filters,
        )


@pytest.fixture(scope="module")
def bridge(tmp_path_factory) -> Bridge:
    kind = os.environ.get("ACR_ANCHOR_LEVEL0_BRIDGE", "source")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_anchor_level0_" + kind) / (
        "bridge" + suffix)
    _compile_bridge(kind, output)
    return Bridge(output)


def _filter(non_null=True, *, offset_58=(10, 20, 90, 180),
            offset_78=(0, 0, 100, 200),
            offset_68=(25, 50, 75, 150),
            offset_88=(0, 0, 100, 200)):
    return oracle.Filter(non_null, offset_58, offset_78,
                         offset_68, offset_88)


def _assert_matches(bridge: Bridge, case: oracle.Input):
    expected_status, expected = oracle.compute(case)
    status, values, raw = bridge.call(case)
    assert status == int(expected_status)
    if expected_status == oracle.Status.OK:
        assert expected is not None
        assert values == (*expected.actual_level0_bounds,
                          *expected.reference_level0_bounds)
    else:
        assert raw == OUTPUT_SENTINEL
    return values


def test_all_gate_combinations_and_hardcoded_mixed_probe(bridge: Bridge):
    actual = (10, 20, 110, 220)
    reference = (7, 13, 207, 413)
    filters = (_filter(),)
    expected_by_gates = {
        (False, False): (actual, reference),
        (True, False): ((20, 40, 100, 200), reference),
        (False, True): (actual, (57, 113, 157, 313)),
        (True, True): ((20, 40, 100, 200), (57, 113, 157, 313)),
    }
    for gates, expected in expected_by_gates.items():
        case = oracle.Input(actual, reference, filters, *gates)
        values = _assert_matches(bridge, case)
        assert values == (*expected[0], *expected[1])

    # This is the constructor asymmetry: actual is filtered while reference is
    # the original int32 rectangle. Both independently differ from the other
    # mixed combination above.
    assert expected_by_gates[(True, False)] == (
        (20, 40, 100, 200), (7, 13, 207, 413))


def test_null_equal_and_multiple_filters_with_nonzero_origins(bridge: Bridge):
    null_filter = _filter(
        False,
        offset_58=(0, 0, 0, 0), offset_78=(0, 0, 0, 0),
        offset_68=(0, 0, 0, 0), offset_88=(0, 0, 0, 0),
    )
    equal_actual = _filter(
        True,
        offset_58=(5, 6, 5, 6), offset_78=(5, 6, 5, 6),
        offset_68=(0, 0, 60, 80), offset_88=(0, 0, 100, 100),
    )
    second = _filter(
        True,
        offset_58=(10, 20, 90, 180), offset_78=(0, 0, 100, 200),
        offset_68=(10, 20, 80, 90), offset_88=(0, 0, 100, 100),
    )
    case = oracle.Input(
        (123, 456, 1123, 2456), (-80, 135, 720, 1735),
        (null_filter, equal_actual, second), True, True,
    )
    _assert_matches(bridge, case)


@pytest.mark.parametrize(
    ("base", "expected"),
    [((0, 0, 1, 1), (1, 1, 1, 1)),
     ((-1, -1, 0, 0), (-1, -1, 0, 0))],
)
def test_positive_and_negative_half_ties_round_away_from_zero(
    bridge: Bridge, base, expected):
    tie_filter = _filter(
        offset_58=(1, 1, 2, 2), offset_78=(0, 0, 2, 2),
        offset_68=(1, 1, 2, 2), offset_88=(0, 0, 2, 2),
    )
    case = oracle.Input(base, (0, 0, 1, 1), (tie_filter,), True, False)
    values = _assert_matches(bridge, case)
    assert values[:4] == expected


def test_signed_differences_allow_negative_and_int32_minimum(bridge: Bridge):
    for rect, expected_height, expected_width in (
        ((8, 9, 2, 1), -6, -8),
        ((0, 0, -(1 << 31), 1), -(1 << 31), 1),
    ):
        case = oracle.Input(rect, (0, 0, 1, 1), (), True, False)
        values = _assert_matches(bridge, case)
        assert values[:4] == rect
        assert values[2] - values[0] == expected_height
        assert values[3] - values[1] == expected_width


@pytest.mark.parametrize(
    "rect",
    [
        (0, -(1 << 31), 1, 0),  # width = INT32_MAX + 1
        (1, 0, -(1 << 31), 1),  # height = INT32_MIN - 1
        (0, (1 << 31) - 1, 1, -2),  # width = INT32_MIN - 1
    ],
)
def test_only_signed_difference_overflow_is_rejected_atomically(
    bridge: Bridge, rect):
    case = oracle.Input(rect, (3, 4, 5, 6), (), True, False)
    expected_status, _ = oracle.compute(case)
    assert expected_status == oracle.Status.INT32_OVERFLOW
    status, _, raw = bridge.call(case)
    assert status == int(oracle.Status.INT32_OVERFLOW)
    assert raw == OUTPUT_SENTINEL


def test_coordinate_rounding_overflow_is_rejected_atomically(bridge: Bridge):
    beyond_i32 = _filter(
        offset_58=(2, 0, 3, 1), offset_78=(0, 0, 1, 1),
        offset_68=(0, 0, 1, 1), offset_88=(0, 0, 1, 1),
    )
    case = oracle.Input(
        ((1 << 31) - 2, 0, (1 << 31) - 1, 1),
        (0, 0, 1, 1), (beyond_i32,), True, False,
    )
    expected_status, _ = oracle.compute(case)
    assert expected_status == oracle.Status.INT32_OVERFLOW
    status, _, raw = bridge.call(case)
    assert status == int(oracle.Status.INT32_OVERFLOW)
    assert raw == OUTPUT_SENTINEL


def test_failure_in_second_chain_keeps_both_outputs_unchanged(bridge: Bridge):
    filter_with_bad_reference_denominator = _filter(
        offset_58=(0, 0, 1, 1), offset_78=(0, 0, 2, 2),
        offset_68=(0, 0, 1, 1), offset_88=(0, 0, 0, 1),
    )
    case = oracle.Input(
        (10, 20, 110, 220), (30, 40, 130, 240),
        (filter_with_bad_reference_denominator,), True, True,
    )
    expected_status, _ = oracle.compute(case)
    assert expected_status == oracle.Status.INVALID_GEOMETRY
    status, _, raw = bridge.call(case)
    assert status == int(oracle.Status.INVALID_GEOMETRY)
    assert raw == OUTPUT_SENTINEL


def test_disabled_chains_do_not_validate_unused_filters(bridge: Bridge):
    malformed_bool_filter = _filter(
        7,
        offset_58=(0, 0, 0, 0), offset_78=(0, 0, 0, 0),
        offset_68=(0, 0, 0, 0), offset_88=(0, 0, 0, 0),
    )
    status, values, _ = bridge.call_raw(
        0, 0, (3, 4, 5, 6), (7, 8, 9, 10),
        (malformed_bool_filter,), filter_count=0x7FFFFFFF,
        filter_span_bytes=0,
    )
    assert status == int(oracle.Status.OK)
    assert values == (3, 4, 5, 6, 7, 8, 9, 10)

    status, _, raw = bridge.call_raw(
        1, 0, (3, 4, 5, 6), (7, 8, 9, 10),
        (malformed_bool_filter,),
    )
    assert status == int(oracle.Status.INVALID_INPUT)
    assert raw == OUTPUT_SENTINEL


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transform_actual": 2},
        {"actual_span_bytes": 15},
        {"reference_span_bytes": 15},
        {"filter_span_bytes": C.sizeof(_FilterInput) - 1},
        {"output_span_bytes": C.sizeof(_I32x8) - 1},
    ],
)
def test_bridge_rejects_bad_flags_and_spans_atomically(bridge: Bridge, kwargs):
    case_kwargs = {
        "transform_actual": 1, "transform_reference": 0,
        "actual": (1, 2, 3, 4), "reference": (5, 6, 7, 8),
        "filters": (_filter(),),
    }
    case_kwargs.update(kwargs)
    status, _, raw = bridge.call_raw(**case_kwargs)
    assert status == int(oracle.Status.INVALID_INPUT)
    assert raw == OUTPUT_SENTINEL


def test_seeded_random_prepared_cases_match_independent_oracle(bridge: Bridge):
    randomizer = random.Random(0xA11C0)

    def source_rect():
        top = randomizer.randrange(0, 80)
        left = randomizer.randrange(0, 80)
        bottom = randomizer.randrange(top + 1, 101)
        right = randomizer.randrange(left + 1, 101)
        return top, left, bottom, right

    for _ in range(240):
        actual_top = randomizer.randrange(-2000, 2001)
        actual_left = randomizer.randrange(-2000, 2001)
        actual_height = randomizer.randrange(1, 5000)
        actual_width = randomizer.randrange(1, 5000)
        reference_top = randomizer.randrange(-2000, 2001)
        reference_left = randomizer.randrange(-2000, 2001)
        reference_height = randomizer.randrange(1, 5000)
        reference_width = randomizer.randrange(1, 5000)
        filters = []
        for _ in range(randomizer.randrange(0, 5)):
            filters.append(_filter(
                randomizer.choice((True, True, False)),
                offset_58=source_rect(), offset_78=(0, 0, 100, 100),
                offset_68=source_rect(), offset_88=(0, 0, 100, 100),
            ))
        case = oracle.Input(
            (actual_top, actual_left,
             actual_top + actual_height, actual_left + actual_width),
            (reference_top, reference_left,
             reference_top + reference_height,
             reference_left + reference_width),
            tuple(filters), bool(randomizer.getrandbits(1)),
            bool(randomizer.getrandbits(1)),
        )
        _assert_matches(bridge, case)
