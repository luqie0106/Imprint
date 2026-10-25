"""Independent contract checks for the dynamic d0684c geometry driver."""

from __future__ import annotations

import ctypes as C
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
from dataclasses import replace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools/acr_render_geometry_bridge.cpp"
GEOMETRY_SOURCE = NATIVE / "src/acr_render_geometry.cpp"
ANCHOR_SOURCE = NATIVE / "src/acr_anchor_geometry.cpp"
MATRIX_SOURCE = NATIVE / "src/acr_resample_matrix.cpp"
OBJECT_DIR = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src"
GEOMETRY_OBJECT = OBJECT_DIR / "acr_render_geometry.cpp.o"
ANCHOR_OBJECT = OBJECT_DIR / "acr_anchor_geometry.cpp.o"
MATRIX_OBJECT = OBJECT_DIR / "acr_resample_matrix.cpp.o"
HEADER = NATIVE / "src/acr_render_geometry.hpp"

sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import render_geometry_oracle as oracle  # noqa: E402


class _Rect(C.Structure):
    _fields_ = [("top", C.c_int32), ("left", C.c_int32),
                ("bottom", C.c_int32), ("right", C.c_int32)]


class _Level(C.Structure):
    _fields_ = [("image", C.c_void_p), ("reference_bounds", _Rect),
                ("actual_bounds", _Rect)]


_Levels = _Level * 6
_QWords7 = C.c_uint64 * 7
_Matrix3 = C.c_double * 9
_I32Pair = C.c_int32 * 2


class _Source(C.Structure):
    _fields_ = [("levels", _Levels), ("field_68a", C.c_uint8),
                ("field_178", C.c_uint32)]


class _Context(C.Structure):
    _fields_ = [
        ("field_00", C.c_uint32), ("field_7c", C.c_uint32),
        ("field_8c", C.c_uint32), ("field_98", C.c_uint32),
        ("field_911", C.c_uint32), ("field_846", C.c_uint8),
        ("field_847", C.c_uint8), ("field_820", C.c_uint8),
        ("field_84c", C.c_int32), ("field_8a8", _QWords7),
        ("field_810", C.c_double), ("field_818", C.c_int32),
        ("field_824", C.c_uint32), ("field_828", C.c_uint32),
    ]


class _Input(C.Structure):
    _fields_ = [
        ("source", _Source), ("context", _Context),
        ("initial_level", C.c_int32), ("extent3", _I32Pair),
        ("extent4", _I32Pair), ("optional_present", C.c_bool),
        ("optional_matrix", _Matrix3), ("param6", C.c_int32),
        ("feature_5a", C.c_uint8),
    ]


class _Result(C.Structure):
    _fields_ = [
        ("level", C.c_uint32), ("image", C.c_void_p),
        ("reference_bounds", _Rect), ("actual_bounds", _Rect),
        ("output_bounds", _Rect), ("matrix_e0", _Matrix3),
        ("matrix_170", _Matrix3), ("matrix_200", _Matrix3),
        ("context_after", _Context), ("flag_0c", C.c_bool),
        ("mask_290", C.c_bool), ("iterations", C.c_uint32),
        ("mask_first_value", C.c_double),
    ]


OUTPUT_SENTINEL = bytes([0x6D]) * C.sizeof(_Result)


def _compile_bridge(kind: str, output: Path) -> list[str]:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the geometry bridge"
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
        inputs = [GEOMETRY_SOURCE, ANCHOR_SOURCE, MATRIX_SOURCE]
        assert all(path.is_file() for path in inputs), "missing geometry sources"
    elif kind == "formal_object":
        inputs = [GEOMETRY_OBJECT, ANCHOR_OBJECT, MATRIX_OBJECT]
        assert all(path.is_file() for path in inputs), (
            "build the formal renderer before ACR_RENDER_GEOMETRY_BRIDGE="
            "formal_object")
        assert GEOMETRY_OBJECT.stat().st_mtime_ns >= max(
            GEOMETRY_SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns
        ), f"rebuild formal renderer: stale object {GEOMETRY_OBJECT.name}"
    else:
        raise AssertionError(
            "ACR_RENDER_GEOMETRY_BRIDGE must be source or formal_object")
    command.extend(str(path) for path in inputs)
    command.extend(["-o", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    return command


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.function = self.library.acr_test_prepare_render_geometry
        self.function.argtypes = [C.POINTER(_Input), C.POINTER(C.c_uint8),
                                  C.c_size_t]
        self.function.restype = C.c_int

    def call(self, case: oracle.RenderGeometryInput
             ) -> tuple[int, bytes, tuple[int | None, ...]]:
        native_input, image_refs = _native_input(case)
        output = (C.c_uint8 * C.sizeof(_Result)).from_buffer_copy(
            OUTPUT_SENTINEL)
        status = int(self.function(C.byref(native_input), output,
                                   C.sizeof(_Result)))
        assert status != 99, "native input or failed result bytes were mutated"
        addresses = tuple(C.addressof(buffer) if buffer is not None else None
                          for buffer in image_refs)
        return status, bytes(output), addresses


@pytest.fixture(scope="module")
def bridge(tmp_path_factory) -> Bridge:
    kind = os.environ.get("ACR_RENDER_GEOMETRY_BRIDGE", "source")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_render_geometry_" + kind) / (
        "bridge" + suffix)
    _compile_bridge(kind, output)
    return Bridge(output)


def _native_input(case: oracle.RenderGeometryInput
                  ) -> tuple[_Input, list[C.Array | None]]:
    native = _Input()
    buffers: list[C.Array | None] = []
    for index, level in enumerate(case.levels):
        if level.image:
            buffer = C.create_string_buffer(1)
            buffers.append(buffer)
            image = C.cast(buffer, C.c_void_p)
        else:
            buffers.append(None)
            image = C.c_void_p()
        native.source.levels[index] = _Level(
            image, _native_rect(level.reference_bounds),
            _native_rect(level.actual_bounds))
    native.source.field_68a = case.source_68a
    native.source.field_178 = case.source_178

    ctx = case.context
    for field in ("field_00", "field_7c", "field_8c", "field_98",
                  "field_911", "field_846", "field_847", "field_820",
                  "field_84c", "field_810", "field_818", "field_824",
                  "field_828"):
        setattr(native.context, field, getattr(ctx, field))
    native.context.field_8a8[:] = ctx.field_8a8
    native.initial_level = case.initial_level
    native.extent3[:] = case.extent3
    native.extent4[:] = case.extent4
    native.optional_present = case.optional_present
    native.optional_matrix[:] = case.optional_matrix
    native.param6 = case.param6
    native.feature_5a = case.feature_5a
    return native, buffers


def _native_rect(rect: oracle.Rect) -> _Rect:
    return _Rect(rect.top, rect.left, rect.bottom, rect.right)


def _rect_values(rect: _Rect) -> tuple[int, int, int, int]:
    return rect.top, rect.left, rect.bottom, rect.right


def _bits(values) -> tuple[int, ...]:
    return struct.unpack("=9Q", struct.pack("=9d", *values))


def _assert_output_matches(raw: bytes, expected: oracle.RenderGeometryOutput,
                           image_addresses: tuple[int | None, ...]) -> None:
    actual = _Result.from_buffer_copy(raw)
    assert actual.level == expected.level
    assert actual.image == image_addresses[expected.level]
    assert _rect_values(actual.reference_bounds) == (
        expected.reference_bounds.top, expected.reference_bounds.left,
        expected.reference_bounds.bottom, expected.reference_bounds.right)
    assert _rect_values(actual.actual_bounds) == (
        expected.actual_bounds.top, expected.actual_bounds.left,
        expected.actual_bounds.bottom, expected.actual_bounds.right)
    assert _rect_values(actual.output_bounds) == (
        expected.output_bounds.top, expected.output_bounds.left,
        expected.output_bounds.bottom, expected.output_bounds.right)
    assert _bits(actual.matrix_e0) == _bits(expected.matrix_e0)
    assert _bits(actual.matrix_170) == _bits(expected.matrix_170)
    assert _bits(actual.matrix_200) == _bits(expected.matrix_200)
    context = actual.context_after
    expected_context = expected.context_after
    for field in ("field_00", "field_7c", "field_8c", "field_98",
                  "field_911", "field_846", "field_847", "field_820",
                  "field_84c", "field_818", "field_824", "field_828"):
        assert getattr(context, field) == getattr(expected_context, field)
    assert struct.pack("=d", context.field_810) == struct.pack(
        "=d", expected_context.field_810)
    assert tuple(context.field_8a8) == expected_context.field_8a8
    assert bool(actual.flag_0c) is expected.flag_0c
    assert bool(actual.mask_290) is expected.mask_290
    assert actual.iterations == expected.iterations
    diagnostic_bytes = C.string_at(
        C.addressof(actual) + _Result.mask_first_value.offset,
        C.sizeof(C.c_double))
    diagnostic_bits = struct.unpack("=Q", diagnostic_bytes)[0]
    assert diagnostic_bits == expected.mask_first_value_bits


def _rect(top: int, left: int, height: int, width: int) -> oracle.Rect:
    return oracle.Rect(top, left, top + height, left + width)


def _case(*, height: int = 1024, width: int = 1024,
          extent3: tuple[int, int] = (256, 256),
          extent4: tuple[int, int] | None = None,
          available: tuple[bool, ...] = (True, True, True, True, True, True),
          initial_level: int = 0,
          context: oracle.Context | None = None,
          source_68a: int = 0, source_178: int = 1, param6: int = 1,
          optional_present: bool = False,
          optional_matrix: tuple[float, ...] | None = None,
          first_qword: int = 0,
          origin: tuple[int, int] = (0, 0),
          invalid_unused_layers: bool = False,
          feature_5a: int = 0) -> oracle.RenderGeometryInput:
    if context is None:
        context = oracle.Context(field_847=1,
                                 field_8a8=(first_qword, 0, 0, 0, 0, 0, 0))
    elif first_qword:
        context = replace(context, field_8a8=(first_qword, *context.field_8a8[1:]))
    levels = []
    for index, is_available in enumerate(available):
        if invalid_unused_layers and not is_available:
            ref = oracle.Rect(0, 0, 0, 0)
            actual = oracle.Rect(0, 0, 0, 0)
        else:
            level_top = min(0x7FFFFFFF, max(-0x80000000,
                                           origin[0] + index * 3))
            level_left = min(0x7FFFFFFF, max(-0x80000000,
                                             origin[1] - index * 2))
            ref = _rect(level_top, level_left, height, width)
            actual_top = max(-0x80000000, level_top - 2)
            actual_left = max(-0x80000000, level_left - 3)
            actual = _rect(actual_top, actual_left,
                           max(1, height - index), max(1, width - index))
        levels.append(oracle.Level(0x1000 + index if is_available else 0,
                                   ref, actual))
    if extent4 is None:
        extent4 = extent3
    optional = ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                if optional_matrix is None else optional_matrix)
    return oracle.RenderGeometryInput(
        levels=tuple(levels), source_68a=source_68a,
        source_178=source_178, context=context,
        initial_level=initial_level, extent3=extent3, extent4=extent4,
        optional_present=optional_present, optional_matrix=optional,
        param6=param6, feature_5a=feature_5a)


def _converted_case(dpi: float, *, feature_5a: int = 0, mode: int = 0,
                    units: int = 0, data: int = 0) -> oracle.RenderGeometryInput:
    context = oracle.Context(
        field_847=1, field_820=1, field_810=dpi, field_818=units,
        field_824=mode, field_828=data,
        field_8a8=(0x7FF8000000000001,) * 7)
    return _case(extent3=(256, 256), context=context,
                 feature_5a=feature_5a)


def _assert_case(bridge: Bridge, case: oracle.RenderGeometryInput,
                 expected_status: oracle.RenderGeometryStatus | None = None
                 ) -> oracle.RenderGeometryOutput | None:
    oracle_status, expected = oracle.prepare_render_geometry(case)
    if expected_status is not None:
        assert oracle_status == expected_status
    status, output_bytes, image_addresses = bridge.call(case)
    assert status == int(oracle_status)
    if expected is None:
        assert output_bytes == OUTPUT_SENTINEL
    else:
        assert status == int(oracle.RenderGeometryStatus.OK)
        _assert_output_matches(output_bytes, expected, image_addresses)
    return expected


def test_fixed_retry_thresholds_and_dynamic_routing(bridge: Bridge) -> None:
    exact = _case(extent3=(256, 256))
    exact_result = _assert_case(bridge, exact)
    assert exact_result is not None and exact_result.level == 0
    assert exact_result.matrix_e0[0] == 0.25

    below = _case(extent3=(255, 255))
    below_result = _assert_case(bridge, below)
    assert below.initial_level == 0
    assert below.source_68a == 0 and below.context.field_847 != 0
    assert below.context.field_846 == 0 and below.context.field_84c == 0
    assert below_result is not None and below_result.level == 1
    assert below_result.context_after.field_84c == 5
    assert below_result.iterations >= 2

    bypass_but_late = _case(
        height=128, width=128, extent3=(1, 1),
        available=(True, True, False, False, False, False),
        context=oracle.Context(field_8c=1, field_847=1,
                               field_8a8=(0, 0, 0, 0, 0, 0, 0)))
    bypass_result = _assert_case(bridge, bypass_but_late)
    assert bypass_result is not None and bypass_result.level == 1
    assert bypass_result.context_after.field_84c == 0


def test_d_context_gate_and_unsigned_version_wrap_do_not_block_late_gate(
        bridge: Bridge) -> None:
    case = _case(
        height=128, width=128, extent3=(1, 1),
        available=(True, True, False, False, False, False),
        context=oracle.Context(field_00=0xFFFFFFFF, field_911=1,
                               field_847=1,
                               field_8a8=(0, 0, 0, 0, 0, 0, 0)))
    result = _assert_case(bridge, case)
    assert result is not None and result.level == 1
    assert result.context_after.field_84c == 0

    field_7c_case = _case(
        height=128, width=128, extent3=(1, 1),
        available=(True, True, False, False, False, False),
        context=oracle.Context(field_00=0xFFFFFFFF, field_7c=7,
                               field_847=1,
                               field_8a8=(0, 0, 0, 0, 0, 0, 0)))
    field_7c_result = _assert_case(bridge, field_7c_case)
    assert field_7c_result is not None and field_7c_result.level == 1


def test_no_next_level_retries_once_without_recomputing_initial_mask(
        bridge: Bridge) -> None:
    no_next = _case(
        height=1024, width=1024, extent3=(1, 1),
        available=(True, False, False, False, False, False),
        first_qword=0x7FF8000000000001)
    result = _assert_case(bridge, no_next)
    assert result is not None
    assert result.level == 0
    assert result.mask_290 is True
    assert result.context_after.field_846 == 1
    assert result.iterations == 2


def test_mask_fold_reset_and_mixed_ratio_branch_order(bridge: Bridge) -> None:
    mixed = (128, 384)
    folded = _case(extent3=(256, 256), extent4=mixed,
                   context=oracle.Context(field_847=0))
    folded_result = _assert_case(bridge, folded)
    assert folded_result is not None
    assert folded_result.matrix_e0[0] == pytest.approx(0.125)
    assert folded_result.matrix_e0[4] == pytest.approx(0.375)
    assert folded_result.matrix_170 == oracle.IDENTITY
    assert folded_result.matrix_200 == oracle.IDENTITY

    retained = _case(extent3=(256, 256), extent4=mixed,
                     first_qword=0x3FF0000000000000)
    retained_result = _assert_case(bridge, retained)
    assert retained_result is not None and retained_result.mask_290
    assert retained_result.matrix_e0[0] == 0.25
    assert retained_result.matrix_e0[4] == 0.25
    assert retained_result.matrix_170[0] == 0.5
    assert retained_result.matrix_170[4] == 1.5
    assert retained_result.matrix_200 == oracle.IDENTITY

    up_ratio = _case(extent3=(128, 128), extent4=(256, 384),
                     context=oracle.Context(field_847=0))
    up_result = _assert_case(bridge, up_ratio)
    assert up_result is not None
    assert up_result.matrix_170 == oracle.IDENTITY
    assert up_result.matrix_200[0] == 2.0
    assert up_result.matrix_200[4] == 3.0


@pytest.mark.parametrize(("field_846", "expected_mask"), [
    (1, False), (2, True), (255, True),
], ids=["846-one-folds", "846-two-retains", "846-ff-retains"])
def test_mask_initial_byte_eor_and_converter_skip(
        bridge: Bridge, field_846: int, expected_mask: bool) -> None:
    # Raw d07024..d07038 initializes byte +0x290 with EOR(field_846, 1),
    # then skips the record/converter whenever the original byte is nonzero.
    # +0x820 is deliberately nonzero to make that skip observable.
    case = _case(
        extent3=(512, 512), extent4=(128, 1024),
        feature_5a=255,
        context=oracle.Context(field_846=field_846, field_847=1,
                               field_820=1, field_810=math.nan,
                               field_818=3, field_824=0xFFFFFFFF,
                               field_828=0xFFFFFFFF,
                               field_8a8=(0x3FF0000000000000,) * 7))
    result = _assert_case(bridge, case)
    assert result is not None
    assert result.mask_290 is expected_mask
    assert result.mask_first_value_bits == 0
    assert result.iterations == 1
    if expected_mask:
        assert result.matrix_e0[0] == 0.5
        assert result.matrix_e0[4] == 0.5
        assert result.matrix_170[0] == 0.25
        assert result.matrix_170[4] == 2.0
    else:
        assert result.matrix_e0[0] == 0.125
        assert result.matrix_e0[4] == 1.0
        assert result.matrix_170 == oracle.IDENTITY
    assert result.matrix_200 == oracle.IDENTITY


def test_converted_mask_first_qword_tables_and_boundaries(bridge: Bridge) -> None:
    default = _assert_case(bridge, _converted_case(
        700.0, feature_5a=0, mode=0xFFFFFFFF))
    assert default is not None
    assert default.mask_first_value_bits == 0x3FF8000000000000  # 1.5

    mode0 = _assert_case(bridge, _converted_case(
        0.0, feature_5a=1, mode=0))
    assert mode0 is not None
    assert mode0.mask_first_value_bits == 0x4008000000000000  # 3.0

    # Any nonzero feature byte selects the mode family.
    feature_ff = _assert_case(bridge, _converted_case(
        300.0, feature_5a=255, mode=1))
    assert feature_ff is not None
    assert feature_ff.mask_first_value_bits == 0x400999999999999A  # 3.2

    expected_by_key = {
        180: 0x3FF3333333333333,
        240: 0x3FFCCCCCCCCCCCCD,
        300: 0x400999999999999A,
        360: 0x400C000000000000,
        480: 0x400F333333333333,
    }
    expected_by_midpoint = {
        210: 0x3FF8000000000000,
        270: 0x4004000000000000,
        330: 0x400ACCCCCCCCCCCD,
        420: 0x400D99999999999A,
    }
    for mode in (1, 2):
        for dpi, expected_bits in (*expected_by_key.items(),
                                   *expected_by_midpoint.items()):
            result = _assert_case(bridge, _converted_case(
                float(dpi), feature_5a=1, mode=mode))
            assert result is not None
            assert result.mask_first_value_bits == expected_bits, (mode, dpi)

        # Exercise adjacent representable DPIs on both sides of every key.
        for key, exact_bits in expected_by_key.items():
            for dpi in (math.nextafter(float(key), -math.inf), float(key),
                        math.nextafter(float(key), math.inf)):
                result = _assert_case(bridge, _converted_case(
                    dpi, feature_5a=1, mode=mode))
                assert result is not None
                if dpi == float(key):
                    assert result.mask_first_value_bits == exact_bits


def test_converter_reciprocal_then_multiply_rounding_is_preserved(
        bridge: Bridge) -> None:
    # Raw b5a7d4..b5a7e0 computes reciprocal(key span), then delta DPI times
    # that rounded reciprocal. Direct division gives ...662 for the first
    # canary, while reciprocal-then-multiply gives ...661; expectations were
    # independently recomputed with libm fma.
    canaries = (
        (0x406DA1109391867C, 0x3FFC53486AF7B661),
        (struct.unpack("=Q", struct.pack("=d", 352.85439626880367))[0],
         0x400BB6D43A8292CD),
    )
    for dpi_bits, expected_bits in canaries:
        dpi = struct.unpack("=d", struct.pack("=Q", dpi_bits))[0]
        result = _assert_case(bridge, _converted_case(
            dpi, feature_5a=1, mode=1))
        assert result is not None
        assert result.mask_first_value_bits == expected_bits


def test_converter_units_feature_mode_and_overflow_scope(bridge: Bridge) -> None:
    converted = _assert_case(bridge, _converted_case(
        100.0, feature_5a=1, mode=1, units=3, data=0xFFFFFFFF))
    raw_units = _assert_case(bridge, _converted_case(
        254.0, feature_5a=1, mode=1, units=0, data=0))
    assert converted is not None and raw_units is not None
    assert converted.mask_first_value_bits == 0x40010369D0369D04
    assert raw_units.mask_first_value_bits == 0x40010369D0369D04
    assert converted.context_after.field_828 == 0xFFFFFFFF
    assert raw_units.context_after.field_828 == 0

    # The caller multiplies even when units is not three. The unselected
    # overflow is ignored; a selected overflow reaches the last endpoint.
    unselected_overflow = _assert_case(bridge, _converted_case(
        1.0e308, feature_5a=1, mode=1, units=2))
    selected_overflow = _assert_case(bridge, _converted_case(
        1.0e308, feature_5a=1, mode=1, units=3))
    assert unselected_overflow is not None and selected_overflow is not None
    assert unselected_overflow.mask_first_value_bits == 0x400F333333333333
    assert selected_overflow.mask_first_value_bits == 0x400F333333333333

    invalid_modes_ignored = _assert_case(bridge, _converted_case(
        123.0, feature_5a=0, mode=0xFFFFFFFF))
    assert invalid_modes_ignored is not None
    assert invalid_modes_ignored.mask_first_value_bits == 0x3FF8000000000000


@pytest.mark.parametrize("field_810", [math.nan, math.inf, -math.inf],
                         ids=["nan-dpi", "positive-infinity-dpi",
                              "negative-infinity-dpi"])
def test_converter_nonfinite_original_dpi_fails_atomically(
        bridge: Bridge, field_810: float) -> None:
    _assert_case(bridge, _converted_case(
        field_810, feature_5a=1, mode=1),
        oracle.RenderGeometryStatus.UNSUPPORTED_MASK_CONVERSION)


def test_converter_unsupported_mode_fails_atomically(bridge: Bridge) -> None:
    _assert_case(bridge, _converted_case(
        200.0, feature_5a=255, mode=3),
        oracle.RenderGeometryStatus.UNSUPPORTED_MASK_CONVERSION)


@pytest.mark.parametrize(("first_qword", "expected_mask"), [
    (0x0000000000000000, False),
    (0x8000000000000000, False),
    (0x7FF8000000000001, True),
], ids=["positive-zero", "negative-zero", "nan"])
def test_mask_first_word_exact_ieee_zero_predicate(
        bridge: Bridge, first_qword: int, expected_mask: bool) -> None:
    case = _case(extent3=(256, 256), first_qword=first_qword)
    result = _assert_case(bridge, case)
    assert result is not None and result.mask_290 is expected_mask
    assert result.mask_first_value_bits == first_qword


def test_int32_minimum_origin_uses_wrapped_negate_bits(bridge: Bridge) -> None:
    minimum = -0x80000000
    case = _case(height=1024, width=1024, extent3=(1, 1),
                 available=(True, False, False, False, False, False),
                 context=oracle.Context(field_847=0),
                 origin=(minimum, minimum))
    result = _assert_case(bridge, case)
    assert result is not None
    assert struct.unpack("=Q", struct.pack("=d", result.matrix_e0[2]))[0] == (
        0xC140000000000000)  # exactly -2^21
    assert struct.unpack("=Q", struct.pack("=d", result.matrix_e0[5]))[0] == (
        0xC140000000000000)


def test_unused_optional_nan_unused_layers_and_converter_scope(
        bridge: Bridge) -> None:
    ignored_nan = _case(
        extent3=(256, 256), optional_present=False,
        optional_matrix=(math.nan,) * 9,
        available=(True, False, False, False, False, False),
        context=oracle.Context(field_847=0, field_820=1,
                               field_810=math.nan, field_818=3,
                               field_824=0xFFFFFFFF,
                               field_828=0xFFFFFFFF,
                               field_8a8=(0x7FF8000000000001,) * 7),
        invalid_unused_layers=True, feature_5a=255)
    result = _assert_case(bridge, ignored_nan)
    assert result is not None and result.mask_290 is False

    converter = _case(
        extent3=(256, 256),
        context=oracle.Context(field_847=1, field_820=1,
                               field_824=3, field_810=200.0,
                               field_8a8=(0x3FF0000000000000,) * 7),
        feature_5a=1)
    _assert_case(bridge, converter,
                 oracle.RenderGeometryStatus.UNSUPPORTED_MASK_CONVERSION)

    unused_converter = _case(
        extent3=(256, 256), param6=0,
        feature_5a=255,
        context=oracle.Context(field_847=1, field_820=1,
                               field_810=math.nan, field_818=3,
                               field_824=0xFFFFFFFF,
                               field_828=0xFFFFFFFF,
                               field_8a8=(0x3FF0000000000000,) * 7))
    result = _assert_case(bridge, unused_converter)
    assert result is not None and result.mask_290 is False


@pytest.mark.parametrize(("case", "status"), [
    (_case(available=(False, True, True, True, True, True)),
     oracle.RenderGeometryStatus.UNAVAILABLE_LEVEL),
    (_case(height=0), oracle.RenderGeometryStatus.INVALID_BOUNDS),
    (_case(extent3=(-1, 8)), oracle.RenderGeometryStatus.INVALID_INPUT),
    (_case(optional_present=True, optional_matrix=(math.nan,) + (0.0,) * 8),
     oracle.RenderGeometryStatus.NON_FINITE_MATRIX),
    (_case(optional_present=True, optional_matrix=(1.0e308,) * 9,
           context=oracle.Context(field_847=0)),
     oracle.RenderGeometryStatus.NON_FINITE_MATRIX),
    (_case(height=1024, width=1024, extent3=(1024, 1024),
           optional_present=True,
           optional_matrix=(9.0e307, 9.0e307, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0)),
     oracle.RenderGeometryStatus.NON_FINITE_MATRIX),
], ids=["missing-current-image", "zero-reference-span", "negative-extent",
       "non-finite-optional", "finite-matrix-overflow",
       "finite-norm-overflow"])
def test_failures_preserve_entire_result_bytes(
        bridge: Bridge, case: oracle.RenderGeometryInput,
        status: oracle.RenderGeometryStatus) -> None:
    _assert_case(bridge, case, status)


def test_invalid_final_actual_bounds_preserve_entire_result_bytes(
        bridge: Bridge) -> None:
    case = _case(extent3=(256, 256),
                 available=(True, False, False, False, False, False),
                 context=oracle.Context(field_847=0))
    levels = list(case.levels)
    levels[0] = replace(levels[0], actual_bounds=oracle.Rect(0, 0, 0, 1024))
    case = replace(case, levels=tuple(levels))
    _assert_case(bridge, case, oracle.RenderGeometryStatus.INVALID_BOUNDS)


def _random_case(rng: random.Random, index: int
                 ) -> oracle.RenderGeometryInput:
    height = rng.choice((32, 64, 128, 256, 512, 1024, 2048))
    width = rng.choice((32, 64, 128, 256, 512, 1024, 2048))
    origin = (rng.randint(-5000, 5000), rng.randint(-5000, 5000))
    converted_branch = (index % 2) == 0
    initial = 0 if converted_branch else rng.randrange(6)
    availability = [rng.random() < 0.68 for _ in range(6)]
    availability[initial] = True
    if converted_branch:
        feature_5a = rng.choice((0, 1, 255))
        mode = (rng.getrandbits(32) if feature_5a == 0
                else rng.randrange(3))
        dpi = rng.choice((rng.uniform(120.0, 520.0), 180.0, 270.0,
                          300.0, 330.0, 360.0, 480.0))
        context = oracle.Context(
            field_00=rng.choice((0, 0x0506FFFF, 0x05070000, 0xFFFFFFFF)),
            field_7c=rng.choice((0, 0, 1, 7)),
            field_8c=rng.choice((0, 0, 1)),
            field_98=rng.choice((0, 0, 1)),
            field_911=rng.choice((0, 0, 1, 0xFFFFFFFF)),
            field_846=0, field_847=1, field_820=1,
            field_84c=rng.choice((0, 0, 2, 5)),
            field_8a8=(rng.getrandbits(64),) * 7,
            field_810=dpi, field_818=rng.choice((-1, 0, 3)),
            field_824=mode, field_828=rng.getrandbits(32))
        source_68a = 0
        param6 = 1
    else:
        feature_5a = rng.choice((0, 1, 255))
        context = oracle.Context(
            field_00=rng.choice((0, 0x0506FFFF, 0x05070000, 0xFFFFFFFF)),
            field_7c=rng.choice((0, 0, 1, 7)),
            field_8c=rng.choice((0, 0, 1)),
            field_98=rng.choice((0, 0, 1)),
            field_911=rng.choice((0, 0, 1, 0xFFFFFFFF)),
            field_846=rng.choice((0, 0, 1, 2, 255)),
            field_847=rng.choice((0, 0, 1)),
            field_820=rng.choice((0, 0, 1)),
            field_84c=rng.choice((0, 0, 2, 5)),
            field_8a8=(rng.choice((0, 0x3FF0000000000000,
                                   0x8000000000000000,
                                   0x7FF8000000000042)),
                       rng.getrandbits(64), rng.getrandbits(64),
                       rng.getrandbits(64), rng.getrandbits(64),
                       rng.getrandbits(64), rng.getrandbits(64)),
            field_810=rng.choice((math.nan, math.inf,
                                  rng.uniform(-100.0, 600.0))),
            field_818=rng.choice((-1, 0, 3)),
            field_824=rng.choice((0, 1, 2, 3, 0xFFFFFFFF)),
            field_828=rng.getrandbits(32))
        source_68a = rng.choice((0, 0, 1))
        param6 = rng.choice((0, 1))
    optional = rng.random() < 0.2
    if optional:
        optional_matrix = (
            1.0 + rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), 0.0,
            rng.uniform(-0.05, 0.05), 1.0 + rng.uniform(-0.1, 0.1), 0.0,
            0.0, 0.0, 1.0)
    else:
        optional_matrix = (math.nan,) * 9
    extent3 = (rng.choice((1, 8, 32, 64, 128, 256, 512)),
               rng.choice((1, 8, 32, 64, 128, 256, 512)))
    extent4 = (rng.choice((1, 8, 32, 64, 128, 256, 512)),
               rng.choice((1, 8, 32, 64, 128, 256, 512)))
    return _case(height=height, width=width, extent3=extent3,
                 extent4=extent4, available=tuple(availability),
                 initial_level=initial, context=context,
                 source_68a=source_68a,
                 source_178=rng.choice((0, 1, 2)),
                 param6=param6, optional_present=optional,
                 optional_matrix=optional_matrix, origin=origin,
                 feature_5a=feature_5a)


def test_seeded_dynamic_random_cases_match_independent_oracle(
        bridge: Bridge) -> None:
    rng = random.Random(20261024)
    for index in range(64):
        case = _random_case(rng, index)
        _assert_case(bridge, case)

