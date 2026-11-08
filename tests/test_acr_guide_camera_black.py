"""Bitwise tests for the prepared Camera -> post-camera-black composition."""

from __future__ import annotations

import ctypes as C
import itertools
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools/acr_guide_camera_black_bridge.cpp"
SOURCE_FILES = (
    "acr_guide_camera_black",
    "acr_camera_combined",
    "acr_postcamera_black",
    "acr_buffer_format",
    "acr_pipe_format",
    "acr_float_lut",
    "acr_u16_lut",
    "acr_tone_kernels",
)
SOURCES = {name: NATIVE / "src" / f"{name}.cpp" for name in SOURCE_FILES}
HEADERS = {name: NATIVE / "src" / f"{name}.hpp" for name in SOURCE_FILES}
OBJECTS = {
    name: NATIVE / "build/CMakeFiles/imprint_renderer.dir/src" / f"{name}.cpp.o"
    for name in SOURCE_FILES
}

sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import buffer_format_oracle as buffer_oracle  # noqa: E402
import camera_combined_oracle as camera_oracle  # noqa: E402
import postcamera_black_oracle as black_oracle  # noqa: E402
import u16_lut_oracle  # noqa: E402


FLOAT = C.c_float
FLOAT_PTR = C.POINTER(FLOAT)
DOUBLE = C.c_double
DOUBLE_PTR = C.POINTER(DOUBLE)
U32 = C.c_uint32
U32_PTR = C.POINTER(U32)
SIZE = C.c_size_t
SIZE_MAX = C.c_size_t(-1).value
UINTPTR_MAX = C.c_size_t(-1).value
SENTINEL = np.float32(-9123.25)

OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
INVALID_STRIDE = 3
BUFFER_TOO_SMALL = 4
ARITHMETIC_OVERFLOW = 5
RESOURCE_LIMIT = 6
INVALID_CONTROLS = 9
CAMERA_FAILURE = 10
BLACK_FAILURE = 12


class Snapshot(C.Structure):
    _fields_ = [
        ("height", SIZE),
        ("width", SIZE),
        ("camera_type", U32),
        ("black_type", U32),
        ("black_encoding", C.c_uint8),
        ("quantized_before_black", C.c_uint8),
        ("rgb_size", SIZE),
        ("pin_type", U32),
    ]


@dataclass
class Result:
    status: int
    height: int
    width: int
    camera_type: int
    black_type: int
    black_encoding: int
    quantized_before_black: int
    rgb: np.ndarray
    rgb_size: int
    pin_type: int

    @property
    def plan(self):
        return (
            self.camera_type,
            self.black_type,
            self.black_encoding,
            self.quantized_before_black,
        )


def _float_ptr(values: np.ndarray) -> FLOAT_PTR:
    return values.ctypes.data_as(FLOAT_PTR)


def _double_ptr(values: np.ndarray) -> DOUBLE_PTR:
    return values.ctypes.data_as(DOUBLE_PTR)


def _u32_ptr(values: np.ndarray) -> U32_PTR:
    return values.ctypes.data_as(U32_PTR)


def _null_float_ptr() -> FLOAT_PTR:
    return C.cast(C.c_void_p(0), FLOAT_PTR)


class NativeBridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.call_fn = self.library.acr_test_guide_camera_black
        self.call_fn.argtypes = [
            FLOAT_PTR,
            SIZE,
            SIZE,
            SIZE,
            SIZE,
            DOUBLE_PTR,
            U32_PTR,
            U32,
            U32,
            SIZE,
            SIZE,
            U32,
            U32,
            C.c_uint8,
            C.c_uint8,
            FLOAT_PTR,
            SIZE,
            U32,
            C.POINTER(Snapshot),
            FLOAT_PTR,
            SIZE,
        ]
        self.call_fn.restype = C.c_int
        self.bind_fn = self.library.acr_test_bind_guide_camera_black
        self.bind_fn.argtypes = [FLOAT_PTR, FLOAT_PTR, U32_PTR, FLOAT_PTR,
                                U32_PTR, U32_PTR, DOUBLE_PTR]
        self.bind_fn.restype = C.c_int

    def bind(self, source_fields, *, order=(2, 0, 1),
             gains=(8192, 4096, 16384), matrix=None,
             sorted_matrix=None, gain_float=None):
        if matrix is None:
            matrix = np.asarray([1.0, 0.125, -0.25, -0.125, 1.0, 0.25,
                                 0.25, -0.5, 1.0], dtype=np.float32)
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if sorted_matrix is None:
            sorted_matrix = matrix.reshape(3, 3)[:, order].reshape(-1)
        sorted_matrix = np.ascontiguousarray(sorted_matrix, dtype=np.float32)
        gains = np.ascontiguousarray(gains, dtype=np.uint32)
        if gain_float is None:
            gain_float = gains.astype(np.float32) / np.float32(4096.0)
        gain_float = np.ascontiguousarray(gain_float, dtype=np.float32)
        order = np.ascontiguousarray(order, dtype=np.uint32)
        source_fields = np.ascontiguousarray(source_fields, dtype=np.uint32)
        output = np.full(18, -1234.5, dtype=np.float64)
        status = self.bind_fn(_float_ptr(matrix), _float_ptr(sorted_matrix),
            _u32_ptr(gains), _float_ptr(gain_float), _u32_ptr(order),
            _u32_ptr(source_fields), _double_ptr(output))
        return status, output

    def raw(
        self,
        *,
        source_ptr: FLOAT_PTR,
        source_count: int,
        source_row_stride: int,
        height: int,
        width: int,
        controls: np.ndarray,
        order: np.ndarray,
        flags: int = 0,
        selector: int = 1,
        old_height: int = 91,
        old_width: int = 87,
        old_camera_type: int = 0xA1B2C3D4,
        old_black_type: int = 0xB2C3D4E5,
        old_black_encoding: int = 0xE7,
        old_quantized_before_black: int = 1,
        old_rgb: np.ndarray | None = None,
        source_is_old_rgb: bool = False,
        result_capacity: int | None = None,
    ) -> Result:
        if old_rgb is None:
            old_rgb = np.asarray(
                [0x80000000, 0x7FC01234, 0xFF800000, 0x3EAAAAAB, 0xFFC54321],
                dtype=np.uint32,
            ).view(np.float32)
        old_rgb = np.ascontiguousarray(old_rgb, dtype=np.float32)
        if result_capacity is None:
            result_capacity = max(old_rgb.size, 1)
            if 0 < height <= 1024 and 0 < width <= 1024:
                result_capacity = max(result_capacity, height * width * 3)
        result_storage = np.full(max(result_capacity, 1), SENTINEL, dtype=np.float32)
        snapshot = Snapshot()
        status = int(
            self.call_fn(
                source_ptr,
                source_count,
                source_row_stride,
                height,
                width,
                _double_ptr(controls),
                _u32_ptr(order),
                flags,
                selector,
                old_height,
                old_width,
                old_camera_type,
                old_black_type,
                old_black_encoding,
                old_quantized_before_black,
                _float_ptr(old_rgb),
                old_rgb.size,
                1 if source_is_old_rgb else 0,
                C.byref(snapshot),
                _float_ptr(result_storage),
                result_capacity,
            )
        )
        if snapshot.rgb_size > result_capacity:
            rgb = result_storage.copy()
        else:
            rgb = result_storage[: snapshot.rgb_size].copy()
        return Result(
            status,
            int(snapshot.height),
            int(snapshot.width),
            int(snapshot.camera_type),
            int(snapshot.black_type),
            int(snapshot.black_encoding),
            int(snapshot.quantized_before_black),
            rgb,
            int(snapshot.rgb_size),
            int(snapshot.pin_type),
        )

    def call(
        self,
        source: np.ndarray | None,
        controls: np.ndarray,
        order: np.ndarray,
        *,
        height: int,
        width: int,
        source_row_stride: int,
        source_count: int | None = None,
        flags: int = 0,
        selector: int = 1,
        **kwargs,
    ) -> Result:
        if source is None:
            pointer = _null_float_ptr()
            count = 0 if source_count is None else source_count
        else:
            source = np.ascontiguousarray(source, dtype=np.float32).reshape(-1)
            pointer = _float_ptr(source)
            count = source.size if source_count is None else source_count
        return self.raw(
            source_ptr=pointer,
            source_count=count,
            source_row_stride=source_row_stride,
            height=height,
            width=width,
            controls=controls,
            order=order,
            flags=flags,
            selector=selector,
            **kwargs,
        )


def _bridge_inputs(kind: str) -> list[Path]:
    for name in SOURCE_FILES:
        assert SOURCES[name].is_file(), f"missing composition source: {SOURCES[name]}"
        assert HEADERS[name].is_file(), f"missing composition header: {HEADERS[name]}"
    assert BRIDGE_SOURCE.is_file(), f"missing test bridge: {BRIDGE_SOURCE}"
    if kind == "source":
        return [BRIDGE_SOURCE, *(SOURCES[name] for name in SOURCE_FILES)]

    for name in SOURCE_FILES:
        obj, source, header = OBJECTS[name], SOURCES[name], HEADERS[name]
        assert obj.is_file(), f"rebuild renderer: missing official object {obj}"
        assert obj.stat().st_mtime_ns >= max(source.stat().st_mtime_ns, header.stat().st_mtime_ns), (
            f"rebuild renderer: stale official object {obj.name}"
        )
    return [BRIDGE_SOURCE, *(OBJECTS[name] for name in SOURCE_FILES)]


@pytest.fixture(scope="module", params=("source", "official_object"), ids=("source", "official_object"))
def native(request, tmp_path_factory) -> NativeBridge:
    inputs = _bridge_inputs(request.param)
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the prepared composition bridge"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_guide_camera_black_" + request.param) / (
        "libguide_camera_black" + suffix
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
        *(str(path) for path in inputs),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return NativeBridge(output)


def make_controls(*, direct: bool, black: float, upper: float = 1.0) -> np.ndarray:
    a, b_gain = (1.0, 1.0) if direct else (4.0, 2.0)
    return np.asarray(
        [a, b_gain, upper, black, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )


def expected_pixels(
    source: np.ndarray,
    controls: np.ndarray,
    order: np.ndarray,
    flags: int,
    selector: int,
    *,
    height: int,
    width: int,
    source_row_stride: int,
) -> np.ndarray:
    camera = camera_oracle.apply(
        source,
        controls,
        order,
        flags,
        height=height,
        width=width,
        src_row_stride=source_row_stride,
        dst_row_stride=width * 3,
        destination=np.full(height * width * 3, SENTINEL, dtype=np.float32),
    )
    camera = camera[: height * width * 3].copy()
    black = float(controls[3])
    has_pin = bool(flags & 1)
    if selector == 1 and (black > 0.0 or has_pin):
        quantized = buffer_oracle.floats_to_u16(camera)
        if black > 0.0:
            black_table = u16_lut_oracle.build(black_oracle.build_lut(black))
            quantized = black_table[quantized]
        return buffer_oracle.u16s_to_float(quantized)
    if has_pin:
        # 00bc4b44 range tracking invokes 00bcb3c0 before the range=0
        # float consumer, not after an out-of-domain unsigned LUT access.
        camera = np.where(camera > 0.0, np.minimum(camera, np.float32(1.0)),
                          np.float32(0.0)).astype(np.float32)
    if not black > 0.0:
        return camera
    return black_oracle.apply(camera.reshape(height, width, 3), black).reshape(-1)


def assert_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32).view(np.uint32),
        np.asarray(expected, dtype=np.float32).view(np.uint32),
    )


def assert_success(
    result: Result,
    expected: np.ndarray,
    *,
    height: int,
    width: int,
    selector: int,
    black: float,
    flags: int = 0,
) -> None:
    has_black = black > 0.0
    assert result.status == OK
    assert (result.height, result.width) == (height, width)
    assert result.plan == (
        2,
        selector if has_black else 0,
        0,
        int(has_black and selector == 1),
    )
    assert result.rgb_size == height * width * 3
    assert result.pin_type == (selector if flags & 1 else 0)
    assert result.rgb.size == result.rgb_size
    assert_bits(result.rgb, expected)


def old_result() -> tuple[dict, np.ndarray]:
    rgb = np.asarray(
        [0x80000000, 0x7FC01234, 0xFF800000, 0x3EAAAAAB, 0xFFC54321],
        dtype=np.uint32,
    ).view(np.float32)
    fields = {
        "old_height": 91,
        "old_width": 87,
        "old_camera_type": 0xA1B2C3D4,
        "old_black_type": 0xB2C3D4E5,
        "old_black_encoding": 0xE7,
        "old_quantized_before_black": 1,
        "old_rgb": rgb,
    }
    return fields, rgb.copy()


def assert_old_result_preserved(result: Result, fields: dict, rgb_before: np.ndarray) -> None:
    assert result.status != OK
    assert (result.height, result.width) == (fields["old_height"], fields["old_width"])
    assert result.plan == (
        fields["old_camera_type"],
        fields["old_black_type"],
        fields["old_black_encoding"],
        fields["old_quantized_before_black"],
    )
    assert result.rgb_size == rgb_before.size
    assert result.pin_type == 0xC3D4E5F6
    assert_bits(result.rgb, rgb_before)


def test_randomized_composition_all_orders_modes_flags_and_odd_padded_geometry(native):
    rng = np.random.default_rng(20261107)
    height, width = 3, 5
    active = width * 3
    stride = active + 4
    source = np.full(height * stride, np.float32(np.nan), dtype=np.float32)
    for row in range(height):
        source[row * stride : row * stride + active] = rng.uniform(
            0.0, 0.93, size=active
        ).astype(np.float32)
    before = source.view(np.uint32).copy()
    black = 4100.0 / 65535.0

    for direct in (True, False):
        controls = make_controls(direct=direct, black=black)
        for order_values in itertools.permutations((0, 1, 2)):
            order = np.asarray(order_values, dtype=np.uint32)
            for flags in (0, 1):
                for selector in (1, 2):
                    expected = expected_pixels(
                        source,
                        controls,
                        order,
                        flags,
                        selector,
                        height=height,
                        width=width,
                        source_row_stride=stride,
                    )
                    actual = native.call(
                        source,
                        controls,
                        order,
                        height=height,
                        width=width,
                        source_row_stride=stride,
                        flags=flags,
                        selector=selector,
                    )
                    assert_success(
                        actual,
                        expected,
                        height=height,
                        width=width,
                        selector=selector,
                        black=black,
                        flags=flags,
                    )
                    assert_bits(source, before.view(np.float32))


def test_constant_near_black_and_threshold_inputs_match_both_independent_modes(native):
    black = 4100.0 / 65535.0
    black32 = np.float32(black)
    near_zero = np.nextafter(np.float32(0.0), np.float32(1.0))
    near_b = np.asarray(
        [
            np.nextafter(black32, np.float32(-np.inf)),
            black32,
            np.nextafter(black32, np.float32(np.inf)),
        ],
        dtype=np.float32,
    )
    grid = np.asarray([0.0, 1.0 / 4096.0, 17.0 / 4096.0, 0.5, 1.0], dtype=np.float32)
    around_grid = np.concatenate(
        [
            np.nextafter(grid, np.float32(-np.inf)),
            grid,
            np.nextafter(grid, np.float32(np.inf)),
        ]
    )
    cases = (
        ("constant", np.full(15, 0.375, dtype=np.float32)),
        (
            "near_black",
            np.asarray([0.0, -0.0, near_zero, *near_b, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float32),
        ),
        ("threshold", around_grid[:15].astype(np.float32)),
    )
    order = np.asarray([2, 0, 1], dtype=np.uint32)
    controls = make_controls(direct=False, black=black)
    height, width, stride = 1, 5, 15
    for _case_name, values in cases:
        source = np.resize(values, height * stride).astype(np.float32)
        before = source.view(np.uint32).copy()
        for selector in (1, 2):
            expected = expected_pixels(
                source,
                controls,
                order,
                0,
                selector,
                height=height,
                width=width,
                source_row_stride=stride,
            )
            result = native.call(
                source,
                controls,
                order,
                height=height,
                width=width,
                source_row_stride=stride,
                selector=selector,
            )
            assert_success(
                result,
                expected,
                height=height,
                width=width,
                selector=selector,
                black=black,
            )
            assert_bits(source, before.view(np.float32))


@pytest.mark.parametrize("black", (0.0, -0.0), ids=("positive-zero", "negative-zero"))
def test_zero_double_black_without_pin_skips_selector1_quantization(native, black):
    source = np.asarray(
        [0.0, -0.0, np.nextafter(np.float32(0.0), np.float32(1.0)), 0.1234567, 0.5, 1.0],
        dtype=np.float32,
    )
    controls = make_controls(direct=True, black=black)
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    expected = expected_pixels(
        source,
        controls,
        order,
        0,
        1,
        height=1,
        width=2,
        source_row_stride=6,
    )
    result = native.call(
        source,
        controls,
        order,
        height=1,
        width=2,
        source_row_stride=6,
        flags=0,
        selector=1,
    )
    assert_success(result, expected, height=1, width=2, selector=1, black=black)
    forced_quantization = buffer_oracle.u16s_to_float(
        buffer_oracle.floats_to_u16(expected)
    )
    assert np.any(expected.view(np.uint32) != forced_quantization.view(np.uint32))


def test_positive_minimum_double_black_still_adds_mode1_quantization(native):
    black = float(np.nextafter(np.float64(0.0), np.float64(1.0)))
    assert float(np.float32(black)) == 0.0
    source = np.asarray([0.1234567, 0.2345678, 0.8765432], dtype=np.float32)
    controls = make_controls(direct=True, black=black)
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    expected = expected_pixels(
        source,
        controls,
        order,
        1,
        1,
        height=1,
        width=1,
        source_row_stride=3,
    )
    camera_only = camera_oracle.apply(
        source,
        controls,
        order,
        1,
        height=1,
        width=1,
        src_row_stride=3,
    )
    result = native.call(
        source,
        controls,
        order,
        height=1,
        width=1,
        source_row_stride=3,
        flags=1,
        selector=1,
    )
    assert_success(result, expected, height=1, width=1, selector=1, black=black, flags=1)
    assert np.any(expected.view(np.uint32) != camera_only.view(np.uint32))


@pytest.mark.parametrize(
    "upper",
    (float(np.float32(32768.0 / 65535.0)), 1.0),
    ids=("lower-bound", "upper-bound"),
)
@pytest.mark.parametrize("selector", (1, 2), ids=("mode1", "mode2"))
def test_camera_upper_control_boundary_values_match_oracles(native, upper, selector):
    black = 4100.0 / 65535.0
    controls = make_controls(direct=False, black=black, upper=upper)
    order = np.asarray([1, 2, 0], dtype=np.uint32)
    source = np.asarray([0.12, 0.24, 0.36, 0.48, 0.6, 0.72], dtype=np.float32)
    expected = expected_pixels(
        source,
        controls,
        order,
        0,
        selector,
        height=1,
        width=2,
        source_row_stride=6,
    )
    result = native.call(
        source,
        controls,
        order,
        height=1,
        width=2,
        source_row_stride=6,
        selector=selector,
    )
    assert_success(result, expected, height=1, width=2, selector=selector, black=black)


def test_mode1_and_mode2_fixed_golden_are_distinct_and_plan_is_resolved_before_gray(native):
    black = 4100.0 / 65535.0
    controls = make_controls(direct=True, black=black)
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    source = np.asarray(
        [0.1234567, 0.2345678, 0.8765432, 0.5, 0.01001, 0.49999, 1.0, 0.01, 0.9999],
        dtype=np.float32,
    )
    expected_mode1 = np.asarray(
        [1032128645, 1044111548, 1063143646, 1055844591, 0, 1055844591,
         1065353216, 0, 1065351424],
        dtype=np.uint32,
    ).view(np.float32)
    expected_mode2 = np.asarray(
        [1032128779, 1044112254, 1063143725, 1055844943, 0, 1055844586,
         1065353216, 0, 1065351426],
        dtype=np.uint32,
    ).view(np.float32)

    actual = {}
    for selector in (1, 2):
        expected = expected_pixels(
            source,
            controls,
            order,
            0,
            selector,
            height=1,
            width=3,
            source_row_stride=9,
        )
        assert_bits(expected, expected_mode1 if selector == 1 else expected_mode2)
        result = native.call(
            source,
            controls,
            order,
            height=1,
            width=3,
            source_row_stride=9,
            selector=selector,
        )
        assert_success(result, expected, height=1, width=3, selector=selector, black=black)
        actual[selector] = result.rgb

    assert np.any(actual[1].view(np.uint32) != actual[2].view(np.uint32))


def test_failure_statuses_leave_all_old_result_fields_and_rgb_bytes_unchanged(native):
    base_controls = make_controls(direct=True, black=4100.0 / 65535.0)
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    source = np.full(24, np.float32(0.25), dtype=np.float32)

    def failure(
        *,
        expected_status: int,
        controls: np.ndarray = base_controls,
        candidate_order: np.ndarray = order,
        source_values: np.ndarray | None = source,
        height: int = 2,
        width: int = 4,
        stride: int = 12,
        count: int | None = None,
        selector: int = 1,
        flags: int = 0,
        source_ptr: FLOAT_PTR | None = None,
    ) -> None:
        fields, rgb_before = old_result()
        if source_ptr is not None:
            result = native.raw(
                source_ptr=source_ptr,
                source_count=source.size if count is None else count,
                source_row_stride=stride,
                height=height,
                width=width,
                controls=controls,
                order=candidate_order,
                selector=selector,
                flags=flags,
                **fields,
            )
        else:
            result = native.call(
                source_values,
                controls,
                candidate_order,
                height=height,
                width=width,
                source_row_stride=stride,
                source_count=count,
                selector=selector,
                flags=flags,
                **fields,
            )
        assert result.status == expected_status
        assert_old_result_preserved(result, fields, rgb_before)

    for selector in (0, 3, 99):
        failure(expected_status=INVALID_CONTROLS, selector=selector)
    for flags in (2, 3, 0xFFFFFFFF):
        failure(expected_status=INVALID_CONTROLS, flags=flags)

    for bad_black in (-0.01, 1.0, 65535.0, float("nan"), float("inf")):
        bad_controls = base_controls.copy()
        bad_controls[3] = bad_black
        failure(expected_status=INVALID_CONTROLS, controls=bad_controls)

    bad_order = np.asarray([0, 0, 2], dtype=np.uint32)
    failure(expected_status=INVALID_CONTROLS, candidate_order=bad_order)
    bad_controls = base_controls.copy()
    bad_controls[0] = 0.5
    failure(expected_status=INVALID_CONTROLS, controls=bad_controls)
    bad_controls = base_controls.copy()
    bad_controls[2] = float(
        np.nextafter(np.float32(32768.0 / 65535.0), np.float32(0.0))
    )
    failure(expected_status=INVALID_CONTROLS, controls=bad_controls)
    bad_controls = base_controls.copy()
    bad_controls[7] = np.nan
    failure(expected_status=INVALID_CONTROLS, controls=bad_controls)

    failure(expected_status=INVALID_DIMENSIONS, height=0)
    failure(expected_status=INVALID_STRIDE, stride=11)
    failure(expected_status=BUFFER_TOO_SMALL, count=23)

    failure(
        expected_status=ARITHMETIC_OVERFLOW,
        height=2,
        width=1,
        stride=SIZE_MAX,
        count=SIZE_MAX,
    )
    near_address = C.cast(C.c_void_p(UINTPTR_MAX - 3), FLOAT_PTR)
    failure(
        expected_status=ARITHMETIC_OVERFLOW,
        height=1,
        width=1,
        stride=3,
        count=3,
        source_ptr=near_address,
    )

    resource_height = 128 * 1024 * 1024 + 1
    failure(
        expected_status=RESOURCE_LIMIT,
        height=resource_height,
        width=1,
        stride=3,
        count=resource_height * 3,
    )

    for bad_value in (np.nan, np.inf, -np.inf):
        bad_source = source.copy()
        bad_source[1] = np.float32(bad_value)
        failure(expected_status=CAMERA_FAILURE, source_values=bad_source)


def test_selector2_pipe_pins_late_overrange_before_unsigned_black_lut(native):
    black = 4100.0 / 65535.0
    controls = make_controls(direct=True, black=black)
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    source = np.asarray(
        [0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 1.25], dtype=np.float32
    )
    camera = camera_oracle.apply(
        source,
        controls,
        order,
        1,
        height=1,
        width=3,
        src_row_stride=9,
    )
    assert camera[-1] > 1.0
    result = native.call(
        source,
        controls,
        order,
        height=1,
        width=3,
        source_row_stride=9,
        flags=1,
        selector=2,
    )
    expected = expected_pixels(
        source, controls, order, 1, 2, height=1, width=3, source_row_stride=9,
    )
    assert_success(result, expected, height=1, width=3, selector=2, black=black, flags=1)
    assert result.rgb[-1] == np.float32(1.0)


@pytest.mark.parametrize("selector", (1, 2), ids=("mode1", "mode2"))
def test_old_result_rgb_alias_is_safe_on_success_and_failure(native, selector):
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    black = 4100.0 / 65535.0
    controls = make_controls(direct=True, black=black)
    source = np.asarray([0.1234567, 0.2345678, 0.8765432, 0.5, 0.25, 0.75], dtype=np.float32)
    fields = {
        "old_height": 17,
        "old_width": 19,
        "old_camera_type": 0x10203040,
        "old_black_type": 0x50607080,
        "old_black_encoding": 0x91,
        "old_quantized_before_black": 0,
        "old_rgb": source.copy(),
        "source_is_old_rgb": True,
    }
    expected = expected_pixels(
        source,
        controls,
        order,
        0,
        selector,
        height=1,
        width=2,
        source_row_stride=6,
    )
    success = native.call(
        None,
        controls,
        order,
        height=1,
        width=2,
        source_row_stride=6,
        source_count=source.size,
        selector=selector,
        **fields,
    )
    assert_success(success, expected, height=1, width=2, selector=selector, black=black)

    if selector == 2:
        failing_source = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, np.nan], dtype=np.float32)
        failure_fields = dict(fields)
        failure_fields["old_rgb"] = failing_source.copy()
        before = failing_source.copy()
        failed = native.call(
            None,
            controls,
            order,
            height=1,
            width=2,
            source_row_stride=6,
            source_count=failing_source.size,
            flags=1,
            selector=selector,
            **failure_fields,
        )
        assert failed.status == CAMERA_FAILURE
        assert_old_result_preserved(failed, failure_fields, before)


@pytest.mark.parametrize("selector", (1, 2), ids=("integer-pin", "float-pin"))
@pytest.mark.parametrize("black", (0.0, -0.0, 4100.0 / 65535.0),
                         ids=("zero", "negative-zero", "positive"))
def test_independent_pin_with_zero_and_positive_black(native, selector, black):
    source = np.asarray(
        [-0.25, -0.0, 0.0, 0.1234567, 0.2345678, 1.25,
         np.nextafter(np.float32(0.0), np.float32(1.0)), 0.5, 1.0],
        dtype=np.float32,
    )
    before = source.view(np.uint32).copy()
    controls = make_controls(direct=True, black=black)
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    expected = expected_pixels(
        source, controls, order, 1, selector, height=1, width=3, source_row_stride=9,
    )
    result = native.call(
        source, controls, order, height=1, width=3, source_row_stride=9,
        flags=1, selector=selector,
    )
    assert_success(result, expected, height=1, width=3, selector=selector,
                   black=black, flags=1)
    assert_bits(source, before.view(np.float32))
    assert np.all((result.rgb >= 0.0) & (result.rgb <= 1.0))
    assert np.all(result.rgb[:3].view(np.uint32) == 0)
    if black == 0.0:
        assert result.plan == (2, 0, 0, 0)  # pin exists without a black table
        assert result.rgb[5] == np.float32(1.0)
        if selector == 1:
            assert result.rgb[3].view(np.uint32) != source[3].view(np.uint32)
        else:
            assert_bits(result.rgb[3:5], source[3:5])


def test_source_binding_precision_and_independent_format_flag(native):
    codes = np.arange(32768, 65536, dtype=np.uint32)
    quotient = codes.astype(np.float32) / np.float32(65535.0)
    reciprocal = codes.astype(np.float32) * np.float32(1.0/65535.0)
    canaries = codes[quotient.view(np.uint32) != reciprocal.view(np.uint32)]
    assert canaries.size == 64
    for code in [32768, 65535, *map(int, canaries)]:
        status, output = native.bind([code, 4100, 1, 3])
        assert status == OK
        assert output[0] == 4.0 and output[1] == 2.0
        assert output[2] == float(np.float32(code) / np.float32(65535.0))
        assert output[3] == 4100.0 * float.fromhex('0x1.0001000100010p-16')
        np.testing.assert_array_equal(output[13:], [2, 0, 1, 1, 1])
        np.testing.assert_array_equal(output[4:13],
            [-0.25, 1.0, 0.125, 0.25, -0.125, 1.0, 1.0, 0.25, -0.5])
    b16 = np.arange(65535, dtype=np.float64)
    product = b16 * float.fromhex('0x1.0001000100010p-16')
    divided = b16 / 65535.0
    different = np.flatnonzero(product.view(np.uint64) != divided.view(np.uint64))
    assert different.size > 0
    for black in [0, 1028, 4100, 65534, *map(int, different[:20])]:
        for flag, image_type in itertools.product((0, 1), (3, 11)):
            status, output = native.bind([65535, black, flag, image_type])
            assert status == OK
            assert output[3] == product[black]
            np.testing.assert_array_equal(output[16:], [flag, 2 if image_type == 11 else 1])


def test_source_binding_rejects_mixed_or_invalid_preparation_atomically(native):
    for fields in ([32767, 0, 0, 3], [65536, 0, 0, 3], [65535, 65535, 0, 3],
                   [65535, 0, 2, 3], [65535, 0, 0, 0], [65535, 0, 0, 8]):
        status, output = native.bind(fields)
        assert status == INVALID_CONTROLS
        np.testing.assert_array_equal(output, np.full(18, -1234.5))
    for kwargs in (
        {'order': (0, 0, 1)}, {'order': (0, 1, 2)},
        {'gains': (4095, 4096, 16384)}, {'gain_float': (2.01, 1, 4)},
        {'sorted_matrix': np.ones(9)}, {'matrix': np.full(9, np.nan)},
    ):
        status, output = native.bind([65535, 4100, 0, 3], **kwargs)
        assert status == INVALID_CONTROLS
        np.testing.assert_array_equal(output, np.full(18, -1234.5))


def test_bound_controls_feed_composition_with_one_plane_permutation(native):
    source = np.asarray([0.1234, 0.3432, 0.5678, 0.7, 0.2, 0.4], dtype=np.float32)
    for order in itertools.permutations((0, 1, 2)):
        gains = np.empty(3, dtype=np.uint32)
        gains[list(order)] = [16384, 8192, 4096]
        for flag, image_type in itertools.product((0, 1), (3, 11)):
            status, bound = native.bind([58982, 4100, flag, image_type],
                                         order=order, gains=gains)
            assert status == OK
            controls = np.ascontiguousarray(bound[:13])
            permutation = np.ascontiguousarray(bound[13:16], dtype=np.uint32)
            selector = int(bound[17])
            expected = expected_pixels(source, controls, permutation, flag, selector,
                height=1, width=2, source_row_stride=6)
            actual = native.call(source, controls, permutation, flags=flag, selector=selector,
                height=1, width=2, source_row_stride=6)
            assert_success(actual, expected, height=1, width=2, selector=selector,
                           black=controls[3], flags=flag)


def test_formal_camera_and_exposure_preparations_bind_without_manual_repacking(native):
    # Exercise actual producer APIs as well as the independent array contract.
    import test_acr_camera_prepare_instructions as prepare_test
    import camera_prepare_instruction_oracle as prepare_oracle

    library_path = NATIVE / "build" / (
        "libimprint_renderer.0.1.0.dylib" if sys.platform == "darwin"
        else "libimprint_renderer.0.1.0.so")
    assert library_path.is_file()
    for name in ("acr_camera_prepare", "acr_exposure", "acr_exposure_api"):
        assert library_path.stat().st_mtime_ns >= (NATIVE / "src" / f"{name}.cpp").stat().st_mtime_ns
    formal = prepare_test.API(library_path)
    status, camera = formal.prepare(prepare_test.CANARY)
    assert status == OK
    expected_camera = prepare_oracle.prepare(prepare_test.CANARY)
    prepare_test._assert_fields(camera, expected_camera)
    exposure = formal.lib.im_native_acr_prepare_exposure
    exposure.argtypes = [DOUBLE_PTR, SIZE, DOUBLE_PTR, SIZE]
    exposure.restype = C.c_int
    for limit in (0.0, 0.5, math.nextafter(0.945, 0.0), 0.945,
                  math.nextafter(0.945, 1.0), 1.0):
        values = np.asarray([0, 0, 0, 1, limit, max(camera[18:21]), 1, 0, 0, 0],
                            dtype=np.float64)
        prepared = np.full(19, -1234.5, dtype=np.float64)
        assert exposure(_double_ptr(values), 10, _double_ptr(prepared), 19) == OK
        rational_value = math.floor(limit * 100.0 + 0.5) / 100.0
        expected_code = min(65535, max(32768, math.floor(rational_value * 65535.0 + 0.5)))
        assert prepared[18] == expected_code
        status, bound = native.bind([int(prepared[18]), 4100, 0, 3],
            matrix=camera[:9], sorted_matrix=camera[9:18],
            gains=camera[18:21], gain_float=camera[21:24],
            order=tuple(map(int, camera[27:30])))
        assert status == OK
        assert bound[2] == float(np.float32(expected_code) / np.float32(65535.0))
        np.testing.assert_array_equal(bound[4:13], camera[9:18])
        np.testing.assert_array_equal(bound[13:16], camera[27:30])
        assert bound[0] == camera[21 + int(camera[27])]
        assert bound[1] == camera[21 + int(camera[28])]
        # ShadowTint-derived compensation is not the retained source B16.
        assert np.all(bound[3] != camera[24:27])
