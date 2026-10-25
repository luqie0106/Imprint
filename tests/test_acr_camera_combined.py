from __future__ import annotations

import ctypes as C
import importlib.util
import itertools
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_camera_combined.cpp"
HEADER = SOURCE.with_suffix(".hpp")
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_camera_combined.cpp.o"
BRIDGE_SOURCE = NATIVE / "tools/acr_camera_combined_bridge.cpp"
FORMAL_LIBRARY = NATIVE / "build" / (
    "libimprint_renderer.0.1.0.dylib"
    if sys.platform == "darwin"
    else "libimprint_renderer.0.1.0.so"
)

ORACLE_SPEC = importlib.util.spec_from_file_location(
    "camera_combined_oracle", ROOT / "tests/acr_support/camera_combined_oracle.py"
)
oracle = importlib.util.module_from_spec(ORACLE_SPEC)
assert ORACLE_SPEC.loader is not None
ORACLE_SPEC.loader.exec_module(oracle)

FLOAT = C.c_float
FLOAT_PTR = C.POINTER(FLOAT)
DOUBLE = C.c_double
DOUBLE_PTR = C.POINTER(DOUBLE)
U32 = C.c_uint32
U32_PTR = C.POINTER(U32)
SIZE = C.c_size_t

CORE_STATUS = {
    "ok": 0,
    "invalid": 1,
    "dims": 2,
    "stride": 3,
    "short": 4,
    "alignment": 5,
    "overlap": 6,
    "overflow": 7,
    "resource": 8,
    "nonfinite_input": 9,
    "invalid_controls": 10,
    "nonfinite_intermediate": 11,
    "alloc": 12,
    "length": 13,
}


def float_ptr(array: np.ndarray) -> FLOAT_PTR:
    return array.ctypes.data_as(FLOAT_PTR)


def double_ptr(array: np.ndarray) -> DOUBLE_PTR:
    return array.ctypes.data_as(DOUBLE_PTR)


def order_ptr(array: np.ndarray) -> U32_PTR:
    return array.ctypes.data_as(U32_PTR)


def _formal_status(core_status: int) -> int:
    if core_status == CORE_STATUS["ok"]:
        return 0
    if core_status == CORE_STATUS["short"]:
        return 4
    if core_status in {
        CORE_STATUS["resource"],
        CORE_STATUS["alloc"],
        CORE_STATUS["length"],
    }:
        return 5
    return 1


class NativeCall:
    def __init__(self, library: C.CDLL, *, formal: bool):
        self.library = library
        self.formal = formal
        symbol = (
            "im_native_acr_camera_combined_prepared_f32"
            if formal
            else "acr_test_camera_combined"
        )
        self.function = getattr(library, symbol)
        self.function.argtypes = [
            FLOAT_PTR,
            SIZE,
            SIZE,
            SIZE,
            SIZE,
            DOUBLE_PTR,
            SIZE,
            U32_PTR,
            U32,
            FLOAT_PTR,
            SIZE,
            SIZE,
        ]
        self.function.restype = C.c_int

    def raw(
        self,
        source: FLOAT_PTR,
        src_count: int,
        src_row_stride: int,
        height: int,
        width: int,
        controls: DOUBLE_PTR,
        control_count: int,
        order3: U32_PTR,
        flags: int,
        destination: FLOAT_PTR,
        dst_count: int,
        dst_row_stride: int,
    ) -> int:
        return int(
            self.function(
                source,
                src_count,
                src_row_stride,
                height,
                width,
                controls,
                control_count,
                order3,
                flags,
                destination,
                dst_count,
                dst_row_stride,
            )
        )

    def call(
        self,
        source: np.ndarray,
        controls: np.ndarray,
        order3: np.ndarray,
        flags: int,
        destination: np.ndarray,
        *,
        height: int,
        width: int,
        src_row_stride: int,
        dst_row_stride: int,
        src_count: int | None = None,
        dst_count: int | None = None,
    ) -> int:
        return self.raw(
            float_ptr(source),
            source.size if src_count is None else src_count,
            src_row_stride,
            height,
            width,
            double_ptr(controls),
            controls.size,
            order_ptr(order3),
            flags,
            float_ptr(destination),
            destination.size if dst_count is None else dst_count,
            dst_row_stride,
        )

    def status(self, core_status: int) -> int:
        return _formal_status(core_status) if self.formal else core_status


@pytest.fixture(scope="module", params=("source", "official_object", "formal_library"))
def native(request, tmp_path_factory):
    if request.param == "formal_library":
        assert FORMAL_LIBRARY.is_file(), f"missing formal renderer library: {FORMAL_LIBRARY}"
        assert SOURCE.is_file(), f"missing combined camera source: {SOURCE}"
        assert HEADER.is_file(), f"missing combined camera header: {HEADER}"
        newest_input = max(SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns)
        assert FORMAL_LIBRARY.stat().st_mtime_ns >= newest_input, (
            "rebuild renderer: stale formal renderer library"
        )
        return NativeCall(C.CDLL(str(FORMAL_LIBRARY)), formal=True)

    assert SOURCE.is_file(), f"missing combined camera source: {SOURCE}"
    assert HEADER.is_file(), f"missing combined camera header: {HEADER}"
    assert BRIDGE_SOURCE.is_file(), f"missing combined camera C ABI bridge: {BRIDGE_SOURCE}"
    if request.param == "official_object":
        assert OBJECT.is_file(), f"rebuild renderer: missing official combined camera object: {OBJECT}"
        assert OBJECT.stat().st_mtime_ns >= max(
            SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns
        ), "rebuild renderer: stale official combined camera object"
        inputs = [OBJECT]
    else:
        inputs = [SOURCE]

    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the combined camera bridge"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_camera_combined_" + request.param) / (
        "libcamera_combined" + suffix
    )
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I",
        str(NATIVE / "src"),
        str(BRIDGE_SOURCE),
        *(str(path) for path in inputs),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    library = C.CDLL(str(output))
    return NativeCall(library, formal=False)


def assert_same_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32).view(np.uint32),
        np.asarray(expected, dtype=np.float32).view(np.uint32),
    )


def make_controls(rng, *, direct: bool, floor: float) -> np.ndarray:
    a = np.float64(1.0) if direct else np.float64(rng.uniform(1.05, 16.0))
    b = np.float64(rng.uniform(1.0, a))
    upper = np.float64(rng.uniform(32768.0 / 65535.0, 1.0))
    matrix = rng.uniform(-0.8, 1.1, size=9)
    return np.asarray([a, b, upper, floor, *matrix], dtype=np.float64)


def test_randomized_instruction_kernel_across_orders_widths_and_flags(native):
    rng = np.random.default_rng(2026102101)
    widths = (1, 3, 7, 8, 11, 12, 13, 17)
    plane_orders = tuple(itertools.permutations((0, 1, 2)))
    seen_widths = set()
    for direct in (True, False):
        for floor_index, floor in enumerate((0.0, -0.0, 0.07125, 0.43)):
            controls = make_controls(rng, direct=direct, floor=floor)
            for order_index, order_values in enumerate(plane_orders):
                order = np.asarray(order_values, dtype=np.uint32)
                width = widths[(order_index + floor_index + 2 * int(direct)) % len(widths)]
                seen_widths.add(width)
                height = 3
                src_stride = width * 3 + 5
                dst_stride = width * 3 + 7
                source = rng.uniform(-2.25, 3.5, size=height * src_stride).astype(np.float32)
                destination_template = rng.uniform(-4.0, 5.0, size=height * dst_stride).astype(np.float32)
                for flags in range(4):
                    expected = oracle.apply(
                        source,
                        controls,
                        order,
                        flags,
                        height=height,
                        width=width,
                        src_row_stride=src_stride,
                        dst_row_stride=dst_stride,
                        destination=destination_template.copy(),
                    )
                    actual = destination_template.copy()
                    status = native.call(
                        source,
                        controls,
                        order,
                        flags,
                        actual,
                        height=height,
                        width=width,
                        src_row_stride=src_stride,
                        dst_row_stride=dst_stride,
                    )
                    assert status == 0
                    assert_same_bits(actual, expected)
    assert seen_widths == set(widths)


def test_hard_coded_fma_and_signed_zero_probes(native):
    # With these binary32 operands, fmaf leaves a 2**-46 residual while a
    # separately rounded multiply plus add returns zero.
    a = np.float32(1.0 + 2.0**-23)
    b = np.float32(1.0 + 2.0**-23)
    c = np.float32(-(1.0 + 2.0**-22))
    fused = oracle._fma(float(a), float(b), float(c))
    split = np.float32(np.float32(a * b) + c)
    assert np.asarray([fused], dtype=np.float32).view(np.uint32)[0] == 0x28800000
    assert np.asarray([split], dtype=np.float32).view(np.uint32)[0] == 0
    # The output clamp's strict < 0 selection retains a pre-existing -0.
    assert np.asarray([oracle.clamp(-0.0)], dtype=np.float32).view(np.uint32)[0] == 0x80000000

    controls = np.asarray(
        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    source = np.full(3, np.float32(-0.0), dtype=np.float32)
    expected = oracle.apply(
        source,
        controls,
        order,
        0,
        height=1,
        width=1,
        src_row_stride=3,
    )
    actual = np.full(3, np.float32(77.0), dtype=np.float32)
    assert native.call(
        source,
        controls,
        order,
        0,
        actual,
        height=1,
        width=1,
        src_row_stride=3,
        dst_row_stride=3,
    ) == 0
    assert_same_bits(actual, expected)
    assert actual.view(np.uint32).tolist() == [0x80000000] * 3

    # A kernel-level matrix canary: column 1 is multiplied first, then the
    # column 0 product is fused into it. Splitting that FMA loses 2**-46.
    fma_controls = np.asarray(
        [
            1.0,
            1.0,
            1.0,
            0.0,
            1.0 + 2.0**-23,
            -(1.0 + 2.0**-22),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    fma_source = np.asarray([1.0 + 2.0**-23, 1.0, 0.0], dtype=np.float32)
    fma_actual = np.full(3, 19.0, dtype=np.float32)
    assert native.call(
        fma_source,
        fma_controls,
        order,
        1,
        fma_actual,
        height=1,
        width=1,
        src_row_stride=3,
        dst_row_stride=3,
    ) == 0
    assert int(fma_actual.view(np.uint32)[0]) == 0x28800000


def test_subnormal_negative_and_overrange_pixels(native):
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
    source = np.asarray(
        [tiny, -tiny, -0.0, -3.0, 2.75, 0.5, 4.0, -2.0, 1.5], dtype=np.float32
    )
    controls = np.asarray(
        [3.0, 1.25, 0.75, 0.0, 1.0, 0.1, -0.2, 0.0, 0.8, 0.1, 0.15, -0.1, 1.1],
        dtype=np.float64,
    )
    order = np.asarray([2, 0, 1], dtype=np.uint32)
    for flags in (0, 1, 2, 3):
        expected = oracle.apply(
            source,
            controls,
            order,
            flags,
            height=1,
            width=3,
            src_row_stride=9,
        )
        actual = np.full_like(source, -99.0)
        assert native.call(
            source,
            controls,
            order,
            flags,
            actual,
            height=1,
            width=3,
            src_row_stride=9,
            dst_row_stride=9,
        ) == 0
        assert_same_bits(actual, expected)


def test_smooth_threshold_binary32_neighbors(native):
    # A=4, B=2, U=1 gives q=1/4, q1=1/8, q2=1/16 in binary32.
    # Since input preparation scales the last two planes by 1/2 and 1/4,
    # source value 1/4 lands on all three thresholds.
    q = np.float32(0.25)
    below = np.nextafter(q, np.float32(-np.inf), dtype=np.float32)
    above = np.nextafter(q, np.float32(np.inf), dtype=np.float32)
    source = np.asarray(
        [below, q, above, q, above, below, above, below, q], dtype=np.float32
    )
    controls = np.asarray(
        [4.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    expected = oracle.apply(
        source,
        controls,
        order,
        1,
        height=1,
        width=3,
        src_row_stride=9,
    )
    actual = np.full(9, -1.0, dtype=np.float32)
    assert native.call(
        source,
        controls,
        order,
        1,
        actual,
        height=1,
        width=3,
        src_row_stride=9,
        dst_row_stride=9,
    ) == 0
    assert_same_bits(actual, expected)


def test_simple_floor_subtract_and_unconditional_A_scale_are_distinct_canaries(native):
    # The captured nonzero-b path normalizes each input by 1/(1-b), scales
    # matrix entries with FNMUL, then independently FADDs bout. A common
    # simplified `subtract b -> b=0 combined camera -> add b` path and an
    # unconditional post-camera A scale both produce different pixels here;
    # neither is used as the oracle.
    controls = np.asarray(
        [2.0, 1.5, 0.8, 0.25, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    source = np.asarray([0.12, 0.18, 0.15], dtype=np.float32)
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    expected = oracle.apply(
        source,
        controls,
        order,
        1,
        height=1,
        width=1,
        src_row_stride=3,
    )
    assert expected.view(np.uint32).tolist() == [0x3DF5C28E, 0x3E4A3D71, 0x3E4CCCCD]

    floor0_controls = controls.copy()
    floor0_controls[3] = 0.0
    shifted_source = np.asarray(source - np.float32(oracle.f32(controls[3])), dtype=np.float32)
    floor0_result = oracle.apply(
        shifted_source,
        floor0_controls,
        order,
        1,
        height=1,
        width=1,
        src_row_stride=3,
    )
    simple_floor_subtract = np.asarray(
        floor0_result + np.float32(oracle.f32(controls[3])), dtype=np.float32
    )
    unconditional_a_scale = np.asarray(
        expected * np.float32(oracle.f32(controls[0])), dtype=np.float32
    )
    assert simple_floor_subtract.view(np.uint32).tolist() == [0x3DF5C290, 0x3E4A3D71, 0x3E4CCCCD]
    assert unconditional_a_scale.view(np.uint32).tolist() == [0x3E75C28E, 0x3ECA3D71, 0x3ECCCCCD]
    assert not np.array_equal(expected.view(np.uint32), simple_floor_subtract.view(np.uint32))
    assert not np.array_equal(expected.view(np.uint32), unconditional_a_scale.view(np.uint32))

    actual = np.full(3, -9.0, dtype=np.float32)
    assert native.call(
        source,
        controls,
        order,
        1,
        actual,
        height=1,
        width=1,
        src_row_stride=3,
        dst_row_stride=3,
    ) == 0
    assert_same_bits(actual, expected)
    assert_same_bits(actual, np.asarray([0x3DF5C28E, 0x3E4A3D71, 0x3E4CCCCD], dtype=np.uint32).view(np.float32))


def test_reference_can_select_roi_without_repacking_inputs():
    source = np.arange(2 * 17, dtype=np.float32) / np.float32(7.0)
    controls = np.asarray(
        [2.0, 1.4, 0.82, 0.125, 1.0, 0.1, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.8],
        dtype=np.float64,
    )
    order = np.asarray([1, 2, 0], dtype=np.uint32)
    seed = np.full(2 * 19, -11.0, dtype=np.float32)
    full = oracle.apply(
        source,
        controls,
        order,
        1,
        height=2,
        width=4,
        src_row_stride=17,
        dst_row_stride=19,
        destination=seed.copy(),
    )
    roi = oracle.apply(
        source,
        controls,
        order,
        1,
        height=2,
        width=4,
        src_row_stride=17,
        dst_row_stride=19,
        destination=seed.copy(),
        pixels=((1, 3), (0, 1)),
    )
    for row, col in ((1, 3), (0, 1)):
        base = row * 19 + col * 3
        assert_same_bits(roi[base : base + 3], full[base : base + 3])
    untouched = {(r, c) for r in range(2) for c in range(4)} - {(1, 3), (0, 1)}
    for row, col in untouched:
        base = row * 19 + col * 3
        assert_same_bits(roi[base : base + 3], seed[base : base + 3])
    for row in range(2):
        assert_same_bits(roi[row * 19 + 12 : row * 19 + 19], seed[row * 19 + 12 : row * 19 + 19])


def test_padding_and_exact_inplace_preserve_geometry(native):
    rng = np.random.default_rng(2026102102)
    height, width = 3, 7
    stride = width * 3 + 4
    order = np.asarray([2, 0, 1], dtype=np.uint32)
    controls = make_controls(rng, direct=False, floor=0.22)
    source = rng.uniform(-1.5, 2.0, height * stride).astype(np.float32)
    source.reshape(height, stride)[:, width * 3 :] = np.float32(np.nan)
    before = source.copy()
    expected = oracle.apply(
        source,
        controls,
        order,
        3,
        height=height,
        width=width,
        src_row_stride=stride,
        destination=source.copy(),
    )
    actual = source.copy()
    assert native.call(
        actual,
        controls,
        order,
        3,
        actual,
        height=height,
        width=width,
        src_row_stride=stride,
        dst_row_stride=stride,
    ) == 0
    assert_same_bits(actual, expected)
    assert_same_bits(actual.reshape(height, stride)[:, width * 3 :], before.reshape(height, stride)[:, width * 3 :])


def test_exact_inplace_direct_zero_floor_all_six_orders(native):
    rng = np.random.default_rng(2026102103)
    controls = make_controls(rng, direct=True, floor=0.0)
    for order_values in itertools.permutations((0, 1, 2)):
        order = np.asarray(order_values, dtype=np.uint32)
        source = rng.uniform(-0.4, 1.6, size=3 * 5).astype(np.float32)
        expected = oracle.apply(
            source,
            controls,
            order,
            0,
            height=1,
            width=5,
            src_row_stride=15,
            destination=source.copy(),
        )
        actual = source.copy()
        assert native.call(
            actual,
            controls,
            order,
            0,
            actual,
            height=1,
            width=5,
            src_row_stride=15,
            dst_row_stride=15,
        ) == 0
        assert_same_bits(actual, expected)


def _assert_error_atomic(
    native,
    *,
    source: FLOAT_PTR,
    src_count: int,
    src_row_stride: int,
    height: int,
    width: int,
    controls: DOUBLE_PTR,
    control_count: int,
    order: U32_PTR,
    flags: int,
    destination: np.ndarray,
    dst_count: int,
    dst_row_stride: int,
    core_error: str,
    before: np.ndarray | None = None,
) -> None:
    if before is None:
        before = destination.copy()
    status = native.raw(
        source,
        src_count,
        src_row_stride,
        height,
        width,
        controls,
        control_count,
        order,
        flags,
        float_ptr(destination),
        dst_count,
        dst_row_stride,
    )
    assert status == native.status(CORE_STATUS[core_error]), (core_error, status)
    assert_same_bits(destination, before)


def test_validation_failures_are_atomic(native):
    controls = np.asarray(
        [2.0, 1.5, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    source = np.full(2 * 12, 0.25, dtype=np.float32)
    destination = np.full(2 * 12, -555.0, dtype=np.float32)
    base_args = dict(
        controls=double_ptr(controls),
        control_count=13,
        order=order_ptr(order),
        flags=0,
        destination=destination,
        dst_count=destination.size,
        dst_row_stride=12,
    )

    _assert_error_atomic(
        native,
        source=float_ptr(source), src_count=24, src_row_stride=12, height=0, width=4,
        core_error="dims", **base_args
    )
    _assert_error_atomic(
        native,
        source=float_ptr(source), src_count=24, src_row_stride=12, height=2, width=0,
        core_error="dims", **base_args
    )
    _assert_error_atomic(
        native,
        source=float_ptr(source), src_count=24, src_row_stride=11, height=2, width=4,
        core_error="stride", **base_args
    )
    _assert_error_atomic(
        native,
        source=float_ptr(source), src_count=23, src_row_stride=12, height=2, width=4,
        core_error="short", **base_args
    )
    _assert_error_atomic(
        native,
        source=float_ptr(source), src_count=24, src_row_stride=12, height=2, width=4,
        dst_count=23,
        core_error="short", **{key: value for key, value in base_args.items() if key != "dst_count"}
    )
    _assert_error_atomic(
        native,
        source=float_ptr(source), src_count=24, src_row_stride=12, height=2, width=4,
        dst_row_stride=11,
        core_error="stride", **{key: value for key, value in base_args.items() if key != "dst_row_stride"}
    )
    _assert_error_atomic(
        native,
        source=float_ptr(source), src_count=24, src_row_stride=12, height=2, width=4,
        control_count=12,
        core_error="invalid_controls", **{key: value for key, value in base_args.items() if key != "control_count"}
    )

    # Width 1 and height 2 make (height-1)*SIZE_MAX + active-width overflow.
    size_max = C.c_size_t(-1).value
    _assert_error_atomic(
        native,
        source=float_ptr(source), src_count=size_max, src_row_stride=size_max,
        height=2, width=1, controls=double_ptr(controls), control_count=13,
        order=order_ptr(order), flags=0, destination=destination,
        dst_count=size_max, dst_row_stride=size_max, core_error="overflow",
    )

    # The configured active-pixel ceiling must reject this before touching the
    # deliberately small buffers.  Lengths describe the virtual full spans.
    huge_height = 128 * 1024 * 1024 + 1
    virtual_count = huge_height * 3
    _assert_error_atomic(
        native,
        source=float_ptr(source), src_count=virtual_count, src_row_stride=3,
        height=huge_height, width=1, controls=double_ptr(controls), control_count=13,
        order=order_ptr(order), flags=0, destination=destination,
        dst_count=virtual_count, dst_row_stride=3, core_error="resource",
    )

    for bad_value in (np.nan, np.inf, -np.inf):
        bad_source = source.copy()
        bad_source[1] = np.float32(bad_value)
        _assert_error_atomic(
            native,
            source=float_ptr(bad_source), src_count=24, src_row_stride=12,
            height=2, width=4, controls=double_ptr(controls), control_count=13,
            order=order_ptr(order), flags=0, destination=destination,
            dst_count=24, dst_row_stride=12, core_error="nonfinite_input",
        )

    invalid_control_cases = []
    invalid_control_values = (
        (0, 0.0),
        (1, 0.9),
        (1, 3.0),  # B > A
        (2, float(np.nextafter(np.float32(32768.0 / 65535.0), np.float32(0.0)))),
        (2, float(np.nextafter(np.float32(1.0), np.float32(2.0)))),
        (3, -1e-9),
        (3, 1.0),
        (4, np.nan),
    )
    for index, value in invalid_control_values:
        candidate = controls.copy()
        candidate[index] = value
        invalid_control_cases.append(candidate)
    for invalid_controls in invalid_control_cases:
        _assert_error_atomic(
            native,
            source=float_ptr(source), src_count=24, src_row_stride=12,
            height=2, width=4, controls=double_ptr(invalid_controls), control_count=13,
            order=order_ptr(order), flags=0, destination=destination,
            dst_count=24, dst_row_stride=12, core_error="invalid_controls",
        )

    # Finite source and controls can still overflow an intermediate matrix
    # product; it must fail before committing any destination pixel.
    max_float = np.finfo(np.float32).max
    huge_source = np.full(3, max_float, dtype=np.float32)
    huge_matrix = controls.copy()
    huge_matrix[4:13] = max_float
    small_destination = np.full(3, -555.0, dtype=np.float32)
    _assert_error_atomic(
        native,
        source=float_ptr(huge_source), src_count=3, src_row_stride=3,
        height=1, width=1, controls=double_ptr(huge_matrix), control_count=13,
        order=order_ptr(order), flags=1, destination=small_destination,
        dst_count=3, dst_row_stride=3, core_error="nonfinite_intermediate",
    )


def test_null_and_unaligned_arguments_are_rejected_atomically(native):
    controls = np.asarray(
        [1.0, 1.0, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    source = np.full(3, 0.25, dtype=np.float32)
    destination = np.full(3, -7.0, dtype=np.float32)
    before = destination.copy()
    null_float = C.cast(C.c_void_p(0), FLOAT_PTR)
    assert native.raw(
        null_float, 3, 3, 1, 1, double_ptr(controls), 13, order_ptr(order), 0,
        float_ptr(destination), 3, 3,
    ) == native.status(CORE_STATUS["invalid"])
    assert_same_bits(destination, before)

    backing = (C.c_uint8 * 64)()
    unaligned_source = C.cast(C.c_void_p(C.addressof(backing) + 1), FLOAT_PTR)
    _assert_error_atomic(
        native,
        source=unaligned_source, src_count=3, src_row_stride=3, height=1, width=1,
        controls=double_ptr(controls), control_count=13, order=order_ptr(order),
        flags=0, destination=destination, dst_count=3, dst_row_stride=3,
        core_error="alignment", before=before,
    )


def test_partial_overlap_and_metadata_aliases_are_rejected_atomically(native):
    controls = np.asarray(
        [1.0, 1.0, 0.75, 0.0, 1.0, 0.1, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.8],
        dtype=np.float64,
    )
    order = np.asarray([0, 1, 2], dtype=np.uint32)
    storage = np.linspace(-1.0, 2.0, 20, dtype=np.float32)
    before = storage.copy()
    status = native.raw(
        float_ptr(storage), 18, 9, 2, 3, double_ptr(controls), 13, order_ptr(order), 0,
        float_ptr(storage[1:]), 18, 9,
    )
    assert status == native.status(CORE_STATUS["overlap"])
    assert_same_bits(storage, before)

    # Controls alias the source span.  The pointer is naturally aligned and
    # the source count covers the metadata bytes, so overlap is the failure.
    source_alias = controls.view(np.float32)
    destination = np.full(24, -44.0, dtype=np.float32)
    before_destination = destination.copy()
    status = native.raw(
        float_ptr(source_alias), source_alias.size, 12, 2, 4,
        double_ptr(controls), 13, order_ptr(order), 0,
        float_ptr(destination), destination.size, 12,
    )
    assert status == native.status(CORE_STATUS["overlap"])
    assert_same_bits(destination, before_destination)

    # Controls may not overlap the output span either.
    output_alias = controls.view(np.float32)
    status = native.raw(
        float_ptr(source_alias), source_alias.size, 12, 2, 4,
        double_ptr(controls), 13, order_ptr(order), 0,
        float_ptr(output_alias), output_alias.size, 12,
    )
    assert status == native.status(CORE_STATUS["overlap"])
    assert_same_bits(controls, np.asarray(
        [1.0, 1.0, 0.75, 0.0, 1.0, 0.1, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.8],
        dtype=np.float64,
    ))

    # The order metadata cannot point into the input or output span.
    order_source = np.zeros(24, dtype=np.float32)
    order_source.view(np.uint32)[:3] = np.asarray([0, 1, 2], dtype=np.uint32)
    order_alias = order_source.view(np.uint32)
    separate_destination = np.full(24, -31.0, dtype=np.float32)
    before_separate = separate_destination.copy()
    status = native.raw(
        float_ptr(order_source), order_source.size, 12, 2, 4,
        double_ptr(controls), 13, order_ptr(order_alias), 0,
        float_ptr(separate_destination), separate_destination.size, 12,
    )
    assert status == native.status(CORE_STATUS["overlap"])
    assert_same_bits(separate_destination, before_separate)

    order_output = order.view(np.float32)
    source_for_order_output = np.full(3, 0.5, dtype=np.float32)
    before_order_output = order_output.copy()
    status = native.raw(
        float_ptr(source_for_order_output), 3, 3, 1, 1,
        double_ptr(controls), 13, order_ptr(order), 0,
        float_ptr(order_output), 3, 3,
    )
    assert status == native.status(CORE_STATUS["overlap"])
    assert_same_bits(order_output, before_order_output)

    # The order array aliases the control array.  Metadata spans may not alias
    # each other even though the first three words are not a valid permutation.
    metadata = controls.copy()
    order_alias = C.cast(C.c_void_p(metadata.ctypes.data), U32_PTR)
    status = native.raw(
        float_ptr(source_alias), source_alias.size, 12, 2, 4,
        double_ptr(metadata), 13, order_alias, 0,
        float_ptr(destination), destination.size, 12,
    )
    assert status == native.status(CORE_STATUS["overlap"])
    assert_same_bits(destination, before_destination)


def test_prepared_source_floor_is_rounded_to_binary32(native):
    # The control is a caller-prepared binary64 value; 013466b4 consumes its
    # FCVT binary32 value.  Keep the original double in the ABI buffer.
    floor64 = 0.30000000000000004
    assert int(np.asarray([oracle.f32(floor64)], dtype=np.float32).view(np.uint32)[0]) == 0x3E99999A
    controls = np.asarray(
        [1.0, 1.0, 0.875, floor64, 0.85, -0.12, 0.05, 0.11, 0.94, -0.08, 0.2, 0.07, 0.9],
        dtype=np.float64,
    )
    source = np.asarray([-0.25, 0.72, 1.8], dtype=np.float32)
    order = np.asarray([2, 1, 0], dtype=np.uint32)
    expected = oracle.apply(
        source,
        controls,
        order,
        1,
        height=1,
        width=1,
        src_row_stride=3,
    )
    actual = np.full(3, -17.0, dtype=np.float32)
    assert native.call(
        source,
        controls,
        order,
        1,
        actual,
        height=1,
        width=1,
        src_row_stride=3,
        dst_row_stride=3,
    ) == 0
    assert_same_bits(actual, expected)
