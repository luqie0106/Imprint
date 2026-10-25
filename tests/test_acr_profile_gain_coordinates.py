"""Independent oracle checks for the PGTM temporary coordinate planes."""

from __future__ import annotations

import ctypes as C
import mmap
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SUPPORT = ROOT / "tests" / "acr_support"
COORD_SOURCE = NATIVE / "src" / "acr_profile_gain_coordinates.cpp"
COORD_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_coordinates_bridge.cpp"
MAP_SOURCE = NATIVE / "src" / "acr_profile_gain_map.cpp"
GAIN_SOURCE = NATIVE / "src" / "acr_profile_gain.cpp"
APPLY_SOURCE = NATIVE / "src" / "acr_profile_gain_apply.cpp"
RESAMPLE_MATRIX_SOURCE = NATIVE / "src" / "acr_resample_matrix.cpp"
MAP_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_map_bridge.cpp"
GAIN_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_bridge.cpp"
APPLY_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_apply_bridge.cpp"

OBJECT_SUFFIX = ".obj" if sys.platform == "win32" else ".o"
OFFICIAL_COORD_OBJECT = NATIVE / (
    f"build/CMakeFiles/imprint_renderer.dir/src/acr_profile_gain_coordinates.cpp{OBJECT_SUFFIX}"
)
OFFICIAL_CHAIN_OBJECTS = (
    NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/acr_resample_matrix.cpp{OBJECT_SUFFIX}",
    NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/acr_profile_gain_map.cpp{OBJECT_SUFFIX}",
    NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/acr_profile_gain.cpp{OBJECT_SUFFIX}",
    NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/acr_profile_gain_apply.cpp{OBJECT_SUFFIX}",
)

sys.path.insert(0, str(SUPPORT))
import profile_gain_coordinates_oracle as coordinate_oracle  # noqa: E402
import profile_gain_apply_oracle as apply_oracle  # noqa: E402
import profile_gain_map_oracle as map_oracle  # noqa: E402
import profile_gain_oracle as gain_oracle  # noqa: E402
import resample_matrix_oracle as matrix_oracle  # noqa: E402


F32_PTR = C.POINTER(C.c_float)
I32_PTR = C.POINTER(C.c_int32)
U32_PTR = C.POINTER(C.c_uint32)
F64_PTR = C.POINTER(C.c_double)
SIZE = C.c_size_t
SIZE_PTR = C.POINTER(SIZE)
F32_PTR_PTR = C.POINTER(F32_PTR)
SIZE_MAX = SIZE(-1).value

OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
INVALID_STRIDE = 3
SHORT_BUFFER = 4
ALIGNMENT = 5
OVERLAP = 6
ARITHMETIC_OVERFLOW = 7
RESOURCE_LIMIT = 8
INVALID_PREPARED = 9
NON_FINITE_INTERMEDIATE = 11
UNSUPPORTED_TRANSFORM = 13

SENTINEL = np.float32(-731.625)
SENTINEL_BITS = np.asarray([SENTINEL], dtype=np.float32).view(np.uint32)[0]


def _typed_pointer(value, pointer_type):
    if value is None:
        return C.cast(C.c_void_p(0), pointer_type)
    if isinstance(value, int):
        return C.cast(C.c_void_p(value), pointer_type)
    if isinstance(value, C.c_void_p):
        return C.cast(value, pointer_type)
    if isinstance(value, np.ndarray):
        return C.cast(C.c_void_p(value.ctypes.data), pointer_type)
    return C.cast(value, pointer_type)


def _f32_ptr(value):
    return _typed_pointer(value, F32_PTR)


def _i32_ptr(value):
    return _typed_pointer(value, I32_PTR)


def _u32_ptr(value):
    return _typed_pointer(value, U32_PTR)


def _compile_bridge(tmp_path: Path, kind: str) -> Path:
    # Missing coordinate implementation/bridge is a hard failure for both
    # source and official-object validation; do not turn it into a skip.
    assert COORD_SOURCE.is_file(), f"missing coordinate implementation: {COORD_SOURCE}"
    assert COORD_BRIDGE_SOURCE.is_file(), f"missing coordinate bridge: {COORD_BRIDGE_SOURCE}"

    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required to compile the coordinate bridge"
    bridges = (COORD_BRIDGE_SOURCE, MAP_BRIDGE_SOURCE, GAIN_BRIDGE_SOURCE, APPLY_BRIDGE_SOURCE)
    implementations = (COORD_SOURCE, RESAMPLE_MATRIX_SOURCE, MAP_SOURCE, GAIN_SOURCE, APPLY_SOURCE)
    for path in (*bridges, *implementations):
        assert path.is_file(), f"missing local PGTM chain source: {path}"

    if kind == "source":
        linked_implementations = implementations
    else:
        assert kind == "official_object"
        if not OFFICIAL_COORD_OBJECT.is_file():
            pytest.skip(f"official coordinate object is unavailable: {OFFICIAL_COORD_OBJECT}")
        missing = [path for path in OFFICIAL_CHAIN_OBJECTS if not path.is_file()]
        if missing:
            pytest.skip("official PGTM chain objects are unavailable: " + ", ".join(map(str, missing)))
        official_pairs = ((OFFICIAL_COORD_OBJECT, COORD_SOURCE),) + tuple(
            zip(OFFICIAL_CHAIN_OBJECTS, implementations[1:])
        )
        stale = [obj for obj, source in official_pairs if obj.stat().st_mtime_ns < source.stat().st_mtime_ns]
        if stale:
            pytest.skip("official PGTM objects are stale; rebuild imprint_renderer: " + ", ".join(p.name for p in stale))
        linked_implementations = tuple(obj for obj, _ in official_pairs)

    if sys.platform == "darwin":
        suffix, shared_flags = ".dylib", ["-dynamiclib"]
    elif sys.platform == "win32":
        suffix, shared_flags = ".dll", ["-shared"]
    else:
        suffix, shared_flags = ".so", ["-shared", "-fPIC"]

    output = tmp_path / f"libprofile_gain_coordinates_{kind}{suffix}"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        *shared_flags,
        "-I",
        str(NATIVE / "src"),
        "-I",
        str(NATIVE / "include"),
        *(str(path) for path in bridges),
        *(str(path) for path in linked_implementations),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))

        self.coordinates_fn = self.library.acr_profile_gain_coordinates_bridge
        self.coordinates_fn.argtypes = [
            F32_PTR, SIZE, I32_PTR, SIZE, C.c_uint32,
            F32_PTR, SIZE, F32_PTR, SIZE, SIZE,
        ]
        self.coordinates_fn.restype = C.c_int

        self.prepare_coordinates_fn = self.library.acr_profile_gain_coordinate_prepare_bridge
        self.prepare_coordinates_fn.argtypes = [
            I32_PTR, SIZE, I32_PTR, SIZE, F64_PTR, SIZE,
            C.c_uint32, F32_PTR, SIZE,
        ]
        self.prepare_coordinates_fn.restype = C.c_int

        self.map_fn = self.library.acr_profile_gain_map_bridge
        self.map_fn.argtypes = [
            C.POINTER(C.c_uint8), SIZE, C.c_uint32, C.c_uint32, I32_PTR,
            C.c_double, F64_PTR, SIZE, F32_PTR, SIZE, U32_PTR, SIZE,
            F32_PTR, SIZE, SIZE_PTR,
        ]
        self.map_fn.restype = C.c_int

        self.gain_fn = self.library.acr_profile_gain_bridge
        self.gain_fn.argtypes = [
            F32_PTR, SIZE, U32_PTR, SIZE, F32_PTR, SIZE,
            F32_PTR_PTR, SIZE_PTR, F32_PTR, SIZE,
            SIZE, SIZE, SIZE, SIZE,
        ]
        self.gain_fn.restype = C.c_int

        self.apply_fn = self.library.acr_profile_gain_apply_bridge
        self.apply_fn.argtypes = [
            F32_PTR_PTR, SIZE_PTR, F32_PTR_PTR, SIZE_PTR,
            SIZE, SIZE, SIZE, SIZE, C.c_double, C.c_uint32, C.c_uint32,
        ]
        self.apply_fn.restype = C.c_int

    def coordinates_raw(
        self,
        coefficients,
        rect,
        out_v,
        out_u,
        *,
        coordinate_task=0,
        coefficient_count=None,
        rect_count=None,
        v_count=None,
        u_count=None,
        row_stride,
    ):
        return int(self.coordinates_fn(
            _f32_ptr(coefficients),
            (len(coefficients) if coefficient_count is None and coefficients is not None else
             (coefficient_count if coefficient_count is not None else 0)),
            _i32_ptr(rect),
            (len(rect) if rect_count is None and rect is not None else
             (rect_count if rect_count is not None else 0)),
            coordinate_task,
            _f32_ptr(out_v),
            (len(out_v) if v_count is None and out_v is not None else
             (v_count if v_count is not None else 0)),
            _f32_ptr(out_u),
            (len(out_u) if u_count is None and out_u is not None else
             (u_count if u_count is not None else 0)),
            row_stride,
        ))

    def prepare_coordinates_raw(
        self,
        zero_rect,
        state_rect,
        matrix50,
        coefficients,
        *,
        coordinate_task=0,
        zero_count=None,
        state_count=None,
        matrix_count=None,
        coefficient_count=None,
    ):
        return int(self.prepare_coordinates_fn(
            _i32_ptr(zero_rect),
            (len(zero_rect) if zero_count is None and zero_rect is not None else
             (zero_count if zero_count is not None else 0)),
            _i32_ptr(state_rect),
            (len(state_rect) if state_count is None and state_rect is not None else
             (state_count if state_count is not None else 0)),
            _typed_pointer(matrix50, F64_PTR),
            (len(matrix50) if matrix_count is None and matrix50 is not None else
             (matrix_count if matrix_count is not None else 0)),
            coordinate_task,
            _f32_ptr(coefficients),
            (len(coefficients) if coefficient_count is None and coefficients is not None else
             (coefficient_count if coefficient_count is not None else 0)),
        ))

    def prepare(self, payload: bytes, tag: int, byteorder: int, rect, scalar: float):
        data = C.create_string_buffer(payload, len(payload))
        rect_array = (C.c_int32 * len(rect))(*rect)
        meta = np.full(23, np.float64(-991.125), dtype=np.float64)
        fields = np.full(19, np.float32(-731.625), dtype=np.float32)
        geometry = np.full(8, np.uint32(0xA5A5A5A5), dtype=np.uint32)
        table = np.full(128, np.float32(-317.25), dtype=np.float32)
        written = SIZE(0x12345678)
        status = int(self.map_fn(
            C.cast(data, C.POINTER(C.c_uint8)), len(payload), tag, byteorder,
            rect_array, scalar, meta.ctypes.data_as(F64_PTR), meta.size,
            fields.ctypes.data_as(F32_PTR), fields.size,
            geometry.ctypes.data_as(U32_PTR), geometry.size,
            table.ctypes.data_as(F32_PTR), table.size, C.pointer(written),
        ))
        return status, meta, fields, geometry, table, int(written.value), data, rect_array

    def evaluate_gain(self, fields, geometry, table, inputs, output, *, outer, inner, value_stride, coord_stride):
        input_ptrs = (F32_PTR * 5)(*(_f32_ptr(item) for item in inputs))
        input_counts = (SIZE * 5)(*(item.size for item in inputs))
        return int(self.gain_fn(
            _f32_ptr(fields), 16, _u32_ptr(geometry), 5, _f32_ptr(table), table.size,
            input_ptrs, input_counts, _f32_ptr(output), output.size,
            outer, inner, value_stride, coord_stride,
        ))

    def apply(self, inputs, outputs, *, outer, inner, value_stride, gain_stride, pivot, mode, clip):
        input_ptrs = (F32_PTR * 4)(*(_f32_ptr(item) for item in inputs))
        input_counts = (SIZE * 4)(*(item.size for item in inputs))
        output_ptrs = (F32_PTR * 3)(*(_f32_ptr(item) for item in outputs))
        output_counts = (SIZE * 3)(*(item.size for item in outputs))
        return int(self.apply_fn(
            input_ptrs, input_counts, output_ptrs, output_counts,
            outer, inner, value_stride, gain_stride, pivot, mode, clip,
        ))


@pytest.fixture(scope="module", params=("source", "official_object"), ids=("source", "official_object"))
def bridge(request, tmp_path_factory):
    path = _compile_bridge(tmp_path_factory.mktemp("acr_profile_gain_coordinates_" + request.param), request.param)
    return Bridge(path)


def _bit_pattern(word: int) -> np.float32:
    return np.asarray([word], dtype=np.uint32).view(np.float32)[0]


def _bits_equal(actual: np.ndarray, expected: np.ndarray, message: str):
    try:
        coordinate_oracle.assert_float32_or_nan(actual, expected)
    except AssertionError as error:
        raise AssertionError(f"{message}: {error}") from error


def _valid_call(bridge, coefficients=None, rect=None, *, row_stride=None, v_count=None, u_count=None):
    coefficients = np.asarray([0.75, -0.125, 0.375, -0.5, 0.25, -0.25], dtype=np.float32) if coefficients is None else coefficients
    rect = np.asarray([-3, 2, 1, 7], dtype=np.int32) if rect is None else rect
    height, width = int(rect[2]) - int(rect[0]), int(rect[3]) - int(rect[1])
    stride = width + 3 if row_stride is None else row_stride
    minimum = (height - 1) * stride + width
    v = np.full(minimum if v_count is None else v_count, SENTINEL, dtype=np.float32)
    u = np.full(minimum if u_count is None else u_count, SENTINEL, dtype=np.float32)
    status = bridge.coordinates_raw(coefficients, rect, v, u, row_stride=stride)
    expected_v, expected_u = coordinate_oracle.reference_planes(
        coefficients, rect, stride, v_count=v.size, u_count=u.size
    )
    assert status == OK
    _bits_equal(v, expected_v, "v plane")
    _bits_equal(u, expected_u, "u plane")
    return coefficients, rect, v, u


def _prepare_once(bridge, zero_rect, state_rect, matrix50, *,
                  zero_count=None, state_count=None, matrix_count=None,
                  coefficient_count=10, coordinate_task=0):
    coefficients = np.full(coefficient_count, SENTINEL, dtype=np.float32)
    status = bridge.prepare_coordinates_raw(
        zero_rect, state_rect, matrix50, coefficients,
        zero_count=zero_count, state_count=state_count,
        matrix_count=matrix_count, coefficient_count=coefficient_count,
        coordinate_task=coordinate_task,
    )
    expected_status, expected = coordinate_oracle.reference_prepare(
        zero_rect, state_rect, matrix50
    )
    assert status == expected_status
    if status == OK:
        assert expected is not None
        _bits_equal(coefficients[:6], np.asarray(expected, dtype=np.float32), "prepared coefficients")
        assert np.all(coordinate_oracle.bits32(coefficients[6:]) == SENTINEL_BITS)
    else:
        assert np.all(coordinate_oracle.bits32(coefficients) == SENTINEL_BITS)
    return coefficients


def test_coordinate_constructor_identity_origins_and_int32_min_wrap(bridge):
    identity_matrix = np.eye(3, dtype=np.float64).reshape(-1)
    same_rect = np.asarray([0, 0, 4, 8], dtype=np.int32)
    coefficients = _prepare_once(bridge, same_rect, same_rect, identity_matrix)
    np.testing.assert_array_equal(
        coordinate_oracle.bits32(coefficients[:6]),
        np.asarray([0x3F800000, 0, 0, 0, 0x3F800000, 0], dtype=np.uint32),
    )

    zero_rect = np.asarray([10, -20, 16, -8], dtype=np.int32)
    state_rect = np.asarray([-4, 7, -2, 10], dtype=np.int32)
    coefficients = _prepare_once(bridge, zero_rect, state_rect, identity_matrix)
    np.testing.assert_array_equal(
        coordinate_oracle.bits32(coefficients[:6]),
        np.asarray([0x40400000, 0, 0x41B00000, 0, 0x40800000, 0xC2400000], dtype=np.uint32),
    )

    zero_rect = np.asarray([0, 0, 2, 2], dtype=np.int32)
    state_rect = np.asarray([-(1 << 31), -(1 << 31), -(1 << 31) + 1, -(1 << 31) + 2], dtype=np.int32)
    coefficients = _prepare_once(bridge, zero_rect, state_rect, identity_matrix)
    assert coordinate_oracle._wrapped_neg32(-(1 << 31)) == -(1 << 31)
    expected_status, expected = coordinate_oracle.reference_prepare(
        zero_rect, state_rect, identity_matrix
    )
    assert expected_status == OK and expected is not None
    _bits_equal(coefficients[:6], np.asarray(expected, dtype=np.float32), "INT32_MIN wrap")


def test_coordinate_constructor_128_independent_geometry_states(bridge):
    rng = np.random.default_rng(0xC04D7A)
    seen = set()

    def origin(state_index, axis):
        if state_index % 8 == 0:
            return int(rng.integers(-(1 << 31), -(1 << 31) + 64))
        if state_index % 8 == 1:
            return int(rng.integers((1 << 31) - 2048, (1 << 31) - 64))
        if state_index % 8 == 2:
            return (-(1 << 31)) if axis == 0 else int(rng.integers(-100, 100))
        return int(rng.integers(-100000, 100000))

    for index in range(128):
        rh, rw, sh, sw = (int(x) for x in rng.integers(1, 1000, size=4))
        ztop, zleft = origin(index, 0), origin(index, 1)
        stop, sleft = origin(index + 3, 0), origin(index + 5, 1)
        ztop = min(ztop, (1 << 31) - 1 - rh)
        zleft = min(zleft, (1 << 31) - 1 - rw)
        stop = min(stop, (1 << 31) - 1 - sh)
        sleft = min(sleft, (1 << 31) - 1 - sw)
        zero_rect = np.asarray([ztop, zleft, ztop + rh, zleft + rw], dtype=np.int32)
        state_rect = np.asarray([stop, sleft, stop + sh, sleft + sw], dtype=np.int32)

        # General affine/projective, diagonally dominant matrices cover every
        # cell and keep inversion well conditioned over this finite range.
        matrix50 = np.asarray([
            rng.uniform(0.65, 1.45), rng.uniform(-0.35, 0.35), rng.uniform(-4.0, 4.0),
            rng.uniform(-0.35, 0.35), rng.uniform(0.65, 1.45), rng.uniform(-4.0, 4.0),
            rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(0.75, 1.35),
        ], dtype=np.float64)
        matrix50 = np.concatenate((matrix50, np.asarray([17.25, -99.5], dtype=np.float64)))
        coefficients = np.full(11, SENTINEL, dtype=np.float32)
        snapshots = (zero_rect.tobytes(), state_rect.tobytes(), matrix50.tobytes())
        status = bridge.prepare_coordinates_raw(
            zero_rect, state_rect, matrix50, coefficients,
            zero_count=4, state_count=4, matrix_count=9, coefficient_count=11,
        )
        expected_status, expected = coordinate_oracle.reference_prepare(zero_rect, state_rect, matrix50)
        assert status == OK == expected_status, f"constructor geometry state {index}"
        assert expected is not None
        _bits_equal(coefficients[:6], np.asarray(expected, dtype=np.float32), f"constructor state {index}")
        assert np.all(coordinate_oracle.bits32(coefficients[6:]) == SENTINEL_BITS)
        assert snapshots == (zero_rect.tobytes(), state_rect.tobytes(), matrix50.tobytes())
        seen.add((tuple(zero_rect), tuple(state_rect), tuple(matrix50[:9])))
    assert len(seen) == 128


def test_coordinate_constructor_fma64_narrowing_counterexamples(bridge):
    # In this projective case, keeping both matrix products in FP64 produces
    # two different low bits from narrowing each product early to FP32.
    zero_rect = np.asarray([-10506, 98891, -10411, 98973], dtype=np.int32)
    state_rect = np.asarray([98516, -23599, 98580, -23516], dtype=np.int32)
    matrix50 = np.asarray([
        1.3806472081255654, -0.4227749711400317, -1.4603487226091287,
        0.41955540689806003, 1.1406956519867433, -1.7134005154750636,
        -0.017673116027748906, -0.043164870498540966, 0.800176690397063,
    ], dtype=np.float64)
    expected_status, expected = coordinate_oracle.reference_prepare(zero_rect, state_rect, matrix50)
    assert expected_status == OK and expected is not None
    expected_bits = np.asarray([0xC3792F59, 0xC5FFC92F, 0xC850DA03, 0x4342CD6D, 0x45C7771C, 0x4822D69B], dtype=np.uint32)
    np.testing.assert_array_equal(coordinate_oracle.bits32(np.asarray(expected)), expected_bits)

    reference_h, reference_w = int(zero_rect[2] - zero_rect[0]), int(zero_rect[3] - zero_rect[1])
    state_h, state_w = int(state_rect[2] - state_rect[0]), int(state_rect[3] - state_rect[1])
    t = (1.0, 0.0, float(zero_rect[0]), 0.0, 1.0, float(zero_rect[1]), 0.0, 0.0, 1.0)
    s = (reference_h / state_h, 0.0, 0.0, 0.0, reference_w / state_w, 0.0, 0.0, 0.0, 1.0)
    u = (1.0, 0.0, float(coordinate_oracle._wrapped_neg32(int(state_rect[0]))),
         0.0, 1.0, float(coordinate_oracle._wrapped_neg32(int(state_rect[1]))), 0.0, 0.0, 1.0)
    _, inverse = matrix_oracle._invert3(tuple(matrix50))
    assert inverse is not None

    def multiply_then_narrow(left, right):
        status, value = matrix_oracle._multiply3(left, right)
        assert status == matrix_oracle.ResampleMatrixStatus.OK and value is not None
        return tuple(float(np.float32(item)) for item in value)

    early_ts = multiply_then_narrow(t, s)
    early_tsu = multiply_then_narrow(early_ts, u)
    early_composed = multiply_then_narrow(early_tsu, inverse)
    early_bits = coordinate_oracle.bits32(np.asarray(early_composed[:6], dtype=np.float32))
    np.testing.assert_array_equal(early_bits, np.asarray([0xC3792F59, 0xC5FFC92F, 0xC850DA03, 0x4342CD6C, 0x45C7771B, 0x4822D69B], dtype=np.uint32))
    assert not np.array_equal(expected_bits, early_bits)

    # This near-cancelled but above-threshold cofactor distinguishes binary64
    # FMA from a separately rounded product and subtraction, before narrowing.
    determinant_block = np.asarray([
        1.0, 0.0, 0.0,
        0.0, 1.3318869383544185, 1.546138110994592,
        0.0, 0.7207181699973391, 0.8366549727997228,
    ], dtype=np.float64)
    a, b, c, d = determinant_block[4], determinant_block[5], determinant_block[7], determinant_block[8]
    negative_product = -(b * c)
    fused_cofactor = matrix_oracle._fma(a, d, negative_product)
    split_cofactor = a * d + negative_product
    assert abs(fused_cofactor) >= matrix_oracle.DETERMINANT_THRESHOLD
    assert fused_cofactor != split_cofactor
    fused_narrow = np.float32(d / fused_cofactor)
    split_narrow = np.float32(d / split_cofactor)
    assert int(coordinate_oracle.bits32([fused_narrow])[0]) == 0x4F3E4BE5
    assert int(coordinate_oracle.bits32([split_narrow])[0]) == 0x4F3E4BE4
    coeff = _prepare_once(
        bridge, np.asarray([0, 0, 1, 1], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.int32), determinant_block,
    )
    assert int(coordinate_oracle.bits32(coeff[4:5])[0]) == 0x4F3E4BE5


def test_coordinate_constructor_singular_and_nonfinite_statuses_are_atomic(bridge):
    rect = np.asarray([0, 0, 2, 3], dtype=np.int32)
    output = np.full(9, SENTINEL, dtype=np.float32)
    matrices = [
        (np.asarray([np.nan, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), INVALID_PREPARED),
        (np.asarray([np.inf, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), INVALID_PREPARED),
        (np.zeros(9, dtype=np.float64), INVALID_PREPARED),
        (np.asarray([1, 0, 0, 0, 1, 0, 0, 0, np.nextafter(1e-10, 0.0)], dtype=np.float64), INVALID_PREPARED),
        (np.asarray([1e308, 0, 0, 0, 1e308, 0, 0, 0, 1], dtype=np.float64), NON_FINITE_INTERMEDIATE),
        (np.asarray([1e-100, 0, 0, 0, 1e100, 0, 0, 0, 1], dtype=np.float64), NON_FINITE_INTERMEDIATE),
    ]
    for matrix50, expected_status in matrices:
        before = output.tobytes()
        status = bridge.prepare_coordinates_raw(rect, rect, matrix50, output)
        assert status == expected_status
        assert output.tobytes() == before

    threshold_matrix = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1e-10], dtype=np.float64)
    expected_status, expected = coordinate_oracle.reference_prepare(rect, rect, threshold_matrix)
    assert expected_status == OK and expected is not None
    assert bridge.prepare_coordinates_raw(rect, rect, threshold_matrix, output) == OK
    _bits_equal(output[:6], np.asarray(expected, dtype=np.float32), "determinant threshold boundary")

    invalid_rects = [
        (np.asarray([0, 0, 0, 3], dtype=np.int32), rect),
        (rect, np.asarray([0, 0, 1, 0], dtype=np.int32)),
        (np.asarray([-(1 << 31), 0, (1 << 31) - 1, 3], dtype=np.int32), rect),
    ]
    for zero_rect, state_rect in invalid_rects:
        output[:] = SENTINEL
        before = output.tobytes()
        assert bridge.prepare_coordinates_raw(zero_rect, state_rect, np.eye(3).reshape(-1), output) == INVALID_DIMENSIONS
        assert output.tobytes() == before


def test_coordinate_constructor_bridge_counts_tasks_and_atomic_failures(bridge):
    zero_rect = np.asarray([0, 0, 2, 3], dtype=np.int32)
    state_rect = np.asarray([-1, 4, 1, 7], dtype=np.int32)
    matrix50 = np.eye(3, dtype=np.float64).reshape(-1)
    output = np.full(10, SENTINEL, dtype=np.float32)

    for kwargs in (
        {"zero_count": 3},
        {"state_count": 3},
        {"matrix_count": 8},
        {"coefficient_count": 5},
    ):
        output[:] = SENTINEL
        before = output.tobytes()
        status = bridge.prepare_coordinates_raw(
            zero_rect, state_rect, matrix50, output, **kwargs
        )
        assert status == SHORT_BUFFER
        assert output.tobytes() == before

    for task, expected_status in ((1, UNSUPPORTED_TRANSFORM), (2, INVALID_ARGUMENT)):
        output[:] = SENTINEL
        before = output.tobytes()
        status = bridge.prepare_coordinates_raw(
            zero_rect, state_rect, matrix50, output, coordinate_task=task
        )
        assert status == expected_status
        assert output.tobytes() == before

    null_cases = (
        {"zero_rect": None, "state_rect": state_rect, "matrix50": matrix50, "coefficients": output},
        {"zero_rect": zero_rect, "state_rect": None, "matrix50": matrix50, "coefficients": output},
        {"zero_rect": zero_rect, "state_rect": state_rect, "matrix50": None, "coefficients": output},
        {"zero_rect": zero_rect, "state_rect": state_rect, "matrix50": matrix50, "coefficients": None},
    )
    for case in null_cases:
        output[:] = SENTINEL
        before = output.tobytes()
        status = bridge.prepare_coordinates_raw(**case)
        assert status == INVALID_ARGUMENT
        assert output.tobytes() == before

    zero_bytes, state_bytes, matrix_bytes, output_bytes = (
        bytearray(4 * C.sizeof(C.c_int32) + 1),
        bytearray(4 * C.sizeof(C.c_int32) + 1),
        bytearray(9 * C.sizeof(C.c_double) + 1),
        bytearray(6 * C.sizeof(C.c_float) + 1),
    )
    zero_ptr = C.cast(C.c_void_p(C.addressof(C.c_char.from_buffer(zero_bytes)) + 1), I32_PTR)
    state_ptr = C.cast(C.c_void_p(C.addressof(C.c_char.from_buffer(state_bytes)) + 1), I32_PTR)
    matrix_ptr = C.cast(C.c_void_p(C.addressof(C.c_char.from_buffer(matrix_bytes)) + 1), F64_PTR)
    output_ptr = C.cast(C.c_void_p(C.addressof(C.c_char.from_buffer(output_bytes)) + 1), F32_PTR)
    before = tuple(bytes(buf) for buf in (zero_bytes, state_bytes, matrix_bytes, output_bytes))
    aligned_zero = (C.c_int32 * 4)(*zero_rect)
    aligned_state = (C.c_int32 * 4)(*state_rect)
    aligned_matrix = (C.c_double * 9)(*matrix50)

    for kwargs in (
        {"zero_rect": zero_ptr, "state_rect": aligned_state, "matrix50": aligned_matrix, "coefficients": output},
        {"zero_rect": aligned_zero, "state_rect": state_ptr, "matrix50": aligned_matrix, "coefficients": output},
        {"zero_rect": aligned_zero, "state_rect": aligned_state, "matrix50": matrix_ptr, "coefficients": output},
        {"zero_rect": aligned_zero, "state_rect": aligned_state, "matrix50": aligned_matrix, "coefficients": output_ptr},
    ):
        assert bridge.prepare_coordinates_raw(
            **kwargs, zero_count=4, state_count=4, matrix_count=9, coefficient_count=6
        ) == ALIGNMENT
    assert tuple(bytes(buf) for buf in (zero_bytes, state_bytes, matrix_bytes, output_bytes)) == before

    overflow_cases = (
        {"zero_rect": zero_rect, "zero_count": SIZE_MAX // 4 + 1},
        {"state_rect": state_rect, "state_count": SIZE_MAX // 4 + 1},
        {"matrix50": matrix50, "matrix_count": SIZE_MAX // 8 + 1},
        {"coefficients": output, "coefficient_count": SIZE_MAX // 4 + 1},
        {"zero_rect": 0xFFFFFFFFFFFFFFF8, "zero_count": 4},
        {"state_rect": 0xFFFFFFFFFFFFFFF8, "state_count": 4},
        {"matrix50": 0xFFFFFFFFFFFFFFF8, "matrix_count": 3},
        {"coefficients": 0xFFFFFFFFFFFFFFFC, "coefficient_count": 6},
    )
    for overrides in overflow_cases:
        args = {
            "zero_rect": zero_rect, "zero_count": 4,
            "state_rect": state_rect, "state_count": 4,
            "matrix50": matrix50, "matrix_count": 9,
            "coefficients": output, "coefficient_count": 10,
        }
        args.update(overrides)
        output[:] = SENTINEL
        before = output.tobytes()
        assert bridge.prepare_coordinates_raw(**args) == ARITHMETIC_OVERFLOW
        assert output.tobytes() == before


def test_coordinate_constructor_full_declared_input_aliases_are_rejected(bridge):
    state_rect = np.asarray([-1, 4, 1, 7], dtype=np.int32)
    matrix50 = np.eye(3, dtype=np.float64).reshape(-1)
    zero_storage = np.zeros(20, dtype=np.int32)
    zero_storage[:4] = [0, 0, 2, 3]
    zero_output = zero_storage[6:12].view(np.float32)
    zero_snapshot = zero_storage.tobytes()
    assert bridge.prepare_coordinates_raw(
        zero_storage, state_rect, matrix50, zero_output,
        zero_count=8, state_count=4, matrix_count=9, coefficient_count=6,
    ) == OVERLAP
    assert zero_storage.tobytes() == zero_snapshot

    state_storage = np.zeros(20, dtype=np.int32)
    state_storage[:4] = state_rect
    state_output = state_storage[6:12].view(np.float32)
    state_snapshot = state_storage.tobytes()
    assert bridge.prepare_coordinates_raw(
        np.asarray([0, 0, 2, 3], dtype=np.int32), state_storage, matrix50, state_output,
        zero_count=4, state_count=8, matrix_count=9, coefficient_count=6,
    ) == OVERLAP
    assert state_storage.tobytes() == state_snapshot

    matrix_storage = np.zeros(12, dtype=np.float64)
    matrix_storage[:9] = np.eye(3, dtype=np.float64).reshape(-1)
    matrix_output = matrix_storage[9:].view(np.float32)
    matrix_snapshot = matrix_storage.tobytes()
    assert bridge.prepare_coordinates_raw(
        np.asarray([0, 0, 2, 3], dtype=np.int32), state_rect, matrix_storage, matrix_output,
        zero_count=4, state_count=4, matrix_count=12, coefficient_count=6,
    ) == OVERLAP
    assert matrix_storage.tobytes() == matrix_snapshot


def test_128_independent_rect_coefficient_and_padding_states(bridge):
    rng = np.random.default_rng(0xC0A4D1)
    states = set()
    for state in range(128):
        if state % 8 == 0:
            top = int(rng.integers(-(1 << 31), -(1 << 31) + 128))
            left = int(rng.integers((1 << 31) - 128, (1 << 31) - 1))
        elif state % 8 == 1:
            top = int(rng.integers((1 << 23) - 8, (1 << 23) + 16))
            left = int(rng.integers((1 << 24) - 8, (1 << 24) + 16))
        else:
            top, left = (int(x) for x in rng.integers(-100000, 100000, size=2))
        height, width = int(rng.integers(1, 5)), int(rng.integers(1, 7))
        # Keep the exclusive bounds in the signed int32 domain.
        top = min(top, (1 << 31) - 1 - height)
        left = min(left, (1 << 31) - 1 - width)
        rect = np.asarray([top, left, top + height, left + width], dtype=np.int32)
        coefficients = rng.uniform(-2.25, 2.25, size=6).astype(np.float32)
        stride = width + int(rng.integers(0, 6))
        count = height * stride + int(rng.integers(0, 5))
        coefficients = np.concatenate((coefficients, np.asarray([12.5, -9.25], dtype=np.float32)))
        coeff_before, rect_before = coefficients.tobytes(), rect.tobytes()
        v = np.full(count, SENTINEL, dtype=np.float32)
        u = np.full(count + 2, SENTINEL, dtype=np.float32)
        v_before, u_before = v.tobytes(), u.tobytes()
        status = bridge.coordinates_raw(
            coefficients, rect, v, u, coefficient_count=8, rect_count=4,
            v_count=v.size, u_count=u.size, row_stride=stride,
        )
        expected_v, expected_u = coordinate_oracle.reference_planes(
            coefficients, rect, stride, v_count=v.size, u_count=u.size
        )
        assert status == OK, f"independent state {state}"
        _bits_equal(v, expected_v, f"state {state} v")
        _bits_equal(u, expected_u, f"state {state} u")
        assert coefficients.tobytes() == coeff_before
        assert rect.tobytes() == rect_before
        assert v_before != v.tobytes() or u_before != u.tobytes()
        states.add((tuple(rect), tuple(coordinate_oracle.bits32(coefficients[:6])), stride, count))
    assert len(states) == 128


def test_single_pixel_identity_negative_and_int32_near_limits(bridge):
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    _valid_call(bridge, identity, np.asarray([-(1 << 31), -(1 << 31), -(1 << 31) + 1, -(1 << 31) + 1], dtype=np.int32), row_stride=1)
    _valid_call(bridge, identity, np.asarray([(1 << 31) - 2, (1 << 31) - 2, (1 << 31) - 1, (1 << 31) - 1], dtype=np.int32), row_stride=1)
    _valid_call(bridge, identity, np.asarray([-16777217, -16777217, -16777215, -16777213], dtype=np.int32), row_stride=4)


def test_hardcoded_identity_rounding_and_recursive_column_counterexample(bridge):
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    # The row path ends at 8,388,607.5 after the distinct -0.5 operation.
    row_rect = np.asarray([1 << 23, 0, (1 << 23) + 1, 1], dtype=np.int32)
    v = np.full(1, SENTINEL, dtype=np.float32)
    u = np.full(1, SENTINEL, dtype=np.float32)
    assert bridge.coordinates_raw(identity, row_rect, v, u, row_stride=1) == OK
    assert int(coordinate_oracle.bits32(v)[0]) == 0x4AFFFFFF
    assert int(coordinate_oracle.bits32(u)[0]) == 0x00000000

    # Starting at 2^24, recursive FP32 +1 stays stuck at 2^24. Reconstructing
    # left + column_index as an integer would produce 2^24+2 at the third lane.
    column_rect = np.asarray([0, 1 << 24, 1, (1 << 24) + 3], dtype=np.int32)
    v = np.full(3, SENTINEL, dtype=np.float32)
    u = np.full(3, SENTINEL, dtype=np.float32)
    assert bridge.coordinates_raw(identity, column_rect, v, u, row_stride=3) == OK
    expected_bits = np.asarray([0x4B800000, 0x4B800000, 0x4B800000], dtype=np.uint32)
    np.testing.assert_array_equal(coordinate_oracle.bits32(v), np.zeros(3, dtype=np.uint32))
    np.testing.assert_array_equal(coordinate_oracle.bits32(u), expected_bits)
    assert int(coordinate_oracle.bits32(np.asarray([np.float32((1 << 24) + 2)]))[0]) == 0x4B800001


def test_fma_and_nonassociative_row_operation_order_counterexamples(bridge):
    # Row 1 distinguishes FMA from a split multiply/add, then the explicitly
    # ordered (C0 + t) - .5 from C0 + (t - .5).
    coefficients = np.asarray([
        _bit_pattern(0xC3BB8D39), _bit_pattern(0x43A97C99),
        _bit_pattern(0x4B2F0C17), 0.0, 0.0, 0.0,
    ], dtype=np.float32)
    row_half = coordinate_oracle.add32(coordinate_oracle.f32(1), 0.5)
    term = coordinate_oracle.fma32(
        coefficients[0], row_half, coordinate_oracle.mul32(coefficients[1], 0.5)
    )
    split_term = coordinate_oracle.add32(
        coordinate_oracle.mul32(coefficients[0], row_half),
        coordinate_oracle.mul32(coefficients[1], 0.5),
    )
    assert int(coordinate_oracle.bits32([term])[0]) == 0xC3C49589
    assert int(coordinate_oracle.bits32([split_term])[0]) == 0xC3C4958A
    base = coordinate_oracle.sub32(
        coordinate_oracle.add32(coefficients[2], term), 0.5
    )
    reassociated_base = coordinate_oracle.add32(
        coefficients[2], coordinate_oracle.sub32(term, 0.5)
    )
    assert int(coordinate_oracle.bits32([base])[0]) == 0x4B2F0A8E
    assert int(coordinate_oracle.bits32([reassociated_base])[0]) == 0x4B2F0A8D
    rect = np.asarray([1, 0, 2, 3], dtype=np.int32)
    v = np.full(3, SENTINEL, dtype=np.float32)
    u = np.full(3, SENTINEL, dtype=np.float32)
    assert bridge.coordinates_raw(coefficients, rect, v, u, row_stride=3) == OK
    expected_v, expected_u = coordinate_oracle.reference_planes(coefficients, rect, 3)
    _bits_equal(v, expected_v, "fixed FMA v")
    _bits_equal(u, expected_u, "fixed FMA u")
    assert coordinate_oracle.bits32(v).tolist() == [0x4B2F0A8E, 0x4B2F0BE1, 0x4B2F0D34]
    assert coordinate_oracle.bits32(u).tolist() == [0xBF000000] * 3

    # The column FMA is independently distinguishable from FMUL then FADD.
    col_coefficients = np.asarray([
        _bit_pattern(0xC3DD9905), _bit_pattern(0xC42250F9),
        _bit_pattern(0xC4248C70), 0.0, 0.0, 0.0,
    ], dtype=np.float32)
    col_base_term = coordinate_oracle.fma32(
        col_coefficients[0], 0.5,
        coordinate_oracle.mul32(col_coefficients[1], 0.5),
    )
    col_base = coordinate_oracle.sub32(
        coordinate_oracle.add32(col_coefficients[2], col_base_term), 0.5
    )
    col = coordinate_oracle.f32(17)
    fused = coordinate_oracle.fma32(col_coefficients[1], col, col_base)
    split = coordinate_oracle.add32(
        coordinate_oracle.mul32(col_coefficients[1], col), col_base
    )
    assert int(coordinate_oracle.bits32([fused])[0]) == 0xC63F49BB
    assert int(coordinate_oracle.bits32([split])[0]) == 0xC63F49BC
    col_rect = np.asarray([0, 17, 1, 20], dtype=np.int32)
    v = np.full(3, SENTINEL, dtype=np.float32)
    u = np.full(3, SENTINEL, dtype=np.float32)
    assert bridge.coordinates_raw(col_coefficients, col_rect, v, u, row_stride=3) == OK
    assert coordinate_oracle.bits32(v).tolist() == [0xC63F49BB, 0xC6496ECB, 0xC65393DB]


def test_ieee_coefficients_propagate_nan_and_infinity(bridge):
    cases = [
        np.asarray([np.nan, 0.25, 0.5, 0.75, 1.0, -0.5], dtype=np.float32),
        np.asarray([np.inf, 0.25, 0.5, 0.75, 1.0, -0.5], dtype=np.float32),
        np.asarray([0.25, -np.inf, 0.5, 0.75, np.inf, -0.5], dtype=np.float32),
        np.asarray([0.25, 0.5, -np.inf, 0.75, 1.0, np.inf], dtype=np.float32),
    ]
    rect = np.asarray([-2, -3, 1, 3], dtype=np.int32)
    saw_nan = saw_infinity = False
    for coefficients in cases:
        before = coefficients.tobytes()
        v = np.full(3 * 8, SENTINEL, dtype=np.float32)
        u = np.full_like(v, SENTINEL)
        assert bridge.coordinates_raw(coefficients, rect, v, u, row_stride=8) == OK
        expected_v, expected_u = coordinate_oracle.reference_planes(
            coefficients, rect, 8, v_count=v.size, u_count=u.size
        )
        _bits_equal(v, expected_v, "IEEE v")
        _bits_equal(u, expected_u, "IEEE u")
        saw_nan |= bool(np.isnan(v[:3]).any() or np.isnan(u[:3]).any())
        saw_infinity |= bool(np.isinf(v[:3]).any() or np.isinf(u[:3]).any())
        assert coefficients.tobytes() == before
    assert saw_nan and saw_infinity


def _expect_atomic(bridge, expected_status, *, coefficients=None, rect=None, out_v=None, out_u=None, **kwargs):
    snapshots = []
    for value in (coefficients, rect, out_v, out_u):
        if isinstance(value, np.ndarray):
            snapshots.append((value, value.tobytes()))
    status = bridge.coordinates_raw(
        coefficients, rect, out_v, out_u,
        row_stride=kwargs.pop("row_stride", 2), **kwargs,
    )
    assert status == expected_status
    for array, before in snapshots:
        assert array.tobytes() == before


@pytest.mark.parametrize("bad_rect", [
    (0, 0, 0, 1),
    (2, 0, 1, 1),
    (0, 2, 1, 1),
    (-(1 << 31), 0, (1 << 31) - 1, 1),
    (0, -(1 << 31), 1, (1 << 31) - 1),
])
def test_invalid_dimensions_are_atomic(bridge, bad_rect):
    coefficients = np.ones(6, dtype=np.float32)
    rect = np.asarray(bad_rect, dtype=np.int32)
    v = np.full(16, SENTINEL, dtype=np.float32)
    u = np.full(16, SENTINEL, dtype=np.float32)
    _expect_atomic(bridge, INVALID_DIMENSIONS, coefficients=coefficients, rect=rect, out_v=v, out_u=u, row_stride=4)


def test_invalid_counts_stride_task_and_short_buffers_are_atomic(bridge):
    coefficients = np.ones(7, dtype=np.float32)
    rect = np.asarray([-1, 4, 1, 7], dtype=np.int32)
    v, u = np.full(12, SENTINEL, dtype=np.float32), np.full(12, SENTINEL, dtype=np.float32)
    for coefficient_count, rect_count in ((5, 4), (6, 3)):
        _expect_atomic(
            bridge, SHORT_BUFFER, coefficients=coefficients, rect=rect, out_v=v, out_u=u,
            coefficient_count=coefficient_count, rect_count=rect_count, row_stride=3,
        )
    _expect_atomic(bridge, INVALID_ARGUMENT, coefficients=coefficients, rect=rect, out_v=v, out_u=u, coordinate_task=2, row_stride=3)
    _expect_atomic(bridge, UNSUPPORTED_TRANSFORM, coefficients=coefficients, rect=rect, out_v=v, out_u=u, coordinate_task=1, row_stride=3)
    _expect_atomic(bridge, INVALID_STRIDE, coefficients=coefficients, rect=rect, out_v=v, out_u=u, row_stride=2)
    _expect_atomic(bridge, INVALID_STRIDE, coefficients=coefficients, rect=np.asarray([0, 0, 1, 1], dtype=np.int32), out_v=v, out_u=u, row_stride=1 << 31)
    _expect_atomic(
        bridge, SHORT_BUFFER, coefficients=coefficients, rect=rect, out_v=v[:5], out_u=u,
        v_count=5, row_stride=3,
    )
    _expect_atomic(
        bridge, SHORT_BUFFER, coefficients=coefficients, rect=rect, out_v=v, out_u=u[:5],
        u_count=5, row_stride=3,
    )
    # Count one is a valid capacity for a 1x1 rectangle; there is no minimum
    # four-float output-size requirement.
    one = np.full(1, SENTINEL, dtype=np.float32)
    one_rect = np.asarray([0, 0, 1, 1], dtype=np.int32)
    assert bridge.coordinates_raw(coefficients, one_rect, one, one.copy(), row_stride=1) == OK


def test_null_alignment_and_declared_span_overflow_are_atomic(bridge):
    coefficients = np.ones(6, dtype=np.float32)
    rect = np.asarray([0, 0, 2, 2], dtype=np.int32)
    v, u = np.full(4, SENTINEL, dtype=np.float32), np.full(4, SENTINEL, dtype=np.float32)
    for kwargs in (
        {"coefficients": None, "rect": rect, "out_v": v, "out_u": u},
        {"coefficients": coefficients, "rect": None, "out_v": v, "out_u": u},
        {"coefficients": coefficients, "rect": rect, "out_v": None, "out_u": u},
        {"coefficients": coefficients, "rect": rect, "out_v": v, "out_u": None},
    ):
        _expect_atomic(bridge, INVALID_ARGUMENT, **kwargs, row_stride=2)

    coeff_bytes = bytearray(6 * 4 + 1)
    rect_bytes = bytearray(4 * 4 + 1)
    out_v_bytes, out_u_bytes = bytearray(4 * 4 + 1), bytearray(4 * 4 + 1)
    aligned_rect = (C.c_int32 * 4)(0, 0, 2, 2)
    aligned_coefficients = (C.c_float * 6)(1, 0, 0, 0, 1, 0)
    misaligned_coefficients = C.cast(C.c_void_p(C.addressof(C.c_char.from_buffer(coeff_bytes)) + 1), F32_PTR)
    misaligned_rect = C.cast(C.c_void_p(C.addressof(C.c_char.from_buffer(rect_bytes)) + 1), I32_PTR)
    misaligned_v = C.cast(C.c_void_p(C.addressof(C.c_char.from_buffer(out_v_bytes)) + 1), F32_PTR)
    misaligned_u = C.cast(C.c_void_p(C.addressof(C.c_char.from_buffer(out_u_bytes)) + 1), F32_PTR)
    alignment_snapshots = tuple(bytes(buf) for buf in (coeff_bytes, rect_bytes, out_v_bytes, out_u_bytes))
    assert bridge.coordinates_raw(
        misaligned_coefficients,
        aligned_rect,
        (C.c_float * 4)(), (C.c_float * 4)(),
        coefficient_count=6, rect_count=4, v_count=4, u_count=4, row_stride=2,
    ) == ALIGNMENT
    assert bridge.coordinates_raw(
        aligned_coefficients,
        misaligned_rect,
        (C.c_float * 4)(), (C.c_float * 4)(),
        coefficient_count=6, rect_count=4, v_count=4, u_count=4, row_stride=2,
    ) == ALIGNMENT
    assert bridge.coordinates_raw(
        aligned_coefficients, aligned_rect, misaligned_v, (C.c_float * 4)(),
        coefficient_count=6, rect_count=4, v_count=4, u_count=4, row_stride=2,
    ) == ALIGNMENT
    assert bridge.coordinates_raw(
        aligned_coefficients, aligned_rect, (C.c_float * 4)(), misaligned_u,
        coefficient_count=6, rect_count=4, v_count=4, u_count=4, row_stride=2,
    ) == ALIGNMENT
    assert tuple(bytes(buf) for buf in (coeff_bytes, rect_bytes, out_v_bytes, out_u_bytes)) == alignment_snapshots

    for kwargs in (
        {"coefficient_count": SIZE_MAX // 4 + 1},
        {"rect_count": SIZE_MAX // 4 + 1},
        {"v_count": 2, "out_v": 0xFFFFFFFFFFFFFFFC},
        {"u_count": 2, "out_u": 0xFFFFFFFFFFFFFFFC},
    ):
        call = {"coefficients": coefficients, "rect": rect, "out_v": v, "out_u": u, "row_stride": 2}
        call.update(kwargs)
        _expect_atomic(bridge, ARITHMETIC_OVERFLOW, **call)


def test_output_input_and_output_output_aliases_are_rejected_atomically(bridge):
    coefficients = np.full(12, np.float32(0.25), dtype=np.float32)
    rect = np.asarray([0, 0, 2, 2], dtype=np.int32)
    other = np.full(4, SENTINEL, dtype=np.float32)
    before = coefficients.tobytes()
    # The overlapping output begins beyond the six values actually read, but
    # inside the complete coefficient_count span declared to the bridge.
    out_v = coefficients[8:12]
    _expect_atomic(
        bridge, OVERLAP, coefficients=coefficients, rect=rect, out_v=out_v,
        out_u=other, coefficient_count=12, row_stride=2,
    )
    assert coefficients.tobytes() == before

    rect_storage = np.asarray([0, 0, 2, 2, 0x3F800000, 0x40000000, 0, 0], dtype=np.int32)
    rect_before = rect_storage.tobytes()
    rect_output = rect_storage[4:].view(np.float32)
    _expect_atomic(
        bridge, OVERLAP, coefficients=np.ones(6, dtype=np.float32),
        rect=rect_storage, out_v=rect_output, out_u=other,
        rect_count=8, v_count=4, row_stride=2,
    )
    assert rect_storage.tobytes() == rect_before

    shared = np.full(4, SENTINEL, dtype=np.float32)
    shared_before = shared.tobytes()
    _expect_atomic(
        bridge, OVERLAP, coefficients=np.ones(6, dtype=np.float32), rect=rect_storage[:4],
        out_v=shared, out_u=shared, row_stride=2,
    )
    assert shared.tobytes() == shared_before


def test_active_sample_resource_limit_is_enforced_before_writes(bridge):
    height, width = 4097, 8192  # 2*height*width is just over 2^26.
    count = height * width
    rect = np.asarray([0, 0, height, width], dtype=np.int32)
    coefficients = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    byte_count = count * np.dtype(np.float32).itemsize
    with mmap.mmap(-1, byte_count) as map_v, mmap.mmap(-1, byte_count) as map_u:
        v = np.frombuffer(map_v, dtype=np.float32)
        u = np.frombuffer(map_u, dtype=np.float32)
        v[0], v[-1], u[0], u[-1] = SENTINEL, SENTINEL, SENTINEL, SENTINEL
        before = (v[0].view(np.uint32), v[-1].view(np.uint32), u[0].view(np.uint32), u[-1].view(np.uint32))
        status = bridge.coordinates_raw(
            coefficients, rect, v, u, v_count=count, u_count=count,
            row_stride=width,
        )
        assert status == RESOURCE_LIMIT
        after = (v[0].view(np.uint32), v[-1].view(np.uint32), u[0].view(np.uint32), u[-1].view(np.uint32))
        np.testing.assert_array_equal(after, before)
        del v, u


def _pack_extended_payload(rows, cols, slices, values, *, byteorder=1):
    prefix = ">" if byteorder else "<"
    header = struct.pack(
        prefix + "II4dI5fI3f",
        rows, cols, 0.75, 1.375, -0.625, 2.25, slices,
        0.03125, 0.0625, 0.015625, 0.046875, 0.0234375,
        3, 1.25, 0.125, 8.0,
    )
    return header + struct.pack(prefix + f"{len(values)}f", *values)


def test_coordinates_map_prepare_gain_evaluation_and_cpp_apply_chain(bridge):
    # Locally links coordinate construction -> coordinate generation ->
    # existing payload parsing/map preparation -> numeric gain evaluation ->
    # existing C++ apply. This is not an Adobe RGB pipeline.
    outer, inner, value_stride, coord_stride = 3, 5, 8, 9
    coord_rect = np.asarray([0, 0, outer, inner], dtype=np.int32)
    coordinate_matrix = np.eye(3, dtype=np.float64).reshape(-1)
    coord_coefficients = np.full(10, SENTINEL, dtype=np.float32)
    assert bridge.prepare_coordinates_raw(
        coord_rect, coord_rect, coordinate_matrix, coord_coefficients,
        coefficient_count=10,
    ) == OK
    prep_status, expected_coefficients = coordinate_oracle.reference_prepare(
        coord_rect, coord_rect, coordinate_matrix
    )
    assert prep_status == OK and expected_coefficients is not None
    _bits_equal(coord_coefficients[:6], np.asarray(expected_coefficients, dtype=np.float32), "integrated prepared coefficients")
    assert np.all(coordinate_oracle.bits32(coord_coefficients[6:]) == SENTINEL_BITS)
    coordinates_v = np.full(outer * coord_stride, SENTINEL, dtype=np.float32)
    coordinates_u = np.full_like(coordinates_v, SENTINEL)
    assert bridge.coordinates_raw(
        coord_coefficients[:6], coord_rect, coordinates_v, coordinates_u,
        row_stride=coord_stride,
    ) == OK
    expected_v, expected_u = coordinate_oracle.reference_planes(
        coord_coefficients[:6], coord_rect, coord_stride,
        v_count=coordinates_v.size, u_count=coordinates_u.size,
    )
    _bits_equal(coordinates_v, expected_v, "integrated coordinate v")
    _bits_equal(coordinates_u, expected_u, "integrated coordinate u")

    rows, cols, slices = 3, 5, 2
    encoded_values = tuple(float(0.3 + 0.11 * index) for index in range(rows * cols * slices))
    payload = _pack_extended_payload(rows, cols, slices, encoded_values)
    rect = (0, 0, 3, 5)
    map_scalar = 0.875
    expected_prepared = map_oracle.payload_to_prepared(
        payload, map_oracle.TAG_EXTENDED, 1, rect, map_scalar
    )
    status, meta, fields, geometry, table, written, _data, _rect = bridge.prepare(
        payload, map_oracle.TAG_EXTENDED, 1, rect, map_scalar
    )
    assert status == map_oracle.OK
    assert written == expected_prepared.table.size
    np.testing.assert_array_equal(map_oracle.expected_bits64(meta[:19]), map_oracle.expected_bits64(expected_prepared.meta))
    np.testing.assert_array_equal(map_oracle.expected_bits32(fields[:16]), map_oracle.expected_bits32(expected_prepared.fields))
    np.testing.assert_array_equal(geometry[:5], expected_prepared.geometry)
    np.testing.assert_array_equal(map_oracle.expected_bits32(table[:written]), map_oracle.expected_bits32(expected_prepared.table))

    rng = np.random.default_rng(0xA52C)
    pixels = [np.full(outer * value_stride, np.float32(-17.25), dtype=np.float32) for _ in range(3)]
    for row in range(outer):
        for column in range(inner):
            offset = row * value_stride + column
            for channel in range(3):
                pixels[channel][offset] = np.float32(rng.uniform(0.05, 0.9))
    gains = np.full(outer * coord_stride, np.float32(-31.0), dtype=np.float32)
    gain_inputs = [*pixels, coordinates_v.copy(), coordinates_u.copy()]
    assert bridge.evaluate_gain(
        fields[:16], geometry[:5], table[:written], gain_inputs, gains,
        outer=outer, inner=inner, value_stride=value_stride, coord_stride=coord_stride,
    ) == OK
    expected_gains = gain_oracle.reference_rows(
        fields[:16], geometry[:5], table[:written], gain_inputs,
        outer, inner, value_stride, coord_stride,
    )
    expected_gains_padded = np.full_like(gains, np.float32(-31.0))
    for row in range(outer):
        expected_gains_padded[row * coord_stride : row * coord_stride + inner] = expected_gains[row]
    _bits_equal(gains, expected_gains_padded, "integrated gains")

    apply_inputs = [channel.copy() for channel in pixels] + [gains.copy()]
    outputs = [np.full(outer * value_stride, SENTINEL, dtype=np.float32) for _ in range(3)]
    source_snapshots = [item.copy() for item in apply_inputs]
    pivot = 0.375
    assert bridge.apply(
        apply_inputs, outputs, outer=outer, inner=inner,
        value_stride=value_stride, gain_stride=coord_stride, pivot=pivot,
        mode=apply_oracle.CPU_VECTOR_MODE, clip=1,
    ) == OK
    expected_outputs = apply_oracle.reference_rows(
        source_snapshots, outer, inner, value_stride, coord_stride,
        pivot, apply_oracle.CPU_VECTOR_MODE, 1,
    )
    for actual, expected in zip(outputs, expected_outputs):
        _bits_equal(actual, expected, "integrated apply output")
