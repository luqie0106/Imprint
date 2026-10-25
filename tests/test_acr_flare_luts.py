"""Instruction-ordered LUT and bounded formal flare-source regression tests."""

from __future__ import annotations

import ctypes as C
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
CORE_SOURCE = NATIVE / "src" / "acr_flare.cpp"
CORE_HEADER = NATIVE / "src" / "acr_flare.hpp"
TABLE_BRIDGE = NATIVE / "tools" / "acr_flare_tables_bridge.cpp"
CORE_OBJECT = NATIVE / "build" / "CMakeFiles" / "imprint_renderer.dir" / "src" / "acr_flare.cpp.o"
FORMAL_API = NATIVE / "src" / "acr_flare_api.cpp"
PUBLIC_HEADER = NATIVE / "include" / "imprint_renderer.h"
FORMAL_LIBRARY = NATIVE / "build" / (
    "libimprint_renderer.0.1.0.dylib" if sys.platform == "darwin"
    else "libimprint_renderer.0.1.0.so"
)

sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
sys.path.insert(0, str(ROOT / "src"))
import flare_lut_oracle as oracle  # noqa: E402
import native_renderer as native  # noqa: E402


U16 = C.c_uint16
U16_PTR = C.POINTER(U16)
F64 = C.c_double
F64_PTR = C.POINTER(F64)
SIZE = C.c_size_t
SIZE_MAX = SIZE(-1).value
TABLE_SIZE = 65536
TABLE_WITH_PADDING = TABLE_SIZE + 4
SENTINEL = np.uint16(0xD39A)
FLOAT_SENTINEL = -753.25

OK = 0
INVALID_ARGUMENT = 1
SIZE_OVERFLOW = 2

BLACKS = (0, 1, 1028, 4100, 32768, 65534)
KINDS = (0, 1, 2)


class FlareResult(C.Structure):
    _fields_ = [
        ("mean", F64),
        ("flare", F64),
        ("epsilon", F64),
        ("last_width", SIZE),
        ("last_height", SIZE),
        ("level_count", C.c_uint32),
    ]


def _u16_ptr(array: np.ndarray | None) -> U16_PTR:
    if array is None:
        return C.cast(C.c_void_p(0), U16_PTR)
    return array.ctypes.data_as(U16_PTR)


def _f64_ptr(array: np.ndarray | None) -> F64_PTR:
    if array is None:
        return C.cast(C.c_void_p(0), F64_PTR)
    return array.ctypes.data_as(F64_PTR)


def _binary64_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def _assert_fresh_object() -> Path:
    assert CORE_OBJECT.is_file(), f"rebuild renderer: missing official flare object {CORE_OBJECT}"
    assert CORE_OBJECT.stat().st_mtime_ns >= max(
        CORE_SOURCE.stat().st_mtime_ns, CORE_HEADER.stat().st_mtime_ns
    ), f"rebuild renderer: stale official flare object {CORE_OBJECT}"
    return CORE_OBJECT


class TableBridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.build_fn = self.library.acr_test_flare_integer_lut
        self.build_fn.argtypes = [U16, C.c_int, U16_PTR, SIZE]
        self.build_fn.restype = C.c_int

        self.calculate_fn = self.library.acr_test_flare_calculate
        self.calculate_fn.argtypes = [
            U16_PTR, SIZE, SIZE, SIZE, SIZE, SIZE, SIZE, U16,
            C.c_int, C.c_int, C.POINTER(FlareResult), U16_PTR, SIZE,
        ]
        self.calculate_fn.restype = C.c_int

    def build_raw(self, black: int, kind: int, output: U16_PTR, count: int) -> int:
        return int(self.build_fn(black, kind, output, count))


@pytest.fixture(scope="module", params=("source", "official_object"), ids=("source", "official_object"))
def table_bridge(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> TableBridge:
    assert TABLE_BRIDGE.is_file(), f"missing flare LUT C ABI bridge: {TABLE_BRIDGE}"
    if request.param == "source":
        inputs = [TABLE_BRIDGE, CORE_SOURCE]
    else:
        inputs = [TABLE_BRIDGE, _assert_fresh_object()]
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the flare LUT bridge"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_flare_lut_" + request.param) / ("libflare_lut" + suffix)
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
    return TableBridge(output)


@pytest.fixture(scope="module")
def formal_library() -> tuple[C.CDLL, str]:
    assert FORMAL_LIBRARY.is_file(), f"build the formal native renderer library: missing {FORMAL_LIBRARY}"
    required = (FORMAL_API, CORE_SOURCE, CORE_HEADER, PUBLIC_HEADER)
    assert FORMAL_LIBRARY.stat().st_mtime_ns >= max(path.stat().st_mtime_ns for path in required), (
        f"rebuild the formal native renderer library: stale artifact {FORMAL_LIBRARY}"
    )
    library = C.CDLL(str(FORMAL_LIBRARY))
    native._configure_library(library)
    assert hasattr(library, "im_native_acr_flare_u16_rgb"), (
        "formal native library is missing im_native_acr_flare_u16_rgb"
    )
    return library, str(FORMAL_LIBRARY)


def _build_table(bridge: TableBridge, black: int, kind: int) -> tuple[int, np.ndarray]:
    output = np.full(TABLE_WITH_PADDING, SENTINEL, dtype=np.uint16)
    status = bridge.build_raw(black, kind, _u16_ptr(output), TABLE_SIZE)
    return status, output


def _formal_call(
    library: C.CDLL,
    source: np.ndarray | None,
    controls: np.ndarray,
    *,
    output: np.ndarray | None = None,
    output_count: int = 6,
    source_count: int | None = None,
    source_pointer: U16_PTR | None = None,
) -> tuple[int, np.ndarray]:
    if output is None:
        output = np.full(6, FLOAT_SENTINEL, dtype=np.float64)
    if source_pointer is None:
        source_pointer = _u16_ptr(source)
    if source_count is None:
        source_count = 0 if source is None else int(source.size)
    function = library.im_native_acr_flare_u16_rgb
    height = 0 if source is None else int(source.shape[0])
    width = 0 if source is None else int(source.shape[1])
    status = function(
        width,
        height,
        source_pointer,
        source_count,
        _f64_ptr(controls),
        int(controls.size),
        _f64_ptr(output),
        output_count,
    )
    return int(status), output


@pytest.mark.parametrize("black", BLACKS)
@pytest.mark.parametrize("kind", KINDS)
def test_every_uint16_table_entry_matches_instruction_oracle(
    table_bridge: TableBridge, black: int, kind: int
) -> None:
    status, actual = _build_table(table_bridge, black, kind)
    assert status == OK, (black, kind)
    np.testing.assert_array_equal(actual[:TABLE_SIZE], oracle.build_u16_table(black, kind))
    assert np.all(actual[TABLE_SIZE:] == SENTINEL)


@pytest.mark.parametrize(
    ("black", "index", "expected"),
    ((4100, 16360, 4096), (0, 15815, 5226)),
)
def test_fixed_native_inverse_fma_rounding_regressions(
    table_bridge: TableBridge, black: int, index: int, expected: int
) -> None:
    status, actual = _build_table(table_bridge, black, 2)
    assert status == OK
    assert int(actual[index]) == expected


def test_known_fmaf_probe_differs_from_split_multiply_add() -> None:
    # Inverse/black-zero special case at x=1/4096.  The separate operations
    # differ by eight binary32 ULPs at this node; keep both observed values
    # fixed so a later refactor cannot accidentally split the two FMAs.
    x = 1.0 / 4096.0
    actual = np.float32(oracle._sample(0, 2, x))
    assert int(actual.view(np.uint32)) == 0x37F0F0F8

    x32 = oracle.f32(x)
    z_split = oracle._add32(oracle._mul32(x32, oracle._K_INVERSE), 1.0 / 16.0)
    result_split = oracle._sub32(oracle._mul32(z_split, z_split), 1.0 / 256.0)
    assert int(np.float32(result_split).view(np.uint32)) == 0x37F0F100
    assert int(actual.view(np.uint32)) != int(np.float32(result_split).view(np.uint32))


@pytest.mark.parametrize(
    ("black", "kind", "count", "expected"),
    [
        (0, 9, TABLE_SIZE, INVALID_ARGUMENT),
        (0, 0, TABLE_SIZE - 1, INVALID_ARGUMENT),
        (65535, 0, TABLE_SIZE, INVALID_ARGUMENT),
    ],
)
def test_bad_lut_requests_preserve_output(
    table_bridge: TableBridge, black: int, kind: int, count: int, expected: int
) -> None:
    output = np.full(TABLE_SIZE, SENTINEL, dtype=np.uint16)
    before = output.view(np.uint8).copy()
    assert table_bridge.build_raw(black, kind, _u16_ptr(output), count) == expected
    np.testing.assert_array_equal(output.view(np.uint8), before)


def test_null_lut_output_is_rejected(table_bridge: TableBridge) -> None:
    assert table_bridge.build_raw(0, 0, _u16_ptr(None), TABLE_SIZE) == INVALID_ARGUMENT


def test_lut_alignment_and_address_overflow_preserve_output(table_bridge: TableBridge) -> None:
    storage = np.full(TABLE_SIZE * 2 + 1, 0xA7, dtype=np.uint8)
    before = storage.copy()
    unaligned = C.cast(C.c_void_p(storage.ctypes.data + 1), U16_PTR)
    assert table_bridge.build_raw(0, 0, unaligned, TABLE_SIZE) == INVALID_ARGUMENT
    np.testing.assert_array_equal(storage, before)

    invalid_address = C.cast(C.c_void_p(SIZE_MAX - 15), U16_PTR)
    assert table_bridge.build_raw(0, 0, invalid_address, TABLE_SIZE) == SIZE_OVERFLOW
    np.testing.assert_array_equal(storage, before)


def test_analytic_constant_small_formal_source_path(formal_library: tuple[C.CDLL, str]) -> None:
    library, _ = formal_library
    source = np.empty((3, 5, 3), dtype=np.uint16)
    source[:] = (37, 32768, 65534)
    controls = np.asarray([1, 1, 2, 1, 0, 1, 1], dtype=np.float64)
    status, output = _formal_call(library, source, controls)
    assert status == 0

    mean = ((37.0 + 32768.0) + 65534.0) / 3.0 * oracle.RECIPROCAL_65535
    flare = min(max(mean / 1048576.0, 0.0), 1.0 / 4096.0)
    np.testing.assert_array_equal(
        output,
        [mean, flare, flare, source.shape[1], source.shape[0], 0.0],
    )


def test_fixed_reciprocal_normalization_rounding_regression(
    table_bridge: TableBridge,
) -> None:
    # Raw 00aef4b8 performs FDIV(1, 65535), then 00aef4bc FMUL(mean, reciprocal).
    # For the constant RGB16 value 33 this differs by one binary64 ULP from
    # dividing the final mean directly by 65535. Pin both encodings so the
    # independent source and official-object paths cannot silently agree with
    # the old operation order.
    expected_mean_bits = 0x3F40801080108010
    expected_flare_bits = 0x3E00801080108010
    legacy_mean_bits = 0x3F40801080108011
    legacy_flare_bits = 0x3E00801080108011
    source = np.full((1, 1, 3), 33, dtype=np.uint16)
    expected = oracle.reference_image(source, (0, 0, 1, 1), 0, False, True)
    assert _binary64_bits(expected["mean"]) == expected_mean_bits
    assert _binary64_bits(expected["flare"]) == expected_flare_bits
    assert _binary64_bits(expected["epsilon"]) == expected_flare_bits

    legacy_mean = ((33.0 + 33.0) + 33.0) / 3.0 / 65535.0
    legacy_flare = legacy_mean / 1048576.0
    assert _binary64_bits(legacy_mean) == legacy_mean_bits
    assert _binary64_bits(legacy_flare) == legacy_flare_bits

    result = FlareResult()
    status = table_bridge.calculate_fn(
        _u16_ptr(source),
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        C.byref(result),
        _u16_ptr(None),
        0,
    )
    assert status == OK
    assert _binary64_bits(result.mean) == expected_mean_bits
    assert _binary64_bits(result.flare) == expected_flare_bits
    assert _binary64_bits(result.epsilon) == expected_flare_bits
    assert (result.last_width, result.last_height, result.level_count) == (1, 1, 0)


def test_fixed_reciprocal_normalization_rounding_formal_api_regression(
    formal_library: tuple[C.CDLL, str],
) -> None:
    library, _ = formal_library
    source = np.full((1, 1, 3), 33, dtype=np.uint16)
    controls = np.asarray([0, 0, 1, 1, 0, 0, 1], dtype=np.float64)
    status, output = _formal_call(library, source, controls)
    assert status == OK

    # These are hard-coded from the raw FDIV->FMUL operation order, not from
    # the C++ implementation or the oracle.
    expected = np.asarray([
        struct.unpack(">d", bytes.fromhex("3f40801080108010"))[0],
        struct.unpack(">d", bytes.fromhex("3e00801080108010"))[0],
        struct.unpack(">d", bytes.fromhex("3e00801080108010"))[0],
        1.0,
        1.0,
        0.0,
    ], dtype=np.float64)
    np.testing.assert_array_equal(output, expected)
    assert tuple(_binary64_bits(value) for value in output[:3]) == (
        0x3F40801080108010,
        0x3E00801080108010,
        0x3E00801080108010,
    )


def test_formal_small_crop_uses_full_actual_image_and_black_lut(
    formal_library: tuple[C.CDLL, str]
) -> None:
    library, _ = formal_library
    rng = np.random.default_rng(20261021)
    source = rng.integers(0, 65536, size=(91, 83, 3), dtype=np.uint16)
    before = source.copy()
    crop = (23, 11, 17, 29)
    controls = np.asarray([*crop, 4100, 1, 0], dtype=np.float64)
    expected = oracle.reference_image(source, crop, 4100, True, False)

    status, output = _formal_call(library, source, controls)
    assert status == 0
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(
        output,
        [expected["mean"], expected["flare"], expected["epsilon"],
         expected["last_width"], expected["last_height"], expected["level_count"]],
    )


def test_formal_odd_nonzero_origin_pyramid_matches_independent_reference(
    formal_library: tuple[C.CDLL, str]
) -> None:
    library, _ = formal_library
    rng = np.random.default_rng(20261022)
    source = rng.integers(0, 65536, size=(129, 149, 3), dtype=np.uint16)
    before = source.copy()
    crop = (61, 17, 87, 103)
    controls = np.asarray([*crop, 1028, 1, 1], dtype=np.float64)
    expected = oracle.reference_image(source, crop, 1028, True, True)

    status, output = _formal_call(library, source, controls)
    assert status == 0
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(
        output,
        [expected["mean"], expected["flare"], expected["epsilon"],
         expected["last_width"], expected["last_height"], expected["level_count"]],
    )


def test_formal_failures_leave_output_and_input_unchanged(
    formal_library: tuple[C.CDLL, str]
) -> None:
    library, _ = formal_library
    source = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    controls = np.asarray([0, 0, 5, 4, 0, 0, 0], dtype=np.float64)
    output = np.full(6, FLOAT_SENTINEL, dtype=np.float64)
    output_before = output.copy()
    source_before = source.copy()

    bad_controls = controls.copy()
    bad_controls[4] = 65535
    status, _ = _formal_call(library, source, bad_controls, output=output)
    assert status == 1
    np.testing.assert_array_equal(output.view(np.uint8), output_before.view(np.uint8))

    status, _ = _formal_call(library, source, controls, output=output, output_count=5)
    assert status == 1
    np.testing.assert_array_equal(output.view(np.uint8), output_before.view(np.uint8))

    status, _ = _formal_call(library, None, controls, output=output, source_count=source.size)
    assert status == 1
    np.testing.assert_array_equal(output.view(np.uint8), output_before.view(np.uint8))

    aliased_output = C.cast(C.c_void_p(source.ctypes.data), F64_PTR)
    status = library.im_native_acr_flare_u16_rgb(
        source.shape[1], source.shape[0], _u16_ptr(source), source.size,
        _f64_ptr(controls), controls.size, aliased_output, 6,
    )
    assert status == 1
    np.testing.assert_array_equal(source, source_before)
