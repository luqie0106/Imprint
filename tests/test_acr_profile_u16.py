"""Mode-1 unsigned wrapper parity for the recovered SDR HSM profile stage."""

from __future__ import annotations

import ctypes as C
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SUPPORT = ROOT / "tests" / "acr_support"
U16_SOURCE = NATIVE / "src" / "acr_profile_u16.cpp"
U16_HEADER = U16_SOURCE.with_suffix(".hpp")
STAGE_SOURCE = NATIVE / "src" / "acr_profile_stage.cpp"
STAGE_HEADER = STAGE_SOURCE.with_suffix(".hpp")
TABLE_SOURCE = NATIVE / "src" / "acr_profile_table.cpp"
TABLE_HEADER = TABLE_SOURCE.with_suffix(".hpp")
U16_BRIDGE = NATIVE / "tools" / "acr_profile_u16_bridge.cpp"
FLOAT_BRIDGE = NATIVE / "tools" / "acr_profile_stage_bridge.cpp"
PUBLIC_HEADER = NATIVE / "include" / "imprint_renderer.h"
API_SOURCE = NATIVE / "src" / "acr_profile_u16_api.cpp"

OBJECTS = (
    NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_profile_u16.cpp.o",
    NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_profile_stage.cpp.o",
    NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_profile_table.cpp.o",
)
FORMAL_LIBRARY = NATIVE / "build" / (
    "libimprint_renderer.dylib" if sys.platform == "darwin" else "libimprint_renderer.so"
)

sys.path.insert(0, str(SUPPORT))
sys.path.insert(0, str(ROOT / "src"))
import profile_bulk_oracle as oracle  # noqa: E402
import native_renderer as native  # noqa: E402


U16 = C.c_uint16
U16_PTR = C.POINTER(U16)
F32_PTR = C.POINTER(C.c_float)
SIZE = C.c_size_t
SIZE_MAX = SIZE(-1).value
SENTINEL = np.uint16(0xA57C)

OK = 0
INVALID_ARGUMENT = 1
INVALID_TABLE = 2
BUFFER_TOO_SMALL = 3
UNSUPPORTED_ALIASING = 4
ARITHMETIC_OVERFLOW = 5

DIMS = ((1, 2, 1), (7, 5, 1), (5, 4, 6))
CONTROL_PAIRS = tuple(itertools.product((0.0, 1.0, 2.0), repeat=2))


def _u16_ptr(array: np.ndarray | None):
    if array is None:
        return C.cast(C.c_void_p(0), U16_PTR)
    return array.ctypes.data_as(U16_PTR)


def _float_ptr(array: np.ndarray | None):
    if array is None:
        return C.cast(C.c_void_p(0), F32_PTR)
    return array.ctypes.data_as(F32_PTR)


def _compiler() -> str:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the profile u16 bridge"
    return compiler


def _bridge_command(compiler: str, output: Path, inputs: list[Path]) -> list[str]:
    command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
               "-fno-fast-math", "-ffp-contract=off", "-fPIC",
               "-I", str(NATIVE / "src")]
    command += ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
    command += [str(path) for path in inputs]
    command += ["-o", str(output)]
    return command


def _assert_fresh(path: Path, sources: tuple[Path, ...], label: str) -> None:
    assert path.is_file(), f"rebuild renderer: missing {label}: {path}"
    newest_source = max(item.stat().st_mtime_ns for item in sources)
    assert path.stat().st_mtime_ns >= newest_source, (
        f"rebuild renderer: stale {label}: {path.name}"
    )


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.apply_fn = self.library.im_native_acr_profile_stage_sdr_u16
        self.apply_fn.argtypes = [
            U16_PTR, SIZE, SIZE, SIZE, SIZE, F32_PTR, SIZE,
            C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32,
            C.c_double, C.c_double, C.c_uint32,
            U16_PTR, SIZE, SIZE,
        ]
        self.apply_fn.restype = C.c_int
        self.float_fn = self.library.acr_test_profile_stage
        self.float_fn.argtypes = [
            F32_PTR, SIZE, F32_PTR, SIZE, SIZE, SIZE, SIZE, C.c_uint,
            C.c_double, C.c_double, F32_PTR, SIZE,
        ]
        self.float_fn.restype = C.c_int

    def apply_raw(self, source, src_count, src_stride, height, width,
                  table, table_values, dims, encoding, gain, amount,
                  pixel_type, destination, dst_count, dst_stride,
                  *, null_source=False, null_table=False, null_destination=False):
        h, s, v = dims
        return int(self.apply_fn(
            _u16_ptr(None if null_source else source), src_count, src_stride,
            height, width, _float_ptr(None if null_table else table), table_values,
            h, s, v, encoding, gain, amount, pixel_type,
            _u16_ptr(None if null_destination else destination), dst_count,
            dst_stride,
        ))

    def apply_image(self, source, table, dims, encoding=0, gain=1.0,
                    amount=1.0, pixel_type=3, *, output=None):
        height, width, _ = source.shape
        row_samples = width * 3
        if output is None:
            output = np.full(source.size, SENTINEL, dtype=np.uint16)
        status = self.apply_raw(
            source, source.size, row_samples, height, width,
            table, table.size, dims, encoding, gain, amount, pixel_type,
            output, output.size, row_samples,
        )
        return status, output.reshape(source.shape)

    def apply_float(self, source, table, dims, encoding, gain, amount):
        output = np.full(source.shape, np.float32(-123.25), dtype=np.float32)
        h, s, v = dims
        status = self.float_fn(
            _float_ptr(source), source.size, _float_ptr(table), table.size,
            h, s, v, encoding, gain, amount,
            _float_ptr(output), output.size,
        )
        return int(status), output


def _bridge_inputs(kind: str) -> list[Path]:
    if kind == "source":
        return [U16_BRIDGE, FLOAT_BRIDGE, U16_SOURCE, STAGE_SOURCE, TABLE_SOURCE]
    freshness = (U16_SOURCE, U16_HEADER, STAGE_SOURCE, STAGE_HEADER,
                 TABLE_SOURCE, TABLE_HEADER)
    for obj in OBJECTS:
        _assert_fresh(obj, freshness, "official profile-stage object")
    return [U16_BRIDGE, FLOAT_BRIDGE, *OBJECTS]


@pytest.fixture(scope="module", params=("source", "fresh_objects"),
                ids=("source_bridge", "fresh_object_bridge"))
def bridge(request, tmp_path_factory):
    kind = request.param
    inputs = _bridge_inputs(kind)
    assert U16_BRIDGE.is_file(), f"missing profile u16 bridge: {U16_BRIDGE}"
    assert FLOAT_BRIDGE.is_file(), f"missing profile float bridge: {FLOAT_BRIDGE}"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_profile_u16_" + kind) / ("libprofileu16" + suffix)
    command = _bridge_command(_compiler(), output, inputs)
    subprocess.run(command, check=True, capture_output=True, text=True)
    return Bridge(output)


@pytest.fixture(scope="module")
def formal_library():
    freshness = (U16_SOURCE, U16_HEADER, STAGE_SOURCE, STAGE_HEADER,
                 TABLE_SOURCE, TABLE_HEADER, API_SOURCE, PUBLIC_HEADER)
    _assert_fresh(FORMAL_LIBRARY, freshness, "formal renderer library")
    library = C.CDLL(str(FORMAL_LIBRARY))
    native._configure_library(library)
    return library


def _pixels_with_edges(seed: int, count: int = 48) -> np.ndarray:
    edge_rows = np.asarray([
        [0, 1, 32767], [32768, 65534, 65535],
        [65535, 0, 32768], [1, 32767, 65534],
        [0, 0, 0], [65535, 65535, 65535],
    ], dtype=np.uint16)
    rng = np.random.default_rng(seed)
    random_rows = rng.integers(0, 65536, size=(count, 3), dtype=np.uint16)
    return np.ascontiguousarray(np.concatenate((edge_rows, random_rows)))


def _same_bits(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    actual = np.ascontiguousarray(actual)
    expected = np.ascontiguousarray(expected)
    assert actual.shape == expected.shape, f"{label}: {actual.shape} != {expected.shape}"
    if actual.dtype == np.float32:
        differing = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
    else:
        differing = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
    assert differing.size == 0, (
        f"{label}: first mismatch at flat index {int(differing[0])}: "
        f"actual={actual.reshape(-1)[differing[0]]!r}, "
        f"expected={expected.reshape(-1)[differing[0]]!r}"
    )


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("encoding", (0, 1))
@pytest.mark.parametrize("gain,amount", CONTROL_PAIRS)
def test_mode1_u16_source_and_fresh_object_bridges_match_independent_oracle(
    bridge, dims, encoding, gain, amount,
):
    table = oracle.make_table(*dims, seed=20261021 + sum(dims)).reshape(
        dims[2], dims[0], dims[1], 3
    )
    source_rows = _pixels_with_edges(0x016D2021 + sum(dims) + encoding)
    source = source_rows.reshape(-1, 1, 3).copy()
    before = source.copy()

    for pixel_type in (3, 8):
        encoded_source = source if pixel_type == 3 else np.bitwise_xor(source, np.uint16(0x8000))
        encoded_before = encoded_source.copy()
        status, actual = bridge.apply_image(
            encoded_source, table, dims, encoding, gain, amount, pixel_type
        )
        expected = oracle.reference_u16(
            encoded_source, table, dims, encoding, gain, amount, pixel_type
        )
        assert status == OK, (dims, encoding, gain, amount, pixel_type, status)
        _same_bits(actual, expected, "mode1 wrapper output")
        _same_bits(encoded_source, encoded_before, "mode1 wrapper input preserved")

    _same_bits(source, before, "base random RGB input preserved")


def test_type3_and_type8_decode_to_identical_rgb_and_outputs_are_xor_related(bridge):
    dims = (5, 4, 6)
    table = oracle.make_table(*dims, seed=0x21A10).reshape(6, 5, 4, 3)
    type3 = _pixels_with_edges(0x8F0321).reshape(-1, 1, 3)
    type8 = np.bitwise_xor(type3, np.uint16(0x8000))
    decoded3 = oracle.decode_u16(type3, 3)
    decoded8 = oracle.decode_u16(type8, 8)
    _same_bits(decoded3, decoded8, "type 3/type 8 decoded float samples")

    status3, output3 = bridge.apply_image(type3, table, dims, 1, 2.0, 1.0, 3)
    status8, output8 = bridge.apply_image(type8, table, dims, 1, 2.0, 1.0, 8)
    assert status3 == status8 == OK
    _same_bits(output8, np.bitwise_xor(output3, np.uint16(0x8000)),
               "type 8 output XOR relation")


@pytest.mark.parametrize("pixel_type", (3, 8))
def test_all_65536_gray_words_pass_through_identity_hsm(bridge, pixel_type):
    dims = (1, 2, 1)
    table = np.tile(np.asarray([0.0, 1.0, 1.0], dtype=np.float32), 2).reshape(1, 1, 2, 3)
    values = np.arange(65536, dtype=np.uint16)
    source = np.repeat(values[:, None, None], 3, axis=2)
    if pixel_type == 8:
        source ^= np.uint16(0x8000)
    before = source.copy()
    status, actual = bridge.apply_image(source, table, dims, 0, 1.0, 1.0, pixel_type)
    assert status == OK
    _same_bits(actual, before, f"all gray words type {pixel_type}")
    _same_bits(source, before, "all gray source preserved")


def test_fixed_wrapper_rounding_and_offset_word_counterexample(bridge):
    """Catch a direct float result or a truncating low-word store."""
    dims = (1, 2, 1)
    table = np.tile(np.asarray([0.0, 1.0, 0.5], dtype=np.float32), 2).reshape(1, 1, 2, 3)
    type3_source = np.full((1, 1, 3), 32767, dtype=np.uint16)
    type8_source = np.bitwise_xor(type3_source, np.uint16(0x8000))

    decoded = oracle.decode_u16(type3_source, 3)
    mapped = oracle.reference_float_image(decoded, table, dims, 0, 1.0, 1.0)
    unbiased = oracle.mul32(mapped[0, 0, 0], oracle.U16_FROM_F32_SCALE)
    assert int(float(unbiased)) == 16383
    assert int(oracle.reference_u16(type3_source, table, dims, 0)[0, 0, 0]) == 16384

    status3, output3 = bridge.apply_image(type3_source, table, dims, 0, 1, 1, 3)
    status8, output8 = bridge.apply_image(type8_source, table, dims, 0, 1, 1, 8)
    assert status3 == status8 == OK
    assert np.all(output3 == 16384)
    assert np.all(output8 == (16384 ^ 0x8000))
    assert not np.array_equal(output8, output3)

def test_float_bulk_reference_rounding_and_hsv_sort_tie_boundaries(bridge):
    # A fused witness makes a separately rounded multiply/add observably wrong.
    a = np.float32(1.0 + 2.0 ** -23)
    b = np.float32(1.0 - 2.0 ** -23)
    assert oracle.fma32(a, b, -1.0).view(np.uint32) == np.uint32(0xA8800000)
    assert oracle.add32(oracle.mul32(a, b), -1.0).view(np.uint32) == np.uint32(0)

    half = np.float32(0.5)
    below = np.nextafter(half, np.float32(0.0))
    above = np.nextafter(half, np.float32(1.0))
    boundary_pixels = np.asarray([
        [0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [half, half, 0], [half, 0, half], [0, half, half],
        [half, below, 0], [half, above, 0], [half, 0, below], [half, 0, above],
        [0, half, below], [0, half, above], [half, below, below],
        [below, half, below], [below, below, half],
    ], dtype=np.float32)
    hsv_bits = [tuple(int(word) for word in np.asarray(
        oracle.hsv_from_rgb(*pixel), dtype=np.float32
    ).view(np.uint32)) for pixel in boundary_pixels[5:8]]
    assert hsv_bits == [
        (0x3F800000, 0x3F800000, 0x3F000000),
        (0x40A00000, 0x3F800000, 0x3F000000),
        (0x40400000, 0x3F800000, 0x3F000000),
    ]

    rng = np.random.default_rng(0x5EED1021)
    pixels = np.ascontiguousarray(np.concatenate((
        boundary_pixels,
        rng.random((64 - len(boundary_pixels), 3), dtype=np.float32),
    )))
    controls = ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (0.5, 0.5))
    cases = 0
    for dims in DIMS:
        table = oracle.make_table(*dims, seed=0x21D4 + sum(dims))
        for encoding in (0, 1):
            for gain, amount in controls:
                status, actual = bridge.apply_float(
                    pixels, table, dims, encoding, gain, amount
                )
                expected = oracle.reference_float_image(
                    pixels, table, dims, encoding, gain, amount
                )
                assert status == OK, (dims, encoding, gain, amount, status)
                _same_bits(actual, expected, "independent float bulk oracle bits")
                cases += 1
    assert cases == 24


def test_row_padding_source_preservation_and_exact_in_place(bridge):
    dims = (7, 5, 1)
    table = oracle.make_table(*dims, seed=0x100021).reshape(1, 7, 5, 3)
    height, width, src_stride, dst_stride = 3, 2, 9, 10
    row_samples = width * 3
    rng = np.random.default_rng(0xF00D21)
    source = np.full(height * src_stride, 0xE11E, dtype=np.uint16)
    active = rng.integers(0, 65536, size=(height, width, 3), dtype=np.uint16)
    for row in range(height):
        source[row * src_stride:row * src_stride + row_samples] = active[row].reshape(-1)
    source_before = source.copy()
    output = np.full(height * dst_stride, SENTINEL, dtype=np.uint16)
    output_before = output.copy()
    status = bridge.apply_raw(
        source, source.size, src_stride, height, width,
        table, table.size, dims, 0, 1.0, 1.0, 3,
        output, output.size, dst_stride,
    )
    assert status == OK
    expected_active = oracle.reference_u16(active, table, dims, 0, 1, 1, 3)
    for row in range(height):
        dst_start = row * dst_stride
        _same_bits(output[dst_start:dst_start + row_samples], expected_active[row].reshape(-1),
                   "padded active output")
        _same_bits(output[dst_start + row_samples:dst_start + dst_stride],
                   output_before[dst_start + row_samples:dst_start + dst_stride],
                   "destination row padding")
    _same_bits(source, source_before, "strided source unchanged")

    inplace = source_before.copy()
    inplace_before = inplace.copy()
    status = bridge.apply_raw(
        inplace, inplace.size, src_stride, height, width,
        table, table.size, dims, 0, 1, 1, 3,
        inplace, inplace.size, src_stride,
    )
    assert status == OK
    for row in range(height):
        start = row * src_stride
        _same_bits(inplace[start:start + row_samples], expected_active[row].reshape(-1),
                   "exact in-place active samples")
        _same_bits(inplace[start + row_samples:start + src_stride],
                   inplace_before[start + row_samples:start + src_stride],
                   "exact in-place row padding")


def test_overlap_count_stride_alignment_and_huge_span_rejections_are_atomic(bridge):
    dims = (1, 2, 1)
    table = np.tile(np.asarray([0.0, 1.0, 1.0], dtype=np.float32), 2)
    one_row = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.uint16)
    output = np.full(12, SENTINEL, dtype=np.uint16)
    before = output.view(np.uint8).copy()

    # Shifted ranges and equal bases with different declared counts both overlap.
    backing = np.arange(24, dtype=np.uint16)
    original = backing.copy()
    status = bridge.apply_raw(
        backing[:12], 12, 6, 1, 2, table, table.size, dims, 0, 1, 1, 3,
        backing[1:13], 12, 6,
    )
    assert status == UNSUPPORTED_ALIASING
    _same_bits(backing, original, "partial overlap atomic")
    status = bridge.apply_raw(
        backing[:12], 12, 6, 1, 2, table, table.size, dims, 0, 1, 1, 3,
        backing[:11], 11, 6,
    )
    assert status == UNSUPPORTED_ALIASING
    _same_bits(backing, original, "same-base unequal-span overlap atomic")

    assert bridge.apply_raw(
        one_row, 5, 6, 1, 2, table, table.size, dims, 0, 1, 1, 3,
        output, output.size, 6,
    ) == BUFFER_TOO_SMALL
    _same_bits(output.view(np.uint8), before, "short source count atomic")
    assert bridge.apply_raw(
        one_row, one_row.size, 6, 1, 2, table, table.size, dims, 0, 1, 1, 3,
        output, 5, 6,
    ) == BUFFER_TOO_SMALL
    _same_bits(output.view(np.uint8), before, "short destination count atomic")
    assert bridge.apply_raw(
        one_row, one_row.size, 5, 1, 2, table, table.size, dims, 0, 1, 1, 3,
        output, output.size, 6,
    ) == INVALID_ARGUMENT
    _same_bits(output.view(np.uint8), before, "short source stride atomic")
    assert bridge.apply_raw(
        one_row, one_row.size, 6, 1, 2, table, table.size, dims, 0, 1, 1, 3,
        output, output.size, 5,
    ) == INVALID_ARGUMENT
    _same_bits(output.view(np.uint8), before, "short destination stride atomic")

    raw = np.zeros(16, dtype=np.uint8)
    misaligned = raw[1:15].view(np.uint16)
    assert bridge.apply_raw(
        misaligned, misaligned.size, 3, 1, 1, table, table.size, dims, 0, 1, 1, 3,
        output, output.size, 3,
    ) == INVALID_ARGUMENT
    _same_bits(output.view(np.uint8), before, "misaligned input atomic")
    misaligned_output = raw[1:15].view(np.uint16)
    assert bridge.apply_raw(
        one_row, one_row.size, 6, 1, 2, table, table.size, dims, 0, 1, 1, 3,
        misaligned_output, misaligned_output.size, 6,
    ) == INVALID_ARGUMENT
    _same_bits(output.view(np.uint8), before, "misaligned output leaves aligned sentinel atomic")

    assert bridge.apply_raw(
        one_row, SIZE_MAX, 6, 1, 2, table, table.size, dims, 0, 1, 1, 3,
        output, output.size, 6,
    ) == ARITHMETIC_OVERFLOW
    _same_bits(output.view(np.uint8), before, "oversized declared span atomic")


def test_invalid_table_controls_and_pixel_type_leave_every_output_byte_unchanged(bridge):
    dims = (1, 2, 1)
    table = np.tile(np.asarray([0.0, 1.0, 1.0], dtype=np.float32), 2)
    source = np.asarray([100, 200, 300], dtype=np.uint16)
    output = np.full(9, SENTINEL, dtype=np.uint16)
    before = output.view(np.uint8).copy()

    nan_table = table.copy()
    nan_table[1] = np.nan
    assert bridge.apply_raw(
        source, source.size, 3, 1, 1, nan_table, nan_table.size, dims, 0, 1, 1, 3,
        output, output.size, 3,
    ) == INVALID_TABLE
    _same_bits(output.view(np.uint8), before, "NaN table atomic")
    assert bridge.apply_raw(
        source, source.size, 3, 1, 1, table, table.size - 3, dims, 0, 1, 1, 3,
        output, output.size, 3,
    ) == INVALID_TABLE
    _same_bits(output.view(np.uint8), before, "table count mismatch atomic")

    invalid_cases = (
        (2, 1.0, 1.0, 3),
        (0, float("nan"), 1.0, 3),
        (0, 1.0, 3.0, 3),
        (0, 1.0, 1.0, 7),
    )
    for encoding, gain, amount, pixel_type in invalid_cases:
        status = bridge.apply_raw(
            source, source.size, 3, 1, 1, table, table.size, dims,
            encoding, gain, amount, pixel_type, output, output.size, 3,
        )
        assert status == INVALID_ARGUMENT, (encoding, gain, amount, pixel_type, status)
        _same_bits(output.view(np.uint8), before, "invalid control atomic")

    assert bridge.apply_raw(
        source, source.size, 3, 1, 1, None, 0, dims, 0, 1, 1, 3,
        output, output.size, 3, null_table=True,
    ) == INVALID_TABLE
    _same_bits(output.view(np.uint8), before, "null table atomic")


def test_formal_capi_and_python_adapter_match_reference_on_strided_input(
    formal_library, monkeypatch,
):
    dims = (5, 4, 6)
    table = oracle.make_table(*dims, seed=0x53545249).reshape(6, 5, 4, 3)
    backing = np.arange(4 * 10 * 3, dtype=np.uint16).reshape(4, 10, 3) * np.uint16(173)
    backing_before = backing.copy()
    type3_source = backing[:, ::2, :]
    encoded_backing = backing.copy()
    encoded_backing ^= np.uint16(0x8000)
    source = encoded_backing[:, ::2, :]
    assert not source.flags.c_contiguous
    before = source.copy()
    expected = oracle.reference_u16(source, table, dims, 1, 2.0, 0.75, 8)

    function = formal_library.im_native_acr_profile_stage_sdr_u16
    packed_source = np.ascontiguousarray(source)
    output = np.full(packed_source.shape, SENTINEL, dtype=np.uint16)
    h, s, v = dims
    status = function(
        _u16_ptr(packed_source), packed_source.size, packed_source.shape[1] * 3,
        packed_source.shape[0], packed_source.shape[1], _float_ptr(table), table.size,
        h, s, v, 1, 2.0, 0.75, 8,
        _u16_ptr(output), output.size, output.shape[1] * 3,
    )
    assert status == 0
    _same_bits(output, expected, "formal C API strided-source result")

    monkeypatch.setattr(native, "_get_library", lambda: (formal_library, str(FORMAL_LIBRARY)))
    adapter_output = native.native_acr_profile_stage_sdr_u16(
        source, table, dims, 1, lookup_gain=2.0, amount=0.75, pixel_type=8
    )
    _same_bits(adapter_output, expected, "Python adapter strided-input result")
    _same_bits(source, before, "Python adapter does not mutate strided input")
    _same_bits(backing, backing_before, "original backing source unchanged")


def test_python_adapter_fails_when_native_u16_symbol_is_missing(monkeypatch):
    dims = (1, 2, 1)
    table = np.tile(np.asarray([0.0, 1.0, 1.0], dtype=np.float32), 2).reshape(1, 1, 2, 3)
    source = np.zeros((1, 1, 3), dtype=np.uint16)
    monkeypatch.setattr(native, "_get_library", lambda: (object(), "missing-u16-symbol"))
    with pytest.raises(native.NativeRendererError, match="does not provide integer ACR profile stage"):
        native.native_acr_profile_stage_sdr_u16(source, table, dims, 0)

