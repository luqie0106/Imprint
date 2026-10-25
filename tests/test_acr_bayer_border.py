from __future__ import annotations

import ctypes as C
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_bayer_border.cpp"
HEADER = NATIVE / "src/acr_bayer_border.hpp"
BRIDGE_SOURCE = NATIVE / "tools/acr_bayer_border_bridge.cpp"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_bayer_border.cpp.o"
U16 = C.POINTER(C.c_uint16)
I32 = C.POINTER(C.c_int32)
SIZE = C.c_size_t
SIZE_MAX = SIZE(-1).value
PTRDIFF_MAX = sys.maxsize
UINTPTR_MAX = C.c_void_p(-1).value
SENTINEL = np.uint16(0xA55A)

_SPEC = importlib.util.spec_from_file_location(
    "bayer_border_oracle", ROOT / "tests/acr_support/bayer_border_oracle.py"
)
oracle = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(oracle)


def pointer_at(address: int):
    return C.cast(C.c_void_p(address), U16)


def array_pointer(array: np.ndarray):
    return array.ctypes.data_as(U16)


def cfa_sample(y: int, x: int, row: int, column: int, seed: int = 0) -> np.uint16:
    """Tag each sample with its global CFA site and a coordinate marker."""
    payload = (row * 257 + column * 31 + seed * 113 + 1) & 0x3FFF
    return np.uint16(((y & 1) << 15) | ((x & 1) << 14) | payload)


def make_source(source_rect, *, stride: int | None = None,
                seed: int = 0) -> np.ndarray:
    top, left, bottom, right = map(int, source_rect)
    height, width = bottom - top, right - left
    if stride is None:
        stride = width
    rows = np.full((height, stride), np.uint16(0xD00D), dtype=np.uint16)
    for row in range(height):
        for column in range(width):
            rows[row, column] = cfa_sample(top + row, left + column,
                                           row, column, seed)
    return rows


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.run = self.library.acr_test_bayer_repeat2
        self.run.argtypes = [U16, SIZE, SIZE, I32, I32, U16, SIZE, SIZE]
        self.run.restype = C.c_int

    def execute(
        self,
        source: np.ndarray,
        source_rect,
        request_rect,
        *,
        destination: np.ndarray | None = None,
        source_count: int | None = None,
        source_stride: int | None = None,
        destination_count: int | None = None,
        destination_stride: int | None = None,
        source_pointer=None,
        destination_pointer=None,
        source_rect_array: np.ndarray | None = None,
        request_rect_array: np.ndarray | None = None,
    ):
        source = np.asarray(source)
        if source_stride is None:
            source_stride = source.shape[1] if source.ndim == 2 else 0
        if source_count is None:
            source_count = source.size
        source_rect_array = (
            np.asarray(source_rect, dtype=np.int32)
            if source_rect_array is None else source_rect_array
        )
        request_rect_array = (
            np.asarray(request_rect, dtype=np.int32)
            if request_rect_array is None else request_rect_array
        )
        qt, ql, qb, qr = map(int, request_rect_array)
        height, width = qb - qt, qr - ql
        if destination_stride is None:
            destination_stride = width
        if destination is None:
            destination = np.full((height, destination_stride), SENTINEL,
                                  dtype=np.uint16)
        else:
            destination = np.asarray(destination)
        if destination_count is None:
            destination_count = destination.size
        if source_pointer is None:
            source_pointer = array_pointer(source)
        if destination_pointer is None:
            destination_pointer = array_pointer(destination)
        status = self.run(
            source_pointer,
            source_count,
            source_stride,
            source_rect_array.ctypes.data_as(I32),
            request_rect_array.ctypes.data_as(I32),
            destination_pointer,
            destination_count,
            destination_stride,
        )
        return status, destination


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    assert SOURCE.exists() and HEADER.exists(), "missing Bayer border source/header"
    assert BRIDGE_SOURCE.exists(), "missing Bayer border test bridge source"
    if request.param == "official_object":
        assert OBJECT.exists(), "rebuild renderer: missing official Bayer border object"
        assert OBJECT.stat().st_mtime_ns >= max(
            SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns
        ), "rebuild renderer: stale official Bayer border object"
        inputs = [OBJECT]
    else:
        inputs = [SOURCE]

    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "C++ compiler required for Bayer border bridge tests"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_bayer_border_" + request.param) / (
        "libacr_bayer_border_test_bridge" + suffix
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
        str(BRIDGE_SOURCE),
        *(str(path) for path in inputs),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return Bridge(output)


def assert_matches(bridge, source, source_rect, request_rect,
                   *, destination_stride=None, seed=0):
    before = source.copy()
    if destination_stride is None:
        destination_stride = int(request_rect[3]) - int(request_rect[1]) + 3
    qt, ql, qb, qr = map(int, request_rect)
    destination = np.full((qb - qt, destination_stride), SENTINEL,
                          dtype=np.uint16)
    status, actual = bridge.execute(
        source,
        source_rect,
        request_rect,
        destination=destination,
        destination_stride=destination_stride,
    )
    expected = oracle.reference(
        source, source_rect, request_rect,
        destination_stride=destination_stride,
    )
    assert status == 0
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before, err_msg="source was modified")

    # The high bits of each sentinel encode its original, global Bayer site.
    for out_y, y in enumerate(range(qt, qb)):
        for out_x, x in enumerate(range(ql, qr)):
            sample = int(actual[out_y, out_x])
            assert ((sample >> 15) & 1) == (y & 1)
            assert ((sample >> 14) & 1) == (x & 1)
    if destination_stride > qr - ql:
        assert np.all(actual[:, qr - ql:] == SENTINEL), "row padding was written"
    return actual


def test_hand_written_4x4_to_8x8_repeats_both_samples_of_each_bayer_edge(bridge):
    source_rect = np.asarray([10, 20, 14, 24], dtype=np.int32)
    request_rect = np.asarray([8, 18, 16, 26], dtype=np.int32)
    source = np.asarray(
        [
            [0x0001, 0x4002, 0x0003, 0x4004],
            [0x8005, 0xC006, 0x8007, 0xC008],
            [0x0009, 0x400A, 0x000B, 0x400C],
            [0x800D, 0xC00E, 0x800F, 0xC010],
        ],
        dtype=np.uint16,
    )
    expected = np.asarray(
        [
            [0x0001, 0x4002, 0x0001, 0x4002, 0x0003, 0x4004, 0x0003, 0x4004],
            [0x8005, 0xC006, 0x8005, 0xC006, 0x8007, 0xC008, 0x8007, 0xC008],
            [0x0001, 0x4002, 0x0001, 0x4002, 0x0003, 0x4004, 0x0003, 0x4004],
            [0x8005, 0xC006, 0x8005, 0xC006, 0x8007, 0xC008, 0x8007, 0xC008],
            [0x0009, 0x400A, 0x0009, 0x400A, 0x000B, 0x400C, 0x000B, 0x400C],
            [0x800D, 0xC00E, 0x800D, 0xC00E, 0x800F, 0xC010, 0x800F, 0xC010],
            [0x0009, 0x400A, 0x0009, 0x400A, 0x000B, 0x400C, 0x000B, 0x400C],
            [0x800D, 0xC00E, 0x800D, 0xC00E, 0x800F, 0xC010, 0x800F, 0xC010],
        ],
        dtype=np.uint16,
    )
    actual = assert_matches(
        bridge, source, source_rect, request_rect, destination_stride=11
    )
    np.testing.assert_array_equal(actual[:, :8], expected)


def test_odd_source_dimensions_and_odd_negative_global_origin_keep_parity(bridge):
    source_rect = np.asarray([-7, 13, -2, 16], dtype=np.int32)
    request_rect = np.asarray([-13, 8, 3, 22], dtype=np.int32)
    source = make_source(source_rect, stride=8, seed=1)
    assert_matches(bridge, source, source_rect, request_rect,
                   destination_stride=19, seed=1)


@pytest.mark.parametrize(
    "source_rect,request_rect",
    (
        ((2147483645, 2147483645, 2147483647, 2147483647),
         (-2147483648, -2147483648, -2147483646, -2147483646)),
        ((-2147483648, -2147483648, -2147483646, -2147483646),
         (2147483645, 2147483645, 2147483647, 2147483647)),
    ),
)
def test_valid_small_rectangles_at_opposite_int32_extremes_keep_parity(
    bridge, source_rect, request_rect
):
    source = make_source(source_rect, stride=4, seed=7)
    assert_matches(bridge, source, source_rect, request_rect,
                   destination_stride=4, seed=7)


@pytest.mark.parametrize(
    "request_rect",
    (
        (-12, 7, 4, 11),  # entirely above
        (9, 7, 15, 11),   # entirely below
        (5, -10, 8, 6),   # entirely left
        (5, 12, 8, 18),   # entirely right
        (-12, -12, 4, 6), # entirely above-left
    ),
)
def test_requests_fully_outside_each_edge_and_corner(bridge, request_rect):
    source_rect = (5, 7, 8, 11)
    source = make_source(source_rect, stride=6, seed=2)
    assert_matches(bridge, source, source_rect, request_rect,
                   destination_stride=request_rect[3] - request_rect[1] + 2)


def test_2x2_checkerboard_survives_large_halos_on_all_sides(bridge):
    source_rect = (-3, 5, -1, 7)
    request_rect = (-27, -19, 18, 31)
    source = make_source(source_rect, stride=5, seed=3)
    actual = assert_matches(bridge, source, source_rect, request_rect,
                            destination_stride=55, seed=3)
    assert np.unique(actual[:, :50]).size == 4


def test_random_rectangles_strides_and_padding_match_coordinate_oracle(bridge):
    rng = np.random.default_rng(0xBA7E2026)
    for case in range(200):
        top = int(rng.integers(-100, 101))
        left = int(rng.integers(-100, 101))
        source_height = int(rng.integers(2, 12))
        source_width = int(rng.integers(2, 13))
        source_rect = (top, left, top + source_height, left + source_width)
        source_stride = source_width + int(rng.integers(0, 8))
        source = make_source(source_rect, stride=source_stride, seed=case)
        request_height = int(rng.integers(1, 20))
        request_width = int(rng.integers(1, 21))
        request_top = int(rng.integers(top - 30, top + source_height + 31))
        request_left = int(rng.integers(left - 30, left + source_width + 31))
        request_rect = (
            request_top,
            request_left,
            request_top + request_height,
            request_left + request_width,
        )
        destination_stride = request_width + int(rng.integers(0, 8))
        assert_matches(bridge, source, source_rect, request_rect,
                       destination_stride=destination_stride, seed=case)


def test_minimal_terminal_sample_capacities_are_valid(bridge):
    source_rect = (1, -3, 3, -1)
    request_rect = (0, -4, 4, 0)
    source = make_source(source_rect, stride=4, seed=5)
    destination = np.full((4, 5), SENTINEL, dtype=np.uint16)
    # The final logical sample is at (height-1)*stride + width - 1.
    src_terminal_count = (2 - 1) * 4 + 2
    dst_terminal_count = (4 - 1) * 5 + 4
    before = source.copy()
    status, actual = bridge.execute(
        source, source_rect, request_rect, destination=destination,
        source_count=src_terminal_count, source_stride=4,
        destination_count=dst_terminal_count, destination_stride=5,
    )
    assert status == 0
    np.testing.assert_array_equal(
        actual,
        oracle.reference(source, source_rect, request_rect,
                         destination_stride=5),
    )
    np.testing.assert_array_equal(source, before)


def _assert_rejected_unchanged(bridge, source, source_rect, request_rect,
                               *, destination=None, **kwargs):
    before_source = source.copy()
    if destination is None:
        qt, ql, qb, qr = map(int, request_rect)
        stride = kwargs.get("destination_stride", max(qr - ql, 1))
        destination = np.full((max(qb - qt, 1), stride), SENTINEL,
                              dtype=np.uint16)
    before_destination = destination.copy()
    status, actual = bridge.execute(
        source, source_rect, request_rect, destination=destination, **kwargs
    )
    assert status != 0
    np.testing.assert_array_equal(source, before_source, err_msg="source changed on error")
    np.testing.assert_array_equal(actual, before_destination,
                                  err_msg="destination changed on error")


@pytest.mark.parametrize(
    "source_rect,request_rect",
    (
        ((0, 0, 1, 2), (0, 0, 2, 2)),
        ((0, 0, 2, 1), (0, 0, 2, 2)),
        ((2, 0, 0, 2), (0, 0, 2, 2)),
        ((0, 2, 2, 0), (0, 0, 2, 2)),
        ((0, 0, 2, 2), (2, 0, 0, 2)),
        ((0, 0, 2, 2), (0, 2, 2, 0)),
        ((0, 0, 0, 2), (0, 0, 1, 1)),
    ),
)
def test_degenerate_reversed_or_empty_rectangles_reject_atomically(
    bridge, source_rect, request_rect
):
    source = np.full((4, 8), 0x1234, dtype=np.uint16)
    _assert_rejected_unchanged(bridge, source, source_rect, request_rect,
                               destination=np.full((4, 8), SENTINEL,
                                                   dtype=np.uint16))


@pytest.mark.parametrize(
    "source_rect,request_rect",
    (
        ((-2147483648, 0, 2147483647, 2), (0, 0, 2, 2)),
        ((0, -2147483648, 2, 2147483647), (0, 0, 2, 2)),
        ((0, 0, 2, 2), (-2147483648, 0, 2147483647, 2)),
        ((0, 0, 2, 2), (0, -2147483648, 2, 2147483647)),
    ),
)
def test_int32_rect_differences_larger_than_int32_reject_atomically(
    bridge, source_rect, request_rect
):
    source = np.full((2, 2), 0x2468, dtype=np.uint16)
    _assert_rejected_unchanged(bridge, source, source_rect, request_rect,
                               destination=np.full((2, 2), SENTINEL,
                                                   dtype=np.uint16))


@pytest.mark.parametrize("bad_capacity", ("source", "destination"))
def test_short_capacity_at_terminal_sample_rejects_atomically(bridge, bad_capacity):
    source_rect = (0, 0, 2, 2)
    request_rect = (0, 0, 2, 2)
    source = make_source(source_rect, stride=4)
    destination = np.full((2, 4), SENTINEL, dtype=np.uint16)
    kwargs = {"source_stride": 4, "destination_stride": 4}
    if bad_capacity == "source":
        kwargs["source_count"] = 5  # required endpoint is sample 5, inclusive
    else:
        kwargs["destination_count"] = 5
    _assert_rejected_unchanged(bridge, source, source_rect, request_rect,
                               destination=destination, **kwargs)


@pytest.mark.parametrize("bad_argument", ("source_stride", "destination_stride"))
def test_stride_shorter_than_its_logical_row_rejects_atomically(
    bridge, bad_argument
):
    source_rect = (0, 0, 2, 2)
    request_rect = (0, 0, 2, 2)
    source = make_source(source_rect)
    destination = np.full((2, 4), SENTINEL, dtype=np.uint16)
    kwargs = {bad_argument: 1}
    _assert_rejected_unchanged(bridge, source, source_rect, request_rect,
                               destination=destination, **kwargs)


@pytest.mark.parametrize("bad_argument", ("source_stride", "destination_stride"))
def test_size_max_stride_rejects_before_access(bridge, bad_argument):
    source_rect = (0, 0, 2, 2)
    request_rect = (0, 0, 2, 2)
    source = make_source(source_rect)
    kwargs = {bad_argument: SIZE_MAX}
    _assert_rejected_unchanged(bridge, source, source_rect, request_rect,
                               destination=np.full((2, 2), SENTINEL,
                                                   dtype=np.uint16), **kwargs)


@pytest.mark.parametrize("bad_argument", ("source_count", "destination_count"))
def test_size_max_sample_count_rejects_before_access(bridge, bad_argument):
    source_rect = (0, 0, 2, 2)
    request_rect = (0, 0, 2, 2)
    source = make_source(source_rect)
    kwargs = {bad_argument: SIZE_MAX}
    _assert_rejected_unchanged(bridge, source, source_rect, request_rect,
                               **kwargs)


@pytest.mark.parametrize("bad_argument", ("source_count", "destination_count"))
def test_byte_span_beyond_ptrdiff_max_rejects_before_access(bridge, bad_argument):
    source_rect = (0, 0, 2, 2)
    request_rect = (0, 0, 2, 2)
    source = make_source(source_rect)
    excessive_sample_count = PTRDIFF_MAX // np.dtype(np.uint16).itemsize + 1
    kwargs = {bad_argument: excessive_sample_count}
    _assert_rejected_unchanged(bridge, source, source_rect, request_rect,
                               **kwargs)


@pytest.mark.parametrize("which_pointer", ("source", "destination"))
def test_unaligned_u16_pointer_rejects_atomically(bridge, which_pointer):
    source_rect = (0, 0, 2, 2)
    request_rect = (0, 0, 2, 2)
    source = make_source(source_rect)
    destination_bytes = np.full(16, 0x5A, dtype=np.uint8)
    destination = destination_bytes.view(np.uint16)
    before_source = source.copy()
    before_bytes = destination_bytes.copy()
    kwargs = {}
    if which_pointer == "source":
        backing = np.full(8, 0x1234, dtype=np.uint16)
        kwargs["source_pointer"] = pointer_at(backing.ctypes.data + 1)
    else:
        kwargs["destination_pointer"] = pointer_at(destination.ctypes.data + 1)
    status, actual = bridge.execute(
        source, source_rect, request_rect, destination=destination, **kwargs
    )
    assert status != 0
    np.testing.assert_array_equal(source, before_source)
    np.testing.assert_array_equal(destination_bytes, before_bytes)
    np.testing.assert_array_equal(actual, before_bytes.view(np.uint16))


@pytest.mark.parametrize("which_pointer", ("source", "destination"))
def test_uintptr_span_wrap_rejects_atomically(bridge, which_pointer):
    source_rect = (0, 0, 2, 2)
    request_rect = (0, 0, 2, 2)
    source = make_source(source_rect)
    destination = np.full((2, 2), SENTINEL, dtype=np.uint16)
    before_source = source.copy()
    before_destination = destination.copy()
    # Aligned address whose 8-byte minimal span wraps uintptr_t.
    near_end = (UINTPTR_MAX - 7) & ~1
    kwargs = {}
    if which_pointer == "source":
        kwargs["source_pointer"] = pointer_at(near_end)
    else:
        kwargs["destination_pointer"] = pointer_at(near_end)
    status, actual = bridge.execute(
        source, source_rect, request_rect, destination=destination, **kwargs
    )
    assert status != 0
    np.testing.assert_array_equal(source, before_source)
    np.testing.assert_array_equal(actual, before_destination)


@pytest.mark.parametrize("alias_kind", ("full", "partial", "source_padding"))
def test_source_and_destination_declared_spans_cannot_overlap_atomically(
    bridge, alias_kind
):
    source_rect = (0, 0, 2, 2)
    request_rect = (0, 0, 2, 2)
    storage = np.arange(24, dtype=np.uint16) + np.uint16(100)
    before = storage.copy()
    kwargs = {
        "source_count": 8,
        "source_stride": 4,
        "destination_count": 8,
        "destination_stride": 4,
    }
    if alias_kind == "full":
        kwargs["destination_pointer"] = array_pointer(storage)
    elif alias_kind == "partial":
        kwargs["destination_pointer"] = pointer_at(storage.ctypes.data + 5 * 2)
    else:
        # Destination starts in the source's first-row padding samples.
        kwargs["destination_pointer"] = pointer_at(storage.ctypes.data + 2 * 2)
    status, actual = bridge.execute(
        storage, source_rect, request_rect, destination=storage, **kwargs
    )
    assert status != 0
    np.testing.assert_array_equal(storage, before,
                                  err_msg="overlap error modified aliased storage")
    np.testing.assert_array_equal(actual, before)


@pytest.mark.parametrize("overlap", ("source_rect", "request_rect"))
def test_destination_cannot_overlap_either_metadata_array(bridge, overlap):
    source_rect = np.asarray([0, 0, 2, 2], dtype=np.int32)
    request_rect = np.asarray([0, 0, 2, 2], dtype=np.int32)
    source = make_source(source_rect)
    destination = np.full(8, SENTINEL, dtype=np.uint16)
    metadata = source_rect if overlap == "source_rect" else request_rect
    destination_pointer = pointer_at(metadata.ctypes.data)
    before_source = source.copy()
    before_source_rect = source_rect.copy()
    before_request_rect = request_rect.copy()
    status, actual = bridge.execute(
        source,
        source_rect,
        request_rect,
        destination=destination,
        destination_count=4,
        destination_stride=2,
        destination_pointer=destination_pointer,
        source_rect_array=source_rect,
        request_rect_array=request_rect,
    )
    assert status != 0
    np.testing.assert_array_equal(source, before_source)
    np.testing.assert_array_equal(source_rect, before_source_rect)
    np.testing.assert_array_equal(request_rect, before_request_rect)
    np.testing.assert_array_equal(actual, np.full(8, SENTINEL, dtype=np.uint16))
