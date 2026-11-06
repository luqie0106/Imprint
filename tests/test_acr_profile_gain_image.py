"""Independent composition tests for the local PGTM RGB image bridge."""

from __future__ import annotations

import ctypes as C
import itertools
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

IMAGE_SOURCE = NATIVE / "src" / "acr_profile_gain_image.cpp"
IMAGE_BRIDGE = NATIVE / "tools" / "acr_profile_gain_image_bridge.cpp"
IMPLEMENTATION_NAMES = (
    "acr_profile_gain_image.cpp",
    "acr_profile_gain_map.cpp",
    "acr_profile_gain_cache.cpp",
    "acr_profile_gain_coordinates.cpp",
    "acr_profile_gain.cpp",
    "acr_profile_gain_apply.cpp",
    "acr_resample_matrix.cpp",
)
IMPLEMENTATIONS = tuple(NATIVE / "src" / name for name in IMPLEMENTATION_NAMES)
OBJECT_SUFFIX = ".obj" if sys.platform == "win32" else ".o"
OFFICIAL_OBJECTS = tuple(
    NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/{name}{OBJECT_SUFFIX}"
    for name in IMPLEMENTATION_NAMES
)

sys.path.insert(0, str(SUPPORT))
import profile_gain_cache_oracle as cache_oracle  # noqa: E402
import profile_gain_coordinates_oracle as coordinates_oracle  # noqa: E402
import profile_gain_image_oracle as image_oracle  # noqa: E402
import profile_gain_map_oracle as map_oracle  # noqa: E402


U8_PTR = C.POINTER(C.c_uint8)
I32_PTR = C.POINTER(C.c_int32)
F32_PTR = C.POINTER(C.c_float)
F64_PTR = C.POINTER(C.c_double)
SIZE = C.c_size_t

OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
INVALID_STRIDE = 3
BUFFER_TOO_SMALL = 4
UNSUPPORTED_ALIGNMENT = 5
OVERLAP = 6
ARITHMETIC_OVERFLOW = 7
RESOURCE_LIMIT = 8
INVALID_PREPARED = 9
NON_FINITE_INPUT = 10
NON_FINITE_INTERMEDIATE = 11
ALLOCATION_FAILURE = 12
UNSUPPORTED_TRANSFORM = 13
INVALID_MAP = 14

SENTINEL = np.float32(-731.625)
SIZE_MAX = SIZE(-1).value
PTRDIFF_MAX = (1 << (C.sizeof(C.c_void_p) * 8 - 1)) - 1


def _compiler() -> str:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required to build the PGTM image bridge"
    return compiler


def _shared_flags() -> tuple[str, list[str]]:
    if sys.platform == "darwin":
        return ".dylib", ["-dynamiclib"]
    if sys.platform == "win32":
        return ".dll", ["-shared"]
    return ".so", ["-shared", "-fPIC"]


def _implementation_paths(kind: str) -> tuple[Path, ...]:
    if kind == "source":
        return IMPLEMENTATIONS
    assert kind == "official_object"
    missing = [path for path in OFFICIAL_OBJECTS if not path.is_file()]
    if missing:
        pytest.skip("official PGTM image objects are unavailable: " + ", ".join(map(str, missing)))
    stale = [
        (obj, source)
        for obj, source in zip(OFFICIAL_OBJECTS, IMPLEMENTATIONS)
        if obj.stat().st_mtime_ns < source.stat().st_mtime_ns
    ]
    if stale:
        pytest.skip("official PGTM image objects are stale: " + ", ".join(obj.name for obj, _ in stale))
    return OFFICIAL_OBJECTS


def _compile_bridge(tmp_path: Path, kind: str) -> Path:
    for path in (*IMPLEMENTATIONS, IMAGE_BRIDGE):
        assert path.is_file(), f"missing PGTM image source: {path}"
    implementations = _implementation_paths(kind)
    suffix, shared_flags = _shared_flags()
    output = tmp_path / f"libprofile_gain_image_{kind}{suffix}"
    command = [
        _compiler(),
        "-std=c++17",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        *shared_flags,
        "-I",
        str(NATIVE / "src"),
        str(IMAGE_BRIDGE),
        *(str(path) for path in implementations),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


def _typed_ptr(value, ctype):
    if value is None:
        return C.cast(C.c_void_p(0), C.POINTER(ctype))
    if isinstance(value, np.ndarray):
        return value.ctypes.data_as(C.POINTER(ctype))
    if isinstance(value, C.c_void_p):
        return C.cast(value, C.POINTER(ctype))
    return C.cast(value, C.POINTER(ctype))


class ImageBridge:
    def __init__(self, library_path: Path):
        self.library = C.CDLL(str(library_path))
        self.process_fn = self.library.acr_profile_gain_image_bridge
        self.process_fn.argtypes = [
            F32_PTR,
            SIZE,
            SIZE,
            F32_PTR,
            SIZE,
            SIZE,
            U8_PTR,
            SIZE,
            C.c_uint32,
            C.c_uint32,
            I32_PTR,
            SIZE,
            F64_PTR,
            SIZE,
            C.c_double,
            C.c_int32,
            C.c_uint32,
            C.c_uint32,
            C.c_uint32,
        ]
        self.process_fn.restype = C.c_int

    def raw(
        self,
        src_ptr,
        src_count: int,
        src_stride: int,
        dst_ptr,
        dst_count: int,
        dst_stride: int,
        payload_ptr,
        payload_bytes: int,
        tag: int,
        byteorder: int,
        rects_ptr,
        rect_count: int,
        matrix_ptr,
        matrix_count: int,
        scalar: float,
        amount: int,
        lookup_flag: int,
        predicate: int,
        has_task: int,
    ) -> int:
        return int(
            self.process_fn(
                _typed_ptr(src_ptr, C.c_float),
                src_count,
                src_stride,
                _typed_ptr(dst_ptr, C.c_float),
                dst_count,
                dst_stride,
                _typed_ptr(payload_ptr, C.c_uint8),
                payload_bytes,
                tag,
                byteorder,
                _typed_ptr(rects_ptr, C.c_int32),
                rect_count,
                _typed_ptr(matrix_ptr, C.c_double),
                matrix_count,
                scalar,
                amount,
                lookup_flag,
                predicate,
                has_task,
            )
        )


@pytest.fixture(scope="module", params=("source", "official_object"), ids=("source", "official_object"))
def bridge(request, tmp_path_factory):
    path = _compile_bridge(tmp_path_factory.mktemp("acr_profile_gain_image_" + request.param), request.param)
    return ImageBridge(path)


def _geometry(tile=(-2, 3, 6, 10)):
    return np.asarray(
        [
            -9, -13, 21, 26,       # map rect
            -8, -12, 12, 20,       # reference-zero rect
            -4, 2, 18, 37,         # state rect
            *tile,                 # tile rect
        ],
        dtype=np.int32,
    )


def _matrix():
    return np.asarray(
        [1.125, 0.0625, -0.25, -0.03125, 0.875, 0.375, 0.001, -0.002, 1.0],
        dtype=np.float64,
    )


def _source(height: int, width: int, stride: int, *, sentinel=SENTINEL):
    values = np.full(height * stride, np.float32(sentinel), dtype=np.float32)
    pattern = np.asarray(
        [0.0, 0.25, 0.5, 1.0, -0.5, 2.0, 3.75, 8192.0, -16.0, 32768.0, 0.875, 1.125],
        dtype=np.float32,
    )
    for row in range(height):
        for column in range(width * 3):
            values[row * stride + column] = pattern[(row * width * 3 + column) % pattern.size]
    return values


def _payload(data_type=3, byteorder=0, *, gamma=1.375, spacing_v=1.25, spacing_h=0.8):
    return cache_oracle.make_payload(
        data_type,
        byteorder,
        weights=(0.125, -0.25, 0.5, 0.75, -1.0),
        gamma=gamma,
        spacing_v=spacing_v,
        spacing_h=spacing_h,
        origin_v=-2.5,
        origin_h=3.25,
    ).payload


def _valid_call(bridge, *, data_type=3, byteorder=0, amount=101, lookup_flag=0,
                predicate=1, tile=(-2, 3, 6, 10), payload=None, source=None,
                src_stride=None, dst_stride=None, scalar=0.75, has_task=0):
    rects = _geometry(tile)
    matrix = _matrix()
    height = tile[2] - tile[0]
    width = tile[3] - tile[1]
    src_stride = width * 3 + 5 if src_stride is None else src_stride
    dst_stride = width * 3 + 7 if dst_stride is None else dst_stride
    source = _source(height, width, src_stride) if source is None else source
    destination = np.full(height * dst_stride, SENTINEL, dtype=np.float32)
    original_source = source.copy()
    payload = _payload(data_type, byteorder) if payload is None else payload
    payload_buffer = C.create_string_buffer(payload, len(payload))
    status = bridge.raw(
        source, source.size, src_stride,
        destination, destination.size, dst_stride,
        payload_buffer, len(payload), map_oracle.TAG_EXTENDED, byteorder,
        rects, rects.size, matrix, matrix.size, scalar, amount,
        lookup_flag, predicate, has_task,
    )
    expected, prepared = image_oracle.reference_image(
        payload, map_oracle.TAG_EXTENDED, byteorder, amount, lookup_flag,
        predicate, rects, matrix, scalar, original_source, src_stride, dst_stride,
        destination_count=destination.size,
    )
    return status, source, original_source, destination, expected, prepared, rects, matrix


@pytest.mark.parametrize(
    "data_type,byteorder,amount,lookup_flag,predicate",
    list(itertools.product(range(4), (0, 1), (1, 100, 101, 200, 0x7FFFFFFF), (0, 1), (0, 1))),
)
def test_full_image_matches_composed_scalar_reference_for_payload_flag_and_amount_domains(
    bridge, data_type, byteorder, amount, lookup_flag, predicate
):
    status, source, original, destination, expected, prepared, rects, matrix = _valid_call(
        bridge,
        data_type=data_type,
        byteorder=byteorder,
        amount=amount,
        lookup_flag=lookup_flag,
        predicate=predicate,
    )
    assert status == OK
    np.testing.assert_array_equal(destination.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(source.view(np.uint32), original.view(np.uint32))
    direct_source_route = amount == 100 and (predicate & 1) == 0
    if direct_source_route:
        assert prepared.cache_exponent is None
        assert prepared.negative_remap is None
    else:
        assert prepared.cache_exponent is not None
        assert prepared.negative_remap is (lookup_flag == 0 and (predicate & 1) != 0)
    assert prepared.prepared_map.table.size == 12
    active = destination.reshape(-1, destination.size // (rects[14] - rects[12]))
    assert np.all(active[:, (rects[15] - rects[13]) * 3:] == SENTINEL)
    assert np.any(original < np.float32(cache_oracle.MIN_GAIN))
    assert np.any(original > np.float32(cache_oracle.MAX_GAIN))


def _single_gain_payload(value_bits=0x3E001CFE):
    value = np.asarray([value_bits], dtype=np.uint32).view(np.float32)[0]
    header = struct.pack(
        "<II4dI5fI3f",
        1, 1, 1.0, 1.0, 0.0, 0.0, 1,
        0.0, 0.0, 0.0, 0.0, 0.0,
        3, 1.0, float(cache_oracle.MIN_GAIN), float(cache_oracle.MAX_GAIN),
    )
    assert len(header) == 80
    return header + struct.pack("<f", float(value))


@pytest.mark.parametrize("amount", (99, 100, 101))
@pytest.mark.parametrize("predicate", (0, 2, 1, 3))
@pytest.mark.parametrize("lookup_flag", (0, 1))
def test_source_pair_route_and_cache_fallback_match_independent_reference(
    bridge, amount, predicate, lookup_flag
):
    """The amount-100 route preserves source bits only when predicate bit 0 is clear."""
    payload = _single_gain_payload()
    payload_buffer = C.create_string_buffer(payload, len(payload))
    source = np.ones(3, dtype=np.float32)
    destination = np.full(3, SENTINEL, dtype=np.float32)
    rects = np.asarray([0, 0, 1, 1] * 4, dtype=np.int32)
    matrix = np.eye(3, dtype=np.float64).reshape(-1)

    status = bridge.raw(
        source, source.size, 3, destination, destination.size, 3,
        payload_buffer, len(payload), map_oracle.TAG_EXTENDED, 0,
        rects, rects.size, matrix, matrix.size, 1.0, amount, lookup_flag,
        predicate, 0,
    )
    expected, prepared = image_oracle.reference_image(
        payload, map_oracle.TAG_EXTENDED, 0, amount, lookup_flag, predicate,
        rects, matrix, 1.0, source, 3, 3, destination_count=3,
    )
    assert status == OK
    np.testing.assert_array_equal(destination.view(np.uint32), expected.view(np.uint32))

    direct_source_route = amount == 100 and (predicate & 1) == 0
    if direct_source_route:
        assert prepared.cache_exponent is None
        assert prepared.negative_remap is None
        np.testing.assert_array_equal(
            destination.view(np.uint32), np.full(3, 0x3E001CFE, dtype=np.uint32)
        )
    else:
        assert prepared.cache_exponent is not None
        assert prepared.negative_remap is (lookup_flag == 0 and (predicate & 1) != 0)


def test_fixed_nonidentity_matrix_and_fma_order_hard_probe(bridge):
    """Fixed bit output catches composition changes beyond randomized mirror tests."""
    tile = (-3, 5, -1, 8)
    rects = _geometry(tile)
    matrix = _matrix()
    source_stride, destination_stride = 11, 13
    source = _source(2, 3, source_stride)
    payload = _payload(3, 1, gamma=1.625, spacing_v=0.625, spacing_h=1.75)
    payload_buffer = C.create_string_buffer(payload, len(payload))
    destination = np.full(2 * destination_stride, SENTINEL, dtype=np.float32)
    status = bridge.raw(
        source, source.size, source_stride, destination, destination.size,
        destination_stride, payload_buffer, len(payload), map_oracle.TAG_EXTENDED,
        1, rects, rects.size, matrix, matrix.size, 1.125, 101, 0, 1, 0,
    )
    assert status == OK
    active = np.concatenate((destination[:9], destination[destination_stride:destination_stride + 9]))
    expected_bits = np.asarray(
        [
            # Filled from the standalone scalar references and fixed here so
            # the bridge is checked against a non-random, bit-exact witness.
            0x00000000, 0x3EC05AAC, 0x3F405AAC, 0x3FC05AAC, 0xBF405AAC,
            0x40405AAC, 0x40B45501, 0x46405AAC, 0xC1C05AAC,
            0x47405AAC, 0x3FA84F56, 0x3FD86602, 0x00000000,
            0x3EC05AAC, 0x3F405AAC, 0x3FC05AAC, 0xBF405AAC, 0x40405AAC,
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(active.view(np.uint32), expected_bits)


def test_full_image_equals_full_width_row_tiles(bridge):
    tile = (-2, 3, 5, 10)
    height, width = 7, 7
    source_stride, destination_stride = width * 3 + 4, width * 3 + 6
    source = _source(height, width, source_stride)
    full_destination = np.full(height * destination_stride, SENTINEL, dtype=np.float32)
    tiled_destination = np.full_like(full_destination, SENTINEL)
    payload = _payload(2, 1, gamma=0.875)
    payload_buffer = C.create_string_buffer(payload, len(payload))
    matrix = _matrix()
    full_rects = _geometry(tile)
    full_status = bridge.raw(
        source, source.size, source_stride, full_destination, full_destination.size,
        destination_stride, payload_buffer, len(payload), map_oracle.TAG_EXTENDED,
        1, full_rects, full_rects.size, matrix, matrix.size, 0.625, 200, 0, 1, 0,
    )
    assert full_status == OK
    for row in range(height):
        row_rects = _geometry((tile[0] + row, tile[1], tile[0] + row + 1, tile[3]))
        source_offset = row * source_stride
        destination_offset = row * destination_stride
        row_status = bridge.raw(
            source[source_offset:], source.size - source_offset, source_stride,
            tiled_destination[destination_offset:], tiled_destination.size - destination_offset,
            destination_stride, payload_buffer, len(payload), map_oracle.TAG_EXTENDED,
            1, row_rects, row_rects.size, matrix, matrix.size, 0.625, 200, 0, 1, 0,
        )
        assert row_status == OK
    np.testing.assert_array_equal(tiled_destination.view(np.uint32), full_destination.view(np.uint32))


def test_exact_inplace_is_supported_and_preserves_padding(bridge):
    tile = (-1, -2, 3, 3)
    height, width = 4, 5
    stride = width * 3 + 5
    image = _source(height, width, stride)
    original = image.copy()
    rects, matrix = _geometry(tile), _matrix()
    payload = _payload(0, 0)
    payload_buffer = C.create_string_buffer(payload, len(payload))
    expected, _ = image_oracle.reference_image(
        payload, map_oracle.TAG_EXTENDED, 0, 100, 1, 0, rects, matrix,
        0.875, original, stride, stride, destination_count=image.size,
    )
    status = bridge.raw(
        image, image.size, stride, image, image.size, stride,
        payload_buffer, len(payload), map_oracle.TAG_EXTENDED, 0,
        rects, rects.size, matrix, matrix.size, 0.875, 100, 1, 0, 0,
    )
    assert status == OK
    np.testing.assert_array_equal(image.view(np.uint32), expected.view(np.uint32))
    for row in range(height):
        np.testing.assert_array_equal(image[row * stride + width * 3:(row + 1) * stride], original[row * stride + width * 3:(row + 1) * stride])


def _atomic_call(bridge, *, src=None, src_count=None, src_stride=6, dst=None,
                 dst_count=None, dst_stride=6, payload=None, payload_bytes=None,
                 tag=map_oracle.TAG_EXTENDED, byteorder=0, rects=None,
                 rect_count=None, matrix=None, matrix_count=None, scalar=0.75,
                 amount=100, lookup_flag=0, predicate=1, has_task=0):
    if src is None:
        src = np.asarray([0.25, 0.5, 0.75, 1.0, 0.125, 0.875], dtype=np.float32)
    if dst is None:
        dst = np.full(6, SENTINEL, dtype=np.float32)
    if payload is None:
        payload = _payload()
    if isinstance(payload, bytes):
        payload_owner = C.create_string_buffer(payload, len(payload))
        payload_ptr = payload_owner
        actual_bytes = len(payload)
    else:
        payload_owner = payload
        payload_ptr = payload
        actual_bytes = payload_bytes if payload_bytes is not None else len(payload)
    if rects is None:
        rects = _geometry((0, 0, 1, 2))
    if matrix is None:
        matrix = _matrix()
    return bridge.raw(
        src, src.size if src_count is None and hasattr(src, "size") else (src_count or 0), src_stride,
        dst, dst.size if dst_count is None and hasattr(dst, "size") else (dst_count or 0), dst_stride,
        payload_ptr, actual_bytes if payload_bytes is None else payload_bytes,
        tag, byteorder, rects, rects.size if rect_count is None and hasattr(rects, "size") else (rect_count or 0),
        matrix, matrix.size if matrix_count is None and hasattr(matrix, "size") else (matrix_count or 0),
        scalar, amount, lookup_flag, predicate, has_task,
    )


def test_task_branch_flags_and_invalid_task_fail_atomically(bridge):
    dst = np.full(6, SENTINEL, dtype=np.float32)
    assert _atomic_call(bridge, dst=dst, has_task=1) == UNSUPPORTED_TRANSFORM
    np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))
    assert _atomic_call(bridge, dst=dst, has_task=2) == INVALID_ARGUMENT
    np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))


@pytest.mark.parametrize("rect_count,matrix_count,expected", [(15, 9, BUFFER_TOO_SMALL), (17, 9, INVALID_ARGUMENT), (16, 8, BUFFER_TOO_SMALL), (16, 10, INVALID_ARGUMENT)])
def test_control_array_counts_are_exact_and_atomic(bridge, rect_count, matrix_count, expected):
    dst = np.full(6, SENTINEL, dtype=np.float32)
    assert _atomic_call(bridge, dst=dst, rect_count=rect_count, matrix_count=matrix_count) == expected
    np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))


@pytest.mark.parametrize("which,expected", [("source", BUFFER_TOO_SMALL), ("destination", BUFFER_TOO_SMALL)])
def test_short_image_buffers_are_atomic(bridge, which, expected):
    src = np.arange(6, dtype=np.float32) / np.float32(8)
    dst = np.full(6, SENTINEL, dtype=np.float32)
    kwargs = {"src": src, "dst": dst}
    if which == "source":
        kwargs["src_count"] = 5
    else:
        kwargs["dst_count"] = 5
    assert _atomic_call(bridge, **kwargs) == expected
    np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))


def test_invalid_amount_flag_dimensions_and_stride_are_atomic(bridge):
    cases = [
        ({"amount": 0}, INVALID_ARGUMENT),
        ({"lookup_flag": 256}, INVALID_ARGUMENT),
        ({"rects": _geometry((0, 0, 0, 2))}, INVALID_DIMENSIONS),
        ({"src_stride": 5}, INVALID_STRIDE),
    ]
    for kwargs, status in cases:
        dst = np.full(6, SENTINEL, dtype=np.float32)
        assert _atomic_call(bridge, dst=dst, **kwargs) == status
        np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))


def test_invalid_map_matrix_nan_and_late_nonfinite_source_are_atomic(bridge):
    dst = np.full(6, SENTINEL, dtype=np.float32)
    payload = bytearray(_payload())
    payload[0:4] = (0).to_bytes(4, "little")  # zero map rows
    assert _atomic_call(bridge, dst=dst, payload=bytes(payload)) == INVALID_MAP
    np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))

    matrix = _matrix()
    matrix[4] = np.nan
    assert _atomic_call(bridge, dst=dst, matrix=matrix) == INVALID_PREPARED
    np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))

    src = np.asarray([0.25, 0.5, 0.75, 1.0, 0.125, np.nan], dtype=np.float32)
    assert _atomic_call(bridge, src=src, dst=dst) == NON_FINITE_INPUT
    np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))


@pytest.mark.parametrize("predicate", (0, 1))
def test_nonfinite_map_preparation_scalar_is_atomic(bridge, predicate):
    dst = np.full(6, SENTINEL, dtype=np.float32)
    assert _atomic_call(
        bridge, dst=dst, scalar=float("inf"), predicate=predicate
    ) == NON_FINITE_INTERMEDIATE
    np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))


def test_partial_source_destination_overlap_is_rejected_atomically(bridge):
    backing = np.arange(7, dtype=np.float32)
    before = backing.copy()
    payload = _payload()
    payload_buffer = C.create_string_buffer(payload, len(payload))
    rects, matrix = _geometry((0, 0, 1, 2)), _matrix()
    status = bridge.raw(
        backing[:6], 6, 6, backing[1:], 6, 6,
        payload_buffer, len(payload), map_oracle.TAG_EXTENDED, 0,
        rects, rects.size, matrix, matrix.size, 0.75, 100, 0, 1, 0,
    )
    assert status == OVERLAP
    np.testing.assert_array_equal(backing.view(np.uint32), before.view(np.uint32))


@pytest.mark.parametrize("control", ("payload", "rects", "matrix"))
def test_output_cannot_overlap_control_inputs(bridge, control):
    src = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
    rects = _geometry((0, 0, 1, 1))
    matrix = _matrix()
    payload = _payload()
    payload_buffer = C.create_string_buffer(payload, len(payload))
    if control == "payload":
        output = payload_buffer
        original = bytes(payload_buffer.raw)
    elif control == "rects":
        output = rects
        original = rects.copy()
    else:
        output = matrix
        original = matrix.copy()
    status = bridge.raw(
        src, src.size, 3, output, 3, 3, payload_buffer, len(payload),
        map_oracle.TAG_EXTENDED, 0, rects, rects.size, matrix, matrix.size,
        0.75, 100, 0, 1, 0,
    )
    assert status == OVERLAP
    if control == "payload":
        assert bytes(payload_buffer.raw) == original
    else:
        np.testing.assert_array_equal(output.view(np.uint8), original.view(np.uint8))


def test_unaligned_pointer_and_declared_span_overflow_fail_before_access(bridge):
    storage = C.create_string_buffer(64)
    src_ptr = C.cast(C.c_void_p(C.addressof(storage) + 1), F32_PTR)
    dst = np.full(6, SENTINEL, dtype=np.float32)
    assert _atomic_call(bridge, src=src_ptr, src_count=6, dst=dst) == UNSUPPORTED_ALIGNMENT
    np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))

    overflow_src = C.cast(C.c_void_p(SIZE_MAX - 7), F32_PTR)
    assert _atomic_call(bridge, src=overflow_src, src_count=2, dst=dst) == ARITHMETIC_OVERFLOW
    np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))


@pytest.mark.parametrize("which", ("source", "destination", "rectangles", "matrix"))
def test_each_typed_pointer_rejects_unaligned_address_atomically(bridge, which):
    src = np.asarray([0.25, 0.5, 0.75, 1.0, 0.125, 0.875], dtype=np.float32)
    dst = np.full(6, SENTINEL, dtype=np.float32)
    rects, matrix = _geometry((0, 0, 1, 2)), _matrix()
    payload = _payload()
    payload_buffer = C.create_string_buffer(payload, len(payload))
    src_ptr = src
    dst_ptr = dst
    rect_ptr = rects
    matrix_ptr = matrix
    owner = None

    if which == "source":
        owner = C.create_string_buffer(src.nbytes + 1)
        C.memmove(C.addressof(owner) + 1, src.ctypes.data, src.nbytes)
        src_ptr = C.cast(C.c_void_p(C.addressof(owner) + 1), F32_PTR)
    elif which == "destination":
        owner = C.create_string_buffer(dst.nbytes + 1)
        before = bytes(owner.raw)
        dst_ptr = C.cast(C.c_void_p(C.addressof(owner) + 1), F32_PTR)
    elif which == "rectangles":
        owner = C.create_string_buffer(rects.nbytes + 1)
        C.memmove(C.addressof(owner) + 1, rects.ctypes.data, rects.nbytes)
        rect_ptr = C.cast(C.c_void_p(C.addressof(owner) + 1), I32_PTR)
    else:
        owner = C.create_string_buffer(matrix.nbytes + 1)
        C.memmove(C.addressof(owner) + 1, matrix.ctypes.data, matrix.nbytes)
        matrix_ptr = C.cast(C.c_void_p(C.addressof(owner) + 1), F64_PTR)

    status = bridge.raw(
        src_ptr, src.size, 6, dst_ptr, dst.size, 6,
        payload_buffer, len(payload), map_oracle.TAG_EXTENDED, 0,
        rect_ptr, rects.size, matrix_ptr, matrix.size, 0.75, 100, 0, 1, 0,
    )
    assert status == UNSUPPORTED_ALIGNMENT
    if which == "destination":
        assert bytes(owner.raw) == before
    else:
        np.testing.assert_array_equal(dst, np.full(6, SENTINEL, dtype=np.float32))


def test_resource_limit_rejects_before_parsing_or_allocating(bridge):
    # The source/destination spans are declared but deliberately unmapped.
    # A compliant implementation checks the 2N coordinate cap before reading
    # either image pointer, parsing payload bytes, or allocating stage arrays.
    width = (1 << 25) + 1
    rects = _geometry((0, 0, 1, width))
    matrix = _matrix()
    payload = _payload()
    payload_buffer = C.create_string_buffer(payload, len(payload))
    fake_src = C.cast(C.c_void_p(0x1000), F32_PTR)
    fake_dst = C.cast(C.c_void_p(0x20000000), F32_PTR)
    status = bridge.raw(
        fake_src, width * 3, width * 3,
        fake_dst, width * 3, width * 3,
        payload_buffer, len(payload), map_oracle.TAG_EXTENDED, 0,
        rects, rects.size, matrix, matrix.size, 0.75, 100, 0, 1, 0,
    )
    assert status == RESOURCE_LIMIT


def test_pointer_count_overflow_is_reported_before_access(bridge):
    src = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
    dst = np.full(3, SENTINEL, dtype=np.float32)
    assert _atomic_call(bridge, src=src, dst=dst, src_count=SIZE_MAX) == ARITHMETIC_OVERFLOW
    np.testing.assert_array_equal(dst, np.full(3, SENTINEL, dtype=np.float32))

