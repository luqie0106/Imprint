"""Bit-exact tests for the scalar ACR float32/u16 buffer conversion paths."""

from __future__ import annotations

import ctypes as C
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_buffer_format.cpp"
HEADER = NATIVE / "src/acr_buffer_format.hpp"
API_SOURCE = NATIVE / "src/acr_buffer_format_api.cpp"
PUBLIC_HEADER = NATIVE / "include/imprint_renderer.h"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_buffer_format.cpp.o"
BRIDGE_SOURCE = NATIVE / "tools/acr_buffer_format_bridge.cpp"
FORMAL_LIBRARY = NATIVE / "build" / (
    "libimprint_renderer.0.1.0.dylib"
    if sys.platform == "darwin"
    else "libimprint_renderer.so"
)
sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
import buffer_format_oracle as oracle  # noqa: E402


FLOAT = C.c_float
U16 = C.c_uint16
SIZE = C.c_size_t
FLOAT_PTR = C.POINTER(FLOAT)
U16_PTR = C.POINTER(U16)
SIZE_MAX = SIZE(-1).value
MAX_ACTIVE_SAMPLES = 128 * 1024 * 1024

OK = 0
INVALID_ARGUMENT = 1
INVALID_DIMENSIONS = 2
INVALID_STRIDE = 3
SHORT_BUFFER = 4
ALIGNMENT = 5
OVERLAP = 6
OVERFLOW = 7
RESOURCE = 8
ALLOC = 9
LENGTH = 10
FORMAL_BUFFER_TOO_SMALL = 4
FORMAL_RUNTIME_ERROR = 5
SENTINEL_F32 = np.float32(-1234.5)
SENTINEL_U16 = np.uint16(0xA55A)


def float_ptr(array):
    return array.ctypes.data_as(FLOAT_PTR)


def u16_ptr(array):
    return array.ctypes.data_as(U16_PTR)


def null_ptr(pointer_type):
    return C.cast(C.c_void_p(0), pointer_type)


def bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def assert_same_float_bits(actual, expected):
    np.testing.assert_array_equal(bits(actual), bits(expected))


def _configure_function(function, *, reverse):
    source_type, destination_type = (U16_PTR, FLOAT_PTR) if reverse else (FLOAT_PTR, U16_PTR)
    function.argtypes = [source_type, SIZE, SIZE, SIZE, SIZE, destination_type, SIZE, SIZE]
    function.restype = C.c_int
    return function


class BufferBridge:
    def __init__(self, library, forward_name, reverse_name, backend):
        self.library = library
        self.forward = _configure_function(getattr(library, forward_name), reverse=False)
        self.reverse = _configure_function(getattr(library, reverse_name), reverse=True)
        self.backend = backend

    def call_forward(
        self,
        source,
        source_count,
        source_stride,
        height,
        row_samples,
        destination,
        destination_count,
        destination_stride,
        *,
        source_address=None,
        destination_address=None,
    ):
        src = float_ptr(source) if source_address is None else C.cast(source_address, FLOAT_PTR)
        dst = u16_ptr(destination) if destination_address is None else C.cast(destination_address, U16_PTR)
        return self.forward(
            src,
            source_count,
            source_stride,
            height,
            row_samples,
            dst,
            destination_count,
            destination_stride,
        )

    def call_reverse(
        self,
        source,
        source_count,
        source_stride,
        height,
        row_samples,
        destination,
        destination_count,
        destination_stride,
        *,
        source_address=None,
        destination_address=None,
    ):
        src = u16_ptr(source) if source_address is None else C.cast(source_address, U16_PTR)
        dst = float_ptr(destination) if destination_address is None else C.cast(destination_address, FLOAT_PTR)
        # The exported reverse converter has the same eight-argument layout,
        # with source and destination element types exchanged.
        return self.reverse(
            src,
            source_count,
            source_stride,
            height,
            row_samples,
            dst,
            destination_count,
            destination_stride,
        )


def _formal_status(bridge, core_status):
    if bridge.backend != "formal_api":
        return core_status
    if core_status == OK:
        return OK
    if core_status == SHORT_BUFFER:
        return FORMAL_BUFFER_TOO_SMALL
    if core_status in (RESOURCE, ALLOC, LENGTH):
        return FORMAL_RUNTIME_ERROR
    return INVALID_ARGUMENT


@pytest.fixture(scope="module", params=("source", "official_object", "formal_api"))
def bridge(request, tmp_path_factory):
    assert SOURCE.is_file(), f"missing buffer-format source: {SOURCE}"
    assert HEADER.is_file(), f"missing buffer-format header: {HEADER}"
    assert BRIDGE_SOURCE.is_file(), f"missing buffer-format test bridge: {BRIDGE_SOURCE}"

    if request.param == "formal_api":
        assert API_SOURCE.is_file(), f"missing formal buffer-format API source: {API_SOURCE}"
        assert PUBLIC_HEADER.is_file(), f"missing renderer public header: {PUBLIC_HEADER}"
        assert FORMAL_LIBRARY.is_file(), f"rebuild renderer: missing formal library {FORMAL_LIBRARY}"
        dependencies = (SOURCE, HEADER, API_SOURCE, PUBLIC_HEADER)
        assert FORMAL_LIBRARY.stat().st_mtime_ns >= max(
            path.stat().st_mtime_ns for path in dependencies
        ), "rebuild renderer: stale formal buffer-format API/library"
        library = C.CDLL(str(FORMAL_LIBRARY))
        return BufferBridge(
            library,
            "im_native_acr_buffer_f32_to_u16",
            "im_native_acr_buffer_u16_to_f32",
            request.param,
        )

    if request.param == "official_object":
        assert OBJECT.is_file(), f"rebuild renderer: missing official object {OBJECT}"
        assert OBJECT.stat().st_mtime_ns >= max(
            SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns
        ), f"rebuild renderer: stale official object {OBJECT.name}"
        inputs = [OBJECT]
    else:
        inputs = [SOURCE]

    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the buffer-format bridge"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_buffer_format_" + request.param) / (
        "libacr_buffer_format_test_bridge" + suffix
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
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    library = C.CDLL(str(output))
    return BufferBridge(
        library,
        "acr_test_buffer_f32_to_u16",
        "acr_test_buffer_u16_to_f32",
        request.param,
    )


def _status(bridge, core_status):
    return _formal_status(bridge, core_status)


def _forward(bridge, source, destination, *, height, row_samples,
             source_stride=None, destination_stride=None,
             source_count=None, destination_count=None,
             source_address=None, destination_address=None):
    source_stride = row_samples if source_stride is None else source_stride
    destination_stride = row_samples if destination_stride is None else destination_stride
    source_count = source.size if source_count is None else source_count
    destination_count = destination.size if destination_count is None else destination_count
    return bridge.call_forward(
        source,
        source_count,
        source_stride,
        height,
        row_samples,
        destination,
        destination_count,
        destination_stride,
        source_address=source_address,
        destination_address=destination_address,
    )


def _reverse(bridge, source, destination, *, height, row_samples,
             source_stride=None, destination_stride=None,
             source_count=None, destination_count=None,
             source_address=None, destination_address=None):
    source_stride = row_samples if source_stride is None else source_stride
    destination_stride = row_samples if destination_stride is None else destination_stride
    source_count = source.size if source_count is None else source_count
    destination_count = destination.size if destination_count is None else destination_count
    return bridge.call_reverse(
        source,
        source_count,
        source_stride,
        height,
        row_samples,
        destination,
        destination_count,
        destination_stride,
        source_address=source_address,
        destination_address=destination_address,
    )


def test_all_u16_values_roundtrip_through_float32(bridge):
    source = np.arange(65536, dtype=np.uint16)
    floats = np.full(source.size, SENTINEL_F32, dtype=np.float32)
    status = _reverse(
        bridge, source, floats, height=1, row_samples=source.size
    )
    assert status == _status(bridge, OK)
    assert_same_float_bits(floats, oracle.u16s_to_float(source))

    restored = np.full(source.size, SENTINEL_U16, dtype=np.uint16)
    status = _forward(
        bridge, floats, restored, height=1, row_samples=source.size
    )
    assert status == _status(bridge, OK)
    np.testing.assert_array_equal(restored, source)


def test_random_float_bit_patterns_match_independent_fmaf_oracle(bridge):
    rng = np.random.default_rng(20261020)
    raw = rng.integers(0, 1 << 32, size=12000, dtype=np.uint32)
    samples = raw.view(np.float32)
    output = np.full(samples.size, SENTINEL_U16, dtype=np.uint16)
    status = _forward(bridge, samples, output, height=1, row_samples=samples.size)
    assert status == _status(bridge, OK)
    np.testing.assert_array_equal(output, oracle.floats_to_u16(samples))


def test_nan_infinities_signed_zero_boundaries_and_out_of_range(bridge):
    half = np.float32(0.5)
    zero = np.float32(0.0)
    one = np.float32(1.0)
    samples = np.asarray(
        [
            np.nan,
            np.float32(np.inf),
            np.float32(-np.inf),
            zero,
            np.float32(-0.0),
            np.nextafter(zero, np.float32(-np.inf)),
            np.nextafter(zero, np.float32(np.inf)),
            np.nextafter(half, np.float32(-np.inf)),
            half,
            np.nextafter(half, np.float32(np.inf)),
            np.nextafter(one, np.float32(-np.inf)),
            one,
            np.nextafter(one, np.float32(np.inf)),
            np.float32(-12.0),
            np.float32(12.0),
        ],
        dtype=np.float32,
    )
    output = np.full(samples.size, SENTINEL_U16, dtype=np.uint16)
    status = _forward(bridge, samples, output, height=1, row_samples=samples.size)
    assert status == _status(bridge, OK)
    np.testing.assert_array_equal(output, oracle.floats_to_u16(samples))
    assert output[0] == 0 and output[1] == 65535 and output[2] == 0
    assert output[3] == output[4] == 0
    assert output[7] == 32767 and output[8] == output[9] == 32768


def test_half_integer_quantization_boundaries_and_nextafter_neighbors(bridge):
    # Probe several output rounding boundaries from dark to bright, including
    # their nearest binary32 neighbors on both sides.
    boundary_indices = np.asarray([0, 1, 2, 255, 4096, 16383, 32767, 32768, 65534])
    centers = np.asarray(
        [(int(index) + 0.5) / 65535.0 for index in boundary_indices],
        dtype=np.float32,
    )
    samples = np.concatenate(
        (
            np.nextafter(centers, np.float32(-np.inf)),
            centers,
            np.nextafter(centers, np.float32(np.inf)),
        )
    )
    output = np.full(samples.size, SENTINEL_U16, dtype=np.uint16)
    status = _forward(bridge, samples, output, height=1, row_samples=samples.size)
    assert status == _status(bridge, OK)
    np.testing.assert_array_equal(output, oracle.floats_to_u16(samples))


def test_fused_and_rounding_mode_counterexamples_are_explicit(bridge):
    # Multiplication rounds to exactly 0.5 here, so np.rint uses ties-to-even
    # and produces 0; the observed fused `x * 65535 + 0.5`, truncated toward
    # zero, produces 1.
    ties_even_probe = np.asarray([0x37000080], dtype=np.uint32).view(np.float32)
    rounded_product = np.float32(ties_even_probe[0] * oracle.U16_SCALE)
    assert rounded_product == np.float32(0.5)
    assert int(np.rint(rounded_product)) == 0
    assert oracle.float_to_u16(ties_even_probe[0]) == 1

    # This nearby input is a hard pre-conversion witness: the fused and
    # separately rounded multiply/add results have distinct binary32 bits.
    # The uint16 conversion truncates both to 1, so comparing only stored
    # samples cannot observe this internal distinction.
    split_probe = np.asarray([0x37000083], dtype=np.uint32).view(np.float32)[0]
    fused = oracle.fma32(split_probe, oracle.U16_SCALE, oracle.HALF)
    split = np.float32(np.float32(split_probe * oracle.U16_SCALE) + oracle.HALF)
    assert int(fused.view(np.uint32)) == 0x3F800001
    assert int(split.view(np.uint32)) == 0x3F800002

    output = np.full(2, SENTINEL_U16, dtype=np.uint16)
    status = _forward(
        bridge, np.asarray([ties_even_probe[0], split_probe], dtype=np.float32),
        output, height=1, row_samples=2,
    )
    assert status == _status(bridge, OK)
    np.testing.assert_array_equal(output, np.asarray([1, 1], dtype=np.uint16))


def test_padded_strides_odd_width_and_padding_preservation(bridge):
    height, row_samples = 3, 7
    source_stride, destination_stride = 11, 9
    source = np.full(height * source_stride + 3, SENTINEL_F32, dtype=np.float32)
    destination = np.full(height * destination_stride + 5, SENTINEL_U16, dtype=np.uint16)
    expected = np.empty((height, row_samples), dtype=np.uint16)
    for row in range(height):
        values = np.asarray(
            [-0.25, 0.0, 0.125, 0.5, 0.731, 1.0, 1.25 + row],
            dtype=np.float32,
        )
        source[row * source_stride:row * source_stride + row_samples] = values
        expected[row] = oracle.floats_to_u16(values)
    before_source = source.copy()

    status = _forward(
        bridge,
        source,
        destination,
        height=height,
        row_samples=row_samples,
        source_stride=source_stride,
        destination_stride=destination_stride,
    )
    assert status == _status(bridge, OK)
    for row in range(height):
        start = row * destination_stride
        np.testing.assert_array_equal(destination[start:start + row_samples], expected[row])
        assert np.all(destination[start + row_samples:start + destination_stride] == SENTINEL_U16)
    assert np.all(destination[height * destination_stride:] == SENTINEL_U16)
    np.testing.assert_array_equal(source, before_source)

    reverse_source = np.full(height * destination_stride + 3, SENTINEL_U16, dtype=np.uint16)
    reverse_destination = np.full(height * source_stride + 5, SENTINEL_F32, dtype=np.float32)
    for row in range(height):
        reverse_source[row * destination_stride:row * destination_stride + row_samples] = (
            expected[row]
        )
    status = _reverse(
        bridge,
        reverse_source,
        reverse_destination,
        height=height,
        row_samples=row_samples,
        source_stride=destination_stride,
        destination_stride=source_stride,
    )
    assert status == _status(bridge, OK)
    for row in range(height):
        start = row * source_stride
        assert_same_float_bits(
            reverse_destination[start:start + row_samples],
            oracle.u16s_to_float(expected[row]),
        )
        assert np.all(reverse_destination[start + row_samples:start + source_stride] == SENTINEL_F32)
    assert np.all(reverse_destination[height * source_stride:] == SENTINEL_F32)


@pytest.mark.parametrize("direction", ("forward", "reverse"))
def test_short_buffers_and_invalid_layouts_are_atomic(bridge, direction):
    source = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    reverse_source = np.asarray([1, 2, 3, 4], dtype=np.uint16)
    source_storage = source if direction == "forward" else reverse_source
    destination_dtype = np.uint16 if direction == "forward" else np.float32

    cases = (
        ("short_source", {"source_count": 3}, SHORT_BUFFER),
        ("short_destination", {"destination_count": 3}, SHORT_BUFFER),
        ("zero_height", {"height": 0}, INVALID_DIMENSIONS),
        ("zero_row_samples", {"row_samples": 0}, INVALID_DIMENSIONS),
        ("source_stride", {"source_stride": 1}, INVALID_STRIDE),
        ("destination_stride", {"destination_stride": 1}, INVALID_STRIDE),
    )
    for _name, changes, core_status in cases:
        destination = np.full(4, SENTINEL_U16 if destination_dtype is np.uint16 else SENTINEL_F32,
                              dtype=destination_dtype)
        before = destination.tobytes()
        kwargs = dict(height=1, row_samples=4)
        kwargs.update(changes)
        if direction == "forward":
            status = _forward(bridge, source_storage, destination, **kwargs)
        else:
            status = _reverse(bridge, source_storage, destination, **kwargs)
        assert status == _status(bridge, core_status), _name
        assert destination.tobytes() == before, _name


@pytest.mark.parametrize("direction", ("forward", "reverse"))
def test_alignment_alias_overflow_and_active_limit_reject_atomically(bridge, direction):
    source_f32 = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    source_u16 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.uint16)
    source = source_f32 if direction == "forward" else source_u16
    destination = np.full(6, SENTINEL_U16 if direction == "forward" else SENTINEL_F32,
                          dtype=np.uint16 if direction == "forward" else np.float32)

    bad_calls = (
        ("unaligned_source", ALIGNMENT, lambda: _call_bad_alignment(
            bridge, direction, source, destination, source_unaligned=True)),
        ("unaligned_destination", ALIGNMENT, lambda: _call_bad_alignment(
            bridge, direction, source, destination, destination_unaligned=True)),
        ("overlap", OVERLAP, lambda: _call_alias(bridge, direction, source, destination)),
        ("size_overflow", OVERFLOW, lambda: _call_overflow(bridge, direction, source, destination)),
        ("extreme_height", OVERFLOW, lambda: _call_extreme_height(bridge, direction, source, destination)),
        ("resource_limit", RESOURCE, lambda: _call_resource_limit(bridge, direction, source, destination)),
    )
    for name, core_status, invoke in bad_calls:
        before = destination.tobytes()
        before_source = source.tobytes()
        status = invoke()
        assert status == _status(bridge, core_status), name
        assert destination.tobytes() == before, name
        assert source.tobytes() == before_source, name


@pytest.mark.parametrize("direction", ("forward", "reverse"))
def test_declared_span_byte_length_overflow_is_atomic(bridge, direction):
    source_f32 = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    source_u16 = np.asarray([1, 2, 3, 4], dtype=np.uint16)
    source = source_f32 if direction == "forward" else source_u16
    destination = np.full(
        4,
        SENTINEL_U16 if direction == "forward" else SENTINEL_F32,
        dtype=np.uint16 if direction == "forward" else np.float32,
    )

    for overflow_side in ("source", "destination"):
        before_source = source.tobytes()
        before_destination = destination.tobytes()
        kwargs = {f"{overflow_side}_count": SIZE_MAX}
        if direction == "forward":
            status = _forward(
                bridge, source, destination, height=1, row_samples=4, **kwargs
            )
        else:
            status = _reverse(
                bridge, source, destination, height=1, row_samples=4, **kwargs
            )
        assert status == _status(bridge, OVERFLOW), overflow_side
        assert source.tobytes() == before_source, overflow_side
        assert destination.tobytes() == before_destination, overflow_side


def _call_bad_alignment(bridge, direction, source, destination,
                        *, source_unaligned=False, destination_unaligned=False):
    src_addr = source.ctypes.data + (1 if source_unaligned else 0)
    dst_addr = destination.ctypes.data + (1 if destination_unaligned else 0)
    if direction == "forward":
        return _forward(bridge, source, destination, height=1, row_samples=4,
                        source_count=4, destination_count=4,
                        source_address=src_addr, destination_address=dst_addr)
    return _reverse(bridge, source, destination, height=1, row_samples=4,
                    source_count=4, destination_count=4,
                    source_address=src_addr, destination_address=dst_addr)


def _call_alias(bridge, direction, source, destination):
    if direction == "forward":
        # Both spans begin at the same aligned address; only validation should
        # run because the declared element types differ.
        dst_addr = source.ctypes.data
        return _forward(bridge, source, destination, height=1, row_samples=4,
                        source_count=4, destination_count=4,
                        destination_address=dst_addr)
    dst_addr = source.ctypes.data
    return _reverse(bridge, source, destination, height=1, row_samples=4,
                    source_count=4, destination_count=4,
                    destination_address=dst_addr)


def _call_overflow(bridge, direction, source, destination):
    if direction == "forward":
        return _forward(bridge, source, destination, height=2, row_samples=4,
                        source_stride=SIZE_MAX, destination_stride=4,
                        source_count=SIZE_MAX, destination_count=8)
    return _reverse(bridge, source, destination, height=2, row_samples=4,
                    source_stride=SIZE_MAX, destination_stride=4,
                    source_count=SIZE_MAX, destination_count=8)


def _call_extreme_height(bridge, direction, source, destination):
    if direction == "forward":
        return _forward(bridge, source, destination, height=SIZE_MAX,
                        row_samples=2, source_stride=2, destination_stride=2,
                        source_count=2, destination_count=2)
    return _reverse(bridge, source, destination, height=SIZE_MAX,
                    row_samples=2, source_stride=2, destination_stride=2,
                    source_count=2, destination_count=2)


def _call_resource_limit(bridge, direction, source, destination):
    if direction == "forward":
        return _forward(bridge, source, destination, height=1,
                        row_samples=MAX_ACTIVE_SAMPLES + 1,
                        source_stride=MAX_ACTIVE_SAMPLES + 1,
                        destination_stride=MAX_ACTIVE_SAMPLES + 1,
                        source_count=MAX_ACTIVE_SAMPLES + 1,
                        destination_count=MAX_ACTIVE_SAMPLES + 1)
    return _reverse(bridge, source, destination, height=1,
                    row_samples=MAX_ACTIVE_SAMPLES + 1,
                    source_stride=MAX_ACTIVE_SAMPLES + 1,
                    destination_stride=MAX_ACTIVE_SAMPLES + 1,
                    source_count=MAX_ACTIVE_SAMPLES + 1,
                    destination_count=MAX_ACTIVE_SAMPLES + 1)


def test_null_pointers_reject_without_writing(bridge):
    destination = np.full(2, SENTINEL_U16, dtype=np.uint16)
    before = destination.tobytes()
    status = bridge.forward(
        null_ptr(FLOAT_PTR), 2, 2, 1, 2, u16_ptr(destination), 2, 2
    )
    assert status == _status(bridge, INVALID_ARGUMENT)
    assert destination.tobytes() == before

    forward_source = np.asarray([0.25, 0.5], dtype=np.float32)
    status = bridge.forward(
        float_ptr(forward_source), 2, 2, 1, 2, null_ptr(U16_PTR), 2, 2
    )
    assert status == _status(bridge, INVALID_ARGUMENT)

    reverse_source = np.asarray([1, 2], dtype=np.uint16)
    reverse_destination = np.full(2, SENTINEL_F32, dtype=np.float32)
    before = reverse_destination.tobytes()
    status = bridge.reverse(
        u16_ptr(reverse_source), 2, 2, 1, 2,
        float_ptr(reverse_destination), 2, 2,
    )
    assert status == _status(bridge, OK)
    assert reverse_destination.tobytes() != before

    before = reverse_destination.tobytes()
    status = bridge.reverse(
        null_ptr(U16_PTR), 2, 2, 1, 2,
        float_ptr(reverse_destination), 2, 2,
    )
    assert status == _status(bridge, INVALID_ARGUMENT)
    assert reverse_destination.tobytes() == before
