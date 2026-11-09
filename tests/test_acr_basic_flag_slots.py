"""Formal CPU ABI for b69684..b696c8 only; no full Prepare/Look claim.

The expectations below express the prepared flag/default contract directly.
They are not an instruction oracle and do not load a temporary test library.
Build the formal native renderer before running the ABI/wrapper tests.
"""

from __future__ import annotations

import ctypes as C
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import native_renderer as native  # noqa: E402

N = 132
HEADER_COUNT = 16
MASK_COUNT = 5
OK = 0
INVALID_ARGUMENT = 1
IP = C.POINTER(C.c_int32)
BP = C.POINTER(C.c_uint8)
UP = C.POINTER(C.c_uint32)
POINTER_TYPES = (IP, BP, IP, UP)
COUNTS = (N, HEADER_COUNT, N, MASK_COUNT)
SPAN_BYTES = (N * 4, HEADER_COUNT, N * 4, MASK_COUNT * 4)
SIZE_MAX = C.c_size_t(-1).value
UINTPTR_MAX = (1 << (8 * C.sizeof(C.c_void_p))) - 1

# Inclusive Basic ID ranges; the final item indexes bytes P+4..P+0x13.
FLAG_GROUPS = (
    (18, 24, 4), (25, 34, 8), (35, 42, 6), (43, 66, 7),
    (67, 80, 5), (81, 94, 9), (95, 102, 10), (103, 111, 11),
    (112, 118, 12),
)
DEFAULTS = np.zeros(N, dtype=np.int32)
# Only mapped, nonzero b5df18(ID, 2) entries are relevant to this operator.
# In particular, ID120's 400 and ID128's 100 MUST NOT be applied here.
for _id, _value in {
    22: 25, 23: 50, 24: 75, 26: 10, 27: 25, 30: 50, 33: 50,
    34: 50, 77: 50, 85: 50, 87: 30, 88: 70, 90: 40, 91: 60,
    100: 100, 104: 25, 105: 50, 107: 50, 108: 50, 110: 1,
}.items():
    DEFAULTS[_id] = _value
DEFAULTS.flags.writeable = False


def _pointer(array, pointer_type, byte_offset=0):
    return C.cast(C.c_void_p(array.ctypes.data + byte_offset), pointer_type)


def _arguments(arrays):
    arguments = []
    for array, pointer_type, count in zip(arrays, POINTER_TYPES, COUNTS):
        arguments.extend((_pointer(array, pointer_type), count))
    return arguments


def _expected(current, header):
    values = current.copy()
    mask_words = [0] * MASK_COUNT
    for first, last, header_index in FLAG_GROUPS:
        if int(header[header_index]) != 0:
            continue
        for slot_id in range(first, last + 1):
            values[slot_id] = DEFAULTS[slot_id]
            mask_words[slot_id // 32] |= 1 << (slot_id % 32)
    return values, np.asarray(mask_words, dtype=np.uint32)


def _marked_ids(mask):
    return {slot_id for slot_id in range(160)
            if (int(mask[slot_id // 32]) >> (slot_id % 32)) & 1}


@pytest.fixture(scope="module")
def library():
    library, path = native._get_library()
    assert Path(path).is_file(), "build the formal imprint_renderer library first"
    function = library.im_native_acr_basic_flag_slots
    function.argtypes = [IP, C.c_size_t, BP, C.c_size_t,
                         IP, C.c_size_t, UP, C.c_size_t]
    function.restype = C.c_int
    return library, path


@pytest.fixture(scope="module")
def function(library):
    return library[0].im_native_acr_basic_flag_slots


def _apply(function, current, header):
    current_before, header_before = current.copy(), header.copy()
    destination = np.full(N + 7, -0x1234567, dtype=np.int32)
    mask = np.full(MASK_COUNT + 7, 0xA5A5A5A5, dtype=np.uint32)
    destination_before, mask_before = destination.copy(), mask.copy()
    arguments = _arguments((current, header, destination, mask))
    assert function(*arguments) == OK
    np.testing.assert_array_equal(current, current_before)
    np.testing.assert_array_equal(header, header_before)
    np.testing.assert_array_equal(destination[N:], destination_before[N:])
    np.testing.assert_array_equal(mask[MASK_COUNT:], mask_before[MASK_COUNT:])
    expected_values, expected_mask = _expected(current_before, header_before)
    np.testing.assert_array_equal(destination[:N], expected_values)
    np.testing.assert_array_equal(mask[:MASK_COUNT], expected_mask)
    assert not (_marked_ids(mask[:MASK_COUNT]) & set(range(132, 160)))
    return destination[:N].copy(), mask[:MASK_COUNT].copy()


@pytest.mark.parametrize("header_index", range(HEADER_COUNT))
def test_every_header_byte_and_group_boundary(function, header_index):
    current = np.arange(N, dtype=np.int32) + 10000
    header = np.ones(HEADER_COUNT, dtype=np.uint8)
    header[header_index] = 0
    _, mask = _apply(function, current, header)
    expected_ids = {slot_id for first, last, index in FLAG_GROUPS
                    if index == header_index for slot_id in range(first, last + 1)}
    assert _marked_ids(mask) == expected_ids


@pytest.mark.parametrize("slot_id", range(N))
def test_every_basic_id_store_and_mask_position(function, slot_id):
    current = np.arange(N, dtype=np.int32) + 10000
    group = next((group for group in FLAG_GROUPS
                  if group[0] <= slot_id <= group[1]), None)
    # Unmapped IDs must survive even when every available flag is zero.
    header = np.zeros(HEADER_COUNT, dtype=np.uint8)
    if group is not None:
        header[:] = 255
        header[group[2]] = 0
    values, mask = _apply(function, current, header)
    assert bool(int(mask[slot_id // 32]) & (1 << (slot_id % 32))) == (group is not None)
    assert values[slot_id] == (DEFAULTS[slot_id] if group is not None else current[slot_id])
    if group is not None:
        assert _marked_ids(mask) == set(range(group[0], group[1] + 1))


@pytest.mark.parametrize("flag", range(256))
def test_only_exact_zero_resets_for_all_256_flag_bytes(function, flag):
    current = np.arange(N, dtype=np.int32) - 20000
    for first, last, index in FLAG_GROUPS:
        header = np.ones(HEADER_COUNT, dtype=np.uint8)
        header[index] = flag
        values, mask = _apply(function, current, header)
        assert _marked_ids(mask) == (set(range(first, last + 1)) if flag == 0 else set())
        if flag != 0:
            np.testing.assert_array_equal(values, current)


@pytest.mark.parametrize("value", [-(1 << 31), -999999, -1000, -1, 0, 1, (1 << 31) - 1])
def test_sentinel_and_int32_extremes_are_not_special(function, value):
    current = np.full(N, value, dtype=np.int32)
    for flag in (0, 1, 2, 128, 255):
        header = np.full(HEADER_COUNT, flag, dtype=np.uint8)
        values, mask = _apply(function, current, header)
        np.testing.assert_array_equal(values[:18], current[:18])
        np.testing.assert_array_equal(values[119:], current[119:])
        assert _marked_ids(mask) == (set(range(18, 119)) if flag == 0 else set())


def test_all_zero_header_uses_mode2_defaults_not_source_seed_defaults(function):
    current = np.full(N, -999999, dtype=np.int32)
    values, mask = _apply(function, current, np.zeros(HEADER_COUNT, dtype=np.uint8))
    assert values[23] == 50 and values[25] == 0 and values[32] == 0
    assert values[30] == values[33] == values[34] == 50
    assert values[77] == 50 and values[100] == 100 and values[110] == 1
    assert values[120] == values[128] == values[130] == -999999
    assert mask.tolist() == [0xFFFC0000, 0xFFFFFFFF, 0xFFFFFFFF, 0x007FFFFF, 0]


def test_unchanged_values_still_mark_every_executed_store(function):
    current = DEFAULTS.copy()
    values, mask = _apply(function, current, np.zeros(HEADER_COUNT, dtype=np.uint8))
    np.testing.assert_array_equal(values, current)
    assert _marked_ids(mask) == set(range(18, 119))


def test_random_mixed_flags_and_raw_int32_bits_do_not_mutate_inputs(function):
    rng = np.random.default_rng(0xB69664)
    for iteration in range(150):
        current = rng.integers(0, 1 << 32, N, dtype=np.uint32).view(np.int32)
        header = (rng.integers(0, 256, HEADER_COUNT, dtype=np.uint8)
                  if iteration % 2 else
                  rng.choice(np.array([0, 0, 1, 2, 127, 128, 255], dtype=np.uint8), HEADER_COUNT))
        header[4 + iteration % 9] = 0
        _apply(function, current, header)


def _invalid_arrays():
    return (np.arange(N, dtype=np.int32) - 700,
            np.zeros(HEADER_COUNT, dtype=np.uint8),
            np.full(N + 7, -0x1234567, dtype=np.int32),
            np.full(MASK_COUNT + 7, 0xA5A5A5A5, dtype=np.uint32))


def _assert_invalid_unchanged(function, arguments, arrays):
    before = [array.copy() for array in arrays]
    assert function(*arguments) == INVALID_ARGUMENT
    for array, saved in zip(arrays, before):
        np.testing.assert_array_equal(array, saved)


@pytest.mark.parametrize("buffer_index", range(4))
def test_counts_are_exact_and_both_outputs_are_atomic(function, buffer_index):
    arrays = _invalid_arrays()
    count = COUNTS[buffer_index]
    for invalid_count in (0, 1, count - 1, count + 1, SIZE_MAX // 2, SIZE_MAX):
        arguments = _arguments(arrays)
        arguments[buffer_index * 2 + 1] = invalid_count
        _assert_invalid_unchanged(function, arguments, arrays)


@pytest.mark.parametrize("buffer_index", range(4))
def test_null_in_any_buffer_preserves_both_outputs(function, buffer_index):
    arrays = _invalid_arrays()
    arguments = _arguments(arrays)
    arguments[buffer_index * 2] = None
    _assert_invalid_unchanged(function, arguments, arrays)


@pytest.mark.parametrize("buffer_index", [0, 2, 3])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_unaligned_typed_buffers_preserve_both_outputs(function, buffer_index, offset):
    arrays = _invalid_arrays()
    arguments = _arguments(arrays)
    arguments[buffer_index * 2] = _pointer(arrays[buffer_index], POINTER_TYPES[buffer_index], offset)
    _assert_invalid_unchanged(function, arguments, arrays)


@pytest.mark.parametrize("buffer_index", range(4))
def test_address_wrap_is_rejected_before_reading_or_writing(function, buffer_index):
    arrays = _invalid_arrays()
    arguments = _arguments(arrays)
    # Keep natural alignment for each typed buffer while its end wraps.
    invalid_address = UINTPTR_MAX - (7 if buffer_index == 1 else 3)
    arguments[buffer_index * 2] = C.cast(C.c_void_p(invalid_address), POINTER_TYPES[buffer_index])
    _assert_invalid_unchanged(function, arguments, arrays)


def test_all_spans_are_checked_before_any_input_is_read(function):
    arrays = _invalid_arrays()
    arguments = _arguments(arrays)
    arguments[0] = C.cast(C.c_void_p(4), IP)  # Aligned, nonwrapping, but unreadable.
    arguments[6] = None
    _assert_invalid_unchanged(function, arguments, arrays)
    arguments = _arguments(arrays)
    arguments[2] = C.cast(C.c_void_p(1), BP)  # Raw header has no alignment restriction.
    arguments[6] = C.cast(C.c_void_p(UINTPTR_MAX - 3), UP)
    _assert_invalid_unchanged(function, arguments, arrays)


@pytest.mark.parametrize("pair", list(combinations(range(4), 2)))
@pytest.mark.parametrize("placement", ["same_start", "forward", "backward", "last_word"])
def test_all_six_buffer_pairs_reject_overlap_atomically(function, pair, placement):
    storage = np.full(4096, 0xA5, dtype=np.uint8)
    offsets = [64, 1024, 2048, 3072]
    first, second = pair
    delta = {"same_start": 0, "forward": 4, "backward": -4,
             "last_word": SPAN_BYTES[first] - 4}[placement]
    offsets[second] = offsets[first] + delta
    arguments = []
    for index in range(4):
        arguments.extend((_pointer(storage, POINTER_TYPES[index], offsets[index]), COUNTS[index]))
    before = storage.copy()
    assert function(*arguments) == INVALID_ARGUMENT
    np.testing.assert_array_equal(storage, before)


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 2, 1, 0)])
def test_adjacent_nonoverlapping_buffers_are_valid(function, order):
    storage = np.full(sum(SPAN_BYTES) + 8, 0xA5, dtype=np.uint8)
    offsets, cursor = [0] * 4, 4
    for index in order:
        offsets[index] = cursor
        cursor += SPAN_BYTES[index]
    arrays = tuple(np.ndarray((count,), dtype=dtype, buffer=storage, offset=offset)
                   for count, dtype, offset in zip(COUNTS, (np.int32, np.uint8, np.int32, np.uint32), offsets))
    current, header, destination, mask = arrays
    current[:] = np.arange(N, dtype=np.int32) + 9000
    header[:] = 0
    before = storage.copy()
    expected_values, expected_mask = _expected(current, header)
    assert function(*_arguments(arrays)) == OK
    np.testing.assert_array_equal(destination, expected_values)
    np.testing.assert_array_equal(mask, expected_mask)
    for index in (0, 1):
        start, end = offsets[index], offsets[index] + SPAN_BYTES[index]
        np.testing.assert_array_equal(storage[start:end], before[start:end])
    np.testing.assert_array_equal(storage[:4], before[:4])
    np.testing.assert_array_equal(storage[-4:], before[-4:])


def test_byte_header_can_start_at_an_odd_address(function):
    header_storage = np.full(HEADER_COUNT + 2, 0xA5, dtype=np.uint8)
    header = header_storage[1:-1]
    assert header.ctypes.data % 2 == 1
    header[:] = 0
    before = header_storage.copy()
    _apply(function, np.arange(N, dtype=np.int32), header)
    np.testing.assert_array_equal(header_storage, before)


def test_wrapper_contiguous_strided_unaligned_and_readonly_inputs(library, monkeypatch):
    monkeypatch.setattr(native, "_get_library", lambda: library)
    current = np.arange(N, dtype=np.int32) - 54321
    header = np.array([0, 255, 1, 2, 0, 1, 0, 255, 0, 0, 2, 0, 0, 1, 0, 255], dtype=np.uint8)
    unaligned = np.ndarray((N,), dtype=np.int32, buffer=bytearray(N * 4 + 1), offset=1)
    unaligned[:] = current
    assert not unaligned.flags.aligned
    readonly = current.copy()
    readonly.flags.writeable = False
    readonly_header = header.copy()
    readonly_header.flags.writeable = False
    cases = ((current, header), (np.repeat(current, 2)[::2], np.repeat(header, 2)[::2]),
             (unaligned, header), (current[::-1], header[::-1]), (readonly, readonly_header))
    for values, flags in cases:
        before = values.copy(), flags.copy()
        result, mask = native.native_acr_basic_flag_slots(values, flags)
        assert result.shape == (N,) and result.dtype == np.int32
        assert mask.shape == (MASK_COUNT,) and mask.dtype == np.uint32
        assert result.flags.c_contiguous and result.flags.aligned
        assert mask.flags.c_contiguous and mask.flags.aligned
        assert not np.shares_memory(result, values)
        assert not np.shares_memory(mask, values)
        assert not np.shares_memory(result, mask)
        expected_values, expected_mask = _expected(values, flags)
        np.testing.assert_array_equal(result, expected_values)
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(values, before[0])
        np.testing.assert_array_equal(flags, before[1])


def test_wrapper_owns_disjoint_snapshots_even_when_readonly_inputs_alias(library, monkeypatch):
    monkeypatch.setattr(native, "_get_library", lambda: library)
    current = np.arange(N, dtype=np.int32) - 1000
    header = current.view(np.uint8)[32:32 + HEADER_COUNT]
    header[:] = 0
    assert np.shares_memory(current, header)
    before = current.copy()
    expected_values, expected_mask = _expected(current, header)
    values, mask = native.native_acr_basic_flag_slots(current, header)
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(current, before)


@pytest.mark.parametrize("bad_current", [
    None, [0] * N, np.zeros(N, np.int64), np.zeros(N, np.uint32),
    np.zeros(N, np.float32), np.zeros(N, np.float64), np.zeros(N, np.bool_),
    np.zeros(N, dtype=">i4"), np.zeros((1, N), np.int32),
    np.zeros(N - 1, np.int32), np.zeros(N + 1, np.int32), np.array(0, np.int32),
])
def test_wrapper_rejects_current_dtype_and_shape_before_loading(monkeypatch, bad_current):
    monkeypatch.setattr(native, "_get_library", lambda: pytest.fail("invalid inputs must not load the library"))
    with pytest.raises(ValueError):
        native.native_acr_basic_flag_slots(bad_current, np.zeros(HEADER_COUNT, np.uint8))


@pytest.mark.parametrize("bad_header", [
    None, [0] * HEADER_COUNT, bytes(HEADER_COUNT), np.zeros(HEADER_COUNT, np.int8),
    np.zeros(HEADER_COUNT, np.uint16), np.zeros(HEADER_COUNT, np.int32),
    np.zeros(HEADER_COUNT, np.float32), np.zeros(HEADER_COUNT, np.bool_),
    np.zeros((4, 4), np.uint8), np.zeros(HEADER_COUNT - 1, np.uint8),
    np.zeros(HEADER_COUNT + 1, np.uint8), np.array(0, np.uint8),
])
def test_wrapper_rejects_header_dtype_and_shape_before_loading(monkeypatch, bad_header):
    monkeypatch.setattr(native, "_get_library", lambda: pytest.fail("invalid inputs must not load the library"))
    with pytest.raises(ValueError):
        native.native_acr_basic_flag_slots(np.zeros(N, np.int32), bad_header)


def test_wrapper_does_not_fallback_when_library_or_symbol_is_missing(monkeypatch):
    def fail_load():
        raise native.NativeRendererError("test library unavailable")

    current, header = np.zeros(N, np.int32), np.zeros(HEADER_COUNT, np.uint8)
    monkeypatch.setattr(native, "_get_library", fail_load)
    with pytest.raises(native.NativeRendererError, match="test library unavailable"):
        native.native_acr_basic_flag_slots(current, header)
    monkeypatch.setattr(native, "_get_library", lambda: (object(), "missing"))
    with pytest.raises(native.NativeRendererError):
        native.native_acr_basic_flag_slots(current, header)


@pytest.mark.parametrize("status", [1, 2, 5])
def test_wrapper_surfaces_native_failures_without_fallback(monkeypatch, status):
    class FailedLibrary:
        @staticmethod
        def im_native_acr_basic_flag_slots(*arguments):
            return status

    monkeypatch.setattr(native, "_get_library", lambda: (FailedLibrary(), "failure"))
    current, header = np.zeros(N, np.int32), np.zeros(HEADER_COUNT, np.uint8)
    before = current.copy(), header.copy()
    with pytest.raises(ValueError if status == INVALID_ARGUMENT else native.NativeRendererError):
        native.native_acr_basic_flag_slots(current, header)
    np.testing.assert_array_equal(current, before[0])
    np.testing.assert_array_equal(header, before[1])
