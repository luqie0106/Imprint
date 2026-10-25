"""Independent tests for PGTM derived cache tables and their C bridge."""

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
SUPPORT = ROOT / "tests" / "acr_support"
MAP_SOURCE = NATIVE / "src" / "acr_profile_gain_map.cpp"
GAIN_SOURCE = NATIVE / "src" / "acr_profile_gain.cpp"
CACHE_SOURCE = NATIVE / "src" / "acr_profile_gain_cache.cpp"
APPLY_SOURCE = NATIVE / "src" / "acr_profile_gain_apply.cpp"
CACHE_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_cache_bridge.cpp"
GAIN_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_bridge.cpp"
APPLY_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_apply_bridge.cpp"
OBJECT_SUFFIX = ".obj" if sys.platform == "win32" else ".o"
OFFICIAL_OBJECTS = tuple(
    NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/{name}{OBJECT_SUFFIX}"
    for name in (
        "acr_profile_gain_map.cpp",
        "acr_profile_gain.cpp",
        "acr_profile_gain_cache.cpp",
        "acr_profile_gain_apply.cpp",
    )
)
IMPLEMENTATION_SOURCES = (MAP_SOURCE, GAIN_SOURCE, CACHE_SOURCE, APPLY_SOURCE)

sys.path.insert(0, str(SUPPORT))
import profile_gain_cache_oracle as oracle  # noqa: E402


U8_PTR = C.POINTER(C.c_uint8)
I32_PTR = C.POINTER(C.c_int32)
U32_PTR = C.POINTER(C.c_uint32)
F32_PTR = C.POINTER(C.c_float)
F64_PTR = C.POINTER(C.c_double)
SIZE = C.c_size_t
SIZE_PTR = C.POINTER(SIZE)
F32_PTR_PTR = C.POINTER(F32_PTR)


def _numpy_ptr(array: np.ndarray | None, ctype):
    if array is None:
        return C.cast(C.c_void_p(0), C.POINTER(ctype))
    return array.ctypes.data_as(C.POINTER(ctype))


def _compiler() -> str:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the ProfileGain cache bridge tests"
    return compiler


def _implementation_paths(kind: str) -> tuple[Path, ...]:
    if kind == "source":
        return IMPLEMENTATION_SOURCES
    assert kind == "official_object"
    missing = [path for path in OFFICIAL_OBJECTS if not path.is_file()]
    if missing:
        pytest.skip("official ProfileGain cache objects unavailable: " + ", ".join(map(str, missing)))
    stale = [
        obj.name
        for obj, source in zip(OFFICIAL_OBJECTS, IMPLEMENTATION_SOURCES)
        if obj.stat().st_mtime_ns < source.stat().st_mtime_ns
    ]
    if stale:
        pytest.skip("official ProfileGain cache objects are stale: " + ", ".join(stale))
    return OFFICIAL_OBJECTS


def _shared_flags() -> tuple[str, list[str]]:
    if sys.platform == "darwin":
        return ".dylib", ["-dynamiclib"]
    if sys.platform == "win32":
        return ".dll", ["-shared"]
    return ".so", ["-shared", "-fPIC"]


def _compile_bridge(tmp_path: Path, kind: str) -> Path:
    for path in (
        *IMPLEMENTATION_SOURCES,
        CACHE_BRIDGE_SOURCE,
        GAIN_BRIDGE_SOURCE,
        APPLY_BRIDGE_SOURCE,
    ):
        assert path.is_file(), f"missing PGTM cache source: {path}"
    implementations = _implementation_paths(kind)
    suffix, shared_flags = _shared_flags()
    output = tmp_path / f"libprofile_gain_cache_{kind}{suffix}"
    command = [
        _compiler(),
        "-std=c++17",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        *shared_flags,
        "-I",
        str(NATIVE / "src"),
        str(CACHE_BRIDGE_SOURCE),
        str(GAIN_BRIDGE_SOURCE),
        str(APPLY_BRIDGE_SOURCE),
        *(str(path) for path in implementations),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


class CacheBridge:
    def __init__(self, library_path: Path):
        self.library = C.CDLL(str(library_path))
        self.cache_fn = self.library.acr_profile_gain_cache_bridge
        self.cache_fn.argtypes = [
            U8_PTR,
            SIZE,
            C.c_uint32,
            C.c_uint32,
            C.c_int32,
            C.c_uint32,
            C.c_uint32,
            I32_PTR,
            C.c_double,
            F64_PTR,
            SIZE,
            F32_PTR,
            SIZE,
            U32_PTR,
            SIZE,
            F32_PTR,
            SIZE,
            SIZE_PTR,
        ]
        self.cache_fn.restype = C.c_int

        self.lookup_fn = self.library.acr_profile_gain_bridge
        self.lookup_fn.argtypes = [
            F32_PTR,
            SIZE,
            U32_PTR,
            SIZE,
            F32_PTR,
            SIZE,
            F32_PTR_PTR,
            SIZE_PTR,
            F32_PTR,
            SIZE,
            SIZE,
            SIZE,
            SIZE,
            SIZE,
        ]
        self.lookup_fn.restype = C.c_int

        self.apply_fn = self.library.acr_profile_gain_apply_bridge
        self.apply_fn.argtypes = [
            F32_PTR_PTR,
            SIZE_PTR,
            F32_PTR_PTR,
            SIZE_PTR,
            SIZE,
            SIZE,
            SIZE,
            SIZE,
            C.c_double,
            C.c_uint32,
            C.c_uint32,
        ]
        self.apply_fn.restype = C.c_int

    def raw(
        self,
        data_ptr,
        data_bytes: int,
        tag: int,
        byteorder: int,
        amount: int,
        lookup_flag: int,
        predicate: int,
        rect_ptr,
        scalar: float,
        meta_ptr,
        meta_count: int,
        fields_ptr,
        fields_count: int,
        geometry_ptr,
        geometry_count: int,
        values_ptr,
        capacity: int,
        written_ptr,
    ) -> int:
        return int(
            self.cache_fn(
                data_ptr,
                data_bytes,
                tag,
                byteorder,
                amount,
                lookup_flag,
                predicate,
                rect_ptr,
                scalar,
                meta_ptr,
                meta_count,
                fields_ptr,
                fields_count,
                geometry_ptr,
                geometry_count,
                values_ptr,
                capacity,
                written_ptr,
            )
        )

    def prepare(self, payload: bytes, byteorder: int, amount: int, lookup_flag: int, predicate: int, *, rect=(0, 0, 64, 128), scalar=0.75, meta_count=25, field_count=19, geometry_count=7, capacity=16):
        data = C.create_string_buffer(payload, len(payload))
        rect_buffer = (C.c_int32 * 4)(*rect)
        meta = np.full(meta_count, oracle.SENTINEL_META, dtype=np.float64)
        fields = np.full(field_count, oracle.SENTINEL_FIELDS, dtype=np.float32)
        geometry = np.full(geometry_count, oracle.SENTINEL_GEOMETRY, dtype=np.uint32)
        values = np.full(capacity, oracle.SENTINEL_VALUES, dtype=np.float32)
        written = SIZE(oracle.SENTINEL_WRITTEN)
        status = self.raw(
            C.cast(data, U8_PTR),
            len(payload),
            oracle.TAG_EXTENDED,
            byteorder,
            amount,
            lookup_flag,
            predicate,
            rect_buffer,
            scalar,
            _numpy_ptr(meta, C.c_double),
            meta_count,
            _numpy_ptr(fields, C.c_float),
            field_count,
            _numpy_ptr(geometry, C.c_uint32),
            geometry_count,
            _numpy_ptr(values, C.c_float),
            capacity,
            C.pointer(written),
        )
        return status, meta, fields, geometry, values, written.value, data, rect_buffer


@pytest.fixture(scope="module", params=("source", "official_object"), ids=("source", "official_object"))
def bridge(request, tmp_path_factory):
    path = _compile_bridge(
        tmp_path_factory.mktemp("acr_profile_gain_cache_" + request.param), request.param
    )
    return CacheBridge(path)


CASES = [
    (data_type, byteorder, amount, lookup_flag, predicate)
    for data_type in range(4)
    for byteorder in range(2)
    for amount in (1, 100, 101, 200, 0x7FFFFFFF)
    for lookup_flag, predicate in ((0, 0), (0, 0xFFFFFF01), (1, 1), (255, 3))
]


@pytest.mark.parametrize(
    "data_type,byteorder,amount,lookup_flag,predicate",
    CASES,
    ids=lambda value: str(value),
)
def test_cache_values_metadata_and_geometry_match_independent_oracle(
    bridge, data_type, byteorder, amount, lookup_flag, predicate
):
    case = oracle.make_payload(data_type, byteorder)
    status, meta, fields, geometry, values, written, _, _ = bridge.prepare(
        case.payload,
        byteorder,
        amount,
        lookup_flag,
        predicate,
        capacity=case.source_values.size + 3,
    )
    expected_values, exponent, negative_remap = oracle.expected_cache_values(
        case.source_values, amount, lookup_flag, predicate
    )
    expected_meta, expected_fields, expected_geometry, _ = oracle.expected_prepared(
        case, expected_values, exponent, negative_remap
    )

    assert status == oracle.OK
    assert written == case.source_values.size
    np.testing.assert_array_equal(meta[:21], expected_meta)
    np.testing.assert_array_equal(oracle.bits32(fields[:16]), oracle.bits32(expected_fields))
    np.testing.assert_array_equal(geometry[:5], expected_geometry)
    np.testing.assert_array_equal(oracle.bits32(values[:written]), oracle.bits32(expected_values))
    assert meta[12] == 2.0, "derived cache metadata must retain datatype 2"
    assert meta[19] == float(exponent)
    assert meta[20] == (1.0 if negative_remap else 0.0)
    assert np.all(meta[21:] == oracle.SENTINEL_META)
    assert np.all(fields[16:] == oracle.SENTINEL_FIELDS)
    assert np.all(geometry[5:] == oracle.SENTINEL_GEOMETRY)
    assert np.all(values[written:] == oracle.SENTINEL_VALUES)


def test_hardcoded_33_vs_32_and_selector_probes(bridge):
    # This pinned scalar distinguishes the observed divisor 33 from 32.
    exponent_101 = oracle.amount_exponent(101)
    exponent_101_bits = int(oracle.bits32([exponent_101])[0])
    exponent_101_with_32 = np.float32((100.0 + 15.0 / 32.0) * 0.01)
    assert exponent_101_bits == 0x3F8094F2
    assert exponent_101_bits != int(oracle.bits32([exponent_101_with_32])[0])

    case = oracle.make_payload(2, 0)
    status, _, _, _, values, written, _, _ = bridge.prepare(
        case.payload, 0, 100, 0, 0, capacity=case.source_values.size
    )
    assert status == oracle.OK
    cached = values[:written]
    # At exponent 1, 2^-12 remaps to 2^-11 when selected; 4096 is unchanged.
    endpoint_values = np.asarray([case.source_values[0], case.source_values[7]], dtype=np.float32)
    assert endpoint_values.tolist() == [float(oracle.MIN_GAIN), float(oracle.MAX_GAIN)]
    for enabled, expected in ((False, (float(oracle.MIN_GAIN), 4096.0)), (True, (float(oracle.MIN_GAIN) * 2.0, 4096.0))):
        flag, predicate = (0, 1) if enabled else (0, 0)
        status, _, _, _, values, written, _, _ = bridge.prepare(
            case.payload, 0, 100, flag, predicate, capacity=case.source_values.size
        )
        assert status == oracle.OK and written == case.source_values.size
        # The two selected indices are asserted by their fixed fixture order.
        assert float(values[0]) == expected[0]
        assert float(values[7]) == expected[1]
    assert float(cached[0]) == float(oracle.MIN_GAIN)

    # These two neighbouring source floats straddle the u < 0 branch boundary.
    near_one = oracle.make_payload(3, 0)
    status, _, _, _, values, _, _, _ = bridge.prepare(
        near_one.payload, 0, 200, 0, 1, capacity=near_one.source_values.size
    )
    assert status == oracle.OK
    assert float(values[3]) == 1.0
    assert float(values[5]) > 1.0


@pytest.mark.parametrize("amount,lookup_flag", ((0, 0), (-1, 0), (100, 256)))
def test_invalid_amount_or_lookup_flag_is_atomic(bridge, amount, lookup_flag):
    case = oracle.make_payload(3, 1)
    status, meta, fields, geometry, values, written, _, _ = bridge.prepare(
        case.payload,
        1,
        amount,
        lookup_flag,
        1,
        capacity=case.source_values.size + 2,
    )
    assert status == oracle.INVALID_ARGUMENT
    assert np.all(meta == oracle.SENTINEL_META)
    assert np.all(fields == oracle.SENTINEL_FIELDS)
    assert np.all(geometry == oracle.SENTINEL_GEOMETRY)
    assert np.all(values == oracle.SENTINEL_VALUES)
    assert written == oracle.SENTINEL_WRITTEN


@pytest.mark.parametrize("short_kind", ("meta", "fields", "geometry", "values"))
def test_short_output_buffers_leave_all_outputs_unchanged(bridge, short_kind):
    case = oracle.make_payload(2, 0)
    kwargs = {"capacity": case.source_values.size + 2}
    if short_kind == "meta":
        kwargs["meta_count"] = 20
    elif short_kind == "fields":
        kwargs["field_count"] = 15
    elif short_kind == "geometry":
        kwargs["geometry_count"] = 4
    else:
        kwargs["capacity"] = case.source_values.size - 1
    status, meta, fields, geometry, values, written, _, _ = bridge.prepare(
        case.payload, 0, 101, 0, 0, **kwargs
    )
    assert status == oracle.BUFFER_TOO_SMALL
    assert np.all(meta == oracle.SENTINEL_META)
    assert np.all(fields == oracle.SENTINEL_FIELDS)
    assert np.all(geometry == oracle.SENTINEL_GEOMETRY)
    assert np.all(values == oracle.SENTINEL_VALUES)
    assert written == oracle.SENTINEL_WRITTEN


def test_wire_reader_range_remains_strict_and_failure_is_atomic(bridge):
    case = oracle.make_payload(2, 1)
    payload = bytearray(case.payload)
    struct.pack_into(">e", payload, 80, 8192.0)
    status, meta, fields, geometry, values, written, _, _ = bridge.prepare(
        bytes(payload), 1, 200, 1, 1, capacity=case.source_values.size + 2
    )
    assert status == oracle.INVALID_VALUE
    assert np.all(meta == oracle.SENTINEL_META)
    assert np.all(fields == oracle.SENTINEL_FIELDS)
    assert np.all(geometry == oracle.SENTINEL_GEOMETRY)
    assert np.all(values == oracle.SENTINEL_VALUES)
    assert written == oracle.SENTINEL_WRITTEN


def test_overlapping_outputs_are_rejected_without_mutation(bridge):
    case = oracle.make_payload(3, 0)
    data = C.create_string_buffer(case.payload, len(case.payload))
    rect = (C.c_int32 * 4)(0, 0, 64, 128)
    shared = C.create_string_buffer(b"q" * 200, 200)
    before = bytes(shared.raw)
    geometry = np.full(5, oracle.SENTINEL_GEOMETRY, dtype=np.uint32)
    values = np.full(case.source_values.size, oracle.SENTINEL_VALUES, dtype=np.float32)
    written = SIZE(oracle.SENTINEL_WRITTEN)
    status = bridge.raw(
        C.cast(data, U8_PTR),
        len(case.payload),
        oracle.TAG_EXTENDED,
        0,
        101,
        0,
        1,
        rect,
        0.75,
        C.cast(shared, F64_PTR),
        21,
        C.cast(shared, F32_PTR),
        16,
        _numpy_ptr(geometry, C.c_uint32),
        5,
        _numpy_ptr(values, C.c_float),
        values.size,
        C.pointer(written),
    )
    assert status == oracle.OVERLAP
    assert bytes(shared.raw) == before
    assert np.all(geometry == oracle.SENTINEL_GEOMETRY)
    assert np.all(values == oracle.SENTINEL_VALUES)
    assert written.value == oracle.SENTINEL_WRITTEN


@pytest.mark.parametrize("failure", ("input_output_overlap", "unaligned", "pointer_overflow"))
def test_pointer_boundary_failures_are_atomic(bridge, failure):
    case = oracle.make_payload(3, 0)
    data = C.create_string_buffer(case.payload, len(case.payload))
    rect = (C.c_int32 * 4)(0, 0, 64, 128)
    meta = np.full(25, oracle.SENTINEL_META, dtype=np.float64)
    fields = np.full(19, oracle.SENTINEL_FIELDS, dtype=np.float32)
    geometry = np.full(7, oracle.SENTINEL_GEOMETRY, dtype=np.uint32)
    values = np.full(case.source_values.size + 2, oracle.SENTINEL_VALUES, dtype=np.float32)
    written = SIZE(oracle.SENTINEL_WRITTEN)
    meta_ptr = _numpy_ptr(meta, C.c_double)
    meta_count = meta.size
    alignment_bytes = None

    if failure == "input_output_overlap":
        # The byte buffer has allocator alignment, and the bridge rejects the
        # output span before reading or writing through this overlapping view.
        meta_ptr = C.cast(data, F64_PTR)
    elif failure == "unaligned":
        alignment_bytes = C.create_string_buffer(b"u" * 201, 201)
        meta_ptr = C.cast(C.byref(alignment_bytes, 1), F64_PTR)
    else:
        meta_count = (SIZE(-1).value // C.sizeof(C.c_double)) + 1

    before_data = bytes(data.raw)
    before_alignment = bytes(alignment_bytes.raw) if alignment_bytes is not None else None
    status = bridge.raw(
        C.cast(data, U8_PTR),
        len(case.payload),
        oracle.TAG_EXTENDED,
        0,
        101,
        0,
        1,
        rect,
        0.75,
        meta_ptr,
        meta_count,
        _numpy_ptr(fields, C.c_float),
        fields.size,
        _numpy_ptr(geometry, C.c_uint32),
        geometry.size,
        _numpy_ptr(values, C.c_float),
        values.size,
        C.pointer(written),
    )

    expected_status = {
        "input_output_overlap": oracle.OVERLAP,
        "unaligned": oracle.UNSUPPORTED_ALIGNMENT,
        "pointer_overflow": oracle.POINTER_OVERFLOW,
    }[failure]
    assert status == expected_status
    assert bytes(data.raw) == before_data
    if alignment_bytes is not None:
        assert bytes(alignment_bytes.raw) == before_alignment
    assert np.all(meta == oracle.SENTINEL_META)
    assert np.all(fields == oracle.SENTINEL_FIELDS)
    assert np.all(geometry == oracle.SENTINEL_GEOMETRY)
    assert np.all(values == oracle.SENTINEL_VALUES)
    assert written.value == oracle.SENTINEL_WRITTEN


@pytest.mark.parametrize("endpoint,lookup_flag,predicate", (("low", 1, 1), ("low", 0, 1), ("high", 1, 1), ("high", 0, 1)))
def test_cached_table_flows_through_existing_lookup_bridge(bridge, endpoint, lookup_flag, predicate):
    case = oracle.make_payload(
        2,
        0,
        weights=(0.0, 0.0, 0.0, 0.0, 1.0),
        gamma=1.0,
        spacing_v=1.0,
        spacing_h=1.0,
        origin_v=0.0,
        origin_h=0.0,
    )
    status, _, fields, geometry, values, written, _, _ = bridge.prepare(
        case.payload,
        0,
        200,
        lookup_flag,
        predicate,
        rect=(0, 0, 1, 1),
        scalar=1.0,
        capacity=case.source_values.size,
    )
    assert status == oracle.OK and written == case.source_values.size

    source_z = np.float32(0.0 if endpoint == "low" else 0.5)
    coord_v = np.float32(-0.5 if endpoint == "low" else 0.5)
    inputs = [
        np.asarray([source_z], dtype=np.float32),
        np.asarray([source_z], dtype=np.float32),
        np.asarray([source_z], dtype=np.float32),
        np.asarray([coord_v], dtype=np.float32),
        np.asarray([-0.5], dtype=np.float32),
    ]
    input_ptrs = (F32_PTR * 5)(*(_numpy_ptr(item, C.c_float) for item in inputs))
    input_counts = (SIZE * 5)(*(item.size for item in inputs))
    output = np.asarray([oracle.SENTINEL_VALUES], dtype=np.float32)
    fields_for_lookup = np.ascontiguousarray(fields[:16])
    geometry_for_lookup = np.ascontiguousarray(geometry[:5])
    values_for_lookup = np.ascontiguousarray(values[:written])
    lookup_status = bridge.lookup_fn(
        _numpy_ptr(fields_for_lookup, C.c_float),
        16,
        _numpy_ptr(geometry_for_lookup, C.c_uint32),
        5,
        _numpy_ptr(values_for_lookup, C.c_float),
        written,
        input_ptrs,
        input_counts,
        _numpy_ptr(output, C.c_float),
        output.size,
        1,
        1,
        1,
        1,
    )
    expected_values, _, negative_remap = oracle.expected_cache_values(
        case.source_values, 200, lookup_flag, predicate
    )
    # The selected coordinates address source table entries 0 (2^-12) and 7
    # (4096), so the lookup result itself exercises values outside the reader
    # domain after cache generation.
    table_index = 0 if endpoint == "low" else 7
    expected_gain = expected_values[table_index]
    assert lookup_status == 0
    np.testing.assert_array_equal(oracle.bits32(output), oracle.bits32([expected_gain]))
    if endpoint == "low":
        assert float(output[0]) < float(oracle.MIN_GAIN)
        if negative_remap:
            no_remap_values, _, _ = oracle.expected_cache_values(
                case.source_values, 200, 1, predicate
            )
            assert float(output[0]) > float(no_remap_values[0])
    else:
        assert float(output[0]) > float(oracle.MAX_GAIN)

    # Feed this exact gain output into the three-plane apply bridge.  Pivot 0
    # and clipping disabled leave a simple, independently calculated f32 path.
    rgb_planes = [
        np.asarray([0.25], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
        np.asarray([0.75], dtype=np.float32),
    ]
    apply_inputs = [*rgb_planes, output]
    apply_input_ptrs = (F32_PTR * 4)(
        *(_numpy_ptr(item, C.c_float) for item in apply_inputs)
    )
    apply_input_counts = (SIZE * 4)(*(item.size for item in apply_inputs))
    apply_outputs = [np.full(1, oracle.SENTINEL_VALUES, dtype=np.float32) for _ in range(3)]
    apply_output_ptrs = (F32_PTR * 3)(
        *(_numpy_ptr(item, C.c_float) for item in apply_outputs)
    )
    apply_output_counts = (SIZE * 3)(*(item.size for item in apply_outputs))
    apply_status = bridge.apply_fn(
        apply_input_ptrs,
        apply_input_counts,
        apply_output_ptrs,
        apply_output_counts,
        1,
        1,
        1,
        1,
        0.0,
        0,
        0,
    )
    expected_outputs = [oracle.apply_pivot_zero(float(plane[0]), expected_gain) for plane in rgb_planes]
    assert apply_status == 0
    for actual, expected in zip(apply_outputs, expected_outputs):
        np.testing.assert_array_equal(oracle.bits32(actual), oracle.bits32([expected]))


API_CONTRACT_HARNESS = r"""
#include "acr_profile_gain_cache.hpp"

#include <array>
#include <cstdlib>
#include <new>
#include <vector>

static bool fail_next_allocation = false;

void *operator new(std::size_t size) {
    if (fail_next_allocation) {
        fail_next_allocation = false;
        throw std::bad_alloc();
    }
    if (void *pointer = std::malloc(size == 0 ? 1 : size)) return pointer;
    throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void *pointer, std::size_t) noexcept { std::free(pointer); }

using namespace imprint::acr;

static ProfileGainMapData basic_map() {
    ProfileGainMapData map;
    map.rows = 1;
    map.columns = 1;
    map.points_n = 2;
    map.data_type = 2;
    map.gamma = 1.0f;
    map.gain_min = 0x1p-12f;
    map.gain_max = 4096.0f;
    map.values = {0x1p-12f, 4096.0f};
    return map;
}

int main() {
    const auto source = basic_map();

    ProfileGainCacheMap failed;
    failed.map.rows = 7;
    failed.map.columns = 8;
    failed.map.points_n = 9;
    failed.map.data_type = 3;
    failed.map.values.reserve(16);
    failed.map.values.push_back(7.25f);
    failed.exponent = 8.5f;
    failed.negative_remap = true;
    const auto *old_data = failed.map.values.data();
    const auto old_capacity = failed.map.values.capacity();
    const auto old_values = failed.map.values;
    fail_next_allocation = true;
    const auto allocation_status = build_profile_gain_cache_map(source, 100, 0, 1, failed);
    if (allocation_status != ProfileGainMapStatus::allocation_failure ||
        failed.map.values.data() != old_data || failed.map.values.capacity() != old_capacity ||
        failed.map.values != old_values || failed.map.rows != 7 || failed.map.columns != 8 ||
        failed.map.points_n != 9 || failed.map.data_type != 3 || failed.exponent != 8.5f ||
        !failed.negative_remap) return 10;

    // Bind the input reference to out.map to exercise the aliasing commit.
    ProfileGainCacheMap in_place;
    in_place.map = basic_map();
    const auto alias_status = build_profile_gain_cache_map(in_place.map, 200, 1, 1, in_place);
    if (alias_status != ProfileGainMapStatus::ok || in_place.map.data_type != 2 ||
        in_place.map.values.size() != 2 || !(in_place.map.values[0] > 0.0f) ||
        !(in_place.map.values[0] < 0x1p-12f) || !(in_place.map.values[1] > 4096.0f)) return 11;

    const std::array<std::int32_t, 4> rect = {0, 0, 1, 1};
    ProfileGainPrepared normal;
    ProfileGainPrepared cached;
    const auto normal_status = prepare_profile_gain_map(in_place.map, rect, 1.0, normal);
    const auto cached_status = prepare_cached_profile_gain_map(in_place.map, rect, 1.0, cached);
    if (normal_status != ProfileGainMapStatus::invalid_value ||
        cached_status != ProfileGainMapStatus::ok || cached.table != in_place.map.values.data() ||
        cached.table_count != 2) return 12;
    return 0;
}
"""


@pytest.mark.parametrize("kind", ("source", "official_object"), ids=("source", "official_object"))
def test_cpp_api_alias_commit_allocation_failure_and_cached_prepare(tmp_path, kind):
    implementations = _implementation_paths(kind)
    source = tmp_path / "cache_api_contract.cpp"
    source.write_text(API_CONTRACT_HARNESS, encoding="utf-8")
    executable = tmp_path / ("cache_api_contract.exe" if sys.platform == "win32" else "cache_api_contract")
    command = [
        _compiler(),
        "-std=c++17",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-I",
        str(NATIVE / "src"),
        str(source),
        *(str(path) for path in implementations),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
