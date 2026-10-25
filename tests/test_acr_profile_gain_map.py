"""Bitwise tests for serialized PGTM decoding and CPU preparation."""

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
MAP_SOURCE = NATIVE / "src" / "acr_profile_gain_map.cpp"
MAP_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_map_bridge.cpp"
CORE_SOURCE = NATIVE / "src" / "acr_profile_gain.cpp"
CORE_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_bridge.cpp"
OBJECT_SUFFIX = ".obj" if sys.platform == "win32" else ".o"
MAP_OBJECT = NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/acr_profile_gain_map.cpp{OBJECT_SUFFIX}"
CORE_OBJECT = NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/acr_profile_gain.cpp{OBJECT_SUFFIX}"
SUPPORT = ROOT / "tests" / "acr_support"

sys.path.insert(0, str(SUPPORT))
import profile_gain_map_oracle as map_oracle  # noqa: E402
import profile_gain_oracle as cpu_oracle  # noqa: E402


U8_PTR = C.POINTER(C.c_uint8)
I32_PTR = C.POINTER(C.c_int32)
U32_PTR = C.POINTER(C.c_uint32)
F32_PTR = C.POINTER(C.c_float)
F64_PTR = C.POINTER(C.c_double)
SIZE = C.c_size_t
SIZE_PTR = C.POINTER(SIZE)

OK = map_oracle.OK
INVALID_ARGUMENT = map_oracle.INVALID_ARGUMENT
INVALID_HEADER = map_oracle.INVALID_HEADER
UNSUPPORTED_TYPE = map_oracle.UNSUPPORTED_TYPE
INVALID_LENGTH = map_oracle.INVALID_LENGTH
INVALID_GAIN_VALUE = map_oracle.INVALID_GAIN_VALUE
INTEGER_OVERFLOW = map_oracle.INTEGER_OVERFLOW
RESOURCE_LIMIT = map_oracle.RESOURCE_LIMIT
SHORT_OUTPUT = map_oracle.SHORT_OUTPUT
ALIAS = map_oracle.ALIAS
ALIGNMENT = map_oracle.ALIGNMENT
POINTER_SPAN_OVERFLOW = map_oracle.POINTER_SPAN_OVERFLOW
INVALID_GEOMETRY = map_oracle.INVALID_GEOMETRY
NUMERIC_FAILURE = map_oracle.NUMERIC_FAILURE

SENTINEL_META = np.float64(-991.125)
SENTINEL_FIELDS = np.float32(-731.625)
SENTINEL_GEOMETRY = np.uint32(0xA5A5A5A5)
SENTINEL_VALUES = np.float32(-317.25)
SENTINEL_WRITTEN = 0x12345678


def _ptr(array: np.ndarray | None, ctype):
    if array is None:
        return C.cast(C.c_void_p(0), C.POINTER(ctype))
    return array.ctypes.data_as(C.POINTER(ctype))


def _compiler() -> str:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ is required to compile the ProfileGain map bridge")
    return compiler


def _compile_bridge(tmp_path: Path, kind: str) -> Path:
    assert MAP_SOURCE.is_file(), f"missing map implementation source: {MAP_SOURCE}"
    assert MAP_BRIDGE_SOURCE.is_file(), f"missing map bridge: {MAP_BRIDGE_SOURCE}"
    assert CORE_SOURCE.is_file(), f"missing existing CPU core: {CORE_SOURCE}"
    assert CORE_BRIDGE_SOURCE.is_file(), f"missing existing CPU bridge: {CORE_BRIDGE_SOURCE}"

    if kind == "source":
        map_impl = MAP_SOURCE
        core_impl = CORE_SOURCE
    else:
        assert kind == "official_object"
        if not MAP_OBJECT.is_file() or not CORE_OBJECT.is_file():
            pytest.skip(
                "official ProfileGain map/core objects are unavailable at the platform CMake path"
            )
        if MAP_OBJECT.stat().st_mtime_ns < MAP_SOURCE.stat().st_mtime_ns:
            pytest.skip("official ProfileGain map object is stale; rebuild imprint_renderer")
        if CORE_OBJECT.stat().st_mtime_ns < CORE_SOURCE.stat().st_mtime_ns:
            pytest.skip("official ProfileGain CPU object is stale; rebuild imprint_renderer")
        map_impl = MAP_OBJECT
        core_impl = CORE_OBJECT

    if sys.platform == "darwin":
        suffix = ".dylib"
        shared_flags = ["-dynamiclib"]
    elif sys.platform == "win32":
        suffix = ".dll"
        shared_flags = ["-shared"]
    else:
        suffix = ".so"
        shared_flags = ["-shared", "-fPIC"]
    output = tmp_path / f"libprofile_gain_map_{kind}{suffix}"
    command = [
        _compiler(),
        "-std=c++17",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        *shared_flags,
        "-I",
        str(NATIVE / "src"),
        str(MAP_BRIDGE_SOURCE),
        str(map_impl),
        str(CORE_BRIDGE_SOURCE),
        str(core_impl),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


class MapBridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.prepare_fn = self.library.acr_profile_gain_map_bridge
        self.prepare_fn.argtypes = [
            U8_PTR,
            SIZE,
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
        self.prepare_fn.restype = C.c_int

        self.apply_fn = self.library.acr_profile_gain_bridge
        self.apply_fn.argtypes = [
            F32_PTR,
            SIZE,
            U32_PTR,
            SIZE,
            F32_PTR,
            SIZE,
            C.POINTER(F32_PTR),
            SIZE_PTR,
            F32_PTR,
            SIZE,
            SIZE,
            SIZE,
            SIZE,
            SIZE,
        ]
        self.apply_fn.restype = C.c_int

    def apply_raw(
        self,
        data_ptr,
        data_bytes: int,
        tag: int,
        byteorder: int,
        rect_ptr,
        scalar: float,
        meta_ptr,
        meta_count: int,
        fields_ptr,
        fields_count: int,
        geometry_ptr,
        geometry_count: int,
        values_ptr,
        values_capacity: int,
        written_ptr,
    ) -> int:
        return int(
            self.prepare_fn(
                data_ptr,
                data_bytes,
                tag,
                byteorder,
                rect_ptr,
                scalar,
                meta_ptr,
                meta_count,
                fields_ptr,
                fields_count,
                geometry_ptr,
                geometry_count,
                values_ptr,
                values_capacity,
                written_ptr,
            )
        )

    def prepare(
        self,
        payload: bytes,
        tag: int,
        byteorder: int,
        rect: tuple[int, int, int, int],
        scalar: float,
        *,
        meta_count: int = 23,
        fields_count: int = 19,
        geometry_count: int = 8,
        values_capacity: int | None = None,
    ):
        data_buffer = C.create_string_buffer(payload, len(payload))
        rect_buffer = (C.c_int32 * 4)(*rect)
        parsed = map_oracle.parse_payload(payload, tag, byteorder)
        value_capacity = parsed.count + 3 if values_capacity is None else values_capacity
        meta = np.full(max(meta_count, 0), SENTINEL_META, dtype=np.float64)
        fields = np.full(max(fields_count, 0), SENTINEL_FIELDS, dtype=np.float32)
        geometry = np.full(max(geometry_count, 0), SENTINEL_GEOMETRY, dtype=np.uint32)
        values = np.full(max(value_capacity, 0), SENTINEL_VALUES, dtype=np.float32)
        written = SIZE(SENTINEL_WRITTEN)
        status = self.apply_raw(
            C.cast(data_buffer, U8_PTR),
            len(payload),
            tag,
            byteorder,
            rect_buffer,
            scalar,
            _ptr(meta, C.c_double),
            meta_count,
            _ptr(fields, C.c_float),
            fields_count,
            _ptr(geometry, C.c_uint32),
            geometry_count,
            _ptr(values, C.c_float),
            value_capacity,
            C.pointer(written),
        )
        return status, meta, fields, geometry, values, written.value, data_buffer, rect_buffer


@pytest.fixture(scope="module", params=("source", "official_object"), ids=("source", "official_object"))
def bridge(request, tmp_path_factory):
    path = _compile_bridge(tmp_path_factory.mktemp("acr_profile_gain_map_" + request.param), request.param)
    return MapBridge(path)


def _pack_payload(
    *,
    tag: int,
    byteorder: int,
    rows: int,
    cols: int,
    slices: int,
    values,
    data_type: int = 3,
    spacing_v: float = 1.0,
    spacing_h: float = 1.0,
    origin_v: float = 0.0,
    origin_h: float = 0.0,
    weights=(0.03, 0.07, 0.02, 0.01, 0.04),
    gamma: float = 1.0,
    gain_min: float = 0.25,
    gain_max: float = 2.25,
) -> bytes:
    prefix = ">" if byteorder else "<"
    header = struct.pack(
        prefix + "II4dI5f",
        rows,
        cols,
        spacing_v,
        spacing_h,
        origin_v,
        origin_h,
        slices,
        *weights,
    )
    if tag == map_oracle.TAG_EXTENDED:
        header += struct.pack(prefix + "I3f", data_type, gamma, gain_min, gain_max)
    else:
        assert tag == map_oracle.TAG_LEGACY
        assert data_type == 3

    if data_type == 0:
        body = bytes(int(x) for x in values)
    elif data_type == 1:
        body = struct.pack(prefix + f"{len(values)}H", *(int(x) for x in values))
    elif data_type == 2:
        body = b"".join(struct.pack(prefix + "e", float(x)) for x in values)
    else:
        body = struct.pack(prefix + f"{len(values)}f", *(float(x) for x in values))
    return header + body


def _valid_payloads(seed: int = 0xA2024):
    rng = np.random.default_rng(seed)
    cases = []
    scalar_values = (-2.0, -0.25, 0.0, 0.5, 1.0, 1.375, 4.0)
    for index in range(120):
        tag = map_oracle.TAG_LEGACY if index % 12 == 0 else map_oracle.TAG_EXTENDED
        byteorder = index % 2
        data_type = 3 if tag == map_oracle.TAG_LEGACY else (index // 2) % 4
        rows = (1, 2, 3, 4)[(index // 3) % 4]
        cols = (1, 2, 3, 5, 7)[(index // 5) % 5]
        slices = (1, 2, 3, 4)[(index // 7) % 4]
        count = rows * cols * slices
        if data_type == 0:
            values = rng.integers(0, 256, size=count, dtype=np.uint8).tolist()
        elif data_type == 1:
            values = rng.integers(0, 65536, size=count, dtype=np.uint16).tolist()
        elif data_type == 2:
            values = rng.uniform(0.01, 100.0, size=count).tolist()
        else:
            values = rng.uniform(0.001, 64.0, size=count).tolist()
        spacing_v = float(rng.choice((0.125, 0.3, 0.7, 1.25, 2.5, 7.0)))
        spacing_h = float(rng.choice((0.2, 0.45, 0.9, 1.7, 3.0, 11.0)))
        origin_v = float(rng.choice((-13.25, -2.0, -0.5, 0.25, 1.125, 9.75)))
        origin_h = float(rng.choice((-8.5, -1.0, 0.375, 2.25, 6.5, 17.0)))
        weights = tuple(float(x) for x in rng.uniform(0.005, 0.12, size=5))
        gain_min = float(rng.uniform(0.02, 1.25))
        gain_max = float(rng.uniform(1.35, 6.0))
        rect_top = int(rng.integers(-100, 101))
        rect_left = int(rng.integers(-100, 101))
        rect = (
            rect_top,
            rect_left,
            rect_top + int(rng.integers(1, 18)),
            rect_left + int(rng.integers(1, 23)),
        )
        scalar = scalar_values[index % len(scalar_values)]
        payload = _pack_payload(
            tag=tag,
            byteorder=byteorder,
            rows=rows,
            cols=cols,
            slices=slices,
            values=values,
            data_type=data_type,
            spacing_v=spacing_v,
            spacing_h=spacing_h,
            origin_v=origin_v,
            origin_h=origin_h,
            weights=weights,
            gamma=float((0.125, 0.25, 0.5, 1.0, 2.0, 8.0)[index % 6]),
            gain_min=gain_min,
            gain_max=gain_max,
        )
        cases.append((payload, tag, byteorder, rect, scalar))
    return cases


def _assert_unchanged(arrays, snapshots, written, written_snapshot):
    for array, snapshot in zip(arrays, snapshots):
        assert array.tobytes() == snapshot
    assert int(written.value) == written_snapshot


def _invoke_atomic(
    bridge: MapBridge,
    payload: bytes,
    tag: int,
    byteorder: int,
    rect: tuple[int, int, int, int],
    scalar: float,
    *,
    meta_count: int = 23,
    fields_count: int = 19,
    geometry_count: int = 8,
    values_capacity: int | None = None,
):
    data_buffer = C.create_string_buffer(payload, len(payload))
    rect_buffer = (C.c_int32 * 4)(*rect)
    value_count = max(64, values_capacity or 64)
    meta = np.full(max(23, meta_count), SENTINEL_META, dtype=np.float64)
    fields = np.full(max(19, fields_count), SENTINEL_FIELDS, dtype=np.float32)
    geometry = np.full(max(8, geometry_count), SENTINEL_GEOMETRY, dtype=np.uint32)
    values = np.full(value_count, SENTINEL_VALUES, dtype=np.float32)
    written = SIZE(SENTINEL_WRITTEN)
    arrays = (meta, fields, geometry, values)
    snapshots = tuple(array.tobytes() for array in arrays)
    written_snapshot = int(written.value)
    status = bridge.apply_raw(
        C.cast(data_buffer, U8_PTR),
        len(payload),
        tag,
        byteorder,
        rect_buffer,
        scalar,
        _ptr(meta, C.c_double),
        meta_count,
        _ptr(fields, C.c_float),
        fields_count,
        _ptr(geometry, C.c_uint32),
        geometry_count,
        _ptr(values, C.c_float),
        value_count if values_capacity is None else values_capacity,
        C.pointer(written),
    )
    return status, arrays, snapshots, written, written_snapshot, data_buffer, rect_buffer


def _assert_prepared(bridge: MapBridge, payload, tag, byteorder, rect, scalar):
    reference = map_oracle.payload_to_prepared(payload, tag, byteorder, rect, scalar)
    status, meta, fields, geometry, values, written, *_ = bridge.prepare(
        payload, tag, byteorder, rect, scalar
    )
    assert status == OK
    assert written == reference.table.size
    assert np.array_equal(
        map_oracle.expected_bits64(meta[:19]), map_oracle.expected_bits64(reference.meta)
    ), "meta must match double bit patterns"
    assert np.array_equal(
        map_oracle.expected_bits32(fields[:16]), map_oracle.expected_bits32(reference.fields)
    ), "prepared fields must match binary32 bit patterns"
    assert np.array_equal(geometry[:5], reference.geometry)
    assert np.array_equal(
        map_oracle.expected_bits32(values[:written]), map_oracle.expected_bits32(reference.table)
    ), "decoded gain table must match binary32 bit patterns"
    assert np.all(meta[19:] == SENTINEL_META)
    assert np.all(fields[16:] == SENTINEL_FIELDS)
    assert np.all(geometry[5:] == SENTINEL_GEOMETRY)
    assert np.all(values[written:] == SENTINEL_VALUES)

    # Close the payload -> prepared fields/table -> existing CPU kernel chain
    # against the separate instruction-order scalar oracle.
    outer, inner = 3, 5
    value_stride, coord_stride = 7, 8
    rng_seed = (0x51A0 + written + rect[0] * 17 + rect[1]) & 0xFFFFFFFFFFFFFFFF
    rng = np.random.default_rng(rng_seed)
    inputs = [
        np.full(outer * value_stride, np.float32(-37.0), dtype=np.float32)
        for _ in range(3)
    ]
    inputs += [
        np.full(outer * coord_stride, np.float32(-43.0), dtype=np.float32)
        for _ in range(2)
    ]
    for row in range(outer):
        for col in range(inner):
            for channel in range(3):
                inputs[channel][row * value_stride + col] = np.float32(
                    rng.uniform(-0.25, 0.9)
                )
            inputs[3][row * coord_stride + col] = np.float32(rng.uniform(-1.0, 2.0))
            inputs[4][row * coord_stride + col] = np.float32(rng.uniform(-1.0, 2.0))
    destination = np.full(outer * coord_stride, np.float32(99.0), dtype=np.float32)
    pointer_array = (F32_PTR * 5)(*(_ptr(array, C.c_float) for array in inputs))
    count_array = (SIZE * 5)(*(array.size for array in inputs))
    core_status = int(
        bridge.apply_fn(
            _ptr(fields, C.c_float),
            16,
            _ptr(geometry, C.c_uint32),
            5,
            _ptr(values, C.c_float),
            written,
            pointer_array,
            count_array,
            _ptr(destination, C.c_float),
            destination.size,
            outer,
            inner,
            value_stride,
            coord_stride,
        )
    )
    assert core_status == 0
    expected = cpu_oracle.reference_rows(
        fields[:16],
        geometry[:5],
        values[:written],
        inputs,
        outer,
        inner,
        value_stride,
        coord_stride,
    )
    expected_padded = np.full_like(destination, np.float32(99.0))
    for row in range(outer):
        expected_padded[row * coord_stride : row * coord_stride + inner] = expected[row]
    assert np.array_equal(
        cpu_oracle.bits32(destination), cpu_oracle.bits32(expected_padded)
    ), "prepared payload output and untouched padding must agree with independent CPU kernel oracle"


def test_at_least_100_payload_header_rect_scalar_combinations_on_both_backends(bridge):
    cases = _valid_payloads()
    assert len(cases) >= 100
    assert len({(payload, rect, scalar) for payload, _, _, rect, scalar in cases}) >= 100
    assert {tag for _, tag, _, _, _ in cases} == {map_oracle.TAG_LEGACY, map_oracle.TAG_EXTENDED}
    assert {byteorder for _, _, byteorder, _, _ in cases} == {0, 1}
    assert {map_oracle.parse_payload(payload, tag, byteorder).data_type for payload, tag, byteorder, _, _ in cases} >= {0, 1, 2, 3}
    for payload, tag, byteorder, rect, scalar in cases:
        before = bytes(payload)
        _assert_prepared(bridge, payload, tag, byteorder, rect, scalar)
        assert payload == before, "decoder must not modify caller-owned payload bytes"


def test_singleton_axes_canonicalize_before_ignoring_encoded_nonfinite_values(bridge):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=1,
        rows=1,
        cols=1,
        slices=2,
        values=(0.25, 2.0),
        data_type=3,
        spacing_v=float("nan"),
        spacing_h=float("inf"),
        origin_v=float("-inf"),
        origin_h=float("nan"),
    )
    parsed = map_oracle.parse_payload(payload, map_oracle.TAG_EXTENDED, 1)
    assert (parsed.spacing_v, parsed.origin_v, parsed.spacing_h, parsed.origin_h) == (1.0, 0.0, 1.0, 0.0)
    _assert_prepared(bridge, payload, map_oracle.TAG_EXTENDED, 1, (8, 12, 11, 17), 0.0)


def test_int32_min_origin_negation_wraps_and_affine_uses_distinct_spacing_and_origin(bridge):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=3,
        cols=4,
        slices=2,
        values=(0.3, 0.5, 0.75, 1.0, 1.2, 1.5, 1.75, 2.0, 0.4, 0.6, 0.8, 1.1,
                1.3, 1.6, 1.9, 2.1, 0.35, 0.55, 0.95, 1.4, 1.8, 2.2, 0.45, 0.85),
        data_type=3,
        spacing_v=0.3125,
        spacing_h=1.875,
        origin_v=-7.25,
        origin_h=3.625,
        weights=(0.03125, 0.0625, 0.09375, 0.015625, 0.046875),
        gamma=1.25,
    )
    rect = (-(1 << 31), -(1 << 31), -(1 << 31) + 5, -(1 << 31) + 9)
    prepared = map_oracle.payload_to_prepared(payload, map_oracle.TAG_EXTENDED, 0, rect, -1.75)
    assert map_oracle._wrap_neg32(rect[0]) == -(1 << 31)
    assert map_oracle._wrap_neg32(rect[1]) == -(1 << 31)
    assert prepared.fields[5] != prepared.fields[7]
    assert prepared.fields[6] != prepared.fields[8]
    _assert_prepared(bridge, payload, map_oracle.TAG_EXTENDED, 0, rect, -1.75)


def test_integer_decode_uses_separate_reciprocal_multiply_then_fmaf(bridge):
    # This input differs by one ULP if the final FMA is accidentally split.
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=1,
        cols=1,
        slices=1,
        values=(115,),
        data_type=0,
        gain_min=53.54279327392578,
        gain_max=882.0284423828125,
    )
    parsed = map_oracle.parse_payload(payload, map_oracle.TAG_EXTENDED, 0)
    actual = parsed.table[0]
    reciprocal = np.asarray([0x3B808081], dtype=np.uint32).view(np.float32)[0]
    normalized = np.float32(np.float32(115.0) * reciprocal)
    delta = np.float32(np.float32(882.0284423828125) - np.float32(53.54279327392578))
    split = np.float32(np.float32(normalized * delta) + np.float32(53.54279327392578))
    assert int(actual.view(np.uint32)) != int(split.view(np.uint32))
    assert int(actual.view(np.uint32)) == 0x43D59639
    _assert_prepared(bridge, payload, map_oracle.TAG_EXTENDED, 0, (2, 3, 5, 8), -0.5)


def test_reversed_gain_min_max_is_not_rejected_or_clamped(bridge):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=1,
        rows=1,
        cols=2,
        slices=1,
        values=(0, 128),
        data_type=0,
        gain_min=500.0,
        gain_max=2.0,
    )
    prepared = map_oracle.payload_to_prepared(payload, map_oracle.TAG_EXTENDED, 1, (0, 0, 3, 4), 1.0)
    assert prepared.table[0] == np.float32(500.0)
    assert prepared.table[1] != np.float32(2.0)
    _assert_prepared(bridge, payload, map_oracle.TAG_EXTENDED, 1, (0, 0, 3, 4), 1.0)


def test_every_valid_binary16_gain_code_and_invalid_representatives(bridge):
    codes = tuple(range(0x0C00, 0x6C01))
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=1,
        rows=1,
        cols=len(codes),
        slices=1,
        values=tuple(struct.unpack(">e", struct.pack(">H", code))[0] for code in codes),
        data_type=2,
    )
    expected = np.fromiter(
        (struct.unpack(">e", struct.pack(">H", code))[0] for code in codes),
        dtype=np.float32,
        count=len(codes),
    )
    parsed = map_oracle.parse_payload(payload, map_oracle.TAG_EXTENDED, 1)
    assert np.array_equal(map_oracle.expected_bits32(parsed.table), map_oracle.expected_bits32(expected))
    status, _, _, _, values, written, *_ = bridge.prepare(
        payload,
        map_oracle.TAG_EXTENDED,
        1,
        (0, 0, 1, len(codes)),
        1.0,
        values_capacity=len(codes) + 3,
    )
    assert status == OK
    assert written == len(codes)
    assert np.array_equal(map_oracle.expected_bits32(values[:written]), map_oracle.expected_bits32(expected))
    assert np.all(values[written:] == SENTINEL_VALUES)

    for code in (0x0000, 0x0001, 0x8000, 0x7C00, 0x7E01, 0xFC00, 0x0BFF, 0x6C01):
        bad = _pack_payload(
            tag=map_oracle.TAG_EXTENDED,
            byteorder=1,
            rows=1,
            cols=1,
            slices=1,
            values=(struct.unpack(">e", struct.pack(">H", code))[0],),
            data_type=2,
        )
        status, arrays, snapshots, written_obj, written_snapshot, *_ = _invoke_atomic(
            bridge, bad, map_oracle.TAG_EXTENDED, 1, (0, 0, 1, 1), 1.0
        )
        assert status == INVALID_GAIN_VALUE
        _assert_unchanged(arrays, snapshots, written_obj, written_snapshot)


def test_float32_boundaries_are_accepted_and_invalid_gain_values_fail_atomically(bridge):
    for value in (2.0**-12, 4096.0):
        payload = _pack_payload(
            tag=map_oracle.TAG_EXTENDED,
            byteorder=0,
            rows=1,
            cols=1,
            slices=1,
            values=(value,),
            data_type=3,
        )
        _assert_prepared(bridge, payload, map_oracle.TAG_EXTENDED, 0, (0, 0, 2, 2), 1.0)
    for value in (0.0, -1.0, float("nan"), float("inf"), -float("inf"), 2.0**-13, 4096.5):
        payload = _pack_payload(
            tag=map_oracle.TAG_EXTENDED,
            byteorder=0,
            rows=1,
            cols=1,
            slices=1,
            values=(value,),
            data_type=3,
        )
        status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
            bridge, payload, map_oracle.TAG_EXTENDED, 0, (0, 0, 1, 1), 1.0
        )
        assert status == INVALID_GAIN_VALUE
        _assert_unchanged(arrays, snapshots, written, written_snapshot)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"rows": 0}, INVALID_HEADER),
        ({"cols": 0}, INVALID_HEADER),
        ({"slices": 0}, INVALID_HEADER),
        ({"spacing_v": 0.0}, INVALID_HEADER),
        ({"spacing_h": -1.0}, INVALID_HEADER),
        ({"spacing_v": float("inf")}, INVALID_HEADER),
        ({"origin_h": float("nan")}, INVALID_HEADER),
        ({"weights": (0.1, 0.2, float("nan"), 0.3, 0.4)}, INVALID_HEADER),
        ({"data_type": 4}, UNSUPPORTED_TYPE),
        ({"gamma": 0.124}, INVALID_GAIN_VALUE),
        ({"gamma": 8.001}, INVALID_GAIN_VALUE),
        ({"gamma": float("nan")}, INVALID_GAIN_VALUE),
        ({"gain_min": 2.0**-13}, INVALID_GAIN_VALUE),
        ({"gain_min": float("nan")}, INVALID_GAIN_VALUE),
        ({"gain_max": 4097.0}, INVALID_GAIN_VALUE),
        ({"gain_max": float("inf")}, INVALID_GAIN_VALUE),
    ],
)
def test_invalid_header_fields_fail_atomically(bridge, kwargs, expected):
    arguments = dict(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=2,
        cols=2,
        slices=1,
        values=(0.5, 0.75, 1.0, 1.25),
        data_type=3,
    )
    arguments.update(kwargs)
    payload = _pack_payload(**arguments)
    status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
        bridge, payload, map_oracle.TAG_EXTENDED, 0, (0, 0, 2, 2), 1.0
    )
    assert status == expected
    _assert_unchanged(arrays, snapshots, written, written_snapshot)


def test_legacy_header_defaults_and_bad_lengths_are_distinguished(bridge):
    valid = _pack_payload(
        tag=map_oracle.TAG_LEGACY,
        byteorder=1,
        rows=2,
        cols=1,
        slices=2,
        values=(0.5, 1.0, 1.5, 2.0),
        data_type=3,
        spacing_v=17.0,
        spacing_h=float("nan"),
        origin_v=2.0,
        origin_h=float("inf"),
        gamma=7.0,
        gain_min=99.0,
        gain_max=101.0,
    )
    parsed = map_oracle.parse_payload(valid, map_oracle.TAG_LEGACY, 1)
    assert parsed.data_type == 3
    assert (parsed.gamma, parsed.gain_min, parsed.gain_max) == (1.0, 1.0, 1.0)
    assert parsed.spacing_h == 1.0 and parsed.origin_h == 0.0
    _assert_prepared(bridge, valid, map_oracle.TAG_LEGACY, 1, (2, 3, 5, 8), 0.25)

    for malformed in (valid[:-1], valid + b"\0"):
        status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
            bridge, malformed, map_oracle.TAG_LEGACY, 1, (2, 3, 5, 8), 0.25
        )
        assert status == INVALID_LENGTH
        _assert_unchanged(arrays, snapshots, written, written_snapshot)
    status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
        bridge, b"\x00" * 12, map_oracle.TAG_LEGACY, 1, (0, 0, 1, 1), 1.0
    )
    assert status == INVALID_LENGTH
    _assert_unchanged(arrays, snapshots, written, written_snapshot)


def test_multiplication_overflow_and_resource_limit_are_rejected_before_length(bridge):
    overflow = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=0x7FFFFFFF,
        cols=3,
        slices=2,
        values=(),
        data_type=3,
    )
    status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
        bridge, overflow, map_oracle.TAG_EXTENDED, 0, (0, 0, 1, 1), 1.0
    )
    assert status == INTEGER_OVERFLOW
    _assert_unchanged(arrays, snapshots, written, written_snapshot)

    resource = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=1,
        rows=1,
        cols=1,
        slices=1 << 24,
        values=(),
        data_type=3,
    )
    status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
        bridge, resource, map_oracle.TAG_EXTENDED, 1, (0, 0, 1, 1), 1.0
    )
    assert status == RESOURCE_LIMIT
    _assert_unchanged(arrays, snapshots, written, written_snapshot)


@pytest.mark.parametrize(
    ("capacity", "expected"),
    [("meta", SHORT_OUTPUT), ("fields", SHORT_OUTPUT), ("geometry", SHORT_OUTPUT), ("values", SHORT_OUTPUT)],
)
def test_short_output_capacity_is_atomic(bridge, capacity, expected):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=1,
        cols=2,
        slices=2,
        values=(0.5, 0.75, 1.0, 1.25),
        data_type=3,
    )
    kwargs = {
        "meta_count": 23,
        "fields_count": 19,
        "geometry_count": 8,
        "values_capacity": 8,
    }
    if capacity == "meta":
        kwargs["meta_count"] = 18
    elif capacity == "fields":
        kwargs["fields_count"] = 15
    elif capacity == "geometry":
        kwargs["geometry_count"] = 4
    else:
        kwargs["values_capacity"] = 3
    status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
        bridge,
        payload,
        map_oracle.TAG_EXTENDED,
        0,
        (0, 0, 2, 2),
        1.0,
        meta_count=kwargs["meta_count"],
        fields_count=kwargs["fields_count"],
        geometry_count=kwargs["geometry_count"],
        values_capacity=kwargs["values_capacity"],
    )
    assert status == expected
    _assert_unchanged(arrays, snapshots, written, written_snapshot)


@pytest.mark.parametrize(
    "rect",
    [
        (0, 0, 0, 3),
        (0, 4, 2, 4),
        (-(1 << 31), 0, (1 << 31) - 1, 4),
        (0, -1, 1, (1 << 31) - 1),
    ],
)
def test_invalid_source_rectangles_return_geometry_status_atomically(bridge, rect):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=2,
        cols=2,
        slices=1,
        values=(0.5, 0.75, 1.0, 1.25),
        data_type=3,
    )
    status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
        bridge, payload, map_oracle.TAG_EXTENDED, 0, rect, 1.0
    )
    assert status == INVALID_GEOMETRY
    _assert_unchanged(arrays, snapshots, written, written_snapshot)


@pytest.mark.parametrize("scalar", [float("inf"), float("nan"), 1e100, -1e100])
def test_nonfinite_or_unrepresentable_scalar_is_numeric_failure(bridge, scalar):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=2,
        cols=2,
        slices=1,
        values=(0.5, 0.75, 1.0, 1.25),
        data_type=3,
    )
    status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
        bridge, payload, map_oracle.TAG_EXTENDED, 0, (0, 0, 2, 3), scalar
    )
    assert status == NUMERIC_FAILURE
    _assert_unchanged(arrays, snapshots, written, written_snapshot)


def test_nonfinite_matrix_from_finite_positive_subnormal_spacing_is_numeric_failure(bridge):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=2,
        cols=2,
        slices=1,
        values=(0.5, 0.75, 1.0, 1.25),
        data_type=3,
        spacing_h=5e-324,
    )
    status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
        bridge, payload, map_oracle.TAG_EXTENDED, 0, (0, 0, 2, 3), 1.0
    )
    assert status == NUMERIC_FAILURE
    _assert_unchanged(arrays, snapshots, written, written_snapshot)


def test_invalid_tag_byteorder_and_null_input_are_invalid_arguments(bridge):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=1,
        cols=1,
        slices=1,
        values=(1.0,),
        data_type=3,
    )
    for tag, byteorder in ((999, 0), (map_oracle.TAG_EXTENDED, 2)):
        status, arrays, snapshots, written, written_snapshot, *_ = _invoke_atomic(
            bridge, payload, tag, byteorder, (0, 0, 1, 1), 1.0
        )
        assert status == INVALID_ARGUMENT
        _assert_unchanged(arrays, snapshots, written, written_snapshot)

    meta = np.full(23, SENTINEL_META, dtype=np.float64)
    fields = np.full(19, SENTINEL_FIELDS, dtype=np.float32)
    geometry = np.full(8, SENTINEL_GEOMETRY, dtype=np.uint32)
    values = np.full(4, SENTINEL_VALUES, dtype=np.float32)
    rect = (C.c_int32 * 4)(0, 0, 1, 1)
    written = SIZE(SENTINEL_WRITTEN)
    arrays = (meta, fields, geometry, values)
    snapshots = tuple(array.tobytes() for array in arrays)
    status = bridge.apply_raw(
        C.cast(C.c_void_p(0), U8_PTR),
        0,
        map_oracle.TAG_EXTENDED,
        0,
        rect,
        1.0,
        _ptr(meta, C.c_double),
        meta.size,
        _ptr(fields, C.c_float),
        fields.size,
        _ptr(geometry, C.c_uint32),
        geometry.size,
        _ptr(values, C.c_float),
        values.size,
        C.pointer(written),
    )
    assert status == INVALID_ARGUMENT
    _assert_unchanged(arrays, snapshots, written, SENTINEL_WRITTEN)


def test_unaligned_rect_and_output_pointer_are_rejected_atomically(bridge):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=1,
        cols=1,
        slices=1,
        values=(1.0,),
        data_type=3,
    )
    data_buffer = C.create_string_buffer(payload, len(payload))
    rect_storage = C.create_string_buffer(4 * C.sizeof(C.c_int32) + 1)
    rect_aligned = (C.c_int32 * 4)(0, 0, 1, 1)
    rect_unaligned = C.cast(C.addressof(rect_storage) + 1, I32_PTR)
    meta = np.full(23, SENTINEL_META, dtype=np.float64)
    fields = np.full(19, SENTINEL_FIELDS, dtype=np.float32)
    geometry = np.full(8, SENTINEL_GEOMETRY, dtype=np.uint32)
    values = np.full(4, SENTINEL_VALUES, dtype=np.float32)
    written = SIZE(SENTINEL_WRITTEN)
    arrays = (meta, fields, geometry, values)
    snapshots = tuple(array.tobytes() for array in arrays)
    status = bridge.apply_raw(
        C.cast(data_buffer, U8_PTR), len(payload), map_oracle.TAG_EXTENDED, 0,
        rect_unaligned, 1.0, _ptr(meta, C.c_double), meta.size,
        _ptr(fields, C.c_float), fields.size, _ptr(geometry, C.c_uint32), geometry.size,
        _ptr(values, C.c_float), values.size, C.pointer(written),
    )
    assert status == ALIGNMENT
    _assert_unchanged(arrays, snapshots, written, SENTINEL_WRITTEN)

    unaligned_meta_storage = C.create_string_buffer(meta.nbytes + 1)
    unaligned_meta = C.cast(C.addressof(unaligned_meta_storage) + 1, F64_PTR)
    status = bridge.apply_raw(
        C.cast(data_buffer, U8_PTR), len(payload), map_oracle.TAG_EXTENDED, 0,
        rect_aligned, 1.0, unaligned_meta, meta.size,
        _ptr(fields, C.c_float), fields.size, _ptr(geometry, C.c_uint32), geometry.size,
        _ptr(values, C.c_float), values.size, C.pointer(written),
    )
    assert status == ALIGNMENT
    assert C.string_at(C.addressof(unaligned_meta_storage), len(unaligned_meta_storage)) == bytes(
        len(unaligned_meta_storage)
    )
    _assert_unchanged((fields, geometry, values), (snapshots[1], snapshots[2], snapshots[3]), written, SENTINEL_WRITTEN)


def test_data_alias_and_output_alias_are_rejected_without_mutation(bridge):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=1,
        cols=1,
        slices=1,
        values=(1.0,),
        data_type=3,
    )
    data_storage = C.create_string_buffer(payload, len(payload))
    data_before = C.string_at(C.addressof(data_storage), len(payload))
    rect = (C.c_int32 * 4)(0, 0, 1, 1)
    meta = np.full(23, SENTINEL_META, dtype=np.float64)
    fields = np.full(19, SENTINEL_FIELDS, dtype=np.float32)
    geometry = np.full(8, SENTINEL_GEOMETRY, dtype=np.uint32)
    written = SIZE(SENTINEL_WRITTEN)
    arrays = (meta, fields, geometry)
    snapshots = tuple(array.tobytes() for array in arrays)
    values_alias = C.cast(C.addressof(data_storage), F32_PTR)
    assert C.addressof(data_storage) % C.alignment(C.c_float) == 0
    status = bridge.apply_raw(
        C.cast(data_storage, U8_PTR), len(payload), map_oracle.TAG_EXTENDED, 0,
        rect, 1.0, _ptr(meta, C.c_double), meta.size,
        _ptr(fields, C.c_float), fields.size, _ptr(geometry, C.c_uint32), geometry.size,
        values_alias, 1, C.pointer(written),
    )
    assert status == ALIAS
    assert C.string_at(C.addressof(data_storage), len(payload)) == data_before
    _assert_unchanged(arrays, snapshots, written, SENTINEL_WRITTEN)

    rect_storage = (C.c_int32 * 4)(0, 0, 1, 1)
    rect_before = C.string_at(C.addressof(rect_storage), C.sizeof(rect_storage))
    values_alias = C.cast(C.addressof(rect_storage), F32_PTR)
    status = bridge.apply_raw(
        C.cast(data_storage, U8_PTR), len(payload), map_oracle.TAG_EXTENDED, 0,
        rect_storage, 1.0, _ptr(meta, C.c_double), meta.size,
        _ptr(fields, C.c_float), fields.size, _ptr(geometry, C.c_uint32), geometry.size,
        values_alias, 1, C.pointer(written),
    )
    assert status == ALIAS
    assert C.string_at(C.addressof(rect_storage), C.sizeof(rect_storage)) == rect_before
    _assert_unchanged(arrays, snapshots, written, SENTINEL_WRITTEN)

    # Two nominal outputs share storage; the function must detect this before
    # committing either region.
    shared = np.full(32, SENTINEL_META, dtype=np.float64)
    shared_before = shared.tobytes()
    fields_alias = C.cast(shared.ctypes.data, F32_PTR)
    values = np.full(4, SENTINEL_VALUES, dtype=np.float32)
    geometry = np.full(8, SENTINEL_GEOMETRY, dtype=np.uint32)
    written = SIZE(SENTINEL_WRITTEN)
    snapshots = (values.tobytes(), geometry.tobytes())
    status = bridge.apply_raw(
        C.cast(data_storage, U8_PTR), len(payload), map_oracle.TAG_EXTENDED, 0,
        rect, 1.0, _ptr(shared, C.c_double), shared.size,
        fields_alias, 19, _ptr(geometry, C.c_uint32), geometry.size,
        _ptr(values, C.c_float), values.size, C.pointer(written),
    )
    assert status == ALIAS
    assert shared.tobytes() == shared_before
    _assert_unchanged((values, geometry), snapshots, written, SENTINEL_WRITTEN)


def test_pointer_span_overflow_is_rejected_before_dereference(bridge):
    payload = _pack_payload(
        tag=map_oracle.TAG_EXTENDED,
        byteorder=0,
        rows=1,
        cols=1,
        slices=1,
        values=(1.0,),
        data_type=3,
    )
    data_buffer = C.create_string_buffer(payload, len(payload))
    rect = (C.c_int32 * 4)(0, 0, 1, 1)
    invalid_meta = C.cast(C.c_void_p((1 << (C.sizeof(C.c_void_p) * 8)) - 8), F64_PTR)
    fields = np.full(19, SENTINEL_FIELDS, dtype=np.float32)
    geometry = np.full(8, SENTINEL_GEOMETRY, dtype=np.uint32)
    values = np.full(4, SENTINEL_VALUES, dtype=np.float32)
    written = SIZE(SENTINEL_WRITTEN)
    arrays = (fields, geometry, values)
    snapshots = tuple(array.tobytes() for array in arrays)
    status = bridge.apply_raw(
        C.cast(data_buffer, U8_PTR), len(payload), map_oracle.TAG_EXTENDED, 0,
        rect, 1.0, invalid_meta, 19,
        _ptr(fields, C.c_float), fields.size, _ptr(geometry, C.c_uint32), geometry.size,
        _ptr(values, C.c_float), values.size, C.pointer(written),
    )
    assert status == POINTER_SPAN_OVERFLOW
    _assert_unchanged(arrays, snapshots, written, SENTINEL_WRITTEN)
