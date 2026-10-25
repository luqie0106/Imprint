"""Independent oracle tests for the PGTM tail-consumer kernel and bridge."""

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

MAP_SOURCE = NATIVE / "src" / "acr_profile_gain_map.cpp"
CORE_SOURCE = NATIVE / "src" / "acr_profile_gain.cpp"
APPLY_SOURCE = NATIVE / "src" / "acr_profile_gain_apply.cpp"
MAP_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_map_bridge.cpp"
CORE_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_bridge.cpp"
APPLY_BRIDGE_SOURCE = NATIVE / "tools" / "acr_profile_gain_apply_bridge.cpp"

OBJECT_SUFFIX = ".obj" if sys.platform == "win32" else ".o"
OFFICIAL_OBJECTS = (
    NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/acr_profile_gain_map.cpp{OBJECT_SUFFIX}",
    NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/acr_profile_gain.cpp{OBJECT_SUFFIX}",
    NATIVE / f"build/CMakeFiles/imprint_renderer.dir/src/acr_profile_gain_apply.cpp{OBJECT_SUFFIX}",
)

sys.path.insert(0, str(SUPPORT))
import profile_gain_apply_oracle as apply_oracle  # noqa: E402
import profile_gain_map_oracle as map_oracle  # noqa: E402
import profile_gain_oracle as scalar_oracle  # noqa: E402


U8_PTR = C.POINTER(C.c_uint8)
I32_PTR = C.POINTER(C.c_int32)
U32_PTR = C.POINTER(C.c_uint32)
F32_PTR = C.POINTER(C.c_float)
F64_PTR = C.POINTER(C.c_double)
SIZE = C.c_size_t
SIZE_PTR = C.POINTER(SIZE)
F32_PTR_PTR = C.POINTER(F32_PTR)
SIZE_MAX = SIZE(-1).value
PTRDIFF_MAX = (1 << (C.sizeof(C.c_void_p) * 8 - 1)) - 1

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

SENTINEL = np.float32(-731.625)
SENTINEL_META = np.float64(-991.125)
SENTINEL_FIELDS = np.float32(-731.625)
SENTINEL_GEOMETRY = np.uint32(0xA5A5A5A5)
SENTINEL_TABLE = np.float32(-317.25)


def _float_ptr(array: np.ndarray | None):
    if array is None:
        return C.cast(C.c_void_p(0), F32_PTR)
    return array.ctypes.data_as(F32_PTR)


def _byte_ptr(array: bytes | bytearray | C.Array | None):
    if array is None:
        return C.cast(C.c_void_p(0), U8_PTR)
    return C.cast(array, U8_PTR)


def _compile_bridge(tmp_path: Path, kind: str) -> Path:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required to compile the ProfileGain apply bridge"
    for bridge_source in (MAP_BRIDGE_SOURCE, CORE_BRIDGE_SOURCE, APPLY_BRIDGE_SOURCE):
        assert bridge_source.is_file(), f"missing ProfileGain bridge source: {bridge_source}"
    for source in (MAP_SOURCE, CORE_SOURCE, APPLY_SOURCE):
        assert source.is_file(), f"missing ProfileGain implementation source: {source}"

    if kind == "source":
        implementations = (MAP_SOURCE, CORE_SOURCE, APPLY_SOURCE)
    else:
        assert kind == "official_object"
        missing = [path for path in OFFICIAL_OBJECTS if not path.is_file()]
        if missing:
            pytest.skip("official ProfileGain objects are unavailable: " + ", ".join(map(str, missing)))
        stale = [
            (obj, source)
            for obj, source in zip(OFFICIAL_OBJECTS, (MAP_SOURCE, CORE_SOURCE, APPLY_SOURCE))
            if obj.stat().st_mtime_ns < source.stat().st_mtime_ns
        ]
        if stale:
            pytest.skip(
                "official ProfileGain object is stale: "
                + ", ".join(obj.name for obj, _ in stale)
            )
        implementations = OFFICIAL_OBJECTS

    if sys.platform == "darwin":
        suffix = ".dylib"
        shared_flags = ["-dynamiclib"]
    elif sys.platform == "win32":
        suffix = ".dll"
        shared_flags = ["-shared"]
    else:
        suffix = ".so"
        shared_flags = ["-shared", "-fPIC"]

    output = tmp_path / f"libprofile_gain_apply_{kind}{suffix}"
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
        str(MAP_BRIDGE_SOURCE),
        str(CORE_BRIDGE_SOURCE),
        str(APPLY_BRIDGE_SOURCE),
        *map(str, implementations),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


class Bridge:
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

        self.scalar_evaluator_fn = self.library.acr_profile_gain_bridge
        self.scalar_evaluator_fn.argtypes = [
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
        self.scalar_evaluator_fn.restype = C.c_int

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

    def prepare(
        self,
        payload: bytes,
        tag: int,
        byteorder: int,
        rect: tuple[int, int, int, int],
        scalar: float,
    ):
        data_buffer = C.create_string_buffer(payload, len(payload))
        rect_buffer = (C.c_int32 * 4)(*rect)
        meta = np.full(23, SENTINEL_META, dtype=np.float64)
        fields = np.full(19, SENTINEL_FIELDS, dtype=np.float32)
        geometry = np.full(8, SENTINEL_GEOMETRY, dtype=np.uint32)
        values = np.full(128, SENTINEL_TABLE, dtype=np.float32)
        written = SIZE(0x12345678)
        status = int(
            self.prepare_fn(
                _byte_ptr(data_buffer),
                len(payload),
                tag,
                byteorder,
                rect_buffer,
                scalar,
                meta.ctypes.data_as(F64_PTR),
                meta.size,
                fields.ctypes.data_as(F32_PTR),
                fields.size,
                geometry.ctypes.data_as(U32_PTR),
                geometry.size,
                values.ctypes.data_as(F32_PTR),
                values.size,
                C.pointer(written),
            )
        )
        return status, meta, fields, geometry, values, int(written.value), data_buffer, rect_buffer

    def evaluate_scalar(
        self,
        fields: np.ndarray,
        geometry: np.ndarray,
        table: np.ndarray,
        inputs: list[np.ndarray],
        output: np.ndarray,
        *,
        outer: int,
        inner: int,
        value_stride: int,
        coord_stride: int,
    ) -> int:
        pointer_array = (F32_PTR * 5)(*(_float_ptr(value) for value in inputs))
        count_array = (SIZE * 5)(*(value.size for value in inputs))
        return int(
            self.scalar_evaluator_fn(
                fields.ctypes.data_as(F32_PTR),
                16,
                geometry.ctypes.data_as(U32_PTR),
                5,
                table.ctypes.data_as(F32_PTR),
                table.size,
                pointer_array,
                count_array,
                output.ctypes.data_as(F32_PTR),
                output.size,
                outer,
                inner,
                value_stride,
                coord_stride,
            )
        )

    def apply_raw(
        self,
        input_ptrs,
        input_counts,
        output_ptrs,
        output_counts,
        outer: int,
        inner: int,
        value_stride: int,
        gain_stride: int,
        pivot: float,
        mode: int,
        clip: int,
        null_input_ptrs: bool = False,
        null_input_counts: bool = False,
        null_output_ptrs: bool = False,
        null_output_counts: bool = False,
    ) -> int:
        null_f32_views = C.cast(C.c_void_p(0), F32_PTR_PTR)
        null_sizes = C.cast(C.c_void_p(0), SIZE_PTR)
        return int(
            self.apply_fn(
                null_f32_views if null_input_ptrs else input_ptrs,
                null_sizes if null_input_counts else input_counts,
                null_f32_views if null_output_ptrs else output_ptrs,
                null_sizes if null_output_counts else output_counts,
                outer,
                inner,
                value_stride,
                gain_stride,
                pivot,
                mode,
                clip,
            )
        )


@pytest.fixture(scope="module", params=("source", "official_object"), ids=("source", "official_object"))
def bridge(request, tmp_path_factory):
    path = _compile_bridge(
        tmp_path_factory.mktemp("acr_profile_gain_apply_" + request.param), request.param
    )
    return Bridge(path)


def _views(inputs, outputs, input_counts=None, output_counts=None):
    input_ptrs = (F32_PTR * 4)(*(_float_ptr(value) for value in inputs))
    output_ptrs = (F32_PTR * 3)(*(_float_ptr(value) for value in outputs))
    in_counts = [0 if value is None else value.size for value in inputs]
    out_counts = [0 if value is None else value.size for value in outputs]
    if input_counts is not None:
        in_counts = [int(value) for value in input_counts]
    if output_counts is not None:
        out_counts = [int(value) for value in output_counts]
    input_count_array = (SIZE * 4)(*in_counts)
    output_count_array = (SIZE * 3)(*out_counts)
    return input_ptrs, input_count_array, output_ptrs, output_count_array


def _call(
    bridge: Bridge,
    inputs,
    outputs,
    *,
    outer=1,
    inner=1,
    value_stride=1,
    gain_stride=1,
    pivot=0.5,
    mode=apply_oracle.SCALAR_MODE,
    clip=0,
    input_counts=None,
    output_counts=None,
    null_input_ptrs=False,
    null_input_counts=False,
    null_output_ptrs=False,
    null_output_counts=False,
):
    input_ptrs, in_counts, output_ptrs, out_counts = _views(
        inputs, outputs, input_counts=input_counts, output_counts=output_counts
    )
    status = bridge.apply_raw(
        input_ptrs,
        in_counts,
        output_ptrs,
        out_counts,
        outer=outer,
        inner=inner,
        value_stride=value_stride,
        gain_stride=gain_stride,
        pivot=pivot,
        mode=mode,
        clip=clip,
        null_input_ptrs=null_input_ptrs,
        null_input_counts=null_input_counts,
        null_output_ptrs=null_output_ptrs,
        null_output_counts=null_output_counts,
    )
    return status, (input_ptrs, in_counts, output_ptrs, out_counts)


def _case_arrays(outer: int, inner: int, value_stride: int, gain_stride: int):
    inputs = [
        np.full(outer * value_stride, SENTINEL, dtype=np.float32) for _ in range(3)
    ]
    inputs.append(np.full(outer * gain_stride, SENTINEL, dtype=np.float32))
    outputs = [
        np.full(outer * value_stride, SENTINEL, dtype=np.float32) for _ in range(3)
    ]
    return inputs, outputs


def _assert_outputs(actual, expected):
    for channel, (got, want) in enumerate(zip(actual, expected)):
        try:
            apply_oracle.assert_float32_results(got, want)
        except AssertionError as exc:
            raise AssertionError(f"output channel {channel}: {exc}") from exc


def _bit_pattern(value: int) -> np.float32:
    return np.asarray([value], dtype=np.uint32).view(np.float32)[0]


def _active_random_inputs(rng, outer, inner, value_stride, gain_stride):
    inputs, outputs = _case_arrays(outer, inner, value_stride, gain_stride)
    for row in range(outer):
        for col in range(inner):
            offset = row * value_stride + col
            for channel in range(3):
                inputs[channel][offset] = np.float32(rng.uniform(-4.0, 5.0))
            inputs[3][row * gain_stride + col] = np.float32(rng.uniform(-3.5, 4.5))
    return inputs, outputs


def test_one_hundred_independent_random_states_all_mode_clip_pairs(bridge):
    rng = np.random.default_rng(0x5047544D)
    seen = set()
    for state in range(100):
        outer = int(rng.integers(1, 6))
        inner = int(2 * rng.integers(0, 18) + 1)  # every tested width is odd
        value_stride = inner + int(rng.integers(0, 9))
        gain_stride = inner + int(rng.integers(0, 8))
        pivot = float(rng.uniform(-1.75, 3.25))
        if state % 17 == 0:
            pivot = 0.0
        elif state % 19 == 0:
            pivot = 1.0
        signature = (outer, inner, value_stride, gain_stride, np.float64(pivot).view(np.uint64).item())
        seen.add(signature)
        inputs, _ = _active_random_inputs(rng, outer, inner, value_stride, gain_stride)
        source_snapshots = [item.copy() for item in inputs]

        for mode in (apply_oracle.SCALAR_MODE, apply_oracle.CPU_VECTOR_MODE):
            for clip in (0, 1):
                outputs = [
                    np.full(outer * value_stride, SENTINEL, dtype=np.float32)
                    for _ in range(3)
                ]
                status, _ = _call(
                    bridge,
                    inputs,
                    outputs,
                    outer=outer,
                    inner=inner,
                    value_stride=value_stride,
                    gain_stride=gain_stride,
                    pivot=pivot,
                    mode=mode,
                    clip=clip,
                )
                assert status == OK, f"random state {state}, mode={mode}, clip={clip}"
                expected = apply_oracle.reference_rows(
                    source_snapshots,
                    outer,
                    inner,
                    value_stride,
                    gain_stride,
                    pivot,
                    mode,
                    clip,
                )
                _assert_outputs(outputs, expected)
        for before, after in zip(source_snapshots, inputs):
            np.testing.assert_array_equal(apply_oracle.bits32(after), apply_oracle.bits32(before))
    assert len(seen) >= 100, "randomized geometries/pivots must be independent states"


def test_hardcoded_fma_and_add_order_counterexamples(bridge):
    # Each lane has a fixed binary32 input chosen from the original operation
    # sequence. The alternate bits are from splitting one FMA, splitting the
    # final FMA, and reassociating gain*factor, respectively.
    p_bits = (0x3E84D145, 0x3FC23C2F, 0x40D6772B)
    gain_bits = (0xC0A33900, 0xBFBA1D7A, 0x40151BDA)
    expected_bits = (0xBF0F8247, 0xBFF32881, 0x41772C84)
    wrong_bits = (0xBF0F8246, 0xBFF32880, 0x41772C85)
    assert all(good != wrong for good, wrong in zip(expected_bits, wrong_bits))

    inner, value_stride, gain_stride = 3, 5, 4
    inputs, outputs = _case_arrays(1, inner, value_stride, gain_stride)
    for col in range(inner):
        p = _bit_pattern(p_bits[col])
        for channel in range(3):
            inputs[channel][col] = p
        inputs[3][col] = _bit_pattern(gain_bits[col])
    status, _ = _call(
        bridge,
        inputs,
        outputs,
        inner=inner,
        value_stride=value_stride,
        gain_stride=gain_stride,
        pivot=0.125,
        mode=apply_oracle.SCALAR_MODE,
        clip=0,
    )
    assert status == OK
    expected = apply_oracle.reference_rows(
        inputs, 1, inner, value_stride, gain_stride, 0.125, 0, 0
    )
    _assert_outputs(outputs, expected)
    for actual in outputs:
        np.testing.assert_array_equal(
            apply_oracle.bits32(actual[:inner]), np.asarray(expected_bits, dtype=np.uint32)
        )
        assert not np.array_equal(
            apply_oracle.bits32(actual[:inner]), np.asarray(wrong_bits, dtype=np.uint32)
        )


def test_pivot_zero_upper_clip_and_pivot_one_select(bridge):
    inputs, outputs = _case_arrays(1, 3, 5, 4)
    inputs[0][:3] = np.asarray([np.finfo(np.float32).max, -2.0, 0.25], dtype=np.float32)
    inputs[1][:3] = inputs[0][:3]
    inputs[2][:3] = inputs[0][:3]
    inputs[3][:3] = np.asarray([np.finfo(np.float32).max, 1.0, 4.0], dtype=np.float32)

    for mode in (apply_oracle.SCALAR_MODE, apply_oracle.CPU_VECTOR_MODE):
        status, _ = _call(
            bridge,
            inputs,
            outputs,
            inner=3,
            value_stride=5,
            gain_stride=4,
            pivot=0.0,
            mode=mode,
            clip=1,
        )
        assert status == OK
        expected = apply_oracle.reference_rows(inputs, 1, 3, 5, 4, 0.0, mode, 1)
        _assert_outputs(outputs, expected)
        np.testing.assert_array_equal(
            apply_oracle.bits32(outputs[0][:3]),
            np.asarray([0x3F800000, 0xC0000000, 0x3F800000], dtype=np.uint32),
        )

        for output in outputs:
            output[:] = SENTINEL
        status, _ = _call(
            bridge,
            inputs,
            outputs,
            inner=3,
            value_stride=5,
            gain_stride=4,
            pivot=1.0,
            mode=mode,
            clip=0,
        )
        assert status == OK
        expected = apply_oracle.reference_rows(inputs, 1, 3, 5, 4, 1.0, mode, 0)
        _assert_outputs(outputs, expected)
        for output in outputs:
            np.testing.assert_array_equal(
                apply_oracle.bits32(output[:3]), np.asarray([0x3F800000] * 3, dtype=np.uint32)
            )


def test_nan_infinity_and_signed_zero_are_allowed_and_classified(bridge):
    inputs, outputs = _case_arrays(1, 3, 5, 4)
    inputs[0][:3] = np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)
    inputs[1][:3] = np.asarray([0.25, 0.5, -0.75], dtype=np.float32)
    inputs[2][:3] = np.asarray([0.5, -np.inf, np.inf], dtype=np.float32)
    inputs[3][:3] = 1.0

    for mode in (apply_oracle.SCALAR_MODE, apply_oracle.CPU_VECTOR_MODE):
        for clip in (0, 1):
            for output in outputs:
                output[:] = SENTINEL
            status, _ = _call(
                bridge,
                inputs,
                outputs,
                inner=3,
                value_stride=5,
                gain_stride=4,
                pivot=0.0,
                mode=mode,
                clip=clip,
            )
            assert status == OK, "NaN/Inf pixel inputs are values, not validation failures"
            expected = apply_oracle.reference_rows(inputs, 1, 3, 5, 4, 0.0, mode, clip)
            _assert_outputs(outputs, expected)
            if clip:
                if mode == apply_oracle.SCALAR_MODE:
                    assert outputs[0][0] == np.float32(1.0)
                else:
                    assert np.isnan(outputs[0][0])
                assert outputs[0][2] == np.float32(-np.inf), "there is no lower clip"

    gain_inputs, gain_outputs = _case_arrays(1, 3, 5, 4)
    for channel in range(3):
        gain_inputs[channel][:3] = np.asarray([0.25, 0.0, 0.5], dtype=np.float32)
    gain_inputs[3][:3] = np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)
    for mode in (apply_oracle.SCALAR_MODE, apply_oracle.CPU_VECTOR_MODE):
        status, _ = _call(
            bridge,
            gain_inputs,
            gain_outputs,
            inner=3,
            value_stride=5,
            gain_stride=4,
            pivot=0.0,
            mode=mode,
            clip=1,
        )
        assert status == OK, "NaN/Inf gain values are accepted as pixel data"
        expected = apply_oracle.reference_rows(gain_inputs, 1, 3, 5, 4, 0.0, mode, 1)
        _assert_outputs(gain_outputs, expected)
        if mode == apply_oracle.SCALAR_MODE:
            np.testing.assert_array_equal(
                apply_oracle.bits32([gain_outputs[0][0], gain_outputs[0][1], gain_outputs[0][2]]),
                np.asarray([0x3F800000, 0x3F800000, 0xFF800000], dtype=np.uint32),
            )
        else:
            assert np.isnan(gain_outputs[0][0]) and np.isnan(gain_outputs[0][1])
            assert gain_outputs[0][2] == np.float32(-np.inf)

    # A negative-zero pivot makes offset -0; +0 times a negative gain then
    # preserves -0 through the final FMA and either upper-only clip.
    zero_inputs, zero_outputs = _case_arrays(1, 1, 3, 2)
    zero_inputs[0][0] = np.float32(+0.0)
    zero_inputs[1][0] = np.float32(+0.0)
    zero_inputs[2][0] = np.float32(+0.0)
    zero_inputs[3][0] = np.float32(-1.0)
    for mode in (0, 1):
        for clip in (0, 1):
            status, _ = _call(
                bridge,
                zero_inputs,
                zero_outputs,
                inner=1,
                value_stride=3,
                gain_stride=2,
                pivot=-0.0,
                mode=mode,
                clip=clip,
            )
            assert status == OK
            for output in zero_outputs:
                assert apply_oracle.bits32(output[:1])[0] == np.uint32(0x80000000)


def _atomic_failure(bridge, *, expected_status, **call_kwargs):
    inputs, outputs = _case_arrays(1, 3, 5, 4)
    inputs[0][:3] = (0.25, 0.5, 0.75)
    inputs[1][:3] = (0.75, 0.5, 0.25)
    inputs[2][:3] = (0.125, 0.375, 0.625)
    inputs[3][:3] = (1.0, 1.25, 0.875)
    all_arrays = [*inputs, *outputs]
    snapshots = [array.tobytes() for array in all_arrays]
    arguments = {
        "outer": 1,
        "inner": 3,
        "value_stride": 5,
        "gain_stride": 4,
        "pivot": 0.5,
        "mode": 0,
        "clip": 0,
    }
    arguments.update(call_kwargs)
    status, refs = _call(bridge, inputs, outputs, **arguments)
    assert status == expected_status
    for array, snapshot in zip(all_arrays, snapshots):
        assert array.tobytes() == snapshot
    return inputs, outputs, refs


@pytest.mark.parametrize("pivot", (float("nan"), float("inf"), -float("inf"), float(np.finfo(np.float32).max) * 2.0))
def test_invalid_pivot_is_atomic(bridge, pivot):
    _atomic_failure(bridge, expected_status=INVALID_PREPARED, pivot=pivot)


@pytest.mark.parametrize(
    "call_kwargs,expected",
    [
        ({"outer": 0}, INVALID_DIMENSIONS),
        ({"inner": 0}, INVALID_DIMENSIONS),
        ({"value_stride": 0}, INVALID_STRIDE),
        ({"value_stride": 2}, INVALID_STRIDE),
        ({"gain_stride": 0}, INVALID_STRIDE),
        ({"gain_stride": 2}, INVALID_STRIDE),
        ({"value_stride": (1 << 31)}, INVALID_STRIDE),
        ({"gain_stride": (1 << 31)}, INVALID_STRIDE),
        ({"mode": 2}, INVALID_ARGUMENT),
        ({"clip": 2}, INVALID_ARGUMENT),
    ],
)
def test_invalid_dimensions_strides_and_flags_are_atomic(bridge, call_kwargs, expected):
    _atomic_failure(bridge, expected_status=expected, **call_kwargs)


@pytest.mark.parametrize(
    "call_kwargs,expected",
    [
        ({"input_counts": [2, 15, 15, 12]}, BUFFER_TOO_SMALL),
        ({"output_counts": [2, 15, 15]}, BUFFER_TOO_SMALL),
        ({"null_input_ptrs": True}, INVALID_ARGUMENT),
        ({"null_input_counts": True}, INVALID_ARGUMENT),
        ({"null_output_ptrs": True}, INVALID_ARGUMENT),
        ({"null_output_counts": True}, INVALID_ARGUMENT),
    ],
)
def test_short_buffers_and_null_views_are_atomic(bridge, call_kwargs, expected):
    _atomic_failure(bridge, expected_status=expected, **call_kwargs)


def test_view_control_array_aliases_are_rejected_atomically(bridge):
    inputs, outputs = _case_arrays(1, 1, 1, 1)
    inputs[0][0], inputs[1][0], inputs[2][0], inputs[3][0] = 0.25, 0.5, 0.75, 1.0
    input_ptrs, input_counts, output_ptrs, output_counts = _views(inputs, outputs)
    snapshots = [array.tobytes() for array in [*inputs, *outputs]]

    # A destination span may not overwrite any pointer/count view descriptor.
    for descriptor in (input_ptrs, input_counts, output_ptrs, output_counts):
        output_ptrs[0] = C.cast(C.addressof(descriptor), F32_PTR)
        descriptor_snapshots = [
            C.string_at(C.addressof(view), C.sizeof(view))
            for view in (input_ptrs, input_counts, output_ptrs, output_counts)
        ]
        status = bridge.apply_raw(
            input_ptrs,
            input_counts,
            output_ptrs,
            output_counts,
            1,
            1,
            1,
            1,
            0.5,
            0,
            0,
        )
        assert status == OVERLAP
        for view, snapshot in zip(
            (input_ptrs, input_counts, output_ptrs, output_counts), descriptor_snapshots
        ):
            assert C.string_at(C.addressof(view), C.sizeof(view)) == snapshot
        for array, snapshot in zip([*inputs, *outputs], snapshots):
            assert array.tobytes() == snapshot
        output_ptrs[0] = _float_ptr(outputs[0])


def test_exact_corresponding_inplace_and_read_read_alias_are_allowed(bridge):
    # Exact same pointer/count is permitted for each corresponding channel.
    outer, inner, value_stride, gain_stride = 2, 3, 5, 4
    inputs, _ = _active_random_inputs(np.random.default_rng(1234), outer, inner, value_stride, gain_stride)
    before = [array.copy() for array in inputs]
    outputs = [inputs[0], inputs[1], inputs[2]]
    status, _ = _call(
        bridge,
        inputs,
        outputs,
        outer=outer,
        inner=inner,
        value_stride=value_stride,
        gain_stride=gain_stride,
        pivot=0.375,
        mode=0,
        clip=1,
    )
    assert status == OK
    expected = apply_oracle.reference_rows(before, outer, inner, value_stride, gain_stride, 0.375, 0, 1)
    _assert_outputs(outputs, expected)
    assert np.array_equal(apply_oracle.bits32(inputs[3]), apply_oracle.bits32(before[3]))

    # Read-only spans can alias one another; writes remain distinct.
    shared = np.asarray([0.125, 0.5, 0.875, SENTINEL], dtype=np.float32)
    other = np.asarray([0.25, 0.75, 0.5, SENTINEL], dtype=np.float32)
    gain = np.asarray([1.25, 0.5, 2.0, SENTINEL], dtype=np.float32)
    read_alias_inputs = [shared, shared, other, gain]
    read_alias_outputs = [np.full(4, SENTINEL, dtype=np.float32) for _ in range(3)]
    status, _ = _call(
        bridge,
        read_alias_inputs,
        read_alias_outputs,
        outer=1,
        inner=3,
        value_stride=4,
        gain_stride=4,
        pivot=0.25,
        mode=1,
        clip=0,
    )
    assert status == OK
    expected = apply_oracle.reference_rows(read_alias_inputs, 1, 3, 4, 4, 0.25, 1, 0)
    _assert_outputs(read_alias_outputs, expected)


@pytest.mark.parametrize("alias_kind", ("partial", "cross_channel", "gain", "outputs"))
def test_disallowed_data_aliases_are_rejected_atomically(bridge, alias_kind):
    inputs, outputs = _case_arrays(1, 3, 5, 4)
    for channel, array in enumerate(inputs):
        array[:3] = np.asarray((0.1 + channel * 0.2, 0.3 + channel * 0.1, 0.7 - channel * 0.1), dtype=np.float32)
    input_ptrs, input_counts, output_ptrs, output_counts = _views(inputs, outputs)
    snapshots = [array.tobytes() for array in [*inputs, *outputs]]

    if alias_kind == "partial":
        output_ptrs[0] = C.cast(C.addressof(inputs[0].ctypes.data_as(F32_PTR).contents) + C.sizeof(C.c_float), F32_PTR)
    elif alias_kind == "cross_channel":
        output_ptrs[0] = _float_ptr(inputs[1])
    elif alias_kind == "gain":
        output_ptrs[0] = _float_ptr(inputs[3])
    else:
        output_ptrs[1] = _float_ptr(outputs[0])

    status = bridge.apply_raw(
        input_ptrs,
        input_counts,
        output_ptrs,
        output_counts,
        1,
        3,
        5,
        4,
        0.5,
        0,
        0,
    )
    assert status == OVERLAP
    for array, snapshot in zip([*inputs, *outputs], snapshots):
        assert array.tobytes() == snapshot


def test_inplace_requires_identical_declared_count_and_misalignment_is_atomic(bridge):
    inputs = [np.full(4, SENTINEL, dtype=np.float32) for _ in range(3)]
    inputs.append(np.full(1, SENTINEL, dtype=np.float32))
    outputs = [np.full(3, SENTINEL, dtype=np.float32) for _ in range(3)]
    inputs[0][0] = 0.25
    inputs[1][0] = 0.5
    inputs[2][0] = 0.75
    inputs[3][0] = 1.0
    input_ptrs, input_counts, output_ptrs, output_counts = _views(inputs, outputs)
    output_ptrs[0] = _float_ptr(inputs[0])
    output_counts[0] = outputs[0].size  # sufficient, but not the identical declared span/count
    snapshots = [array.tobytes() for array in [*inputs, *outputs]]
    status = bridge.apply_raw(input_ptrs, input_counts, output_ptrs, output_counts, 1, 1, 1, 1, 0.5, 0, 0)
    assert status == OVERLAP
    for array, snapshot in zip([*inputs, *outputs], snapshots):
        assert array.tobytes() == snapshot

    backing = np.full(8, 0xA5, dtype=np.uint8)
    misaligned = C.cast(C.c_void_p(backing.ctypes.data + 1), F32_PTR)
    input_ptrs[0] = misaligned
    output_counts[0] = outputs[0].size
    for array in outputs:
        array[:] = SENTINEL
    snapshots = [array.tobytes() for array in [*inputs[1:], *outputs]]
    status = bridge.apply_raw(input_ptrs, input_counts, output_ptrs, output_counts, 1, 1, 1, 1, 0.5, 0, 0)
    assert status == UNSUPPORTED_ALIGNMENT
    for array, snapshot in zip([*inputs[1:], *outputs], snapshots):
        assert array.tobytes() == snapshot

    # Exact in-place views are still atomic when a later validation fails.
    in_place_inputs = [np.asarray([0.25, 0.5, 0.75], dtype=np.float32) for _ in range(3)]
    in_place_inputs.append(np.asarray([1.0, 1.25, 0.875], dtype=np.float32))
    in_place_ptrs, in_place_counts, in_place_outputs, in_place_output_counts = _views(
        in_place_inputs, in_place_inputs[:3]
    )
    snapshots = [array.tobytes() for array in in_place_inputs]
    status = bridge.apply_raw(
        in_place_ptrs,
        in_place_counts,
        in_place_outputs,
        in_place_output_counts,
        1,
        1,
        1,
        1,
        float("nan"),
        0,
        0,
    )
    assert status == INVALID_PREPARED
    for array, snapshot in zip(in_place_inputs, snapshots):
        assert array.tobytes() == snapshot


def test_checked_span_overflow_and_resource_limit_statuses(bridge):
    _atomic_failure(
        bridge,
        expected_status=ARITHMETIC_OVERFLOW,
        input_counts=[SIZE_MAX, 15, 15, 12],
    )
    _atomic_failure(
        bridge,
        expected_status=ARITHMETIC_OVERFLOW,
        input_counts=[PTRDIFF_MAX // C.sizeof(C.c_float) + 1, 15, 15, 12],
    )

    # Reserve one sparse anonymous span with disjoint declared views. The
    # consumer must reject its active-work bound before dereferencing pixels.
    inner = (1 << 26) // 3 + 1
    span_bytes = inner * C.sizeof(C.c_float)
    gap_bytes = 4096
    view_count = 7
    total_bytes = view_count * (span_bytes + gap_bytes)
    try:
        storage = mmap.mmap(-1, total_bytes)
    except (OSError, OverflowError, ValueError) as exc:
        pytest.skip(f"cannot reserve sparse spans for resource-limit validation: {exc}")
    try:
        base = C.addressof(C.c_char.from_buffer(storage))
        input_ptrs = (F32_PTR * 4)()
        output_ptrs = (F32_PTR * 3)()
        for index in range(4):
            input_ptrs[index] = C.cast(C.c_void_p(base + index * (span_bytes + gap_bytes)), F32_PTR)
        for index in range(3):
            output_ptrs[index] = C.cast(
                C.c_void_p(base + (index + 4) * (span_bytes + gap_bytes)), F32_PTR
            )
        input_counts = (SIZE * 4)(*[inner] * 4)
        output_counts = (SIZE * 3)(*[inner] * 3)
        status = bridge.apply_raw(
            input_ptrs,
            input_counts,
            output_ptrs,
            output_counts,
            1,
            inner,
            inner,
            inner,
            0.5,
            0,
            0,
        )
        assert status == RESOURCE_LIMIT
    finally:
        storage.close()


def _pack_payload(*, rows, cols, slices, values, byteorder=0):
    prefix = ">" if byteorder else "<"
    header = struct.pack(
        prefix + "II4dI5fI3f",
        rows,
        cols,
        0.75,
        1.375,
        -0.625,
        2.25,
        slices,
        0.03125,
        0.0625,
        0.015625,
        0.046875,
        0.0234375,
        3,
        1.25,
        0.125,
        8.0,
    )
    return header + struct.pack(prefix + f"{len(values)}f", *values)


def test_payload_prepare_scalar_evaluator_then_cpp_apply_chain(bridge):
    # This closes the local PGTM payload -> prepared fields/table -> scalar
    # map evaluation -> C++ tail consumer chain. It is not a product RGB-pipeline
    # closure or an Adobe/Camera Raw runtime comparison.
    rows, cols, slices = 3, 5, 2
    values = tuple(float(0.3 + 0.11 * index) for index in range(rows * cols * slices))
    payload = _pack_payload(rows=rows, cols=cols, slices=slices, values=values, byteorder=1)
    rect = (17, -23, 59, 84)
    map_scalar = 0.875
    expected_prepared = map_oracle.payload_to_prepared(
        payload, map_oracle.TAG_EXTENDED, 1, rect, map_scalar
    )
    status, meta, fields, geometry, table_with_padding, written, *_ = bridge.prepare(
        payload, map_oracle.TAG_EXTENDED, 1, rect, map_scalar
    )
    assert status == map_oracle.OK
    assert written == expected_prepared.table.size
    np.testing.assert_array_equal(map_oracle.expected_bits64(meta[:19]), map_oracle.expected_bits64(expected_prepared.meta))
    np.testing.assert_array_equal(map_oracle.expected_bits32(fields[:16]), map_oracle.expected_bits32(expected_prepared.fields))
    np.testing.assert_array_equal(geometry[:5], expected_prepared.geometry)
    np.testing.assert_array_equal(
        map_oracle.expected_bits32(table_with_padding[:written]),
        map_oracle.expected_bits32(expected_prepared.table),
    )
    assert np.all(meta[19:] == SENTINEL_META)
    assert np.all(fields[16:] == SENTINEL_FIELDS)
    assert np.all(geometry[5:] == SENTINEL_GEOMETRY)
    assert np.all(table_with_padding[written:] == SENTINEL_TABLE)

    outer, inner, value_stride, coord_stride = 3, 5, 8, 7
    rng = np.random.default_rng(0xA991)
    map_inputs = [
        np.full(outer * value_stride, np.float32(-17.25), dtype=np.float32)
        for _ in range(3)
    ]
    map_inputs.extend(
        np.full(outer * coord_stride, np.float32(-19.5), dtype=np.float32)
        for _ in range(2)
    )
    for row in range(outer):
        for col in range(inner):
            offset = row * value_stride + col
            for channel in range(3):
                map_inputs[channel][offset] = np.float32(rng.uniform(0.05, 0.9))
            map_inputs[3][row * coord_stride + col] = np.float32(rng.uniform(-0.5, 2.5))
            map_inputs[4][row * coord_stride + col] = np.float32(rng.uniform(-0.75, 3.5))

    gains = np.full(outer * coord_stride, np.float32(-31.0), dtype=np.float32)
    scalar_status = bridge.evaluate_scalar(
        fields[:16],
        geometry[:5],
        table_with_padding[:written],
        map_inputs,
        gains,
        outer=outer,
        inner=inner,
        value_stride=value_stride,
        coord_stride=coord_stride,
    )
    assert scalar_status == OK
    expected_gains = scalar_oracle.reference_rows(
        fields[:16],
        geometry[:5],
        table_with_padding[:written],
        map_inputs,
        outer,
        inner,
        value_stride,
        coord_stride,
    )
    expected_gains_padded = np.full_like(gains, np.float32(-31.0))
    for row in range(outer):
        expected_gains_padded[row * coord_stride : row * coord_stride + inner] = expected_gains[row]
    np.testing.assert_array_equal(scalar_oracle.bits32(gains), scalar_oracle.bits32(expected_gains_padded))

    # The prepared gain plane is read with its independent row stride by the
    # tail consumer. Its three output channels come from the C++ bridge.
    apply_inputs = [
        np.full(outer * value_stride, np.float32(-23.75), dtype=np.float32)
        for _ in range(3)
    ]
    apply_inputs.append(np.full(outer * coord_stride, np.float32(-31.0), dtype=np.float32))
    for channel in range(3):
        apply_inputs[channel][:] = map_inputs[channel]
    apply_inputs[3][:] = gains
    outputs = [
        np.full(outer * value_stride, SENTINEL, dtype=np.float32) for _ in range(3)
    ]
    source_snapshots = [item.copy() for item in apply_inputs]
    pivot = 0.375
    status, _ = _call(
        bridge,
        apply_inputs,
        outputs,
        outer=outer,
        inner=inner,
        value_stride=value_stride,
        gain_stride=coord_stride,
        pivot=pivot,
        mode=apply_oracle.CPU_VECTOR_MODE,
        clip=1,
    )
    assert status == OK
    expected_outputs = apply_oracle.reference_rows(
        source_snapshots,
        outer,
        inner,
        value_stride,
        coord_stride,
        pivot,
        apply_oracle.CPU_VECTOR_MODE,
        1,
    )
    _assert_outputs(outputs, expected_outputs)


def test_wrong_pointer_span_overflow_is_atomic(bridge):
    if C.sizeof(C.c_void_p) != 8:
        pytest.skip("this synthetic pointer-span probe is 64-bit only")
    inputs, outputs = _case_arrays(1, 1, 1, 1)
    inputs[0][0] = 0.25
    inputs[1][0] = 0.5
    inputs[2][0] = 0.75
    inputs[3][0] = 1.0
    input_ptrs, input_counts, output_ptrs, output_counts = _views(inputs, outputs)
    input_ptrs[0] = C.cast(C.c_void_p(0xFFFFFFFFFFFFFFFC), F32_PTR)
    snapshots = [array.tobytes() for array in [*inputs, *outputs]]
    status = bridge.apply_raw(input_ptrs, input_counts, output_ptrs, output_counts, 1, 1, 1, 1, 0.5, 0, 0)
    assert status == ARITHMETIC_OVERFLOW
    for array, snapshot in zip([*inputs, *outputs], snapshots):
        assert array.tobytes() == snapshot
