"""Bit-exact validation of the five recovered optional SDR LUT contexts."""

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
SOURCE = NATIVE / "src/acr_exposure_sdr_mask.cpp"
HEADER = NATIVE / "src/acr_exposure_sdr_mask.hpp"
API_SOURCE = NATIVE / "src/acr_exposure_sdr_mask_api.cpp"
BRIDGE_SOURCE = NATIVE / "tools/acr_exposure_sdr_mask_bridge.cpp"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_exposure_sdr_mask.cpp.o"
API_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_exposure_sdr_mask_api.cpp.o"
PUBLIC_HEADER = NATIVE / "include/imprint_renderer.h"
sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
sys.path.insert(0, str(ROOT / "src"))
import exposure_sdr_mask_oracle as oracle  # noqa: E402
import native_renderer as native  # noqa: E402


FLOAT = C.c_float
FLOAT_PTR = C.POINTER(FLOAT)
SIZE = C.c_size_t
U32 = C.c_uint32
LUT_BOUNDS = 5
NONFINITE = 8
ALIAS = 3
INVALID = 1
SHORT = 2
SENTINEL = np.float32(-1234.5)
UNSET = object()


class LutDescriptor(C.Structure):
    _fields_ = [
        ("values", FLOAT_PTR),
        ("count", SIZE),
        ("max_index", U32),
        ("scale", FLOAT),
    ]


def _null_float_ptr():
    return C.cast(C.c_void_p(0), FLOAT_PTR)


def _null_lut_ptr():
    return C.cast(C.c_void_p(0), C.POINTER(LutDescriptor))


def _ptr(array):
    return array.ctypes.data_as(FLOAT_PTR) if array is not None else _null_float_ptr()


def _bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def _assert_bytes_unchanged(actual, before):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint8),
                                  np.asarray(before).view(np.uint8))


def _controls(*, gain=1.0, scale=3.0, rational=1.0, sqrt=1.0,
              denominator=1.0, epsilon=0.125):
    return np.asarray([gain, scale, rational, sqrt, denominator, epsilon], np.float32)


def _main_curve():
    return np.asarray([0.03, 0.28, 0.19, 0.82, 0.57, 0.94, 0.71], np.float32)


def _lut(values, scale, max_index=None):
    table = np.asarray(values, dtype=np.float32)
    if max_index is None:
        max_index = table.size - 2
    return table, int(max_index), float(scale)


def _context_records(descriptors, descriptor_type=LutDescriptor):
    if descriptors is None:
        descriptors = (None,) * 5
    if len(descriptors) != 5:
        raise ValueError("exactly five context descriptor definitions are required")
    records = (descriptor_type * 5)()
    owners = []
    for index, descriptor in enumerate(descriptors):
        if descriptor is None:
            # Data pointer is the activation field; metadata is deliberately
            # poison to ensure inactive descriptors are not inspected.
            records[index] = descriptor_type(_null_float_ptr(), 0, 0xFFFFFFFF, np.nan)
            continue
        values, max_index, scale = descriptor
        table = np.ascontiguousarray(values, dtype=np.float32)
        owners.append(table)
        records[index] = descriptor_type(_ptr(table), table.size, int(max_index), float(scale))
    return records, owners


def _compile_bridge(kind: str, output: Path) -> None:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the SDR context bridge"
    assert BRIDGE_SOURCE.is_file(), f"missing SDR context C ABI bridge: {BRIDGE_SOURCE}"
    assert HEADER.is_file() and SOURCE.is_file(), "missing SDR context source/header"
    if kind == "source":
        inputs = [SOURCE]
    else:
        assert OBJECT.is_file(), f"rebuild renderer: missing official object {OBJECT.name}"
        assert OBJECT.stat().st_mtime_ns >= max(SOURCE.stat().st_mtime_ns,
                                                HEADER.stat().st_mtime_ns), (
            f"rebuild renderer: stale official object {OBJECT.name}"
        )
        inputs = [OBJECT]
    command = [
        compiler, "-std=c++17", "-O2", "-fPIC", "-fno-fast-math", "-ffp-contract=off",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", str(NATIVE / "src"), "-I", str(NATIVE / "include"), str(BRIDGE_SOURCE),
        *(str(path) for path in inputs), "-o", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.apply_fn = self.library.acr_test_exposure_sdr_mask_context
        self.apply_fn.argtypes = [
            FLOAT_PTR, SIZE, SIZE, SIZE, SIZE, FLOAT_PTR, SIZE, SIZE,
            FLOAT_PTR, SIZE, FLOAT_PTR, SIZE, C.POINTER(LutDescriptor), SIZE,
            FLOAT_PTR, SIZE, SIZE,
        ]
        self.apply_fn.restype = C.c_int
        self.legacy_fn = self.library.acr_test_exposure_sdr_mask
        self.legacy_fn.argtypes = [
            FLOAT_PTR, SIZE, SIZE, SIZE, SIZE, FLOAT_PTR, SIZE, SIZE,
            FLOAT_PTR, SIZE, FLOAT_PTR, SIZE, FLOAT_PTR, SIZE, SIZE,
        ]
        self.legacy_fn.restype = C.c_int

    def raw(self, src, dst, controls, *, mask, main_lut=None, descriptors=None,
            height=1, width=None, src_stride=None, dst_stride=None, mask_stride=None,
            src_count=None, dst_count=None, mask_count=None, control_count=6,
            main_count=None, descriptor_count=5, descriptor_storage=None,
            descriptor_pointer=UNSET, src_pointer=UNSET, dst_pointer=UNSET,
            mask_pointer=UNSET, controls_pointer=UNSET, main_pointer=UNSET):
        width = (src.size // 3) if width is None else int(width)
        src_stride = width * 3 if src_stride is None else int(src_stride)
        dst_stride = width * 3 if dst_stride is None else int(dst_stride)
        mask_stride = width if mask_stride is None else int(mask_stride)
        src_count = src.size if src_count is None else int(src_count)
        dst_count = dst.size if dst_count is None else int(dst_count)
        mask_count = (0 if mask is None else mask.size) if mask_count is None else int(mask_count)
        main_count = (0 if main_lut is None else main_lut.size) if main_count is None else int(main_count)
        if descriptor_storage is None:
            records, owners = _context_records(descriptors)
        else:
            records, owners = descriptor_storage, []
        record_pointer = (C.cast(records, C.POINTER(LutDescriptor))
                          if descriptor_pointer is UNSET else descriptor_pointer)
        pointers = [
            _ptr(src) if src_pointer is UNSET else src_pointer,
            _ptr(mask) if mask_pointer is UNSET else mask_pointer,
            _ptr(controls) if controls_pointer is UNSET else controls_pointer,
            _ptr(main_lut) if main_pointer is UNSET else main_pointer,
            record_pointer,
            _ptr(dst) if dst_pointer is UNSET else dst_pointer,
        ]
        del owners  # Keep referenced NumPy arrays alive through the native call.
        return self.apply_fn(
            pointers[0], src_count, src_stride, height, width,
            pointers[1], mask_count, mask_stride, pointers[2], control_count,
            pointers[3], main_count, pointers[4], descriptor_count,
            pointers[5], dst_count, dst_stride,
        )

    def apply(self, rgb, controls, *, mask, main_lut=None, pre_lut=None,
              channel_luts=(None, None, None), post_lut=None):
        source = np.ascontiguousarray(rgb, dtype=np.float32)
        masks = np.ascontiguousarray(mask, dtype=np.float32)
        parameters = np.ascontiguousarray(controls, dtype=np.float32)
        table = None if main_lut is None else np.ascontiguousarray(main_lut, dtype=np.float32)
        output = np.full(source.size, SENTINEL, dtype=np.float32)
        descriptors = (pre_lut, *channel_luts, post_lut)
        status = self.raw(source, output, parameters, mask=masks, main_lut=table,
                          descriptors=descriptors, height=source.shape[0],
                          width=source.shape[1])
        return status, output.reshape(source.shape), source, masks, parameters, table

    def legacy(self, rgb, controls, *, mask, main_lut=None):
        source = np.ascontiguousarray(rgb, dtype=np.float32)
        masks = np.ascontiguousarray(mask, dtype=np.float32)
        parameters = np.ascontiguousarray(controls, dtype=np.float32)
        table = None if main_lut is None else np.ascontiguousarray(main_lut, dtype=np.float32)
        output = np.full(source.size, SENTINEL, dtype=np.float32)
        status = self.legacy_fn(
            _ptr(source), source.size, source.shape[1] * 3, source.shape[0], source.shape[1],
            _ptr(masks), masks.size, source.shape[1], _ptr(parameters), parameters.size,
            _ptr(table), 0 if table is None else table.size, _ptr(output), output.size,
            source.shape[1] * 3,
        )
        return status, output.reshape(source.shape)


@pytest.fixture(scope="module", params=("source", "official_object"),
                ids=("source", "official"))
def bridge(request, tmp_path_factory):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_exposure_sdr_context_" + request.param) / (
        "libsdr_context_exposure" + suffix
    )
    _compile_bridge(request.param, output)
    return Bridge(output)


@pytest.fixture(scope="module")
def formal_library():
    path = NATIVE / "build" / (
        "libimprint_renderer.0.1.0.dylib" if sys.platform == "darwin"
        else "libimprint_renderer.0.1.0.so"
    )
    assert path.is_file(), f"missing formal native renderer artifact: {path}"
    for artifact in (SOURCE, HEADER, API_SOURCE, PUBLIC_HEADER):
        assert artifact.is_file(), f"missing formal SDR context input: {artifact}"
        assert path.stat().st_mtime_ns >= artifact.stat().st_mtime_ns, (
            f"rebuild formal native renderer: stale relative to {artifact.name}"
        )
    for obj, inputs in ((OBJECT, (SOURCE, HEADER)),
                        (API_OBJECT, (API_SOURCE, PUBLIC_HEADER))):
        assert obj.is_file(), f"missing formal SDR context object: {obj}"
        assert obj.stat().st_mtime_ns >= max(item.stat().st_mtime_ns for item in inputs), (
            f"rebuild formal native renderer: stale object {obj.name}"
        )
        assert path.stat().st_mtime_ns >= obj.stat().st_mtime_ns, (
            f"relink formal native renderer: stale relative to {obj.name}"
        )
    library = C.CDLL(str(path))
    native._configure_library(library)
    return library, str(path)


def _formal_apply(library, rgb, mask, controls, *, main_lut=None,
                  pre_lut=None, channel_luts=(None, None, None), post_lut=None,
                  source_count=None, descriptor_count=5, destination=None,
                  destination_count=None, control_count=None):
    source = np.ascontiguousarray(rgb, dtype=np.float32)
    masks = np.ascontiguousarray(mask, dtype=np.float32)
    parameters = np.ascontiguousarray(controls, dtype=np.float32)
    table = None if main_lut is None else np.ascontiguousarray(main_lut, dtype=np.float32)
    definitions = (pre_lut, *channel_luts, post_lut)
    records, owners = _context_records(definitions, native._AcrExposureLutF32)
    output = (np.full(source.size, SENTINEL, np.float32) if destination is None
              else destination)
    h, w = source.shape[:2]
    fn = library.im_native_acr_sdr_mask_exposure_context_prepared_f32
    status = fn(
        _ptr(source), source.size if source_count is None else source_count,
        w * 3, h, w, _ptr(masks), masks.size, w, _ptr(parameters),
        parameters.size if control_count is None else control_count,
        _ptr(table), 0 if table is None else table.size, records, descriptor_count,
        _ptr(output), output.size if destination_count is None else destination_count,
        w * 3,
    )
    del owners
    return status, output, source, masks, parameters, table


def _expected(rgb, mask, controls, *, main_lut=None, pre_lut=None,
              channel_luts=(None, None, None), post_lut=None):
    return oracle.apply(rgb, mask, controls, main_lut=main_lut, pre_lut=pre_lut,
                        channel_luts=channel_luts, post_lut=post_lut)


def _assert_bridge_matches(bridge, rgb, mask, controls, *, main_lut=None,
                           pre_lut=None, channel_luts=(None, None, None), post_lut=None):
    status, actual, *_ = bridge.apply(
        rgb, controls, mask=mask, main_lut=main_lut, pre_lut=pre_lut,
        channel_luts=channel_luts, post_lut=post_lut,
    )
    assert status == 0
    expected = _expected(rgb, mask, controls, main_lut=main_lut, pre_lut=pre_lut,
                         channel_luts=channel_luts, post_lut=post_lut)
    _assert_bits(actual, expected)
    return expected


def test_context_oracle_identity_keeps_legacy_default_and_all_empty_equivalent(bridge):
    rng = np.random.default_rng(2026102201)
    rgb = rng.uniform(0.0, 0.8, size=(4, 7, 3)).astype(np.float32)
    mask = rng.uniform(-0.5, 0.5, size=(4, 7)).astype(np.float32)
    controls = _controls(gain=1.16, scale=3.0, rational=1.0, sqrt=1.25)
    expected = oracle.apply(rgb, mask, controls)
    explicit_empty = oracle.apply(
        rgb, mask, controls, pre_lut=None,
        channel_luts=(None, None, None), post_lut=None,
    )
    _assert_bits(explicit_empty, expected)
    context_status, context_result, *_ = bridge.apply(rgb, controls, mask=mask)
    legacy_status, legacy_result = bridge.legacy(rgb, controls, mask=mask)
    assert context_status == legacy_status == 0
    _assert_bits(context_result, expected)
    _assert_bits(context_result, legacy_result)


def test_empty_context_preserves_masked_unconditional_middle_reconstruction(bridge):
    rgb = np.asarray([[[0x3F25EA66, 0x3EC872E8, 0x3D051DF0]]], np.uint32).view(np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls(gain=1.0, scale=1.0, rational=1.0,
                         sqrt=1.0, denominator=1.0, epsilon=1.0e-6)
    expected = oracle.apply(rgb, mask, controls)
    assert list(_bits(expected).flat) == [0x3F25EA66, 0x3EC872E9, 0x3D051DF0]
    status, actual, *_ = bridge.apply(rgb, controls, mask=mask)
    assert status == 0
    _assert_bits(actual, expected)


@pytest.mark.parametrize("slot", range(5), ids=("pre", "red", "green", "blue", "post"))
def test_each_optional_context_slot_runs_only_when_enabled(bridge, slot):
    rgb = np.asarray([[[0.16, 0.37, 0.69], [0.73, 0.24, 0.48]]], np.float32)
    mask = np.zeros((1, 2), np.float32)
    controls = _controls()
    flat = [None] * 5
    if slot == 0:
        flat[0] = _lut([0.2, 0.6, 0.9, 1.1], 1.5, 2)
    elif slot == 4:
        flat[4] = _lut([0.15, 0.7, 0.95, 1.2], 1.5, 2)
    else:
        for index in range(1, 4):
            flat[index] = _lut([0.1 + index * 0.03, 0.35 + index * 0.05,
                               0.72 - index * 0.04, 0.95], 1.0, 2)
    actual_context = (flat[0], tuple(flat[1:4]), flat[4])
    expected = _assert_bridge_matches(
        bridge, rgb, mask, controls, pre_lut=actual_context[0],
        channel_luts=actual_context[1], post_lut=actual_context[2],
    )
    base = oracle.apply(rgb, mask, controls)
    assert np.any(_bits(expected) != _bits(base))


def test_rgb_context_group_is_all_or_nothing_and_ignores_missing_group_metadata(bridge):
    rgb = np.asarray([[[0.2, 0.4, 0.7]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls()
    source = rgb.reshape(-1).copy()
    output = np.full(3, SENTINEL, np.float32)
    records, owners = _context_records((None,) * 5)
    poison = np.asarray([np.nan], np.float32)
    owners.append(poison)
    records[1] = LutDescriptor(_ptr(poison), poison.size, 0xFFFFFFFF, np.nan)
    records[2] = LutDescriptor(_null_float_ptr(), 0, 0xFFFFFFFF, np.nan)
    records[3] = LutDescriptor(_ptr(poison), 0, 0xFFFFFFFF, -np.inf)
    before = output.copy()
    status = bridge.raw(source, output, controls, mask=mask.reshape(-1), width=1,
                        descriptor_storage=records)
    assert status == 0
    _assert_bits(output, oracle.apply(rgb, mask, controls).reshape(-1))
    assert not np.array_equal(_bits(output), _bits(before))
    partial_oracle = oracle.apply(rgb, mask, controls,
                                  channel_luts=(object(), None, object()))
    _assert_bits(partial_oracle, oracle.apply(rgb, mask, controls))

    # The adapter suppresses the entire incomplete group before it examines
    # any of these intentionally unusable descriptor values.
    adapted = native.native_acr_sdr_mask_exposure_context_prepared(
        rgb, mask, controls, channel_luts=(object(), None, object()),
    )
    _assert_bits(adapted, oracle.apply(rgb, mask, controls))
    assert before[0] == SENTINEL


def test_context_luma_and_channel_order_has_a_bit_locked_order_witness(bridge):
    rgb = np.asarray([[[0.21, 0.43, 0.67]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls()
    pre = _lut([0.18, 0.62, 0.41, 0.93, 1.08], 3.0, 3)
    channels = (
        _lut([0.31, 0.74, 0.52, 0.88, 1.03], 2.0, 3),
        _lut([0.12, 0.46, 0.81, 0.59, 0.97], 2.0, 3),
        _lut([0.27, 0.69, 0.38, 0.91, 1.07], 2.0, 3),
    )
    post = _lut([0.25, 0.58, 0.86, 0.44, 1.12], 3.0, 3)
    expected = _assert_bridge_matches(
        bridge, rgb, mask, controls, pre_lut=pre,
        channel_luts=channels, post_lut=post,
    )
    # Pin the raw instruction sequence: pre-luma, R/G/B, post-luma, final
    # post-only upper clip. The literal is independent of the native result.
    assert list(_bits(expected).flat) == [0x3F47652D, 0x3F3663CE, 0x3F2CF446]

    # Swapping the two luma contexts gives a distinguishable result even when
    # the same RGB tables remain between them.
    swapped = oracle.apply(rgb, mask, controls, pre_lut=post,
                           channel_luts=channels, post_lut=pre)
    assert not np.array_equal(_bits(expected), _bits(swapped))

    # Also execute the reverse block order explicitly: post (including its
    # clamp), R/G/B, pre. The pinned value shows that moving whole contexts
    # across each other changes the result.
    reversed_values = list(oracle.apply(rgb, mask, controls)[0, 0])
    epsilon = controls[oracle.CONTROL_EPSILON]
    post_context = oracle._prepare_context(post, "reverse post")
    channel_contexts = tuple(oracle._prepare_context(item, "reverse channel")
                             for item in channels)
    pre_context = oracle._prepare_context(pre, "reverse pre")
    oracle._context_luma(reversed_values, post_context, epsilon, "reverse post")
    reversed_values = [value if value < np.float32(1.0) else np.float32(1.0)
                       for value in reversed_values]
    for channel, context in enumerate(channel_contexts):
        reversed_values[channel] = oracle._context_channel(
            reversed_values[channel], context, "reverse channel"
        )
    oracle._context_luma(reversed_values, pre_context, epsilon, "reverse pre")
    reversed_result = np.asarray(reversed_values, np.float32)
    assert list(_bits(reversed_result)) == [0x3F0DEB80, 0x3F0A1683, 0x3EB7FACC]
    assert not np.array_equal(_bits(expected[0, 0]), _bits(reversed_result))


def test_luma_addition_order_has_a_bit_locked_rounding_witness(bridge):
    rgb = np.asarray([[[0x3F1B9E77, 0x3E2B13A5, 0x3F7C6987]]], np.uint32).view(np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls(gain=1.0, scale=3.0, rational=1.0,
                         sqrt=1.0, denominator=1.0, epsilon=1.0e-6)
    table = _lut([
        0.8916738, 1.1119207, -0.2807523, 0.3289146, 0.9625386,
        0.7928209, 0.8984618, -0.24340515, 0.17570452, -0.45355526,
        -0.40733606, 0.81870353,
    ], 8.0, 10)
    expected = _assert_bridge_matches(bridge, rgb, mask, controls, pre_lut=table)
    assert list(_bits(expected).flat) == [0x3F8CA799, 0x3E9AA041, 0x3FE423E8]

    # The alternate but algebraically equivalent ((R+B)+G)+G order differs by
    # one ULP in luma and produces a distinct RGB bit pattern.
    base = oracle.apply(rgb, mask, controls)[0, 0]
    alternate_total = oracle.add32(base[0], base[2], "alternate luma R+B")
    alternate_total = oracle.add32(base[1], alternate_total, "alternate luma +G")
    alternate_total = oracle.add32(base[1], alternate_total, "alternate luma +G")
    alternate_luma = oracle.mul32(alternate_total, np.float32(0.25),
                                  "alternate luma quarter")
    context = oracle._prepare_context(table, "alternate luma")
    coordinate = oracle.mul32(context.scale, alternate_luma)
    index, fraction = oracle._context_index(coordinate, context, "alternate luma")
    mapped = oracle._context_samples(context, index, fraction, "alternate luma")
    denominator = (alternate_luma if alternate_luma > controls[oracle.CONTROL_EPSILON]
                   else controls[oracle.CONTROL_EPSILON])
    gain = oracle.div32(mapped, denominator)
    alternate = np.asarray([oracle.mul32(value, gain) for value in base], np.float32)
    assert list(_bits(alternate)) == [0x3F8CA79A, 0x3E9AA043, 0x3FE423EA]


def test_context_interpolation_uses_the_recovered_fma_operand_order(bridge):
    rgb = np.asarray([[[0.25, 0.5, 0.75]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls()
    red = _lut([3.012157, -6.7526484, 9.322026, 0.1], 1.5, 2)
    identity = _lut([0.0, 1.0, 1.0, 1.0], 1.0, 2)
    channels = (red, identity, identity)
    expected = _assert_bridge_matches(bridge, rgb, mask, controls,
                                      channel_luts=channels)
    assert list(_bits(expected).flat) == [0xBF264F25, 0x3F000000, 0x3F400000]

    fraction = np.float32(0.375)
    complement = oracle.sub32(np.float32(1.0), fraction)
    fraction_times_next = oracle.mul32(fraction, red[0][1])
    separate = oracle.add32(
        oracle.mul32(red[0][0], complement), fraction_times_next,
    )
    fused = oracle.fma32(red[0][0], complement, fraction_times_next)
    assert int(_bits([separate])[0]) == 0xBF264F24
    assert int(_bits([fused])[0]) == 0xBF264F25


def test_luma_addition_order_and_negative_cross_context_coordinates(bridge):
    rgb = np.asarray([[[0.5, 0.25, 0.75]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls()
    # This context produces negative RGB. Positive channel scales then create
    # q values in (-1, 0), which FCVTZS truncates to index zero.
    pre = _lut([-0.4, -0.1, 0.2, 0.5], 1.0, 2)
    channels = (
        _lut([-0.3, 0.5, 0.7], 0.5, 1),
        _lut([-0.2, 0.4, 0.8], 0.5, 1),
        _lut([-0.1, 0.6, 0.9], 0.5, 1),
    )
    expected = _assert_bridge_matches(bridge, rgb, mask, controls,
                                      pre_lut=pre, channel_luts=channels)
    assert np.all(expected < 0.0)
    for value, context in zip(expected[0, 0], channels):
        coordinate = oracle.mul32(context[2], value)
        assert -1.0 < float(coordinate) < 0.0
        index, _ = oracle._context_index(coordinate, oracle._prepare_context(context, "test"),
                                         "negative-q witness")
        assert index == 0

    # For q <= -1, FCVTZS yields a negative index and the active table fails
    # atomically rather than wrapping to an unsigned array index.
    bad_channels = tuple(_lut(item[0], 4.0, item[1]) for item in channels)
    status, actual, *_ = bridge.apply(rgb, controls, mask=mask,
                                     pre_lut=pre, channel_luts=bad_channels)
    assert status == LUT_BOUNDS
    assert np.all(actual == SENTINEL)


def test_context_lut_index_is_inclusive_but_requires_the_adjacent_guard(bridge):
    rgb = np.asarray([[[0.25, 0.5, 0.75]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls()
    table = _lut([0.2, 0.8], 0.0, 0)
    _assert_bridge_matches(bridge, rgb, mask, controls, pre_lut=table)

    # max_index is inclusive: q=0.5 accesses index 0 and its guard at 1.
    q_half = _lut([0.2, 0.8], 1.0, 0)
    _assert_bridge_matches(bridge, rgb, mask, controls, pre_lut=q_half)

    # Coordinate 1 exceeds max_index even though a second sample exists.
    over_max = _lut([0.2, 0.8, 0.9], 4.0, 0)
    status, output, *_ = bridge.apply(rgb, controls, mask=mask, pre_lut=over_max)
    assert status == LUT_BOUNDS
    assert np.all(output == SENTINEL)

    # max_index 1 asks for the next guard at index 2; count two is insufficient.
    missing_guard = _lut([0.2, 0.8], 2.0, 1)
    status, output, *_ = bridge.apply(rgb, controls, mask=mask, pre_lut=missing_guard)
    assert status == LUT_BOUNDS
    assert np.all(output == SENTINEL)


def test_only_accessed_context_samples_must_be_finite(bridge):
    rgb = np.asarray([[[0.25, 0.5, 0.75]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls()
    declared_range_only = _lut([0.2, 0.8], 0.0, 128)
    _assert_bridge_matches(bridge, rgb, mask, controls, pre_lut=declared_range_only)
    unused_nan = _lut([0.2, 0.8, np.nan, np.nan], 0.0, 128)
    _assert_bridge_matches(bridge, rgb, mask, controls, pre_lut=unused_nan)

    accessed_nan = _lut([0.2, np.nan], 0.0, 0)
    status, output, *_ = bridge.apply(rgb, controls, mask=mask, pre_lut=accessed_nan)
    assert status == NONFINITE
    assert np.all(output == SENTINEL)


def test_post_context_alone_adds_the_final_upper_clip(bridge):
    rgb = np.asarray([[[0.2, 0.4, 0.7]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls()
    pre = _lut([3.0, 3.0, 3.0], 0.0, 1)
    without_post = _assert_bridge_matches(bridge, rgb, mask, controls, pre_lut=pre)
    assert np.any(without_post > 1.0)

    post = _lut([3.0, 3.0, 3.0], 0.0, 1)
    with_post = _assert_bridge_matches(bridge, rgb, mask, controls,
                                      pre_lut=pre, post_lut=post)
    assert np.all(with_post <= 1.0)
    assert np.any(with_post == 1.0)


@pytest.mark.parametrize("slot", ["main", "pre", "red", "green", "blue", "post"])
def test_output_overlap_with_each_active_table_is_rejected_atomically(bridge, slot):
    rgb = np.asarray([[[0.2, 0.4, 0.7]]], np.float32)
    source = rgb.reshape(-1).copy()
    mask = np.zeros(1, np.float32)
    controls = _controls(rational=1.2 if slot == "main" else 1.0,
                         scale=1.0)
    shared = np.asarray([0.2, 0.5, 0.8, 0.4], np.float32)
    shared_before = shared.copy()
    destination = shared[:3]
    main_lut = _main_curve() if slot == "main" else None
    descriptors = [None] * 5
    if slot == "main":
        main_lut = shared
        controls[2] = np.float32(1.2)
    else:
        index = {"pre": 0, "red": 1, "green": 2, "blue": 3, "post": 4}[slot]
        descriptors[index] = (shared, 2, 0.0)
        if index in (1, 2, 3):
            for channel_index in (1, 2, 3):
                if descriptors[channel_index] is None:
                    descriptors[channel_index] = (_lut([0.2, 0.4, 0.7], 0.0, 1))
    records, owners = _context_records(descriptors)
    # Replace the selected record's copied data with the shared output span.
    selected_index = {"pre": 0, "red": 1, "green": 2, "blue": 3, "post": 4}.get(slot)
    if selected_index is not None:
        records[selected_index] = LutDescriptor(_ptr(shared), shared.size, 2, 0.0)
    before = destination.copy()
    status = bridge.raw(source, destination, controls, mask=mask, width=1,
                        main_lut=main_lut,
                        main_count=shared.size if slot == "main" else None,
                        descriptor_storage=records)
    assert status == ALIAS
    _assert_bytes_unchanged(destination, before)
    _assert_bytes_unchanged(shared, shared_before)
    del owners


def test_output_overlap_with_descriptor_array_is_rejected_atomically(bridge):
    rgb = np.asarray([[[0.2, 0.4, 0.7]]], np.float32)
    source = rgb.reshape(-1).copy()
    mask = np.zeros(1, np.float32)
    controls = _controls()
    record_bytes = C.sizeof(LutDescriptor) * 5
    raw = (C.c_ubyte * (record_bytes + 32))()
    address = C.addressof(raw)
    records = C.cast(address, C.POINTER(LutDescriptor))
    for index in range(5):
        records[index] = LutDescriptor(_null_float_ptr(), 0, 0xFFFFFFFF, np.nan)
    destination = np.frombuffer(raw, dtype=np.float32, count=3,
                                offset=record_bytes - 4)
    before = destination.copy()
    status = bridge.raw(source, destination, controls, mask=mask, width=1,
                        descriptor_pointer=records)
    assert status == ALIAS
    _assert_bytes_unchanged(destination, before)


def test_random_nonmonotone_main_curve_rejects_out_of_context_range_atomically(bridge):
    rng = np.random.default_rng(2026102202)
    rgb = rng.uniform(0.01, 0.82, size=(5, 7, 3)).astype(np.float32)
    mask = rng.uniform(-0.4, 0.4, size=(5, 7)).astype(np.float32)
    controls = _controls(gain=1.13, rational=1.18, sqrt=1.27,
                         denominator=0.77, epsilon=0.015625)
    main = rng.uniform(0.08, 0.96, size=13).astype(np.float32)
    pre = _lut(rng.uniform(0.05, 0.98, 34), 6.0, 32)
    channels = tuple(_lut(rng.uniform(0.06, 0.97, 34), 8.0, 32)
                     for _ in range(3))
    post = _lut(rng.uniform(0.04, 0.99, 34), 6.0, 32)
    # This deterministic random curve is intentionally not monotone. The
    # recovered main stage can produce a large finite component; the context
    # descriptor declares only a finite supported range. Preserve that domain
    # boundary as an atomic failure in both oracle and native implementation.
    with pytest.raises(oracle.LutBoundsError):
        _expected(rgb, mask, controls, main_lut=main, pre_lut=pre,
                  channel_luts=channels, post_lut=post)
    source = rgb.reshape(-1).copy()
    output = np.full(source.size, SENTINEL, np.float32)
    before = output.copy()
    status = bridge.raw(source, output, controls, mask=mask.reshape(-1),
                        width=rgb.shape[1], main_lut=main,
                        descriptors=(pre, *channels, post))
    assert status == LUT_BOUNDS
    _assert_bytes_unchanged(output, before)


def test_random_all_five_contexts_match_independent_uint32_oracle(bridge):
    rng = np.random.default_rng(2026102206)
    rgb = rng.uniform(0.01, 0.82, size=(5, 7, 3)).astype(np.float32)
    mask = rng.uniform(-0.25, 0.25, size=(5, 7)).astype(np.float32)
    controls = _controls(gain=1.0, scale=3.0, rational=1.0, sqrt=1.25,
                         denominator=1.0, epsilon=0.015625)
    pre = _lut(rng.uniform(0.05, 0.98, 18), 3.0, 16)
    channels = tuple(_lut(rng.uniform(0.06, 0.97, 18), 4.0, 16)
                     for _ in range(3))
    post = _lut(rng.uniform(0.04, 0.99, 18), 3.0, 16)
    _assert_bridge_matches(bridge, rgb, mask, controls, pre_lut=pre,
                           channel_luts=channels, post_lut=post)


def test_formal_api_and_python_adapter_match_context_oracle(formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    rng = np.random.default_rng(2026102203)
    rgb = rng.uniform(0.01, 0.78, size=(4, 6, 3)).astype(np.float32)
    mask = rng.uniform(-0.35, 0.35, size=(4, 6)).astype(np.float32)
    controls = _controls(gain=1.11, rational=1.16, sqrt=1.22,
                         denominator=0.73, epsilon=0.0078125)
    main = rng.uniform(0.07, 0.94, size=12).astype(np.float32)
    pre = _lut(rng.uniform(0.08, 0.97, 34), 6.0, 32)
    channels = tuple(_lut(rng.uniform(0.06, 0.98, 34), 8.0, 32)
                     for _ in range(3))
    post = _lut(rng.uniform(0.04, 0.99, 34), 6.0, 32)
    expected = _expected(rgb, mask, controls, main_lut=main,
                         pre_lut=pre, channel_luts=channels, post_lut=post)

    status, actual, *_ = _formal_apply(
        library, rgb, mask, controls, main_lut=main, pre_lut=pre,
        channel_luts=channels, post_lut=post,
    )
    assert status == 0
    _assert_bits(actual.reshape(rgb.shape), expected)
    adapted = native.native_acr_sdr_mask_exposure_context_prepared(
        rgb, mask, controls, main_lut=main, pre_lut=pre,
        channel_luts=channels, post_lut=post,
    )
    _assert_bits(adapted, expected)


def test_formal_empty_context_matches_legacy_api_and_adapter(formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    rgb = np.asarray([[[0.12, 0.37, 0.76], [0.81, 0.49, 0.23]]], np.float32)
    mask = np.asarray([[0.125, -0.25]], np.float32)
    controls = _controls(gain=1.14, sqrt=1.22)
    expected = oracle.apply(rgb, mask, controls)
    context_status, context_output, *_ = _formal_apply(
        library, rgb, mask, controls,
    )
    legacy_output = native.native_acr_sdr_mask_exposure_prepared(
        rgb, mask, controls,
    )
    context_adapter = native.native_acr_sdr_mask_exposure_context_prepared(
        rgb, mask, controls, pre_lut=None, channel_luts=(None, None, None), post_lut=None,
    )
    assert context_status == 0
    _assert_bits(context_output.reshape(rgb.shape), expected)
    _assert_bits(legacy_output, expected)
    _assert_bits(context_adapter, legacy_output)


def test_formal_context_status_mapping_and_atomic_short_output(formal_library):
    library, _ = formal_library
    rgb = np.asarray([[[0.2, 0.4, 0.7]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls()
    output = np.full(3, SENTINEL, np.float32)
    before = output.copy()
    status, actual, *_ = _formal_apply(
        library, rgb, mask, controls, destination=output.copy(), descriptor_count=4,
    )
    assert status == 1
    _assert_bytes_unchanged(actual, before)
    status, actual, *_ = _formal_apply(
        library, rgb, mask, controls, destination=output.copy(), destination_count=2,
    )
    assert status == 4
    _assert_bytes_unchanged(actual, before)
