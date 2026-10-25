"""Bit-exact checks for the distinct prepared 01372428 no-mask baseline."""

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
SENTINEL = np.float32(-1234.5)
UNSET = object()
OK = 0
INVALID = 1
SHORT = 2
ALIAS = 3
OVERFLOW = 4
LUT_BOUNDS = 5
RESOURCE = 7
NONFINITE = 8


class LutDescriptor(C.Structure):
    _fields_ = [
        ("values", FLOAT_PTR),
        ("count", SIZE),
        ("max_index", C.c_uint32),
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


def _lut(values, scale, max_index=None):
    table = np.asarray(values, dtype=np.float32)
    if max_index is None:
        max_index = table.size - 2
    return table, int(max_index), float(scale)


def _context_records(descriptors):
    if descriptors is None:
        descriptors = (None,) * 5
    if len(descriptors) != 5:
        raise ValueError("exactly five context descriptors are required")
    records = (LutDescriptor * 5)()
    owners = []
    for index, descriptor in enumerate(descriptors):
        if descriptor is None:
            records[index] = LutDescriptor(_null_float_ptr(), 0, 0xFFFFFFFF, np.nan)
            continue
        values, max_index, scale = descriptor
        table = np.ascontiguousarray(values, dtype=np.float32)
        owners.append(table)
        records[index] = LutDescriptor(_ptr(table), table.size, max_index, scale)
    return records, owners


def _compile_bridge(kind: str, output: Path) -> None:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the no-mask bridge"
    assert BRIDGE_SOURCE.is_file(), f"missing no-mask C ABI bridge: {BRIDGE_SOURCE}"
    assert SOURCE.is_file() and HEADER.is_file(), "missing no-mask core source/header"
    if kind == "source":
        inputs = [SOURCE]
    else:
        assert OBJECT.is_file(), f"rebuild renderer: missing fresh official object {OBJECT.name}"
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
        self.apply_fn = self.library.acr_test_exposure_sdr_nomask_context
        self.apply_fn.argtypes = [
            FLOAT_PTR, SIZE, SIZE, SIZE, SIZE, FLOAT_PTR, SIZE,
            FLOAT_PTR, SIZE, C.POINTER(LutDescriptor), SIZE,
            FLOAT_PTR, SIZE, SIZE,
        ]
        self.apply_fn.restype = C.c_int

    def raw(self, src, dst, controls, *, main_lut=None, descriptors=None,
            height=1, width=None, src_stride=None, dst_stride=None,
            src_count=None, dst_count=None, control_count=6, main_count=None,
            descriptor_count=5, descriptor_storage=None,
            descriptor_pointer=UNSET, src_pointer=UNSET, dst_pointer=UNSET,
            controls_pointer=UNSET, main_pointer=UNSET):
        width = (src.size // 3) if width is None else int(width)
        src_stride = width * 3 if src_stride is None else int(src_stride)
        dst_stride = width * 3 if dst_stride is None else int(dst_stride)
        src_count = src.size if src_count is None else int(src_count)
        dst_count = dst.size if dst_count is None else int(dst_count)
        main_count = (0 if main_lut is None else main_lut.size) if main_count is None else int(main_count)
        if descriptor_storage is None:
            records, owners = _context_records(descriptors)
        else:
            records, owners = descriptor_storage, []
        table_pointer = (C.cast(records, C.POINTER(LutDescriptor))
                         if descriptor_pointer is UNSET else descriptor_pointer)
        pointers = (
            _ptr(src) if src_pointer is UNSET else src_pointer,
            _ptr(controls) if controls_pointer is UNSET else controls_pointer,
            _ptr(main_lut) if main_pointer is UNSET else main_pointer,
            table_pointer,
            _ptr(dst) if dst_pointer is UNSET else dst_pointer,
        )
        status = self.apply_fn(
            pointers[0], src_count, src_stride, height, width,
            pointers[1], control_count, pointers[2], main_count,
            pointers[3], descriptor_count, pointers[4], dst_count, dst_stride,
        )
        del owners
        return status

    def apply(self, rgb, controls, *, main_lut=None, pre_lut=None,
              channel_luts=(None, None, None), post_lut=None):
        source = np.ascontiguousarray(rgb, dtype=np.float32)
        parameters = np.ascontiguousarray(controls, dtype=np.float32)
        table = None if main_lut is None else np.ascontiguousarray(main_lut, dtype=np.float32)
        descriptors = (pre_lut, *channel_luts, post_lut)
        output = np.full(source.size, SENTINEL, dtype=np.float32)
        status = self.raw(source.reshape(-1), output, parameters, main_lut=table,
                          descriptors=descriptors, height=source.shape[0],
                          width=source.shape[1])
        return status, output.reshape(source.shape), source, parameters, table


@pytest.fixture(scope="module", params=("source", "official_object"),
                ids=("source", "official"))
def bridge(request, tmp_path_factory):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_exposure_sdr_nomask_" + request.param) / (
        "libsdr_nomask_exposure" + suffix
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
        assert artifact.is_file(), f"missing formal no-mask input: {artifact}"
        assert path.stat().st_mtime_ns >= artifact.stat().st_mtime_ns, (
            f"rebuild formal native renderer: stale relative to {artifact.name}"
        )
    for obj, inputs in ((OBJECT, (SOURCE, HEADER)),
                        (API_OBJECT, (API_SOURCE, PUBLIC_HEADER))):
        assert obj.is_file(), f"missing formal no-mask object: {obj}"
        assert obj.stat().st_mtime_ns >= max(item.stat().st_mtime_ns for item in inputs), (
            f"rebuild formal native renderer: stale object {obj.name}"
        )
        assert path.stat().st_mtime_ns >= obj.stat().st_mtime_ns, (
            f"relink formal native renderer: stale relative to {obj.name}"
        )
    library = C.CDLL(str(path))
    native._configure_library(library)
    function = library.im_native_acr_sdr_nomask_exposure_context_prepared_f32
    function.argtypes = [
        FLOAT_PTR, SIZE, SIZE, SIZE, SIZE, FLOAT_PTR, SIZE, FLOAT_PTR, SIZE,
        C.POINTER(native._AcrExposureLutF32), SIZE, FLOAT_PTR, SIZE, SIZE,
    ]
    function.restype = C.c_int
    return library, str(path)


def _formal_apply(library, rgb, controls, *, main_lut=None, pre_lut=None,
                  channel_luts=(None, None, None), post_lut=None,
                  destination=None, destination_count=None, control_count=6,
                  descriptor_count=5, src_count=None, src_stride=None,
                  dst_stride=None, height=None, width=None):
    source = np.ascontiguousarray(rgb, dtype=np.float32)
    parameters = np.ascontiguousarray(controls, dtype=np.float32)
    table = None if main_lut is None else np.ascontiguousarray(main_lut, dtype=np.float32)
    definitions = (pre_lut, *channel_luts, post_lut)
    records, owners = _context_records(definitions)
    # Mirror the public record layout while letting ctypes enforce the adapter's
    # actual private record type at the formal entry point.
    formal_records = (native._AcrExposureLutF32 * 5)()
    for index, record in enumerate(records):
        formal_records[index] = native._AcrExposureLutF32(
            record.values, record.count, record.max_index, record.scale
        )
    output = (np.full(source.size, SENTINEL, np.float32) if destination is None
              else destination)
    h, w = source.shape[:2]
    h = h if height is None else height
    w = w if width is None else width
    function = library.im_native_acr_sdr_nomask_exposure_context_prepared_f32
    status = function(
        _ptr(source), source.size if src_count is None else src_count,
        w * 3 if src_stride is None else src_stride, h, w,
        _ptr(parameters), control_count, _ptr(table), 0 if table is None else table.size,
        formal_records, descriptor_count, _ptr(output),
        output.size if destination_count is None else destination_count,
        w * 3 if dst_stride is None else dst_stride,
    )
    del owners
    return status, output


def _assert_matches(bridge, rgb, controls, **kwargs):
    status, actual, *_ = bridge.apply(rgb, controls, **kwargs)
    assert status == OK
    expected = oracle.apply_nomask(rgb, controls, **kwargs)
    _assert_bits(actual, expected)
    return expected


def test_oracle_identity_and_null_mask_dispatch_witness(bridge):
    assert oracle.NOMASK_REFERENCE_ID == "acr-01372428-sdr-no-mask-five-context-f32-v1"
    assert oracle.NOMASK_BASELINE_ENTRY == 0x01372428
    rgb = np.asarray([[[0x3F25EA66, 0x3EC872E8, 0x3D051DF0]]], np.uint32).view(np.float32)
    controls = _controls(gain=1.0, scale=1.0, rational=1.0, sqrt=1.0,
                         denominator=1.0, epsilon=1.0e-6)
    no_mask = oracle.apply_nomask(rgb, controls)
    zero_mask = oracle.apply(rgb, np.zeros((1, 1), np.float32), controls)
    assert list(_bits(no_mask).flat) == [0x3F25EA66, 0x3EC872E8, 0x3D051DF0]
    assert list(_bits(zero_mask).flat) == [0x3F25EA66, 0x3EC872E9, 0x3D051DF0]
    _assert_bits(_assert_matches(bridge, rgb, controls), no_mask)


def test_negative_zero_gain_keeps_direct_rgb_sign_bits(bridge):
    rgb = np.asarray([[[0.25, 0.5, 0.75]]], np.float32)
    controls = _controls(gain=np.float32(-0.0), scale=1.0, rational=1.0,
                         sqrt=1.0, denominator=1.0, epsilon=1.0e-6)
    expected = oracle.apply_nomask(rgb, controls)
    masked = oracle.apply(rgb, np.zeros((1, 1), np.float32), controls)
    assert list(_bits(expected).flat) == [0x80000000, 0x80000000, 0x80000000]
    assert list(_bits(masked).flat) == [0x80000000, 0x00000000, 0x80000000]
    _assert_bits(_assert_matches(bridge, rgb, controls), expected)


@pytest.mark.parametrize("rgb,expected_bits", [
    ([0.25, 0.5, 0.75], [0x3E800000, 0x3F000000, 0x3F400000]),
    ([0.25, 0.5, 1.0], [0x3E800000, 0x3F000000, 0x3F800000]),
    ([0.25, 0.75, 1.25], [0x3E800000, 0x3F200000, 0x3F800000]),
], ids=("high-below-1", "high-equals-1", "high-above-1"))
def test_upper_clip_boundary_is_conditional(bridge, rgb, expected_bits):
    source = np.asarray(rgb, np.float32).reshape(1, 1, 3)
    controls = _controls(gain=1.0, scale=1.0, rational=1.0, sqrt=1.0,
                         denominator=1.0, epsilon=1.0e-6)
    expected = _assert_matches(bridge, source, controls)
    assert list(_bits(expected).flat) == expected_bits


def test_rational_stage_can_cross_one_before_conditional_upper_clip(bridge):
    rgb = np.asarray([[[0.2, 0.45, 0.8]]], np.float32)
    controls = _controls(gain=1.0, scale=1.0, rational=1.0, sqrt=1.0,
                         denominator=0.5, epsilon=1.0e-6)
    expected = _assert_matches(bridge, rgb, controls)
    assert _bits(expected).tolist() == [[[0x3E6EEEEE, 0x3F0D82D8, 0x3F800000]]]


@pytest.mark.parametrize("denominator", [0.75, 1.0, 1.25],
                         ids=("Q-below-1", "Q-equal-1", "Q-above-1"))
@pytest.mark.parametrize("rational", [0.8, 1.0, 1.3],
                         ids=("A-below-1", "A-equal-1", "A-above-1"))
@pytest.mark.parametrize("sqrt", [0.8, 1.0, 1.35],
                         ids=("B-below-1", "B-equal-1", "B-above-1"))
def test_Q_A_B_boundaries_match_original_no_mask_order(
    bridge, denominator, rational, sqrt
):
    rgb = np.asarray([[[0.07, 0.31, 0.68], [0.73, 0.22, 0.49],
                       [0.25, 0.25, 0.25], [0.81, 0.6, 0.14]]], np.float32)
    controls = _controls(gain=1.13, rational=rational, sqrt=sqrt,
                         denominator=denominator, epsilon=0.0625)
    table = np.linspace(0.1, 0.9, 17, dtype=np.float32) if rational > 1 else None
    _assert_matches(bridge, rgb, controls, main_lut=table)


@pytest.mark.parametrize("slot", range(5), ids=("pre", "red", "green", "blue", "post"))
def test_each_of_five_context_slots_is_observed(bridge, slot):
    rgb = np.asarray([[[0.16, 0.37, 0.69], [0.73, 0.24, 0.48]]], np.float32)
    controls = _controls(gain=1.0, scale=1.0, rational=1.0, sqrt=1.0,
                         denominator=1.0, epsilon=0.015625)
    values = [0.2, 0.61, 0.43, 0.88, 1.04]
    active = _lut(values, 2.0, 3)
    identity = _lut(np.linspace(0.0, 1.0, 5, dtype=np.float32), 4.0, 3)
    options = {"pre_lut": None, "channel_luts": (None, None, None), "post_lut": None}
    if slot == 0:
        options["pre_lut"] = active
    elif slot == 4:
        options["post_lut"] = active
    else:
        channels = [identity, identity, identity]
        channels[slot - 1] = active
        options["channel_luts"] = tuple(channels)
    expected = _assert_matches(bridge, rgb, controls, **options)
    baseline = oracle.apply_nomask(rgb, controls)
    assert np.any(_bits(expected) != _bits(baseline))


def test_all_five_contexts_preserve_pre_rgb_post_order(bridge):
    rgb = np.asarray([[[0.21, 0.43, 0.67]]], np.float32)
    controls = _controls(gain=1.0, scale=1.0, rational=1.0, sqrt=1.0,
                         denominator=1.0, epsilon=0.015625)
    pre = _lut([0.18, 0.62, 0.41, 0.93, 1.08], 3.0, 3)
    channels = (
        _lut([0.31, 0.74, 0.52, 0.88, 1.03], 2.0, 3),
        _lut([0.12, 0.46, 0.81, 0.59, 0.97], 2.0, 3),
        _lut([0.27, 0.69, 0.38, 0.91, 1.07], 2.0, 3),
    )
    post = _lut([0.25, 0.58, 0.86, 0.44, 1.12], 3.0, 3)
    expected = _assert_matches(bridge, rgb, controls, pre_lut=pre,
                               channel_luts=channels, post_lut=post)
    assert list(_bits(expected).flat) == [0x3F47652D, 0x3F3663CE, 0x3F2CF446]


def test_random_original_order_and_nonidentity_main_curve(bridge):
    rng = np.random.default_rng(2026102208)
    rgb = rng.uniform(0.03, 0.72, size=(5, 7, 3)).astype(np.float32)
    controls = _controls(gain=1.21, scale=4.0, rational=1.35, sqrt=1.3,
                         denominator=0.72, epsilon=0.015625)
    main = np.linspace(0.08, 0.95, 20, dtype=np.float32)
    pre = _lut(rng.uniform(0.12, 0.92, 34), 3.0, 32)
    channels = tuple(_lut(rng.uniform(0.14, 0.94, 34), 4.0, 32)
                     for _ in range(3))
    post = _lut(rng.uniform(0.1, 0.9, 34), 3.0, 32)
    _assert_matches(bridge, rgb, controls, main_lut=main, pre_lut=pre,
                    channel_luts=channels, post_lut=post)


def test_partial_rgb_context_group_ignores_inactive_descriptor_metadata(bridge):
    rgb = np.asarray([[[0.2, 0.4, 0.7]]], np.float32)
    controls = _controls()
    actual = oracle.apply_nomask(rgb, controls, channel_luts=(object(), None, object()))
    baseline = oracle.apply_nomask(rgb, controls)
    _assert_bits(actual, baseline)
    adapted = native.native_acr_sdr_nomask_exposure_context_prepared(
        rgb, controls, channel_luts=(object(), None, object()),
    )
    _assert_bits(adapted, baseline)


def test_padding_in_place_and_atomic_short_output(bridge):
    rgb = np.asarray([
        [0.15, 0.45, 0.78, np.nan, np.nan, np.nan, 91.0, 92.0],
        [0.31, 0.62, 0.22, np.nan, np.nan, np.nan, 93.0, 94.0],
    ], np.float32)
    controls = _controls(gain=1.12, scale=1.0, rational=1.0, sqrt=1.0,
                         denominator=1.0, epsilon=0.015625)
    destination = np.full(18, SENTINEL, np.float32)
    status = bridge.raw(rgb.reshape(-1), destination, controls, height=2, width=1,
                        src_stride=8, dst_stride=9, src_count=16, dst_count=18)
    assert status == OK
    expected = oracle.apply_nomask(rgb[:, :3].reshape(2, 1, 3), controls).reshape(-1)
    _assert_bits(destination.reshape(2, 9)[:, :3].reshape(-1), expected)
    _assert_bits(destination.reshape(2, 9)[:, 3:], np.full((2, 6), SENTINEL, np.float32))

    in_place = np.asarray([0.15, 0.45, 0.78], np.float32)
    before = in_place.copy()
    status = bridge.raw(in_place, in_place, controls, width=1)
    assert status == OK
    _assert_bits(in_place.reshape(1, 1, 3),
                 oracle.apply_nomask(before.reshape(1, 1, 3), controls))

    short_output = np.full(2, SENTINEL, np.float32)
    short_before = short_output.copy()
    status = bridge.raw(np.asarray([0.2, 0.4, 0.6], np.float32), short_output,
                        controls, width=1, dst_count=2)
    assert status == SHORT
    _assert_bytes_unchanged(short_output, short_before)


def test_alias_invalid_overflow_resource_and_nonfinite_failures_are_atomic(bridge):
    source = np.asarray([0.2, 0.4, 0.6], np.float32)
    controls = _controls()
    shared = controls.copy()
    before = shared.copy()
    status = bridge.raw(source, shared, shared, width=1,
                        dst_pointer=_ptr(shared), dst_count=3,
                        controls_pointer=_ptr(shared))
    assert status == ALIAS
    _assert_bytes_unchanged(shared, before)

    output = np.full(3, SENTINEL, np.float32)
    output_before = output.copy()
    invalid_controls = _controls(gain=-1.0)
    status = bridge.raw(source, output, invalid_controls, width=1)
    assert status == INVALID
    _assert_bytes_unchanged(output, output_before)
    negative_source = np.asarray([-0.1, 0.4, 0.6], np.float32)
    status = bridge.raw(negative_source, output, controls, width=1)
    assert status == INVALID
    _assert_bytes_unchanged(output, output_before)

    max_count = (1 << (C.sizeof(SIZE) * 8)) - 1
    status = bridge.raw(source, output, controls, height=2, width=1,
                        src_stride=max_count, src_count=max_count,
                        dst_stride=3, dst_count=6)
    assert status == OVERFLOW
    _assert_bytes_unchanged(output, output_before)

    status = bridge.raw(source, output, controls, width=(1 << 26) + 1,
                        src_count=3, dst_count=3, src_stride=3, dst_stride=3)
    assert status == RESOURCE
    _assert_bytes_unchanged(output, output_before)

    overflowing = _controls(gain=np.finfo(np.float32).max)
    huge_source = np.asarray([2.0, 0.4, 0.6], np.float32)
    with pytest.raises(oracle.NonfiniteIntermediateError):
        oracle.apply_nomask(huge_source.reshape(1, 1, 3), overflowing)
    status = bridge.raw(huge_source, output, overflowing, width=1)
    assert status == NONFINITE
    _assert_bytes_unchanged(output, output_before)

    misaligned_storage = (C.c_ubyte * 16)()
    misaligned_before = bytes(misaligned_storage)
    status = bridge.raw(
        source, output, controls, width=1,
        dst_pointer=C.cast(C.c_void_p(C.addressof(misaligned_storage) + 1), FLOAT_PTR),
    )
    assert status == INVALID
    assert bytes(misaligned_storage) == misaligned_before


def test_formal_api_and_python_adapter_match_nomask_oracle(formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    rng = np.random.default_rng(2026102209)
    rgb = rng.uniform(0.04, 0.75, size=(4, 6, 3)).astype(np.float32)
    controls = _controls(gain=1.1, scale=4.0, rational=1.2, sqrt=1.18,
                         denominator=0.8, epsilon=0.0078125)
    main = np.linspace(0.08, 0.94, 20, dtype=np.float32)
    pre = _lut(rng.uniform(0.1, 0.9, 34), 3.0, 32)
    channels = tuple(_lut(rng.uniform(0.12, 0.93, 34), 4.0, 32)
                     for _ in range(3))
    post = _lut(rng.uniform(0.12, 0.9, 34), 3.0, 32)
    expected = oracle.apply_nomask(rgb, controls, main_lut=main, pre_lut=pre,
                                   channel_luts=channels, post_lut=post)
    status, actual = _formal_apply(library, rgb, controls, main_lut=main,
                                   pre_lut=pre, channel_luts=channels,
                                   post_lut=post)
    assert status == 0
    _assert_bits(actual.reshape(rgb.shape), expected)
    adapted = native.native_acr_sdr_nomask_exposure_context_prepared(
        rgb, controls, main_lut=main, pre_lut=pre,
        channel_luts=channels, post_lut=post,
    )
    _assert_bits(adapted, expected)


def test_formal_status_mapping_and_atomic_short_output(formal_library):
    library, _ = formal_library
    rgb = np.asarray([[[0.2, 0.4, 0.7]]], np.float32)
    controls = _controls()
    output = np.full(3, SENTINEL, np.float32)
    before = output.copy()
    status, actual = _formal_apply(library, rgb, controls, destination=output.copy(),
                                   destination_count=2)
    assert status == 4
    _assert_bytes_unchanged(actual, before)
    status, actual = _formal_apply(library, rgb, controls, destination=output.copy(),
                                   descriptor_count=4)
    assert status == 1
    _assert_bytes_unchanged(actual, before)

    status, actual = _formal_apply(
        library, rgb, controls, destination=output.copy(), src_count=3,
        src_stride=3, dst_stride=3, height=1, width=(1 << 26) + 1,
    )
    assert status == 5
    _assert_bytes_unchanged(actual, before)
