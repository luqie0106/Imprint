"""Bit-exact tests for the recovered prepared signed Exposure operator."""

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
SOURCE = NATIVE / "src/acr_exposure_signed.cpp"
HEADER = NATIVE / "src/acr_exposure_signed.hpp"
API_SOURCE = NATIVE / "src/acr_exposure_signed_api.cpp"
BRIDGE_SOURCE = NATIVE / "tools/acr_exposure_signed_bridge.cpp"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_exposure_signed.cpp.o"
API_OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_exposure_signed_api.cpp.o"
PUBLIC_HEADER = NATIVE / "include/imprint_renderer.h"
sys.path.insert(0, str(ROOT / "tests" / "acr_support"))
sys.path.insert(0, str(ROOT / "src"))
import exposure_signed_oracle as oracle  # noqa: E402
import native_renderer as native  # noqa: E402


FLOAT = C.c_float
FLOAT_PTR = C.POINTER(FLOAT)
SIZE = C.c_size_t
U32 = C.c_uint32
SENTINEL = np.float32(-1234.5)
OK = 0


class ExposureLut(C.Structure):
    _fields_ = [
        ("values", FLOAT_PTR),
        ("capacity", SIZE),
        ("max_index", U32),
        ("scale", FLOAT),
    ]


FormalExposureLut = native._AcrExposureLutF32


class PreparedSignedExposure(C.Structure):
    _fields_ = [
        ("gain", FLOAT),
        ("compression", FLOAT),
        ("expansion", FLOAT),
        ("epsilon", FLOAT),
        ("channel", ExposureLut * 3),
        ("gray", ExposureLut),
    ]


def _ptr(array):
    return array.ctypes.data_as(FLOAT_PTR) if array is not None else _null_ptr()


def _null_ptr():
    return C.cast(C.c_void_p(0), FLOAT_PTR)


def _bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def _assert_bytes_unchanged(actual, before):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint8), np.asarray(before).view(np.uint8))


def _prepared(controls=(1.0, 1.0, 1.0, 0.25), channel_luts=(None, None, None), gray_lut=None):
    prepared = PreparedSignedExposure(*map(float, controls))
    retained = []

    def descriptor(spec):
        if spec is None:
            return ExposureLut(_null_ptr(), 0, 0, 1.0)
        values, max_index, scale = spec
        table = np.ascontiguousarray(values, dtype=np.float32)
        retained.append(table)
        return ExposureLut(_ptr(table), table.size, int(max_index), float(scale))

    for index, spec in enumerate(channel_luts):
        prepared.channel[index] = descriptor(spec)
    prepared.gray = descriptor(gray_lut)
    return prepared, retained


def _descriptors(channel_luts=(None, None, None), gray_lut=None):
    descriptors = (FormalExposureLut * 4)()
    retained = []
    for index, spec in enumerate([*channel_luts, gray_lut]):
        if spec is None:
            descriptors[index] = FormalExposureLut(_null_ptr(), 0, 0, 1.0)
            continue
        values, max_index, scale = spec
        table = np.ascontiguousarray(values, dtype=np.float32)
        retained.append(table)
        descriptors[index] = FormalExposureLut(_ptr(table), table.size, int(max_index), float(scale))
    return descriptors, retained


def _compile_sources(kind, output):
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the signed Exposure bridge"
    assert BRIDGE_SOURCE.is_file(), f"missing signed Exposure C ABI bridge: {BRIDGE_SOURCE}"
    assert SOURCE.is_file() and HEADER.is_file(), "missing signed Exposure source/header"
    if kind == "source":
        inputs = [SOURCE]
    else:
        assert OBJECT.is_file(), f"rebuild renderer: missing official object {OBJECT.name}"
        assert OBJECT.stat().st_mtime_ns >= max(SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns), (
            f"rebuild renderer: stale official object {OBJECT.name}"
        )
        inputs = [OBJECT]
    command = [
        compiler, "-std=c++17", "-O2", "-fPIC", "-fno-fast-math", "-ffp-contract=off",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", str(NATIVE / "src"), str(BRIDGE_SOURCE),
        *(str(path) for path in inputs), "-o", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


class Bridge:
    def __init__(self, path):
        self.library = C.CDLL(str(path))
        self.apply_fn = self.library.acr_test_signed_exposure
        self.apply_fn.argtypes = [FLOAT_PTR, FLOAT_PTR, SIZE, FLOAT_PTR, SIZE,
                                  C.POINTER(PreparedSignedExposure)]
        self.apply_fn.restype = C.c_int

    def apply_raw(self, source, destination, pixel_count, mask, mask_count, prepared):
        return self.apply_fn(_ptr(source), _ptr(destination), pixel_count,
                             _ptr(mask), mask_count, C.byref(prepared))

    def apply(self, source, controls=(1.0, 1.0, 1.0, 0.25), *, mask=None,
              channel_luts=(None, None, None), gray_lut=None):
        source = np.ascontiguousarray(source, dtype=np.float32)
        prepared, retained = _prepared(controls, channel_luts, gray_lut)
        mask_values = None if mask is None else np.ascontiguousarray(mask, dtype=np.float32)
        output = np.full(source.size, SENTINEL, dtype=np.float32)
        status = self.apply_raw(source, output, source.size // 3, mask_values,
                                0 if mask_values is None else mask_values.size, prepared)
        return status, output.reshape(source.shape), source, mask_values, prepared, retained


@pytest.fixture(scope="module", params=("source", "official_object"), ids=("source", "official"))
def bridge(request, tmp_path_factory):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_exposure_signed_" + request.param) / ("libsigned_exposure" + suffix)
    _compile_sources(request.param, output)
    return Bridge(output)


@pytest.fixture(scope="module")
def formal_library():
    path = NATIVE / "build" / (
        "libimprint_renderer.0.1.0.dylib" if sys.platform == "darwin"
        else "libimprint_renderer.0.1.0.so"
    )
    assert path.is_file(), f"missing formal native renderer artifact: {path}"
    for artifact in (SOURCE, HEADER, API_SOURCE, PUBLIC_HEADER):
        assert artifact.is_file(), f"missing formal signed Exposure input: {artifact}"
        assert path.stat().st_mtime_ns >= artifact.stat().st_mtime_ns, (
            f"rebuild formal native renderer: stale relative to {artifact.name}"
        )
    for obj, inputs in ((OBJECT, (SOURCE, HEADER)), (API_OBJECT, (API_SOURCE, PUBLIC_HEADER))):
        assert obj.is_file(), f"missing formal signed Exposure object: {obj}"
        assert obj.stat().st_mtime_ns >= max(item.stat().st_mtime_ns for item in inputs), (
            f"rebuild formal native renderer: stale object {obj.name}"
        )
        assert path.stat().st_mtime_ns >= obj.stat().st_mtime_ns, (
            f"relink formal native renderer: stale relative to {obj.name}"
        )
    library = C.CDLL(str(path))
    native._configure_library(library)
    return library, str(path)


def _formal_apply(library, source, controls, *, mask=None,
                  channel_luts=(None, None, None), gray_lut=None,
                  source_samples=None, pixel_count=None, control_count=4,
                  table_count=4, mask_count=None, destination=None,
                  destination_samples=None, control_owner=None, descriptors=None):
    source = np.ascontiguousarray(source, dtype=np.float32)
    controls = np.ascontiguousarray(controls, dtype=np.float32) if control_owner is None else control_owner
    if descriptors is None:
        descriptors, retained = _descriptors(channel_luts, gray_lut)
    else:
        retained = []
    mask_values = None if mask is None else np.ascontiguousarray(mask, dtype=np.float32)
    output = np.full(source.size, SENTINEL, dtype=np.float32) if destination is None else destination
    fn = library.im_native_acr_signed_exposure_prepared_f32
    result = fn(
        _ptr(source), source.size if source_samples is None else source_samples,
        source.size // 3 if pixel_count is None else pixel_count,
        _ptr(controls), control_count, descriptors, table_count,
        _ptr(mask_values), (0 if mask_values is None else mask_values.size)
        if mask_count is None else mask_count,
        _ptr(output), output.size if destination_samples is None else destination_samples,
    )
    return result, output, source, controls, mask_values, descriptors, retained


def _table(values=(0.05, 0.2, 0.45, 0.7, 0.9), scale=4.0, max_index=3):
    return np.asarray(values, dtype=np.float32), max_index, scale


def _expansion_rounding_case():
    minimum_normal = np.finfo(np.float32).tiny
    expansion = np.nextafter(minimum_normal, np.float32(np.inf))
    epsilon = np.nextafter(np.float32(0.0), np.float32(1.0))
    gray = np.float32(2.0 ** -24)
    source = np.full((1, 1, 3), gray, dtype=np.float32)
    controls = np.asarray([1.0, 1.0, expansion, epsilon], dtype=np.float32)
    return source, controls


def test_identity_preserves_signed_zero_and_clamps_finite_input(bridge):
    source = np.asarray(
        [[[-0.0, 0.0, -0.0], [-1001.0, 1001.0, 0.5], [-1000.0, 1000.0, -1000.0]]],
        dtype=np.float32,
    )
    before = source.copy()
    status, actual, _, _, _, _ = bridge.apply(source)
    expected = oracle.apply(source, [1.0, 1.0, 1.0, 0.25])
    assert status == OK
    _assert_bits(actual, expected)
    assert list(_bits(actual[0, 0])) == [0x80000000, 0x00000000, 0x80000000]
    _assert_bits(source, before)


def test_output_is_clamped_after_exposure(bridge):
    source = np.asarray([[[-900.0, 600.0, -0.25], [1000.0, -1000.0, 12.0]]], np.float32)
    controls = [2.0, 1.0, 1.0, 0.25]
    status, actual, *_ = bridge.apply(source, controls)
    assert status == OK
    _assert_bits(actual, oracle.apply(source, controls))
    assert np.max(actual) <= 1000.0 and np.min(actual) >= -1000.0


def test_compression_expansion_precedence_and_exact_branch_boundaries(bridge):
    source = np.asarray([[[0.125, 0.3, 0.7], [0.8, 0.4, 0.2], [0.5, 0.5, 0.5]]], np.float32)
    one_below = np.nextafter(np.float32(1.0), np.float32(0.0))
    cases = (
        [1.25, 1.0, 1.0, 0.125],
        [1.25, one_below, 1.0, 0.125],
        [1.25, 1.0, one_below, 0.125],
        [1.25, 0.65, 0.45, 0.125],  # compression branch takes precedence
    )
    for controls in cases:
        status, actual, *_ = bridge.apply(source, controls)
        assert status == OK
        _assert_bits(actual, oracle.apply(source, controls))
    linear_status, linear, *_ = bridge.apply(source, cases[0])
    assert linear_status == OK
    _assert_bits(linear, np.multiply(source, np.float32(1.25), dtype=np.float32))


def test_expansion_half_is_rounded_before_adding_delta_and_root(bridge):
    source, controls = _expansion_rounding_case()
    status, actual, *_ = bridge.apply(source, controls)
    assert status == OK
    _assert_bits(actual, oracle.apply(source, controls))
    # The raw sequence computes float(expansion * 0.5) first. Reassociating to
    # 0.5 * (delta + root) before multiplying gives 0x1a000000 instead.
    assert list(_bits(actual).flat) == [0x19FFFFFF, 0x19FFFFFF, 0x19FFFFFF]


def test_negative_mask_floor_and_fma_counterexamples_are_locked(bridge):
    source = np.asarray([[[0.5, 0.5, 0.5]]], np.float32)
    mask = np.asarray([[-0.125]], np.float32)
    status, actual, *_ = bridge.apply(source, mask=mask)
    expected = oracle.apply(source, [1.0, 1.0, 1.0, 0.25], mask=mask)
    assert status == OK
    _assert_bits(actual, expected)
    # This fixed output changes if mask*4 is decomposed with truncation to zero.
    assert int(_bits(actual).flat[0]) == 0x3EB5209B

    table = np.asarray([0.0, 7.817676067352295, -6.387670516967773], np.float32)
    source = np.asarray([[[0.25, 1.0, 1.0]]], np.float32)
    spec = (table, 1, 4.5)
    status, actual, *_ = bridge.apply(source, channel_luts=(spec, spec, spec))
    assert status == OK
    # A product-plus-add interpolation yields 0x40c15820; the recovered FMA is one ULP higher.
    assert int(_bits(actual).flat[0]) == 0x40C15821
    _assert_bits(actual, oracle.apply(source, [1, 1, 1, 0.25], channel_luts=(spec, spec, spec)))


def test_mask_sign_fraction_integer_and_clamp_ranges(bridge):
    samples = np.asarray([-0.125, -0.25, -0.375, -0.5, 0.125, 0.25, 0.5, -16.0, 16.0, -18.0, 18.0], np.float32)
    source = np.tile(np.asarray([[[0.125, 0.5, 0.875]]], np.float32), (1, samples.size, 1))
    mask = samples.reshape(1, -1)
    status, actual, *_ = bridge.apply(source, mask=mask)
    assert status == OK
    _assert_bits(actual, oracle.apply(source, [1, 1, 1, 0.25], mask=mask))


def test_channel_group_is_all_or_none_and_gray_table_is_independent(bridge):
    lut = _table()
    source = np.asarray([[[0.2, 0.4, 0.6], [0.7, 0.5, 0.25]]], np.float32)
    for channel_luts, gray_lut in (
        ((lut, lut, lut), None),
        ((lut, lut, None), None),
        ((None, None, None), lut),
        ((lut, lut, lut), lut),
    ):
        status, actual, *_ = bridge.apply(source, channel_luts=channel_luts, gray_lut=gray_lut)
        assert status == OK
        _assert_bits(actual, oracle.apply(source, [1, 1, 1, 0.25],
                                          channel_luts=channel_luts, gray_lut=gray_lut))

    unused = (np.asarray([0.4], np.float32), 7, np.nan)
    partial = (unused, None, None)
    status, actual, *_ = bridge.apply(source, channel_luts=partial, gray_lut=lut)
    assert status == OK
    _assert_bits(actual, oracle.apply(source, [1, 1, 1, 0.25],
                                      channel_luts=partial, gray_lut=lut))


def test_lut_branches_skip_at_one_and_use_the_last_guard_node(bridge):
    flat = _table(values=(0.125, 0.125, 0.125, 0.125), scale=2.0, max_index=2)
    source = np.asarray([[[0.5, 1.0, 0.25], [1.0, 1.0, 1.0]]], np.float32)
    status, actual, *_ = bridge.apply(source, channel_luts=(flat, flat, flat), gray_lut=flat)
    assert status == OK
    _assert_bits(actual, oracle.apply(source, [1, 1, 1, 0.25],
                                      channel_luts=(flat, flat, flat), gray_lut=flat))

    guarded = _table(values=(0.0, 0.25, 0.75), scale=4.0, max_index=1)
    source = np.asarray([[[0.375, 0.375, 0.375]]], np.float32)
    status, actual, *_ = bridge.apply(source, channel_luts=(guarded,) * 3)
    assert status == OK
    _assert_bits(actual, oracle.apply(source, [1, 1, 1, 0.25], channel_luts=(guarded,) * 3))


def test_signed_channel_lut_values_allow_negative_gray_fraction_and_guard_lower_index(bridge):
    negative = _table(values=(-0.2, -0.2, -0.2, -0.2), scale=1.0, max_index=2)
    gray = _table(values=(0.1, 0.3, 0.6, 0.9), scale=1.0, max_index=2)
    source = np.asarray([[[0.25, 0.25, 0.25]]], np.float32)
    status, actual, *_ = bridge.apply(source, channel_luts=(negative,) * 3, gray_lut=gray)
    assert status == OK
    _assert_bits(actual, oracle.apply(source, [1, 1, 1, 0.25],
                                      channel_luts=(negative,) * 3, gray_lut=gray))

    too_negative = _table(values=(-0.6, -0.6, -0.6, -0.6), scale=1.0, max_index=2)
    failing_gray = _table(values=(0.1, 0.3, 0.6, 0.9), scale=2.0, max_index=2)
    prepared, retained = _prepared(channel_luts=(too_negative,) * 3, gray_lut=failing_gray)
    output = np.full(source.size, SENTINEL, np.float32)
    before = output.copy()
    status = bridge.apply_raw(source, output, 1, None, 0, prepared)
    assert status != OK  # -1.2 truncates to signed index -1 after gray LUT scaling.
    _assert_bytes_unchanged(output, before)
    assert retained


def test_lut_index_overrun_fails_without_writing_output(bridge):
    table = np.asarray([0.0, 0.5], np.float32)
    spec = (table, 0, 4.0)
    source = np.asarray([[[0.75, 0.75, 0.75]]], np.float32)
    prepared, retained = _prepared(channel_luts=(spec, spec, spec))
    output = np.full(source.size, SENTINEL, np.float32)
    before = output.copy()
    status = bridge.apply_raw(source, output, 1, None, 0, prepared)
    assert status != OK
    _assert_bytes_unchanged(output, before)
    assert retained[0] is table or np.array_equal(retained[0], table)


def test_only_accessed_lut_values_must_be_finite(bridge):
    table = np.asarray([0.25, 0.5, np.nan], np.float32)
    spec = (table, 0, 0.0)
    source = np.asarray([[[0.25, 0.25, 0.25]]], np.float32)
    status, actual, *_ = bridge.apply(source, channel_luts=(spec,) * 3)
    assert status == OK
    _assert_bits(actual, oracle.apply(source, [1, 1, 1, 0.25], channel_luts=(spec,) * 3))

    bad = table.copy()
    bad[0] = np.nan
    bad_spec = (bad, 0, 0.0)
    prepared, _ = _prepared(channel_luts=(bad_spec,) * 3)
    output = np.full(3, SENTINEL, np.float32)
    before = output.copy()
    status = bridge.apply_raw(source, output, 1, None, 0, prepared)
    assert status != OK
    _assert_bytes_unchanged(output, before)


@pytest.mark.parametrize("seed", [2026102101, 2026102102, 2026102103])
def test_random_finite_pixels_masks_controls_and_table_combinations_are_bit_exact(bridge, seed):
    rng = np.random.default_rng(seed)
    source = rng.uniform(-0.8, 0.8, size=(7, 11, 3)).astype(np.float32)
    mask = rng.uniform(-1.0, 1.0, size=(7, 11)).astype(np.float32)
    controls = np.asarray([
        rng.uniform(0.3, 1.8), rng.uniform(0.45, 1.45),
        rng.uniform(0.45, 1.45), rng.uniform(0.05, 0.4),
    ], np.float32)
    lut = _table(values=(0.02, 0.12, 0.28, 0.53, 0.78), scale=4.0, max_index=3)
    modes = (
        ((None, None, None), None),
        ((lut, lut, lut), None),
        ((lut, None, lut), None),
        ((None, None, None), lut),
        ((lut, lut, lut), lut),
    )
    for channel_luts, gray_lut in modes:
        status, actual, *_ = bridge.apply(source, controls, mask=mask,
                                          channel_luts=channel_luts, gray_lut=gray_lut)
        assert status == OK
        expected = oracle.apply(source, controls, mask=mask,
                                channel_luts=channel_luts, gray_lut=gray_lut)
        _assert_bits(actual, expected)


def test_exact_in_place_is_supported_and_partial_overlap_is_rejected(bridge):
    source = np.asarray([[[-0.25, 0.5, 0.75], [0.1, -0.0, 0.4]]], np.float32)
    expected = oracle.apply(source, [1.5, 0.8, 1.2, 0.25])
    prepared, _ = _prepared([1.5, 0.8, 1.2, 0.25])
    in_place = source.copy().reshape(-1)
    status = bridge.apply_raw(in_place, in_place, 2, None, 0, prepared)
    assert status == OK
    _assert_bits(in_place, expected.reshape(-1))

    backing = np.arange(source.size + 1, dtype=np.float32)
    before = backing.copy()
    status = bridge.apply_raw(backing[:-1], backing[1:], 2, None, 0, prepared)
    assert status != OK
    _assert_bytes_unchanged(backing, before)


def test_nonfinite_inputs_masks_and_bad_prepared_controls_fail_atomically(bridge):
    good_source = np.asarray([[[0.25, 0.5, 0.75]]], np.float32)
    cases = [
        (good_source.copy(), np.asarray([[np.inf]], np.float32), [1, 1, 1, 0.25]),
        (np.asarray([[[np.nan, 0.5, 0.75]]], np.float32), None, [1, 1, 1, 0.25]),
    ]
    for controls in ([-1, 1, 1, 0.25], [1, 0, 1, 0.25], [1, 1, 0, 0.25],
                     [1, 1, 1, 0], [np.nan, 1, 1, 0.25]):
        cases.append((good_source.copy(), None, controls))
    for source, mask, controls in cases:
        prepared, retained = _prepared(controls)
        mask_values = None if mask is None else np.ascontiguousarray(mask)
        output = np.full(source.size, SENTINEL, np.float32)
        before = output.copy()
        status = bridge.apply_raw(source, output, 1, mask_values,
                                  0 if mask_values is None else mask_values.size, prepared)
        assert status != OK
        _assert_bytes_unchanged(output, before)

    prepared, _ = _prepared()
    output = np.full(3, SENTINEL, np.float32)
    before = output.copy()
    status = bridge.apply_raw(good_source, output, 1, np.zeros(1, np.float32), 0, prepared)
    assert status != OK
    _assert_bytes_unchanged(output, before)


def test_finite_extreme_mask_and_tiny_expansion_keep_original_ieee_final_clamp(bridge):
    controls = [np.finfo(np.float32).max, 1.0, 1.0, 0.25]
    source = np.asarray([[[0.0, 0.5, -0.5]]], np.float32)
    mask = np.asarray([[16.0]], np.float32)
    status, actual, *_ = bridge.apply(source, controls, mask=mask)
    assert status == OK
    _assert_bits(actual, oracle.apply(source, controls, mask=mask))
    assert list(_bits(actual[0, 0])) == [0x447A0000, 0x447A0000, 0xC47A0000]

    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    controls = [1.0, 1.0, tiny, tiny]
    source = np.asarray([[[0.0, 0.5, 1.0]]], np.float32)
    status, actual, *_ = bridge.apply(source, controls)
    assert status == OK
    _assert_bits(actual, oracle.apply(source, controls))


def test_bridge_rejects_bad_pixel_count_alignment_and_active_buffer_overlap_atomically(bridge):
    prepared, _ = _prepared()
    source = np.asarray([[[0.25, 0.5, 0.75]]], np.float32)
    output = np.full(3, SENTINEL, np.float32)
    before = output.copy()
    for pixel_count, mask_count in ((0, 0), ((1 << 26) + 1, 0)):
        status = bridge.apply_raw(source, output, pixel_count, None, mask_count, prepared)
        assert status != OK
        _assert_bytes_unchanged(output, before)

    source_bytes = np.zeros(16, np.uint8)
    source_owner = (C.c_ubyte * source_bytes.size).from_buffer(source_bytes)
    unaligned_source = C.cast(C.addressof(source_owner) + 1, FLOAT_PTR)
    status = bridge.apply_fn(unaligned_source, _ptr(output), 1, _null_ptr(), 0, C.byref(prepared))
    assert status != OK
    _assert_bytes_unchanged(output, before)

    destination_bytes = np.full(16, 0x5A, np.uint8)
    destination_owner = (C.c_ubyte * destination_bytes.size).from_buffer(destination_bytes)
    unaligned_destination = C.cast(C.addressof(destination_owner) + 1, FLOAT_PTR)
    destination_before = destination_bytes.copy()
    status = bridge.apply_fn(_ptr(source), unaligned_destination, 1, _null_ptr(), 0,
                             C.byref(prepared))
    assert status != OK
    _assert_bytes_unchanged(destination_bytes, destination_before)

    mask_storage = np.asarray([0.25, 0.5, 0.75, 0.125], np.float32)
    mask_before = mask_storage.copy()
    status = bridge.apply_fn(_ptr(source), _ptr(mask_storage), 1, _ptr(mask_storage), 1,
                             C.byref(prepared))
    assert status != OK
    _assert_bytes_unchanged(mask_storage, mask_before)

    table = np.asarray([0.0, 0.5, 1.0], np.float32)
    spec = (table, 1, 2.0)
    prepared, retained = _prepared(channel_luts=(spec, spec, spec))
    table_before = retained[0].copy()
    status = bridge.apply_fn(_ptr(source), _ptr(retained[0]), 1, _null_ptr(), 0,
                             C.byref(prepared))
    assert status != OK
    _assert_bytes_unchanged(retained[0], table_before)


def test_formal_c_api_and_python_adapter_match_the_same_oracle(formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    rng = np.random.default_rng(2026102117)
    source_backing = rng.uniform(-0.9, 0.9, size=(9, 13, 3)).astype(np.float32)
    source = source_backing[:, ::2, :]
    before = source_backing.copy()
    mask = rng.uniform(-0.5, 0.5, size=source.shape[:2]).astype(np.float32)
    controls = np.asarray([1.15, 0.72, 0.9, 0.125], np.float32)
    lut = _table(values=(0.03, 0.18, 0.32, 0.57, 0.83), scale=4.0, max_index=3)
    channel_luts = (lut, lut, lut)
    expected = oracle.apply(source, controls, mask=mask,
                            channel_luts=channel_luts, gray_lut=lut)

    status, actual, *_ = _formal_apply(library, source, controls, mask=mask,
                                       channel_luts=channel_luts, gray_lut=lut)
    assert status == OK
    _assert_bits(actual.reshape(source.shape), expected)

    adapted = native.native_acr_signed_exposure_prepared(
        source, controls, mask=mask, channel_luts=channel_luts, gray_lut=lut
    )
    _assert_bits(adapted, expected)
    _assert_bits(source_backing, before)

    tiny_source, tiny_controls = _expansion_rounding_case()
    status, actual, *_ = _formal_apply(library, tiny_source, tiny_controls)
    assert status == OK
    assert list(_bits(actual).flat) == [0x19FFFFFF, 0x19FFFFFF, 0x19FFFFFF]
    tiny_adapted = native.native_acr_signed_exposure_prepared(tiny_source, tiny_controls)
    assert list(_bits(tiny_adapted).flat) == [0x19FFFFFF, 0x19FFFFFF, 0x19FFFFFF]


def test_partial_invalid_channel_group_is_bypassed_in_formal_api_and_adapter(formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    source = np.asarray([[[0.2, 0.4, 0.6], [0.7, 0.5, 0.25]]], np.float32)
    controls = np.asarray([1.0, 1.0, 1.0, 0.25], np.float32)
    unused = (np.asarray([0.4], np.float32), 7, np.nan)
    partial = (unused, None, None)
    gray = _table()
    expected = oracle.apply(source, controls, channel_luts=partial, gray_lut=gray)

    status, actual, *_ = _formal_apply(
        library, source, controls, channel_luts=partial, gray_lut=gray,
    )
    assert status == OK
    _assert_bits(actual.reshape(source.shape), expected)
    adapted = native.native_acr_signed_exposure_prepared(
        source, controls, channel_luts=partial, gray_lut=gray,
    )
    _assert_bits(adapted, expected)


def test_formal_api_rejects_bad_counts_aliases_and_misalignment_atomically(formal_library):
    library, _ = formal_library
    source = np.asarray([[[0.25, 0.5, 0.75]]], np.float32)
    controls = np.asarray([1, 1, 1, 0.25], np.float32)
    output = np.full(3, SENTINEL, np.float32)
    before = output.copy()
    failures = (
        dict(source_samples=2), dict(pixel_count=0), dict(control_count=3),
        dict(table_count=3), dict(mask=np.zeros((1, 1), np.float32), mask_count=0),
        dict(destination_samples=2),
    )
    for overrides in failures:
        status, result, *_ = _formal_apply(library, source, controls, destination=output.copy(), **overrides)
        assert status != OK
        _assert_bytes_unchanged(result, before)

    for bad_source, bad_mask, channel_luts in (
        (np.asarray([[[np.nan, 0.5, 0.75]]], np.float32), None, (None, None, None)),
        (source, np.asarray([[np.inf]], np.float32), (None, None, None)),
        (source, None, ((np.asarray([np.nan, 0.5], np.float32), 0, 0.0),) * 3),
    ):
        status, result, *_ = _formal_apply(
            library, bad_source, controls, mask=bad_mask, channel_luts=channel_luts,
            destination=np.full(3, SENTINEL, np.float32),
        )
        assert status != OK
        _assert_bytes_unchanged(result, before)

    mask_owner = np.asarray([0.25, SENTINEL, SENTINEL], np.float32)
    mask_before = mask_owner.copy()
    status, result, *_ = _formal_apply(
        library, source, controls, mask=mask_owner[:1], destination=mask_owner[:3],
    )
    assert status != OK
    _assert_bytes_unchanged(mask_owner, mask_before)

    table_owner = np.asarray([0.0, 0.5, 1.0, SENTINEL, SENTINEL, SENTINEL], np.float32)
    table_before = table_owner.copy()
    table_spec = (table_owner[:3], 1, 2.0)
    status, result, *_ = _formal_apply(
        library, source, controls, channel_luts=(table_spec,) * 3,
        destination=table_owner[:3],
    )
    assert status != OK
    _assert_bytes_unchanged(table_owner, table_before)

    owner = np.full(16, SENTINEL, np.float32)
    owner[:4] = controls
    status, result, *_ = _formal_apply(
        library, source, controls, destination=owner[:3], control_owner=owner,
    )
    assert status != OK
    _assert_bytes_unchanged(owner, np.concatenate((controls, np.full(12, SENTINEL, np.float32))))

    byte_owner = np.zeros(16, np.uint8)
    raw = (C.c_ubyte * byte_owner.size).from_buffer(byte_owner)
    unaligned_source = C.cast(C.addressof(raw) + 1, FLOAT_PTR)
    descriptors, retained = _descriptors()
    status = library.im_native_acr_signed_exposure_prepared_f32(
        unaligned_source, 3, 1, _ptr(controls), controls.size, descriptors, 4,
        _null_ptr(), 0, _ptr(output), output.size,
    )
    assert status != OK
    _assert_bytes_unchanged(output, before)

    output_bytes = np.full(16, 0xA5, np.uint8)
    output_raw = (C.c_ubyte * output_bytes.size).from_buffer(output_bytes)
    unaligned_output = C.cast(C.addressof(output_raw) + 1, FLOAT_PTR)
    output_bytes_before = output_bytes.copy()
    status = library.im_native_acr_signed_exposure_prepared_f32(
        _ptr(source), source.size, 1, _ptr(controls), controls.size, descriptors, 4,
        _null_ptr(), 0, unaligned_output, 3,
    )
    assert status != OK
    _assert_bytes_unchanged(output_bytes, output_bytes_before)


@pytest.mark.parametrize("bad_controls", [
    [-1, 1, 1, 0.25], [1, 0, 1, 0.25], [1, 1, 0, 0.25], [1, 1, 1, 0],
    [np.inf, 1, 1, 0.25],
])
def test_formal_api_rejected_control_state_is_atomic(formal_library, bad_controls):
    library, _ = formal_library
    source = np.asarray([[[0.25, 0.5, 0.75]]], np.float32)
    controls = np.asarray(bad_controls, np.float32)
    output = np.full(3, SENTINEL, np.float32)
    before = output.copy()
    status, actual, *_ = _formal_apply(library, source, controls, destination=output)
    assert status != OK
    _assert_bytes_unchanged(actual, before)


def test_python_adapter_rejects_bad_shapes_and_descriptors_before_loading_library(monkeypatch):
    monkeypatch.setattr(native, "_get_library", lambda: pytest.fail("invalid adapter input loaded native library"))
    source = np.zeros((1, 1, 3), np.float32)
    with pytest.raises(ValueError):
        native.native_acr_signed_exposure_prepared(source, [1, 1, 1])
    with pytest.raises(ValueError):
        native.native_acr_signed_exposure_prepared(source, [1, 1, 1, 0.25], channel_luts=(None, None))
    with pytest.raises(ValueError):
        native.native_acr_signed_exposure_prepared(source, [1, 1, 1, 0.25], mask=np.zeros((2, 1), np.float32))
    with pytest.raises(ValueError):
        native.native_acr_signed_exposure_prepared(
            source, [1, 1, 1, 0.25], channel_luts=((np.zeros(2, np.float32), True, 1.0), None, None)
        )
