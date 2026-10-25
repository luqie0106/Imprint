"""Bit-exact validation for the prepared ACR SDR mask exposure subset."""

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

OK = 0
INVALID = 1
SHORT = 2
ALIAS = 3
ARITHMETIC = 4
LUT_BOUNDS = 5
ALLOC = 6
RESOURCE = 7
NONFINITE = 8

UNSET = object()


def _null_ptr():
    return C.cast(C.c_void_p(0), FLOAT_PTR)


def _ptr(array):
    return array.ctypes.data_as(FLOAT_PTR) if array is not None else _null_ptr()


def _bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def _assert_bytes_unchanged(actual, before):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint8),
                                  np.asarray(before).view(np.uint8))


def _compile_bridge(kind: str, output: Path) -> None:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the SDR mask bridge"
    assert BRIDGE_SOURCE.is_file(), f"missing SDR mask C ABI bridge: {BRIDGE_SOURCE}"
    assert HEADER.is_file() and SOURCE.is_file(), "missing SDR mask source/header"
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
        "-I", str(NATIVE / "src"), str(BRIDGE_SOURCE),
        *(str(path) for path in inputs), "-o", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.apply_fn = self.library.acr_test_exposure_sdr_mask
        self.apply_fn.argtypes = [
            FLOAT_PTR, SIZE, SIZE, SIZE, SIZE, FLOAT_PTR, SIZE, SIZE,
            FLOAT_PTR, SIZE, FLOAT_PTR, SIZE, FLOAT_PTR, SIZE, SIZE,
        ]
        self.apply_fn.restype = C.c_int

    def raw(self, src, dst, controls, *, mask, table=None,
            height=1, width=None, src_stride=None, dst_stride=None,
            mask_stride=None, src_count=None, dst_count=None, mask_count=None,
            control_count=6, table_count=None, src_pointer=UNSET,
            dst_pointer=UNSET, mask_pointer=UNSET, controls_pointer=UNSET,
            table_pointer=UNSET):
        width = (src.size // 3) if width is None else int(width)
        src_stride = width * 3 if src_stride is None else int(src_stride)
        dst_stride = width * 3 if dst_stride is None else int(dst_stride)
        mask_stride = width if mask_stride is None else int(mask_stride)
        src_count = src.size if src_count is None else int(src_count)
        dst_count = dst.size if dst_count is None else int(dst_count)
        mask_count = (0 if mask is None else mask.size) if mask_count is None else int(mask_count)
        table_count = (0 if table is None else table.size) if table_count is None else int(table_count)
        pointers = [
            _ptr(src) if src_pointer is UNSET else src_pointer,
            _ptr(mask) if mask_pointer is UNSET else mask_pointer,
            _ptr(controls) if controls_pointer is UNSET else controls_pointer,
            _ptr(table) if table_pointer is UNSET else table_pointer,
            _ptr(dst) if dst_pointer is UNSET else dst_pointer,
        ]
        return self.apply_fn(
            pointers[0], src_count, src_stride, height, width,
            pointers[1], mask_count, mask_stride,
            pointers[2], control_count, pointers[3], table_count,
            pointers[4], dst_count, dst_stride,
        )

    def apply(self, rgb, controls, *, mask, main_lut=None):
        source = np.ascontiguousarray(rgb, dtype=np.float32)
        masks = np.ascontiguousarray(mask, dtype=np.float32)
        parameters = np.ascontiguousarray(controls, dtype=np.float32)
        table = None if main_lut is None else np.ascontiguousarray(main_lut, dtype=np.float32)
        output = np.full(source.size, SENTINEL, dtype=np.float32)
        status = self.raw(source, output, parameters, mask=masks, table=table,
                          height=source.shape[0], width=source.shape[1])
        return status, output.reshape(source.shape), source, masks, parameters, table


@pytest.fixture(scope="module", params=("source", "official_object"),
                ids=("source", "official"))
def bridge(request, tmp_path_factory):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_exposure_sdr_mask_" + request.param) / (
        "libsdr_mask_exposure" + suffix
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
        assert artifact.is_file(), f"missing formal SDR mask input: {artifact}"
        assert path.stat().st_mtime_ns >= artifact.stat().st_mtime_ns, (
            f"rebuild formal native renderer: stale relative to {artifact.name}"
        )
    for obj, inputs in ((OBJECT, (SOURCE, HEADER)),
                        (API_OBJECT, (API_SOURCE, PUBLIC_HEADER))):
        assert obj.is_file(), f"missing formal SDR mask object: {obj}"
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
                  source_count=None, source_stride=None, height=None, width=None,
                  mask_count=None, mask_stride=None, control_count=None,
                  table_count=None, destination=None, destination_count=None,
                  destination_stride=None):
    source = np.ascontiguousarray(rgb, dtype=np.float32)
    masks = np.ascontiguousarray(mask, dtype=np.float32)
    parameters = np.ascontiguousarray(controls, dtype=np.float32)
    table = None if main_lut is None else np.ascontiguousarray(main_lut, dtype=np.float32)
    output = (np.full(source.size, SENTINEL, dtype=np.float32) if destination is None
              else destination)
    h, w = source.shape[:2]
    fn = library.im_native_acr_sdr_mask_exposure_prepared_f32
    status = fn(
        _ptr(source), source.size if source_count is None else source_count,
        w * 3 if source_stride is None else source_stride,
        h if height is None else height, w if width is None else width,
        _ptr(masks), masks.size if mask_count is None else mask_count,
        w if mask_stride is None else mask_stride,
        _ptr(parameters), parameters.size if control_count is None else control_count,
        _ptr(table), (0 if table is None else table.size)
        if table_count is None else table_count,
        _ptr(output), output.size if destination_count is None else destination_count,
        w * 3 if destination_stride is None else destination_stride,
    )
    return status, output, source, masks, parameters, table


def _controls(*, gain=1.0, scale=3.0, rational=1.0, sqrt=1.0,
              denominator=1.0, epsilon=0.125):
    return np.asarray([gain, scale, rational, sqrt, denominator, epsilon], np.float32)


def _curve():
    # Deterministic nonidentity curve with both rising and falling segments.
    return np.asarray([0.03, 0.28, 0.19, 0.82, 0.57, 0.94, 0.71], np.float32)


def test_oracle_is_version_pinned_and_negative_mask_floor_is_explicit():
    assert oracle.REFERENCE_ID == "acr-013729ac-sdr-mask-no-context-f32-v1"
    assert oracle.BASELINE_ENTRY == 0x013729AC
    assert int(oracle.MASK_COEFFICIENT_BITS) == 0xBEADF655
    assert int(_bits(oracle.MASK_COEFFICIENT)) == 0xBEADF655
    source = np.asarray([[[0.5, 0.5, 0.5]]], np.float32)
    mask = np.asarray([[-0.125]], np.float32)
    expected = oracle.apply(source, mask, _controls())
    assert int(_bits(expected).flat[0]) == 0x3EB5209B


def test_mask_zero_negative_floor_and_signed_clamps_match_raw_bridge(bridge):
    samples = np.asarray([-0.125, -0.25, -0.375, -0.5, 0.0, 0.125,
                          0.25, 0.5, -16.0, 16.0, -18.0, 18.0], np.float32)
    rgb = np.tile(np.asarray([[[0.125, 0.5, 0.875]]], np.float32),
                  (1, samples.size, 1))
    mask = samples.reshape(1, -1)
    status, actual, *_ = bridge.apply(rgb, _controls(), mask=mask)
    assert status == OK
    _assert_bits(actual, oracle.apply(rgb, mask, _controls()))


@pytest.mark.parametrize("denominator", [0.5, 1.0, 1.5], ids=["Q-below-1", "Q-equal-1", "Q-above-1"])
@pytest.mark.parametrize("rational", [0.75, 1.0, 1.25], ids=["A-below-1", "A-equal-1", "A-above-1"])
@pytest.mark.parametrize("sqrt", [0.75, 1.0, 1.5], ids=["B-below-1", "B-equal-1", "B-above-1"])
def test_branch_boundaries_for_Q_A_and_B(bridge, denominator, rational, sqrt):
    rgb = np.asarray([[[0.07, 0.31, 0.68], [0.73, 0.22, 0.49],
                       [0.25, 0.25, 0.25], [0.81, 0.6, 0.14]]], np.float32)
    mask = np.asarray([[0.125, -0.125, 0.0, 0.375]], np.float32)
    controls = _controls(gain=1.13, rational=rational, sqrt=sqrt,
                         denominator=denominator, epsilon=0.0625)
    table = _curve() if rational > 1 else None
    status, actual, *_ = bridge.apply(rgb, controls, mask=mask, main_lut=table)
    assert status == OK
    _assert_bits(actual, oracle.apply(rgb, mask, controls, main_lut=table))


def test_finite_negative_sqrt_amount_is_valid_and_bypasses_the_transform(bridge):
    rgb = np.asarray([[[0.0, 0.25, 0.75], [0.8, 0.4, 0.1]]], np.float32)
    mask = np.asarray([[0.0, -0.125]], np.float32)
    controls = _controls(gain=1.15, sqrt=-np.finfo(np.float32).max)
    status, actual, *_ = bridge.apply(rgb, controls, mask=mask)
    assert status == OK
    _assert_bits(actual, oracle.apply(rgb, mask, controls))


def test_grayscale_colored_ties_and_all_six_pair_tie_branches_match_oracle(bridge):
    ties = np.asarray([
        [0.75, 0.75, 0.25],  # R == G > B
        [0.75, 0.25, 0.75],  # R == B > G
        [0.25, 0.75, 0.75],  # G == B > R
        [0.25, 0.25, 0.75],  # R == G < B
        [0.25, 0.75, 0.25],  # R == B < G
        [0.75, 0.25, 0.25],  # G == B < R
        [0.5, 0.5, 0.5],     # all equal: raw tree selects lo=B, mid=G, hi=R
        [0.8, 0.5, 0.2],     # strictly colored
    ], dtype=np.float32).reshape(1, 8, 3)
    mask = np.asarray([[0.125, -0.125, 0.25, -0.25,
                        0.0, 0.375, -0.375, 0.0625]], np.float32)
    controls = _controls(gain=1.17, scale=3.5, rational=1.3, sqrt=1.4,
                         denominator=0.55, epsilon=0.03125)
    table = _curve()
    status, actual, *_ = bridge.apply(ties, controls, mask=mask, main_lut=table)
    assert status == OK
    _assert_bits(actual, oracle.apply(ties, mask, controls, main_lut=table))


def test_main_curve_and_sqrt_use_random_nonidentity_table_without_resorting(bridge):
    rng = np.random.default_rng(2026102201)
    table = rng.uniform(0.08, 0.98, size=17).astype(np.float32)
    rgb = rng.uniform(0.01, 0.94, size=(5, 7, 3)).astype(np.float32)
    mask = rng.uniform(-0.55, 0.55, size=(5, 7)).astype(np.float32)
    controls = _controls(gain=1.21, scale=11.25, rational=1.35, sqrt=1.8,
                         denominator=0.72, epsilon=0.0078125)
    status, actual, *_ = bridge.apply(rgb, controls, mask=mask, main_lut=table)
    assert status == OK
    _assert_bits(actual, oracle.apply(rgb, mask, controls, main_lut=table))


def test_main_curve_fma_rounding_probe_is_bit_locked(bridge):
    rgb = np.asarray([[[0.5, 0.5, 0.5]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls(scale=np.float32(2.751891613006592), rational=1.25)
    table = np.asarray([
        -3.8350019454956055, 3.608353853225708, -5.974758625030518,
        -1.1657330989837646, 4.157245635986328,
    ], np.float32)
    expected = oracle.apply(rgb, mask, controls, main_lut=table)
    assert list(_bits(expected).flat) == [0x3BB83F95, 0x3BB83F95, 0x3BB83F95]

    coordinate = oracle.mul32(rgb[0, 0, 0], controls[1])
    index = int(float(coordinate))  # This canary coordinate is positive.
    fraction = oracle.sub32(coordinate, np.float32(index))
    delta = oracle.sub32(table[index + 1], table[index])
    separate = oracle.add32(oracle.mul32(fraction, delta), table[index])
    fused = oracle.fma32(fraction, delta, table[index])
    assert int(_bits(separate)) != int(_bits(fused))

    status, actual, *_ = bridge.apply(rgb, controls, mask=mask, main_lut=table)
    assert status == OK
    _assert_bits(actual, expected)


def test_active_main_curve_reads_only_guarded_samples_and_ignores_unused_nan(bridge):
    rgb = np.asarray([[[0.2, 0.4, 0.65]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls(scale=1.0, rational=1.2)
    table = np.asarray([0.1, 0.7, np.nan], np.float32)
    status, actual, *_ = bridge.apply(rgb, controls, mask=mask, main_lut=table)
    assert status == OK
    _assert_bits(actual, oracle.apply(rgb, mask, controls, main_lut=table))

    missing_guard = np.asarray([0.1], np.float32)
    destination = np.full(3, SENTINEL, np.float32)
    before = destination.copy()
    source = rgb.reshape(-1).copy()
    parameters = _controls(scale=4.0, rational=1.2)
    status = bridge.raw(source, destination, parameters, mask=mask.reshape(-1),
                        table=missing_guard, width=1)
    assert status == LUT_BOUNDS
    _assert_bytes_unchanged(destination, before)

    status = bridge.raw(source.reshape(-1), destination, parameters,
                        mask=mask.reshape(-1), table=None, width=1)
    assert status == LUT_BOUNDS
    _assert_bytes_unchanged(destination, before)

    accessed_nan = np.asarray([0.1, np.nan, 0.7], np.float32)
    parameters = _controls(scale=1.0, rational=1.2)
    status = bridge.raw(source.reshape(-1), destination, parameters,
                        mask=mask.reshape(-1), table=accessed_nan, width=1)
    assert status == NONFINITE
    _assert_bytes_unchanged(destination, before)


def test_inactive_main_lut_is_fully_ignored_including_nan_and_short_storage(bridge):
    rgb = np.asarray([[[0.2, 0.4, 0.65], [0.7, 0.1, 0.35]]], np.float32)
    mask = np.zeros((1, 2), np.float32)
    controls = _controls(scale=3.0, rational=1.0)
    table = np.asarray([np.nan], np.float32)
    status, actual, *_ = bridge.apply(rgb, controls, mask=mask, main_lut=table)
    assert status == OK
    _assert_bits(actual, oracle.apply(rgb, mask, controls, main_lut=table))


def test_negative_and_nonmonotonic_curve_samples_are_preserved(bridge):
    rgb = np.asarray([[[0.12, 0.34, 0.67], [0.2, 0.55, 0.9]]], np.float32)
    mask = np.zeros((1, 2), np.float32)
    controls = _controls(scale=4.0, rational=1.4, sqrt=1.0)
    table = np.asarray([-0.2, 0.4, -0.1, 0.8, 0.3, 0.6], np.float32)
    status, actual, *_ = bridge.apply(rgb, controls, mask=mask, main_lut=table)
    assert status == OK
    _assert_bits(actual, oracle.apply(rgb, mask, controls, main_lut=table))


def test_finite_rational_and_sqrt_intermediate_overflow_returns_eight_atomically(bridge):
    cases = [
        (np.asarray([[[2.0, 0.5, 0.25]]], np.float32),
         _controls(gain=np.finfo(np.float32).max), np.zeros((1, 1), np.float32), None),
        (np.asarray([[[np.finfo(np.float32).max, 0.5, 0.25]]], np.float32),
         _controls(denominator=0.5), np.zeros((1, 1), np.float32), None),
        (np.asarray([[[0.2, 0.5, 0.8]]], np.float32),
         _controls(rational=2.0, scale=3.0), np.zeros((1, 1), np.float32),
         np.asarray([0.0, np.finfo(np.float32).max, -np.finfo(np.float32).max,
                     0.5, 0.75], np.float32)),
        (np.asarray([[[0.25, 0.5, 0.75]]], np.float32),
         _controls(scale=1.0, rational=1.2, sqrt=2.0), np.zeros((1, 1), np.float32),
         np.full(3, np.float32(2e20), np.float32)),
    ]
    for source, controls, mask, table in cases:
        if controls[2] > 1.0 and table is None:
            table = _curve()
        output = np.full(source.size, SENTINEL, np.float32)
        before = output.copy()
        status = bridge.raw(source.reshape(-1), output, controls, mask=mask.reshape(-1),
                            table=table, width=1)
        assert status == NONFINITE
        _assert_bytes_unchanged(output, before)

    late_failure = np.asarray([[[0.25, 0.5, 0.75], [2.0, 0.5, 0.25]]], np.float32)
    late_mask = np.zeros((1, 2), np.float32)
    large_gain = _controls(gain=np.finfo(np.float32).max)
    output = np.full(late_failure.size, SENTINEL, np.float32)
    before = output.copy()
    status = bridge.raw(late_failure.reshape(-1), output, large_gain,
                        mask=late_mask.reshape(-1), width=2)
    assert status == NONFINITE
    _assert_bytes_unchanged(output, before)
    with pytest.raises(oracle.NonfiniteIntermediateError):
        oracle.apply(late_failure, late_mask, large_gain)


def test_nonfinite_mask_or_rgb_and_bad_controls_fail_atomically(bridge):
    source = np.asarray([[[0.25, 0.5, 0.75]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    cases = [
        (source.copy(), np.asarray([[np.inf]], np.float32), _controls(), INVALID),
        (np.asarray([[[np.nan, 0.5, 0.75]]], np.float32), mask.copy(), _controls(), INVALID),
        (np.asarray([[[-0.1, 0.5, 0.75]]], np.float32), mask.copy(), _controls(), INVALID),
        (source.copy(), mask.copy(), _controls(gain=-1), INVALID),
        (source.copy(), mask.copy(), _controls(scale=-1), INVALID),
        (source.copy(), mask.copy(), _controls(rational=0), INVALID),
        (source.copy(), mask.copy(), _controls(denominator=0), INVALID),
        (source.copy(), mask.copy(), _controls(epsilon=0), INVALID),
        (source.copy(), mask.copy(), np.asarray([1, 1, 1, np.inf, 1, 0.1], np.float32), INVALID),
    ]
    for rgb, bad_mask, controls, expected_status in cases:
        output = np.full(rgb.size, SENTINEL, np.float32)
        before = output.copy()
        status = bridge.raw(rgb.reshape(-1), output, controls, mask=bad_mask.reshape(-1),
                            width=1)
        assert status == expected_status
        _assert_bytes_unchanged(output, before)


def test_nonzero_padded_strides_preserve_source_and_output_padding(bridge):
    height, width = 4, 5
    source_stride, mask_stride, destination_stride = 3 * width + 4, width + 3, 3 * width + 6
    source_storage = np.full(height * source_stride, np.float32(91.25), np.float32)
    mask_storage = np.full(height * mask_stride, np.float32(-81.5), np.float32)
    destination = np.full(height * destination_stride, SENTINEL, np.float32)
    rng = np.random.default_rng(2026102202)
    packed = rng.uniform(0.0, 0.8, size=(height, width, 3)).astype(np.float32)
    mask = rng.uniform(-0.5, 0.5, size=(height, width)).astype(np.float32)
    for y in range(height):
        source_storage[y * source_stride:y * source_stride + width * 3] = packed[y].reshape(-1)
        mask_storage[y * mask_stride:y * mask_stride + width] = mask[y]
    source_before = source_storage.copy()
    mask_before = mask_storage.copy()
    destination_before = destination.copy()
    controls = _controls(gain=1.1, scale=3.0, rational=1.2, sqrt=1.25,
                         denominator=0.7)
    table = _curve()
    status = bridge.raw(source_storage, destination, controls, mask=mask_storage,
                        table=table, height=height, width=width,
                        src_stride=source_stride, mask_stride=mask_stride,
                        dst_stride=destination_stride)
    assert status == OK
    expected = oracle.apply(packed, mask, controls, main_lut=table)
    for y in range(height):
        _assert_bits(destination[y * destination_stride:y * destination_stride + width * 3],
                     expected[y].reshape(-1))
        _assert_bytes_unchanged(
            destination[y * destination_stride + width * 3:(y + 1) * destination_stride],
            destination_before[y * destination_stride + width * 3:(y + 1) * destination_stride],
        )
    _assert_bytes_unchanged(source_storage, source_before)
    _assert_bytes_unchanged(mask_storage, mask_before)


def test_exact_in_place_is_supported(bridge):
    source = np.asarray([0.2, 0.5, 0.75, 0.8, 0.1, 0.4], np.float32)
    masks = np.asarray([0.125, -0.25], np.float32)
    controls = _controls(gain=1.2, rational=1.0)
    expected = oracle.apply(source.reshape(1, 2, 3), masks.reshape(1, 2), controls)
    before_source = source.copy()
    status = bridge.raw(source, source, controls, mask=masks, width=2)
    assert status == OK
    _assert_bits(source, expected.reshape(-1))
    assert not np.array_equal(_bits(source), _bits(before_source))


def test_source_mask_lut_and_output_aliases_are_rejected_without_writes(bridge):
    controls = _controls(rational=1.2, scale=1.0)
    source = np.asarray([0.2, 0.4, 0.6], np.float32)
    mask = np.asarray([0.0], np.float32)
    table = np.asarray([0.1, 0.4, 0.7], np.float32)

    shared = np.arange(8, dtype=np.float32)
    shared_before = shared.copy()
    status = bridge.raw(shared[:-1], shared[1:], controls, mask=mask,
                        table=table, width=1,
                        src_count=shared.size - 1, dst_count=shared.size - 1)
    assert status == ALIAS
    _assert_bytes_unchanged(shared, shared_before)

    mask_owner = np.asarray([0.0, SENTINEL, SENTINEL], np.float32)
    mask_before = mask_owner.copy()
    status = bridge.raw(source, mask_owner, _controls(rational=1.0),
                        mask=mask_owner[:1], width=1)
    assert status == ALIAS
    _assert_bytes_unchanged(mask_owner, mask_before)

    table_owner = np.asarray([0.1, 0.4, 0.7, SENTINEL], np.float32)
    table_before = table_owner.copy()
    status = bridge.raw(source, table_owner, controls, mask=mask, table=table_owner[:3],
                        width=1)
    assert status == ALIAS
    _assert_bytes_unchanged(table_owner, table_before)


def _misaligned_float_pointer(storage: np.ndarray):
    owner = (C.c_ubyte * storage.size).from_buffer(storage)
    return owner, C.cast(C.addressof(owner) + 1, FLOAT_PTR)


def test_misaligned_source_mask_controls_table_and_output_are_rejected(bridge):
    source = np.asarray([0.25, 0.5, 0.75], np.float32)
    mask = np.asarray([0.0], np.float32)
    controls = _controls()
    destination = np.full(3, SENTINEL, np.float32)
    before = destination.copy()

    source_bytes = np.full(16, 0xA5, np.uint8)
    source_owner, bad_source = _misaligned_float_pointer(source_bytes)
    status = bridge.raw(source, destination, controls, mask=mask,
                        width=1, src_pointer=bad_source)
    assert status == INVALID
    _assert_bytes_unchanged(destination, before)

    mask_bytes = np.full(16, 0xA5, np.uint8)
    mask_owner, bad_mask = _misaligned_float_pointer(mask_bytes)
    status = bridge.raw(source, destination, controls, mask=mask,
                        width=1, mask_pointer=bad_mask)
    assert status == INVALID
    _assert_bytes_unchanged(destination, before)

    controls_bytes = np.full(32, 0xA5, np.uint8)
    controls_owner, bad_controls = _misaligned_float_pointer(controls_bytes)
    status = bridge.raw(source, destination, controls, mask=mask,
                        width=1, controls_pointer=bad_controls)
    assert status == INVALID
    _assert_bytes_unchanged(destination, before)

    table_bytes = np.full(32, 0xA5, np.uint8)
    table_owner, bad_table = _misaligned_float_pointer(table_bytes)
    active = _controls(rational=1.2, scale=1.0)
    status = bridge.raw(source, destination, active, mask=mask,
                        table=np.asarray([0.1, 0.5, 0.8], np.float32),
                        width=1, table_pointer=bad_table, table_count=3)
    assert status == INVALID
    _assert_bytes_unchanged(destination, before)

    destination_bytes = np.full(16, 0xA5, np.uint8)
    destination_owner, bad_destination = _misaligned_float_pointer(destination_bytes)
    destination_bytes_before = destination_bytes.copy()
    status = bridge.raw(source, destination, controls, mask=mask,
                        width=1, dst_pointer=bad_destination)
    assert status == INVALID
    _assert_bytes_unchanged(destination_bytes, destination_bytes_before)


def test_counts_strides_overflow_and_resource_limit_return_distinct_statuses(bridge):
    source = np.asarray([0.25, 0.5, 0.75], np.float32)
    mask = np.asarray([0.0], np.float32)
    controls = _controls()
    destination = np.full(3, SENTINEL, np.float32)
    before = destination.copy()

    status = bridge.raw(source, destination, controls, mask=mask, width=1,
                        control_count=5)
    assert status == INVALID
    _assert_bytes_unchanged(destination, before)

    status = bridge.raw(source, destination, controls, mask=mask, width=1,
                        src_count=2)
    assert status == SHORT
    _assert_bytes_unchanged(destination, before)
    status = bridge.raw(source, destination, controls, mask=mask, width=1,
                        mask_count=0)
    assert status == SHORT
    _assert_bytes_unchanged(destination, before)
    status = bridge.raw(source, destination, controls, mask=mask, width=1,
                        dst_count=2)
    assert status == SHORT
    _assert_bytes_unchanged(destination, before)

    status = bridge.raw(source, destination, controls, mask=mask, width=1,
                        src_stride=2)
    assert status == INVALID
    _assert_bytes_unchanged(destination, before)
    status = bridge.raw(source, destination, controls, mask=mask, width=1,
                        dst_stride=2)
    assert status == INVALID
    _assert_bytes_unchanged(destination, before)
    status = bridge.raw(source, destination, controls, mask=mask, width=1,
                        mask_stride=0)
    assert status == INVALID
    _assert_bytes_unchanged(destination, before)

    size_max = C.c_size_t(-1).value
    status = bridge.raw(source, destination, controls, mask=mask, width=1,
                        src_count=size_max)
    assert status == ARITHMETIC
    _assert_bytes_unchanged(destination, before)
    status = bridge.raw(source, destination, controls, mask=mask,
                        height=2, width=size_max // 2 + 1,
                        src_count=size_max, dst_count=size_max, mask_count=size_max)
    assert status == ARITHMETIC
    _assert_bytes_unchanged(destination, before)

    status = bridge.raw(source, destination, controls, mask=mask, width=(1 << 26) + 1,
                        src_count=3, dst_count=3, mask_count=1)
    assert status == RESOURCE
    _assert_bytes_unchanged(destination, before)


def test_formal_api_and_python_adapter_match_oracle(formal_library, monkeypatch):
    library, path = formal_library
    monkeypatch.setattr(native, "_get_library", lambda: (library, path))
    rng = np.random.default_rng(2026102203)
    rgb = rng.uniform(0.0, 0.8, size=(6, 9, 3)).astype(np.float32)
    mask = rng.uniform(-0.4, 0.4, size=(6, 9)).astype(np.float32)
    controls = _controls(gain=1.19, scale=7.5, rational=1.27, sqrt=1.36,
                         denominator=0.71, epsilon=0.00390625)
    table = np.random.default_rng(2026102204).uniform(0.04, 0.95, 12).astype(np.float32)
    expected = oracle.apply(rgb, mask, controls, main_lut=table)

    status, actual, *_ = _formal_apply(library, rgb, mask, controls, main_lut=table)
    assert status == OK
    _assert_bits(actual.reshape(rgb.shape), expected)
    adapted = native.native_acr_sdr_mask_exposure_prepared(
        rgb, mask, controls, main_lut=table,
    )
    _assert_bits(adapted, expected)

    inactive = _controls(rational=1.0)
    inactive_expected = oracle.apply(rgb, mask, inactive)
    adapted = native.native_acr_sdr_mask_exposure_prepared(
        rgb, mask, inactive, main_lut="ignored when A <= 1",
    )
    _assert_bits(adapted, inactive_expected)


def test_formal_api_maps_short_and_invalid_states_and_keeps_output_unchanged(formal_library):
    library, _ = formal_library
    rgb = np.asarray([[[0.25, 0.5, 0.75]]], np.float32)
    mask = np.zeros((1, 1), np.float32)
    controls = _controls()
    output = np.full(3, SENTINEL, np.float32)
    before = output.copy()

    status, result, *_ = _formal_apply(
        library, rgb, mask, controls, destination=output.copy(), source_count=2,
    )
    assert status == 4  # im_status::buffer_too_small
    _assert_bytes_unchanged(result, before)

    status, result, *_ = _formal_apply(
        library, rgb, mask, controls, destination=output.copy(), control_count=5,
    )
    assert status == 1  # im_status::invalid_argument
    _assert_bytes_unchanged(result, before)

    bad_source = np.asarray([[[-0.25, 0.5, 0.75]]], np.float32)
    status, result, *_ = _formal_apply(
        library, bad_source, mask, controls, destination=output.copy(),
    )
    assert status == 1
    _assert_bytes_unchanged(result, before)

