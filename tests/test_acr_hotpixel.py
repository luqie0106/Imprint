from __future__ import annotations

import ctypes as C
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
SOURCE = NATIVE / "src/acr_hotpixel.cpp"
HEADER = NATIVE / "src/acr_hotpixel.hpp"
BRIDGE_SOURCE = NATIVE / "tools/acr_hotpixel_bridge.cpp"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_hotpixel.cpp.o"
U16 = C.POINTER(C.c_uint16)
SIZE = C.c_size_t
SIZE_MAX = SIZE(-1).value
SENTINEL = np.uint16(0xA55A)

_SPEC = importlib.util.spec_from_file_location(
    "hotpixel_oracle", ROOT / "tests/acr_support/hotpixel_oracle.py"
)
oracle = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(oracle)


def pointer_at(address: int):
    return C.cast(C.c_void_p(address), U16)


def array_pointer(array: np.ndarray):
    return array.ctypes.data_as(U16)


def set_sample(rows: np.ndarray, x: int, dy: int, dx: int, value: int) -> None:
    rows[dy + 2, x + dx + 2] = value


def single_window(fill: int = 1000) -> np.ndarray:
    return np.full((5, 5), fill, dtype=np.uint16)


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.run = self.library.acr_test_hotpixel
        self.run.argtypes = [U16, SIZE, SIZE, SIZE, C.c_uint32,
                             C.c_uint16, C.c_uint16, U16, SIZE]
        self.run.restype = C.c_int

    def execute(
        self,
        source: np.ndarray,
        width: int,
        phase: int,
        hot: int,
        cold: int,
        *,
        dst: np.ndarray | None = None,
        count: int | None = None,
        stride: int | None = None,
        dst_count: int | None = None,
        src_ptr=None,
        dst_ptr=None,
    ):
        if dst is None:
            dst = np.full(width + 2, SENTINEL, dtype=np.uint16)
        if src_ptr is None:
            src_ptr = array_pointer(source)
        if dst_ptr is None:
            dst_ptr = array_pointer(dst)
        if count is None:
            count = source.size
        if stride is None:
            stride = source.shape[1]
        if dst_count is None:
            dst_count = width
        status = self.run(src_ptr, count, stride, width, phase, hot, cold,
                          dst_ptr, dst_count)
        return status, dst


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    if request.param == "official_object":
        assert SOURCE.exists() and HEADER.exists(), "missing hotpixel source/header"
        assert OBJECT.exists(), "rebuild renderer: missing official hotpixel object"
        assert OBJECT.stat().st_mtime_ns >= max(
            SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns
        ), "rebuild renderer: stale official hotpixel object"
        inputs = [OBJECT]
    else:
        assert SOURCE.exists() and HEADER.exists(), "missing hotpixel source/header"
        inputs = [SOURCE]

    assert BRIDGE_SOURCE.exists(), "missing hotpixel test bridge source"
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "C++ compiler required for hotpixel bridge tests"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_hotpixel_" + request.param) / (
        "libacr_hotpixel_test_bridge" + suffix
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
    subprocess.run(command, check=True, capture_output=True, text=True)
    return Bridge(output)


def assert_matches(bridge, source, width, phase, hot, cold):
    original = source.copy()
    status, actual = bridge.execute(source, width, phase, hot, cold)
    expected = oracle.reference(source, width, phase, hot, cold)
    assert status == 0
    np.testing.assert_array_equal(actual[:width], expected)
    assert np.all(actual[width:] == SENTINEL), "bridge wrote beyond output width"
    np.testing.assert_array_equal(source, original)
    return actual[:width]


@pytest.mark.parametrize("phase", (0, 1))
@pytest.mark.parametrize("kind", ("hot", "cold"))
def test_fixed_hot_and_cold_pixels_use_phase_specific_four_neighbor_extreme(
    bridge, phase, kind
):
    hot_neighbors = (3100, 6200, 4100, 5300)
    cold_neighbors = (45000, 50000, 47000, 48000)
    replacement = hot_neighbors if kind == "hot" else cold_neighbors
    center = 50000 if kind == "hot" else 1000
    rows = single_window(1000 if kind == "hot" else 60000)
    set_sample(rows, 0, 0, 0, center)
    for (dy, dx), value in zip(oracle._REPLACEMENT[phase], replacement):
        set_sample(rows, 0, dy, dx, value)

    actual = assert_matches(
        bridge, rows, 1, phase, hot=10000 if kind == "hot" else 50000,
        cold=50000 if kind == "hot" else 2000,
    )
    expected_extreme = max(replacement) if kind == "hot" else min(replacement)
    assert actual[0] == expected_extreme


def test_phase0_counterexample_is_max_four_not_median_five(bridge):
    rows = single_window(1000)
    set_sample(rows, 0, 0, 0, 65535)
    for (dy, dx), value in (
        ((-2, 0), 47923),
        ((2, 0), 9067),
        ((0, -2), 54993),
        ((0, 2), 18027),
    ):
        set_sample(rows, 0, dy, dx, value)

    expected = oracle.reference(rows, 1, phase=0, hot=1000, cold=0)
    assert expected.tolist() == [54993]
    assert sorted((65535, 47923, 9067, 54993, 18027))[2] == 47923
    actual = assert_matches(bridge, rows, 1, phase=0, hot=1000, cold=0)
    assert actual.tolist() == [54993]


def test_phase_xors_for_each_column_inside_a_multi_pixel_window(bridge):
    rows = np.full((5, 6), 1000, dtype=np.uint16)
    # The high center is at output x=1. Its phase-1 replacement maximum is
    # 50000, while its phase-0 replacement maximum is 1400.
    set_sample(rows, 1, 0, 0, 60000)
    phase1_neighbors = (
        ((-1, -1), 20000), ((-1, 1), 30000),
        ((1, -1), 40000), ((1, 1), 50000),
    )
    phase0_neighbors = (
        ((-2, 0), 1100), ((2, 0), 1200),
        ((0, -2), 1300), ((0, 2), 1400),
    )
    for (dy, dx), value in (*phase1_neighbors, *phase0_neighbors):
        set_sample(rows, 1, dy, dx, value)

    from_phase_zero = assert_matches(
        bridge, rows, 2, phase=0, hot=5000, cold=0
    )
    from_phase_one = assert_matches(
        bridge, rows, 2, phase=1, hot=5000, cold=0
    )
    assert from_phase_zero.tolist() == [1000, 50000]
    assert from_phase_one.tolist() == [1000, 1400]


@pytest.mark.parametrize("kind", ("hot", "cold"))
def test_strict_same_color_threshold_equality_does_not_correct(bridge, kind):
    if kind == "hot":
        # Every other same-color sample equals C - hot, so strict '<' fails.
        rows = single_window(5000)
        set_sample(rows, 0, 0, 0, 10000)
        hot, cold = 5000, 0
    else:
        # Every other same-color sample equals C + cold, so strict '<' fails.
        rows = single_window(2000)
        set_sample(rows, 0, 0, 0, 1000)
        hot, cold = 0, 1000
    actual = assert_matches(bridge, rows, 1, phase=0, hot=hot, cold=cold)
    assert actual[0] == rows[2, 2]


@pytest.mark.parametrize("kind", ("hot", "cold"))
def test_strict_residual_threshold_equality_does_not_correct(bridge, kind):
    rows = single_window(1000 if kind == "hot" else 1004)
    if kind == "hot":
        set_sample(rows, 0, 0, 0, 11000)
        for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            set_sample(rows, 0, *offset, 2875)
        # R = 7500 and T = ((11000 - 1000) * 3 + 2) // 4 = 7500.
        hot, cold = 1000, 0
    else:
        set_sample(rows, 0, 0, 0, 1000)
        # Keep both group medians at 1004 while making R = -3.
        for offset in ((-1, 0), (1, 0), (0, -1)):
            set_sample(rows, 0, *offset, 1003)
        # T = ((1000 - 1004) * 3 + 2) // 4 = -3.
        hot, cold = 0, 0

    actual = assert_matches(bridge, rows, 1, phase=0, hot=hot, cold=cold)
    assert actual[0] == rows[2, 2]


def test_negative_threshold_uses_floor_division(bridge):
    rows = single_window(1001)
    set_sample(rows, 0, 0, 0, 1000)
    # T = (-3 + 2) // 4 == -1. With truncation toward zero, T would be 0
    # and this cold pixel would incorrectly remain unchanged.
    actual = assert_matches(bridge, rows, 1, phase=0, hot=0, cold=0)
    assert actual[0] == 1001


@pytest.mark.parametrize("phase", (0, 1))
@pytest.mark.parametrize("width", (1, 2, 5, 8))
def test_random_u16_windows_match_for_odd_even_widths_and_start_phases(
    bridge, width, phase
):
    rng = np.random.default_rng(0xAC720 + width * 7 + phase)
    stride = width + 4
    source = rng.integers(0, 65536, size=(5, stride), dtype=np.uint16)
    # Mix extremes and unrestricted samples to exercise the alternating phase
    # along multi-column windows.
    source.ravel()[::5] = np.resize(
        np.asarray([0, 1, 32767, 65534, 65535], dtype=np.uint16),
        source.ravel()[::5].size,
    )
    assert_matches(bridge, source, width, phase, hot=8000, cold=12000)


def test_unused_trailing_padding_and_output_sentinels_are_preserved(bridge):
    width, stride = 3, 11
    rng = np.random.default_rng(45109)
    source = rng.integers(0, 65536, size=(5, stride), dtype=np.uint16)
    changed_padding = source.copy()
    changed_padding[:, width + 4:] ^= np.uint16(0xFFFF)
    before = source.copy()

    actual = assert_matches(bridge, source, width, phase=1, hot=5000, cold=7000)
    changed = assert_matches(
        bridge, changed_padding, width, phase=1, hot=5000, cold=7000
    )
    np.testing.assert_array_equal(actual, changed)
    np.testing.assert_array_equal(source, before)


def test_nonoverlapping_output_at_source_access_end_is_allowed(bridge):
    width, stride = 3, 7
    endpoint = 4 * stride + width + 4
    storage = np.arange(endpoint + width, dtype=np.uint16)
    source = storage[:endpoint].reshape(5, stride)
    before = storage.copy()
    destination = pointer_at(storage.ctypes.data + endpoint * storage.itemsize)
    status, _ = bridge.execute(
        source, width, phase=0, hot=0, cold=0, dst=storage,
        count=endpoint, dst_count=width, dst_ptr=destination,
    )
    assert status == 0
    np.testing.assert_array_equal(storage[endpoint:endpoint + width],
                                  oracle.reference(source, width, 0, 0, 0))
    np.testing.assert_array_equal(storage[:endpoint], before[:endpoint])


def test_short_count_and_layout_failures_are_atomic(bridge):
    width, stride = 3, 7
    endpoint = 4 * stride + width + 4
    source = np.arange(5 * stride, dtype=np.uint16).reshape(5, stride)

    cases = (
        # Insufficient final source element, stride smaller than width + 4,
        # zero width, insufficient destination capacity, and invalid phase.
        dict(count=endpoint - 1, stride=stride, width=width, phase=0,
             dst_count=width),
        dict(count=source.size, stride=width + 3, width=width, phase=0,
             dst_count=width),
        dict(count=source.size, stride=stride, width=0, phase=0,
             dst_count=0),
        dict(count=source.size, stride=stride, width=width, phase=0,
             dst_count=width - 1),
        dict(count=source.size, stride=stride, width=width, phase=2,
             dst_count=width),
    )
    for case in cases:
        output = np.full(width + 2, SENTINEL, dtype=np.uint16)
        original = output.copy()
        status, _ = bridge.execute(
            source, case["width"], case["phase"], hot=0, cold=0,
            dst=output, count=case["count"], stride=case["stride"],
            dst_count=case["dst_count"],
        )
        assert status != 0
        np.testing.assert_array_equal(output, original)


@pytest.mark.parametrize(
    "stride,width",
    ((SIZE_MAX, 1), (SIZE_MAX, SIZE_MAX), (SIZE_MAX, SIZE_MAX - 2)),
)
def test_size_t_stride_and_width_overflow_fail_atomically(bridge, stride, width):
    source = np.zeros((5, 5), dtype=np.uint16)
    output = np.full(4, SENTINEL, dtype=np.uint16)
    before = output.copy()
    status, _ = bridge.execute(
        source, width, phase=0, hot=0, cold=0, dst=output,
        count=SIZE_MAX, stride=stride, dst_count=SIZE_MAX,
    )
    assert status != 0
    np.testing.assert_array_equal(output, before)


def test_unaligned_source_and_destination_fail_atomically(bridge):
    width, stride = 3, 7
    endpoint = 4 * stride + width + 4
    source = np.arange(5 * stride, dtype=np.uint16).reshape(5, stride)
    output = np.full(width, SENTINEL, dtype=np.uint16)
    before_output = output.copy()

    raw_source = C.create_string_buffer(source.nbytes + 1)
    C.memmove(C.addressof(raw_source) + 1, source.ctypes.data, source.nbytes)
    status, _ = bridge.execute(
        source, width, phase=0, hot=0, cold=0, dst=output,
        count=endpoint, stride=stride, src_ptr=pointer_at(C.addressof(raw_source) + 1),
    )
    assert status != 0
    np.testing.assert_array_equal(output, before_output)

    raw_destination = C.create_string_buffer(width * 2 + 1)
    C.memset(C.addressof(raw_destination), 0xA5, C.sizeof(raw_destination))
    before_destination = raw_destination.raw
    status, _ = bridge.execute(
        source, width, phase=0, hot=0, cold=0,
        count=endpoint, stride=stride,
        dst_ptr=pointer_at(C.addressof(raw_destination) + 1), dst_count=width,
    )
    assert status != 0
    assert raw_destination.raw == before_destination


def test_full_and_partial_source_destination_overlap_fail_atomically(bridge):
    width, stride = 3, 7
    endpoint = 4 * stride + width + 4
    backing = np.arange(endpoint + width + 2, dtype=np.uint16)
    before = backing.copy()

    for destination_offset in (0, endpoint - 1):
        status, _ = bridge.execute(
            backing[:endpoint].reshape(5, stride), width, phase=0, hot=0, cold=0,
            count=endpoint, stride=stride,
            dst_ptr=pointer_at(backing.ctypes.data + destination_offset * backing.itemsize),
            dst_count=width,
        )
        assert status != 0
        np.testing.assert_array_equal(backing, before)

