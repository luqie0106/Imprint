from __future__ import annotations

import ctypes
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BUILD = NATIVE / "build"
SPEC = importlib.util.spec_from_file_location(
    "core3_refine_oracle", ROOT / "tests/acr_support/core3_refine_oracle.py")
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)

PTR = ctypes.POINTER(ctypes.c_float)
SHARED_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
OFFICIAL_OBJECT = BUILD / (
    "CMakeFiles/imprint_renderer.dir/src/acr_stage3_core3_refine.cpp.o")
BRIDGE_CPP = NATIVE / "tools/acr_stage3_core3_refine_bridge.cpp"
REFINE_CPP = NATIVE / "src/acr_stage3_core3_refine.cpp"


def _compile_bridge(target: Path, linked_source: Path) -> Path:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "a C++ compiler is required to build the core3 refine bridge"
    shared_flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    command = [
        compiler, "-std=c++17", "-O2", "-fPIC", "-fno-fast-math",
        "-ffp-contract=off", shared_flag, "-I", str(NATIVE / "src"),
        str(BRIDGE_CPP), str(linked_source), "-o", str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, (
        "core3 refine bridge link failed:\n" + completed.stdout + completed.stderr)
    return target


class Bridge:
    def __init__(self, path: Path):
        self.library = ctypes.CDLL(str(path))
        self.function = self.library.core3_refine_bridge
        self.function.argtypes = [
            ctypes.c_uint32, PTR, PTR, PTR, PTR,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32,
            PTR, PTR, PTR, PTR,
        ]
        self.function.restype = ctypes.c_int

    def call(self, kernel, guide, channel_a, channel_b, mask, stride, width,
             phase, controls, output_guide, output_a, output_b,
             *, guide_pointer=None, a_pointer=None, b_pointer=None,
             mask_pointer=None, controls_pointer=None,
             output_guide_pointer=None, output_a_pointer=None,
             output_b_pointer=None):
        if guide_pointer is None and guide is not None:
            guide_pointer = oracle.pointer(guide)
        if a_pointer is None and channel_a is not None:
            a_pointer = oracle.pointer(channel_a)
        if b_pointer is None and channel_b is not None:
            b_pointer = oracle.pointer(channel_b)
        if mask_pointer is None and mask is not None:
            mask_pointer = oracle.pointer(mask)
        if controls_pointer is None and controls is not None:
            controls_pointer = controls.ctypes.data_as(PTR)
        if output_guide_pointer is None and output_guide is not None:
            output_guide_pointer = output_guide.ctypes.data_as(PTR)
        if output_a_pointer is None and output_a is not None:
            output_a_pointer = output_a.ctypes.data_as(PTR)
        if output_b_pointer is None and output_b is not None:
            output_b_pointer = output_b.ctypes.data_as(PTR)
        return self.function(
            kernel, guide_pointer, a_pointer, b_pointer, mask_pointer,
            stride, width, phase, controls_pointer,
            output_guide_pointer, output_a_pointer, output_b_pointer)

    def execute(self, kernel, guide, channel_a, channel_b, mask, width, phase,
                lower=0.4, upper=0.4):
        stride = guide.shape[1]
        controls = np.asarray([lower, upper], dtype=np.float32)
        out_guide = np.full(width + 8, np.float32(-111.0), dtype=np.float32)
        out_a = np.full(width + 8, np.float32(-222.0), dtype=np.float32)
        out_b = np.full(width + 8, np.float32(-333.0), dtype=np.float32)
        status = self.call(
            kernel, guide, channel_a, channel_b, mask, stride, width, phase,
            controls, out_guide, out_a, out_b)
        return status, out_guide, out_a, out_b


@pytest.fixture(scope="module", params=("source", "official_object"))
def bridge(request, tmp_path_factory):
    temp = tmp_path_factory.mktemp("core3_refine_" + request.param)
    target = temp / ("libcore3_refine" + SHARED_SUFFIX)
    if request.param == "source":
        assert REFINE_CPP.exists(), f"refine source is missing: {REFINE_CPP}"
        return Bridge(_compile_bridge(target, REFINE_CPP))

    assert OFFICIAL_OBJECT.exists(), (
        "formal imprint_renderer core3 refine object is missing; build it before "
        "running official_object cases")
    assert OFFICIAL_OBJECT.stat().st_mtime_ns >= REFINE_CPP.stat().st_mtime_ns, (
        "formal imprint_renderer core3 refine object is stale; rebuild it before "
        "running official_object cases")
    return Bridge(_compile_bridge(target, OFFICIAL_OBJECT))


def _inputs(seed, width, *, pad=0):
    stride = width + 16 + (pad % 4) * 4
    guide = oracle.aligned_rows(seed, width, stride, low=-0.5, high=1.5)
    a = oracle.aligned_rows(seed + 1, width, stride, low=-1.5, high=2.5)
    b = oracle.aligned_rows(seed + 2, width, stride, low=-1.5, high=2.5)
    mask = oracle.aligned_rows(seed + 3, width, stride, low=0.0, high=1.0)
    rng = np.random.default_rng(seed + 4)
    mask[:] = rng.choice(np.asarray([0.0, 0.25, 0.5, 1.0], np.float32), mask.shape)
    return guide, a, b, mask


def _assert_output(bridge, kernel, planes, width, phase, lower=0.4, upper=0.4):
    guide, a, b, mask = planes
    before = [p.view(np.uint32).copy() for p in planes]
    status, out_g, out_a, out_b = bridge.execute(
        kernel, guide, a, b, mask, width, phase, lower, upper)
    assert status == oracle.OK
    expected = oracle.reference(kernel, guide, a, b, mask, width, phase, lower, upper)
    for actual, reference in zip((out_g, out_a, out_b), expected):
        oracle.assert_ieee_equal(actual[:width], reference)
    assert np.all(out_g[width:] == np.float32(-111.0))
    assert np.all(out_a[width:] == np.float32(-222.0))
    assert np.all(out_b[width:] == np.float32(-333.0))
    for plane, saved in zip(planes, before):
        assert np.array_equal(plane.view(np.uint32), saved)
    return out_g, out_a, out_b


@pytest.mark.parametrize("kernel", [0, 1], ids=["00814230", "008145b4"])
def test_random_strides_phases_and_float32_rows_match_oracle(bridge, kernel):
    for width in (4, 8, 12, 20):
        for seed in range(3):
            planes = _inputs(5100 + seed * 31 + width, width, pad=seed)
            for phase in (0, 1):
                _assert_output(bridge, kernel, planes, width, phase)


@pytest.mark.parametrize("kernel", [0, 1], ids=["00814230", "008145b4"])
def test_mask_zero_one_and_noncentral_mask_inputs(bridge, kernel):
    width = 8
    guide, a, b, mask = _inputs(72, width)
    guide.fill(np.float32(0.5))
    a.fill(np.float32(0.8))
    b.fill(np.float32(0.8))
    mask.fill(np.float32(0.0))
    x = oracle.GUARD + 1
    y = oracle.CENTER_ROW
    a[y, x] = b[y, x] = np.float32(0.2)
    # The center is zero. One noncentral neighbor is one, so the second row
    # kernel gets a 0.25 mask mean while the first kernel gets exactly zero.
    mask[y - 1, x] = np.float32(1.0)
    out_g, out_a, out_b = _assert_output(
        bridge, kernel, (guide, a, b, mask), width, phase=1 if kernel == 0 else 0,
        lower=1.0, upper=1.0)
    assert out_a[1] == np.float32(0.2) if kernel == 0 else out_a[1] > np.float32(0.2)
    assert out_b[1] == np.float32(0.2) if kernel == 0 else out_b[1] > np.float32(0.2)

    mask.fill(np.float32(1.0))
    _assert_output(bridge, kernel, (guide, a, b, mask), width,
                   phase=1 if kernel == 0 else 0, lower=1.0, upper=1.0)


def test_mask_mean_uses_right_association(bridge):
    width = 8
    guide, a, b, mask = _inputs(73, width)
    guide.fill(np.float32(0.5))
    a.fill(np.float32(0.8))
    b.fill(np.float32(0.8))
    mask.fill(np.float32(0.0))
    y, x = oracle.CENTER_ROW, oracle.GUARD + 1
    a[y, x] = b[y, x] = np.float32(0.2)
    mask[y - 1, x] = np.float32(1.0e20)
    mask[y, x - 1] = np.float32(-1.0e20)
    mask[y, x + 1] = np.float32(3.0)
    mask[y + 1, x] = np.float32(4.0)
    _, out_a, _ = _assert_output(
        bridge, 1, (guide, a, b, mask), width, phase=0,
        lower=1.0, upper=1.0)
    # Right association loses the small 7 next to 1e20 and yields a zero mask;
    # a left-associated sum would yield 7 (mask 1.75) and clamp toward one.
    assert out_a[1].view(np.uint32) == np.float32(0.2).view(np.uint32)


def test_denominator_and_numerator_use_right_association(bridge):
    # Construct a stable cancellation-sensitive witness at a corrected lane.
    width = 4
    witness = None
    changed_channels = ()
    for seed in range(200):
        planes = _inputs(8000 + seed, width, pad=seed % 4)
        rng = np.random.default_rng(9100 + seed)
        for plane in planes[:3]:
            plane[:] = rng.uniform(-0.75, 1.25, plane.shape).astype(np.float32)
        planes[3][:] = np.float32(1.0)
        right = oracle.reference(0, *planes, width, phase=0,
                                 lower=10.0, upper=10.0)
        left = oracle.reference(0, *planes, width, phase=0,
                                lower=10.0, upper=10.0,
                                association="left")
        changed_channels = tuple(
            i for i, (r, l) in enumerate(zip(right[1:], left[1:]))
            if not np.array_equal(r.view(np.uint32), l.view(np.uint32)))
        if changed_channels:
            witness = planes
            break
    assert witness is not None, "could not construct right-association counterexample"
    actual = _assert_output(bridge, 0, witness, width, phase=0,
                            lower=10.0, upper=10.0)
    expected = oracle.reference(0, *witness, width, phase=0,
                                lower=10.0, upper=10.0)
    alternate = oracle.reference(0, *witness, width, phase=0,
                                 lower=10.0, upper=10.0,
                                 association="left")
    for channel_index in changed_channels:
        oracle.assert_ieee_equal(actual[channel_index + 1][:width], expected[channel_index + 1])
        assert not np.array_equal(
            expected[channel_index + 1].view(np.uint32),
            alternate[channel_index + 1].view(np.uint32))


@pytest.mark.parametrize("kernel", [0, 1], ids=["00814230", "008145b4"])
def test_fma_rounding_matches_single_precision_fmaf(bridge, kernel):
    width = 4
    witness = None
    for seed in range(300):
        planes = _inputs(12000 + seed, width, pad=seed % 4)
        rng = np.random.default_rng(12100 + seed)
        for plane in planes[:3]:
            plane[:] = rng.uniform(0.0, 1.0, plane.shape).astype(np.float32)
        planes[3].fill(np.float32(1.0))
        fused = oracle.reference(kernel, *planes, width, phase=kernel,
                                 lower=1.0, upper=1.0)
        separate = oracle.reference(kernel, *planes, width, phase=kernel,
                                    lower=1.0, upper=1.0, fused=False)
        if any(not np.array_equal(f.view(np.uint32), s.view(np.uint32))
               for f, s in zip(fused[1:], separate[1:])):
            witness = planes
            break
    assert witness is not None, "could not construct single-rounding FMA counterexample"
    actual = _assert_output(bridge, kernel, witness, width, phase=kernel,
                            lower=1.0, upper=1.0)
    expected = oracle.reference(kernel, *witness, width, phase=kernel,
                                lower=1.0, upper=1.0)
    separate = oracle.reference(kernel, *witness, width, phase=kernel,
                                lower=1.0, upper=1.0, fused=False)
    oracle.assert_ieee_equal(actual[1][:width], expected[1])
    assert not np.array_equal(expected[1].view(np.uint32), separate[1].view(np.uint32))


@pytest.mark.parametrize("kernel", [0, 1], ids=["00814230", "008145b4"])
def test_guide_raw_copies_out_of_range_values_nan_payload_and_nonfinite_class(bridge, kernel):
    width = 8
    guide, a, b, mask = _inputs(94, width)
    guide.fill(np.float32(0.5))
    a.fill(np.float32(0.25))
    b.fill(np.float32(0.75))
    mask.fill(np.float32(0.5))
    y, x = oracle.CENTER_ROW, oracle.GUARD
    # Corrected paths see out-of-range and nonfinite arithmetic. The opposite
    # phase lanes must preserve the original payload bits exactly.
    a[y, x + 1] = np.float32(-3.0)
    b[y, x + 1] = np.float32(4.0)
    payload_nan_a = np.asarray([0x7FC12345], dtype=np.uint32).view(np.float32)[0]
    payload_nan_b = np.asarray([0xFFC54321], dtype=np.uint32).view(np.float32)[0]
    a[y, x + 3] = payload_nan_a
    b[y, x + 3] = payload_nan_b
    a[y, x + 2] = np.float32(np.inf)
    b[y, x + 2] = np.float32(-np.inf)
    guide[y, x + 4] = np.float32(-0.0)

    for phase in (0, 1):
        outputs = _assert_output(bridge, kernel, (guide, a, b, mask), width,
                                 phase, lower=2.0, upper=2.0)
        expected = oracle.reference(kernel, guide, a, b, mask, width, phase,
                                    2.0, 2.0)
        for actual, reference in zip(outputs, expected):
            oracle.assert_ieee_equal(actual[:width], reference)
        # Guide output is always a bit-for-bit row copy, including signed zero.
        assert np.array_equal(outputs[0][:width].view(np.uint32),
                              guide[y, x:x + width].view(np.uint32))
        # At least one raw-copy channel lane carries each original NaN payload.
        copied_a = outputs[1][:width].view(np.uint32)
        copied_b = outputs[2][:width].view(np.uint32)
        for col in range(width):
            should_copy = (((col ^ phase) & 1) != 0) if kernel == 0 else (((col ^ phase) & 1) == 0)
            if should_copy and col == 3:
                assert copied_a[col] == np.uint32(0x7FC12345)
                assert copied_b[col] == np.uint32(0xFFC54321)


def test_rejections_are_atomic_for_huge_stride_phase_controls_and_pointers(bridge):
    width = 8
    guide, a, b, mask = _inputs(111, width)
    controls = np.asarray([0.25, 0.5], dtype=np.float32)
    out_g = np.full(width + 8, np.float32(-111.0), dtype=np.float32)
    out_a = np.full(width + 8, np.float32(-222.0), dtype=np.float32)
    out_b = np.full(width + 8, np.float32(-333.0), dtype=np.float32)
    sentinels = [p.view(np.uint32).copy() for p in (out_g, out_a, out_b)]

    def rejected(expected, *, call_width=width, stride=None, phase=0,
                 control_values=controls, guide_pointer=None):
        out_g.fill(np.float32(-111.0))
        out_a.fill(np.float32(-222.0))
        out_b.fill(np.float32(-333.0))
        control_array = None if control_values is None else np.asarray(
            control_values, dtype=np.float32)
        status = bridge.call(
            0, guide, a, b, mask,
            guide.shape[1] if stride is None else stride,
            call_width, phase, control_array, out_g, out_a, out_b,
            guide_pointer=guide_pointer)
        if expected is None:
            assert status != oracle.OK
        else:
            assert status == expected
        for output, saved in zip((out_g, out_a, out_b), sentinels):
            assert np.array_equal(output.view(np.uint32), saved)

    rejected(oracle.INVALID_DIMENSIONS, call_width=3)
    rejected(oracle.INVALID_PHASE, phase=2)
    rejected(oracle.INVALID_PARAMETERS, control_values=(-0.01, 0.5))
    rejected(oracle.INVALID_PARAMETERS, control_values=(0.5, -0.01))
    rejected(oracle.INVALID_PARAMETERS, control_values=(np.nan, 0.5))
    rejected(oracle.INVALID_PARAMETERS, control_values=(0.5, np.inf))
    rejected(oracle.INVALID_ARGUMENT, control_values=None)
    null_pointer = ctypes.cast(ctypes.c_void_p(0), PTR)
    rejected(oracle.INVALID_ARGUMENT, guide_pointer=null_pointer)

    max_ptrdiff_samples = int(np.iinfo(np.intp).max) // np.dtype(np.float32).itemsize
    huge_stride = max_ptrdiff_samples + 1
    rejected(None, stride=huge_stride)

    unaligned = oracle.pointer(guide, aligned=False)
    address = ctypes.cast(unaligned, ctypes.c_void_p).value
    unaligned = ctypes.cast(ctypes.c_void_p(address + ctypes.sizeof(ctypes.c_float)), PTR)
    rejected(oracle.UNSUPPORTED_ALIGNMENT, guide_pointer=unaligned)
