"""Instruction-level checks for the recovered ACR id1 WarpRectilinear path."""
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
SOURCE = NATIVE / "src/acr_warp.cpp"
HEADER = SOURCE.with_suffix(".hpp")
BRIDGE_SOURCE = NATIVE / "tools/acr_warp_bridge.cpp"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_warp.cpp.o"
API_SOURCE = NATIVE / "src/acr_warp_api.cpp"
PUBLIC_HEADER = NATIVE / "include/imprint_renderer.h"
FORMAL_LIBRARY = NATIVE / "build" / (
    "libimprint_renderer.0.1.0.dylib"
    if sys.platform == "darwin"
    else "libimprint_renderer.0.1.0.so"
)

FLOAT = C.POINTER(C.c_float)
INT32 = C.POINTER(C.c_int32)
UINT32 = C.POINTER(C.c_uint32)
DOUBLE = C.POINTER(C.c_double)
SIZE = C.c_size_t
SENTINEL = np.float32(-731.25)

OK = 0
INVALID_ARGUMENT = 1
INVALID_RECT = 2
INVALID_DIMENSIONS = 3
INVALID_STRIDE = 4
BUFFER_TOO_SMALL = 5
INVALID_METADATA = 6
UNSUPPORTED_ALIGNMENT = 7
OVERLAP = 8
ARITHMETIC_OVERFLOW = 9
RESOURCE_LIMIT = 10
NON_FINITE_INPUT = 11
UNSUPPORTED_DOMAIN = 12
ALLOCATION_FAILURE = 13
LENGTH_ERROR = 14

_SPEC = importlib.util.spec_from_file_location(
    "warp_oracle", ROOT / "tests/acr_support/warp_oracle.py"
)
oracle = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(oracle)


def float_ptr(array: np.ndarray):
    return array.ctypes.data_as(FLOAT)


def assert_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


class Bridge:
    def __init__(self, path: Path, symbol: str = "acr_test_warp"):
        self.library = C.CDLL(str(path))
        self.run = getattr(self.library, symbol)
        self.run.argtypes = [
            FLOAT,
            SIZE,
            SIZE,
            INT32,
            UINT32,
            DOUBLE,
            DOUBLE,
            SIZE,
            SIZE,
            SIZE,
            FLOAT,
            SIZE,
        ]
        self.run.restype = C.c_int

    def execute(
        self,
        source: np.ndarray,
        bounds,
        image_planes: int,
        param_planes: int,
        geometry,
        coeffs: np.ndarray,
        *,
        row_stride: int | None = None,
        tile_width: int = 64,
        tile_height: int = 64,
        destination: np.ndarray | None = None,
        source_count: int | None = None,
        coeff_count: int | None = None,
        destination_count: int | None = None,
        source_pointer=None,
        destination_pointer=None,
        rects_array: np.ndarray | None = None,
        controls_array: np.ndarray | None = None,
        geometry_array: np.ndarray | None = None,
        coeffs_array: np.ndarray | None = None,
    ):
        source = np.asarray(source)
        if rects_array is None:
            rects_array = np.asarray(bounds, dtype=np.int32)
        if controls_array is None:
            controls_array = np.asarray((image_planes, param_planes), dtype=np.uint32)
        if geometry_array is None:
            geometry_array = np.asarray(geometry, dtype=np.float64)
        if coeffs_array is None:
            coeffs_array = np.asarray(coeffs, dtype=np.float64).reshape(-1)

        top, left, bottom, right = map(int, rects_array[:4])
        height = max(0, bottom - top)
        width = max(0, right - left)
        count = height * width * int(image_planes)
        if destination is None:
            destination = np.full(count + 4, SENTINEL, dtype=np.float32)
        else:
            destination = np.asarray(destination)
        if row_stride is None:
            row_stride = width * int(image_planes)
        if source_count is None:
            source_count = source.size
        if coeff_count is None:
            coeff_count = coeffs_array.size
        if destination_count is None:
            destination_count = destination.size
        if source_pointer is None:
            source_pointer = float_ptr(source.reshape(-1))
        if destination_pointer is None:
            destination_pointer = float_ptr(destination.reshape(-1))

        status = self.run(
            source_pointer,
            int(source_count),
            int(row_stride),
            rects_array.ctypes.data_as(INT32),
            controls_array.ctypes.data_as(UINT32),
            geometry_array.ctypes.data_as(DOUBLE),
            coeffs_array.ctypes.data_as(DOUBLE),
            int(coeff_count),
            int(tile_width),
            int(tile_height),
            destination_pointer,
            int(destination_count),
        )
        return status, destination


@pytest.fixture(scope="module", params=("source", "official_object", "formal_api"))
def bridge(request, tmp_path_factory):
    assert SOURCE.is_file(), "missing Warp source; finish the native implementation first"
    assert HEADER.is_file(), "missing Warp API header"
    assert BRIDGE_SOURCE.is_file(), "missing Warp test bridge source"
    if request.param == "official_object":
        assert OBJECT.is_file(), "rebuild renderer: missing official Warp object"
        assert OBJECT.stat().st_mtime_ns >= max(
            SOURCE.stat().st_mtime_ns, HEADER.stat().st_mtime_ns
        ), "rebuild renderer: stale official Warp object"
        inputs = [OBJECT]
    elif request.param == "source":
        inputs = [SOURCE]
    else:
        assert FORMAL_LIBRARY.is_file(), "rebuild renderer: missing formal renderer library"
        assert API_SOURCE.is_file(), "missing Warp public API source"
        assert PUBLIC_HEADER.is_file(), "missing renderer public header"
        assert FORMAL_LIBRARY.stat().st_mtime_ns >= max(
            SOURCE.stat().st_mtime_ns,
            HEADER.stat().st_mtime_ns,
            API_SOURCE.stat().st_mtime_ns,
            PUBLIC_HEADER.stat().st_mtime_ns,
        ), "rebuild renderer: stale formal renderer library"
        result = Bridge(FORMAL_LIBRARY, "im_native_acr_warp_prepared_f32")
        result.backend = request.param
        return result

    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "C++ compiler required for Warp bridge tests"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_warp_" + request.param) / (
        "libacr_warp_test_bridge" + suffix
    )
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-frounding-math",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I",
        str(NATIVE / "src"),
        str(BRIDGE_SOURCE),
        *(str(path) for path in inputs),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = Bridge(output)
    result.backend = request.param
    return result


def api_expected_status(bridge, core_status):
    """Map WarpStatus to the smaller formal im_status API where needed."""
    if bridge.backend != "formal_api":
        return core_status
    if core_status == OK:
        return 0
    if core_status == BUFFER_TOO_SMALL:
        return 4
    if core_status in (RESOURCE_LIMIT, ALLOCATION_FAILURE, LENGTH_ERROR):
        return 5
    return 1


def make_case(
    height=15,
    width=18,
    planes=3,
    *,
    top=0,
    left=0,
    row_padding=0,
    param_planes=None,
    geometry=(0.37, 0.61, 1.0),
    coeffs=None,
    values=None,
):
    if param_planes is None:
        param_planes = planes
    stride = width * planes + row_padding
    source = np.full((height, stride), np.float32(-0.875), dtype=np.float32)
    if values is None:
        values = np.linspace(
            -0.42, 1.18, height * width * planes, dtype=np.float32
        )
    values = np.asarray(values, dtype=np.float32).reshape(height, width, planes)
    source[:, : width * planes] = values.reshape(height, width * planes)
    if coeffs is None:
        coeffs = np.zeros((param_planes, 6), dtype=np.float64)
        for plane in range(param_planes):
            coeffs[plane, :4] = (
                1.0 + 0.025 * plane,
                0.12 - 0.035 * plane,
                -0.025 + 0.01 * plane,
                0.004,
            )
    bounds = (top, left, top + height, left + width)
    return source, bounds, planes, param_planes, geometry, coeffs, stride


def run_and_compare(bridge, case, *, tile_width=64, tile_height=64, only_tiles=None):
    source, bounds, planes, param_planes, geometry, coeffs, stride = case
    status, destination = bridge.execute(
        source,
        bounds,
        planes,
        param_planes,
        geometry,
        coeffs,
        row_stride=stride,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    assert status == OK
    expected = oracle.reference(
        source,
        bounds,
        planes,
        param_planes,
        geometry,
        coeffs,
        source_stride=stride,
        tile_width=tile_width,
        tile_height=tile_height,
        only_tiles=only_tiles,
    ).reshape(-1)
    assert_bits(destination[: expected.size], expected)
    return destination[: expected.size].reshape(
        bounds[2] - bounds[0], bounds[3] - bounds[1], planes
    ), expected.reshape(bounds[2] - bounds[0], bounds[3] - bounds[1], planes)


def test_random_nonzero_origin_padded_stride_multiplane_and_tiles(bridge):
    rng = np.random.default_rng(0xAC120)
    values = rng.uniform(-0.35, 1.25, size=(14, 17, 3)).astype(np.float32)
    case = make_case(
        14,
        17,
        3,
        top=-9,
        left=23,
        row_padding=7,
        geometry=(0.19, 0.74, 1.23),
        values=values,
    )
    run_and_compare(bridge, case, tile_width=6, tile_height=5)


def test_selected_tiles_keep_full_image_geometry(bridge):
    case = make_case(
        13,
        16,
        3,
        top=-5,
        left=14,
        row_padding=3,
        geometry=(0.27, 0.79, 1.16),
    )
    source, bounds, planes, param_planes, geometry, coeffs, stride = case
    tw, th = 6, 5
    status, actual = bridge.execute(
        source,
        bounds,
        planes,
        param_planes,
        geometry,
        coeffs,
        row_stride=stride,
        tile_width=tw,
        tile_height=th,
    )
    assert status == api_expected_status(bridge, OK)
    selected = {(bounds[0], bounds[1]), (bounds[0] + th, bounds[1] + tw)}
    partial = oracle.reference(
        source,
        bounds,
        planes,
        param_planes,
        geometry,
        coeffs,
        source_stride=stride,
        tile_width=tw,
        tile_height=th,
        only_tiles=selected,
    )
    actual = actual[: partial.size].reshape(partial.shape)
    for tile_top, tile_left in selected:
        ys = slice(tile_top - bounds[0], min(tile_top + th, bounds[2]) - bounds[0])
        xs = slice(tile_left - bounds[1], min(tile_left + tw, bounds[3]) - bounds[1])
        assert_bits(actual[ys, xs], partial[ys, xs])
    unselected = np.isnan(partial)
    assert np.isfinite(partial[~unselected]).all()
    assert unselected.any()


def test_nonmonotonic_radial_area_uses_edge_interior_extrema(bridge):
    height, width = 13, 17
    coeffs = np.asarray(
        [[1.0, -1.35, 0.42, -0.06, 0.0, 0.0]], dtype=np.float64
    )
    case = make_case(
        height,
        width,
        1,
        top=-4,
        left=8,
        param_planes=1,
        geometry=(0.36, 0.68, 1.0),
        coeffs=coeffs,
    )
    got, _ = run_and_compare(bridge, case, tile_width=64, tile_height=64)
    bounds = case[1]
    tile = (bounds[0], bounds[1], bounds[2], bounds[3])
    area = oracle.source_area(bounds, tile, coeffs, case[4])
    prepared = oracle._prepared_geometry(bounds, case[4])
    top_values = [
        oracle._map_point(bounds[0], x, coeffs[0], bounds, prepared)[0]
        for x in range(bounds[1], bounds[3])
    ]
    assert min(top_values) < min(
        oracle._map_point(bounds[0], x, coeffs[0], bounds, prepared)[0]
        for x in (bounds[1], bounds[3] - 1)
    )
    expected_top = max(bounds[0], int(np.floor(min(top_values))) - 2)
    assert area[0] == expected_top
    assert np.isfinite(got).all()


@pytest.mark.parametrize("planes", (1, 2, 4))
def test_parameter_broadcast_and_per_plane_controls(bridge, planes):
    broadcast = make_case(
        12,
        12,
        planes,
        top=4,
        left=-7,
        param_planes=1,
        geometry=(0.55, 0.42, 0.83),
    )
    run_and_compare(bridge, broadcast, tile_width=4, tile_height=3)

    per_plane = make_case(
        12,
        10,
        planes,
        param_planes=planes,
        geometry=(0.55, 0.42, 0.83),
    )
    run_and_compare(bridge, per_plane, tile_width=5, tile_height=4)


def test_identity_nop_is_bitwise_and_accepts_sub_kernel_image(bridge):
    # A 2x3 image cannot enter the 4x4 sampling consumer, but an all-plane
    # identity is returned by the opcode NOP host short-circuit.
    values = np.asarray(
        [0.0, -0.0, 0.25, -0.75, 1.0, 0.5, -1.0, 0.125, 0.875, 0.0, 0.2, 0.4],
        dtype=np.float32,
    ).reshape(2, 3, 2)
    case = make_case(
        2,
        3,
        2,
        param_planes=1,
        coeffs=np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        values=values,
    )
    got, expected = run_and_compare(bridge, case, tile_width=1, tile_height=1)
    assert_bits(got, expected)
    assert_bits(got, values)


def test_boundary_fraction_and_suspect_maxanchor_b_minus_five(bridge):
    coeffs = np.asarray([[1.45, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    values = np.empty((9, 11, 1), dtype=np.float32)
    for row in range(9):
        for column in range(11):
            values[row, column, 0] = np.float32(row * 0.07 + column * 0.013)
    case = make_case(
        9,
        11,
        1,
        top=-3,
        left=5,
        param_planes=1,
        geometry=(0.41, 0.63, 1.0),
        coeffs=coeffs,
        values=values,
    )
    got, expected = run_and_compare(bridge, case, tile_width=32, tile_height=32)
    top, left, bottom, right = case[1]
    area = oracle.source_area(case[1], (top, left, bottom, right), coeffs, case[4])
    assert area[2] == bottom and area[3] == right
    # On the bottom/right edge the map clamps to the last source coordinate.
    # Its requested anchor is beyond b/r-5; the consumer zeroes that fraction
    # and clamps the anchor, so the last source row/column is excluded.
    assert np.isfinite(got).all()
    assert_bits(got, expected)


def test_instruction_probe_differs_from_bilinear_and_float_coefficients(bridge):
    height, width = 11, 13
    yy, xx = np.mgrid[:height, :width]
    values = (((xx * 7 + yy * 11) % 9) / 8.0).astype(np.float32)[..., None]
    precise = np.asarray(
        [[1.0, 0.187500007450580596923828125, -0.04375000037252902984619140625,
          0.0062500000931322574615478515625, 0.0, 0.0]],
        dtype=np.float64,
    )
    case = make_case(
        height,
        width,
        1,
        param_planes=1,
        geometry=(0.43, 0.58, 1.17),
        coeffs=precise,
        values=values,
    )
    got, _ = run_and_compare(bridge, case, tile_width=32, tile_height=32)

    # A double-coefficient oracle and its float-narrowed variant must expose
    # the coefficient precision boundary on at least one hard-probe sample.
    # Compare the same recovered source coordinates with bilinear sampling as
    # a negative control. This isolates the kernel from geometry differences.
    prepared = oracle._prepared_geometry(case[1], case[4])
    bilinear = np.empty_like(got)
    compact = case[0][:, : width].reshape(height, width, 1)
    for row in range(height):
        for column in range(width):
            sy, sx = oracle._map_point(row, column, precise[0], case[1], prepared)
            sy, sx = oracle._clamp_source_coordinates(sy, sx, case[1])
            y0, x0 = int(np.floor(sy)), int(np.floor(sx))
            y1, x1 = min(y0 + 1, height - 1), min(x0 + 1, width - 1)
            fy, fx = sy - y0, sx - x0
            p00 = float(compact[y0, x0, 0])
            p01 = float(compact[y0, x1, 0])
            p10 = float(compact[y1, x0, 0])
            p11 = float(compact[y1, x1, 0])
            value = (
                (1.0 - fy) * ((1.0 - fx) * p00 + fx * p01)
                + fy * ((1.0 - fx) * p10 + fx * p11)
            )
            bilinear[row, column, 0] = np.float32(min(max(value, 0.0), 1.0))
    assert np.any(np.abs(got.astype(np.float64) - bilinear.astype(np.float64)) > 1e-5)

    # This canary lies just above 1.0 in binary64 but narrows to 1.0f. The
    # fraction table boundary then changes for thousands of integer-grid
    # samples, making accidental float coefficient storage observable.
    large_height, large_width = 128, 131
    yy, xx = np.mgrid[:large_height, :large_width]
    large_values = (((xx * 7 + yy * 11) % 9) / 8.0).astype(np.float32)[..., None]
    double_canary = np.asarray(
        [[1.0000000596046448, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64
    )
    large_case = make_case(
        large_height,
        large_width,
        1,
        param_planes=1,
        geometry=(0.37, 0.61, 1.0),
        coeffs=double_canary,
        values=large_values,
    )
    large_got, _ = run_and_compare(
        bridge,
        large_case,
        tile_width=large_width,
        tile_height=large_height,
    )
    narrowed = double_canary.astype(np.float32).astype(np.float64)
    narrowed_ref = oracle.reference(
        large_case[0], large_case[1], large_case[2], large_case[3],
        large_case[4], narrowed, source_stride=large_case[6],
        tile_width=large_width, tile_height=large_height,
    )
    assert np.count_nonzero(
        large_got.view(np.uint32) != narrowed_ref.view(np.uint32)
    ) > 100


def _failure(bridge, case, expected_status, **overrides):
    source, bounds, planes, param_planes, geometry, coeffs, stride = case
    count = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * planes
    destination = np.full(count + 5, SENTINEL, dtype=np.float32)
    before = destination.view(np.uint32).copy()
    call_kwargs = {"row_stride": stride, "destination": destination}
    call_kwargs.update(overrides)
    status, actual_destination = bridge.execute(
        source, bounds, planes, param_planes, geometry, coeffs, **call_kwargs
    )
    assert status == api_expected_status(bridge, expected_status)
    assert np.array_equal(actual_destination.view(np.uint32), before)


def test_rejections_leave_destination_untouched(bridge):
    case = make_case(8, 9, 3)
    source, bounds, planes, param_planes, geometry, coeffs, stride = case
    _failure(bridge, case, INVALID_RECT, rects_array=np.asarray((0, 0, 8, 0), np.int32))
    _failure(bridge, case, INVALID_METADATA, controls_array=np.asarray((0, 1), np.uint32))
    _failure(bridge, case, INVALID_METADATA, controls_array=np.asarray((3, 2), np.uint32))
    _failure(bridge, case, INVALID_STRIDE, row_stride=1)
    _failure(bridge, case, BUFFER_TOO_SMALL, source_count=1)
    _failure(
        bridge,
        case,
        BUFFER_TOO_SMALL,
        destination_count=(bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * planes - 1,
    )
    _failure(bridge, case, BUFFER_TOO_SMALL, coeff_count=coeffs.size - 1)
    _failure(bridge, case, INVALID_METADATA, coeff_count=coeffs.size + 1)
    _failure(bridge, case, INVALID_METADATA, tile_width=0)
    _failure(bridge, case, INVALID_ARGUMENT, source_pointer=FLOAT())
    _failure(
        bridge,
        case,
        INVALID_METADATA,
        geometry_array=np.asarray((1.01, 0.5, 1.0), dtype=np.float64),
    )
    _failure(
        bridge,
        case,
        INVALID_METADATA,
        geometry_array=np.asarray((0.5, 0.5, 0.0), dtype=np.float64),
    )

    nonzero_tangent = coeffs.copy()
    nonzero_tangent[0, 4] = 1e-12
    tangent_case = (*case[:5], nonzero_tangent, case[6])
    _failure(bridge, tangent_case, INVALID_METADATA)

    for bad in (np.float32(np.nan), np.float32(np.inf), np.float32(-np.inf)):
        nonfinite = source.copy()
        nonfinite[0, 0] = bad
        nonfinite_case = (nonfinite, *case[1:])
        _failure(bridge, nonfinite_case, NON_FINITE_INPUT)


def test_invalid_geometry_and_coefficients_are_not_silently_accepted(bridge):
    case = make_case(8, 9, 1, param_planes=1)
    coeffs = case[5].copy()
    coeffs[0, 1] = np.nan
    _failure(bridge, (*case[:5], coeffs, case[6]), INVALID_METADATA)


def test_source_area_below_five_is_rejected_without_writing(bridge):
    case = make_case(
        8,
        9,
        1,
        top=4,
        left=-7,
        param_planes=1,
        geometry=(0.55, 0.42, 0.83),
    )
    source, bounds, planes, param_planes, geometry, coeffs, stride = case
    destination = np.full(8 * 9 + 4, SENTINEL, dtype=np.float32)
    before = destination.view(np.uint32).copy()
    status, actual = bridge.execute(
        source,
        bounds,
        planes,
        param_planes,
        geometry,
        coeffs,
        row_stride=stride,
        tile_width=4,
        tile_height=3,
        destination=destination,
    )
    assert status == api_expected_status(bridge, UNSUPPORTED_DOMAIN)
    assert np.array_equal(actual.view(np.uint32), before)


def test_unaligned_input_returns_status_without_writing(bridge):
    case = make_case(8, 9, 1, param_planes=1)
    source, bounds, planes, param_planes, geometry, coeffs, stride = case
    raw = bytearray(source.nbytes + 8)
    unaligned = np.ndarray(source.shape, dtype=np.float32, buffer=raw, offset=1)
    unaligned[:] = source
    count = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * planes
    destination = np.full(count + 2, SENTINEL, dtype=np.float32)
    before = destination.view(np.uint32).copy()
    status, actual = bridge.execute(
        unaligned,
        bounds,
        planes,
        param_planes,
        geometry,
        coeffs,
        row_stride=stride,
        destination=destination,
    )
    assert status == api_expected_status(bridge, UNSUPPORTED_ALIGNMENT)
    assert np.array_equal(actual.view(np.uint32), before)


def test_source_destination_alias_is_rejected_atomically(bridge):
    height, width, planes = 6, 7, 1
    values = np.arange(height * width * planes, dtype=np.float32)
    case = make_case(height, width, planes, param_planes=1, values=values)
    source, bounds, _, param_planes, geometry, coeffs, stride = case
    source_storage = np.zeros(height * width, dtype=np.float32)
    source_storage[:] = source[:, :width].reshape(-1)
    before = source_storage.view(np.uint32).copy()
    status, _ = bridge.execute(
        source_storage,
        bounds,
        planes,
        param_planes,
        geometry,
        coeffs,
        row_stride=stride,
        source_count=source_storage.size,
        destination=source_storage,
        destination_count=source_storage.size,
        destination_pointer=float_ptr(source_storage),
    )
    assert status == api_expected_status(bridge, OVERLAP)
    assert np.array_equal(source_storage.view(np.uint32), before)


@pytest.mark.parametrize("overlap_kind", ("rects", "coefficients"))
def test_destination_metadata_alias_is_rejected_atomically(bridge, overlap_kind):
    case = make_case(6, 7, 1, param_planes=1)
    source, bounds, planes, param_planes, geometry, coeffs, stride = case
    count = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * planes
    raw = bytearray(max(count * 4, 64))
    if overlap_kind == "rects":
        rects = np.ndarray((4,), dtype=np.int32, buffer=raw, offset=0)
        rects[:] = bounds
        coeff_array = np.ascontiguousarray(coeffs.reshape(-1), dtype=np.float64)
        destination_pointer = C.cast(C.addressof(C.c_char.from_buffer(raw)), FLOAT)
        before = bytes(raw)
        status, _ = bridge.execute(
            source,
            bounds,
            planes,
            param_planes,
            geometry,
            coeffs,
            row_stride=stride,
            rects_array=rects,
            coeffs_array=coeff_array,
            destination_pointer=destination_pointer,
            destination_count=count,
        )
    else:
        coeff_array = np.ndarray((6,), dtype=np.float64, buffer=raw, offset=0)
        coeff_array[:] = coeffs.reshape(-1)
        destination_pointer = C.cast(C.addressof(C.c_char.from_buffer(raw)), FLOAT)
        before = bytes(raw)
        status, _ = bridge.execute(
            source,
            bounds,
            planes,
            param_planes,
            geometry,
            coeffs,
            row_stride=stride,
            coeffs_array=coeff_array,
            destination_pointer=destination_pointer,
            destination_count=count,
        )
    assert status == api_expected_status(bridge, OVERLAP)
    assert bytes(raw) == before


def test_checked_overflow_and_resource_limit_statuses(bridge):
    overflow_source = np.zeros(8 * 6, dtype=np.float32)
    coeffs = np.asarray([[1.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    status, destination = bridge.execute(
        overflow_source,
        ((1 << 31) - 9, 0, (1 << 31) - 1, 6),
        1,
        1,
        (0.5, 0.5, 1.2),
        coeffs,
        row_stride=6,
        source_count=overflow_source.size,
        destination=np.full(8 * 6, SENTINEL, dtype=np.float32),
    )
    assert status == api_expected_status(bridge, ARITHMETIC_OVERFLOW)
    assert np.all(destination.view(np.uint32) == np.float32(SENTINEL).view(np.uint32))

    tiny = np.zeros(4, dtype=np.float32)
    huge = (0, 0, 400_000, 400)
    status, destination = bridge.execute(
        tiny,
        huge,
        1,
        1,
        (0.5, 0.5, 1.0),
        coeffs,
        row_stride=400,
        source_count=4,
        destination=np.full(8, SENTINEL, dtype=np.float32),
        destination_count=0,
    )
    assert status == api_expected_status(bridge, RESOURCE_LIMIT)
    assert np.all(destination.view(np.uint32) == np.float32(SENTINEL).view(np.uint32))

