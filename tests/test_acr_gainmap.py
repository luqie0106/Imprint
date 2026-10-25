"""Instruction-level checks for the ACR id9 GainMap CPU operator."""
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
SOURCE = NATIVE / "src/acr_gainmap.cpp"
HEADER = SOURCE.with_suffix(".hpp")
DEPENDENCY_HEADER = NATIVE / "src/acr_bayer_border.hpp"
BRIDGE_SOURCE = NATIVE / "tools/acr_gainmap_bridge.cpp"
OBJECT = NATIVE / "build/CMakeFiles/imprint_renderer.dir/src/acr_gainmap.cpp.o"
API_SOURCE = NATIVE / "src/acr_gainmap_api.cpp"
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
SIZE_MAX = SIZE(-1).value
SENTINEL = np.float32(-731.25)

_SPEC = importlib.util.spec_from_file_location(
    "gainmap_oracle", ROOT / "tests/acr_support/gainmap_oracle.py"
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
    def __init__(self, path: Path, symbol: str = "acr_test_gainmap"):
        self.library = C.CDLL(str(path))
        self.run = getattr(self.library, symbol)
        self.run.argtypes = [
            FLOAT, SIZE, SIZE, INT32, UINT32, DOUBLE, FLOAT, SIZE,
            C.c_uint16, SIZE, SIZE, FLOAT, SIZE,
        ]
        self.run.restype = C.c_int

    def execute(
        self,
        source: np.ndarray,
        rects8,
        controls9,
        geometry4,
        gain_map: np.ndarray,
        *,
        black_floor: int = 0,
        tile_w: int = 64,
        tile_h: int = 64,
        destination: np.ndarray | None = None,
        source_count: int | None = None,
        map_count: int | None = None,
        destination_count: int | None = None,
        source_pointer=None,
        destination_pointer=None,
        rects_array: np.ndarray | None = None,
        controls_array: np.ndarray | None = None,
        geometry_array: np.ndarray | None = None,
    ):
        source = np.asarray(source)
        gain_map = np.asarray(gain_map)
        rects_array = (
            np.asarray(rects8, dtype=np.int32)
            if rects_array is None else rects_array
        )
        controls_array = (
            np.asarray(controls9, dtype=np.uint32)
            if controls_array is None else controls_array
        )
        geometry_array = (
            np.asarray(geometry4, dtype=np.float64)
            if geometry_array is None else geometry_array
        )
        if destination is None:
            image_top, image_left, image_bottom, image_right = map(
                int, rects_array[:4]
            )
            image_planes = int(controls_array[0])
            output_count = (
                (image_bottom - image_top)
                * (image_right - image_left)
                * image_planes
            )
            destination = np.full(output_count + 3, SENTINEL, dtype=np.float32)
        else:
            destination = np.asarray(destination)
        if source_count is None:
            source_count = source.size
        if map_count is None:
            map_count = gain_map.size
        if destination_count is None:
            destination_count = destination.size
        if source_pointer is None:
            source_pointer = float_ptr(source)
        if destination_pointer is None:
            destination_pointer = float_ptr(destination)
        status = self.run(
            source_pointer,
            source_count,
            int(source.shape[1]) if source.ndim == 2 else 0,
            rects_array.ctypes.data_as(INT32),
            controls_array.ctypes.data_as(UINT32),
            geometry_array.ctypes.data_as(DOUBLE),
            float_ptr(gain_map.reshape(-1)),
            map_count,
            black_floor,
            tile_w,
            tile_h,
            destination_pointer,
            destination_count,
        )
        return status, destination


@pytest.fixture(
    scope="module",
    params=("source", "official_object", "formal_api"),
)
def bridge(request, tmp_path_factory):
    assert SOURCE.is_file(), "missing GainMap source; finish the native implementation first"
    assert HEADER.is_file(), "missing GainMap API header"
    assert DEPENDENCY_HEADER.is_file(), "missing Bayer-border rectangle dependency"
    assert BRIDGE_SOURCE.is_file(), "missing GainMap test bridge source"
    if request.param == "official_object":
        assert OBJECT.is_file(), "rebuild renderer: missing official GainMap object"
        assert OBJECT.stat().st_mtime_ns >= max(
            SOURCE.stat().st_mtime_ns,
            HEADER.stat().st_mtime_ns,
            DEPENDENCY_HEADER.stat().st_mtime_ns,
        ), "rebuild renderer: stale official GainMap object"
        inputs = [OBJECT]
    elif request.param == "source":
        inputs = [SOURCE]
    else:
        assert FORMAL_LIBRARY.is_file(), "rebuild renderer: missing formal renderer library"
        assert API_SOURCE.is_file(), "missing GainMap public API source"
        assert PUBLIC_HEADER.is_file(), "missing renderer public header"
        assert FORMAL_LIBRARY.stat().st_mtime_ns >= max(
            SOURCE.stat().st_mtime_ns,
            HEADER.stat().st_mtime_ns,
            DEPENDENCY_HEADER.stat().st_mtime_ns,
            API_SOURCE.stat().st_mtime_ns,
            PUBLIC_HEADER.stat().st_mtime_ns,
        ), "rebuild renderer: stale formal renderer library"
        return Bridge(FORMAL_LIBRARY, "im_native_acr_gainmap_prepared_f32")

    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "C++ compiler required for GainMap bridge tests"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = tmp_path_factory.mktemp("acr_gainmap_" + request.param) / (
        "libacr_gainmap_test_bridge" + suffix
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
    return Bridge(output)


def make_case(
    height=4,
    width=5,
    planes=3,
    *,
    top=0,
    left=0,
    row_padding=2,
    first_plane=0,
    area_planes=None,
    row_pitch=1,
    col_pitch=1,
    map_shape=(2, 3, 1),
    area=None,
    geometry=None,
    source_values=None,
    map_values=None,
    list_stage=1,
):
    stride = width * planes + row_padding
    source = np.full((height, stride), np.float32(-0.375), dtype=np.float32)
    if source_values is None:
        logical = np.linspace(-0.1, 0.75, height * width * planes, dtype=np.float32)
    else:
        logical = np.asarray(source_values, dtype=np.float32).reshape(-1)
        assert logical.size == height * width * planes
    for y in range(height):
        start = y * width * planes
        source[y, : width * planes] = logical[start : start + width * planes]

    if area is None:
        area = (top, left, top + height, left + width)
    rects = np.asarray(
        (top, left, top + height, left + width, *area), dtype=np.int32
    )
    if area_planes is None:
        area_planes = planes
    controls = np.asarray(
        (
            planes,
            first_plane,
            area_planes,
            row_pitch,
            col_pitch,
            *map_shape,
            list_stage,
        ),
        dtype=np.uint32,
    )
    map_v, map_h, map_planes = map_shape
    if geometry is None:
        spacing_v = 1.0 / max(1, map_v - 1)
        spacing_h = 1.0 / max(1, map_h - 1)
        geometry = (spacing_v, spacing_h, 0.0, 0.0)
    geometry = np.asarray(geometry, dtype=np.float64)
    if map_values is None:
        map_values = np.linspace(
            0.75, 1.75, map_v * map_h * map_planes, dtype=np.float32
        )
    gain_map = np.asarray(map_values, dtype=np.float32).reshape(
        map_v, map_h, map_planes
    )
    return source, rects, controls, geometry, gain_map


def assert_matches(
    bridge,
    source,
    rects,
    controls,
    geometry,
    gain_map,
    *,
    black_floor=0,
    tile_w=64,
    tile_h=64,
):
    before = source.copy()
    status, destination = bridge.execute(
        source,
        rects,
        controls,
        geometry,
        gain_map,
        black_floor=black_floor,
        tile_w=tile_w,
        tile_h=tile_h,
    )
    expected = oracle.reference(
        source,
        source.shape[1],
        rects,
        controls,
        geometry,
        gain_map,
        black_floor=black_floor,
        tile_w=tile_w,
        tile_h=tile_h,
    )
    assert status == 0
    assert_bits(destination[: expected.size], expected.reshape(-1))
    assert np.all(destination[expected.size :] == SENTINEL), "output tail changed"
    assert_bits(source, before)
    return destination[: expected.size].reshape(expected.shape)


def test_hand_calculated_2x2_map_uses_centered_coordinates_and_segment_slope(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=2,
        width=2,
        planes=1,
        row_padding=0,
        map_shape=(2, 2, 1),
        geometry=(1.0, 1.0, 0.0, 0.0),
        source_values=[0.25] * 4,
        map_values=[1.0, 3.0, 5.0, 7.0],
    )
    actual = assert_matches(bridge, source, rects, controls, geometry, gain_map)
    assert_bits(actual[:, :, 0], np.asarray([[0.625, 0.875], [1.0, 1.0]], np.float32))


def test_constant_map_preserves_negative_results_without_lower_clamp(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=2,
        width=3,
        planes=1,
        map_shape=(1, 1, 1),
        geometry=(np.nan, np.nan, np.inf, -np.inf),
        source_values=[-0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        map_values=[2.0],
    )
    actual = assert_matches(bridge, source, rects, controls, geometry, gain_map)
    assert actual[0, 0, 0] == np.float32(-1.0)


def test_half_pixel_clamps_and_signed_map_origins(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=4,
        width=5,
        planes=1,
        top=-3,
        left=5,
        map_shape=(3, 3, 1),
        geometry=(0.4, 0.3, -0.2, 0.45),
        map_values=np.asarray(
            [[0.5, 1.0, 1.5], [1.25, 1.75, 2.25], [2.0, 2.5, 3.0]],
            dtype=np.float32,
        ),
    )
    actual = assert_matches(bridge, source, rects, controls, geometry, gain_map)
    assert actual[0, 0, 0] < actual[-1, -1, 0]


def test_nonzero_and_negative_horizontal_origins_select_both_clamps(bridge):
    for origin_h in (-0.4, 0.4):
        source, rects, controls, geometry, gain_map = make_case(
            height=3,
            width=5,
            planes=1,
            top=-2,
            left=4,
            map_shape=(2, 2, 1),
            geometry=(0.5, 0.25, -0.125, origin_h),
            map_values=[1.0, 4.0, 2.0, 8.0],
        )
        assert_matches(bridge, source, rects, controls, geometry, gain_map)


def test_fifty_seeded_small_images_match_instruction_oracle(bridge):
    rng = np.random.default_rng(20261020)
    for _ in range(50):
        height = int(rng.integers(1, 8))
        width = int(rng.integers(1, 9))
        planes = int(rng.integers(1, 5))
        map_v = int(rng.integers(1, 5))
        map_h = int(rng.integers(1, 5))
        map_planes = int(rng.integers(1, 4))
        geometry = (
            float(rng.uniform(0.2, 1.2)),
            float(rng.uniform(0.2, 1.2)),
            float(rng.uniform(-0.6, 0.6)),
            float(rng.uniform(-0.6, 0.6)),
        )
        source, rects, controls, geometry, gain_map = make_case(
            height=height,
            width=width,
            planes=planes,
            top=int(rng.integers(-8, 8)),
            left=int(rng.integers(-8, 8)),
            row_padding=int(rng.integers(0, 4)),
            map_shape=(map_v, map_h, map_planes),
            geometry=geometry,
            source_values=rng.uniform(-0.5, 1.5, height * width * planes).astype(
                np.float32
            ),
            map_values=rng.uniform(0.25, 2.5, map_v * map_h * map_planes).astype(
                np.float32
            ),
        )
        assert_matches(
            bridge,
            source,
            rects,
            controls,
            geometry,
            gain_map,
            tile_w=int(rng.integers(1, min(5, width) + 1)),
            tile_h=int(rng.integers(1, min(5, height) + 1)),
        )


@pytest.mark.parametrize("map_planes", (1, 2))
def test_absolute_plane_index_broadcast_and_repeated_tail_plane(bridge, map_planes):
    height, width, planes = 3, 4, 5
    values = np.empty((2, 2, map_planes), dtype=np.float32)
    for p in range(map_planes):
        values[:, :, p] = np.float32(p + 1)
    source_values = np.full(height * width * planes, 0.2, dtype=np.float32)
    source, rects, controls, geometry, gain_map = make_case(
        height=height,
        width=width,
        planes=planes,
        first_plane=1,
        area_planes=4,
        map_shape=(2, 2, map_planes),
        source_values=source_values,
        map_values=values,
        geometry=(1.0, 1.0, 0.0, 0.0),
    )
    actual = assert_matches(bridge, source, rects, controls, geometry, gain_map)
    assert_bits(actual[:, :, 0], source_values.reshape(height, width, planes)[:, :, 0])
    if map_planes == 2:
        assert np.all(actual[:, :, 1] <= actual[:, :, 2])


def test_pitched_area_is_clipped_and_aligned_to_area_origin_per_tile(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=8,
        width=9,
        planes=2,
        top=-4,
        left=3,
        row_pitch=3,
        col_pitch=2,
        area=(-2, 5, 8, 14),
        map_shape=(3, 4, 2),
        geometry=(0.45, 0.3, -0.1, 0.15),
    )
    assert_matches(
        bridge,
        source,
        rects,
        controls,
        geometry,
        gain_map,
        tile_w=4,
        tile_h=3,
    )


def test_pitches_are_limited_by_nonempty_area_dimensions(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=6,
        width=7,
        planes=2,
        top=-2,
        left=3,
        area=(0, 5, 2, 7),
        row_pitch=99,
        col_pitch=99,
        map_shape=(2, 3, 2),
    )
    assert_matches(
        bridge,
        source,
        rects,
        controls,
        geometry,
        gain_map,
        tile_w=3,
        tile_h=2,
    )


def test_col_pitch_crosses_segment_boundary_one_absolute_column_at_a_time(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=1,
        width=9,
        planes=1,
        row_padding=0,
        col_pitch=3,
        map_shape=(1, 3, 1),
        geometry=(np.nan, 2.0 / 9.0, np.nan, 0.0),
        source_values=np.full(9, 0.125, dtype=np.float32),
        map_values=[0.25, 2.528853178024292, 3.1443991661071777],
    )
    actual = assert_matches(
        bridge,
        source,
        rects,
        controls,
        geometry,
        gain_map,
        tile_w=9,
        tile_h=1,
    )
    # The first map-knot boundary falls at absolute column 2, between the
    # selected columns 0 and 3. Advancing through column 2 resets the segment
    # there, so column 3 retains the f32 slope/FMA rounding history.
    assert int(actual[0, 3, 0].view(np.uint32)) == 0x3EBF6490
    direct_at_selected_column = np.float32(2.9905126094818115 * 0.125)
    assert int(direct_at_selected_column.view(np.uint32)) == 0x3EBF648F


@pytest.mark.parametrize(
    "area",
    (
        (4, -2, 3, 9),  # reversed vertical range
        (-3, 5, 7, 5),  # empty horizontal range
        (5, 8, 1, 2),   # reversed on both axes
    ),
)
def test_empty_or_reversed_area_means_the_whole_current_tile(bridge, area):
    source, rects, controls, geometry, gain_map = make_case(
        height=6,
        width=7,
        planes=2,
        top=-3,
        left=2,
        area=area,
        row_pitch=3,
        col_pitch=2,
        map_shape=(2, 3, 2),
    )
    assert_matches(
        bridge,
        source,
        rects,
        controls,
        geometry,
        gain_map,
        tile_w=3,
        tile_h=2,
    )


def test_floor_zero_and_stage_one_skip_black_wrap_stage_three_applies_it(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=3,
        width=4,
        planes=1,
        map_shape=(2, 2, 1),
        source_values=np.full(12, 0.4, dtype=np.float32),
        map_values=np.full(4, 0.75, dtype=np.float32),
        list_stage=1,
    )
    stage_one = assert_matches(
        bridge, source, rects, controls, geometry, gain_map, black_floor=4100
    )
    controls_three = controls.copy()
    controls_three[8] = 3
    stage_three = assert_matches(
        bridge,
        source,
        rects,
        controls_three,
        geometry,
        gain_map,
        black_floor=4100,
    )
    zero_floor = assert_matches(
        bridge,
        source,
        rects,
        controls_three,
        geometry,
        gain_map,
        black_floor=0,
    )
    assert np.any(stage_one != stage_three)
    assert_bits(stage_one, zero_floor)


def test_finite_negative_map_and_pixels_are_accepted_and_only_upper_clamped(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=2,
        width=3,
        planes=1,
        map_shape=(1, 2, 1),
        geometry=(np.nan, 0.5, np.nan, -0.2),
        source_values=[-2.0, -1.0, 0.1, 0.5, 1.0, 2.0],
        map_values=[-1.0, 2.0],
    )
    actual = assert_matches(bridge, source, rects, controls, geometry, gain_map)
    assert np.any(actual < 0.0)
    assert actual[-1, -1, 0] == 1.0


def test_non_grid_coordinate_distinguishes_direct_fma_from_1_over_32_sampling(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=1,
        width=7,
        planes=1,
        row_padding=0,
        map_shape=(1, 2, 1),
        geometry=(np.nan, 1.0, np.nan, 0.0),
        source_values=np.ones(7, dtype=np.float32),
        map_values=[1.0, 5.0],
    )
    actual = assert_matches(
        bridge, source, rects, controls, geometry, gain_map, tile_w=7, tile_h=1
    )
    q = 2.5 / 7.0
    quantized = np.float32(1.0 + 4.0 * (round(q * 32.0) / 32.0))
    assert actual[0, 2, 0] != quantized


def test_binary64_fma_base_keeps_residual_lost_by_separate_multiply_add(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=1,
        width=3,
        planes=1,
        row_padding=0,
        map_shape=(1, 2, 1),
        geometry=(np.nan, 1.0, np.nan, 0.0),
        source_values=[1.0 / 128.0] * 3,
        map_values=[-37.01329803466797, 938.3847045898438],
    )
    actual = assert_matches(bridge, source, rects, controls, geometry, gain_map)
    assert int(actual[0, 0, 0].view(np.uint32)) == 0x3F7B1B27
    separate = np.float32(
        (float(gain_map[0, 1, 0]) - float(gain_map[0, 0, 0]))
        * (0.5 / 3.0)
        + float(gain_map[0, 0, 0])
    )
    assert int(np.float32(separate * np.float32(1.0 / 128.0)).view(np.uint32)) == 0x3F7B1B28


def test_row_padding_and_input_remain_unchanged_and_output_is_compact(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=5,
        width=6,
        planes=3,
        row_padding=5,
        map_shape=(3, 4, 2),
        row_pitch=2,
        col_pitch=3,
    )
    before = source.copy()
    actual = assert_matches(bridge, source, rects, controls, geometry, gain_map)
    assert actual.shape == (5, 6, 3)
    assert_bits(source, before)
    assert np.all(source[:, 18:] == np.float32(-0.375))


def test_nonfinite_padding_is_ignored_and_preserved(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=4,
        width=5,
        planes=2,
        row_padding=3,
        map_shape=(2, 3, 2),
    )
    source[:, 10:] = np.nan
    before = source.copy()
    actual = assert_matches(bridge, source, rects, controls, geometry, gain_map)
    assert actual.shape == (4, 5, 2)
    assert np.isnan(source[:, 10:]).all()
    assert_bits(source[:, :10], before[:, :10])


def test_oracle_only_tiles_keeps_full_image_coordinate_domain():
    source, rects, controls, geometry, gain_map = make_case(
        height=4,
        width=6,
        planes=2,
        top=-3,
        left=5,
        map_shape=(3, 4, 2),
        geometry=(0.4, 0.3, -0.1, 0.2),
        source_values=np.linspace(-0.2, 0.8, 4 * 6 * 2, dtype=np.float32),
    )
    kwargs = dict(
        source_stride=source.shape[1],
        rects8=rects,
        controls9=controls,
        geometry4=geometry,
        gain_map=gain_map,
        tile_w=3,
        tile_h=2,
    )
    full = oracle.reference(source, **kwargs)
    partial = oracle.reference(source, only_tiles={(0, 1)}, **kwargs)
    expected = source[:, : 6 * 2].reshape(4, 6, 2).copy()
    expected[0:2, 3:6] = full[0:2, 3:6]
    assert_bits(partial, expected)


def _unchanged_failure(bridge, source, rects, controls, geometry, gain_map, **kwargs):
    count = (int(rects[2]) - int(rects[0])) * (
        int(rects[3]) - int(rects[1])
    ) * int(controls[0])
    destination = np.full(count + 2, SENTINEL, dtype=np.float32)
    before = destination.copy()
    status, result = bridge.execute(
        source,
        rects,
        controls,
        geometry,
        gain_map,
        destination=destination,
        **kwargs,
    )
    assert status != 0
    assert_bits(result, before)


@pytest.mark.parametrize(
    "change",
    (
        "zero_map_height",
        "zero_map_planes",
        "zero_area_planes",
        "short_map",
        "nan_map",
        "nan_source",
        "zero_spacing",
        "negative_spacing",
        "infinite_origin",
        "black_floor_full_scale",
        "zero_pitch",
        "oversized_tile",
        "unrepresentable_segment_end",
    ),
)
def test_bad_metadata_fails_without_changing_destination(bridge, change):
    source, rects, controls, geometry, gain_map = make_case(
        height=3,
        width=4,
        planes=1,
        map_shape=(2, 2, 1),
        geometry=(0.5, 0.5, 0.0, 0.0),
    )
    kwargs = {}
    if change == "zero_map_height":
        controls[5] = 0
    elif change == "zero_map_planes":
        controls[7] = 0
    elif change == "zero_area_planes":
        controls[2] = 0
    elif change == "short_map":
        kwargs["map_count"] = gain_map.size - 1
    elif change == "nan_map":
        gain_map[0, 0, 0] = np.nan
    elif change == "nan_source":
        source[0, 0] = np.nan
    elif change == "zero_spacing":
        geometry[0] = 0.0
    elif change == "negative_spacing":
        geometry[1] = -0.5
    elif change == "infinite_origin":
        geometry[0] = np.inf
    elif change == "black_floor_full_scale":
        controls[8] = 3
        kwargs["black_floor"] = 65535
    elif change == "zero_pitch":
        controls[3] = 0
    elif change == "oversized_tile":
        kwargs["tile_w"] = 1025
    elif change == "unrepresentable_segment_end":
        rects = np.asarray((0, 0, 1, 1, 0, 0, 1, 1), dtype=np.int32)
        source = np.asarray([[0.25]], dtype=np.float32)
        controls = np.asarray((1, 0, 1, 1, 1, 1, 2, 1, 1), dtype=np.uint32)
        geometry = np.asarray((1.0, 3.0e9, 0.0, 0.0), dtype=np.float64)
        gain_map = np.asarray([[[1.0], [2.0]]], dtype=np.float32)
    _unchanged_failure(
        bridge, source, rects, controls, geometry, gain_map, **kwargs
    )


@pytest.mark.parametrize("short", ("source", "destination", "stride"))
def test_short_buffers_and_stride_fail_atomically(bridge, short):
    source, rects, controls, geometry, gain_map = make_case(
        height=4, width=5, planes=2, map_shape=(2, 3, 2)
    )
    count = 4 * 5 * 2
    destination = np.full(count + 1, SENTINEL, dtype=np.float32)
    before = destination.copy()
    kwargs = {}
    if short == "source":
        required = 3 * source.shape[1] + 2 * 5
        kwargs["source_count"] = required - 1
    elif short == "destination":
        kwargs["destination_count"] = count - 1
    else:
        kwargs["source_count"] = source.size
        kwargs["source_pointer"] = float_ptr(source)
        # The bridge takes the physical row stride from source.shape[1].
        # A compact view with a too-short row exercises that preflight.
        source = source[:, : 2 * 5 - 1]
        kwargs["source_pointer"] = float_ptr(source)
    status, result = bridge.execute(
        source,
        rects,
        controls,
        geometry,
        gain_map,
        destination=destination,
        **kwargs,
    )
    assert status != 0
    assert_bits(result, before)


def test_size_overflow_and_oversized_image_are_rejected_atomically(bridge):
    source, rects, controls, geometry, gain_map = make_case(
        height=1, width=2, planes=1, map_shape=(1, 2, 1)
    )
    destination = np.full(2, SENTINEL, dtype=np.float32)
    before = destination.copy()
    status, result = bridge.execute(
        source,
        rects,
        controls,
        geometry,
        gain_map,
        destination=destination,
        source_count=SIZE_MAX,
        destination_count=SIZE_MAX,
        tile_w=SIZE_MAX,
        tile_h=1,
    )
    assert status != 0
    assert_bits(result, before)

    huge_rects = np.asarray(
        (0, 0, 1, 128_000_001, 0, 0, 1, 128_000_001), dtype=np.int32
    )
    huge_controls = np.asarray((1, 0, 1, 1, 1, 1, 1, 1, 0), dtype=np.uint32)
    status, result = bridge.execute(
        source,
        huge_rects,
        huge_controls,
        np.asarray((1.0, 1.0, 0.0, 0.0)),
        np.ones((1, 1, 1), dtype=np.float32),
        destination=destination,
        source_count=SIZE_MAX,
        destination_count=SIZE_MAX,
    )
    assert status != 0
    assert_bits(result, before)


@pytest.mark.parametrize("offset", (0, 1))
def test_source_destination_alias_is_rejected_without_mutation(bridge, offset):
    source, rects, controls, geometry, gain_map = make_case(
        height=3, width=4, planes=1, map_shape=(2, 2, 1)
    )
    flat = np.concatenate((source.reshape(-1), np.full(32, SENTINEL, np.float32)))
    before = flat.copy()
    source_ptr = C.cast(C.c_void_p(flat.ctypes.data), FLOAT)
    destination_ptr = C.cast(
        C.c_void_p(flat.ctypes.data + offset * np.dtype(np.float32).itemsize),
        FLOAT,
    )
    status, _ = bridge.execute(
        source,
        rects,
        controls,
        geometry,
        gain_map,
        destination=flat,
        source_pointer=source_ptr,
        destination_pointer=destination_ptr,
        destination_count=flat.size,
    )
    assert status != 0
    assert_bits(flat, before)
