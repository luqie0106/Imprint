"""Independent float instruction model for the recovered ACR id9 GainMap CPU pass.

This model is based on the archived ARM64 findings and instruction listings.
It does not load or inspect the native implementation.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np


F = np.float32
_libm = ctypes.CDLL(ctypes.util.find_library("m") or None)
_fmaf = _libm.fmaf
_fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_fmaf.restype = ctypes.c_float
_fma = _libm.fma
_fma.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
_fma.restype = ctypes.c_double

MAX_MAP_SAMPLES = 1_000_000
MAX_IMAGE_SAMPLES = 128_000_000
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def fma32(a: float, b: float, c: float) -> np.float32:
    """One correctly rounded binary32 fused multiply-add."""
    return F(_fmaf(float(F(a)), float(F(b)), float(F(c))))


def fma64(a: float, b: float, c: float) -> float:
    """One correctly rounded binary64 fused multiply-add."""
    return float(_fma(float(a), float(b), float(c)))


def _checked_ceil_i32(value: float) -> int:
    result = math.ceil(value)
    if result < INT32_MIN or result > INT32_MAX:
        raise ValueError("GainMap segment end is outside int32")
    return result


def _axis_parameters(
    map_extent: int, spacing: float, origin: float
) -> tuple[float, float]:
    if map_extent == 1:
        return 1.0, 0.0
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("GainMap spacing must be finite and positive")
    if not math.isfinite(origin):
        raise ValueError("GainMap origin must be finite")
    return float(spacing), float(origin)


def _coordinate(
    extent: int,
    origin_bound: int,
    coordinate: int,
    spacing: float,
    map_origin: float,
) -> float:
    inv_extent = 1.0 / float(extent)
    centered = (0.5 - float(origin_bound)) + float(coordinate)
    return fma64(inv_extent, centered, -map_origin) / spacing


def _vertical_sample(
    gain_map: np.ndarray, qv: float, map_column: int, map_plane: int
) -> np.float32:
    map_height = gain_map.shape[0]
    if qv <= 0.0:
        upper = lower = 0
        fraction = F(0.0)
    elif qv >= float(map_height - 1):
        upper = lower = map_height - 1
        fraction = F(0.0)
    else:
        upper = int(qv)
        lower = upper + 1
        fraction = F(qv - float(upper))

    upper_weight = F(F(1.0) - fraction)
    lower_term = F(fraction * gain_map[lower, map_column, map_plane])
    return fma32(
        gain_map[upper, map_column, map_plane], upper_weight, lower_term
    )


def _horizontal_segment(
    gain_map: np.ndarray,
    qh: float,
    qv: float,
    map_plane: int,
    *,
    image_width: int,
    image_left: int,
    map_origin: float,
    spacing: float,
) -> tuple[np.float32, np.float32, int]:
    map_width = gain_map.shape[1]

    if qh <= 0.0:
        base = _vertical_sample(gain_map, qv, 0, map_plane)
        inv_width = 1.0 / float(image_width)
        end = _checked_ceil_i32(
            map_origin / inv_width - (0.5 - float(image_left))
        )
        return F(base), F(0.0), end

    if qh >= float(map_width - 1):
        base = _vertical_sample(gain_map, qv, map_width - 1, map_plane)
        return F(base), F(0.0), INT32_MAX

    map_column = int(qh)
    left = _vertical_sample(gain_map, qv, map_column, map_plane)
    right = _vertical_sample(gain_map, qv, map_column + 1, map_plane)
    delta = float(right) - float(left)
    base = F(fma64(delta, qh - float(map_column), float(left)))
    inv_width = 1.0 / float(image_width)
    slope = F((inv_width * delta) / spacing)
    next_node = map_column + 1
    end = _checked_ceil_i32(
        fma64(float(next_node), spacing, map_origin) / inv_width
        - (0.5 - float(image_left))
    )
    return base, slope, end


def _selected_interval(
    tile_start: int,
    tile_end: int,
    area_start: int,
    area_end: int,
    pitch: int,
    *,
    empty_area: bool,
) -> tuple[int, int, int] | None:
    if empty_area:
        # The helper returns the complete tile bounds for an empty area, but
        # the parsed row/column pitches still control sampling. The first
        # sample is the tile origin; there is no area-origin alignment.
        return tile_start, tile_end, pitch

    clipped_start = max(tile_start, area_start)
    clipped_end = min(tile_end, area_end)
    if clipped_start >= clipped_end:
        return None
    first = clipped_start + (-(clipped_start - area_start)) % pitch
    if first >= clipped_end:
        return None
    last = clipped_end - 1 - ((clipped_end - 1 - first) % pitch)
    return first, last + 1, pitch


def reference(
    source: np.ndarray,
    source_stride: int,
    rects8: np.ndarray | tuple[int, ...],
    controls9: np.ndarray | tuple[int, ...],
    geometry4: np.ndarray | tuple[float, ...],
    gain_map: np.ndarray,
    *,
    black_floor: int = 0,
    tile_w: int = 64,
    tile_h: int = 64,
    only_tiles: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Return compact HWC output for one complete image.

    ``source`` contains a row-padded HWC image: each row has
    ``source_stride`` float samples, while the logical row has
    ``image_width * image_planes``. The returned array is compact HWC.
    """
    source = np.asarray(source)
    if source.dtype != np.float32:
        raise ValueError("source must contain float32 samples")
    source = source.reshape(-1)
    rects = tuple(int(x) for x in rects8)
    controls = tuple(int(x) for x in controls9)
    geometry = tuple(float(x) for x in geometry4)
    gains = np.asarray(gain_map)
    if gains.dtype != np.float32:
        raise ValueError("GainMap must contain float32 samples")

    if len(rects) != 8 or len(controls) != 9 or len(geometry) != 4:
        raise ValueError("invalid GainMap metadata array length")
    image_top, image_left, image_bottom, image_right = rects[:4]
    area_top, area_left, area_bottom, area_right = rects[4:]
    image_planes, first_plane, area_planes, row_pitch, col_pitch = controls[:5]
    map_v, map_h, map_planes, list_stage = controls[5:]
    image_height = image_bottom - image_top
    image_width = image_right - image_left
    logical_samples = image_height * image_width * image_planes

    if (
        image_height <= 0
        or image_width <= 0
        or image_planes <= 0
        or logical_samples > MAX_IMAGE_SAMPLES
    ):
        raise ValueError("invalid or oversized image bounds")
    if tile_w < 1 or tile_h < 1 or tile_w > 1024 or tile_h > 1024:
        raise ValueError("tile dimensions must be in [1, 1024]")
    if source_stride < image_width * image_planes:
        raise ValueError("source stride is shorter than a logical row")
    required_source = (image_height - 1) * source_stride + image_width * image_planes
    if required_source > source.size:
        raise ValueError("short source buffer")
    logical_row_samples = image_width * image_planes
    for row in range(image_height):
        row_start = row * source_stride
        if not np.isfinite(
            source[row_start : row_start + logical_row_samples]
        ).all():
            raise ValueError("source contains nonfinite samples")
    if (
        map_v <= 0
        or map_h <= 0
        or map_planes <= 0
        or map_v * map_h * map_planes > MAX_MAP_SAMPLES
    ):
        raise ValueError("invalid or oversized GainMap")
    if gains.size < map_v * map_h * map_planes:
        raise ValueError("short GainMap buffer")
    gains = gains[: map_v * map_h * map_planes].reshape(
        map_v, map_h, map_planes
    )
    if not np.isfinite(gains).all():
        raise ValueError("GainMap contains nonfinite values")
    spacing_v, spacing_h, origin_v, origin_h = geometry
    spacing_v, origin_v = _axis_parameters(map_v, spacing_v, origin_v)
    spacing_h, origin_h = _axis_parameters(map_h, spacing_h, origin_h)
    if not 0 <= black_floor <= 65535:
        raise ValueError("invalid black floor")
    if list_stage >= 2 and black_floor == 65535:
        raise ValueError("black floor leaves no positive range")
    if row_pitch <= 0 or col_pitch <= 0:
        raise ValueError("area pitches must be positive")

    output = np.empty(logical_samples, dtype=np.float32)
    for y in range(image_height):
        src_at = y * source_stride
        dst_at = y * image_width * image_planes
        output[dst_at : dst_at + image_width * image_planes] = source[
            src_at : src_at + image_width * image_planes
        ]
    output = output.reshape(image_height, image_width, image_planes)

    empty_area = area_top >= area_bottom or area_left >= area_right
    if not empty_area:
        area_height = area_bottom - area_top
        area_width = area_right - area_left
        row_pitch = min(row_pitch, area_height)
        col_pitch = min(col_pitch, area_width)

    plane_end = min(first_plane + area_planes, image_planes)
    if first_plane > image_planes:
        raise ValueError("invalid first plane")

    black_z = F(F(black_floor) / F(65535.0))
    black_scale = F(F(1.0) - black_z)
    if list_stage >= 2 and black_floor:
        black_inv = F(F(1.0) / black_scale)
        black_bias = F(F(1.0) - black_inv)
    else:
        black_inv = F(1.0)
        black_bias = F(0.0)

    for tile_row_index, tile_top in enumerate(
        range(image_top, image_bottom, tile_h)
    ):
        tile_bottom = min(tile_top + tile_h, image_bottom)
        selected_rows = _selected_interval(
            tile_top,
            tile_bottom,
            area_top,
            area_bottom,
            row_pitch,
            empty_area=empty_area,
        )
        if selected_rows is None:
            continue
        row_start, row_end, row_step = selected_rows

        for tile_column_index, tile_left in enumerate(
            range(image_left, image_right, tile_w)
        ):
            if only_tiles is not None and (
                tile_row_index,
                tile_column_index,
            ) not in only_tiles:
                continue
            tile_right = min(tile_left + tile_w, image_right)
            selected_columns = _selected_interval(
                tile_left,
                tile_right,
                area_left,
                area_right,
                col_pitch,
                empty_area=empty_area,
            )
            if selected_columns is None:
                continue
            col_start, col_end, col_step = selected_columns

            for row in range(row_start, row_end, row_step):
                qv = _coordinate(
                    image_bottom - image_top,
                    image_top,
                    row,
                    spacing_v,
                    origin_v,
                )
                for plane in range(first_plane, plane_end):
                    map_plane = min(plane, map_planes - 1)
                    segment_base: np.float32 | None = None
                    segment_slope = F(0.0)
                    segment_end = INT32_MIN
                    previous_column: int | None = None
                    offset = F(0.0)

                    for column in range(col_start, col_end, col_step):
                        if segment_base is None:
                            offset = F(0.0)
                            advance_columns = (column,)
                        else:
                            assert previous_column is not None
                            advance_columns = range(previous_column + 1, column + 1)

                        for advance_column in advance_columns:
                            if segment_base is None or advance_column >= segment_end:
                                qh = _coordinate(
                                    image_right - image_left,
                                    image_left,
                                    advance_column,
                                    spacing_h,
                                    origin_h,
                                )
                                segment_base, segment_slope, segment_end = (
                                    _horizontal_segment(
                                        gains,
                                        qh,
                                        qv,
                                        map_plane,
                                        image_width=image_right - image_left,
                                        image_left=image_left,
                                        map_origin=origin_h,
                                        spacing=spacing_h,
                                    )
                                )
                                offset = F(0.0)
                            else:
                                offset = F(offset + F(1.0))

                        value = output[
                            row - image_top, column - image_left, plane
                        ]
                        if list_stage >= 2 and black_floor:
                            value = fma32(value, black_inv, black_bias)
                        gain = fma32(segment_slope, offset, segment_base)
                        product = F(gain * value)
                        result = F(np.fmin(product, F(1.0)))
                        if list_stage >= 2 and black_floor:
                            result = fma32(result, black_scale, black_z)
                        output[
                            row - image_top, column - image_left, plane
                        ] = result
                        previous_column = column

    return output
