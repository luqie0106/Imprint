"""Independent instruction model for the ACR id1 WarpRectilinear CPU path.

The model follows the archived ARM64 listings in
``research/acr/reports/evidence/2026-10-20-warp-spec``.  It deliberately does
not import the product warp, OpenCV, or another image-resampling package.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
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

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
MAX_IMAGE_SAMPLES = 128 * 1024 * 1024
MAX_TILE_EXTENT = 1024
_RADIUS = 2
_FILTER_SIDE = 4
_FRACTION_SCALE = 32


def fma32(a: float, b: float, c: float) -> np.float32:
    """One correctly rounded binary32 fused multiply-add."""
    return F(_fmaf(float(F(a)), float(F(b)), float(F(c))))


def fma64(a: float, b: float, c: float) -> float:
    """One correctly rounded binary64 fused multiply-add."""
    return float(_fma(float(a), float(b), float(c)))


def _checked_i32(value: float, label: str) -> int:
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite")
    result = int(value)
    if result < INT32_MIN or result > INT32_MAX:
        raise ValueError(f"{label} is outside int32")
    return result


def _round_away_positive(value: float) -> int:
    """ARM FCVTZS after adding 0.5 for the positive aspect-height case."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("aspect-scaled image height must be finite and positive")
    return _checked_i32(math.floor(value + 0.5), "aspect-scaled image height")


def _fminnm64(value: float, other: float) -> float:
    # ARM FMINNM returns the numeric operand when exactly one input is NaN.
    if np.isnan(value):
        return float(other)
    if np.isnan(other):
        return float(value)
    return float(value if value < other else other)


def _fminnm32(value: np.float32, other: np.float32) -> np.float32:
    if np.isnan(value):
        return F(other)
    if np.isnan(other):
        return F(value)
    return F(value if value < other else other)


def _cubic_kernel(value: float) -> float:
    """The radius-2, a=-0.75 kernel, with the recovered double FMADD order."""
    t = abs(float(value))
    if t >= 2.0:
        return 0.0
    if t < 1.0:
        z = fma64(t, 1.25, -2.25)
        z = t * z
        return fma64(z, t, 1.0)
    z = fma64(t, -0.75, 3.75)
    z = fma64(z, t, -6.0)
    return fma64(z, t, 3.0)


@lru_cache(maxsize=1)
def _weight_table() -> np.ndarray:
    """Build all 32x32 float tables with the original normalization order."""
    table = np.empty((32, 32, 4, 4), dtype=np.float32)
    for fy in range(32):
        for fx in range(32):
            raw = np.empty((4, 4), dtype=np.float32)
            total = 0.0
            for i in range(4):
                y = float(i - 1) - float(fy) / 32.0
                wy = F(_cubic_kernel(y))
                for j in range(4):
                    x = float(j - 1) - float(fx) / 32.0
                    wx = F(_cubic_kernel(x))
                    product = F(wx * wy)
                    raw[i, j] = product
                    # The assembly converts each float product to double and
                    # accumulates row then column, without fused accumulation.
                    total = total + float(product)
            norm = F(1.0 / total)
            for i in range(4):
                for j in range(4):
                    table[fy, fx, i, j] = F(raw[i, j] * norm)
    table.flags.writeable = False
    return table


def _radial_factor(r2: float, coeff: np.ndarray) -> float:
    """Evaluate all fifteen double Horner terms, including zero terms."""
    radius = math.sqrt(min(max(float(r2), 0.0), 1.0))
    expanded = [
        float(coeff[0]), 0.0,
        float(coeff[1]), 0.0,
        float(coeff[2]), 0.0,
        float(coeff[3]),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    value = 0.0
    for coefficient in reversed(expanded):
        value = fma64(value, radius, coefficient)
    return value


def _prepared_geometry(
    bounds: tuple[int, int, int, int], geometry: tuple[float, float, float]
) -> tuple[float, float, float, float, float]:
    top, left, bottom, right = bounds
    cx, cy, aspect = geometry
    height = bottom - top
    width = right - left
    scaled_height = _round_away_positive(aspect * float(height))

    center_y = fma64(cy, float(height), float(top))
    center_x = fma64(cx, float(width), float(left))
    center_scaled_y = fma64(cy, float(scaled_height), float(top))
    bottom_scaled = top + scaled_height
    if bottom_scaled < INT32_MIN or bottom_scaled > INT32_MAX:
        raise ValueError("aspect-scaled image bottom is outside int32")

    top_dy = float(top) - center_scaled_y
    bottom_dy = float(top + scaled_height) - center_scaled_y
    left_dx = float(left) - center_x
    right_dx = float(right) - center_x

    left_dx2 = left_dx * left_dx
    top_left = fma64(top_dy, top_dy, left_dx2)
    bottom_left = fma64(bottom_dy, bottom_dy, left_dx2)
    right_dx2 = right_dx * right_dx
    bottom_right = fma64(bottom_dy, bottom_dy, right_dx2)
    top_right = fma64(top_dy, top_dy, right_dx2)
    distance = math.sqrt(max(top_left, bottom_left, bottom_right, top_right))
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("warp radius must be finite and positive")
    return center_y, center_x, aspect, distance, 1.0 / distance


def _map_point(
    row: int,
    column: int,
    coeff: np.ndarray,
    bounds: tuple[int, int, int, int],
    prepared: tuple[float, float, float, float, float],
) -> tuple[float, float]:
    top, left, bottom, right = bounds
    center_y, center_x, aspect, _distance, inv_distance = prepared
    dy = float(row) - center_y
    dx = float(column) - center_x
    ny = dy * inv_distance
    nx = dx * inv_distance
    v = ny * aspect
    u = nx
    r2 = _fminnm64(v * v + u * u, 1.0)
    radial = _radial_factor(r2, coeff)
    # The NOP-tangential coordinate branch has separate FMUL and FADD.
    source_y = dy * radial + center_y
    source_x = dx * radial + center_x
    return source_y, source_x


def _clamp_source_coordinates(
    source_y: float,
    source_x: float,
    bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    top, left, bottom, right = bounds
    source_y = min(max(source_y, float(top)), float(bottom - 1))
    source_x = min(max(source_x, float(left)), float(right - 1))
    return source_y, source_x


def source_area(
    bounds: tuple[int, int, int, int],
    tile: tuple[int, int, int, int],
    coeffs: np.ndarray,
    geometry: tuple[float, float, float],
) -> tuple[int, int, int, int]:
    """Compute the full SrcArea for one absolute-coordinate output tile."""
    top, left, bottom, right = bounds
    tile_top, tile_left, tile_bottom, tile_right = tile
    prepared = _prepared_geometry(bounds, geometry)
    min_y = min_x = INT32_MAX
    max_y = max_x = INT32_MIN

    for coeff in coeffs:
        # The producer scans the top edge only for floor(y), bottom only for
        # ceil(y), left only for floor(x), and right only for ceil(x).
        # It does not infer these extrema from four corners or a full border.
        for column in range(tile_left, tile_right):
            y, _x = _map_point(tile_top, column, coeff, bounds, prepared)
            min_y = min(min_y, _checked_i32(math.floor(y), "SrcArea minimum y"))
        last_row = tile_bottom - 1
        for column in range(tile_left, tile_right):
            y, _x = _map_point(last_row, column, coeff, bounds, prepared)
            max_y = max(max_y, _checked_i32(math.ceil(y), "SrcArea maximum y"))
        for row in range(tile_top, tile_bottom):
            _y, x = _map_point(row, tile_left, coeff, bounds, prepared)
            min_x = min(min_x, _checked_i32(math.floor(x), "SrcArea minimum x"))
        last_column = tile_right - 1
        for row in range(tile_top, tile_bottom):
            _y, x = _map_point(row, last_column, coeff, bounds, prepared)
            max_x = max(max_x, _checked_i32(math.ceil(x), "SrcArea maximum x"))

    area_top = max(top, _checked_i32(min_y - _RADIUS, "SrcArea top"))
    area_left = max(left, _checked_i32(min_x - _RADIUS, "SrcArea left"))
    area_bottom = min(
        bottom, _checked_i32(max_y + _RADIUS + 1, "SrcArea bottom")
    )
    area_right = min(
        right, _checked_i32(max_x + _RADIUS + 1, "SrcArea right")
    )
    if area_bottom - area_top < 5 or area_right - area_left < 5:
        raise ValueError(
            "source area is smaller than the consumer support: "
            f"{(area_top, area_left, area_bottom, area_right)}"
        )
    return area_top, area_left, area_bottom, area_right


def _validate(
    source: np.ndarray,
    source_stride: int,
    bounds: tuple[int, int, int, int],
    image_planes: int,
    param_planes: int,
    geometry: tuple[float, float, float],
    coeffs: np.ndarray,
    tile_width: int,
    tile_height: int,
) -> tuple[int, int]:
    if source.dtype != np.float32:
        raise ValueError("source must contain float32 samples")
    top, left, bottom, right = bounds
    height = bottom - top
    width = right - left
    if height <= 0 or width <= 0:
        raise ValueError("image bounds must have positive height and width")
    if image_planes < 1 or image_planes > 4:
        raise ValueError("image planes must be in [1, 4]")
    if param_planes not in (1, image_planes):
        raise ValueError("parameter planes must be 1 or image_planes")
    if coeffs.shape != (param_planes, 6):
        raise ValueError("coefficients must have shape (param_planes, 6)")
    if not np.isfinite(coeffs).all():
        raise ValueError("warp coefficients must be finite")
    if np.any(coeffs[:, 4:] != 0.0):
        raise ValueError("nonzero tangential coefficients are unsupported")
    if len(geometry) != 3 or not all(math.isfinite(x) for x in geometry):
        raise ValueError("geometry must contain three finite values")
    cx, cy, aspect = geometry
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and aspect > 0.0):
        raise ValueError("invalid normalized center or aspect")
    if tile_width < 1 or tile_height < 1:
        raise ValueError("tile dimensions must be positive")
    if tile_width > MAX_TILE_EXTENT or tile_height > MAX_TILE_EXTENT:
        raise ValueError("tile dimensions exceed the supported limit")
    if source_stride < width * image_planes:
        raise ValueError("source row stride is shorter than a logical row")
    samples = height * width * image_planes
    if samples <= 0 or samples > MAX_IMAGE_SAMPLES:
        raise ValueError("image is oversized")
    required = (height - 1) * source_stride + width * image_planes
    flat = source.reshape(-1)
    if required > flat.size:
        raise ValueError("source buffer is too small")
    for row in range(height):
        start = row * source_stride
        if not np.isfinite(flat[start : start + width * image_planes]).all():
            raise ValueError("source contains nonfinite values")
    identity = bool(
        np.all(coeffs[:, 0] == 1.0)
        and np.all(coeffs[:, 1:4] == 0.0)
        and np.all(coeffs[:, 4:6] == 0.0)
    )
    if not identity and (height < 5 or width < 5):
        raise ValueError("nonidentity image dimensions must be at least 5")
    return height, width


def reference(
    source: np.ndarray,
    bounds: tuple[int, int, int, int],
    image_planes: int,
    param_planes: int,
    geometry: tuple[float, float, float],
    coeffs: np.ndarray,
    *,
    source_stride: int | None = None,
    tile_width: int = 64,
    tile_height: int = 64,
    only_tiles: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Return compact HWC output, optionally evaluating selected tile origins.

    ``source`` is a flat or row-padded float32 buffer. ``bounds`` and tile
    origins use absolute image coordinates. ``only_tiles`` limits pixel
    production for comparison; each selected tile still gets a complete
    independently computed SrcArea.
    """
    source = np.asarray(source)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    bounds = tuple(int(x) for x in bounds)
    geometry = tuple(float(x) for x in geometry)
    if len(bounds) != 4:
        raise ValueError("bounds must contain top, left, bottom, right")
    top, left, bottom, right = bounds
    stride = (right - left) * image_planes if source_stride is None else int(source_stride)
    height, width = _validate(
        source,
        stride,
        bounds,
        int(image_planes),
        int(param_planes),
        geometry,
        coeffs,
        int(tile_width),
        int(tile_height),
    )
    flat = source.reshape(-1)
    output = np.full((height, width, image_planes), np.nan, dtype=np.float32)
    identity = bool(
        np.all(coeffs[:, 0] == 1.0)
        and np.all(coeffs[:, 1:4] == 0.0)
        and np.all(coeffs[:, 4:6] == 0.0)
    )
    if identity:
        # Preserve every binary32 bit, including signed zero.
        output_bits = output.view(np.uint32).reshape(-1)
        for row in range(height):
            src_start = row * stride
            dst_start = row * width * image_planes
            count = width * image_planes
            output_bits[dst_start : dst_start + count] = flat[
                src_start : src_start + count
            ].view(np.uint32)
        return output

    if only_tiles is None:
        output.fill(0.0)

    prepared = _prepared_geometry(bounds, geometry)
    weights = _weight_table()
    for tile_top in range(top, bottom, tile_height):
        tile_bottom = min(bottom, tile_top + tile_height)
        for tile_left in range(left, right, tile_width):
            tile_right = min(right, tile_left + tile_width)
            tile = (tile_top, tile_left, tile_bottom, tile_right)
            if only_tiles is not None and (tile_top, tile_left) not in only_tiles:
                continue
            area_top, area_left, area_bottom, area_right = source_area(
                bounds, tile, coeffs, geometry
            )
            for row in range(tile_top, tile_bottom):
                for column in range(tile_left, tile_right):
                    for plane in range(image_planes):
                        coeff = coeffs[0 if param_planes == 1 else plane]
                        source_y, source_x = _map_point(
                            row, column, coeff, bounds, prepared
                        )
                        source_y, source_x = _clamp_source_coordinates(
                            source_y, source_x, bounds
                        )
                        iy = math.floor(source_y)
                        ix = math.floor(source_x)
                        fy = int((source_y - float(iy)) * 32.0)
                        fx = int((source_x - float(ix)) * 32.0)
                        if not (0 <= fy < 32 and 0 <= fx < 32):
                            raise ValueError("sample fraction is outside [0, 32)")

                        anchor_y = iy + 1 - _RADIUS
                        anchor_x = ix + 1 - _RADIUS
                        max_anchor_y = area_bottom - _FILTER_SIDE - 1
                        max_anchor_x = area_right - _FILTER_SIDE - 1
                        if anchor_y < area_top or anchor_y > max_anchor_y:
                            fy = 0
                        if anchor_x < area_left or anchor_x > max_anchor_x:
                            fx = 0
                        anchor_y = min(max(anchor_y, area_top), max_anchor_y)
                        anchor_x = min(max(anchor_x, area_left), max_anchor_x)

                        total = F(0.0)
                        for i in range(4):
                            sample_y = anchor_y + i
                            local_y = sample_y - top
                            for j in range(4):
                                sample_x = anchor_x + j
                                local_x = sample_x - left
                                src_index = (
                                    local_y * stride
                                    + local_x * image_planes
                                    + plane
                                )
                                total = fma32(
                                    weights[fy, fx, i, j], flat[src_index], total
                                )
                        value = _fminnm32(total, F(1.0))
                        if value < F(0.0):
                            value = F(0.0)
                        output[row - top, column - left, plane] = value
    return output

