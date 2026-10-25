"""Independent prepared-input composition oracle for the six core3 kernels.

The row and prefix math comes from the versioned instruction-derived oracle
modules beside this file. This module composes those references according to
the recovered tile schedule; it never calls native code to create expected
pixels.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_reference(module_name: str):
    path = Path(__file__).with_name(f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load instruction reference {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prefix = _load_reference("core3_prefix_oracle")
_green = _load_reference("core3_green_oracle")
_colour = _load_reference("core3_colour_oracle")
_refine = _load_reference("core3_refine_oracle")

_F = np.float32
_U16_SCALE_BITS = 0x37800080
_U16_SCALE = np.asarray([_U16_SCALE_BITS], dtype=np.uint32).view(np.float32)[0]


def _refine_row(kernel: int, guide: np.ndarray, channel_a: np.ndarray,
                channel_b: np.ndarray, mask: np.ndarray, y: int, x: int,
                width: int, phase: int, lower: float, upper: float):
    """Run an instruction reference at an arbitrary tile row and column."""
    # The versioned row oracle addresses a fixed center at [2, 8]. Slice only
    # the needed neighborhood so its center matches the schedule. In
    # particular, refine 2 starts at tile x=12 while the row oracle's fixed
    # horizontal center is x=8.
    ys = slice(y - 2, y + 3)
    xs = slice(x - 8, None)
    return _refine.reference(
        kernel,
        guide[ys, xs], channel_a[ys, xs], channel_b[ys, xs], mask[ys, xs],
        width, phase, _F(lower), _F(upper),
    )


def _reference_prepared(tile: np.ndarray, *, width: int | None,
                        green_parameters, lower: float, upper: float,
                        row_origin_parity: int, column_origin_parity: int,
                        input_kind: str) -> dict[str, object]:
    if tile.ndim != 2:
        raise ValueError("prepared tile must be a two-dimensional array")
    height, row_stride = tile.shape
    if width is None:
        width = row_stride
    width = int(width)
    if width <= 0 or width > row_stride:
        raise ValueError("width must be positive and no larger than row stride")
    if row_origin_parity not in (0, 1) or column_origin_parity not in (0, 1):
        raise ValueError("origin parity values must be zero or one")
    if len(green_parameters) != 4:
        raise ValueError("green_parameters must contain t, k, m, q")

    t, k, m, q = map(_F, green_parameters)
    lower, upper = _F(lower), _F(upper)
    margin = 8 if m == _F(0.0) else 12
    if width < 2 * margin + 4 or height < 2 * margin + 1:
        raise ValueError("tile dimensions are below the selected schedule minimum")
    if width % 4:
        raise ValueError("tile width must be a multiple of four")

    # Retain quiet-NaN guards for the independent prefix oracle. The uint16
    # interface applies the recovered reciprocal multiply exactly once;
    # prepared float input is copied as-is with no scale or quantization.
    padded = np.full((height, width + 8), np.float32(np.nan), dtype=np.float32)
    if input_kind == "u16":
        if tile.dtype != np.uint16:
            raise ValueError("tile_u16 must be a two-dimensional uint16 array")
        with np.errstate(all="ignore"):
            prepared = np.multiply(tile[:, :width].astype(np.float32), _U16_SCALE,
                                   dtype=np.float32)
    elif input_kind == "f32":
        if tile.dtype != np.float32:
            raise ValueError("tile_f32 must be a two-dimensional float32 array")
        prepared = np.array(tile[:, :width], dtype=np.float32, copy=True)
        if not np.isfinite(prepared).all():
            raise ValueError("tile_f32 valid pixels must all be finite")
    else:
        raise ValueError(f"unknown prepared input kind: {input_kind}")
    padded[:, 4:4 + width] = prepared
    source = padded[:, 4:]

    workspace = _prefix.reference(padded, width)
    planes = {plane: workspace[:, plane, :] for plane in range(9)}

    # Green stage: aligned x=4, with only the contract's inset-7 lanes claimed
    # valid. Its extra aligned lanes are retained in the workspace for fidelity.
    green_width = width - 8
    for y in range(7, height - 7):
        phase = (row_origin_parity + column_origin_parity + y + 4) & 1
        green, aux = _green.reference(
            source, planes, y, 4, green_width, phase, (t, k, m, q))
        planes[4][y, 4:4 + green_width] = green
        if m != _F(0.0):
            planes[6][y, 4:4 + green_width] = aux

    # Colour stage: A/B swap into P3/P5 with the absolute row parity.
    colour_width = width - 16
    for y in range(8, height - 8):
        phase = (row_origin_parity + column_origin_parity + y + 8) & 1
        a, b = _colour.reference(
            source[y - 1:y + 2, :], planes[4][y - 1:y + 2, :],
            colour_width, phase, center_row=1, offset=8)
        if ((row_origin_parity + y) & 1) == 0:
            planes[3][y, 8:8 + colour_width] = a
            planes[5][y, 8:8 + colour_width] = b
        else:
            planes[5][y, 8:8 + colour_width] = a
            planes[3][y, 8:8 + colour_width] = b

    if m == _F(0.0):
        rgb_planes = (planes[3], planes[4], planes[5])
    else:
        first = {plane: np.full_like(planes[4], np.float32(np.nan))
                 for plane in (0, 1, 2)}
        first_width = width - 16
        for y in range(10, height - 10):
            row_parity = (row_origin_parity + y) & 1
            phase = (row_origin_parity + column_origin_parity + y + 8) & 1
            if row_parity == 0:
                guide, channel_a, channel_b = planes[3], planes[4], planes[5]
                output_guide, output_a, output_b = 0, 1, 2
            else:
                guide, channel_a, channel_b = planes[5], planes[4], planes[3]
                output_guide, output_a, output_b = 2, 1, 0
            out_g, out_a, out_b = _refine_row(
                0, guide, channel_a, channel_b, planes[6], y, 8,
                first_width, phase, lower, upper)
            columns = slice(8, 8 + first_width)
            first[output_guide][y, columns] = out_g
            first[output_a][y, columns] = out_a
            first[output_b][y, columns] = out_b

        second_width = width - 24
        second = {plane: np.full_like(planes[4], np.float32(np.nan))
                  for plane in (0, 1, 2)}
        for y in range(12, height - 12):
            phase = (row_origin_parity + column_origin_parity + y + 12) & 1
            out_g, out_a, out_b = _refine_row(
                1, first[1], first[0], first[2], planes[6], y, 12,
                second_width, phase, lower, upper)
            columns = slice(12, 12 + second_width)
            second[1][y, columns] = out_g
            second[0][y, columns] = out_a
            second[2][y, columns] = out_b
        rgb_planes = (second[0], second[1], second[2])

    top = left = margin
    bottom, right = height - margin, width - margin
    rgb_float = np.empty((bottom - top, right - left, 3), dtype=np.float32)
    for channel, plane in enumerate(rgb_planes):
        rgb_float[..., channel] = plane[top:bottom, left:right]
    if not np.isfinite(rgb_float).all():
        raise ValueError("reference schedule left a nonfinite sample in its valid ROI")
    if np.any(rgb_float < _F(0.0)) or np.any(rgb_float > _F(1.0)):
        raise ValueError("reference RGB lies outside the supported normalized domain")

    scaled = np.multiply(rgb_float, _F(65535.0), dtype=np.float32)
    rgb_u16 = np.rint(scaled).astype(np.uint16)
    return {
        "rgb_float": rgb_float,
        "rgb_u16": rgb_u16,
        "input_width": width,
        "input_height": height,
        "valid_region": (top, left, bottom, right),
        "output_width": right - left,
        "output_height": bottom - top,
        "sample_count": int(rgb_float.size),
    }


def reference(tile_u16: np.ndarray, *, width: int | None = None,
              green_parameters=(25.0, 0.03, 0.0, 0.2),
              lower: float = 0.4, upper: float = 0.4,
              row_origin_parity: int = 0,
              column_origin_parity: int = 0) -> dict[str, object]:
    """Return compact float/u16 RGB plus the tile-logical valid ROI.

    ``tile_u16`` has one uint16 row per tile row. ``width`` may be smaller
    than its row stride to model ignored input padding. The parity values are
    the independent row and column origin bits for the canonical RGGB lattice.
    """
    tile = np.asarray(tile_u16)
    if tile.dtype != np.uint16 or tile.ndim != 2:
        raise ValueError("tile_u16 must be a two-dimensional uint16 array")
    return _reference_prepared(
        tile, width=width, green_parameters=green_parameters, lower=lower,
        upper=upper, row_origin_parity=row_origin_parity,
        column_origin_parity=column_origin_parity, input_kind="u16")


def reference_f32(tile_f32: np.ndarray, *, width: int | None = None,
                  green_parameters=(25.0, 0.03, 0.0, 0.2),
                  lower: float = 0.4, upper: float = 0.4,
                  row_origin_parity: int = 0,
                  column_origin_parity: int = 0) -> dict[str, object]:
    """Return the reference composition for already-prepared float32 pixels.

    Only the first ``width`` values in each row are read. Those valid pixels
    must be finite; padding is ignored and may contain any bit pattern. The
    input is neither scaled nor quantized before entering the common schedule.
    """
    tile = np.asarray(tile_f32)
    if tile.dtype != np.float32 or tile.ndim != 2:
        raise ValueError("tile_f32 must be a two-dimensional float32 array")
    return _reference_prepared(
        tile, width=width, green_parameters=green_parameters, lower=lower,
        upper=upper, row_origin_parity=row_origin_parity,
        column_origin_parity=column_origin_parity, input_kind="f32")
